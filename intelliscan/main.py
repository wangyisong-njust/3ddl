#!/usr/bin/env python3
from __future__ import annotations

"""
3D-IntelliScan Main Pipeline

This module serves as the main entry point for the 3D-IntelliScan semiconductor
metrology and defect detection pipeline. It orchestrates the complete workflow
from 3D NII to 2D conversion, object detection, 3D bounding box generation,
segmentation, and metrology analysis.

Pipeline Workflow:
1. Convert 3D NIfTI volumes to 2D slices (horizontal & vertical views)
2. Run YOLO object detection on 2D slices (single model for both views)
3. Generate 3D bounding boxes from 2D detections
4. Perform 3D semantic segmentation within bounding boxes
5. Compute metrology measurements (BLT, void ratio, defects)
6. Generate PDF reports with analysis

Usage:
    # Process files from files.txt (batch mode)
    python main.py

    # Process a single file
    python main.py /path/to/input.nii

    # Force reprocessing
    python main.py /path/to/input.nii --force

    # Use as module
    from main import process_single_file
    result = process_single_file("/path/to/input.nii", force=False)
"""

import argparse
import json
import re
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path

# Set matplotlib backend to Agg (non-interactive) to prevent GUI errors in threads
import matplotlib
matplotlib.use("Agg")

import nibabel as nib
import numpy as np
import pandas as pd

from detection import (
    load_yolo_model,
    run_yolo_detection,
    run_yolo_detection_folder_to_memory,
    run_yolo_detection_inmemory_from_volume,
)
from merge import generate_bb3d, generate_bb3d_inmemory
from metrology import MAKE_CLEAN_DEFAULT, compute_metrology_from_array, compute_metrology_info
from report import generate_pdf_report
from segmentation import (
    SegmentationConfig,
    SegmentationInference,
    assemble_full_volume,
    segment_bboxes,
)
from utils import PipelineLogbook, PipelineLogger, PipelineMetrics, create_folder_structure, log, nii2jpg


@dataclass
class PipelineConfig:
    """Pipeline configuration settings."""

    output_base: str = "output"
    detection_model: str = "models/detection_model.pt"
    segmentation_model: str = "models/segmentation_model.ckpt"
    tag: str = ""
    use_combined_seg_metrology: bool = True
    use_inmemory_detection: bool = True
    detection_image_workers: int = 4
    detection_batch_prefetch: bool = True
    use_ramdisk_detection: bool = False
    use_adaptive_margin: bool = False
    use_guarded_adaptive_margin: bool = False
    guard_max_loss_xy: int = 4
    guard_max_loss_z: int = 8
    force_sliding_window: bool = False
    force_direct_full_crop: bool = False
    direct_roi_size: tuple[int, int, int] | None = None
    save_bump_predictions: bool = False
    verbose: bool = True
    clean_mask: bool = MAKE_CLEAN_DEFAULT
    use_trt: bool = False
    trt_engine_path: str | None = None
    use_compile: bool = False
    compile_mode: str = "reduce-overhead"
    enable_ai_analysis: bool = False


# Default configuration
DEFAULT_CONFIG = PipelineConfig()


METROLOGY_COLUMNS = [
    "filename",
    "is_memory",
    "void_ratio_defect",
    "solder_extrusion_defect",
    "pad_misalignment_defect",
    "bb",
    "BLT",
    "Pad_misalignment",
    "Void_to_solder_ratio",
    "solder_extrusion_copper_pillar",
    "pillar_width",
    "pillar_height",
]


def normalize_bboxes(bboxes: np.ndarray | list) -> np.ndarray:
    """Normalize bbox arrays to shape [N, 6], including empty and single-box cases."""
    arr = np.asarray(bboxes)
    if arr.size == 0:
        return np.empty((0, 6), dtype=np.int32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def build_metrology_record(filename: str, measurements: dict, bbox: np.ndarray | list) -> dict:
    """Build a stable metrology CSV record for both file-based and combined paths."""
    bbox_value = bbox.tolist() if hasattr(bbox, "tolist") else list(bbox)
    return {
        "filename": filename,
        "is_memory": measurements["is_memory"],
        "void_ratio_defect": measurements["void_ratio_defect"],
        "solder_extrusion_defect": measurements["solder_extrusion_defect"],
        "pad_misalignment_defect": measurements["pad_misalignment_defect"],
        "bb": bbox_value,
        "BLT": measurements["blt"],
        "Pad_misalignment": measurements["pad_misalignment"],
        "Void_to_solder_ratio": measurements["void_solder_ratio"],
        "solder_extrusion_copper_pillar": [
            measurements["solder_extrusion_left"],
            measurements["solder_extrusion_right"],
            measurements["solder_extrusion_front"],
            measurements["solder_extrusion_back"],
        ],
        "pillar_width": measurements["pillar_width"],
        "pillar_height": measurements["pillar_height"],
    }


def records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Create a metrology dataframe with stable column ordering."""
    records = sorted(records, key=lambda record: record["filename"])
    return pd.DataFrame.from_records(records, columns=METROLOGY_COLUMNS)


def extract_pred_index(filename: str) -> int | None:
    """Extract bbox index from a `pred_<idx>.nii.gz` style filename."""
    match = re.search(r"pred_(\d+)\.nii(?:\.gz)?$", filename)
    if match is None:
        return None
    return int(match.group(1))


def save_segmentation_region_manifest(results: list, output_path: Path) -> dict[str, dict]:
    """Persist compact per-bbox region metadata for report reconstruction."""
    rows = []
    for result in sorted(results, key=lambda item: item.bbox_index):
        rows.append(
            {
                "bbox_index": int(result.bbox_index),
                "filename": f"pred_{int(result.bbox_index)}.nii.gz",
                "original_bbox": [int(v) for v in np.asarray(result.original_bbox).tolist()],
                "expanded_bbox": [int(v) for v in result.expanded_bbox],
                "crop_shape": [int(v) for v in result.crop_shape],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return {str(row["filename"]): row for row in rows}


def process_segmentation_and_metrology_combined(
    engine: SegmentationInference,
    volume: np.ndarray,
    bboxes: np.ndarray,
    metrology_output_dir: Path,
    clean_mask: bool = MAKE_CLEAN_DEFAULT,
    save_predictions: bool = False,
) -> tuple[list, pd.DataFrame]:
    """
    Combined segmentation and metrology processing per bbox.

    Processes each bbox: segment -> compute metrology.
    This reduces I/O by computing metrology directly from in-memory predictions
    instead of saving then reloading each prediction file. Optional per-bbox
    prediction NIfTIs can still be kept for debugging or legacy artifact needs.

    Args:
        engine: Initialized SegmentationInference instance
        volume: Full 3D volume array
        bboxes: Array of bboxes [N, 6] format [x_min, x_max, y_min, y_max, z_min, z_max]
        metrology_output_dir: Directory to save metrology CSV
        clean_mask: Whether to apply morphological cleaning to masks

    Returns:
        Tuple of (segmentation_results, metrology_dataframe)
    """
    seg_output_dir = metrology_output_dir.parent / "mmt"
    seg_output_dir.mkdir(parents=True, exist_ok=True)
    if save_predictions:
        (seg_output_dir / "pred").mkdir(parents=True, exist_ok=True)
    metrology_output_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the normal segmentation path so batching, direct/sliding routing,
    # and per-run instrumentation stay identical to the baseline pipeline.
    results = segment_bboxes(
        engine=engine,
        volume=volume,
        bboxes=bboxes,
        save_dir=seg_output_dir,
        save_predictions=save_predictions,
        save_raw_crops=False,
    )

    records = []
    for result in results:
        try:
            measurements = compute_metrology_from_array(
                result.prediction.copy(),
                clean_mask=clean_mask,
            )
            records.append(
                build_metrology_record(
                    filename=f"pred_{result.bbox_index}.nii.gz",
                    measurements=measurements,
                    bbox=result.original_bbox,
                )
            )
        except Exception as e:
            log(f"Error computing metrology for bbox {result.bbox_index}: {e}", level="error")

    metrology_df = records_to_dataframe(records)
    metrology_df.to_csv(metrology_output_dir / "metrology.csv", index=False)
    return results, metrology_df


def process_metrology(
    input_folder: Path,
    output_folder: Path,
    clean_out_path: Path,
    clean_mask: bool,
    bb_3d_list: np.ndarray,
) -> pd.DataFrame:
    """
    Process 3D segmented files and compute metrology measurements.

    Iterates through all .nii.gz segmentation files, computes metrology
    measurements (BLT, void ratio, pad misalignment, etc.).

    Args:
        input_folder: Path to folder containing segmented .nii.gz files
        output_folder: Path to save metrology CSV reports
        clean_out_path: Path to save cleaned/processed masks
        clean_mask: Whether to apply morphological cleaning to masks
        bb_3d_list: Array of 3D bounding boxes [num_classes, num_samples]

    Returns:
        DataFrame with metrology results for all samples

    Output Files:
        - metrology.csv: Metrology data for all samples (memory and logic dies)
    """
    segmented_files = sorted(input_folder.rglob("*.nii.gz"))
    num_files = len(segmented_files)
    log(f"Number of segmented files to process: {num_files}")

    # Create output folders if they don't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    if clean_mask:
        clean_out_path.mkdir(parents=True, exist_ok=True)

    records = []

    for i, segmented_file in enumerate(segmented_files):
        log(f"Processing file {i + 1}/{num_files}: {segmented_file}", level="debug")
        rel_path = segmented_file.relative_to(input_folder)
        output_path = clean_out_path / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            measurements = compute_metrology_info(
                nii_file=str(segmented_file),
                output_path=str(output_path) if clean_mask else None,
                clean_mask=clean_mask,
            )

            bbox_index = extract_pred_index(rel_path.name)
            bbox_value = bb_3d_list[0, bbox_index] if bbox_index is not None else bb_3d_list[0, i]
            records.append(
                build_metrology_record(
                    filename=str(rel_path),
                    measurements=measurements,
                    bbox=bbox_value,
                )
            )

        except Exception as e:
            log(f"Error processing file {segmented_file}: {e}", level="error")
            log(f"Stack trace: {traceback.format_exc()}", level="debug")
            continue

    metrology_df = records_to_dataframe(records)
    metrology_df.to_csv(output_folder / "metrology.csv", index=False)

    return metrology_df


def process_single_file(
    input_file: str | Path,
    config: PipelineConfig | None = None,
    force: bool = False,
) -> dict:
    """Process a single input file through the pipeline.

    This is the main entry point for processing a single file. It can be called
    directly as a module or via CLI.

    Args:
        input_file: Path to input NIfTI file
        config: Pipeline configuration (uses DEFAULT_CONFIG if None)
        force: Force reprocessing even if already completed

    Returns:
        Dictionary with processing result:
            - status: "completed", "skipped", or "failed"
            - output_dir: Path to output directory
            - reason: Reason for status (if skipped or failed)
            - metrics: Processing metrics (if completed)
    """
    config = config or DEFAULT_CONFIG
    input_file = Path(input_file)

    # Initialize logbook for job tracking
    logbook = PipelineLogbook(config.output_base)

    # Check if should process
    tag = config.tag
    should_run, reason = logbook.should_process(input_file, force=force, tag=tag)
    if not should_run:
        return {"status": "skipped", "input_file": str(input_file), "reason": reason}

    # Get output directory using sample ID extraction + tag
    outfolder = logbook.get_output_dir(input_file, tag=tag)
    outfolder.mkdir(parents=True, exist_ok=True)

    # Mark job as started
    logbook.mark_started(
        input_file,
        outfolder,
        config={
            "tag": tag,
            "detection_model": config.detection_model,
            "segmentation_model": config.segmentation_model,
            "use_combined_seg_metrology": config.use_combined_seg_metrology,
            "use_inmemory_detection": config.use_inmemory_detection,
            "detection_image_workers": config.detection_image_workers,
            "detection_batch_prefetch": config.detection_batch_prefetch,
            "use_ramdisk_detection": config.use_ramdisk_detection,
            "clean_mask": config.clean_mask,
            "use_compile": config.use_compile,
            "compile_mode": config.compile_mode,
        },
        tag=tag,
    )

    views = ["view1", "view2"]

    # Initialize logger and metrics for this task
    logger = PipelineLogger(log_file=outfolder / "execution.log", verbose=config.verbose)
    metrics = PipelineMetrics(task_id=str(input_file))

    try:
        with metrics.phase("Folder Creation"):
            folder_structure = create_folder_structure(str(outfolder), views)

        # Load volume data once for reuse across pipeline stages
        log(f"Loading volume: {input_file}")
        img = nib.load(input_file)
        data3d = img.get_fdata()
        if data3d.ndim == 4:
            data3d = data3d[..., 3]
        data_max = data3d.max()

        if config.use_inmemory_detection:
            # True in-memory detection pipeline: slice -> model -> merge without
            # writing JPGs or per-slice detection text files.
            with metrics.phase("2D Detection Inference") as phase:
                log("Running object detection inference (true in-memory)...")
                detection_model = load_yolo_model(config.detection_model)

                log("Running detection on view1 from volume slices...")
                view1_detections, view1_num_slices, view1_image_dimensions = run_yolo_detection_inmemory_from_volume(
                    model=detection_model,
                    volume=data3d,
                    view=0,
                    data_max=data_max,
                    image_workers=config.detection_image_workers,
                    batch_prefetch=config.detection_batch_prefetch,
                )

                log("Running detection on view2 from volume slices...")
                view2_detections, view2_num_slices, _ = run_yolo_detection_inmemory_from_volume(
                    model=detection_model,
                    volume=data3d,
                    view=1,
                    data_max=data_max,
                    image_workers=config.detection_image_workers,
                    batch_prefetch=config.detection_batch_prefetch,
                )

                num_slices = view1_num_slices + view2_num_slices
                phase.complete(count=num_slices)

            with metrics.phase("3D Bounding Box Generation") as phase:
                log("Generating 3D bounding boxes (in-memory)...")

                bb_3d = generate_bb3d_inmemory(
                    view1_detections=view1_detections,
                    view2_detections=view2_detections,
                    view1_num_slices=view1_num_slices,
                    view2_num_slices=view2_num_slices,
                    image_dimensions=view1_image_dimensions,
                    output_file=str(outfolder / "3d_bounding_boxes.npy"),
                    is_normalized=False,
                )
                bb_3d = normalize_bboxes(bb_3d)

                # Save as single array
                np.save(outfolder / "bb3d.npy", bb_3d)

                # Free detection memory after merging
                del view1_detections, view2_detections

                effective_slices = view1_num_slices + view2_num_slices
                phase.complete(count=effective_slices)

            # Explicit cleanup before loading segmentation model
            import gc

            gc.collect()

        elif config.use_ramdisk_detection:
            # Ramdisk-backed detection path: keep baseline JPEG generation and
            # YOLO `source=<folder>` behavior, but move temporary images off
            # physical disk and avoid writing per-slice detection text files.
            ramdisk_parent = "/dev/shm" if Path("/dev/shm").is_dir() else None
            with tempfile.TemporaryDirectory(prefix="3ddl_detect_", dir=ramdisk_parent) as ramdisk_root:
                ramdisk_structure = create_folder_structure(ramdisk_root, views)

                with metrics.phase("NII to JPG Conversion") as phase:
                    log("Converting NII to JPG in temporary ramdisk workspace...")
                    view1_slices = nii2jpg(data3d, ramdisk_structure["view1"]["input_images"], 0, data_max)
                    view2_slices = nii2jpg(data3d, ramdisk_structure["view2"]["input_images"], 1, data_max)
                    num_slices = view1_slices + view2_slices
                    phase.complete(count=num_slices)

                with metrics.phase("2D Detection Inference") as phase:
                    log("Running object detection inference (ramdisk folder path)...")
                    detection_model = load_yolo_model(config.detection_model)

                    log("Running detection on view1 from temporary folder...")
                    view1_detections, view1_num_slices, view1_image_dimensions = run_yolo_detection_folder_to_memory(
                        model=detection_model,
                        input_folder=ramdisk_structure["view1"]["input_images"],
                    )

                    log("Running detection on view2 from temporary folder...")
                    view2_detections, view2_num_slices, _ = run_yolo_detection_folder_to_memory(
                        model=detection_model,
                        input_folder=ramdisk_structure["view2"]["input_images"],
                    )

                    num_slices = view1_num_slices + view2_num_slices
                    phase.complete(count=num_slices)

                with metrics.phase("3D Bounding Box Generation") as phase:
                    log("Generating 3D bounding boxes (ramdisk detections)...")
                    bb_3d = generate_bb3d_inmemory(
                        view1_detections=view1_detections,
                        view2_detections=view2_detections,
                        view1_num_slices=view1_num_slices,
                        view2_num_slices=view2_num_slices,
                        image_dimensions=view1_image_dimensions,
                        output_file=str(outfolder / "3d_bounding_boxes.npy"),
                        is_normalized=False,
                    )
                    bb_3d = normalize_bboxes(bb_3d)
                    np.save(outfolder / "bb3d.npy", bb_3d)

                    del view1_detections, view2_detections

                    effective_slices = view1_num_slices + view2_num_slices
                    phase.complete(count=effective_slices)

            import gc

            gc.collect()

        else:
            # File-based detection pipeline (original behavior)
            with metrics.phase("NII to JPG Conversion") as phase:
                log("Converting NII to JPG...")
                view1_slices = nii2jpg(data3d, folder_structure["view1"]["input_images"], 0, data_max)
                view2_slices = nii2jpg(data3d, folder_structure["view2"]["input_images"], 1, data_max)
                num_slices = view1_slices + view2_slices
                phase.complete(count=num_slices)

            with metrics.phase("2D Detection Inference") as phase:
                log("Running object detection inference...")

                for view in views:
                    input_folder = folder_structure[view]["input_images"]
                    output_folder = folder_structure[view]["detections"]

                    log(f"Running detection on {view}...")
                    run_yolo_detection(
                        model_path=config.detection_model,
                        input_folder=input_folder,
                        output_folder=output_folder,
                    )

                view1_dir = Path(folder_structure["view1"]["input_images"])
                view2_dir = Path(folder_structure["view2"]["input_images"])
                num_slices = len(list(view1_dir.glob("*.jpg"))) + len(list(view2_dir.glob("*.jpg")))
                phase.complete(count=num_slices)

            with metrics.phase("3D Bounding Box Generation") as phase:
                log("Generating 3D bounding boxes...")
                view1_dir = Path(folder_structure["view1"]["input_images"])
                view2_dir = Path(folder_structure["view2"]["input_images"])
                view1_num_slices = len(list(view1_dir.glob("*.jpg")))
                view2_num_slices = len(list(view2_dir.glob("*.jpg")))
                view1_params = [
                    folder_structure["view1"]["detections"],
                    folder_structure["view1"]["input_images"],
                    0,
                    view1_num_slices,
                ]
                view2_params = [
                    folder_structure["view2"]["detections"],
                    folder_structure["view2"]["input_images"],
                    0,
                    view2_num_slices,
                ]
                bb_3d = generate_bb3d(
                    view1_params, view2_params, str(outfolder / "3d_bounding_boxes.npy"), is_normalized=False
                )
                bb_3d = normalize_bboxes(bb_3d)
                np.save(outfolder / "bb3d.npy", bb_3d)
                effective_slices = view1_num_slices + view2_num_slices
                phase.complete(count=effective_slices)

        num_bboxes = bb_3d.shape[0]
        log(f"3D bounding boxes shape: {bb_3d.shape}")

        seg_output_dir = outfolder / "mmt"
        seg_config = SegmentationConfig(
            use_trt=config.use_trt,
            trt_engine_path=config.trt_engine_path,
            use_compile=config.use_compile,
            compile_mode=config.compile_mode,
            direct_roi_size=config.direct_roi_size,
            use_adaptive_margin=config.use_adaptive_margin,
            use_guarded_adaptive_margin=config.use_guarded_adaptive_margin,
            guard_max_loss_xy=config.guard_max_loss_xy,
            guard_max_loss_z=config.guard_max_loss_z,
            force_sliding_window=config.force_sliding_window,
            force_direct_full_crop=config.force_direct_full_crop,
        )
        engine = SegmentationInference(config.segmentation_model, seg_config)

        if config.use_combined_seg_metrology:
            # Combined segmentation + metrology in single phase
            with metrics.phase("3D Segmentation + Metrology") as phase:
                log("Running combined 3D segmentation and metrology...")

                results, metrology_df = process_segmentation_and_metrology_combined(
                    engine=engine,
                    volume=data3d,
                    bboxes=bb_3d,
                    metrology_output_dir=outfolder / "metrology",
                    clean_mask=config.clean_mask,
                    save_predictions=config.save_bump_predictions,
                )

                # Assemble into full volume and save
                full_segmentation = assemble_full_volume(results, data3d.shape)
                nib.save(
                    nib.Nifti1Image(full_segmentation, np.eye(4)),
                    str(outfolder / "segmentation.nii.gz"),
                )
                log(f"Combined segmentation + metrology completed, saved to {outfolder}")
                phase.complete(count=num_bboxes)

        else:
            # Separate segmentation and metrology phases
            with metrics.phase("3D Segmentation") as phase:
                log("Running 3D segmentation...")

                # The legacy separate path still saves per-bbox prediction masks
                # because file-based metrology consumes them.
                (seg_output_dir / "pred").mkdir(parents=True, exist_ok=True)
                results = segment_bboxes(
                    engine,
                    data3d,
                    bb_3d,
                    save_dir=seg_output_dir,
                    save_predictions=True,
                    save_raw_crops=False,
                )

                # Assemble into full volume and save
                full_segmentation = assemble_full_volume(results, data3d.shape)
                nib.save(
                    nib.Nifti1Image(full_segmentation, np.eye(4)),
                    str(outfolder / "segmentation.nii.gz"),
                )
                log(f"Segmentation completed, saved to {outfolder}")
                phase.complete(count=num_bboxes)

            with metrics.phase("Metrology") as phase:
                log("Computing metrology measurements...")
                metrology_input_path = seg_output_dir / "pred"
                if not metrology_input_path.exists():
                    log(f"Metrology input folder does not exist: {metrology_input_path}", level="error")
                else:
                    mask_files = list(metrology_input_path.rglob("*.nii.gz"))
                    log(f"Found {len(mask_files)} .nii.gz files to process")
                    if len(mask_files) == 0:
                        log("No files found! Check if previous steps generated files.", level="warning")

                metrology_df = process_metrology(
                    input_folder=metrology_input_path,
                    output_folder=outfolder / "metrology",
                    clean_out_path=outfolder / "metrology" / "clean",
                    clean_mask=config.clean_mask,
                    bb_3d_list=bb_3d[np.newaxis, ...],
                )
                phase.complete(count=len(results))

        region_lookup = save_segmentation_region_manifest(results, outfolder / "mmt" / "segmentation_regions.json")

        with metrics.phase("GLTF Generation") as phase:
            from gltf_utils import generate_all_bump_gltfs_from_results, generate_gltf_for_sample

            # Use outfolder directly (already includes tag if present)
            # Extract the folder name and base to match gltf_utils API
            output_base_path = outfolder.parent
            sample_folder_name = outfolder.name
            num_bumps = 0
            try:
                log("Generating main GLTF model...")
                generate_gltf_for_sample(sample_folder_name, output_base_path)
                log("Generating bump GLTF models from in-memory segmentation results...")
                num_bumps = generate_all_bump_gltfs_from_results(
                    sample_folder_name,
                    results,
                    output_base_path,
                )
            except Exception as e:
                log(f"GLTF generation failed: {e}", level="warning")
            phase.complete(count=num_bumps)

        report_path = outfolder / "metrology" / "metrology_report.pdf"
        with metrics.phase("Report Generation"):
            generate_pdf_report(
                str(outfolder / "metrology" / "metrology.csv"),
                str(report_path),
                input_filename=input_file.name,
                enable_ai_analysis=config.enable_ai_analysis,
                input_volume_path=str(input_file),
                segmentation_volume_path=str(outfolder / "segmentation.nii.gz"),
                region_manifest_path=str(outfolder / "mmt" / "segmentation_regions.json"),
                input_volume_array=data3d,
                segmentation_volume_array=full_segmentation,
                region_lookup=region_lookup,
            )

        # Save metrics for this task
        metrics.write_log(str(outfolder / "timing.log"))
        metrics.save(str(outfolder / "metrics.json"))

        log(f"Successfully processed {input_file}")
        log(f"Report generated at {report_path.parent}")

        # Mark job as completed
        logbook.mark_completed(input_file, metrics.summary(), tag=tag)

        return {
            "status": "completed",
            "input_file": str(input_file),
            "output_dir": str(outfolder),
            "metrics": metrics.summary(),
        }

    except Exception as e:
        error_msg = f"{e}\n{traceback.format_exc()}"
        log(f"Error processing {input_file}: {e}", level="error")

        # Still save partial metrics on error
        metrics.write_log(str(outfolder / "timing.log"))
        metrics.save(str(outfolder / "metrics.json"))

        # Mark job as failed
        logbook.mark_failed(input_file, str(e), tag=tag)

        return {
            "status": "failed",
            "input_file": str(input_file),
            "output_dir": str(outfolder),
            "error": error_msg,
        }

    finally:
        logger.close()


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="3D-IntelliScan Pipeline: Detection, Segmentation, and Metrology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Process all files in files.txt
  python main.py /path/to/input.nii       # Process single file
  python main.py /path/to/input.nii --force  # Force reprocessing
  python main.py --list                   # List all jobs in logbook
        """,
    )
    parser.add_argument("input_file", nargs="?", help="Input NIfTI file to process")
    parser.add_argument("--force", "-f", action="store_true", help="Force reprocessing")
    parser.add_argument("--output", "-o", default="output", help="Output base directory")
    parser.add_argument("--verbose", "-v", action="store_true", default=True, help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode (only essential output)")
    parser.add_argument("--list", action="store_true", help="List all jobs in logbook")
    parser.add_argument(
        "--combined-seg-metrology",
        dest="combined_seg_metrology",
        action="store_true",
        default=None,
        help="Use combined segmentation + in-memory metrology path (default)",
    )
    parser.add_argument(
        "--separate-seg-metrology",
        dest="combined_seg_metrology",
        action="store_false",
        help="Use legacy separate segmentation and file-based metrology path",
    )
    parser.add_argument(
        "--inmemory",
        dest="inmemory_detection",
        action="store_true",
        default=None,
        help="Use true in-memory detection and reuse one YOLO model across both views (default)",
    )
    parser.add_argument(
        "--file-detection",
        dest="inmemory_detection",
        action="store_false",
        help="Use the legacy file-based detection path with JPG slices and detection txt outputs",
    )
    parser.add_argument(
        "--ramdisk-detection",
        action="store_true",
        help="Use temporary ramdisk JPGs with YOLO folder input and in-memory bbox merging",
    )
    parser.add_argument(
        "--detection-image-workers",
        type=int,
        default=None,
        help="Number of CPU workers for in-memory slice-to-image preparation; only affects --inmemory (default: 4)",
    )
    parser.add_argument(
        "--detection-batch-prefetch",
        dest="detection_batch_prefetch",
        action="store_true",
        default=None,
        help="Overlap preparation of the next in-memory detection batch with current YOLO inference (default)",
    )
    parser.add_argument(
        "--no-detection-batch-prefetch",
        dest="detection_batch_prefetch",
        action="store_false",
        help="Disable next-batch prefetch for in-memory detection",
    )
    parser.add_argument("--adaptive-margin", action="store_true", help="Enable experimental adaptive bbox margin for segmentation")
    parser.add_argument(
        "--guarded-adaptive-margin",
        action="store_true",
        help="Enable guarded adaptive bbox margin with conservative context-loss limits",
    )
    parser.add_argument(
        "--guard-max-loss-xy",
        type=int,
        default=4,
        help="Maximum total context removed on x/y before guarded adaptive margin falls back to baseline",
    )
    parser.add_argument(
        "--guard-max-loss-z",
        type=int,
        default=8,
        help="Maximum total context removed on z before guarded adaptive margin falls back to baseline",
    )
    parser.add_argument("--force-sliding-window", action="store_true", help="Force sliding-window inference for all segmentation crops")
    parser.add_argument("--force-direct-full-crop", action="store_true", help="Force single-pass full-crop PyTorch inference to isolate path effects")
    parser.add_argument(
        "--direct-roi-size",
        type=int,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Experimental direct-path ROI size. Keeps sliding-window roi_size unchanged and only enlarges the direct padded shell.",
    )
    parser.add_argument(
        "--save-bump-predictions",
        action="store_true",
        help="Keep per-bbox prediction NIfTIs under mmt/pred even on the combined path",
    )
    parser.add_argument("--trt", action="store_true", help="Use TensorRT backend for segmentation")
    parser.add_argument("--trt-engine", type=str, default=None, help="Path to TensorRT engine file")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile() for the PyTorch segmentation backend")
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        help="torch.compile mode for the PyTorch segmentation backend",
    )
    parser.add_argument("--ai-analysis", action="store_true", help="Enable AI analysis in PDF report")
    parser.add_argument("--detection-model", type=str, default="models/detection_model.pt", help="Path to YOLO detection model")
    parser.add_argument("--segmentation-model", type=str, default="models/segmentation_model.ckpt", help="Path to segmentation model checkpoint")
    parser.add_argument("--tag", "-t", type=str, default="", help="Tag to differentiate output folders (e.g. output/SN002_<tag>/)")

    args = parser.parse_args()

    if args.inmemory_detection is True and args.ramdisk_detection:
        parser.error("--inmemory and --ramdisk-detection cannot be used together")
    if args.inmemory_detection is False and args.ramdisk_detection:
        parser.error("--file-detection and --ramdisk-detection cannot be used together")
    if args.adaptive_margin and args.guarded_adaptive_margin:
        parser.error("--adaptive-margin and --guarded-adaptive-margin cannot be used together")
    if args.force_sliding_window and args.force_direct_full_crop:
        parser.error("--force-sliding-window and --force-direct-full-crop cannot be used together")
    if args.trt and args.force_direct_full_crop:
        parser.error("--force-direct-full-crop is only supported with the PyTorch segmentation backend")
    if args.direct_roi_size is not None and args.trt:
        parser.error("--direct-roi-size experiments are only supported with the PyTorch segmentation backend")
    if args.trt and args.compile:
        parser.error("--compile is only supported with the PyTorch segmentation backend")

    # Create config from args
    config = PipelineConfig(
        output_base=args.output,
        detection_model=args.detection_model,
        segmentation_model=args.segmentation_model,
        tag=args.tag,
        use_combined_seg_metrology=(
            DEFAULT_CONFIG.use_combined_seg_metrology
            if args.combined_seg_metrology is None
            else args.combined_seg_metrology
        ),
        verbose=not args.quiet,
        use_inmemory_detection=(
            DEFAULT_CONFIG.use_inmemory_detection
            if args.inmemory_detection is None
            else args.inmemory_detection
        ),
        detection_image_workers=(
            DEFAULT_CONFIG.detection_image_workers
            if args.detection_image_workers is None
            else max(1, args.detection_image_workers)
        ),
        detection_batch_prefetch=(
            DEFAULT_CONFIG.detection_batch_prefetch
            if args.detection_batch_prefetch is None
            else args.detection_batch_prefetch
        ),
        use_ramdisk_detection=args.ramdisk_detection,
        use_adaptive_margin=args.adaptive_margin,
        use_guarded_adaptive_margin=args.guarded_adaptive_margin,
        guard_max_loss_xy=args.guard_max_loss_xy,
        guard_max_loss_z=args.guard_max_loss_z,
        force_sliding_window=args.force_sliding_window,
        force_direct_full_crop=args.force_direct_full_crop,
        direct_roi_size=tuple(args.direct_roi_size) if args.direct_roi_size is not None else None,
        save_bump_predictions=args.save_bump_predictions,
        use_trt=args.trt,
        trt_engine_path=args.trt_engine,
        use_compile=args.compile,
        compile_mode=args.compile_mode,
        enable_ai_analysis=args.ai_analysis,
    )

    # List jobs mode
    if args.list:
        logbook = PipelineLogbook(config.output_base)
        jobs = logbook.list_jobs()
        if not jobs:
            print("No jobs in logbook")
        else:
            print(f"{'Status':<12} {'Sample':<20} {'Started':<20} {'Input File'}")
            print("-" * 90)
            for job in jobs:
                sample_id = PipelineLogbook.extract_sample_id(job.get("input_path", ""))
                job_tag = job.get("config", {}).get("tag", "")
                label = f"{sample_id}_{job_tag}" if job_tag else sample_id
                status = job.get("status", "unknown")
                started = job.get("started_at", "")
                input_path = job.get("input_path", "")
                print(f"{status:<12} {label:<20} {started:<20} {input_path}")
        return

    # Determine input files
    if args.input_file:
        # Single file mode
        input_files = [args.input_file]
    else:
        # Batch mode from files.txt
        try:
            with open("files.txt") as f:
                input_files = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        except FileNotFoundError:
            print("Error: No input file specified and files.txt not found")
            print("Usage: python main.py <input_file> or create files.txt")
            return

    print(f"Found {len(input_files)} file(s) to process")

    # Process each file
    for input_file in input_files:
        print(f"\n{'=' * 60}")
        print(f"Processing: {input_file}")
        print(f"{'=' * 60}")

        result = process_single_file(input_file, config=config, force=args.force)

        if result["status"] == "skipped":
            print(f"Skipped: {result['reason']}")
        elif result["status"] == "completed":
            print(f"Completed: output at {result['output_dir']}")
        else:
            print(f"Failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
