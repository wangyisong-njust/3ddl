#!/usr/bin/env python3
"""
2D Object Detection Inference for 3D Semiconductor Inspection

This module performs batch 2D object detection on image slices extracted from 3D
volumes. It uses YOLO models to detect semiconductor components (copper pillars,
solder bumps, etc.) in horizontal and vertical view slices.

The detection workflow processes all images in a directory, generates bounding box
predictions, and saves both coordinate files and visualizations.

Key Features:
- YOLO-based object detection for semiconductor components
- Batch processing of image directories (horizontal/vertical views)
- Automatic bounding box coordinate extraction (class_id, x1, y1, x2, y2)
- Visualization generation for quality control
- Multi-class detection with confidence scoring

Output Format:
- Detection files: <image_name>.txt with format: class_id x1 y1 x2 y2 (one bbox per line)
- Visualizations: Annotated images saved to visualize/ subdirectory
"""

import io
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from utils import log


def load_yolo_model(model_path: str) -> YOLO:
    """Load and return a YOLO model once for reuse across views."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    log(f"Loading YOLO model from: {model_path}")
    return YOLO(model_path)


def _quantize_detections_like_txt(detection_data: np.ndarray) -> np.ndarray:
    """Match the precision loss introduced by the legacy `.txt` detection files.

    The file-based baseline serializes detections with `%d %.2f %.2f %.2f %.2f`
    before merge. Without this step, in-memory branches feed slightly different
    coordinates into `generate_bb3d_inmemory`, which can move borderline boxes.
    """
    if detection_data.size == 0:
        return detection_data

    quantized = detection_data.astype(np.float32, copy=True)
    quantized[:, 0] = np.round(quantized[:, 0])
    quantized[:, 1:] = np.round(quantized[:, 1:] * 100.0) / 100.0
    return quantized


def _extract_detection_array(result, filename: str) -> tuple[np.ndarray, int]:
    """Convert one Ultralytics result to [class_id, x1, y1, x2, y2]."""
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 5), dtype=np.float32), 0

    class_ids = result.boxes.cls.cpu().numpy()
    bboxes = result.boxes.xyxy.cpu().numpy()
    detection_data = np.column_stack((class_ids.reshape(-1, 1), bboxes)).astype(np.float32)
    detection_data = _quantize_detections_like_txt(detection_data)
    return detection_data, int(len(class_ids))


def _slice_to_model_image(slice_data: np.ndarray, data_max: float) -> np.ndarray:
    """Convert one slice to the same in-memory image representation used by file-based YOLO.

    The original pipeline saves float slices to JPEG and then lets YOLO reload them
    from disk. To keep the true in-memory path numerically close to that behavior,
    we reproduce the same PIL transform and JPEG encode/decode roundtrip in memory,
    then decode via OpenCV so Ultralytics receives the same BGR ndarray style that
    the file-based path gets from `source=<folder>`.
    """
    if data_max == 0:
        raise ValueError("The input data contains only zero values, normalization not possible.")

    scaled = (slice_data / data_max) * 255.0
    image = Image.fromarray(scaled).rotate(90, expand=True).transpose(Image.Transpose.FLIP_TOP_BOTTOM).convert("RGB")

    # Mirror the file-based detector input without touching disk.
    jpeg_buffer = io.BytesIO()
    image.save(jpeg_buffer, format="JPEG")
    encoded = np.frombuffer(jpeg_buffer.getvalue(), dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("OpenCV failed to decode the in-memory JPEG image.")
    return decoded


def _resolve_view_slices(
    volume: np.ndarray,
    view: int,
):
    """Return the slice accessor for one detection view."""
    if view == 0:
        num_slices = volume.shape[2]
        slice_func = lambda i: volume[:, :, i]
    elif view == 1:
        num_slices = volume.shape[1]
        slice_func = lambda i: volume[:, i, :]
    else:
        raise ValueError(f"Invalid view parameter: {view}. Must be 0 (axial) or 1 (coronal).")

    return num_slices, slice_func


def iter_detection_batches_from_volume(
    volume: np.ndarray,
    view: int,
    data_max: float,
    batch_size: int,
    image_workers: int = 1,
):
    """Yield detection-ready image batches for one view.

    `image_workers` parallelizes the CPU-side slice -> JPEG -> OpenCV decode
    preparation while preserving per-slice ordering. It does not change the
    model input geometry or the detector merge logic.
    """
    num_slices, slice_func = _resolve_view_slices(volume, view)
    image_workers = max(1, int(image_workers))

    def build_item(index: int) -> tuple[str, np.ndarray]:
        return f"image{index}", _slice_to_model_image(slice_func(index), data_max)

    executor = ThreadPoolExecutor(max_workers=image_workers) if image_workers > 1 else None
    try:
        for batch_start in range(0, num_slices, batch_size):
            batch_stop = min(batch_start + batch_size, num_slices)
            indices = range(batch_start, batch_stop)
            if executor is None:
                yield [build_item(index) for index in indices]
            else:
                yield list(executor.map(build_item, indices))
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


def _prepare_detection_batch(
    slice_func,
    data_max: float,
    batch_start: int,
    batch_size: int,
    num_slices: int,
    image_executor: ThreadPoolExecutor | None,
) -> list[tuple[str, np.ndarray]]:
    """Build one ordered batch of detection-ready images."""
    batch_stop = min(batch_start + batch_size, num_slices)
    indices = range(batch_start, batch_stop)

    def build_item(index: int) -> tuple[str, np.ndarray]:
        return f"image{index}", _slice_to_model_image(slice_func(index), data_max)

    if image_executor is None:
        return [build_item(index) for index in indices]
    return list(image_executor.map(build_item, indices))


def run_yolo_detection_inmemory_from_volume(
    model: YOLO,
    volume: np.ndarray,
    view: int,
    data_max: float,
    batch_size: int = 64,
    image_workers: int = 1,
    batch_prefetch: bool = False,
) -> tuple[dict[str, np.ndarray], int, tuple[int, int]]:
    """Run true in-memory detection directly from a 3D volume view."""
    detections: dict[str, np.ndarray] = {}
    num_detections_total = 0
    num_images_processed = 0
    image_dimensions: tuple[int, int] | None = None

    def process_batch(batch_items: list[tuple[str, np.ndarray]]) -> None:
        nonlocal num_detections_total, num_images_processed, image_dimensions
        if not batch_items:
            return

        names = [name for name, _ in batch_items]
        images = [image for _, image in batch_items]
        if image_dimensions is None and images:
            image_dimensions = (int(images[0].shape[1]), int(images[0].shape[0]))

        results = model.predict(source=images, batch=len(images), verbose=False)
        if len(results) != len(names):
            msg = f"YOLO returned {len(results)} results for {len(names)} in-memory images"
            raise ValueError(msg)

        for name, result in zip(names, results, strict=True):
            detection_data, count = _extract_detection_array(result, name)
            detections[name] = detection_data
            num_detections_total += count
            num_images_processed += 1

    num_slices, slice_func = _resolve_view_slices(volume, view)
    image_workers = max(1, int(image_workers))
    image_executor = ThreadPoolExecutor(max_workers=image_workers) if image_workers > 1 else None
    prefetch_executor = ThreadPoolExecutor(max_workers=1) if batch_prefetch else None

    try:
        if prefetch_executor is None:
            for batch_start in range(0, num_slices, batch_size):
                batch = _prepare_detection_batch(
                    slice_func=slice_func,
                    data_max=data_max,
                    batch_start=batch_start,
                    batch_size=batch_size,
                    num_slices=num_slices,
                    image_executor=image_executor,
                )
                process_batch(batch)
        else:
            next_batch_start = 0

            def submit_batch(start_index: int) -> Future:
                return prefetch_executor.submit(
                    _prepare_detection_batch,
                    slice_func,
                    data_max,
                    start_index,
                    batch_size,
                    num_slices,
                    image_executor,
                )

            next_batch_future = submit_batch(next_batch_start)
            while next_batch_start < num_slices:
                batch = next_batch_future.result()
                next_batch_start += len(batch)
                if next_batch_start < num_slices:
                    next_batch_future = submit_batch(next_batch_start)
                process_batch(batch)
    finally:
        if prefetch_executor is not None:
            prefetch_executor.shutdown(wait=True)
        if image_executor is not None:
            image_executor.shutdown(wait=True)

    if num_images_processed == 0:
        raise ValueError("No in-memory slices were processed for detection.")

    log(f"Processed {num_images_processed} images with {num_detections_total} total detections")
    return detections, num_images_processed, image_dimensions or (0, 0)


def run_yolo_detection_folder_to_memory(
    model: YOLO,
    input_folder: str,
    batch_size: int = 64,
    verbose: bool = False,
) -> tuple[dict[str, np.ndarray], int, tuple[int, int]]:
    """Run folder-based YOLO inference and keep detections in memory.

    This preserves the baseline `source=<folder>` Ultralytics code path while
    avoiding per-slice detection text files. It is used by the ramdisk-backed
    experiment path to separate "disk I/O" from "source dispatch" effects.
    """
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    log(f"Running detection on images in: {input_folder}")
    results = model.predict(source=input_folder, batch=batch_size, verbose=verbose)
    if len(results) == 0:
        raise ValueError(f"No images processed from {input_folder}. Check folder contents.")

    detections: dict[str, np.ndarray] = {}
    num_detections_total = 0
    image_dimensions: tuple[int, int] | None = None

    for result in results:
        filename = Path(result.path).stem
        detection_data, count = _extract_detection_array(result, filename)
        detections[filename] = detection_data
        num_detections_total += count

        if image_dimensions is None:
            height, width = result.orig_shape[:2]
            image_dimensions = (int(width), int(height))

    log(f"Processed {len(results)} images with {num_detections_total} total detections")
    return detections, len(results), image_dimensions or (0, 0)


def run_yolo_detection_inmemory(
    model_path: str,
    input_folder: str,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """
    Perform batch YOLO detection and return results in memory.

    This function processes all images in the input folder and returns
    detection results as a dictionary, avoiding file I/O overhead.

    Args:
        model_path: Path to trained YOLO model weights (.pt file)
        input_folder: Path to folder containing input images (JPG/PNG)
        batch_size: Number of images to process in parallel (default: 64)

    Returns:
        Dictionary mapping image filename (without extension) to detection array.
        Each array has shape [N, 5] with columns [class_id, x1, y1, x2, y2].
        Empty detections return an empty array with shape [0, 5].

    Raises:
        FileNotFoundError: If model_path or input_folder doesn't exist
        ValueError: If input_folder contains no valid images

    Example:
        >>> detections = run_yolo_detection_inmemory(
        ...     model_path="models/detector.pt",
        ...     input_folder="data/view1/input_images"
        ... )
        >>> detections["image0"]  # array([[0, 100.5, 200.3, 150.2, 250.1], ...])
    """
    # Validate inputs
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    # Load YOLO model
    model = load_yolo_model(model_path)

    detections, _, _ = run_yolo_detection_folder_to_memory(
        model=model,
        input_folder=input_folder,
        batch_size=batch_size,
        verbose=True,
    )
    return detections


def run_yolo_detection(
    model_path: str,
    input_folder: str,
    output_folder: str,
    batch_size: int = 64,
) -> None:
    """
    Perform batch YOLO object detection on all images in a folder.

    This function processes all images in the input folder, runs YOLO detection,
    and saves bounding box coordinates as text files.

    Args:
        model_path: Path to trained YOLO model weights (.pt file)
        input_folder: Path to folder containing input images (JPG/PNG)
        output_folder: Path to folder where detection results will be saved
        batch_size: Number of images to process in parallel (default: 16)

    Output Structure:
        output_folder/
            ├── image001.txt  (bounding boxes: class_id x1 y1 x2 y2)
            ├── image002.txt
            └── ...

    Raises:
        FileNotFoundError: If model_path or input_folder doesn't exist
        ValueError: If input_folder contains no valid images

    Example:
        >>> run_yolo_detection(
        ...     model_path="models/detector.pt",
        ...     input_folder="data/view1/input_images",
        ...     output_folder="data/view1/detections"
        ... )
    """
    # Validate inputs
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    log(f"Detection output folder: {output_folder}")

    # Load YOLO model
    model = load_yolo_model(model_path)

    # Run batch prediction
    log(f"Running detection on images in: {input_folder}")
    results = model.predict(source=input_folder, batch=batch_size, verbose=True)

    if len(results) == 0:
        raise ValueError(f"No images processed from {input_folder}. Check folder contents.")

    # Process each result
    num_detections_total = 0
    for result in results:
        filename = Path(result.path).stem

        if result.boxes is None or len(result.boxes) == 0:
            # No detections for this image - create empty file
            output_path = os.path.join(output_folder, f"{filename}.txt")
            Path(output_path).touch()
            continue

        # Extract detection information
        class_ids = result.boxes.cls.cpu().numpy()
        bboxes = result.boxes.xyxy.cpu().numpy()  # xyxy format: x1, y1, x2, y2
        num_detections = len(class_ids)
        num_detections_total += num_detections

        # Save bounding box coordinates: [class_id, x1, y1, x2, y2]
        output_path = os.path.join(output_folder, f"{filename}.txt")
        detection_data = np.column_stack((class_ids.reshape(-1, 1), bboxes))
        np.savetxt(output_path, detection_data, fmt="%d %.2f %.2f %.2f %.2f")

    log(f"✓ Processed {len(results)} images with {num_detections_total} total detections")
    log(f"  Detection files saved to: {output_folder}")


# ============================================================================
# Convenience Functions
# ============================================================================


def detect_from_views(
    model_path: str,
    view1_folder: str,
    view2_folder: str,
    output_base_folder: str,
    batch_size: int = 16,
) -> tuple[str, str]:
    """
    Run detection on both horizontal and vertical view folders.

    Convenience function to process both views with a single model.

    Args:
        model_path: Path to YOLO model weights
        view1_folder: Path to horizontal view images
        view2_folder: Path to vertical view images
        output_base_folder: Base output folder (will create view1/view2 subfolders)
        batch_size: Batch size for inference

    Returns:
        Tuple of (view1_output_folder, view2_output_folder)

    Example:
        >>> detect_from_views(
        ...     model_path="models/detector.pt",
        ...     view1_folder="output/sample1/view1/input_images",
        ...     view2_folder="output/sample1/view2/input_images",
        ...     output_base_folder="output/sample1"
        ... )
    """
    view1_output = os.path.join(output_base_folder, "view1", "detections")
    view2_output = os.path.join(output_base_folder, "view2", "detections")

    log("\n" + "=" * 60)
    log("Running Object Detection - View 1 (Horizontal)")
    log("=" * 60)
    run_yolo_detection(model_path, view1_folder, view1_output, batch_size)

    log("\n" + "=" * 60)
    log("Running Object Detection - View 2 (Vertical)")
    log("=" * 60)
    run_yolo_detection(model_path, view2_folder, view2_output, batch_size)

    return view1_output, view2_output


# ============================================================================
# Test/Demo
# ============================================================================
if __name__ == "__main__":
    import sys

    log("2D Object Detection Module v2.0")
    log("=" * 60)

    if len(sys.argv) < 4:
        log("Usage: python infer_batch_new.py <model_path> <input_folder> <output_folder>")
        log("\nExample:")
        log("  python infer_batch_new.py models/detector.pt data/view1/images output/detections")
        sys.exit(1)

    model_path = sys.argv[1]
    input_folder = sys.argv[2]
    output_folder = sys.argv[3]

    try:
        run_yolo_detection(model_path, input_folder, output_folder)
        log("\n✓ Detection completed successfully!")
    except Exception as e:
        log(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
