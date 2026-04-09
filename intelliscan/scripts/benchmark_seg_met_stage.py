#!/usr/bin/env python3
"""Benchmark the combined segmentation + metrology stage from a cached bb3d."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import process_segmentation_and_metrology_combined  # noqa: E402
from segmentation import SegmentationConfig, SegmentationInference, assemble_full_volume  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-nii", required=True, help="Input 3D NIfTI volume")
    parser.add_argument("--bb3d", required=True, help="Cached bb3d.npy path")
    parser.add_argument("--segmentation-model", default="models/segmentation_model.ckpt")
    parser.add_argument("--repeats", type=int, default=3, help="Benchmark repeats")
    parser.add_argument("--output-json", required=True, help="Where to save the benchmark JSON")
    parser.add_argument(
        "--direct-roi-size",
        nargs=3,
        type=int,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional direct padded ROI; sliding-window ROI stays at the default",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_nii = Path(args.input_nii).resolve()
    bb3d_path = Path(args.bb3d).resolve()
    output_json = Path(args.output_json).resolve()

    volume = np.asanyarray(nib.load(str(input_nii)).dataobj)
    bboxes = np.load(bb3d_path)

    seg_cfg = SegmentationConfig(
        direct_roi_size=tuple(args.direct_roi_size) if args.direct_roi_size is not None else None,
    )
    engine = SegmentationInference(args.segmentation_model, seg_cfg)
    engine.load_model()

    elapsed_seconds: list[float] = []
    last_stats: dict[str, object] | None = None
    for repeat_idx in range(args.repeats):
        with tempfile.TemporaryDirectory(prefix=f"segmet_bench_{repeat_idx}_") as tmpdir:
            start = time.perf_counter()
            results, _ = process_segmentation_and_metrology_combined(
                engine=engine,
                volume=volume,
                bboxes=bboxes,
                metrology_output_dir=Path(tmpdir) / "metrology",
                clean_mask=True,
                save_predictions=False,
            )
            _ = assemble_full_volume(results, volume.shape)
            elapsed_seconds.append(time.perf_counter() - start)
            if engine.last_segmentation_stats is not None:
                last_stats = engine.last_segmentation_stats

    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_nii": str(input_nii),
        "bb3d": str(bb3d_path),
        "direct_roi_size": list(args.direct_roi_size) if args.direct_roi_size is not None else None,
        "repeats": args.repeats,
        "elapsed_seconds": [round(v, 4) for v in elapsed_seconds],
        "mean_seconds": round(float(np.mean(elapsed_seconds)), 4),
        "std_seconds": round(float(np.std(elapsed_seconds)), 4),
        "last_segmentation_stats": last_stats,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
