#!/usr/bin/env python3
"""Compare segmentation and metrology outputs across ablation variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Reference output directory")
    parser.add_argument(
        "--compare",
        required=True,
        nargs="+",
        help="One or more output directories to compare against the reference",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Classes to include in the Dice report",
    )
    return parser.parse_args()


def load_segmentation(path: Path) -> np.ndarray:
    image = nib.load(str(path / "segmentation.nii.gz"))
    return np.asanyarray(image.dataobj)


def load_metrology_rows(path: Path) -> list[dict[str, str]]:
    with open(path / "metrology" / "metrology.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: row["filename"])
    return rows


def dice_for_class(reference: np.ndarray, candidate: np.ndarray, cls: int) -> float:
    ref_mask = reference == cls
    cmp_mask = candidate == cls
    denominator = int(ref_mask.sum()) + int(cmp_mask.sum())
    if denominator == 0:
        return 1.0
    intersection = int(np.logical_and(ref_mask, cmp_mask).sum())
    return 2.0 * intersection / denominator


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def compare_metrology(
    reference_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
) -> dict[str, float | int]:
    candidate_by_name = {row["filename"]: row for row in candidate_rows}
    pad_flip_count = 0
    solder_flip_count = 0
    blt_abs_delta: list[float] = []
    pad_misalignment_abs_delta: list[float] = []

    for reference in reference_rows:
        candidate = candidate_by_name[reference["filename"]]
        pad_flip_count += int(
            parse_bool(reference["pad_misalignment_defect"])
            != parse_bool(candidate["pad_misalignment_defect"])
        )
        solder_flip_count += int(
            parse_bool(reference["solder_extrusion_defect"])
            != parse_bool(candidate["solder_extrusion_defect"])
        )
        blt_abs_delta.append(abs(float(reference["BLT"]) - float(candidate["BLT"])))
        pad_misalignment_abs_delta.append(
            abs(float(reference["Pad_misalignment"]) - float(candidate["Pad_misalignment"]))
        )

    total = len(reference_rows)
    return {
        "count": total,
        "pad_flip_count": pad_flip_count,
        "pad_flip_rate": pad_flip_count / total,
        "solder_flip_count": solder_flip_count,
        "solder_flip_rate": solder_flip_count / total,
        "blt_max_abs_delta": max(blt_abs_delta) if blt_abs_delta else 0.0,
        "blt_mean_abs_delta": float(np.mean(blt_abs_delta)) if blt_abs_delta else 0.0,
        "pad_misalignment_max_abs_delta": (
            max(pad_misalignment_abs_delta) if pad_misalignment_abs_delta else 0.0
        ),
        "pad_misalignment_mean_abs_delta": (
            float(np.mean(pad_misalignment_abs_delta)) if pad_misalignment_abs_delta else 0.0
        ),
    }


def compare_variant(reference_dir: Path, candidate_dir: Path, classes: list[int]) -> dict[str, object]:
    reference_segmentation = load_segmentation(reference_dir)
    candidate_segmentation = load_segmentation(candidate_dir)
    reference_rows = load_metrology_rows(reference_dir)
    candidate_rows = load_metrology_rows(candidate_dir)

    return {
        "reference": str(reference_dir),
        "candidate": str(candidate_dir),
        "voxel_agreement": float((reference_segmentation == candidate_segmentation).mean()),
        "class_dice": {
            str(cls): dice_for_class(reference_segmentation, candidate_segmentation, cls)
            for cls in classes
        },
        "metrology": compare_metrology(reference_rows, candidate_rows),
    }


def main() -> None:
    args = parse_args()
    reference_dir = Path(args.reference).resolve()
    comparisons = {
        Path(candidate).resolve().name: compare_variant(reference_dir, Path(candidate).resolve(), args.classes)
        for candidate in args.compare
    }
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
