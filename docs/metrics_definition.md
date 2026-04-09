# Metrics Definition

## Evidence Status Labels

- `Validated`: directly read from a file in this workspace
- `Computed in this pass`: recomputed from files in this workspace during documentation setup
- `Reported but not revalidated in this pass`: present in an existing report or saved artifact, but raw reproduction inputs are missing or were not rerun
- `User-reported result`: came from the user and is not backed by local files

## Pipeline Timing Metrics

Source format: `intelliscan/.../metrics.json`

- `total_elapsed`: sum of recorded phase durations
- `NII to JPG Conversion`: time spent writing slice images
- `2D Detection Inference`: detection pass over all generated slices
- `3D Bounding Box Generation`: merge of per-view detections into 3D boxes
- `3D Segmentation`: per-bbox 3D segmentation stage
- `Metrology`: metrology measurements from predicted masks
- `Report Generation`: PDF report generation

## Segmentation Comparison Metrics

Used when comparing two saved `segmentation.nii.gz` volumes.

- `voxel_accuracy`: fraction of equal voxels across the full volume
- `class Dice`: per-class overlap, more useful than voxel accuracy for defect-sensitive classes
- `class IoU`: secondary overlap measure

Important interpretation rule:

- High `voxel_accuracy` is not enough when class `0` dominates the volume.
- Always inspect class `3` and class `4` when evaluating acceleration or compression changes.

## Metrology Comparison Metrics

Used when comparing two `metrology.csv` outputs for the same sample.

- `BLT`
- `Pad_misalignment`
- `Void_to_solder_ratio`
- `pillar_width`
- `pillar_height`
- boolean defect columns:
  - `void_ratio_defect`
  - `solder_extrusion_defect`
  - `pad_misalignment_defect`

Preferred comparison summary:

- mean absolute delta for continuous fields
- defect label flip count for boolean fields

## Required Comparison Protocol For Future Experiments

Every future optimization experiment should record:

1. Baseline configuration and variant configuration
2. Same sample set for both runs
3. Same hardware and visible GPU selection
4. Same output type and post-processing path
5. Raw evidence paths such as `metrics.json`, `timing.log`, `metrology.csv`, and if relevant `segmentation.nii.gz`

If any of those are missing, the result should be marked `non-comparable` or `reported only`.
