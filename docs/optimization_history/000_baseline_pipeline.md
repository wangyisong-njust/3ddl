# 000 Baseline Pipeline

## Objective

Record the current default pipeline behavior that later optimizations should be compared against.

## Target Files

- `intelliscan/main.py`
- `intelliscan/detection.py`
- `intelliscan/merge.py`
- `intelliscan/segmentation.py`
- `intelliscan/metrology.py`
- `intelliscan/utils.py`
- `wp5-seg/train.py`
- `wp5-seg/eval.py`

## Code Change Summary

- No new code. This note is the baseline reference.

## Why This Exists

- Optimization claims are not meaningful without a stable baseline path.

## Baseline Behavior

- `intelliscan` default flow is:
  - load NIfTI volume
  - convert to JPG slices
  - run file-based YOLO detection
  - merge 2D detections into 3D boxes
  - segment each bbox
  - save per-bbox crops and predictions under `mmt/`
  - run metrology from saved masks
  - generate report
- Current default flags in `PipelineConfig`:
  - `use_inmemory_detection=False`
  - `use_combined_seg_metrology=False`
  - `use_trt=False`

## Optimized Behavior

- None. This note anchors comparisons.

## How It Was Tested

- Code inspection
- Existing smoke run
- Existing formal SN009 baseline run

## Sample(s) Used

- `SN21_I33_image.nii` smoke run
- `SN009` formal run

## Key Timing Numbers

| Sample | Total | Detection | Segmentation | Metrology |
| --- | --- | --- | --- | --- |
| SN21_I33 smoke | `2.32s` | `1.95s` | `0.12s` | `0.07s` |
| SN009 formal baseline | `93.23s` | `14.85s` | `33.79s` | `16.95s` |

## Accuracy / Quality Impact

- Not applicable. This is the reference path.

## Risks / Caveats

- Default pipeline is I/O-heavy.
- The current default path does not use several optimization hooks already present in code.

## Conclusion

- The current baseline is a valid comparison target for future changes, but it already contains multiple optional optimization branches that need explicit activation and testing.

## Evidence Paths

- `intelliscan/README.md`
- `intelliscan/output_smoke/images/metrics.json`
- `intelliscan/output_formal/SN009_current_gpu/metrics.json`
