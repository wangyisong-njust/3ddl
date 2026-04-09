# 050 Sliding Window Bypass

## Objective

Skip MONAI `sliding_window_inference` when a crop already fits the model ROI.

## Target Files

- `intelliscan/segmentation.py`
- `wp5-seg/reports/segmentation_acceleration_report.md`

## Code Change Summary

- `SegmentationInference.infer_crop()` and `segment_bboxes()` both contain a direct padded-inference path for crops that fit the ROI.
- Oversized crops still fall back to `sliding_window_inference`.

## Why This Change Was Made

- Sliding-window bookkeeping adds overhead when only one padded crop would be enough.

## Baseline Behavior

- All oversized crops use `sliding_window_inference`.

## Optimized Behavior

- Direct symmetric padding and one forward call for ROI-fitting crops.

## How It Was Tested

- Code inspection
- Formal SN009 log inspection
- Historical repo report cross-reference

## Sample(s) Used

- `SN009` for current code-path validation
- `SN002` in repo report for historical claim

## Key Timing Numbers

Validated:

- SN009 current formal run did not activate this optimization because all `114` expanded crops were oversized.

Reported but not revalidated in this pass:

- Repo report states symmetric padding preserved segmentation quality on SN002 and contributed a small segmentation speed gain.

## Accuracy / Quality Impact

- Current code is written to use symmetric padding, which is the safer behavior.

## Risks / Caveats

- The optimization only matters if crops fit ROI after expansion.
- Historical quality claim from the report should be rerun before reuse in publication material.

## Conclusion

- The bypass exists and is likely useful, but adaptive margin is the real enabling change for current samples.

## Evidence Paths

- `intelliscan/segmentation.py`
- `intelliscan/output_formal/SN009_current_gpu/execution.log`
- `wp5-seg/reports/segmentation_acceleration_report.md`
