# 040 Batch Inference

## Objective

Reduce per-bbox segmentation overhead by stacking multiple crops into a single forward pass.

## Target Files

- `intelliscan/segmentation.py`

## Code Change Summary

- `segment_bboxes()` separates crops into:
  - `batch_items` that fit the ROI
  - `large_items` that still require sliding-window inference
- Batchable crops are symmetrically padded to ROI size and stacked into batches.

## Why This Change Was Made

- Many small crops should be cheaper as batched inference than as many isolated forward passes.

## Baseline Behavior

- Per-bbox inference without meaningful batching when crops are classified as oversized.

## Optimized Behavior

- Use batched direct inference when crops fit within `roi_size=(112, 112, 80)`.

## How It Was Tested

- Code inspection
- Formal SN009 run log inspection

## Sample(s) Used

- `SN009`

## Key Timing Numbers

- SN009 baseline log reports:
  - `Batchable crops: 0`
  - `oversized crops: 114`

## Accuracy / Quality Impact

- No separate local quality artifact exists for the batch path, because it did not activate on SN009.

## Risks / Caveats

- Current `margin=15` expansion causes the batch path to be bypassed for the tested formal sample.
- This optimization exists in code but was ineffective on the current measured case.

## Conclusion

- Batch inference implementation is present, but adaptive crop sizing is needed before it becomes a meaningful real-sample optimization.

## Evidence Paths

- `intelliscan/segmentation.py`
- `intelliscan/output_formal/SN009_current_gpu/execution.log`
