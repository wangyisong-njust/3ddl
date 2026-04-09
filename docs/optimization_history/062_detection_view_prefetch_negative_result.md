# Detection View-Prefetch Negative Result

## Objective

Test whether a higher-level detection overlap can improve throughput beyond the kept default path:

- quantized in-memory detection
- `detection_image_workers=4`
- `detection_batch_prefetch=True`

The candidate idea was to overlap batch preparation across `view1` and `view2` while still sharing one YOLO model.

## Why This Was Tested

`061_detection_batch_prefetch_overlap.md` already showed that overlapping next-batch CPU preparation with current YOLO inference is safe and beneficial.

The next natural question was whether a broader `view1/view2` overlap could unlock more throughput. This experiment was designed to answer that directly instead of assuming more concurrency would help.

## Experiment Status

Not kept in code.

The temporary `--detection-view-prefetch` experiment path was benchmarked, then removed because it was slower on the target stage.

## Temporary Target Files

- `intelliscan/detection.py`
- `intelliscan/main.py`

## Candidate Behavior

The rejected candidate preserved:

- the same in-memory JPEG-roundtrip preprocessing
- the same two-decimal bbox quantization
- the same single shared YOLO model
- the same downstream merge / segmentation / metrology logic

The only intended change was scheduling:

- prepare batches for both views concurrently
- consume those batches through the shared model

## Benchmark Setup

### Full-Pipeline A/B

Sample:

- `SN009`

Environment:

- `CUDA_VISIBLE_DEVICES=3`
- same codebase
- same model weights

Evidence:

- reference: `intelliscan/output_viewprefetch/SN009_prefetch4_ref/metrics.json`
- candidate: `intelliscan/output_viewprefetch/SN009_viewprefetch/metrics.json`
- output comparison: `intelliscan/output_viewprefetch/analysis/SN009_viewprefetch_compare.json`

### Detection-Only Repeated Benchmark

Samples:

- `SN009`
- `SN002`

Method:

- detection stage only
- same loaded YOLO model
- same loaded volume
- alternating `baseline / candidate / baseline / candidate / baseline / candidate`
- compare mean runtime per variant

Evidence:

- `intelliscan/output_viewprefetch/analysis/detection_view_prefetch_stage_benchmark.json`

## Measured Results

### Full-Pipeline SN009

| Variant | Total | Detection | 3D BBox | Seg+Met |
| --- | ---: | ---: | ---: | ---: |
| reference | `81.83s` | `9.89s` | `1.88s` | `69.25s` |
| candidate | `74.71s` | `21.19s` | `1.85s` | `50.84s` |

Interpretation:

- the target stage got much slower
- total pipeline time is not a valid judgment signal here because downstream segmentation time fluctuated strongly on this shared machine

### Detection-Only Repeated Benchmark

| Sample | Baseline Mean | Candidate Mean | Result |
| --- | ---: | ---: | --- |
| `SN009` | `8.69s` | `9.62s` | `10.62%` slower |
| `SN002` | `8.64s` | `8.95s` | `3.67%` slower |

## Quality Validation

The candidate was exact.

- `SN009` full-pipeline comparison:
  - voxel agreement `1.0`
  - class Dice `1.0` for classes `0..4`
  - no pad/solder defect flips
  - zero BLT delta
- detection-only benchmark hashes also matched exactly on:
  - `SN009`
  - `SN002`

This means the experiment was numerically safe but throughput-negative.

## Why It Likely Failed

The current kept detection path already overlaps work inside each view:

- per-slice CPU image preparation is parallelized with `4` workers
- next-batch preparation is overlapped with current YOLO inference

The rejected cross-view experiment added more CPU concurrency, but it did not add more GPU inference parallelism because both views still shared one YOLO model on one GPU.

So the likely effect under the current implementation is:

- more CPU contention
- more scheduling overhead
- no new detector-side parallel execution

## Conclusion

Cross-view detection overlap is not worth keeping under the current single-YOLO, single-GPU path.

- outputs remained exact
- the target detection stage became slower on both benchmarked samples
- the experiment code was removed

## Recommendation

Do not spend more local optimization time on same-GPU cross-view detection overlap in the current design.

If overlap work continues later, it should target:

- multi-file stage overlap
- heterogeneous-resource overlap
- or a design that truly separates GPU execution resources rather than only increasing CPU scheduling complexity
