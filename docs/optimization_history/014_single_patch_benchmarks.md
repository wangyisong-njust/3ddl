# 014 Single-Patch Benchmarks

## Objective

Track model-level latency benchmarks independently from end-to-end pipeline timing.

## Target Files

- `wp5-seg/pruning/benchmark.py`
- `wp5-seg/pruning/benchmark_trt.py`
- `wp5-seg/pruning/output/benchmark_test.json`
- `wp5-seg/reports/segmentation_acceleration_report.md`

## Code Change Summary

- `benchmark.py` measures PyTorch FP32 and AMP.
- `benchmark_trt.py` measures TensorRT engines against PyTorch baselines.

## Why This Change Was Made

- Model-level latency helps separate network improvements from pipeline I/O overhead.

## Baseline Behavior

From `benchmark_test.json`:

- Original PyTorch FP32 mean: `17.39ms`

## Optimized Behavior

From `benchmark_test.json`:

- Pruned PyTorch FP32 mean: `6.31ms`
- Recorded speedup: `2.76x`

## How It Was Tested

- Existing benchmark artifact inspection
- Existing repo report cross-reference

## Sample(s) Used

- Patch-level synthetic or dummy benchmark input of shape `(1, 1, 112, 112, 80)`

## Key Timing Numbers

Validated artifact:

| Method | Mean Latency |
| --- | --- |
| Original PyTorch FP32 | `17.39ms` |
| Original PyTorch AMP | `33.71ms` |
| Pruned PyTorch FP32 | `6.31ms` |
| Pruned PyTorch AMP | `16.94ms` |

Reported but not revalidated in this pass:

- Original TRT FP32: `14.54ms`
- Original TRT FP16: `5.20ms`
- Original TRT INT8: `3.39ms`
- Pruned TRT FP16: `1.72ms`
- Pruned TRT INT8: `1.72ms`

## Accuracy / Quality Impact

- Patch benchmark does not measure segmentation quality drift.

## Risks / Caveats

- `benchmark_test.json` references a baseline checkpoint path that is not present in the workspace anymore.
- Current artifact does not support any claim that AMP is faster in this environment.
- TRT benchmark table currently relies on the repo report rather than local engine artifacts.

## Conclusion

- Pruning-based PyTorch speedup is supported by a local artifact.
- TRT patch numbers should remain tagged as reported until local engine artifacts are regenerated and benchmarked.

## Evidence Paths

- `wp5-seg/pruning/output/benchmark_test.json`
- `wp5-seg/reports/segmentation_acceleration_report.md`
