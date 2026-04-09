# 020 TRT FP16 Integration

## Objective

Use a TensorRT FP16 segmentation backend inside the `intelliscan` pipeline.

## Target Files

- `intelliscan/main.py`
- `intelliscan/segmentation.py`
- `wp5-seg/pruning/export_onnx.py`
- `wp5-seg/pruning/build_trt_engine.py`

## Code Change Summary

- `intelliscan` exposes `use_trt` and `trt_engine_path` in `PipelineConfig`.
- `SegmentationInference` can wrap a TensorRT predictor instead of a PyTorch `BasicUNet`.

## Why This Change Was Made

- Formal pipeline timing shows segmentation is one of the main runtime components.

## Baseline Behavior

- SN009 baseline run uses the PyTorch checkpoint path and records:
  - total `93.23s`
  - segmentation `33.79s`

## Optimized Behavior

- SN009 TRT FP16 run records:
  - total `81.15s`
  - segmentation `24.15s`

## How It Was Tested

- Validated from current formal output files under `intelliscan/output_formal/`

## Sample(s) Used

- `SN009`

## Key Timing Numbers

| Variant | Total | Detection | Segmentation | Metrology |
| --- | --- | --- | --- | --- |
| Baseline | `93.23s` | `14.85s` | `33.79s` | `16.95s` |
| TRT FP16 | `81.15s` | `13.91s` | `24.15s` | `16.79s` |

## Accuracy / Quality Impact

- Speed improves, but current SN009 evidence shows non-trivial segmentation and metrology drift.

## Risks / Caveats

- The engine used for the validated run is not stored under this workspace root.
- This optimization is not safe to describe as lossless until more validation is done.

## Conclusion

- TRT FP16 is a real and validated acceleration path for current `intelliscan`, but it should currently be treated as `fast but not yet accuracy-safe`.

## Evidence Paths

- `intelliscan/output_formal/SN009_current_gpu/metrics.json`
- `intelliscan/output_formal/SN009_current_gpu_trt/metrics.json`
- `intelliscan/output_formal/SN009_current_gpu_trt/timing.log`
