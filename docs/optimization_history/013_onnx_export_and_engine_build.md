# 013 ONNX Export And Engine Build

## Objective

Provide a deployment path from PyTorch checkpoints to TensorRT engines.

## Target Files

- `wp5-seg/pruning/export_onnx.py`
- `wp5-seg/pruning/build_trt_engine.py`
- `wp5-seg/pruning/benchmark_trt.py`

## Code Change Summary

- ONNX export uses a static input shape of `(1, 1, 112, 112, 80)` and opset `18`.
- TensorRT build script supports `fp32`, `fp16`, and `int8`.
- Benchmark script can compare PyTorch and TensorRT engines at patch level.

## Why This Change Was Made

- TensorRT is the intended deployment backend for low-latency segmentation inference.

## Baseline Behavior

- Without export/build, deployment stays on PyTorch inference.

## Optimized Behavior

- Export and engine build path exists in code.
- `intelliscan` can consume a TensorRT engine through its `--trt` path.

## How It Was Tested

- Script inspection only in this documentation pass
- Engine consumption was validated indirectly through the SN009 TRT run output

## Sample(s) Used

- Static patch shape `(1, 1, 112, 112, 80)`

## Key Timing Numbers

- No local ONNX or engine build artifact was regenerated in this pass.

## Accuracy / Quality Impact

- Current INT8 build path is not deployment-safe by default because calibration uses random tensors.

## Risks / Caveats

- No ONNX or `.engine` artifact is stored in this workspace.
- Current INT8 calibrator in code uses random data, which is not acceptable for defect-sensitive trust claims.

## Conclusion

- The deployment toolchain is present, but INT8 trustworthiness depends on replacing the current calibration path with real crop data.

## Evidence Paths

- `wp5-seg/pruning/export_onnx.py`
- `wp5-seg/pruning/build_trt_engine.py`
- `wp5-seg/pruning/benchmark_trt.py`
- `intelliscan/output_formal/SN009_current_gpu_trt/metrics.json`
