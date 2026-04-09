# Project Overview

## Goal

This workspace tracks acceleration of a 3D semiconductor defect detection pipeline. The work has two layers:

1. `wp5-seg/`: make the segmentation model smaller and faster to deploy.
2. `intelliscan/`: reduce end-to-end pipeline latency on real NIfTI samples.

## Repository Roles

### `intelliscan/`

- Orchestrates `NII -> 2D detection -> 3D bbox -> 3D segmentation -> metrology -> PDF report`
- Primary entry point: `intelliscan/main.py`
- Key modules:
  - `intelliscan/detection.py`
  - `intelliscan/merge.py`
  - `intelliscan/segmentation.py`
  - `intelliscan/metrology.py`
  - `intelliscan/report.py`

### `wp5-seg/`

- Trains and evaluates the 3D segmentation model used by `intelliscan`
- Key entry points:
  - `wp5-seg/train.py`
  - `wp5-seg/eval.py`
  - `wp5-seg/pruning/run_pruning_pipeline.sh`
- Compression / deployment scripts live under `wp5-seg/pruning/`

## Optimization Layers

### Model Compression Layer

- Structured pruning
- Finetuning after pruning
- ONNX export
- TensorRT engine build
- Patch-level benchmarking

### Pipeline Engineering Layer

- Reduce intermediate file I/O
- Use GPU-side normalization
- Batch per-bbox inference when crops fit ROI
- Bypass sliding-window inference when possible
- Move toward combined segmentation + metrology
- Investigate in-memory detection / merge path
- Integrate TensorRT FP16 into the pipeline
- Profile formal runs on real samples such as SN009

## Current Grounded Evidence

- Historical summary report:
  - `wp5-seg/reports/segmentation_acceleration_report.md`
- Current formal pipeline outputs:
  - `intelliscan/output_formal/SN009_current_gpu/...`
  - `intelliscan/output_formal/SN009_current_gpu_inmemory/...`
  - `intelliscan/output_formal/SN009_current_gpu_trt/...`
- Current smoke evaluations:
  - `intelliscan/output_smoke/images/...`
  - `wp5-seg/runs/smoke_eval/metrics/summary.json`
  - `wp5-seg/runs/smoke_eval_gpu/metrics/summary.json`
- Current pruning artifacts:
  - `wp5-seg/pruning/output/pruned_50pct.ckpt`
  - `wp5-seg/pruning/output/pruned_50pct.json`
  - `wp5-seg/pruning/output/benchmark_test.json`

## Documentation Convention

- `Validated`: directly backed by a file in this workspace
- `Computed in this pass`: recomputed during this documentation setup from files in this workspace
- `Reported but not revalidated in this pass`: present in an existing report or artifact, but raw supporting outputs are not available here
- `User-reported result`: came from the user and could not be checked from local files
