# 020 Runtime Backend Calibration, Compile, and Energy Study

## Objective

Evaluate whether three runtime-side directions are worth keeping for the paper track:

- real-data INT8 calibration
- `torch.compile()` inference
- power / energy-aware backend comparison

This note follows the pruning-ratio and deployment-ranking work. The goal is not just lower patch latency. The goal is to identify which runtime backend is worth carrying into later `intelliscan` deployment validation.

## Why This Experiment Was Needed

The current compression line already has:

- retained students at `r=0.375` and `r=0.5`
- ONNX and TensorRT build scripts
- patch-level latency artifacts

But three gaps remained:

1. INT8 calibration was still using random tensors by default.
2. `torch.compile()` had only been probed lightly.
3. Runtime speed had not been tied to power / energy or full-case backend quality in one place.

## Code and Tooling Added

- Real-data calibration loader in `../../pruning/build_trt_engine.py`
- Shared runtime helpers in `../../pruning/runtime_support.py`
- Power / energy-aware PyTorch benchmark in `../../pruning/benchmark.py`
- Power / energy-aware TensorRT benchmark in `../../pruning/benchmark_trt.py`
- Full-case backend evaluator in `../../pruning/eval_runtime.py`

## Experimental Setup

- GPU: physical `GPU 1` via `CUDA_VISIBLE_DEVICES=1`
- PyTorch compile modes tested:
  - `reduce-overhead`
  - `max-autotune-no-cudagraphs`
- Models:
  - teacher: `../../../intelliscan/models/segmentation_model.ckpt`
  - retained `r=0.375`: `../../runs/paper_fulltrain_kd_r0p375_e10_b2/best.ckpt`
  - retained `r=0.5`: `../../runs/paper_fulltrain_kd_c34_2_2_e10_b2/best.ckpt`
- Unified summary JSON:
  - `./runtime_backend_calibration_summary_20260409.json`

## Compile Mode Selection

Evidence:

- `../../runs/paper_runtime_mode_sweep/r0p5_reduce_overhead.json`
- `../../runs/paper_runtime_mode_sweep/r0p5_max_autotune_no_cudagraphs.json`

| Mode | Mean latency | First-run compile cost | Decision |
| --- | ---: | ---: | --- |
| `reduce-overhead` | `4.76 ms` | `4339.66 ms` | Keep |
| `max-autotune-no-cudagraphs` | `5.96 ms` | `41273.07 ms` | Reject |

`reduce-overhead` is clearly the better compile mode here. It is faster in steady state and far cheaper to warm up.

Smoke-only AMP compile evidence:

- `../../runs/paper_runtime_smoke/benchmark_smoke.json`

That smoke showed `compile + AMP` slower than `compile + FP32` on both teacher and student, so it is not kept as an inference backend.

## Formal Compile Results

Evidence:

- `../../runs/paper_runtime_formal/teacher_vs_r0p375_compile.json`
- `../../runs/paper_runtime_formal/teacher_vs_r0p5_compile.json`

### Teacher vs `r=0.375`

| Variant | Mean latency | Avg power |
| --- | ---: | ---: |
| teacher eager FP32 | `17.40 ms` | `128.65 W` |
| teacher compile FP32 | `13.37 ms` | `119.90 W` |
| `r=0.375` eager FP32 | `14.82 ms` | `131.97 W` |
| `r=0.375` compile FP32 | `13.79 ms` | `147.95 W` |

Interpretation:

- compile helps the teacher meaningfully
- compile only gives a small benefit on `r=0.375`
- compiled `r=0.375` is still slightly slower than compiled teacher

### Teacher vs `r=0.5`

| Variant | Mean latency | Avg power |
| --- | ---: | ---: |
| teacher eager FP32 | `17.40 ms` | `144.46 W` |
| teacher compile FP32 | `13.36 ms` | `115.11 W` |
| `r=0.5` eager FP32 | `6.94 ms` | `93.23 W` |
| `r=0.5` compile FP32 | `4.80 ms` | `85.57 W` |

Interpretation:

- `r=0.5` plus `torch.compile(reduce-overhead)` is a strong PyTorch fallback backend
- relative to compiled teacher, compiled `r=0.5` is much faster and lower power

## Calibration Strategies

Evidence:

- `../../runs/paper_runtime_formal/r0p5_int8_real_calibration_summary.json`
- `../../runs/paper_runtime_formal/r0p375_int8_real_calibration_summary.json`
- `../../runs/paper_runtime_formal/r0p375_int8_fitonly64_calibration_summary.json`

### Generic Real Calibration

- selected `64` training samples
- `26` of those still required center-crop adaptation
- `cropped_voxels_total = 982`

### Fit-Only Calibration

- selected `64` training samples that already fit the ROI
- `0` cropped samples
- `cropped_voxels_total = 0`

Interpretation:

- “real calibration” is not one thing
- the calibration sample selection policy matters
- for this pipeline, fit-only calibration is a cleaner INT8 policy than mixing in adapted samples that require ROI cropping

## Backend Latency, Power, and Energy

Evidence:

- `../../runs/paper_runtime_formal/r0p375_backend_benchmark.json`
- `../../runs/paper_runtime_formal/r0p375_int8_fitonly64_benchmark.json`
- `../../runs/paper_runtime_formal/r0p5_backend_benchmark.json`

### `r=0.375`

| Backend | Mean latency | Avg power | J / iter |
| --- | ---: | ---: | ---: |
| PyTorch FP32 | `16.47 ms` | `198.55 W` | `3.2719` |
| PyTorch compile FP32 | `14.31 ms` | `176.95 W` | `2.5341` |
| TRT FP16 | `3.15 ms` | `96.61 W` | `0.3282` |
| TRT INT8 real | `2.79 ms` | `108.31 W` | `0.3270` |
| TRT INT8 fit-only64 | `2.79 ms` | `106.93 W` | `0.3235` |

### `r=0.5`

| Backend | Mean latency | Avg power | J / iter |
| --- | ---: | ---: | ---: |
| PyTorch FP32 | `8.87 ms` | `121.16 W` | `1.0757` |
| PyTorch compile FP32 | `4.86 ms` | `110.52 W` | `0.5386` |
| TRT FP16 | `1.62 ms` | `85.75 W` | `0.1592` |
| TRT INT8 random | `1.63 ms` | `101.80 W` | `0.1900` |
| TRT INT8 real | `1.63 ms` | `89.64 W` | `0.1670` |

Interpretation:

- TRT FP16 is the cleanest retained deployment backend on both retained students
- `torch.compile()` is useful as a PyTorch fallback, especially for `r=0.5`
- INT8 does not automatically beat FP16 on latency or energy

## Full-Case Backend Quality

Evidence:

- `../../runs/paper_runtime_formal/eval_r0p5_pytorch/metrics/summary.json`
- `../../runs/paper_runtime_formal/eval_r0p5_trt_fp16_hostrelay/metrics/summary.json`
- `../../runs/paper_runtime_formal/eval_r0p5_trt_int8_random_hostrelay/metrics/summary.json`
- `../../runs/paper_runtime_formal/eval_r0p5_trt_int8_real_hostrelay/metrics/summary.json`
- `../../runs/paper_runtime_formal/eval_r0p375_pytorch/metrics/summary.json`
- `../../runs/paper_runtime_formal/eval_r0p375_trt_fp16_hostrelay/metrics/summary.json`
- `../../runs/paper_runtime_formal/eval_r0p375_trt_int8_real_hostrelay/metrics/summary.json`
- `../../runs/paper_runtime_formal/eval_r0p375_trt_int8_fitonly64_hostrelay/metrics/summary.json`

### `r=0.5`

| Backend | Avg Dice | Dice c3 | Dice c4 |
| --- | ---: | ---: | ---: |
| PyTorch | `0.889527` | `0.754505` | `0.874937` |
| TRT FP16 | `0.889533` | `0.754498` | `0.874958` |
| TRT INT8 random | `0.889529` | `0.754481` | `0.874955` |
| TRT INT8 real | `0.889529` | `0.754481` | `0.874955` |

Conclusion:

- `r=0.5` FP16 is effectively parity-level with PyTorch
- `r=0.5` INT8 also stays at parity-level quality
- but INT8 does not buy a latency win over FP16, so FP16 remains the preferred backend

### `r=0.375`

| Backend | Avg Dice | Dice c3 | Dice c4 |
| --- | ---: | ---: | ---: |
| PyTorch | `0.895134` | `0.786901` | `0.871692` |
| TRT FP16 | `0.895146` | `0.786925` | `0.871728` |
| TRT INT8 real | `0.888706` | `0.759882` | `0.869126` |
| TRT INT8 fit-only64 | `0.892145` | `0.773414` | `0.870663` |

Conclusion:

- `r=0.375` FP16 is effectively lossless relative to PyTorch
- generic real INT8 hurts quality materially
- fit-only calibration recovers part of that gap, especially on class `3`
- even after that recovery, INT8 still does not beat FP16 as the retained backend

## Important Runtime Caveat

The first TensorRT full-case evaluator used a GPU-direct predictor path and produced an invalid result:

- `../../runs/paper_runtime_formal/eval_r0p5_trt_fp16/metrics/summary.json`

That artifact collapsed to average Dice `0.291838` and is not used as a backend-quality result.

Kept conclusion:

- use the host-relay TRT predictor path for quality evaluation under MONAI sliding-window
- treat the GPU-direct evaluator as a validated runtime-compatibility negative result, not as evidence about FP16 accuracy

## Retained Decisions

Keep:

- `torch.compile(reduce-overhead)` as the PyTorch fallback backend, especially for `r=0.5`
- TRT FP16 as the preferred deployment backend for both retained students
- fit-only calibration as a better INT8 calibration policy than generic real calibration for `r=0.375`

Do not keep:

- `max-autotune-no-cudagraphs`
- `compile + AMP` inference
- GPU-direct TRT evaluation for quality claims
- generic real INT8 as the preferred `r=0.375` backend
- INT8 as the preferred `r=0.5` backend

## Consequence for the Paper Track

This pass changes the runtime story in a useful way:

- calibration is no longer just a future idea; it now has one real positive result and one real negative result
- compile is no longer just a probe; it is a retained fallback backend for one of the kept students
- power / energy can now be reported alongside latency and Dice, rather than as a separate future promise

## Next Step

Do not treat backend selection as finished at the patch level.

The next meaningful step is to take the retained runtime candidates:

- `r=0.375` + TRT FP16
- `r=0.5` + TRT FP16
- `r=0.5` + PyTorch compile

and rank them again under the `intelliscan` whole-pipeline gate, not just under `wp5-seg` full-case Dice.
