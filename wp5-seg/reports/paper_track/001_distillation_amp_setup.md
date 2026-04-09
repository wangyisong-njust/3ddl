# 001 Distillation + AMP Setup

## Objective

Create the first paper-track training scaffold for `wp5-seg` by adding:

- CUDA AMP/BF16 training controls
- teacher-student distillation support for pruned-model finetuning
- a lightweight evaluation cap for quick research smoke runs

## Why This Step

Current local code already supports structured pruning and TensorRT deployment, but it does not yet support:

- mixed-precision training in `train.py`
- distillation in `pruning/finetune_pruned.py`
- fast-turnaround smoke experiments on small validation subsets

These are prerequisite controls for the next research comparisons.

## Code Changes

- `train.py`
  - add `--amp`
  - add `--amp-dtype {fp16,bf16}`
  - run training forward/loss under autocast when enabled
- `pruning/finetune_pruned.py`
  - add `--amp`
  - add `--amp-dtype {fp16,bf16}`
  - add `--teacher_model_path`
  - add `--teacher_model_format`
  - add `--distill-weight`
  - add `--distill-temperature`
  - add `--subset_ratio`
  - add `--max_eval_cases`
  - add masked KL distillation loss on valid voxels only
  - save `eval_history.json` and `best_metrics.json` for future per-class traceability

## Research Hypothesis

The most promising near-term paper direction is not more uniform pruning by itself, but:

1. prune aggressively enough to create deployment gain
2. recover performance with distillation rather than plain finetuning only
3. evaluate with defect-sensitive metrics, not average Dice alone

## Local Evidence Collected In This Pass

- Existing pruning metadata: `../pruning/output/pruned_50pct.json`
- Existing PyTorch benchmark: `../pruning/output/benchmark_test.json`
- New compile probe on current machine: `../compile_probe_20260407.json`

### Compile Probe Result

Patch-level forward benchmark on current local GPU with `112x112x80` input:

| Model | Eager FP32 | `torch.compile` FP32 |
| --- | ---: | ---: |
| original | `39.31 ms` | `31.26 ms` |
| pruned | `15.46 ms` | `12.98 ms` |

This suggests PyTorch runtime still has some headroom, but compile is not the main paper contribution compared with pruning + distillation + deployable quantization.

## Not Yet Revalidated

- No new full training or finetuning run was completed in this pass.
- No new Dice / IoU / per-class recovery result is claimed yet for distillation or AMP.

## Next Experiments

1. smoke finetune: pruned only vs pruned + distillation
2. same smoke with `--amp --amp-dtype bf16`
3. if smoke is stable, run full comparison matrix:
   - full baseline
   - pruned only
   - pruned + finetune
   - pruned + distillation
   - pruned + distillation + real INT8 PTQ
