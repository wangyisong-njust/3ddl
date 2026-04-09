# 005 Delayed KD Ablation

## Objective

Test whether a supervised warmup before KD is better than immediate KD.

Decision rule for this pass:

- if delayed KD beats the current retained baseline, keep it
- if it does not, record it as a negative result and do not keep it as the preferred method

## Setup

- script: `pruning/finetune_pruned.py`
- GPU: local `GPU 2`
- student: `pruning/output/pruned_50pct.ckpt`
- teacher: `../intelliscan/models/segmentation_model.ckpt`
- subset: `subset_ratio=0.05` (`30` train samples)
- eval cap: `max_eval_cases=10`
- epochs: `5`
- batch size: `1`
- precision: FP32
- KD weight: `0.1`
- KD temperature: `2.0`

Variants:

- plain finetune
- immediate KD (`distill_start_epoch=1`)
- delayed KD start at epoch `2`
- delayed KD start at epoch `3`

Evidence:

- `./delayed_kd_summary_20260408.json`
- `../runs/paper_delayed_kd_plain/finetune_report.json`
- `../runs/paper_delayed_kd_immediate_kd_w0p1/finetune_report.json`
- `../runs/paper_delayed_kd_delayed_kd_w0p1_s2/finetune_report.json`
- `../runs/paper_delayed_kd_delayed_kd_w0p1_s3/finetune_report.json`

## Results

| Variant | Best Dice | Recovery | Class 3 | Class 4 |
| --- | ---: | ---: | ---: | ---: |
| plain | `0.6037` | `+0.2873` | `0.2007` | `0.2186` |
| immediate KD | `0.6892` | `+0.3729` | `0.1174` | `0.6473` |
| delayed KD s2 | `0.6727` | `+0.3563` | `0.1832` | `0.4824` |
| delayed KD s3 | `0.6853` | `+0.3690` | `0.2077` | `0.5256` |

## Interpretation

- KD is now a positive direction at this experiment scale: all KD variants beat plain finetune in average Dice.
- Delaying KD does **not** beat immediate KD in this ablation.
- `distill_start_epoch=3` comes close, but still stays below immediate KD.
- Immediate KD gives the strongest overall gain, driven mainly by a much stronger class-4 Dice.
- Delayed KD preserves class-3 Dice better than immediate KD, but the total average still loses.

## Decision

### Keep

- `KD` as an active paper-track direction
- immediate KD with `distill_weight=0.1` as the current retained KD baseline

### Do Not Keep As Preferred Method

- delayed KD warmup policy (`distill_start_epoch > 1`)

The delayed-KD control can remain in code as an ablation tool, but it is not the selected method after this pass.

## Next Step

The next paper-track experiment should keep immediate KD and improve the objective itself, for example:

1. class-aware KD weighting for rare classes
2. feature-level KD
3. boundary-aware KD

Do not spend more time tuning delayed-KD schedules unless later evidence shows a clear need.
