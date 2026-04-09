# 003 KD Weight Sweep (BF16)

## Objective

Run a small but more meaningful KD sweep than the tiny one-epoch smoke.

This pass keeps the teacher and training recipe fixed, and varies only `distill_weight` under BF16 training.

## Setup

- branch: `research/wp5-compression-paper`
- script: `pruning/finetune_pruned.py`
- GPU: local `GPU 2`
- student: `pruning/output/pruned_50pct.ckpt`
- teacher: `../intelliscan/models/segmentation_model.ckpt`
- data: `3ddl-dataset/data`
- subset: `subset_ratio=0.05` (`30` train samples)
- eval cap: `max_eval_cases=10`
- epochs: `3`
- batch size: `1`
- precision: `--amp --amp-dtype bf16`
- temperature: `2.0`

## Sweep

- `w=0.0`
- `w=0.05`
- `w=0.1`
- `w=0.2`
- `w=0.3`

Evidence:

- `./kd_weight_sweep_summary_20260408.json`
- `../runs/paper_kd_sweep_bf16_w0p0/finetune_report.json`
- `../runs/paper_kd_sweep_bf16_w0p05/finetune_report.json`
- `../runs/paper_kd_sweep_bf16_w0p1/finetune_report.json`
- `../runs/paper_kd_sweep_bf16_w0p2/finetune_report.json`
- `../runs/paper_kd_sweep_bf16_w0p3/finetune_report.json`

## Results

| KD Weight | Best Dice | Recovery | Train Time |
| --- | ---: | ---: | ---: |
| `0.0` | `0.5655` | `+0.2491` | `8.82s` |
| `0.05` | `0.5550` | `+0.2386` | `11.35s` |
| `0.1` | `0.5577` | `+0.2413` | `11.26s` |
| `0.2` | `0.5482` | `+0.2318` | `11.24s` |
| `0.3` | `0.5416` | `+0.2252` | `11.30s` |

## Interpretation

- In this small-to-medium experiment, plain finetuning (`w=0.0`) is still the best setting.
- The best non-zero KD setting is `w=0.1`, but it still underperforms plain finetuning by about `0.0078` Dice.
- Larger KD weights degrade more clearly.
- KD also increases train time in this setup because the teacher forward is always present.

## Conclusion

Current masked-logit KD is functional but not yet a win.

This is a useful negative result for the paper track:

- pruning + naive KD should not be assumed to help automatically
- KD weight is sensitive
- future work should change the KD recipe, not just push the weight higher

## Recommended Next Step

Do not expand this exact KD recipe to large runs yet.

Instead, test one of these more targeted variants:

1. lower or scheduled KD weight
2. delayed KD after a plain supervised warmup
3. class-aware KD emphasizing rare/defect classes
4. feature-level or boundary-aware KD instead of logits-only KD
