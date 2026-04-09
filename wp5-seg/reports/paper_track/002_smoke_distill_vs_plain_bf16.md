# 002 Smoke Distill vs Plain BF16

## Objective

Verify that the new paper-track finetuning controls actually run, and compare a minimal:

- pruned + plain finetune
- pruned + teacher distillation

under the same tiny smoke setup.

## Setup

- script: `pruning/finetune_pruned.py`
- GPU: local `GPU 3` (selected from currently lower-utilization GPUs)
- student: `pruning/output/pruned_50pct.ckpt`
- teacher: `../intelliscan/models/segmentation_model.ckpt`
- dataset: `3ddl-dataset/data`
- train subset: `subset_ratio=0.01` (`6` train samples)
- eval cap: `max_eval_cases=2`
- epochs: `1`
- batch size: `1`
- precision: `--amp --amp-dtype bf16`

## Commands

### Plain

```bash
python pruning/finetune_pruned.py \
  --pruned_model_path pruning/output/pruned_50pct.ckpt \
  --data_dir 3ddl-dataset/data \
  --output_dir runs/paper_smoke_plain_bf16 \
  --epochs 1 --batch_size 1 --subset_ratio 0.01 \
  --num_workers 0 --eval_interval 1 --max_eval_cases 2 \
  --amp --amp-dtype bf16 --distill-weight 0.0 --no_timestamp
```

### Distill

```bash
python pruning/finetune_pruned.py \
  --pruned_model_path pruning/output/pruned_50pct.ckpt \
  --teacher_model_path ../intelliscan/models/segmentation_model.ckpt \
  --teacher_model_format state_dict \
  --data_dir 3ddl-dataset/data \
  --output_dir runs/paper_smoke_distill_bf16 \
  --epochs 1 --batch_size 1 --subset_ratio 0.01 \
  --num_workers 0 --eval_interval 1 --max_eval_cases 2 \
  --amp --amp-dtype bf16 --distill-weight 0.3 \
  --distill-temperature 2.0 --no_timestamp
```

## Results

| Variant | Pre Dice | Best Dice | Recovery | Train Time |
| --- | ---: | ---: | ---: | ---: |
| plain bf16 | `0.2133` | `0.4589` | `+0.2457` | `1.51s` |
| distill bf16 | `0.2133` | `0.4114` | `+0.1981` | `3.38s` |

Evidence:

- `../runs/paper_smoke_plain_bf16/finetune_report.json`
- `../runs/paper_smoke_distill_bf16/finetune_report.json`

## Interpretation

- The new KD path is functional: checkpoint loading, teacher forward, masked KL loss, AMP/BF16, and capped evaluation all ran successfully.
- On this tiny smoke, KD did **not** beat plain finetuning.
- This result is not strong enough to reject KD, because the setup is intentionally too small:
  - only `6` train samples
  - only `2` eval cases
  - only `1` epoch
  - only one KD hyperparameter setting (`weight=0.3`, `temperature=2.0`)

## Practical Conclusion

This pass validates the research infrastructure, not the final method claim.

The next justified step is a controlled KD sweep, not a bigger engineering rewrite:

1. sweep `distill_weight` over a small range
2. compare `bf16` vs FP32 stability
3. increase `subset_ratio` and `epochs` to a still-manageable medium experiment
4. inspect per-class Dice, especially classes `3` and `4`
