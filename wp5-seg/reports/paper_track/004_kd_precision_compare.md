# 004 KD Precision Compare (FP32 vs BF16)

## Objective

Check whether BF16 itself is helping or hurting the current best non-zero KD candidate.

The candidate chosen from the BF16 sweep is `distill_weight=0.1`.

## Setup

Shared settings:

- student: `pruning/output/pruned_50pct.ckpt`
- teacher: `../intelliscan/models/segmentation_model.ckpt`
- data: `3ddl-dataset/data`
- subset: `subset_ratio=0.05`
- eval cap: `max_eval_cases=10`
- epochs: `3`
- batch size: `1`
- KD weight: `0.1`
- temperature: `2.0`

Evidence:

- `./kd_fp32_vs_bf16_summary_20260408.json`
- `../runs/paper_kd_sweep_bf16_w0p1/finetune_report.json`
- `../runs/paper_kd_sweep_fp32_w0p1/finetune_report.json`

## Results

| Precision | Best Dice | Recovery | Train Time |
| --- | ---: | ---: | ---: |
| BF16 | `0.5577` | `+0.2413` | `11.26s` |
| FP32 | `0.5571` | `+0.2407` | `6.94s` |

## Interpretation

- Accuracy is nearly identical in this setup.
- BF16 is slightly higher in Dice, but the gap is negligible at this scale.
- BF16 is noticeably slower here.

This means:

- mixed precision training should not be assumed to improve training throughput for this exact small-batch KD setup
- BF16 is still useful as an option, but it is not currently a proven efficiency win for the paper track

## Practical Conclusion

For the current KD recipe and experiment scale:

- precision is not the main blocker
- the KD formulation itself is the bigger issue

So the next paper-track iteration should focus on improving the distillation objective or schedule, not on spending more time tuning BF16.
