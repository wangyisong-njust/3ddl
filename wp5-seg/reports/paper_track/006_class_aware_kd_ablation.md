# 006 Class-Aware KD Ablation

## Objective

Test whether KD should be weighted more heavily on defect-relevant classes instead of treating all classes equally.

Constraint for this pass:

- keep the method only if it improves the practical precision tradeoff over the current retained KD baseline
- otherwise discard it

## Setup

- script: `pruning/finetune_pruned.py`
- GPU: local `GPU 2`
- student: `pruning/output/pruned_50pct.ckpt`
- teacher: `../intelliscan/models/segmentation_model.ckpt`
- subset: `subset_ratio=0.05`
- eval cap: `max_eval_cases=10`
- epochs: `5`
- batch size: `1`
- precision: FP32
- KD weight: `0.1`
- KD temperature: `2.0`
- KD start epoch: `1`

Evidence:

- `./classaware_kd_summary_20260408.json`
- `../runs/paper_classaware_kd_uniform/finetune_report.json`
- `../runs/paper_classaware_kd_c34_2_2/finetune_report.json`
- `../runs/paper_classaware_kd_c34_2_4/finetune_report.json`
- `../runs/paper_classaware_kd_c34_4_2/finetune_report.json`

## Variants

- uniform KD: `1,1,1,1,1`
- mild class-aware: `1,1,1,2,2`
- pad-heavier: `1,1,1,2,4`
- void-heavier: `1,1,1,4,2`

Weights refer to GT-based KD voxel weights for classes `0..4`.

## Results

| Variant | Avg Dice | Class 3 | Class 4 |
| --- | ---: | ---: | ---: |
| uniform | `0.6892` | `0.1174` | `0.6473` |
| `1,1,1,2,2` | `0.7242` | `0.2312` | `0.6885` |
| `1,1,1,2,4` | `0.6949` | `0.1767` | `0.6721` |
| `1,1,1,4,2` | `0.7316` | `0.2709` | `0.6728` |

## Interpretation

- Class-aware KD is a clear positive direction in this experiment.
- All three class-aware variants beat the current uniform KD baseline in average Dice.
- The best variant is `1,1,1,4,2`.
- It improves both critical classes relative to uniform KD:
  - class 3: `0.1174 -> 0.2709`
  - class 4: `0.6473 -> 0.6728`
- It also improves total average Dice:
  - `0.6892 -> 0.7316`

## Decision

### Keep

- class-aware KD as the new retained KD direction
- current best setting: `kd_class_weights=1,1,1,4,2`

### Drop

- uniform KD as the preferred KD baseline
- delayed KD as the preferred scheduling policy

Uniform KD and delayed KD remain useful as ablation baselines only.

## Practical Conclusion

This is the first KD variant in the paper track that is clearly worth keeping.

It satisfies the acceptance rule for this pass:

- average Dice improves
- class 3 improves
- class 4 improves

## Next Step

Use `kd_class_weights=1,1,1,4,2` as the retained KD baseline and move on to the next stronger research direction:

1. feature-level KD
2. real INT8 calibration / PTQ
3. QAT if PTQ still drifts
