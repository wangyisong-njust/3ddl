# 010 Feature Attention KD Ablation

## Objective

Test whether adding feature-level KD on top of the retained class-aware logit KD can close the remaining gap to the teacher baseline.

Decision rule for this pass:

- keep it only if it beats the current retained full-train KD baseline
- otherwise discard it as a preferred method

## Implementation

Code change:

- `../../pruning/finetune_pruned.py`

New controls:

- `--feature-distill-weight`
- `--feature-distill-layers`

The implemented feature KD is a channel-agnostic attention-transfer loss:

- capture intermediate features from matching teacher/student modules
- convert each feature tensor to a spatial attention map via channel-wise mean of squared activations
- normalize the flattened attention map
- compute masked L2 distance on valid voxels

This design was chosen because the pruned student and full teacher have different channel widths.

Layers used in this pass:

- `conv_0`
- `down_1.convs`
- `down_2.convs`
- `down_3.convs`
- `down_4.convs`

The retained logit KD baseline was kept unchanged:

- `--distill-weight 0.1`
- `--distill-temperature 2.0`
- `--kd-class-weights 1,1,1,2,2`

## Pilot Search

Small pilot setup:

- training subset: `0.05`
- test set: full `174` cases
- epochs: `5`
- batch size: `1`
- eval interval: `5`

Evidence:

- `./feature_kd_pilot_summary_20260408.json`
- `../../runs/paper_feature_kd_w25/finetune_report.json`
- `../../runs/paper_feature_kd_w50/finetune_report.json`
- `../../runs/paper_feature_kd_w100/finetune_report.json`
- `../../runs/paper_feature_kd_w200/finetune_report.json`
- retained small full-case baseline: `../../runs/paper_full_eval_c34_2_2/metrics/summary.json`

| Variant | Avg Dice | Class 3 | Class 4 |
| --- | ---: | ---: | ---: |
| retained logit KD baseline | `0.69335` | `0.14692` | `0.68842` |
| feature KD `w=25` | `0.67847` | `0.18835` | `0.59006` |
| feature KD `w=50` | `0.68236` | `0.16978` | `0.62626` |
| feature KD `w=100` | `0.68061` | `0.16450` | `0.62166` |
| feature KD `w=200` | `0.69366` | `0.21171` | `0.63009` |

Pilot interpretation:

- small weights were clearly worse
- `w=200` barely matched or slightly exceeded average Dice
- but even `w=200` still reduced class 4 strongly in the pilot

So `w=200` was the only candidate worth a full-train confirmation run.

## Full-Train Validation

Full-train setup:

- training subset: `1.0`
- test set: full `174` cases
- epochs: `10`
- batch size: `2`
- same logit KD baseline settings as the retained method
- extra feature KD: `--feature-distill-weight 200`

Evidence:

- `../../runs/paper_fulltrain_featurekd_w200_e10_b2/best_metrics.json`
- `../../runs/paper_fulltrain_featurekd_w200_e10_b2/finetune_report.json`
- `../../runs/paper_fulltrain_featurekd_w200_e10_b2_eval/metrics/summary.json`
- retained full-train KD baseline: `../../runs/paper_fulltrain_kd_c34_2_2_e10_b2_eval/metrics/summary.json`
- teacher full baseline: `../../runs/paper_recovery_eval_teacher_full/metrics/summary.json`
- `./feature_kd_fulltrain_compare_20260408.json`

| Variant | Avg Dice | Class 3 | Class 4 | Gap vs teacher |
| --- | ---: | ---: | ---: | ---: |
| teacher full | `0.90538` | `0.79370` | `0.89656` | `0.00000` |
| retained full-train logit KD | `0.88953` | `0.75450` | `0.87494` | `-0.01586` |
| feature KD `w=200` | `0.88351` | `0.74183` | `0.87068` | `-0.02187` |

Delta versus retained KD:

- average Dice: `-0.00601`
- class 3 Dice: `-0.01268`
- class 4 Dice: `-0.00426`

## Interpretation

- This minimal feature attention KD is not strong enough in the current form.
- It did not improve the retained full-train baseline.
- It did not reduce the teacher gap.
- It also moved the critical classes in the wrong direction.

One useful observation remains:

- feature loss magnitude stayed very small (`~2e-05` at the best full-train epoch), so naive feature attention alignment may simply be too weak or too indirect for this task

## Decision

Keep:

- the implementation hooks and flags as experiment infrastructure

Do not keep as a preferred method:

- feature attention KD in this current form

Current retained method remains:

- immediate class-aware logit KD
- `--distill-weight 0.1`
- `--distill-temperature 2.0`
- `--kd-class-weights 1,1,1,2,2`

## Next Step

Do not continue tuning this exact feature attention KD recipe.

The next candidate should be more targeted than generic activation matching, for example:

- decoder-side or boundary-focused feature KD
- class-conditioned feature KD
- downstream-task-aware objectives instead of generic feature similarity
