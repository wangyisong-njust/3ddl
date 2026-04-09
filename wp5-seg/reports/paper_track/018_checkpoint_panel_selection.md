# 018 Checkpoint Panel Selection

## Objective

Expand checkpoint selection from the `best.ckpt` vs `last.ckpt` probe into a small deployment panel, then check whether any epoch checkpoint is clearly better under the whole-pipeline gate.

This note follows `017_metrology_aware_checkpoint_selection_probe.md`.

## Code Support

The panel run used the retained helper in `../../pruning/finetune_pruned.py`:

- `--save-eval-checkpoints`

This saves `epoch_XXX.ckpt` together with the evaluation payload for downstream selection studies.

## Evidence

- Summary JSON: `./checkpoint_panel_selection_summary_20260408.json`
- Panel run: `../../runs/paper_fulltrain_kd_c34_2_2_e10_b2_ckptpanel`
- Eval history: `../../runs/paper_fulltrain_kd_c34_2_2_e10_b2_ckptpanel/eval_history.json`
- `SN009` teacher: `../../../intelliscan/output_student_deploy/SN009_SN009_teacher_ref`
- `SN009` panel checkpoints:
  - `../../../intelliscan/output_student_deploy/SN009_SN009_student_r0p5_epoch001_panel`
  - `../../../intelliscan/output_student_deploy/SN009_SN009_student_r0p5_epoch005`
  - `../../../intelliscan/output_student_deploy/SN009_SN009_student_r0p5_epoch006_panel`
  - `../../../intelliscan/output_student_deploy/SN009_SN009_student_r0p5_epoch007_panel`
  - `../../../intelliscan/output_student_deploy/SN009_SN009_student_r0p5_epoch010_panel`
- `SN002` teacher: `../../../intelliscan/output_student_deploy/SN002_SN002_teacher_ref`
- `SN002` cross-check candidate: `../../../intelliscan/output_student_deploy/SN002_SN002_student_r0p5_epoch005_panel`

## Full-Case Panel Snapshot

The panel does not change the training recipe. It only exposes more checkpoints from the same retained run.

| Checkpoint | Avg Dice | Class 3 | Class 4 |
| --- | ---: | ---: | ---: |
| `epoch_001` | `0.8680` | `0.6449` | `0.8774` |
| `epoch_005` | `0.8848` | `0.7271` | `0.8801` |
| `epoch_006` | `0.8828` | `0.7310` | `0.8805` |
| `epoch_007` | `0.8724` | `0.6637` | `0.8811` |
| `epoch_010` | `0.8894` | `0.7418` | `0.8813` |

At full-case level, `epoch_010` looks strongest in average Dice.

## `SN009` Screen

Whole-pipeline behavior does not follow the full-case ranking.

| Checkpoint | Total (s) | Seg+Met (s) | Dice c3 | Dice c4 | Pad flips | Solder flips | BLT mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old `best.ckpt` | n/a | n/a | `0.0630` | `0.5570` | `63/114` | `22/114` | `2.0263` |
| old `last.ckpt` | `40.79` | `28.38` | `0.0051` | `0.5310` | `55/114` | `13/114` | `1.0500` |
| `epoch_001` | `41.10` | `28.59` | `0.2456` | `0.5657` | `58/114` | `100/114` | `1.4307` |
| `epoch_005` | `40.72` | `28.15` | `0.3026` | `0.5928` | `58/114` | `23/114` | `0.8842` |
| `epoch_006` | `41.26` | `28.55` | `0.3049` | `0.5052` | `56/114` | `75/114` | `1.6579` |
| `epoch_007` | n/a | n/a | `0.1205` | `0.4612` | `37/114` | `86/114` | `1.4430` |
| `epoch_010` | `40.13` | `27.82` | `0.0079` | `0.4209` | `63/114` | `25/114` | `1.8728` |

Takeaways from `SN009` alone:

- `epoch_005` is the best mid-panel compromise.
- `epoch_005` strongly improves class-sensitive segmentation and BLT relative to both old `best.ckpt` and old `last.ckpt`.
- `last.ckpt` still has the lowest defect flips on this sample.
- `epoch_010` looks good at full-case level, but is clearly not good after deployment.

## `SN002` Cross-Check

To avoid overfitting the decision to `SN009`, only the strongest new candidate was expanded to `SN002`.

| Checkpoint | Total (s) | Seg+Met (s) | Dice c3 | Dice c4 | Pad flips | Solder flips | BLT mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old `best.ckpt` | n/a | n/a | `0.0622` | `0.1284` | `2/95` | `65/95` | `2.7116` |
| old `last.ckpt` | `23.83` | `13.20` | `0.4760` | `0.1416` | `1/95` | `35/95` | `3.8168` |
| `epoch_005` | `23.96` | `12.99` | `0.5861` | `0.1277` | `0/95` | `32/95` | `2.6232` |

On `SN002`, `epoch_005` improves over both old checkpoints on the metrics that matter most here:

- lowest pad flips
- lowest solder flips
- lowest BLT mean delta
- strongest class 3 recovery

## Interpretation

This panel confirms three things.

1. Highest full-case Dice is not a safe deployment selector.
2. The old `best.ckpt` is dominated by newer panel candidates and should not be revisited.
3. `epoch_005` is the current best checkpoint-selection candidate, but it is still not deployment-safe.

The main unresolved tradeoff is:

- `epoch_005` is clearly better than old `best.ckpt`
- `epoch_005` is better than old `last.ckpt` on `SN002`
- but on `SN009`, `epoch_005` still has substantially more defect flips than old `last.ckpt`

So the checkpoint panel is useful, but it does not yet solve the deployment problem.

## Decision

Retain:

- deployment-aware checkpoint panel selection as a valid method
- `epoch_005` as the current preferred panel candidate for future downstream-sensitive studies

Do not retain:

- any new pruned-student checkpoint as deployment-ready

Reason:

- no single checkpoint clears both samples cleanly under the whole-pipeline gate

## Consequence For The Paper Track

This is a strong paper result even though it is not a final deployment win:

- checkpoint choice inside the same run materially changes whole-pipeline behavior
- full-case Dice ranking and downstream ranking diverge
- deployment-aware selection is necessary, but still insufficient without better training objectives

## Next Step

If this direction is revisited, selection should be based on a small raw-scan validation panel with a joint rule over:

- full-case Dice
- class 3 / class 4 Dice
- pad defect flips
- solder defect flips
- BLT / pad-misalignment deltas

Do not promote the current student to TRT deployment yet.
