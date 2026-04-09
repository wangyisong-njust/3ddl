# 019 Checkpoint Panel Third-Sample Check

## Objective

Check whether the `018_checkpoint_panel_selection.md` conclusion still holds after adding a third raw scan.

The goal here is narrow:

- compare only the two strongest surviving checkpoint candidates
- avoid re-expanding the full panel
- test whether `epoch_005` keeps its advantage outside `SN002` and `SN009`

## Candidates

- old `last.ckpt`: `../../runs/paper_fulltrain_kd_c34_2_2_e10_b2/last.ckpt`
- panel `epoch_005`: `../../runs/paper_fulltrain_kd_c34_2_2_e10_b2_ckptpanel/epoch_005.ckpt`

## Sample

- `SN010`: `../../../intelliscan/output_student_deploy/SN010_SN010_teacher_ref`

## Evidence

- Summary JSON: `./checkpoint_panel_extended_summary_20260408.json`
- `SN010` teacher: `../../../intelliscan/output_student_deploy/SN010_SN010_teacher_ref`
- `SN010` last: `../../../intelliscan/output_student_deploy/SN010_SN010_student_r0p5_last`
- `SN010` epoch_005: `../../../intelliscan/output_student_deploy/SN010_SN010_student_r0p5_epoch005_panel`

## Result

| Checkpoint | Voxel agreement | Dice c3 | Dice c4 | Pad flips | Solder flips | BLT mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| old `last.ckpt` | `0.99650` | `0.8043` | `0.2709` | `8/110` | `51/110` | `4.7218` |
| `epoch_005` | `0.99625` | `0.8292` | `0.2519` | `6/110` | `53/110` | `4.1491` |

## Important Note

The `SN010` student runs were launched concurrently on the same GPU.

That means:

- quality and metrology comparisons are still valid
- wall-clock timing from this sample is not used as evidence

This note is about checkpoint quality ranking only.

## Interpretation

The third sample keeps the same pattern seen in `018`.

- `epoch_005` again improves class 3 and BLT mean relative to old `last.ckpt`
- `epoch_005` again improves pad flips relative to old `last.ckpt`
- but `epoch_005` still does not dominate `last.ckpt` on every downstream metric, because solder flips are slightly worse here

So the checkpoint-selection signal is now more stable than before:

- `epoch_005` wins clearly on `SN002`
- `epoch_005` is mixed but still strong on `SN009`
- `epoch_005` is mixed but still strong on `SN010`

## Decision

Keep:

- `epoch_005` as the current preferred checkpoint-selection candidate

Do not keep:

- any current student checkpoint as deployment-ready

Reason:

- even after adding a third sample, the ranking is still sample- and metric-dependent
- the whole-pipeline gate is not yet cleared

## Consequence

This strengthens the paper claim that:

- downstream checkpoint ranking is real
- downstream checkpoint ranking is not equivalent to full-case Dice ranking
- but checkpoint selection alone is still not enough to make the current compressed student safe

## Next Step

Do not add more checkpoint-panel runs right now.

The next meaningful move is to change the training or selection objective in a downstream-sensitive way, then re-run the same whole-pipeline gate.
