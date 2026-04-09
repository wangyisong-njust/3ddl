# 013 Intelliscan Student Deployment Ablation

## Objective

Check whether the best current pruned students from `wp5-seg` remain acceptable after deployment into the full `intelliscan` pipeline.

This note answers a stricter question than patch-level evaluation:

- do pruned students actually speed up end-to-end raw-scan inference
- do they preserve segmentation behavior relative to the teacher inside `intelliscan`
- do they preserve downstream metrology and defect flags

## Candidates

Deployed candidates:

- teacher reference: `../../intelliscan/models/segmentation_model.ckpt`
- quality-oriented student: `../../runs/paper_fulltrain_kd_r0p375_e10_b2/best.ckpt`
- balanced student: `../../runs/paper_fulltrain_kd_c34_2_2_e10_b2/best.ckpt`

All student checkpoints use the retained KD recipe from earlier notes:

- `--distill-weight 0.1`
- `--distill-temperature 2.0`
- `--kd-class-weights 1,1,1,2,2`

## Samples

- `SN002`: `../../../intelliscan/output_student_deploy/SN002_SN002_teacher_ref`
- `SN009`: `../../../intelliscan/output_student_deploy/SN009_SN009_teacher_ref`

## Evidence

- Unified deployment summary: `../../../intelliscan/output_student_deploy/analysis/student_deployment_summary_20260408.json`
- `SN002` teacher: `../../../intelliscan/output_student_deploy/SN002_SN002_teacher_ref`
- `SN002` `r=0.375`: `../../../intelliscan/output_student_deploy/SN002_SN002_student_r0p375`
- `SN002` `r=0.5`: `../../../intelliscan/output_student_deploy/SN002_SN002_student_r0p5`
- `SN009` teacher: `../../../intelliscan/output_student_deploy/SN009_SN009_teacher_ref`
- `SN009` `r=0.375`: `../../../intelliscan/output_student_deploy/SN009_SN009_student_r0p375`
- `SN009` `r=0.5`: `../../../intelliscan/output_student_deploy/SN009_SN009_student_r0p5`

## Runtime Summary

| Sample | Variant | Total (s) | Seg+Met (s) | Relative to teacher |
| --- | --- | ---: | ---: | ---: |
| `SN002` | teacher | `25.03` | `14.13` | reference |
| `SN002` | `r=0.375` | `25.03` | `13.94` | total `-0.01%`, seg+met `-1.35%` |
| `SN002` | `r=0.5` | `24.16` | `13.03` | total `-3.48%`, seg+met `-7.83%` |
| `SN009` | teacher | `48.55` | `36.17` | reference |
| `SN009` | `r=0.375` | `46.37` | `34.12` | total `-4.48%`, seg+met `-5.67%` |
| `SN009` | `r=0.5` | `41.11` | `28.37` | total `-15.32%`, seg+met `-21.57%` |

## Drift Summary Vs Teacher

| Sample | Variant | Voxel agreement | Dice c3 | Dice c4 | Pad flips | Solder flips | BLT mean abs delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `SN002` | `r=0.375` | `0.99717` | `0.28666` | `0.05481` | `16/95` | `50/95` | `7.21` |
| `SN002` | `r=0.5` | `0.99695` | `0.06222` | `0.12835` | `2/95` | `65/95` | `2.71` |
| `SN009` | `r=0.375` | `0.98960` | `0.50067` | `0.50452` | `32/114` | `14/114` | `1.61` |
| `SN009` | `r=0.5` | `0.98829` | `0.06303` | `0.55703` | `63/114` | `22/114` | `2.03` |

## Interpretation

- Patch-level recovery and full-case Dice did not translate into deployment-safe behavior.
- The teacher-student gap looks moderate at dataset level, but class-sensitive disagreement becomes much larger after the masks are turned into per-bump metrology.
- `r=0.5` is the fastest student, but it is clearly the least stable on class 3 and downstream defect flags.
- `r=0.375` is safer than `r=0.5`, but still not acceptable for direct deployment.

## Why Metrology Amplifies Drift

- `intelliscan` metrology is not a smooth average metric.
- It depends on orientation selection, connected components, class presence, and thresholded measurements in `../../../intelliscan/metrology.py`.
- Defect flags are triggered by hard thresholds such as:
  - `PAD_MISALIGNMENT_THRESHOLD = 0.10`
  - `SOLDER_EXTRUSION_THRESHOLD = 0.10`
- Small class-boundary changes can therefore flip a binary decision even when global voxel agreement remains high.
- This makes raw-scan deployment much more sensitive than patch-level Dice alone.

## Decision

Do not keep direct `intelliscan` deployment of the current pruned students as a retained method.

Rejected as deployment-ready:

- `r=0.375`
- `r=0.5`

Reason:

- both candidates introduce too many metrology flips relative to the teacher
- the speed gains are real, but the downstream quality cost is too high

## Consequence For The Paper Track

This is an important negative result:

- full-case Dice alone is insufficient for deployment ranking
- the deployment gate must include metrology stability
- future student selection should be defect-aware, not only segmentation-aware

## Next Step

Do not move these current students into TRT deployment yet.

The next promising direction is to add downstream-sensitive training signals before revisiting deployment, for example:

- class-sensitive supervised reweighting
- boundary-sensitive loss
- metrology-aware checkpoint selection or auxiliary objectives
