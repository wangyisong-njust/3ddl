# 014 Selection Gate: Whole Pipeline First

## Decision

Future `wp5-seg` compression and deployment work must be judged with a two-level gate:

1. `wp5-seg` quality gate
2. `intelliscan` whole-pipeline gate

Passing the first gate is necessary but not sufficient.

## Why This Is Needed

The deployment ablation in [013_intelliscan_student_deployment_ablation.md](./013_intelliscan_student_deployment_ablation.md) showed that:

- a student can look competitive on full-case Dice
- yet still cause large class-sensitive drift after deployment
- and that drift can be amplified into metrology and defect-flag flips

Therefore, model selection cannot stop at segmentation metrics alone.

## Required Retention Criteria

### `wp5-seg` gate

- evaluate on the full test set, not only a small slice
- check average Dice
- check class 3 and class 4 explicitly

### `intelliscan` gate

- compare against the teacher on real raw scans
- check total pipeline time and segmentation-stage time
- check segmentation agreement
- check class 3 / class 4 stability
- check metrology drift:
  - `pad_misalignment_defect`
  - `solder_extrusion_defect`
  - `BLT` delta

## Practical Rule

Do not retain a method as a preferred deployment candidate if it improves `wp5-seg` metrics but fails the `intelliscan` metrology gate.

Record it as a negative result instead.

## Consequence For Next Steps

The next optimization round should target methods that are more downstream-sensitive, such as:

- class-sensitive supervised reweighting
- boundary-sensitive objectives
- metrology-aware checkpoint selection or auxiliary objectives

These directions are more aligned with the real deployment objective than generic segmentation recovery alone.
