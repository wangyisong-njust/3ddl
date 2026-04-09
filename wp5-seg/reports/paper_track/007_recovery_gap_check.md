# 007 Recovery Gap Check

## Objective

Verify whether the current retained student actually recovers enough of the full-model baseline, rather than only beating other pruned variants.

## Setup

Shared evaluation slice:

- data: `3ddl-dataset/data`
- eval range: first `10` test cases
- metric mode: fast Dice/IoU only

Models compared:

- teacher / full model: `../intelliscan/models/segmentation_model.ckpt`
- current retained student: `runs/paper_classaware_kd_c34_4_2/best.ckpt`

Evidence:

- `./recovery_compare_teacher_vs_classaware_20260408.json`
- `../runs/paper_recovery_eval_teacher/metrics/summary.json`
- `../runs/paper_recovery_eval_student_c34_4_2/metrics/summary.json`

## Results

| Model | Avg Dice | Class 1 | Class 2 | Class 3 | Class 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| teacher | `0.7813` | `0.9040` | `0.9028` | `0.2645` | `0.8470` |
| class-aware KD student | `0.7316` | `0.8856` | `0.8484` | `0.2709` | `0.6728` |

Gap (student - teacher):

- average Dice: `-0.0497`
- class 1: `-0.0183`
- class 2: `-0.0544`
- class 3: `+0.0065`
- class 4: `-0.1742`

## Interpretation

- The retained student is now clearly stronger than earlier pruned baselines.
- But it is still **not** recovered enough to be called full-baseline-equivalent.
- The biggest remaining problem is class 4.
- Class 3 is already slightly above the teacher on this shared slice, so the remaining recovery work should focus more on class-4 retention than on class-3 rescue.

## Decision

- Keep class-aware KD as the current best pruned-student direction.
- Do **not** claim that recovery is complete.
- Do **not** treat the current student as deployment-ready if the target is to stay close to the full baseline.

## Next Step

The next retained research move should be chosen for class-4 recovery:

1. feature-level KD
2. real-data PTQ / INT8 calibration only after the student itself is stronger
3. larger-scale recovery runs once the KD recipe stabilizes
