# 008 Teacher Baseline Revalidation

## Objective

Resolve an apparent contradiction:

- historical report baseline is around `0.90`
- a recent quick recovery check showed teacher Dice around `0.78`

## Root Cause

The low `0.7813` value came from a **10-case slice**, not from the full test set.

That number was useful only for same-slice student/teacher comparisons. It should not be read as the full baseline quality of the teacher model.

## Evidence

- `./teacher_slice_bias_summary_20260408.json`
- `../runs/paper_recovery_eval_teacher/metrics/summary.json`
- `../runs/paper_recovery_eval_teacher_50cases/metrics/summary.json`
- `../runs/paper_recovery_eval_teacher_full/metrics/summary.json`

## Results

| Eval Range | Avg Dice | Avg IoU |
| --- | ---: | ---: |
| first 10 test cases | `0.7813` | `0.7208` |
| first 50 test cases | `0.8825` | `0.8296` |
| full 174 test cases | `0.9054` | `0.8540` |

Full-set per-class Dice:

- class 0: `0.9917`
- class 1: `0.9411`
- class 2: `0.9038`
- class 3: `0.7937`
- class 4: `0.8966`

## Conclusion

- The current teacher checkpoint is consistent with the historical `~0.9` baseline story.
- The earlier `0.7813` number was a slice-selection artifact, not a checkpoint failure.
- Future recovery claims must distinguish clearly between:
  - same-slice comparison numbers
  - full-test-set baseline numbers

## Impact On Current Research Line

- The retained class-aware KD direction remains valid as a pruned-student improvement.
- But full recovery should be judged against the full-set teacher baseline, not the 10-case slice alone.
