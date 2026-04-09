# 012 Pruning Ratio Pareto Ranking

## Objective

Turn the pruning-ratio screening and full-train recovery runs into a first real Pareto ranking.

This note answers:

- which ratio is currently best for quality
- which ratio is currently best for speed
- which candidates should move into `intelliscan` deployment tests

## Evidence

- Teacher full baseline: `../../runs/paper_recovery_eval_teacher_full/metrics/summary.json`
- Ratio screening summary: `./pruning_ratio_pareto_summary_20260408.json`
- `r=0.375` full-train eval: `../../runs/paper_fulltrain_kd_r0p375_e10_b2_eval/metrics/summary.json`
- `r=0.5` retained full-train eval: `../../runs/paper_fulltrain_kd_c34_2_2_e10_b2_eval/metrics/summary.json`
- `r=0.625` full-train eval: `../../runs/paper_fulltrain_kd_r0p625_e10_b2_eval/metrics/summary.json`

All three use the same retained KD recipe:

- `--distill-weight 0.1`
- `--distill-temperature 2.0`
- `--kd-class-weights 1,1,1,2,2`

## Teacher Reference

Full teacher baseline on `174` test cases:

| Metric | Value |
| --- | ---: |
| average Dice | `0.90538` |
| class 3 Dice | `0.79370` |
| class 4 Dice | `0.89656` |

## Pareto Table

| Ratio | Params | Reduction | FP32 ms | FP32 speedup | Avg Dice | Class 3 | Class 4 | Gap vs teacher |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.375` | `2,247,285` | `60.91%` | `15.28` | `1.14x` | `0.89513` | `0.78690` | `0.87169` | `-0.01025` |
| `0.5` | `1,438,853` | `74.97%` | `6.39` | `2.72x` | `0.88953` | `0.75450` | `0.87494` | `-0.01586` |
| `0.625` | `809,909` | `85.91%` | `5.00` | `3.47x` | `0.88210` | `0.74877` | `0.86175` | `-0.02328` |

## Interpretation

- `r=0.375` is the current quality winner.
  - It is the closest to the teacher on average Dice.
  - It is also the closest to the teacher on class 3.
- `r=0.5` is the strongest middle point.
  - It gives a much larger speedup than `r=0.375`.
  - Its quality drop versus `r=0.375` is real but still moderate.
- `r=0.625` is the aggressive edge of the current front.
  - It gives the best patch-latency result.
  - But the full-case quality drop is now large enough that it should not be the first deployment candidate.

## Current Pareto View

The current first-pass front is:

- quality-oriented point: `r=0.375`
- balanced point: `r=0.5`
- aggressive speed point: `r=0.625`

But for the next stage, not all three should be treated equally.

## Decision

Move first into `intelliscan` deployment tests:

- `r=0.375`
- `r=0.5`

Keep as a secondary aggressive candidate:

- `r=0.625`

Do not continue:

- `r=0.25`

## Why `r=0.375` And `r=0.5` Go First

- `r=0.375` tells us how much end-to-end latency we can save while staying very close to the teacher.
- `r=0.5` tells us whether the strongest current compression point survives real raw-scan metrology checks.
- Together they are enough to start the deployment story without over-expanding the experiment matrix.

## Next Step

Start the second paper-track line:

1. deploy `r=0.375` and `r=0.5` into `intelliscan`
2. compare them against the teacher model on real raw scans
3. measure:
   - total pipeline time
   - segmentation stage time
   - class-sensitive output drift
   - metrology / defect flip stability

Only after that should the deployment path expand to TRT FP16 / INT8.
