# 011 Pruning Ratio Screening

## Objective

Start the pruning-ratio Pareto line for the paper track.

This pass is only a screening pass:

- prune several ratios from the same teacher checkpoint
- benchmark patch latency
- use the result to decide which ratios deserve full-train recovery

## Implementation

New helper:

- `../../pruning/run_ratio_sweep.py`

This helper reuses existing code instead of duplicating pruning logic:

- `../../pruning/prune_basicunet.py`
- `../../pruning/benchmark.py`

## Setup

- teacher checkpoint: `../../../intelliscan/models/segmentation_model.ckpt`
- benchmark GPU: local `GPU 3`
- input shape: `(1, 1, 112, 112, 80)`
- benchmark runs: `40`
- warmup runs: `10`
- benchmark modes:
  - FP32
  - AMP

Evidence:

- `../../runs/paper_ratio_sweep_screen_v1/ratio_sweep_summary.json`

## Results

| Ratio | Params | Reduction | FP32 ms | FP32 speedup | AMP ms | AMP speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.25` | `3,235,205` | `43.73%` | `19.93` | `0.87x` | `25.21` | `1.34x` |
| `0.375` | `2,247,285` | `60.91%` | `15.28` | `1.14x` | `21.07` | `1.60x` |
| `0.5` | `1,438,853` | `74.97%` | `6.39` | `2.72x` | `17.13` | `1.97x` |
| `0.625` | `809,909` | `85.91%` | `5.00` | `3.47x` | `13.77` | `2.45x` |

## Interpretation

- `0.25` is not worth more work. It preserves too much width and is slower than the teacher in FP32.
- `0.375` is the first ratio that gives a real patch-latency gain while staying much larger than the current `0.5` student.
- `0.5` remains the strongest known middle point and is already backed by full-train recovery evidence.
- `0.625` is the most aggressive screened candidate and deserves a real accuracy test because its latency gain is strong enough to justify the risk.

## Decision

Do not continue:

- ratio `0.25`

Continue to full-train recovery:

- ratio `0.375`
- ratio `0.625`

Retained reference point:

- ratio `0.5`
- full-train class-aware immediate KD

## Runs Launched

To turn this into a real Pareto ranking, two full-train recovery runs were started in this pass:

- `../../runs/paper_fulltrain_kd_r0p375_e10_b2`
- `../../runs/paper_fulltrain_kd_r0p625_e10_b2`

Both use the current retained KD recipe:

- `--distill-weight 0.1`
- `--distill-temperature 2.0`
- `--kd-class-weights 1,1,1,2,2`

## Next Step

When the two full-train runs finish, compare:

- full-case Dice
- class 3 Dice
- class 4 Dice
- parameter count
- patch latency

Then keep only the ratios that are actually on the latency-quality Pareto front.
