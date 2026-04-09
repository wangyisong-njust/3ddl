# 017 Metrology-Aware Checkpoint Selection Probe

## Objective

Test whether checkpoint choice inside the same retained training run can materially change downstream `intelliscan` stability, even when full-case Dice is nearly unchanged.

This is a low-risk probe:

- no new training loss
- no new deployment model family
- only compare `best.ckpt` vs `last.ckpt`

## Evidence

- Summary JSON: `./checkpoint_selection_probe_20260408.json`
- Retained run: `../../runs/paper_fulltrain_kd_c34_2_2_e10_b2`
- `SN002` best deployment: `../../../intelliscan/output_student_deploy/SN002_SN002_student_r0p5`
- `SN002` last deployment: `../../../intelliscan/output_student_deploy/SN002_SN002_student_r0p5_last`
- `SN009` best deployment: `../../../intelliscan/output_student_deploy/SN009_SN009_student_r0p5`
- `SN009` last deployment: `../../../intelliscan/output_student_deploy/SN009_SN009_student_r0p5_last`

## Full-Case Difference

Inside the same run:

| Checkpoint | Epoch | Avg Dice | Class 3 | Class 4 |
| --- | ---: | ---: | ---: | ---: |
| `best.ckpt` | `7` | `0.88953` | `0.75450` | `0.87494` |
| `last.ckpt` | `10` | `0.88831` | `0.74634` | `0.87428` |

The full-case gap is very small:

- average Dice delta: about `-0.00121`

## Whole-Pipeline Result

### `SN002`

| Checkpoint | Pad flips | Solder flips | BLT mean abs delta |
| --- | ---: | ---: | ---: |
| `best.ckpt` | `2/95` | `65/95` | `2.71` |
| `last.ckpt` | `1/95` | `35/95` | `3.82` |

### `SN009`

| Checkpoint | Pad flips | Solder flips | BLT mean abs delta |
| --- | ---: | ---: | ---: |
| `best.ckpt` | `63/114` | `22/114` | `2.03` |
| `last.ckpt` | `55/114` | `13/114` | `1.05` |

## Interpretation

- The checkpoint with the best segmentation Dice is not automatically the best deployment checkpoint.
- On both raw scans, `last.ckpt` reduced defect flips relative to `best.ckpt`.
- The tradeoff is not uniformly better on every metric:
  - `SN002` BLT mean delta became worse
  - `SN009` class 3 Dice became even worse

## Decision

Keep metrology-aware checkpoint selection as a promising direction.

Do not yet retain `last.ckpt` as the new default deployment checkpoint.

Reason:

- the signal is promising
- but it is not yet consistently better across all deployment-sensitive metrics
- the sample count is still small

## Consequence

This is the first strong sign that whole-pipeline ranking may need a deployment-aware checkpoint selector, not just the highest full-case Dice checkpoint.

## Next Step

Expand this direction in a controlled way:

1. compare more checkpoints from the same run, not only `best` vs `last`
2. evaluate them on a small raw-scan validation panel
3. select checkpoints using a joint rule over:
   - full-case Dice
   - class 3 / class 4
   - metrology flip counts
   - BLT / misalignment deltas
