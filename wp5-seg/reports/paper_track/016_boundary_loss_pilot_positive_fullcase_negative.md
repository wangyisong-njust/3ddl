# 016 Boundary Loss: Pilot Positive, Full-Case Negative

## Objective

Test whether a simple boundary-sensitive objective can improve the current retained `r=0.5` KD student under the new whole-pipeline-first selection rule.

The extra loss was:

- GT boundary mask built from classes `1,2,3,4`
- masked CE on those boundary voxels

## Evidence

- Summary JSON: `./boundary_loss_summary_20260408.json`
- Pilot baseline: `../../runs/paper_supervised_reweight_r0p5_plainpilot`
- Pilot `bw=0.25`: `../../runs/paper_boundaryloss_r0p5_bw025pilot`
- Pilot `bw=0.5`: `../../runs/paper_boundaryloss_r0p5_bw05pilot`
- Full-train `bw=0.25`: `../../runs/paper_fulltrain_boundary_bw025_e10_b2`
- Retained full-train baseline: `../../runs/paper_fulltrain_kd_c34_2_2_e10_b2`

## Pilot Result

Small pilot setup:

- `subset_ratio=0.05`
- `epochs=3`
- `max_eval_cases=10`

| Variant | Boundary weight | Best avg Dice | Class 3 | Class 4 |
| --- | ---: | ---: | ---: | ---: |
| baseline | `0.0` | `0.7674` | `0.1753` | `0.8666` |
| candidate | `0.25` | `0.7830` | `0.2405` | `0.8736` |
| candidate | `0.5` | `0.7800` | `0.2605` | `0.8573` |

Pilot conclusion:

- both boundary-loss candidates beat the pilot baseline
- `0.25` was the better of the two

## Full-Case Validation

Full-train setup for the promoted candidate:

- `subset_ratio=1.0`
- `epochs=10`
- full test set

| Variant | Full-case avg Dice | Class 3 | Class 4 | Gap vs retained baseline |
| --- | ---: | ---: | ---: | ---: |
| retained KD baseline | `0.8895` | `0.7545` | `0.8749` | reference |
| boundary loss `0.25` | `0.8895` pre-finetune, then `0.8556` at epoch 5, `0.8350` at epoch 10 | `0.6614` / `0.5306` | `0.8352` / `0.8646` | worse |

Important detail:

- the best full-case checkpoint did not improve beyond the training start point
- in practice, this means the retained baseline still wins

## Interpretation

- The boundary objective looked promising in the small pilot.
- That pilot signal did not survive full-case training.
- This is exactly the kind of false positive the new selection gate is meant to reject.

## Decision

Do not keep the current boundary-loss method as a retained training strategy.

Reason:

- it passed the pilot screen
- but failed the full-case gate
- so it should not advance to `intelliscan` deployment

## Code Handling

The temporary experimental control used for this boundary-loss study is discarded after recording the result.

## Consequence

This is another strong reminder that:

- small pilot wins are useful for screening
- but they are not enough to retain a method
- full-case validation remains mandatory before deployment testing

## Next Step

Move to the next downstream-sensitive direction:

- metrology-aware checkpoint selection

This is lower-risk than adding another new loss immediately, and it aligns directly with the whole-pipeline gate.
