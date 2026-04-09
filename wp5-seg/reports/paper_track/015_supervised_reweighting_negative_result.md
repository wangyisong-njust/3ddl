# 015 Supervised Reweighting Negative Result

## Objective

Test whether class-sensitive supervised reweighting can improve the current retained `r=0.5` KD student before revisiting full-case ranking or `intelliscan` deployment.

The goal was simple:

- keep the retained KD recipe fixed
- change only the supervised CE weighting
- keep the method only if it beats the current pilot baseline

## Setup

Fixed settings for all runs:

- student: `../../runs/paper_fulltrain_kd_c34_2_2_e10_b2/best.ckpt`
- teacher: `../../../intelliscan/models/segmentation_model.ckpt`
- KD: `--distill-weight 0.1 --distill-temperature 2.0 --kd-class-weights 1,1,1,2,2`
- pilot scale:
  - `subset_ratio=0.05`
  - `epochs=3`
  - `max_eval_cases=10`
  - `batch_size=2`
  - `bf16 AMP`

## Evidence

- Summary JSON: `./supervised_reweight_pilot_summary_20260408.json`
- Baseline pilot: `../../runs/paper_supervised_reweight_r0p5_plainpilot`
- `1,1,1,2,2`: `../../runs/paper_supervised_reweight_r0p5_supc22pilot`
- `1,1,1,2,4`: `../../runs/paper_supervised_reweight_r0p5_supc24pilot`
- `1,1,1,4,2`: `../../runs/paper_supervised_reweight_r0p5_supc42pilot`

## Pilot Ranking

| Variant | Supervised class weights | Best avg Dice | Recovery | Class 3 | Class 4 |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | `1,1,1,1,1` | `0.7674` | `+0.0172` | `0.1753` | `0.8666` |
| candidate | `1,1,1,2,2` | `0.7644` | `+0.0141` | `0.1566` | `0.8688` |
| candidate | `1,1,1,2,4` | `0.7639` | `+0.0136` | `0.1538` | `0.8653` |
| candidate | `1,1,1,4,2` | `0.7621` | `+0.0118` | `0.1448` | `0.8690` |

## Interpretation

- None of the supervised reweighting candidates beat the baseline pilot.
- The small gains on class 4 in some runs did not compensate for the drop in average Dice.
- Stronger emphasis on class 3 also did not improve the overall pilot ranking.

## Decision

Do not keep class-sensitive supervised reweighting as a retained method at this stage.

Rejected variants:

- `1,1,1,2,2`
- `1,1,1,2,4`
- `1,1,1,4,2`

Reason:

- all of them lost to the baseline pilot under the same KD setup
- this fails the first gate, so they should not move to full-case or `intelliscan` validation

## Code Handling

The temporary experimental control used for this pilot is discarded after recording the result.

## Consequence

The next direction should not be more generic class reweighting.

More promising downstream-sensitive directions remain:

- boundary-sensitive objectives
- metrology-aware checkpoint selection
- metrology-aware auxiliary objectives
