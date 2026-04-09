# 009 Full-Case Recovery Ranking

## Objective

Judge pruned-student recovery against the real baseline:

- full teacher checkpoint
- full `174`-case test set
- not a 10-case slice

This note answers the question: which current KD variant is actually best once everything is compared on the full evaluation set?

## Evidence

- Teacher full baseline: `../../runs/paper_recovery_eval_teacher_full/metrics/summary.json`
- Small full-eval plain: `../../runs/paper_full_eval_plain/metrics/summary.json`
- Small full-eval uniform KD: `../../runs/paper_full_eval_uniform/metrics/summary.json`
- Small full-eval class-aware `1,1,1,4,2`: `../../runs/paper_full_eval_c34_4_2/metrics/summary.json`
- Small full-eval class-aware `1,1,1,2,2`: `../../runs/paper_full_eval_c34_2_2/metrics/summary.json`
- Full-train plain: `../../runs/paper_fulltrain_plain_e10_b2/best_metrics.json`
- Full-train plain report: `../../runs/paper_fulltrain_plain_e10_b2/finetune_report.json`
- Full-train plain standalone eval: `../../runs/paper_fulltrain_plain_e10_b2_eval/metrics/summary.json`
- Full-train KD `1,1,1,2,2`: `../../runs/paper_fulltrain_kd_c34_2_2_e10_b2/best_metrics.json`
- Full-train KD report: `../../runs/paper_fulltrain_kd_c34_2_2_e10_b2/finetune_report.json`
- Full-train KD standalone eval: `../../runs/paper_fulltrain_kd_c34_2_2_e10_b2_eval/metrics/summary.json`

## Teacher Reference

Full teacher baseline on all `174` test cases:

| Metric | Value |
| --- | ---: |
| average Dice | `0.9054` |
| class 3 Dice | `0.7937` |
| class 4 Dice | `0.8966` |

## Small-Scale Candidate Ranking

These checkpoints came from reduced training experiments, but were re-evaluated on the full test set:

| Variant | Avg Dice | Class 3 | Class 4 | Gap vs teacher |
| --- | ---: | ---: | ---: | ---: |
| plain | `0.5952` | `0.1585` | `0.2708` | `-0.3102` |
| uniform KD | `0.6930` | `0.1585` | `0.6807` | `-0.2124` |
| class-aware `1,1,1,4,2` | `0.6893` | `0.1416` | `0.6744` | `-0.2161` |
| class-aware `1,1,1,2,2` | `0.6934` | `0.1469` | `0.6884` | `-0.2120` |

Conclusion from the full-case ranking:

- the earlier 10-case preference for `1,1,1,4,2` does **not** hold on the full test set
- among the small-scale KD candidates, `1,1,1,2,2` is the best retained choice

## Full-Train Recovery Comparison

Both runs used the full training set (`subset_ratio=1.0`) and full-test evaluation each epoch.

| Variant | Best Epoch | Avg Dice | Class 3 | Class 4 | Gap vs teacher |
| --- | ---: | ---: | ---: | ---: | ---: |
| full-train plain | `10` | `0.8785` | `0.7122` | `0.8707` | `-0.0269` |
| full-train KD `1,1,1,2,2` | `7` | `0.8895` | `0.7545` | `0.8749` | `-0.0159` |

Standalone `eval.py` revalidation matched these full-case rankings:

- full-train plain standalone eval average Dice: `0.878458`
- full-train KD standalone eval average Dice: `0.889527`

So the retained ranking does not depend on the training-loop callback alone.

Training cost:

| Variant | Train Time |
| --- | ---: |
| full-train plain | `341.28s` |
| full-train KD `1,1,1,2,2` | `440.02s` |

## Interpretation

- Full-case testing changes the decision boundary. Small-slice ranking alone was not reliable enough.
- After full training, KD is no longer a weak idea or a toy-only win. It beats plain finetuning on:
  - average Dice
  - class 3 Dice
  - class 4 Dice
- The remaining teacher gap is now much smaller:
  - plain: `-0.0269`
  - KD `1,1,1,2,2`: `-0.0159`

## Keep / Discard

Keep:

- immediate KD
- class-aware KD
- current best retained setting: `--distill-weight 0.1 --distill-temperature 2.0 --kd-class-weights 1,1,1,2,2`

Discard as preferred choices:

- delayed KD
- `1,1,1,4,2` as the leading class-aware configuration
- any claim based only on the first `10` test cases

## Current Position

The best current compressed student is:

- `pruned_50pct.ckpt`
- full-train finetune with class-aware immediate KD
- full-case best Dice `0.8895`

This is still below the teacher `0.9054`, so it is not a full-replacement claim yet. But it is now close enough to justify the next paper-stage exploration.

## Next Step

Only methods that beat the current retained KD baseline should continue.

The next candidate should target the remaining recovery gap, especially:

- better preservation of class 4
- possibly feature-level KD
- later, real-data PTQ/QAT on top of the best recovered student
