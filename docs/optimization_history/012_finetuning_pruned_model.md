# 012 Finetuning Pruned Model

## Objective

Recover segmentation quality after structured pruning.

## Target Files

- `wp5-seg/pruning/finetune_pruned.py`
- `wp5-seg/pruning/run_pruning_pipeline.sh`
- `wp5-seg/reports/segmentation_acceleration_report.md`

## Code Change Summary

- The finetuning script reloads the pruned architecture from the saved `features` tuple and trains it with the same data pipeline and loss family as baseline training.

## Why This Change Was Made

- Pruning reduces capacity; finetuning is required to recover usable segmentation quality.

## Baseline Behavior

- Reported pre-finetune Dice after pruning: `0.3341`

## Optimized Behavior

- Reported post-finetune Dice: `0.9080`
- Reported baseline reference Dice: `0.9038`

## How It Was Tested

- Script inspection plus existing repo report

## Sample(s) Used

- Reported aggregate result on `174` test samples

## Key Timing Numbers

- Reported finetune schedule: `50` epochs, `lr=1e-4`

## Accuracy / Quality Impact

- Reported recovery is strong, but the raw finetuning run output is not stored in this workspace.

## Risks / Caveats

- This is a `Reported but not revalidated in this pass` result.
- There is no local finetune run directory to audit here.

## Conclusion

- Finetuning is an essential part of the pruning pipeline, but the headline Dice recovery should be reproduced from current scripts before reuse in a paper or slide deck.

## Evidence Paths

- `wp5-seg/pruning/finetune_pruned.py`
- `wp5-seg/pruning/run_pruning_pipeline.sh`
- `wp5-seg/reports/segmentation_acceleration_report.md`
