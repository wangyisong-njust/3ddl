# 011 Structured Pruning

## Objective

Reduce segmentation model parameters and checkpoint size through structured channel pruning.

## Target Files

- `wp5-seg/pruning/prune_basicunet.py`
- `wp5-seg/pruning/output/pruned_50pct.ckpt`
- `wp5-seg/pruning/output/pruned_50pct.json`

## Code Change Summary

- The pruning script builds a smaller BasicUNet by pruning channels symmetrically and saving the new `features` tuple alongside the pruned state dict.

## Why This Change Was Made

- Parameter count and model size directly affect deployment cost and can reduce inference latency.

## Baseline Behavior

- Original features: `(32, 32, 64, 128, 256, 32)`
- Original parameters: `5,749,509`

## Optimized Behavior

- Pruned features: `(16, 16, 32, 64, 128, 16)`
- Pruned parameters: `1,438,853`

## How It Was Tested

- Validated from the saved pruning metadata artifact in workspace

## Sample(s) Used

- Model-level artifact; no sample-specific timing file is attached to this note

## Key Timing Numbers

- None stored for pruning itself

## Accuracy / Quality Impact

- Accuracy after pruning alone is not captured in this artifact.
- See `012_finetuning_pruned_model.md` for the reported recovery path.

## Risks / Caveats

- Pruning without finetuning can substantially degrade segmentation quality.

## Conclusion

- The parameter reduction is strongly evidenced by local artifacts and is a valid baseline for all later compression work.

## Evidence Paths

- `wp5-seg/pruning/output/pruned_50pct.json`
- `wp5-seg/pruning/output/pruned_50pct.ckpt`
