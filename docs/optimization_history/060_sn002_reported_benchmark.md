# 060 SN002 Reported Benchmark

## Objective

Preserve the historical SN002 benchmark as a reference target without overstating its current reproducibility.

## Target Files

- `wp5-seg/reports/segmentation_acceleration_report.md`

## Code Change Summary

- No new code in this pass. This note captures an existing report result.

## Why This Change Was Made

- SN002 appears in the existing acceleration report and is useful as a continuity reference for future reruns.

## Baseline Behavior

Reported in the repo report for SN002 with `95` bbox:

- total `59.59s`
- segmentation `14.14s`

## Optimized Behavior

Reported in the repo report after stacked optimizations:

- total `31.71s`
- segmentation `8.60s`

## How It Was Tested

- Not rerun in this documentation pass

## Sample(s) Used

- `SN002` as named in the repo report

## Key Timing Numbers

All numbers in this note are `Reported but not revalidated in this pass`.

## Accuracy / Quality Impact

- Repo report presents the optimized path as accuracy-preserving on its evaluated set.
- That claim has not been rerun from current workspace artifacts.

## Risks / Caveats

- No raw `metrics.json`, `timing.log`, or `metrology.csv` for SN002 is stored in this workspace.

## Conclusion

- SN002 remains a useful historical target and should be reproduced with current code so it can be promoted from reported reference to validated benchmark.

## Evidence Paths

- `wp5-seg/reports/segmentation_acceleration_report.md`
