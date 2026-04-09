# 030 Pipeline I/O Optimization

## Objective

Reduce intermediate file writing and reading between segmentation and metrology.

## Target Files

- `intelliscan/main.py`
- `intelliscan/segmentation.py`
- `wp5-seg/reports/segmentation_acceleration_report.md`

## Code Change Summary

- The codebase contains a `use_combined_seg_metrology` branch intended to keep segmentation and metrology more tightly coupled.
- The current default path still writes per-bbox `nii.gz` files under `mmt/` and then re-reads them for metrology.

## Why This Change Was Made

- Existing report material identifies intermediate `nib.save()` usage as a major cost.

## Baseline Behavior

- Current default path:
  - writes `mmt/img/*.nii.gz`
  - writes `mmt/pred/*.nii.gz`
  - reads `mmt/pred/*.nii.gz` again for metrology

## Optimized Behavior

- Intended direction:
  - reduce or remove intermediate NIfTI traffic
  - make combined segmentation + metrology practical

## How It Was Tested

- Current default path verified by code inspection and SN009 outputs
- Historical improvement numbers come from repo report only

## Sample(s) Used

- SN009 for current code-path validation
- SN002 in repo report for historical timing claims

## Key Timing Numbers

Validated:

- SN009 baseline metrology stage: `16.95s`
- SN009 baseline output includes `114` saved prediction files under `mmt/pred/`

Reported but not revalidated in this pass:

- SN002 segmentation `14.14s -> 10.60s` after removing intermediate NIfTI writes

## Accuracy / Quality Impact

- No current local evidence proves that the no-write path is output-equivalent.

## Risks / Caveats

- Current default code path is still I/O-heavy.
- Historical I/O savings should not be presented as current-code performance without rerunning the same experiment.

## Conclusion

- I/O remains a first-class bottleneck in the current workspace.
- This note records the original direction only; the validated default combined path now lives in `optimization_history/031_combined_seg_metrology_default.md`.

## Evidence Paths

- `intelliscan/main.py`
- `intelliscan/output_formal/SN009_current_gpu/metrics.json`
- `intelliscan/output_formal/SN009_current_gpu/metrology/metrology.csv`
- `wp5-seg/reports/segmentation_acceleration_report.md`
