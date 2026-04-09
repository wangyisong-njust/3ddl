# 031 Combined Segmentation + Metrology Default

## Objective

Make the combined segmentation + metrology path usable as the default `intelliscan` behavior without changing segmentation geometry or inference-path selection.

## Target Files

- `intelliscan/main.py`
- `intelliscan/segmentation.py`

## Code Change Summary

- `PipelineConfig.use_combined_seg_metrology` now defaults to `True`.
- CLI now exposes:
  - `--combined-seg-metrology` (default path)
  - `--separate-seg-metrology` (legacy fallback)
- The combined path now reuses `segment_bboxes(...)` instead of running `infer_crop(...)` one bbox at a time.
- `segment_bboxes(...)` can now write `segmentation_stats.json` without requiring a separate metrology reread path.
- `process_metrology(...)` now parses the bbox index from `pred_<idx>.nii.gz` so the `bb` column matches the correct prediction order.

## Why This Change Was Made

- The old combined branch existed, but it bypassed the batched segmentation path and was not ready to make default.
- The legacy default path saved per-bbox NIfTI masks and then re-read them for metrology.
- Earlier experiments and repo notes identified intermediate mask I/O as a real bottleneck.

## Baseline Behavior

- Segment all bboxes and save `mmt/img/*.nii.gz` and `mmt/pred/*.nii.gz`
- Re-read `mmt/pred/*.nii.gz` for metrology
- Keep combined path disabled by default

## Optimized Behavior

- Segment all bboxes through the same `segment_bboxes(...)` logic as baseline
- Keep the per-bbox files needed by the current PDF report
- Compute metrology directly from in-memory predictions instead of re-reading `mmt/pred/*.nii.gz`

## How It Was Tested

- Same-machine GPU runs with:
  - `--separate-seg-metrology`
  - default combined path
- Samples:
  - `SN009` (`114` sliding-window crops)
  - `SN002` (`95` direct-padded crops)

## Key Timing Numbers

| Sample | Baseline Seg+Met | Combined Seg+Met | Speedup |
| --- | ---: | ---: | ---: |
| `SN009` | `64.41s` | `47.64s` | `26.04%` |
| `SN002` | `17.97s` | `17.62s` | `1.99%` |

## Accuracy / Quality Impact

- `segmentation.nii.gz` is identical on both validated samples.
- Metrology CSV is identical after rebuilding the legacy separate CSV with the fixed bbox-index mapping.
- The original separate-run CSVs had a `bb` column pairing bug caused by lexicographic filename order (`pred_10` before `pred_2`).

## Risks / Caveats

- This note captures the point where combined seg+metrology became the default.
- Later notes removed the report dependency on `mmt/img` and the combined-path GLTF dependency on `mmt/pred`:
  - `032_report_without_bbox_nifti.md`
  - `033_gltf_without_bbox_pred_nifti.md`
- End-to-end total time is still influenced by JPG conversion and detection noise; the clean comparison target for this note is the combined segmentation + metrology stage.

## Conclusion

- The combined segmentation + metrology path is validated enough to keep as the default `intelliscan` path.
- It is a safe engineering optimization because it preserves segmentation outputs and metrology values on the validated samples.
- Later follow-up work did remove those remaining per-bbox report / GLTF dependencies from the combined default path; see `032` and `033`.

## Evidence Paths

- `intelliscan/main.py`
- `intelliscan/segmentation.py`
- `intelliscan/output_combined/analysis/combined_seg_metrology_summary.json`
- `intelliscan/output_combined/SN009_separate_ref/metrics.json`
- `intelliscan/output_combined/SN009_combined_default/metrics.json`
- `intelliscan/output_combined/SN002_separate_ref/metrics.json`
- `intelliscan/output_combined/SN002_combined_default/metrics.json`
- `intelliscan/output_combined/SN009_separate_ref/metrology_rebuilt_v2/metrology.csv`
- `intelliscan/output_combined/SN002_separate_ref/metrology_rebuilt_v2/metrology.csv`
