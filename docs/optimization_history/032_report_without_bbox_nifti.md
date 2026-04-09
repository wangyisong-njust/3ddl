# 032 Report Without Per-BBox NIfTI Dependency

## Objective

Remove report dependence on `mmt/img/*.nii.gz` and `mmt/pred/*.nii.gz`, so report generation can reconstruct defect views from the original input volume and the assembled full-volume segmentation.

## Target Files

- `intelliscan/main.py`
- `intelliscan/report.py`
- `intelliscan/segmentation.py`

## Code Change Summary

- `main.py` now saves a compact `mmt/segmentation_regions.json` manifest with:
  - `filename`
  - `original_bbox`
  - `expanded_bbox`
  - `crop_shape`
- `report.py` now accepts:
  - `input_volume_path`
  - `segmentation_volume_path`
  - `region_manifest_path`
- Report defect pages now crop directly from:
  - the original input volume already loaded in memory for the current run
  - the assembled `segmentation.nii.gz` volume already available in memory
  - exact expanded-bbox metadata from `segmentation_regions.json`
- The pipeline no longer writes `mmt/img/*.nii.gz`.
- At the time of this change, `mmt/pred/*.nii.gz` was still kept for GLTF generation and the legacy separate metrology path.

## Why This Change Was Made

- After combined segmentation + metrology became the default, `mmt/img` was only kept for report rendering.
- Those raw crop files add I/O and artifact clutter even though the same information already exists in the original volume plus bbox metadata.

## Baseline Behavior

- Report defect pages reload `mmt/img/img_<idx>.nii.gz`
- Report defect pages reload `mmt/pred/pred_<idx>.nii.gz`
- Combined path still writes raw crop NIfTI files only for report visualization

## New Behavior

- Report defect pages reconstruct crops on demand from the original volume and `segmentation.nii.gz`.
- The exact crop region comes from `mmt/segmentation_regions.json`.
- `mmt/img` is no longer produced.
- During the same pipeline run, report rendering reuses the in-memory arrays rather than reloading large NIfTI files from disk.

## Benchmark Setup

- Old reference:
  - `intelliscan/output_combined/SN002_combined_default/`
  - `intelliscan/output_combined/SN009_combined_default/`
- New validation runs:
  - intermediate disk-backed reconstruction:
    - `intelliscan/output_reportless/SN002_reportless/`
    - `intelliscan/output_reportless/SN009_reportless/`
  - final checked-in memory-reuse version:
    - `intelliscan/output_reportless/SN002_reportless_mem/`
    - `intelliscan/output_reportless/SN009_reportless_mem/`
- Summary artifact:
  - `intelliscan/output_reportless/analysis/report_without_bbox_nii_summary.json`

## Measured Results

| Sample | `mmt/img` | `bb3d.npy` vs old | `segmentation.nii.gz` vs old | `metrology.csv` vs old | Report Pages |
| --- | --- | --- | --- | --- | --- |
| `SN002` | not created | identical | identical | equal after sort | `4 -> 4` |
| `SN009` | not created | identical | identical | equal after sort | `4 -> 4` |

## Timing

### Intermediate Disk-Backed Reconstruction

| Sample | Old Seg+Met | New Seg+Met | Old Report | New Report | Old Total | New Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `SN002` | `17.62s` | `14.53s` | `0.86s` | `9.99s` | `55.02s` | `62.99s` |
| `SN009` | `47.64s` | `36.94s` | `0.92s` | `5.56s` | `86.99s` | `82.62s` |

### Final Checked-In Memory-Reuse Version

| Sample | Old Seg+Met | New Seg+Met | Old Report | New Report | Old Total | New Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `SN002` | `17.62s` | `14.95s` | `0.86s` | `0.81s` | `55.02s` | `54.38s` |
| `SN009` | `47.64s` | `37.11s` | `0.92s` | `0.81s` | `86.99s` | `81.15s` |

## Accuracy / Quality Impact

- This report change does not alter detection, segmentation, or metrology computation.
- On both validated samples:
  - `bb3d.npy` is identical to the earlier combined-default reference
  - `segmentation.nii.gz` is identical
  - `metrology.csv` is equal after sorting by `filename`
  - `final_report.pdf` still exists and has the same page count

## Risks / Caveats

- A first disk-backed reconstruction attempt was safe but slow; that version should be treated as an intermediate experiment, not the final design.
- The checked-in version avoids that regression by reusing in-memory arrays already present in the pipeline run.
- This note only removes the report dependency.
- The later `033_gltf_without_bbox_pred_nifti.md` note removes the combined-path GLTF dependency on `mmt/pred`.

## Conclusion

- Report no longer depends on per-bbox raw crop NIfTI files.
- This change is safe to keep because validated segmentation and metrology outputs are unchanged.
- It reduces artifact sprawl and removes one class of unnecessary intermediate files.
- The final checked-in memory-reuse version is also faster than the earlier combined-default reference on both validated samples.

## Next Follow-Up Actions

1. If further I/O reduction is needed, remove GLTF dependence on per-bbox `mmt/pred/*.nii.gz`.
   Status: completed later in `033_gltf_without_bbox_pred_nifti.md`.
2. Expand validation to more raw-scan samples.
3. Keep the distinction clear: this change is post-inference artifact reconstruction, unlike the experimental true in-memory detection path.

## Evidence Paths

- `intelliscan/output_reportless/analysis/report_without_bbox_nii_summary.json`
- `intelliscan/output_reportless/SN002_reportless/timing.log`
- `intelliscan/output_reportless/SN009_reportless/timing.log`
- `intelliscan/output_reportless/SN002_reportless_mem/timing.log`
- `intelliscan/output_reportless/SN009_reportless_mem/timing.log`
