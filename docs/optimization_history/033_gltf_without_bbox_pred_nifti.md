# 033 GLTF Without Per-BBox Prediction NIfTI Dependency

## Objective

Remove GLTF generation dependence on `mmt/pred/pred_<idx>.nii.gz` so the combined pipeline can stop writing per-bbox prediction masks by default.

## Target Files

- `intelliscan/main.py`
- `intelliscan/gltf_utils.py`

## Code Change Summary

- Added an in-memory GLTF export helper in `gltf_utils.py` that accepts a labeled 3D volume directly.
- Added `generate_all_bump_gltfs_from_results(...)` so bump GLTF export can use segmentation `results` already present in memory.
- Combined segmentation + metrology now accepts `save_predictions=False` and no longer creates `mmt/pred` by default.
- CLI now exposes `--save-bump-predictions` as an explicit legacy/debug fallback.
- The separate segmentation + metrology path still writes `mmt/pred` because file-based metrology requires it.

## Why This Change Was Made

- After report rendering stopped needing `mmt/pred`, bump GLTF export was the last remaining reason the default combined path still wrote per-bbox prediction NIfTIs.
- That made the combined path cleaner than baseline, but still not a true no-per-bbox-mask default.

## Baseline Behavior

- Combined path saved `mmt/pred/pred_<idx>.nii.gz` for every bbox.
- GLTF generation scanned `mmt/pred/*.nii.gz` and converted those files one by one.

## New Behavior

- Combined path no longer saves per-bbox prediction NIfTIs unless `--save-bump-predictions` is set.
- GLTF generation now uses the in-memory segmentation results list and keeps the same output naming for `.gltf` files.
- The output manifest `mmt/segmentation_regions.json` remains available for report reconstruction.

## Benchmark Setup

- Environment:
  - `CUDA_VISIBLE_DEVICES=3`
  - GPU selected from `nvidia-smi` as the lowest-memory-use device at run start in this pass
- Full pipeline A/B:
  - `intelliscan/output_gltfless/SN002_gltfpred_ref/`
  - `intelliscan/output_gltfless/SN002_gltfmem_default/`
  - `intelliscan/output_gltfless/SN009_gltfpred_ref/`
  - `intelliscan/output_gltfless/SN009_gltfmem_default/`
- Isolated stage benchmark:
  - `intelliscan/output_gltfless/analysis/gltfless_stage_only_benchmark.json`
  - same volume
  - same `bb3d.npy`
  - same loaded segmentation model
  - both save/no-save orders executed to reduce order bias

## Measured Results

### Output Parity

| Sample | `bb3d.npy` | `segmentation.nii.gz` | `metrology.csv` | `mmt/pred` after change |
| --- | --- | --- | --- | --- |
| `SN002` | identical | identical | equal after sort | not created |
| `SN009` | identical | identical | equal after sort | not created |

### Full-Pipeline Single Runs

| Sample | Save `mmt/pred` | No `mmt/pred` | Seg+Met Stage |
| --- | ---: | ---: | ---: |
| `SN002` total | `63.68s` | `68.14s` | `18.08s -> 17.39s` |
| `SN009` total | `83.10s` | `85.39s` | `40.78s -> 43.81s` |

Interpretation:

- These single-run end-to-end totals are noisy because `NII -> JPG` and detection dominate wall-clock time and vary between runs.
- They should not be treated as the cleanest proof for this optimization.

### Isolated Combined Segmentation + Metrology Benchmark

| Sample | Mean Save `mmt/pred` | Mean No `mmt/pred` | Relative Speedup |
| --- | ---: | ---: | ---: |
| `SN002` | `7.8960s` | `7.2006s` | `8.81%` |
| `SN009` | `35.4865s` | `33.5964s` | `5.33%` |

## Accuracy / Quality Impact

- On both validated samples:
  - `bb3d.npy` is identical
  - `segmentation.nii.gz` is identical
  - `metrology.csv` is equal after sorting by `filename`
- This is expected because the change only removes artifact writes and changes GLTF inputs from file-backed masks to the same in-memory prediction arrays.

## Risks / Caveats

- PyVista is not installed in this workspace, so actual bump GLTF file generation was not revalidated end-to-end in this pass.
- The dependency removal is validated in code and by the absence of `mmt/pred` in successful combined runs, but `.gltf` output creation itself still needs a PyVista-enabled environment to be rechecked.
- Full-pipeline wall-clock comparisons are noisier than the isolated stage benchmark.

## Conclusion

- This is a safe engineering cleanup worth keeping.
- The default combined path no longer needs `mmt/pred`.
- The isolated benchmark shows the target `3D Segmentation + Metrology` stage is meaningfully faster when per-bbox prediction NIfTIs are skipped.
- For future benchmarking and writing, use the isolated stage numbers as the primary evidence and treat single full-pipeline totals as secondary.

## Next Follow-Up Actions

1. Revalidate actual `.gltf` output creation in an environment with PyVista installed.
2. Expand the same no-`mmt/pred` validation across more raw-scan samples.
3. Keep moving safe engineering work toward geometry-preserving runtime gains rather than context-trimming shortcuts.

## Evidence Paths

- `intelliscan/output_gltfless/analysis/gltfless_summary.json`
- `intelliscan/output_gltfless/analysis/gltfless_stage_only_benchmark.json`
- `intelliscan/output_gltfless/SN002_gltfpred_ref/metrics.json`
- `intelliscan/output_gltfless/SN002_gltfmem_default/metrics.json`
- `intelliscan/output_gltfless/SN009_gltfpred_ref/metrics.json`
- `intelliscan/output_gltfless/SN009_gltfmem_default/metrics.json`
