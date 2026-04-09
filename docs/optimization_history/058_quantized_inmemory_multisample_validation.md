# 058 Quantized In-Memory Multisample Validation

## Objective

Validate whether the txt-compatible in-memory detection path is strong enough to become the default `intelliscan` detection mode, and quantify how much it improves the pipeline on the local raw-scan set.

## Why This Follows 057

- `057` showed the remaining no-write drift came from skipping the legacy detection `.txt` precision loss.
- After restoring that precision before 3D merge, the next required step was broader raw-scan validation.

## Target Files

- `intelliscan/main.py`
- `intelliscan/detection.py`

## Code Change Summary

- Switched `PipelineConfig.use_inmemory_detection` default from `False` to `True`.
- Kept the quantized no-write detection path as the new default.
- Added `--file-detection` as an explicit legacy fallback.
- Kept `--ramdisk-detection` as an experiment / isolation harness.

## Benchmark Setup

- Environment: `conda intelliscan`, `CUDA_VISIBLE_DEVICES=3`
- Samples:
  - `SN002`
  - `SN003`
  - `SN008`
  - `SN009`
  - `SN010`
  - `SN011`
  - `SN012`
  - `SN061`
- Output base:
  - `intelliscan/output_inmemory_validate/`
- Main summary:
  - `intelliscan/output_inmemory_validate/analysis/multisample_quantized_inmemory_summary.json`

## Measured Results

| Sample | Current File Default | Quantized In-Memory | Total Speedup | Front-End Speedup | Exact |
| --- | ---: | ---: | ---: | ---: | --- |
| `SN002` | `54.01s` | `38.18s` | `29.31%` | `40.51%` | Yes |
| `SN003` | `57.00s` | `45.63s` | `19.95%` | `30.98%` | Yes |
| `SN008` | `60.93s` | `46.75s` | `23.26%` | `34.09%` | Yes |
| `SN009` | `75.22s` | `61.47s` | `18.29%` | `35.31%` | Yes |
| `SN010` | `53.12s` | `39.98s` | `24.73%` | `35.91%` | Yes |
| `SN011` | `53.06s` | `38.43s` | `27.58%` | `37.93%` | Yes |
| `SN012` | `74.26s` | `61.04s` | `17.79%` | `32.76%` | Yes |
| `SN061` | `70.31s` | `56.95s` | `19.00%` | `34.90%` | Yes |

## Aggregate Summary

- Exact samples: `8 / 8`
- Mean total speedup: `22.49%`
- Median total speedup: `21.61%`
- Pooled total speedup: `21.99%`
- Mean front-end speedup: `35.30%`
- Pooled front-end speedup: `35.31%`

## Accuracy / Quality Impact

All eight validated samples are exact:

- `bb3d.npy` identical
- `segmentation.nii.gz` identical
- class Dice `1.0` for classes `0..4`
- `pad_misalignment_defect` flips: `0`
- `solder_extrusion_defect` flips: `0`
- BLT max abs delta: `0.0`

## Historical Workspace Comparison

These are computed from existing workspace outputs, not rerun in this pass:

- `SN002`
  - historical baseline -> current file default: `1.47%` total gain
  - current file default -> quantized in-memory: `29.31%`
  - historical baseline -> quantized in-memory: `30.35%`
- `SN009`
  - historical baseline -> current file default: `12.96%` total gain
  - current file default -> quantized in-memory: `18.29%`
  - historical baseline -> quantized in-memory: `28.88%`

## Interpretation

- The quantized in-memory path is no longer just a promising experiment.
- On the available local raw-scan set, it is both:
  - materially faster
  - exact at the bbox, segmentation, and metrology levels
- `--ramdisk-detection` is no longer the preferred optimization path. It remains useful only as a factor-isolation harness.

## Conclusion

- Keep the txt-compatible quantized in-memory detector path.
- It is now strong enough to use as the default `intelliscan` detection mode.
- Keep `--file-detection` as a legacy fallback until broader future data continues to agree.

## Next Follow-Up Actions

1. Validate the new default on any newly added raw scans or production-only cases.
2. Shift engineering work from detection parity to the next throughput target, such as pipeline parallelization / overlap.
3. Revalidate actual GLTF artifact generation in a PyVista-enabled environment.

## Evidence Paths

- `intelliscan/output_inmemory_validate/analysis/multisample_quantized_inmemory_summary.json`
- `intelliscan/output_inmemory_validate/analysis/cumulative_engineering_gain_summary.json`
- `intelliscan/output_inmemory_validate/analysis/SN002_compare.json`
- `intelliscan/output_inmemory_validate/analysis/SN003_compare.json`
- `intelliscan/output_inmemory_validate/analysis/SN008_compare.json`
- `intelliscan/output_inmemory_validate/analysis/SN009_compare.json`
- `intelliscan/output_inmemory_validate/analysis/SN010_compare.json`
- `intelliscan/output_inmemory_validate/analysis/SN011_compare.json`
- `intelliscan/output_inmemory_validate/analysis/SN012_compare.json`
- `intelliscan/output_inmemory_validate/analysis/SN061_compare.json`
