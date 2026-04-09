# 057 Detection Parity Quantization

## Objective

Resolve the remaining small output drift in no-write detection paths without giving back the front-end speedup.

## Target Files

- `intelliscan/detection.py`
- `intelliscan/main.py`

## Code Change Summary

- Added `_quantize_detections_like_txt(...)` in `intelliscan/detection.py`.
- Applied the same `%d %.2f %.2f %.2f %.2f` precision loss used by the legacy detection `.txt` files before in-memory bbox merge.
- Kept both no-write paths explicit:
  - `--inmemory`
  - `--ramdisk-detection`
- Default production behavior is unchanged.

## Why This Change Was Made

- Pre-quantization `--ramdisk-detection` preserved YOLO `source=<folder>` but still reproduced the same small `SN002` drift seen in the earlier `--inmemory` path.
- That ruled out “disk vs memory” as the main remaining cause.
- The file-based baseline writes detections to `.txt` with two-decimal coordinates, then re-loads them before 3D merge.
- The in-memory branches were feeding full-precision YOLO coordinates into `generate_bb3d_inmemory(...)`, which is enough to move borderline merged boxes.

## Baseline Behavior

- Write JPG slices to disk.
- Run YOLO on folders.
- Save per-slice detection `.txt` files with `%d %.2f %.2f %.2f %.2f`.
- Reload those rounded coordinates during 3D bbox merge.

## New Behavior

- No-write detection branches still skip JPG / detection `.txt` artifacts.
- Before bbox merge, they now quantize coordinates to the same precision that the legacy `.txt` path would have produced.
- This keeps the speed benefit while restoring parity with the baseline merge inputs.

## Benchmark Setup

- Environment: `conda intelliscan`, `CUDA_VISIBLE_DEVICES=3`
- Samples:
  - `SN002`: `output_ramdisk_detect/SN002_*`
  - `SN009`: `output_ramdisk_detect/SN009_*`
- Summary artifact:
  - `intelliscan/output_ramdisk_detect/analysis/detection_parity_quantization_summary.json`

## Measured Results

### Final Kept Variant: `--inmemory` + Txt-Equivalent Quantization

| Sample | Baseline Total | Quantized `--inmemory` Total | Baseline Front-End | Quantized Front-End | Total Speedup | Front-End Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `SN002` | `52.51s` | `38.38s` | `37.57s` | `23.45s` | `26.91%` | `37.58%` |
| `SN009` | `76.17s` | `61.59s` | `39.16s` | `24.72s` | `19.14%` | `36.88%` |

### Intermediate Isolation Result: Pre-Quantization `--ramdisk-detection`

| Sample | Result |
| --- | --- |
| `SN009` | Exact already |
| `SN002` | Same small drift pattern as the old no-write path: voxel agreement `0.999996`, class-4 Dice `0.995715`, BLT max abs delta `1.4`, no defect flips |

## Accuracy / Quality Impact

### Quantized `--inmemory`

Validated exact on both `SN002` and `SN009`:

- `bb3d.npy`
- `segmentation.nii.gz`
- class Dice `1.0` for classes `0..4`
- `pad_misalignment_defect` flips: `0`
- `solder_extrusion_defect` flips: `0`
- BLT max abs delta: `0.0`

### Interpretation

- The remaining no-write drift was not caused primarily by “YOLO sees memory instead of files”.
- The critical mismatch was that the baseline merge path consumes rounded two-decimal detections, while the no-write branches originally consumed full-precision detections.
- Reproducing baseline serialization precision was enough to recover parity on both validated samples.

## Risks / Caveats

- Exactness is validated on `SN002` and `SN009`, not yet on the full raw-scan set.
- `--ramdisk-detection` is now mainly an isolation harness; it is slower than quantized `--inmemory` because it still pays JPG generation cost.
- The default production path is still unchanged in this pass.

## Conclusion

- Txt-equivalent detection quantization fixes the previously observed `SN002` drift in no-write detection paths.
- Quantized `--inmemory` is now the strongest current detection-stage optimization:
  - exact on the two validated samples
  - materially faster than the file-based front-end
  - simpler than the ramdisk harness
- Keep `--inmemory` and this quantization fix.
- Treat `--ramdisk-detection` as a debugging / isolation tool, not the preferred acceleration path.

## Next Follow-Up Actions

1. Validate quantized `--inmemory` on more raw-scan samples.
2. If parity holds, consider making quantized `--inmemory` the default detection path.
3. Use the ramdisk harness only if a future detection regression needs factor isolation.

## Evidence Paths

- `intelliscan/output_ramdisk_detect/analysis/detection_parity_quantization_summary.json`
- `intelliscan/output_ramdisk_detect/analysis/SN002_inmemory_quantized_compare.json`
- `intelliscan/output_ramdisk_detect/analysis/SN009_inmemory_quantized_compare.json`
- `intelliscan/output_ramdisk_detect/analysis/SN002_ramdisk_detect_compare.json`
- `intelliscan/output_ramdisk_detect/analysis/SN009_ramdisk_detect_compare.json`
- `intelliscan/output_ramdisk_detect/SN002_inmemory_quantized/metrics.json`
- `intelliscan/output_ramdisk_detect/SN009_inmemory_quantized/metrics.json`
