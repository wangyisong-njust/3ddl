# Detection Batch Prefetch Overlap

## Objective

Reduce the remaining `2D Detection Inference` cost of the validated quantized in-memory + `4`-worker detection path by overlapping CPU batch preparation with current YOLO inference.

This follows `059_detection_image_worker_parallelism.md`. After parallelizing per-slice image preparation, the next remaining inefficiency was strictly serial batch handling:

- prepare batch `N`
- run YOLO on batch `N`
- prepare batch `N+1`
- run YOLO on batch `N+1`

The goal here was to hide part of the CPU preparation time behind the GPU detection call without changing detector inputs or downstream geometry.

## Target Files

- `intelliscan/detection.py`
- `intelliscan/main.py`
- `intelliscan/README.md`

## Code Change Summary

- Added `batch_prefetch` support to `run_yolo_detection_inmemory_from_volume(...)`.
- Added `_prepare_detection_batch(...)` so batch preparation can be queued explicitly.
- Added a one-thread prefetch executor that prepares batch `N+1` while batch `N` is inside `model.predict(...)`.
- Added CLI controls:
  - `--detection-batch-prefetch`
  - `--no-detection-batch-prefetch`
- Promoted the validated setting to the new default: `detection_batch_prefetch=True`.

The kept implementation does not:

- change YOLO weights
- change slice ordering
- change JPEG-roundtrip preprocessing
- change legacy txt-equivalent bbox quantization
- change 3D bbox merge
- change segmentation, metrology, or report logic

## Benchmark Setup

Environment:

- GPU: `CUDA_VISIBLE_DEVICES=3`
- detection path: quantized in-memory
- image workers: `4`

Validation samples:

- `SN002`
- `SN003`
- `SN008`
- `SN009`
- `SN010`
- `SN011`
- `SN012`
- `SN061`

Reference:

- explicit `workers=4` quantized in-memory path under `intelliscan/output_prefetch/`

Candidate:

- same path plus batch prefetch under `intelliscan/output_prefetch2/`

## Measured Results

Multi-sample summary versus the previous `workers=4` default:

- exact key-output parity on `8 / 8` samples for:
  - `bb3d.npy`
  - `segmentation.nii.gz`
  - `metrology/metrology.csv`
- mean total speedup: `14.20%`
- pooled total speedup: `13.67%`
- mean detection-stage speedup: `34.64%`
- pooled detection-stage speedup: `34.73%`

Representative samples:

| Sample | Workers4 Total | Prefetch Total | Workers4 Detection | Prefetch Detection |
| --- | ---: | ---: | ---: | ---: |
| `SN002` | `29.36s` | `24.97s` | `14.06s` | `9.53s` |
| `SN009` | `54.89s` | `48.15s` | `15.30s` | `9.71s` |
| `SN012` | `52.18s` | `46.98s` | `15.38s` | `10.40s` |

Evidence:

- `intelliscan/output_prefetch2/analysis/detection_batch_prefetch_multisample_summary.json`
- `intelliscan/output_prefetch2/analysis/detection_batch_prefetch_output_parity.json`

## Quality Validation

Explicit full-output comparisons were rerun for:

- `SN002`
- `SN009`
- `SN012`

All were exact at the segmentation and metrology levels.

For the full eight-sample set, the kept persisted outputs are file-identical:

- `bb3d.npy`
- `segmentation.nii.gz`
- `metrology.csv`

## Cumulative Gain

Compared with the earlier file-based detection reference stored under `intelliscan/output_inmemory_validate/*_file_default/`:

- mean total reduction on the eight-sample set: `45.13%`
- pooled total reduction: `44.20%`
- mean detection-stage reduction: `26.59%`
- pooled detection-stage reduction: `26.61%`

Evidence:

- `intelliscan/output_prefetch2/analysis/cumulative_engineering_gain_prefetch4_summary.json`

## Risks / Caveats

- This change only affects the in-memory detection path.
- The new overlap is intentionally narrow: it overlaps CPU batch preparation with detection inference, not cross-view execution or downstream stages.
- Validation is strong on the current local eight-sample raw-scan set, but future production-only cases should still be spot-checked against `--file-detection` when first introduced.

## Conclusion

Batch prefetch overlap is a safe kept optimization under the current validated workspace conditions.

- It preserves the already-fixed quantized in-memory detection parity.
- It further reduces the front-end bottleneck after the `workers=4` optimization.
- It is now the default behavior for the quantized in-memory detection path.

## Next Follow-Up

The next throughput step should move beyond intra-view batch overlap:

- view-level overlap
- multi-file stage overlap
- continued validation on any newly added raw scans
