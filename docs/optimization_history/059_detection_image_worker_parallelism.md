# Detection Image Worker Parallelism

## Objective

Reduce the remaining front-end cost of the default quantized in-memory detection path without changing detector inputs, bbox merge logic, segmentation geometry, or metrology behavior.

This follows `058_quantized_inmemory_multisample_validation.md`. After txt-compatible bbox quantization made the no-write path exact, the next bottleneck inside `2D Detection Inference` was CPU-side slice preparation:

- float slice normalization
- PIL rotate / flip
- in-memory JPEG encode
- OpenCV decode

## Target Files

- `intelliscan/detection.py`
- `intelliscan/main.py`
- `intelliscan/README.md`

## Code Change Summary

- Added `image_workers` support to `run_yolo_detection_inmemory_from_volume(...)`.
- Parallelized only the per-slice CPU image preparation step with `ThreadPoolExecutor`.
- Preserved slice ordering and the exact JPEG-roundtrip image transform already used by the validated quantized in-memory path.
- Added CLI control: `--detection-image-workers N`.
- Promoted the validated setting to the new default: `detection_image_workers=4`.

The implementation does not:

- change YOLO weights
- change bbox quantization
- change 3D merge
- change segmentation routing
- change report or metrology logic

## Candidate Search

The first sweep was kept small and controlled on `SN009`:

| Candidate | Total | Detection | Result |
| --- | ---: | ---: | --- |
| `workers=1` | `61.47s` | `22.89s` | reference |
| `workers=4` | `54.89s` | `15.30s` | best |
| `workers=8` | `59.06s` | `20.63s` | slower than `4` |

Evidence:

- `intelliscan/output_prefetch/analysis/SN009_detection_image_workers_search.json`

## Validation Samples

- `SN002`
- `SN003`
- `SN008`
- `SN009`
- `SN010`
- `SN011`
- `SN012`
- `SN061`

## Measured Results

Multi-sample summary:

- exact key-output parity on `8 / 8` samples for:
  - `bb3d.npy`
  - `segmentation.nii.gz`
  - `metrology/metrology.csv`
- mean total speedup vs previous quantized in-memory default: `17.84%`
- pooled total speedup: `17.14%`
- mean detection-phase speedup: `36.00%`
- pooled detection-phase speedup: `36.07%`

Per-sample total speedup range:

- min: `10.70%` (`SN009`)
- max: `23.09%` (`SN002`)

Evidence:

- `intelliscan/output_prefetch/analysis/detection_image_workers4_multisample_summary.json`

## Quality Validation

For `SN002`, `SN009`, and `SN012`, explicit full-output comparisons were rerun and were exact.

For the full eight-sample set, the kept production artifacts are file-identical:

- `bb3d.npy`
- `segmentation.nii.gz`
- `metrology.csv`

This is sufficient for the current pipeline because segmentation and metrology are downstream of those persisted outputs.

## Cumulative Gain

Compared with the older historical file-based baseline still stored in the workspace:

- `SN002`: `54.82s -> 29.36s` (`46.43%` cumulative total reduction)
- `SN009`: `86.42s -> 54.89s` (`36.48%` cumulative total reduction)

Evidence:

- `intelliscan/output_prefetch/analysis/cumulative_engineering_gain_workers4_summary.json`

## Risks / Caveats

- This optimization only affects the in-memory detection path.
- It is a CPU-side throughput optimization, not a model or geometry change.
- The `workers=1/4/8` search was only explicitly benchmarked on `SN009`; `workers=4` was then validated across the local eight-sample set.
- The current evidence is strong enough for the local raw-scan set, but future production-only scans should still be checked against the fallback `--file-detection` path when first introduced.

## Conclusion

`detection_image_workers=4` is a safe kept optimization under the current validated workspace conditions.

- It preserves the already-fixed quantized in-memory detection parity.
- It materially reduces the dominant front-end stage.
- It is now the default setting for the quantized in-memory detection path.

## Next Follow-Up

The next throughput target should move beyond single-stage CPU preparation and into pipeline overlap:

- view-level overlap
- detection / downstream stage overlap across multiple files
- broader validation on newly added raw scans
