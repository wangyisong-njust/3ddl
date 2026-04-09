# 056 True In-Memory Detection

## Update

This note captures the first checked-in no-write detection path before txt-equivalent bbox quantization was added.
For the current parity-fixed state, see `optimization_history/057_detection_parity_quantization.md`.

## Objective

Remove detection-stage JPG and `.txt` churn, and reuse one YOLO instance across both views without changing the default production path.

## Target Files

- `intelliscan/detection.py`
- `intelliscan/main.py`

## Code Change Summary

- `--inmemory` now slices the volume directly in memory instead of calling `nii2jpg(...)`.
- A single YOLO model is loaded once and reused for both detection views.
- Detection outputs stay in memory and feed `generate_bb3d_inmemory(...)` directly.
- The checked-in in-memory path now mirrors file-based detector inputs more closely:
  - PIL transform matching `save_image(...)`
  - in-memory JPEG encode
  - OpenCV decode to BGR ndarray before YOLO

## Why This Change Was Made

- The previous `--inmemory` path was only partial. It still wrote JPGs first and still loaded YOLO once per view.
- Detection remained the main remaining I/O-heavy stage after combined segmentation + metrology became the default.

## Baseline Behavior

- Convert the full NIfTI volume to JPG slices for both views.
- Run YOLO on folders from disk.
- Save per-slice detection `.txt` files.
- Re-read detections during 3D bbox merge.

## New Behavior

- Generate per-slice detector inputs in memory.
- Reuse one YOLO model for both views.
- Keep detections in memory until bbox merge.
- Do not write JPG slices or detection `.txt` files when `--inmemory` is enabled.

## Benchmark Setup

- Environment: `conda intelliscan`, `CUDA_VISIBLE_DEVICES=3`
- Samples:
  - `SN009`: `output_inmemory_true/SN009_*`
  - `SN002`: `output_inmemory_true/SN002_*`
- Summary artifact: `intelliscan/output_inmemory_true/analysis/true_inmemory_detection_summary.json`

## Measured Results

### Final Checked-In Variant: OpenCV-Decoded In-Memory JPEG

| Sample | Baseline Total | In-Memory Total | Baseline Front-End | In-Memory Front-End | Total Speedup | Front-End Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `SN009` | `86.42s` | `73.36s` | `38.14s` | `24.73s` | `15.11%` | `35.16%` |
| `SN002` | `54.82s` | `42.23s` | `36.37s` | `23.77s` | `22.97%` | `34.64%` |

### Intermediate Attempts In This Pass

| Variant | Sample | Result |
| --- | --- | --- |
| Direct in-memory PIL / uint8 feed | `SN009` | Unsafe: bbox count changed from `114` to `112` |
| In-memory JPEG + PIL decode | `SN009` | Lossless on this sample |
| In-memory JPEG + PIL decode | `SN002` | Still non-identical |

## Accuracy / Quality Impact

### `SN009`

- Final checked-in in-memory path is byte-identical to baseline for:
  - `bb3d.npy`
  - `segmentation.nii.gz`
  - `metrology/metrology.csv`

### `SN002`

- Final checked-in in-memory path is not byte-identical to baseline.
- Measured drift remains small:
  - voxel agreement: `0.999996`
  - class Dice:
    - class 1: `0.999498`
    - class 2: `0.999630`
    - class 3: `0.999032`
    - class 4: `0.995715`
  - defect flips:
    - `pad_misalignment_defect`: `0 / 95`
    - `solder_extrusion_defect`: `0 / 95`
  - BLT delta:
    - max abs delta: `1.4`
    - mean abs delta: `0.0147`

## Risks / Caveats

- Current `--inmemory` is faster, but not yet proven lossless across tested samples.
- Matching the JPEG transform alone was not enough; `SN002` still drifted after both PIL and OpenCV decode variants.
- The remaining gap is likely tied to Ultralytics internals for `source=<folder>` versus `source=<in-memory images>`, but that has not been isolated in this pass.

## Conclusion

- True in-memory detection and single-YOLO reuse are now implemented.
- The checked-in OpenCV-decoded variant is materially faster than the file-based front-end.
- It is safe on `SN009`, but not yet default-safe because `SN002` still shows small non-zero output drift.
- Keep `--inmemory` as an experimental flag, not a production default.

## Next Follow-Up Actions

1. Decide whether exact detector parity is required; if yes, isolate Ultralytics source-path differences.
2. If exact parity is not required, validate defect/metrology stability on more raw-scan samples.
3. If further no-write gains are needed, remove report dependence on per-bbox `mmt/img` and `mmt/pred`.

## Evidence Paths

- `intelliscan/output_inmemory_true/analysis/true_inmemory_detection_summary.json`
- `intelliscan/output_inmemory_true/SN009_baseline_file/timing.log`
- `intelliscan/output_inmemory_true/SN009_inmemory_true_cv2/timing.log`
- `intelliscan/output_inmemory_true/SN002_baseline_file/timing.log`
- `intelliscan/output_inmemory_true/SN002_inmemory_true_cv2/timing.log`
