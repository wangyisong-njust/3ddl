# 055 In-Memory Path Investigation

Superseded in part by `optimization_history/056_true_inmemory_detection.md`, which documents the later true in-memory implementation and validation pass.

## Objective

Reduce detection-stage file I/O and avoid unnecessary merge-time parsing.

## Target Files

- `intelliscan/main.py`
- `intelliscan/detection.py`
- `intelliscan/merge.py`
- `intelliscan/utils.py`

## Code Change Summary

- `--inmemory` switches detection output from per-image `.txt` files to in-memory detection dictionaries consumed directly by the bbox merge path.

## Why This Change Was Made

- File-based detection generates many small files and requires a second read pass during bbox generation.

## Baseline Behavior

- Convert NIfTI to JPG
- Run detection and write `.txt` files
- Re-read those `.txt` files during merge

## Optimized Behavior

- Keep detection outputs in memory during merge

## How It Was Tested

- Validated from current SN009 baseline and `--inmemory` outputs
- Full-volume segmentation equality was recomputed in this pass

## Sample(s) Used

- `SN009`

## Key Timing Numbers

| Variant | Total | NII->JPG | Detection | BBox | Segmentation |
| --- | --- | --- | --- | --- | --- |
| Baseline | `93.23s` | `24.43s` | `14.85s` | `2.15s` | `33.79s` |
| `--inmemory` | `89.86s` | `3.51s` | `16.49s` | `1.88s` | `49.58s` |

## Accuracy / Quality Impact

- Baseline and `--inmemory` full-volume segmentation outputs are identical on SN009.

## Risks / Caveats

- The current path is not truly in-memory because JPG conversion still happens before detection.
- YOLO is still loaded once per view.
- End-to-end win is small because segmentation time increased in the current measured run.

## Conclusion

- The current `--inmemory` flag is a partial engineering step, not the final form of a no-I/O pipeline.

## Evidence Paths

- `intelliscan/output_formal/SN009_current_gpu/metrics.json`
- `intelliscan/output_formal/SN009_current_gpu_inmemory/metrics.json`
- `intelliscan/output_formal/SN009_current_gpu/segmentation.nii.gz`
- `intelliscan/output_formal/SN009_current_gpu_inmemory/segmentation.nii.gz`
