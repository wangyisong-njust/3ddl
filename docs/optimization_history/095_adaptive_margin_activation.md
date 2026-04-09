# 095 Adaptive Margin Activation

## Objective

Activate the existing batch/direct segmentation path on real formal samples by shrinking bbox context per dimension when fixed `margin=15` would otherwise push the crop beyond `roi_size=(112, 112, 80)`.

## Target Files

- `intelliscan/segmentation.py`
- `intelliscan/main.py`

## Code Change Summary

- Added `expand_bbox_fixed()` to preserve the old fixed-margin rule for comparison.
- Added adaptive `expand_bbox(..., roi_size)` plus `_expand_interval_adaptive()`:
  - if raw bbox size already exceeds the ROI on one axis, do not add context on that axis
  - otherwise, allow at most `2 * margin` voxels of total context on that axis
  - cap the final size so that it never exceeds the ROI on that axis
  - clamp by volume boundaries and reuse leftover context budget on the opposite side when one side hits a boundary
- Added lightweight traceability in `segment_bboxes()`:
  - raw-fit count
  - fixed-margin-fit count
  - adaptive-fit count
  - direct vs sliding counts
  - crop-size summaries
- Save run stats to `mmt/segmentation_stats.json`.
- Keep the feature behind `--adaptive-margin`; final code does not enable it by default.

## Why This Change Was Made

SN009 showed that the direct path existed in code but was not reachable in practice. Computed in this pass from `intelliscan/output_formal/SN009_adaptive_margin_baseline/bb3d.npy`:

- raw bbox fit count: `113 / 114`
- fixed-margin fit count: `0 / 114`

The fixed-margin crop mean was `[106.46, 118.45, 105.19]`, which exceeds the ROI on `y` and `z`, so nearly every crop was forced into sliding-window inference.

## Baseline Behavior

- Expand every bbox by fixed `margin=15` on all axes.
- Classify fast-path eligibility after expansion.
- Fresh SN009 baseline run in this pass used the old behavior:
  - output: `intelliscan/output_formal/SN009_adaptive_margin_baseline/`
  - direct path count: `0`
  - sliding-window count: `114`

The direct/sliding count above was recomputed in this pass from the saved bbox file and the fixed-margin rule because `segmentation_stats.json` was introduced by this change.

## Optimized Behavior

- With `--adaptive-margin`, `113 / 114` crops fit the ROI and use the direct/batch path.
- Only one bbox still falls back to sliding-window. Its raw size is `[84, 88, 82]`, so it still exceeds the ROI on `z`.

## How It Was Tested

- Fresh GPU baseline run on `SN009`:
  - `intelliscan/output_formal/SN009_adaptive_margin_baseline/`
- Fresh GPU adaptive run on the same sample in the same environment:
  - `intelliscan/output_formal/SN009_adaptive_margin_enabled/`
- Compared:
  - `metrics.json`
  - `timing.log`
  - `bb3d.npy`
  - `segmentation.nii.gz`
  - `metrology/metrology.csv`
  - `mmt/segmentation_stats.json`

## Sample(s) Used

- `SN009` from `/home/kaixin/yj/wys/data/SN009_3D_May24/2_die_interposer_3Drecon_txm.nii`

`SN002` was not rerun in this pass.

## Key Timing Numbers

| Variant | Total | Segmentation | Metrology | Direct | Sliding |
| --- | --- | --- | --- | --- | --- |
| Baseline | `91.78s` | `33.37s` | `16.89s` | `0` | `114` |
| Adaptive margin | `75.44s` | `20.29s` | `12.79s` | `113` | `1` |

Measured deltas on SN009:

- total pipeline: `-16.35s` (`17.81%` faster)
- segmentation stage: `-13.08s` (`39.20%` faster)
- metrology stage: `-4.10s` (`24.30%` faster)

## Accuracy / Quality Impact

The output is **not** identical to baseline.

- full-volume voxel agreement: `0.9948`
- class Dice:
  - class `0`: `0.9985`
  - class `1`: `0.9265`
  - class `2`: `0.9135`
  - class `3`: `0.6618`
  - class `4`: `0.8094`

Metrology also changed:

- `void_ratio_defect`: `0 -> 0`
- `solder_extrusion_defect`: `3 -> 4`, with `5` row-level flips
- `pad_misalignment_defect`: `70 -> 67`, with `13` row-level flips
- `BLT` max absolute delta: `7.0`
- `BLT` mean absolute delta: `2.89`

## Risks / Caveats

- The speedup is real, but this is not a lossless optimization on SN009.
- Background dominance makes voxel agreement look better than the defect-sensitive class Dice.
- Many adaptive crops collapse to exactly `112 x 112 x 80`, which reduces context relative to the baseline fixed-margin crops.
- Only one formal sample was rerun in this pass.

## Conclusion

Adaptive margin successfully activates the fast path and materially improves SN009 latency, but it currently changes segmentation and metrology outputs enough that it should not be treated as production-safe or lossless.

## Recommendation

- Keep the implementation and instrumentation for continued experiments.
- Keep the feature opt-in behind `--adaptive-margin`.
- Do not switch the default pipeline to adaptive margin until a quality-preserving rule is found and validated on more than one sample.

## Evidence Paths

- `intelliscan/segmentation.py`
- `intelliscan/main.py`
- `intelliscan/output_formal/SN009_adaptive_margin_baseline/metrics.json`
- `intelliscan/output_formal/SN009_adaptive_margin_baseline/timing.log`
- `intelliscan/output_formal/SN009_adaptive_margin_baseline/bb3d.npy`
- `intelliscan/output_formal/SN009_adaptive_margin_baseline/segmentation.nii.gz`
- `intelliscan/output_formal/SN009_adaptive_margin_baseline/metrology/metrology.csv`
- `intelliscan/output_formal/SN009_adaptive_margin_enabled/metrics.json`
- `intelliscan/output_formal/SN009_adaptive_margin_enabled/timing.log`
- `intelliscan/output_formal/SN009_adaptive_margin_enabled/bb3d.npy`
- `intelliscan/output_formal/SN009_adaptive_margin_enabled/segmentation.nii.gz`
- `intelliscan/output_formal/SN009_adaptive_margin_enabled/metrology/metrology.csv`
- `intelliscan/output_formal/SN009_adaptive_margin_enabled/mmt/segmentation_stats.json`
