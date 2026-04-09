# 034 Larger Direct ROI Experiment

## Objective

Test whether a slightly larger direct-path padded ROI can open more fast-path crops without trimming any true crop context.

## Target Files

- `intelliscan/main.py`
- `intelliscan/segmentation.py`
- `intelliscan/scripts/benchmark_seg_met_stage.py`

## Code Change Summary

- Added `direct_roi_size` to the segmentation config and pipeline config.
- Added CLI control: `--direct-roi-size X Y Z`.
- Kept sliding-window ROI fixed at `112 x 112 x 80`.
- Made the direct-padded path use `direct_roi_size` only for symmetric zero padding.
- Added `scripts/benchmark_seg_met_stage.py` to benchmark the combined segmentation + metrology stage from a cached `bb3d.npy`.
- Extended `segmentation_stats.json` to record both `roi_size` and `direct_roi_size`.

## Why This Change Was Made

- The adaptive-margin ablation showed that trimming true context is the main reason outputs drift.
- A larger direct ROI looked like a safer alternative:
  - keep the same fixed-margin crop
  - do not clip any real crop voxels
  - only enlarge the padded shell for the direct path

## Baseline Behavior

- Fixed-margin crops larger than `112 x 112 x 80` fall back to sliding-window.
- Direct-padded crops are always padded to `112 x 112 x 80`.

## Experimental Behavior

- Fixed-margin crop geometry stays unchanged.
- Sliding-window still uses `112 x 112 x 80`.
- Direct-padded crops can be padded to a larger shell such as `118 x 118 x 80` or `124 x 124 x 80`.
- Default production behavior is unchanged unless `--direct-roi-size` is set.

## Benchmark Setup

- Environment:
  - `CUDA_VISIBLE_DEVICES=3`
  - current combined default pipeline
- Full pipeline runs:
  - `intelliscan/output_directroi/SN009_roi112_ref2/`
  - `intelliscan/output_directroi/SN009_roi118_xy/`
  - `intelliscan/output_directroi/SN009_roi124_xy/`
  - `intelliscan/output_directroi/SN010_roi112_ref2/`
  - `intelliscan/output_directroi/SN010_roi118_xy/`
  - `intelliscan/output_directroi/SN010_roi124_xy/`
  - `intelliscan/output_directroi/SN002_roi124_xy/`
- Repeated stage-only benchmark:
  - `intelliscan/scripts/benchmark_seg_met_stage.py`
  - `intelliscan/output_directroi/analysis_SN010_stage_roi112.json`
  - `intelliscan/output_directroi/analysis_SN010_stage_roi124.json`

## Measured Results

### Full Pipeline Summary

| Sample | Direct ROI | Direct / Sliding | `3D Seg + Met` | Quality Summary |
| --- | --- | --- | ---: | --- |
| `SN009` | `112x112x80` | `0 / 114` | `36.24s` | reference |
| `SN009` | `118x118x80` | `2 / 114` | `35.37s` | no defect flips; very small drift |
| `SN009` | `124x124x80` | `2 / 114` | `35.99s` | no defect flips; small drift |
| `SN010` | `112x112x80` | `91 / 19` | `16.38s` | reference |
| `SN010` | `118x118x80` | `91 / 19` | `16.26s` | strong drift despite unchanged direct/sliding counts |
| `SN010` | `124x124x80` | `94 / 16` | `16.39s` | no meaningful speedup; drift worsens |
| `SN002` | `124x124x80` | `95 / 0` | `14.72s` | all-direct control still drifts heavily |

### Drift Summary Versus Reference

| Sample | Variant | Voxel | Dice c3 | Dice c4 | Pad flips | Solder flips | BLT max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `SN009` | `roi118` | `0.999963` | `0.999914` | `0.998089` | `0 / 114` | `0 / 114` | `0.0` |
| `SN009` | `roi124` | `0.999949` | `0.999862` | `0.997388` | `0 / 114` | `0 / 114` | `2.8` |
| `SN010` | `roi118` | `0.998862` | `0.906228` | `0.861601` | `5 / 110` | `21 / 110` | `25.2` |
| `SN010` | `roi124` | `0.998470` | `0.861556` | `0.836034` | `6 / 110` | `35 / 110` | `28.7` |
| `SN002` | `roi124` | `0.998698` | `0.775452` | `0.386047` | `0 / 95` | `36 / 95` | `21.7` |

### Repeated Stage-Only Benchmark on `SN010`

| Variant | Mean | Std | Direct / Sliding |
| --- | ---: | ---: | --- |
| `roi112` | `8.0760s` | `0.2082s` | `91 / 19` |
| `roi124` | `8.3869s` | `0.2616s` | `94 / 16` |

Interpretation:

- On the sample where larger direct ROI was most likely to help (`SN010`), repeated stage-only timing was `3.85%` slower, not faster.
- On `SN009`, enlarging only `x/y` did not materially open the fast path, because the real bottleneck remained elsewhere.

## Accuracy / Quality Impact

- `SN009` shows that a larger direct ROI can be almost harmless when it barely changes routing.
- `SN010 roi118` is the key negative result:
  - direct/sliding counts stay exactly `91 / 19`
  - but class Dice and metrology drift materially
- `SN002 roi124` is the control that confirms the issue:
  - every bbox was already direct under baseline
  - only the padded shell changed
  - class-4 Dice drops to `0.386047`
  - `36 / 95` solder-defect flags flip

This means crop preservation alone is not enough to preserve outputs.

## Why The Drift Happens

- The real crop is unchanged, but the direct path pads it to a larger all-zero canvas.
- The model was trained and validated with the original direct-path canvas size.
- Increasing that canvas changes the spatial statistics seen by the network, especially through convolution and normalization layers inside the 3D UNet.
- `SN010 roi118` proves this directly:
  - same crop geometry
  - same direct/sliding routing
  - only larger padded shell
  - still substantial output drift

## Risks / Caveats

- These results are grounded in the current PyTorch direct path only; TensorRT was intentionally excluded.
- `SN002` reference comes from the current validated combined baseline under `output_gltfless/`, not from a fresh rerun in this pass.
- PyVista is still not installed in this workspace, so GLTF creation remains unvalidated end to end.

## Conclusion

- Do not treat larger direct ROI as a safe runtime optimization under the current model and direct-padded implementation.
- The idea is not only low gain; it is actively unsafe on validated samples.
- The strongest negative finding is:
  - preserving crop geometry is insufficient
  - the padded direct-path canvas is itself a parity-sensitive variable

## Recommendation

Keep `--direct-roi-size` only as an explicit experiment flag. Do not pursue this as a default optimization unless one of these changes happens first:

1. redesign direct-path numerics so output is insensitive to the padded canvas size, or
2. retrain / recalibrate the segmentation model for the larger direct canvas.

## Evidence Paths

- `intelliscan/output_directroi/analysis/direct_roi_experiment_summary.json`
- `intelliscan/output_directroi/analysis_SN009_directroi_compare.json`
- `intelliscan/output_directroi/analysis_SN010_directroi_compare.json`
- `intelliscan/output_directroi/analysis_SN002_directroi_compare.json`
- `intelliscan/output_directroi/analysis_SN010_stage_roi112.json`
- `intelliscan/output_directroi/analysis_SN010_stage_roi124.json`
- `intelliscan/output_directroi/SN009_roi112_ref2/metrics.json`
- `intelliscan/output_directroi/SN009_roi118_xy/metrics.json`
- `intelliscan/output_directroi/SN009_roi124_xy/metrics.json`
- `intelliscan/output_directroi/SN010_roi112_ref2/metrics.json`
- `intelliscan/output_directroi/SN010_roi118_xy/metrics.json`
- `intelliscan/output_directroi/SN010_roi124_xy/metrics.json`
- `intelliscan/output_directroi/SN002_roi124_xy/metrics.json`
