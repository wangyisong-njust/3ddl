# 096 Adaptive Margin Factor Ablation

## Objective

Determine why the current adaptive-margin experiment drifts from baseline by isolating three factors:

1. crop/context change
2. inference-path change
3. downstream metrology sensitivity

Default production behavior remains unchanged. All experiment controls are opt-in.

## Target Files and Controls Added

- `intelliscan/segmentation.py`
- `intelliscan/main.py`
- `intelliscan/scripts/compare_ablation_outputs.py`

Experiment controls added in code:

- `--adaptive-margin`: switch crop selection from fixed margin to adaptive margin
- `--force-sliding-window`: keep all crops on the sliding-window path
- `--force-direct-full-crop`: force single-pass full-crop PyTorch inference

`--force-direct-full-crop` is an ablation-only control. It is not the same as the production fast path, which uses symmetric padding to `roi_size` and then runs direct/batch inference.

## Experiment Matrix

| Variant | Margin rule | Inference policy | Purpose | Notes |
| --- | --- | --- | --- | --- |
| A | Fixed | Auto | Production baseline | Current default behavior |
| B | Fixed | `--force-direct-full-crop` | Path-only approximation | Closest grounded same-crop path-only variant when fixed-margin crop does not fit ROI |
| C | Adaptive | `--force-sliding-window` | Context-only ablation | Changes crop/context, keeps sliding-window |
| D | Adaptive | Auto | Full adaptive experiment | Reproduces the current adaptive-margin behavior |

## Samples Used

- `SN009`: `/home/kaixin/yj/wys/data/SN009_3D_May24/2_die_interposer_3Drecon_txm.nii`
- `SN002`: `/home/kaixin/yj/wys/data/SN002_3D_Feb24/4_die_stack_0.75um_3Drecon_txm.nii`

## Why Fast-Path Activation Fails on SN009

Measured from `intelliscan/output_ablation/SN009_A_fixed_auto/mmt/segmentation_stats.json`:

- raw-fit count: `113 / 114`
- fixed-margin-fit count: `0 / 114`
- adaptive-fit count: `113 / 114`

The fixed crop mean is `[106.46, 118.45, 105.19]`, while `roi_size=(112, 112, 80)`. The fixed-margin rule pushes almost every SN009 crop beyond the ROI on `y` and `z`, so baseline falls back to sliding-window for all `114` boxes.

SN002 behaves differently:

- raw-fit count: `95 / 95`
- fixed-margin-fit count: `95 / 95`
- adaptive-fit count: `95 / 95`

On SN002, adaptive margin does not change the crop at all. That makes SN002 a useful second sample for isolating path-only effects under identical crop/context.

## Benchmark Table

### SN009

| Variant | Total | Segmentation | Metrology | Direct path | Sliding path |
| --- | --- | --- | --- | --- | --- |
| A | `90.64s` | `33.36s` | `17.16s` | `0` | `114` |
| B | `81.46s` | `24.35s` | `17.03s` | `114` (`full`) | `0` |
| C | `73.74s` | `20.24s` | `12.34s` | `0` | `114` |
| D | `73.35s` | `20.77s` | `13.12s` | `113` (`batch`) | `1` |

### SN002

| Variant | Total | Segmentation | Metrology | Direct path | Sliding path |
| --- | --- | --- | --- | --- | --- |
| A | `55.55s` | `13.39s` | `4.62s` | `95` (`batch`) | `0` |
| B | `56.47s` | `12.48s` | `4.67s` | `95` (`full`) | `0` |
| C | `57.92s` | `13.49s` | `4.79s` | `0` | `95` |
| D | `57.37s` | `13.53s` | `4.76s` | `95` (`batch`) | `0` |

## Segmentation Drift Table

### SN009

| Compare | Voxel agreement | Dice c1 | Dice c2 | Dice c3 | Dice c4 |
| --- | --- | --- | --- | --- | --- |
| A vs B | `0.993243` | `0.914608` | `0.901531` | `0.694916` | `0.703799` |
| A vs C | `0.994806` | `0.926523` | `0.913540` | `0.661772` | `0.809370` |
| A vs D | `0.994806` | `0.926526` | `0.913531` | `0.661819` | `0.809364` |
| C vs D | `0.999997` | `0.999956` | `0.999926` | `0.999851` | `0.999871` |

### SN002

| Compare | Voxel agreement | Dice c1 | Dice c2 | Dice c3 | Dice c4 |
| --- | --- | --- | --- | --- | --- |
| A vs B | `0.996983` | `0.671565` | `0.755167` | `0.775055` | `0.205143` |
| A vs C | `0.999999` | `0.999867` | `0.999877` | `0.999723` | `0.999578` |
| A vs D | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `1.000000` |
| C vs D | `0.999999` | `0.999867` | `0.999877` | `0.999723` | `0.999578` |

## Metrology Drift Table

### SN009

| Compare | Pad flips | Solder flips | BLT max abs delta | BLT mean abs delta |
| --- | --- | --- | --- | --- |
| A vs B | `48 / 114` | `5 / 114` | `10.5` | `4.22` |
| A vs C | `13 / 114` | `5 / 114` | `7.0` | `2.89` |
| A vs D | `13 / 114` | `5 / 114` | `7.0` | `2.89` |
| C vs D | `0 / 114` | `0 / 114` | `0.0` | `0.0` |

### SN002

| Compare | Pad flips | Solder flips | BLT max abs delta | BLT mean abs delta |
| --- | --- | --- | --- | --- |
| A vs B | `0 / 95` | `46 / 95` | `35.0` | `10.67` |
| A vs C | `0 / 95` | `0 / 95` | `0.0` | `0.0` |
| A vs D | `0 / 95` | `0 / 95` | `0.0` | `0.0` |
| C vs D | `0 / 95` | `0 / 95` | `0.0` | `0.0` |

## Interpretation

- On `SN009`, `A vs C` and `A vs D` are effectively the same. The full adaptive drift is already present before the fast path is enabled.
- On `SN009`, `C vs D` is nearly identical at both segmentation and metrology level. After the adaptive crop is chosen, switching from sliding-window to the actual direct/batch fast path adds negligible extra drift.
- On `SN002`, adaptive margin does not change the crop at all. `A vs C` therefore isolates direct-padded vs sliding-window under the same crop and shows near-identity with zero defect flips.
- `B` is materially different on both samples. Forcing direct full-crop inference causes much larger drift than the production direct-padded path and should remain an ablation tool only.

## Dominant Cause Analysis

Measured in this pass, the dominant cause of the observed adaptive-margin drift is **context shrinkage**, not the direct-padded inference path.

Evidence:

- `SN009 A vs C` already reproduces the class-3/class-4 and metrology drift seen in `A vs D`.
- `SN009 C vs D` shows near-perfect agreement and zero metrology flips.
- `SN002 A vs C` shows that direct-padded vs sliding-window under the same crop is almost identical and causes no business-level drift.

The main limitation is that `SN009` cannot support an exact fixed-margin `direct_padded` path-only comparison because the fixed-margin crop does not fit the ROI. `SN009 B` uses `--force-direct-full-crop` as the closest grounded approximation and shows that direct-full is unsafe, but it should not be confused with the production adaptive fast path.

## Recommendation

The best next step is a **guarded adaptive-margin policy**, not generic direct-path tuning and not `force_direct_full_crop`.

The guardrail should be driven by measured context loss:

- keep adaptive margin experimental
- focus on limiting per-dimension context shrinkage, especially `z`
- allow fast-path activation only when the crop can be reduced with small, controlled context loss
- leave larger context-loss cases on sliding-window

## Caveats

- This pass covers two samples, not a full benchmark suite.
- `SN009 B` is a grounded approximation, not an exact direct-padded path-only comparison.
- Small voxel differences can still matter downstream; class-3/class-4 Dice and metrology remain the acceptance criteria.

## Evidence Paths

- `intelliscan/segmentation.py`
- `intelliscan/main.py`
- `intelliscan/scripts/compare_ablation_outputs.py`
- `intelliscan/output_ablation/SN009_A_fixed_auto/metrics.json`
- `intelliscan/output_ablation/SN009_A_fixed_auto/mmt/segmentation_stats.json`
- `intelliscan/output_ablation/SN009_B_fixed_direct_full/metrics.json`
- `intelliscan/output_ablation/SN009_B_fixed_direct_full/mmt/segmentation_stats.json`
- `intelliscan/output_ablation/SN009_C_adaptive_sliding/metrics.json`
- `intelliscan/output_ablation/SN009_C_adaptive_sliding/mmt/segmentation_stats.json`
- `intelliscan/output_ablation/SN009_D_adaptive_auto/metrics.json`
- `intelliscan/output_ablation/SN009_D_adaptive_auto/mmt/segmentation_stats.json`
- `intelliscan/output_ablation/analysis/SN009_reference_vs_variants.json`
- `intelliscan/output_ablation/analysis/SN009_context_vs_context_plus_path.json`
- `intelliscan/output_ablation/SN002_A_fixed_auto/metrics.json`
- `intelliscan/output_ablation/SN002_A_fixed_auto/mmt/segmentation_stats.json`
- `intelliscan/output_ablation/SN002_B_fixed_direct_full/metrics.json`
- `intelliscan/output_ablation/SN002_B_fixed_direct_full/mmt/segmentation_stats.json`
- `intelliscan/output_ablation/SN002_C_adaptive_sliding/metrics.json`
- `intelliscan/output_ablation/SN002_C_adaptive_sliding/mmt/segmentation_stats.json`
- `intelliscan/output_ablation/SN002_D_adaptive_auto/metrics.json`
- `intelliscan/output_ablation/SN002_D_adaptive_auto/mmt/segmentation_stats.json`
- `intelliscan/output_ablation/analysis/SN002_reference_vs_variants.json`
- `intelliscan/output_ablation/analysis/SN002_context_vs_context_plus_path.json`
