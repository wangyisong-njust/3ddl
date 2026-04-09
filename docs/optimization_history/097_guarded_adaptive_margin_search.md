# 097 Guarded Adaptive Margin Search

## Objective

Follow `096_adaptive_margin_factor_ablation.md` with a constrained policy search:

- keep the production default unchanged
- keep full adaptive available for comparison
- recover part of the fast-path gain only when context shrinkage is small enough to be plausibly safe

The previous pass showed that adaptive-margin drift is dominated by context shrinkage, especially `z` clipping on `SN009`. This pass tests whether a conservative guardrail can recover useful speed while avoiding most of that drift.

## Code Controls Added

- `intelliscan/main.py`
- `intelliscan/segmentation.py`

Explicit experiment controls:

- `--guarded-adaptive-margin`
- `--guard-max-loss-xy`
- `--guard-max-loss-z`

Guarded policy implemented in code:

1. build the baseline fixed-margin crop
2. build the adaptive crop that would fit `roi_size`
3. if the raw bbox itself does not fit the ROI, keep the fixed crop and use sliding-window
4. only examine dimensions where the fixed-margin crop overflows the ROI
5. compute per-dimension context loss as `fixed_crop_size - adaptive_crop_size`
6. accept shrinkage only when each overflowing `x/y` loss is `<= guard_max_loss_xy` and each overflowing `z` loss is `<= guard_max_loss_z`
7. if accepted, only the overflowing dimensions are replaced with the adaptive bounds; other dimensions keep the fixed crop
8. otherwise fall back to the original fixed-margin crop and baseline sliding-window behavior

This keeps the guardrail local and conservative. It does not force the fast path at all costs.

## Candidate Guardrails

The candidate sweep was grounded in the `SN009` context-loss measurements from `096`:

- full adaptive mean total context loss vs fixed margin: about `[1.18, 6.45, 25.56]` voxels on `[x, y, z]`
- `z` was the dominant clipped dimension

Candidates tested:

| ID | `guard_max_loss_xy` | `guard_max_loss_z` | Intent |
| --- | --- | --- | --- |
| `G1` | `4` | `8` | Very conservative |
| `G2` | `6` | `12` | Conservative |
| `G3` | `8` | `16` | Conservative-moderate |
| `G4` | `10` | `20` | Moderate |
| `G5` | `12` | `24` | Loosest tested guardrail |

## Samples Used

- `SN009`: `/home/kaixin/yj/wys/data/SN009_3D_May24/2_die_interposer_3Drecon_txm.nii`
- `SN002`: `/home/kaixin/yj/wys/data/SN002_3D_Feb24/4_die_stack_0.75um_3Drecon_txm.nii`

## Benchmark Table

Primary timing metric for this optimization is segmentation time. Total pipeline time is reported for completeness, but it also includes unchanged stages such as NIfTI-to-JPG conversion and 2D detection.

### SN009

| Variant | Total | Segmentation | Direct path | Sliding path | Guard accepted |
| --- | --- | --- | --- | --- | --- |
| `A` fixed+auto | `99.43s` | `37.71s` | `0` | `114` | n/a |
| `D` adaptive+auto | `95.50s` | `22.29s` | `113` | `1` | n/a |
| `G1` | `95.58s` | `37.62s` | `1` | `113` | `1` |
| `G2` | `95.30s` | `37.68s` | `2` | `112` | `2` |
| `G3` | `95.86s` | `36.81s` | `2` | `112` | `2` |
| `G4` | `121.36s` | `41.93s` | `2` | `112` | `2` |
| `G5` | `87.23s` | `31.13s` | `28` | `86` | `28` |

### SN002

| Variant | Total | Segmentation | Direct path | Sliding path | Guard accepted |
| --- | --- | --- | --- | --- | --- |
| `A` fixed+auto | `57.49s` | `13.46s` | `95` | `0` | n/a |
| `D` adaptive+auto | `33.05s` | `13.46s` | `95` | `0` | n/a |
| `G1` | `55.45s` | `13.42s` | `95` | `0` | `0` |
| `G2` | `55.98s` | `13.39s` | `95` | `0` | `0` |
| `G3` | `56.66s` | `13.44s` | `95` | `0` | `0` |
| `G4` | `55.66s` | `13.53s` | `95` | `0` | `0` |
| `G5` | `56.32s` | `13.44s` | `95` | `0` | `0` |

On `SN002`, fixed margin already fits the ROI for every bbox, so guarded mode becomes a no-op. The total-time spread is therefore not meaningful evidence for the guardrail itself; the segmentation stage and output comparison are the relevant checks.

## Segmentation Drift Table

### SN009, baseline `A` vs candidate

| Variant | Voxel agreement | Dice c3 | Dice c4 |
| --- | --- | --- | --- |
| `D` | `0.994806` | `0.661819` | `0.809364` |
| `G1` | `0.999995` | `0.999998` | `0.999724` |
| `G2` | `0.999986` | `0.999969` | `0.999393` |
| `G3` | `0.999986` | `0.999969` | `0.999393` |
| `G4` | `0.999986` | `0.999969` | `0.999393` |
| `G5` | `0.998937` | `0.931830` | `0.955980` |

### SN002, baseline `A` vs candidate

All tested guarded candidates are identical to baseline:

- voxel agreement: `1.0`
- Dice c3: `1.0`
- Dice c4: `1.0`

## Metrology Drift Table

### SN009, baseline `A` vs candidate

| Variant | Pad flips | Solder flips | BLT max abs delta | BLT mean abs delta |
| --- | --- | --- | --- | --- |
| `D` | `13 / 114` | `5 / 114` | `7.0` | `2.89` |
| `G1` | `0 / 114` | `0 / 114` | `0.7` | `0.006` |
| `G2` | `0 / 114` | `0 / 114` | `1.4` | `0.018` |
| `G3` | `0 / 114` | `0 / 114` | `1.4` | `0.018` |
| `G4` | `0 / 114` | `0 / 114` | `1.4` | `0.018` |
| `G5` | `1 / 114` | `0 / 114` | `4.2` | `0.461` |

### SN002, baseline `A` vs candidate

All tested guarded candidates are identical to baseline:

- pad flips: `0 / 95`
- solder flips: `0 / 95`
- BLT max abs delta: `0.0`
- BLT mean abs delta: `0.0`

## Tradeoff Analysis

- `G1` to `G4` are effectively too strict on `SN009`. They preserve quality almost perfectly, but they only activate `1-2` fast-path crops and do not produce a meaningful segmentation-speed gain.
- `G5` is the first candidate that recovers a non-trivial part of the fast path on `SN009`: `28 / 114` crops use the direct path and segmentation time drops from `37.71s` to `31.13s`.
- Relative to full adaptive `D`, `G5` gives up some speed but removes most of the quality risk:
  - class-3 Dice improves from `0.661819` to `0.931830`
  - class-4 Dice improves from `0.809364` to `0.955980`
  - pad flips improve from `13` to `1`
  - solder flips improve from `5` to `0`
  - BLT mean abs delta improves from `2.89` to `0.461`
- `SN002` confirms that guarded mode does not destabilize already-fit crops. Every guarded candidate is identical to baseline because no shrinkage is needed at all.

## Selected Best Candidate

The single best candidate in this pass is **`G5`**:

- command shape: `--guarded-adaptive-margin --guard-max-loss-xy 12 --guard-max-loss-z 24`
- evidence-based reason: it is the only candidate that recovers a meaningful fraction of the fast path on `SN009` while keeping segmentation and metrology drift materially below full adaptive

This candidate should be kept as an **experimental option**, not as a production default.

## Recommendation

Current recommendation:

1. keep `G5` available as an explicit experiment flag
2. do not switch the default path
3. validate `G5` on more clipped samples before considering broader use
4. if follow-up tuning is needed, focus on the `z` threshold region around the current acceptance boundary rather than on direct-path numerical alignment

## Caveats

- This pass covers only two samples.
- `SN009` remains the only clipped sample in this guarded search where the guardrail actually changes crop selection.
- `G5` still causes one pad-defect flip on `SN009`, so it is not default-safe.
- Total pipeline time is secondary evidence here because this optimization only changes segmentation crop selection; unchanged upstream stages can vary because of warm caches and repeated file-system access.

## Evidence Paths

- `intelliscan/segmentation.py`
- `intelliscan/main.py`
- `intelliscan/output_guarded/SN009_A_fixed_auto/metrics.json`
- `intelliscan/output_guarded/SN009_A_fixed_auto/mmt/segmentation_stats.json`
- `intelliscan/output_guarded/SN009_D_adaptive_auto/metrics.json`
- `intelliscan/output_guarded/SN009_D_adaptive_auto/mmt/segmentation_stats.json`
- `intelliscan/output_guarded/SN009_G1_guard_xy4_z8/metrics.json`
- `intelliscan/output_guarded/SN009_G1_guard_xy4_z8/mmt/segmentation_stats.json`
- `intelliscan/output_guarded/SN009_G2_guard_xy6_z12/metrics.json`
- `intelliscan/output_guarded/SN009_G2_guard_xy6_z12/mmt/segmentation_stats.json`
- `intelliscan/output_guarded/SN009_G3_guard_xy8_z16/metrics.json`
- `intelliscan/output_guarded/SN009_G3_guard_xy8_z16/mmt/segmentation_stats.json`
- `intelliscan/output_guarded/SN009_G4_guard_xy10_z20/metrics.json`
- `intelliscan/output_guarded/SN009_G4_guard_xy10_z20/mmt/segmentation_stats.json`
- `intelliscan/output_guarded/SN009_G5_guard_xy12_z24/metrics.json`
- `intelliscan/output_guarded/SN009_G5_guard_xy12_z24/mmt/segmentation_stats.json`
- `intelliscan/output_guarded/SN002_A_fixed_auto/metrics.json`
- `intelliscan/output_guarded/SN002_A_fixed_auto/mmt/segmentation_stats.json`
- `intelliscan/output_guarded/SN002_D_adaptive_auto/metrics.json`
- `intelliscan/output_guarded/SN002_D_adaptive_auto/mmt/segmentation_stats.json`
- `intelliscan/output_guarded/SN002_G1_guard_xy4_z8/metrics.json`
- `intelliscan/output_guarded/SN002_G2_guard_xy6_z12/metrics.json`
- `intelliscan/output_guarded/SN002_G3_guard_xy8_z16/metrics.json`
- `intelliscan/output_guarded/SN002_G4_guard_xy10_z20/metrics.json`
- `intelliscan/output_guarded/SN002_G5_guard_xy12_z24/metrics.json`
- `intelliscan/output_guarded/analysis/SN009_guarded_benchmark_summary.json`
- `intelliscan/output_guarded/analysis/SN009_guarded_vs_baseline.json`
- `intelliscan/output_guarded/analysis/SN002_guarded_benchmark_summary.json`
- `intelliscan/output_guarded/analysis/SN002_guarded_vs_baseline.json`
