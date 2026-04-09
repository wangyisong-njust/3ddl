# 098 Guarded Margin Z Boundary

## Objective

Refine the best guarded candidate from `097_guarded_adaptive_margin_search.md`.

`097` showed that:

- `G1` to `G4` were almost lossless but nearly no-op on `SN009`
- `G5` (`xy<=12`, `z<=24`) was the first useful speed-vs-quality point

This pass checks whether the useful region opens smoothly below `z=24`, or whether `z=24` is the first threshold that materially activates the fast path.

## Setup

- sample: `SN009`
- fixed `guard_max_loss_xy=12`
- sweep `guard_max_loss_z` over `21`, `22`, `23`, `24`

The local workspace only exposes two NIfTI samples (`SN002`, `SN009`), and only `SN009` is a clipped case for this guardrail. No extra clipped sample was available locally in this pass.

## Benchmark Summary

| Variant | `guard_max_loss_z` | Total | Segmentation | Direct path | Sliding path | Guard accepted |
| --- | --- | --- | --- | --- | --- | --- |
| `G5a` | `21` | `90.72s` | `33.06s` | `2` | `112` | `2` |
| `G5b` | `22` | `89.46s` | `33.20s` | `2` | `112` | `2` |
| `G5c` | `23` | `90.59s` | `33.08s` | `2` | `112` | `2` |
| `G5` | `24` | `87.23s` | `31.13s` | `28` | `86` | `28` |

## Output Drift vs Baseline

### Segmentation

| Variant | Voxel agreement | Dice c3 | Dice c4 |
| --- | --- | --- | --- |
| `G5a` | `0.999986` | `0.999969` | `0.999393` |
| `G5b` | `0.999986` | `0.999969` | `0.999393` |
| `G5c` | `0.999986` | `0.999969` | `0.999393` |
| `G5` | `0.998937` | `0.931830` | `0.955980` |

### Metrology

| Variant | Pad flips | Solder flips | BLT max abs delta | BLT mean abs delta |
| --- | --- | --- | --- | --- |
| `G5a` | `0 / 114` | `0 / 114` | `1.4` | `0.018` |
| `G5b` | `0 / 114` | `0 / 114` | `1.4` | `0.018` |
| `G5c` | `0 / 114` | `0 / 114` | `1.4` | `0.018` |
| `G5` | `1 / 114` | `0 / 114` | `4.2` | `0.461` |

## Interpretation

- `z=21`, `22`, and `23` are functionally the same in this sample:
  - `2` accepted boxes
  - `112` sliding-window boxes
  - nearly identical segmentation and metrology deltas
- The useful region does **not** open gradually in the tested range.
- `z=24` is the first tested threshold that materially changes activation behavior:
  - accepted boxes jump from `2` to `28`
  - segmentation time improves from about `33.1s` to `31.13s`
  - quality cost also jumps, though it remains much better than full adaptive

## Conclusion

In the tested region, the guarded `z` threshold behaves like a **cliff**, not a smooth tradeoff curve:

- `z<=23`: near-baseline / near-no-op
- `z=24`: first meaningful speedup point, with small but non-zero business-level drift

So `G5` remains the best current experimental policy, and this pass did **not** find a better intermediate threshold below it.

## Recommendation

Do not spend more time on finer local tuning below `z=24` unless the crop-selection logic changes.

The next useful step is to validate `G5` on more clipped samples, because the current workspace only contains one clipped case.

## Evidence Paths

- `intelliscan/output_guarded/SN009_G5a_guard_xy12_z21/metrics.json`
- `intelliscan/output_guarded/SN009_G5a_guard_xy12_z21/mmt/segmentation_stats.json`
- `intelliscan/output_guarded/SN009_G5b_guard_xy12_z22/metrics.json`
- `intelliscan/output_guarded/SN009_G5b_guard_xy12_z22/mmt/segmentation_stats.json`
- `intelliscan/output_guarded/SN009_G5c_guard_xy12_z23/metrics.json`
- `intelliscan/output_guarded/SN009_G5c_guard_xy12_z23/mmt/segmentation_stats.json`
- `intelliscan/output_guarded/SN009_G5_guard_xy12_z24/metrics.json`
- `intelliscan/output_guarded/SN009_G5_guard_xy12_z24/mmt/segmentation_stats.json`
- `intelliscan/output_guarded/analysis/SN009_guarded_z_boundary_summary.json`
- `intelliscan/output_guarded/analysis/SN009_guarded_z_boundary_vs_baseline.json`
