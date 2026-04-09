# 070 SN009 Formal Benchmark

## Objective

Record the current validated formal benchmark on SN009 across baseline, `--inmemory`, and TRT FP16 variants.

## Target Files

- `intelliscan/output_formal/SN009_current_gpu/metrics.json`
- `intelliscan/output_formal/SN009_current_gpu_inmemory/metrics.json`
- `intelliscan/output_formal/SN009_current_gpu_trt/metrics.json`

## Code Change Summary

- No new code in this note. This is a measured-results document.

## Why This Change Was Made

- SN009 is the strongest currently validated end-to-end comparison available in this workspace.

## Baseline Behavior

- Current PyTorch formal run under `intelliscan/output_formal/SN009_current_gpu/`

## Optimized Behavior

- `--inmemory` formal run under `intelliscan/output_formal/SN009_current_gpu_inmemory/`
- TRT FP16 formal run under `intelliscan/output_formal/SN009_current_gpu_trt/`

## How It Was Tested

- Read current saved output files
- Recomputed baseline vs `--inmemory` segmentation equality in this pass

## Sample(s) Used

- `SN009`

## Key Timing Numbers

| Variant | Total | NII->JPG | Detection | BBox | Segmentation | Metrology | Report |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | `93.23s` | `24.43s` | `14.85s` | `2.15s` | `33.79s` | `16.95s` | `1.05s` |
| `--inmemory` | `89.86s` | `3.51s` | `16.49s` | `1.88s` | `49.58s` | `17.47s` | `0.93s` |
| TRT FP16 | `81.15s` | `22.73s` | `13.91s` | `2.14s` | `24.15s` | `16.79s` | `1.42s` |

## Accuracy / Quality Impact

- Baseline vs `--inmemory`: full-volume segmentation is identical on SN009.
- TRT FP16: see `080_trt_accuracy_risk.md` for drift details.

## Risks / Caveats

- `--inmemory` is not yet a complete no-I/O pipeline.
- TRT FP16 should not currently be treated as lossless.

## Conclusion

- Current SN009 evidence supports three conclusions:
  - segmentation is still a major latency target
  - current `--inmemory` only modestly improves end-to-end latency
  - TRT FP16 improves speed materially, but quality risk remains

## Evidence Paths

- `intelliscan/output_formal/SN009_current_gpu/metrics.json`
- `intelliscan/output_formal/SN009_current_gpu_inmemory/metrics.json`
- `intelliscan/output_formal/SN009_current_gpu_trt/metrics.json`
- `intelliscan/output_formal/SN009_current_gpu/timing.log`
- `intelliscan/output_formal/SN009_current_gpu_inmemory/timing.log`
- `intelliscan/output_formal/SN009_current_gpu_trt/timing.log`
