# Multi-GPU Batch Throughput

## Objective

Improve end-to-end throughput for multi-sample processing without changing the single-sample pipeline itself.

This is intentionally a throughput optimization, not a single-sample latency optimization. Each file still runs through the same validated default path:

- quantized in-memory detection
- `detection_image_workers=4`
- `detection_batch_prefetch=True`
- combined segmentation + metrology

The only change is orchestration across multiple GPUs.

## Why This Was Tested

After `062_detection_view_prefetch_negative_result.md`, the evidence was:

- same-GPU higher-order overlap inside one sample is not a good next step
- the current single-sample pipeline is already numerically stable
- the workspace has multiple available GPUs

So the next practical question became:

- can we improve wall-clock time for a batch of samples by dispatching independent files across multiple GPUs?

## Target Files

- `intelliscan/scripts/run_batch_multi_gpu.py`

No production inference code was changed for this experiment.

## Code Change Summary

Added a lightweight batch runner that:

- launches independent `main.py` processes
- pins each job to one GPU via `CUDA_VISIBLE_DEVICES`
- records per-job wall-clock timing, logs, assigned GPU, and output directory
- writes a summary JSON for batch-throughput comparison

The script does not:

- change model weights
- change per-sample pipeline logic
- change bbox quantization
- change segmentation routing
- change metrology or reporting logic

## Benchmark Setup

Samples:

- `SN002`
- `SN003`
- `SN009`
- `SN010`

Environment:

- sequential baseline: `GPU 3`
- parallel candidate: `GPU 3 + GPU 2`
- at benchmark start, these were the two least-used available GPUs

Runs:

- sequential summary: `intelliscan/output_multigpu/analysis/seq1gpu_summary.json`
- parallel summary: `intelliscan/output_multigpu/analysis/par2gpu_summary.json`
- parity + throughput summary: `intelliscan/output_multigpu/analysis/multigpu_batch_throughput_summary.json`

## Measured Results

| Mode | GPUs | Batch Wall Clock |
| --- | --- | ---: |
| sequential baseline | `3` | `163.54s` |
| parallel candidate | `3,2` | `90.52s` |

Throughput gain:

- wall-clock reduction: `44.65%`

Per-job elapsed times remained in the same range as the sequential run, which is expected:

- this optimization improves batch throughput
- it does not claim lower single-sample latency

## Quality Validation

Key persisted outputs were file-identical on all `4 / 4` samples:

- `bb3d.npy`
- `segmentation.nii.gz`
- `metrology/metrology.csv`

This was checked by SHA-256 hash in:

- `intelliscan/output_multigpu/analysis/multigpu_batch_throughput_summary.json`

## Risks / Caveats

- This is only useful when multiple files are waiting to be processed.
- It depends on multiple GPUs being available.
- It improves throughput, not per-file latency.
- The current validation covers four samples and two GPUs; it is not yet a full scaling study.

## Conclusion

Multi-GPU batch dispatch is a safe kept throughput optimization under the current validated workspace conditions.

- it preserves per-sample outputs
- it materially reduces batch wall-clock time
- it avoids touching the already-stabilized single-sample inference path

## Next Follow-Up

The next logical extensions are:

- a larger multi-sample throughput sweep
- `1 GPU` vs `2 GPU` vs `3 GPU` scaling
- queueing policies for mixed heavy/light samples
