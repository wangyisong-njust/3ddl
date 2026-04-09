# 010 Model Compression Summary

## Objective

Summarize the `wp5-seg` compression and deployment toolchain in one place.

## Target Files

- `wp5-seg/pruning/prune_basicunet.py`
- `wp5-seg/pruning/finetune_pruned.py`
- `wp5-seg/pruning/export_onnx.py`
- `wp5-seg/pruning/build_trt_engine.py`
- `wp5-seg/pruning/benchmark.py`
- `wp5-seg/pruning/benchmark_trt.py`
- `wp5-seg/pruning/run_pruning_pipeline.sh`

## Status By Step

| Step | Status | Notes |
| --- | --- | --- |
| Structured pruning | Validated artifact | Pruned checkpoint metadata exists locally |
| Finetuning | Reported but not revalidated in this pass | Script exists, raw finetune run output not present locally |
| ONNX export | Script present | No ONNX artifact stored in workspace |
| TensorRT engine build | Script present | No local engine artifact stored in workspace |
| Patch benchmark | Partly validated | PyTorch benchmark artifact exists; TRT table currently comes from repo report |

## Why This Change Family Matters

- End-to-end pipeline optimization depends on segmentation latency.
- Model compression and pipeline engineering are separate bottlenecks and should be tracked separately.

## Baseline Behavior

- Original model format: full BasicUNet checkpoint
- Model size in workspace: `22M` for `intelliscan/models/segmentation_model.ckpt`

## Optimized Behavior

- Pruned checkpoint artifact exists and is substantially smaller.
- TensorRT deployment path exists in code and has already been used in a formal SN009 run.

## How It Was Tested

- Inspected scripts and artifacts already stored in workspace
- Cross-referenced current formal SN009 TRT run and existing repo report

## Key Numbers

- Pruned checkpoint metadata: `74.97%` parameter reduction
- Current file sizes:
  - original checkpoint: `22M`
  - pruned checkpoint: `5.6M`
  - external TRT FP16 engine used in formal run: `4.6M` (observed during this pass, engine not stored under workspace root)

## Accuracy / Quality Impact

- Compression quality is not safe to summarize with average Dice alone.
- Current SN009 TRT evidence shows speed gain with defect-sensitive output drift.

## Risks / Caveats

- Some reported results rely on `wp5-seg/reports/segmentation_acceleration_report.md`.
- The workspace does not currently contain the ONNX export or TensorRT engine outputs used to produce all historical numbers.

## Conclusion

- The compression toolchain is real and mostly present in code, but the strongest speed claims should remain tagged as reported unless raw artifacts are restored or regenerated.

## Evidence Paths

- `wp5-seg/pruning/output/pruned_50pct.json`
- `wp5-seg/pruning/output/benchmark_test.json`
- `wp5-seg/reports/segmentation_acceleration_report.md`
- `intelliscan/output_formal/SN009_current_gpu_trt/metrics.json`
