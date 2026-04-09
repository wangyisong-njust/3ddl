# Documentation Index

## Recommended Reading Order

1. `project_overview.md`
2. `publication_strategy.md`
3. `current_status.md`
4. `roadmap.md`
5. `decision_log.md`
6. `3ddl_acceleration_report.md`
7. `formal_report/3ddl_acceleration_report_submission.pdf`
8. `pipeline_demo_walkthrough.md`
9. `optimization_history/031_combined_seg_metrology_default.md`
10. `optimization_history/032_report_without_bbox_nifti.md`
11. `optimization_history/033_gltf_without_bbox_pred_nifti.md`
12. `optimization_history/034_larger_direct_roi_experiment.md`
13. `optimization_history/056_true_inmemory_detection.md`
14. `optimization_history/057_detection_parity_quantization.md`
15. `optimization_history/058_quantized_inmemory_multisample_validation.md`
16. `optimization_history/059_detection_image_worker_parallelism.md`
17. `optimization_history/061_detection_batch_prefetch_overlap.md`
18. `optimization_history/062_detection_view_prefetch_negative_result.md`
19. `optimization_history/063_multigpu_batch_throughput.md`
20. `optimization_history/095_adaptive_margin_activation.md`
21. `optimization_history/096_adaptive_margin_factor_ablation.md`
22. `optimization_history/097_guarded_adaptive_margin_search.md`

## Top-Level Files

| File | Purpose |
| --- | --- |
| `README.md` | Scope, evidence rules, and navigation |
| `project_overview.md` | Cross-repo architecture and optimization layers |
| `publication_strategy.md` | Paper/report framing, innovation map, and contribution packaging |
| `3ddl_acceleration_report.md` | Full report-format draft for thesis/report writing |
| `formal_report/3ddl_acceleration_report_submission.pdf` | Formal LaTeX submission build with auto TOC, list of figures, and list of tables |
| `formal_report/3ddl_acceleration_report_submission.tex` | Generated LaTeX source for the formal submission build |
| `pipeline_demo_walkthrough.md` | One concrete raw-scan demo showing input, artifact flow, stage timing, and final outputs |
| `current_status.md` | Fast current-state summary |
| `roadmap.md` | Ranked engineering and research tasks |
| `experiment_registry.md` | Registry of measured runs, artifacts, and reported references |
| `decision_log.md` | Short decision records |
| `metrics_definition.md` | Definitions for timing, segmentation, and metrology comparisons |

## Optimization History

| File | Scope |
| --- | --- |
| `optimization_history/000_baseline_pipeline.md` | Baseline pipeline and default code path |
| `optimization_history/010_model_compression_summary.md` | Summary of the `wp5-seg` compression toolchain |
| `optimization_history/011_structured_pruning.md` | Structured pruning artifact and parameter reduction |
| `optimization_history/012_finetuning_pruned_model.md` | Finetuning recovery path and reported accuracy recovery |
| `optimization_history/013_onnx_export_and_engine_build.md` | ONNX export and TensorRT build pipeline |
| `optimization_history/014_single_patch_benchmarks.md` | Patch-level benchmark artifacts and reported TRT table |
| `optimization_history/020_trt_fp16_integration.md` | TensorRT FP16 use inside `intelliscan` |
| `optimization_history/030_pipeline_io_optimization.md` | I/O bottleneck and combined seg+metrology direction |
| `optimization_history/031_combined_seg_metrology_default.md` | Validated combined seg+metrology default path and legacy `bb`-mapping fix |
| `optimization_history/032_report_without_bbox_nifti.md` | Report reconstruction from original/full-volume outputs without per-bbox raw crop NIfTIs |
| `optimization_history/033_gltf_without_bbox_pred_nifti.md` | Combined-path GLTF decoupling from per-bbox prediction NIfTIs plus isolated stage benchmark |
| `optimization_history/034_larger_direct_roi_experiment.md` | Larger direct padded ROI study showing padded-canvas drift and no worthwhile speedup |
| `optimization_history/040_batch_inference.md` | Batch crop inference implementation and current limitation |
| `optimization_history/050_sliding_window_bypass.md` | Direct padded inference path when crop fits ROI |
| `optimization_history/055_inmemory_path_investigation.md` | Current `--inmemory` path and why it is incomplete |
| `optimization_history/056_true_inmemory_detection.md` | True in-memory detection implementation, YOLO reuse, and current cross-sample validation status |
| `optimization_history/057_detection_parity_quantization.md` | Detection parity fix showing that legacy txt precision must be reproduced in no-write branches |
| `optimization_history/058_quantized_inmemory_multisample_validation.md` | Eight-sample validation and default switch for quantized in-memory detection |
| `optimization_history/059_detection_image_worker_parallelism.md` | CPU-side detection image-preparation parallelism, `workers=1/4/8` search, and eight-sample validation for the new default |
| `optimization_history/061_detection_batch_prefetch_overlap.md` | Intra-view next-batch overlap for in-memory detection, eight-sample validation, and default switch |
| `optimization_history/062_detection_view_prefetch_negative_result.md` | Negative-result note showing that same-GPU cross-view overlap is exact but slower than the kept detection path |
| `optimization_history/063_multigpu_batch_throughput.md` | Multi-GPU batch runner and validated batch-throughput gain with unchanged per-sample outputs |
| `optimization_history/060_sn002_reported_benchmark.md` | Historical SN002 benchmark from repo report |
| `optimization_history/070_sn009_formal_benchmark.md` | Current validated SN009 formal benchmark |
| `optimization_history/080_trt_accuracy_risk.md` | TRTFP16 speed gain vs segmentation/metrology drift |
| `optimization_history/090_next_optimization_targets.md` | Ranked proposals and follow-up experiment targets |
| `optimization_history/095_adaptive_margin_activation.md` | Adaptive-margin fast-path activation, benchmark, and quality risk |
| `optimization_history/096_adaptive_margin_factor_ablation.md` | Factor-isolation ablation separating context change from path change |
| `optimization_history/097_guarded_adaptive_margin_search.md` | Guarded adaptive-margin candidate sweep and selected best experimental policy |
| `optimization_history/098_guarded_margin_z_boundary.md` | Fine sweep around the best guarded `z` threshold on clipped sample `SN009` |

## Templates

| File | Use |
| --- | --- |
| `templates/experiment_template.md` | General experiment logging |
| `templates/optimization_note_template.md` | Optimization implementation note |
| `templates/benchmark_template.md` | Benchmark result note |

## Current Evidence Inventory

### Existing Reports

- `wp5-seg/reports/segmentation_acceleration_report.md`

### Benchmark / Optimization Scripts

- `wp5-seg/train.py`
- `wp5-seg/eval.py`
- `wp5-seg/pruning/prune_basicunet.py`
- `wp5-seg/pruning/finetune_pruned.py`
- `wp5-seg/pruning/export_onnx.py`
- `wp5-seg/pruning/build_trt_engine.py`
- `wp5-seg/pruning/benchmark.py`
- `wp5-seg/pruning/benchmark_trt.py`
- `wp5-seg/pruning/eval_runtime.py`
- `wp5-seg/pruning/runtime_support.py`
- `wp5-seg/pruning/run_pruning_pipeline.sh`
- `intelliscan/main.py`
- `intelliscan/detection.py`
- `intelliscan/merge.py`
- `intelliscan/segmentation.py`
- `intelliscan/scripts/compare_ablation_outputs.py`
- `intelliscan/scripts/benchmark_seg_met_stage.py`

### Measured Outputs Present In Workspace

- `intelliscan/output_formal/SN009_current_gpu/metrics.json`
- `intelliscan/output_formal/SN009_current_gpu_inmemory/metrics.json`
- `intelliscan/output_formal/SN009_current_gpu_trt/metrics.json`
- `intelliscan/output_formal/SN009_adaptive_margin_baseline/metrics.json`
- `intelliscan/output_formal/SN009_adaptive_margin_enabled/metrics.json`
- `intelliscan/output_formal/SN009_adaptive_margin_enabled/mmt/segmentation_stats.json`
- `intelliscan/output_ablation/SN009_A_fixed_auto/metrics.json`
- `intelliscan/output_ablation/SN009_D_adaptive_auto/mmt/segmentation_stats.json`
- `intelliscan/output_ablation/SN002_A_fixed_auto/metrics.json`
- `intelliscan/output_ablation/SN002_D_adaptive_auto/mmt/segmentation_stats.json`
- `intelliscan/output_ablation/analysis/SN009_reference_vs_variants.json`
- `intelliscan/output_ablation/analysis/SN009_context_vs_context_plus_path.json`
- `intelliscan/output_ablation/analysis/SN002_reference_vs_variants.json`
- `intelliscan/output_ablation/analysis/SN002_context_vs_context_plus_path.json`
- `intelliscan/output_guarded/analysis/SN009_guarded_benchmark_summary.json`
- `intelliscan/output_guarded/analysis/SN009_guarded_vs_baseline.json`
- `intelliscan/output_guarded/analysis/SN009_guarded_z_boundary_summary.json`
- `intelliscan/output_guarded/analysis/SN009_guarded_z_boundary_vs_baseline.json`
- `intelliscan/output_guarded/analysis/SN002_guarded_benchmark_summary.json`
- `intelliscan/output_guarded/analysis/SN002_guarded_vs_baseline.json`
- `intelliscan/output_combined/analysis/combined_seg_metrology_summary.json`
- `intelliscan/output_reportless/analysis/report_without_bbox_nii_summary.json`
- `intelliscan/output_gltfless/analysis/gltfless_summary.json`
- `intelliscan/output_gltfless/analysis/gltfless_stage_only_benchmark.json`
- `intelliscan/output_inmemory_true/analysis/true_inmemory_detection_summary.json`
- `intelliscan/output_ramdisk_detect/analysis/detection_parity_quantization_summary.json`
- `intelliscan/output_inmemory_validate/analysis/multisample_quantized_inmemory_summary.json`
- `intelliscan/output_inmemory_validate/analysis/cumulative_engineering_gain_summary.json`
- `intelliscan/output_prefetch/analysis/SN009_detection_image_workers_search.json`
- `intelliscan/output_prefetch/analysis/detection_image_workers4_multisample_summary.json`
- `intelliscan/output_prefetch/analysis/cumulative_engineering_gain_workers4_summary.json`
- `intelliscan/output_prefetch2/analysis/detection_batch_prefetch_multisample_summary.json`
- `intelliscan/output_prefetch2/analysis/detection_batch_prefetch_output_parity.json`
- `intelliscan/output_prefetch2/analysis/cumulative_engineering_gain_prefetch4_summary.json`
- `intelliscan/output_prefetch2/analysis/SN002_default_prefetch4_compare.json`
- `intelliscan/output_viewprefetch/analysis/detection_view_prefetch_stage_benchmark.json`
- `intelliscan/output_viewprefetch/analysis/SN009_viewprefetch_compare.json`
- `intelliscan/output_multigpu/analysis/seq1gpu_summary.json`
- `intelliscan/output_multigpu/analysis/par2gpu_summary.json`
- `intelliscan/output_multigpu/analysis/multigpu_batch_throughput_summary.json`
- `intelliscan/output_directroi/analysis/direct_roi_experiment_summary.json`
- `intelliscan/output_directroi/analysis_SN009_directroi_compare.json`
- `intelliscan/output_directroi/analysis_SN010_directroi_compare.json`
- `intelliscan/output_directroi/analysis_SN002_directroi_compare.json`
- `intelliscan/output_directroi/analysis_SN010_stage_roi112.json`
- `intelliscan/output_directroi/analysis_SN010_stage_roi124.json`
- `intelliscan/output_formal/SN009_current_gpu/metrology/metrology.csv`
- `intelliscan/output_formal/SN009_current_gpu_trt/metrology/metrology.csv`
- `intelliscan/output_smoke/images/metrics.json`
- `wp5-seg/runs/smoke_eval/metrics/summary.json`
- `wp5-seg/runs/smoke_eval_gpu/metrics/summary.json`
- `wp5-seg/pruning/output/pruned_50pct.json`
- `wp5-seg/pruning/output/benchmark_test.json`
- `wp5-seg/runs/paper_runtime_formal/teacher_vs_r0p375_compile.json`
- `wp5-seg/runs/paper_runtime_formal/teacher_vs_r0p5_compile.json`
- `wp5-seg/runs/paper_runtime_formal/r0p375_backend_benchmark.json`
- `wp5-seg/runs/paper_runtime_formal/r0p5_backend_benchmark.json`
- `wp5-seg/reports/paper_track/runtime_backend_calibration_summary_20260409.json`
- `intelliscan/output_runtime_rank/analysis/runtime_candidate_ranking_summary.json`
- `intelliscan/output_runtime_rank/analysis/runtime_candidate_ranking_table.csv`

### Evidence Gaps To Remember

- No ONNX model is stored under this workspace.
- No TensorRT engine is stored under this workspace.
- Historical SN002 report numbers are still not linked to their original raw output directory in this workspace.
- Some benchmark artifacts point to model paths that are not present locally anymore.
