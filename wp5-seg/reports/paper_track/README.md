# WP5 Compression Paper Track

This directory stores branch-local research memory for the `research/wp5-compression-paper` line.

## Scope

Focus on paper-worthy compression and deployment work for `wp5-seg`, especially:

- structured pruning beyond uniform ratios
- teacher-student distillation for pruned models
- AMP/BF16 training efficiency
- real-data INT8 calibration and QAT
- defect-aware and metrology-aware evaluation

## Logging Rules

- Add one numbered markdown file per meaningful change or experiment.
- Separate measured results from ideas.
- Mark any claim without raw local evidence as `Reported but not revalidated`.
- Prefer relative paths to code, logs, and artifacts.

## Retention Gate

For this paper track, a method is only retained if it clears both levels:

- `wp5-seg` level: strong full-case segmentation quality, not just small-slice or patch-level wins
- `intelliscan` level: acceptable whole-pipeline deployment behavior, especially class-sensitive stability and metrology / defect consistency

Methods that improve patch or full-case Dice but fail downstream metrology are recorded as negative results and not kept as preferred deployment candidates.

## Entries

- [001_distillation_amp_setup.md](./001_distillation_amp_setup.md)
- [002_smoke_distill_vs_plain_bf16.md](./002_smoke_distill_vs_plain_bf16.md)
- [003_kd_weight_sweep_bf16.md](./003_kd_weight_sweep_bf16.md)
- [004_kd_precision_compare.md](./004_kd_precision_compare.md)
- [005_delayed_kd_ablation.md](./005_delayed_kd_ablation.md)
- [006_class_aware_kd_ablation.md](./006_class_aware_kd_ablation.md)
- [007_recovery_gap_check.md](./007_recovery_gap_check.md)
- [008_teacher_baseline_revalidation.md](./008_teacher_baseline_revalidation.md)
- [009_fullcase_recovery_ranking.md](./009_fullcase_recovery_ranking.md)
- [010_feature_attention_kd_ablation.md](./010_feature_attention_kd_ablation.md)
- [011_pruning_ratio_screening.md](./011_pruning_ratio_screening.md)
- [012_pruning_ratio_pareto_ranking.md](./012_pruning_ratio_pareto_ranking.md)
- [013_intelliscan_student_deployment_ablation.md](./013_intelliscan_student_deployment_ablation.md)
- [014_selection_gate_whole_pipeline_first.md](./014_selection_gate_whole_pipeline_first.md)
- [015_supervised_reweighting_negative_result.md](./015_supervised_reweighting_negative_result.md)
- [016_boundary_loss_pilot_positive_fullcase_negative.md](./016_boundary_loss_pilot_positive_fullcase_negative.md)
- [017_metrology_aware_checkpoint_selection_probe.md](./017_metrology_aware_checkpoint_selection_probe.md)
- [018_checkpoint_panel_selection.md](./018_checkpoint_panel_selection.md)
- [019_checkpoint_panel_third_sample_check.md](./019_checkpoint_panel_third_sample_check.md)
- [020_runtime_backend_calibration_compile_energy.md](./020_runtime_backend_calibration_compile_energy.md)
- [021_intelliscan_runtime_candidate_ranking.md](./021_intelliscan_runtime_candidate_ranking.md)
