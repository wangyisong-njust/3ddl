# 090 Next Optimization Targets

## Objective

Capture the highest-value next tasks and the evidence that motivates them.

## 1. Adaptive Margin

- Objective: let real samples enter the batchable / direct-inference path
- Target files: `intelliscan/segmentation.py`
- Current behavior: SN009 had `0` batchable crops because fixed `margin=15` pushed all crops beyond ROI
- Proposed behavior: clamp or adapt margin per bbox so crops stay within ROI when safe
- How to test: baseline vs adaptive-margin run on SN009; compare timing and segmentation/metrology outputs
- Risk: tighter context might change class-3/class-4 outputs
- Evidence: `intelliscan/output_formal/SN009_current_gpu/execution.log`

## 2. Combined Segmentation + Metrology Default

- Objective: remove unnecessary segmentation output traffic between stages
- Target files: `intelliscan/main.py`, `intelliscan/metrology.py`, `intelliscan/report.py`
- Current behavior: default path still writes `mmt/pred/*.nii.gz` and re-reads them
- Proposed behavior: benchmark and harden the existing combined path, then consider making it the default
- How to test: same-sample timing and metrology equivalence
- Risk: report generation and debug visibility may need a delayed-save path
- Evidence: `intelliscan/main.py`, `wp5-seg/reports/segmentation_acceleration_report.md`

## 3. Truly In-Memory Detection

- Objective: remove remaining JPG/txt churn from the detection side
- Target files: `intelliscan/main.py`, `intelliscan/detection.py`, `intelliscan/utils.py`
- Current behavior: `--inmemory` still performs NIfTI->JPG conversion and loads YOLO per view
- Proposed behavior: pass arrays directly or at least reuse one detector and eliminate redundant disk steps
- How to test: same-sample total time, bbox equality, downstream segmentation equality
- Risk: detection preprocessing may change if image serialization is removed
- Evidence: `intelliscan/output_formal/SN009_current_gpu_inmemory/metrics.json`

## 4. Pipeline Parallelization

- Objective: overlap work across views and stages where independence exists
- Target files: `intelliscan/main.py`
- Current behavior: stages run serially
- Proposed behavior: parallelize view1/view2 detection and study CPU metrology overlap with GPU segmentation
- How to test: throughput and wall-clock on the same sample set
- Risk: memory pressure and logging complexity
- Evidence: stage breakdowns in `intelliscan/output_formal/SN009_current_gpu/metrics.json`

## 5. AMP / BF16 Training And Eval Study

- Objective: reduce training cost and possibly evaluation cost in `wp5-seg`
- Target files: `wp5-seg/train.py`, `wp5-seg/eval.py`
- Current behavior: training loop is FP32; current patch benchmark artifact does not show AMP inference gains
- Proposed behavior: add controlled AMP/BF16 experiments with explicit baseline comparisons
- How to test: training time, eval time, final per-class Dice, and energy where possible
- Risk: mixed precision may not help inference on this exact setup and may change numerical behavior
- Evidence: `wp5-seg/pruning/output/benchmark_test.json`

## 6. Real Calibration Dataset For TensorRT INT8

- Objective: replace random INT8 calibration with real bump crops
- Target files: `wp5-seg/pruning/build_trt_engine.py`
- Current behavior: calibration data is random noise
- Proposed behavior: build a calibration set from real training or validation crops
- How to test: FP16 vs INT8 latency and defect-sensitive metrics on the same samples
- Risk: calibration data selection bias
- Evidence: `wp5-seg/pruning/build_trt_engine.py`

## 7. Task-Aware Compression

- Objective: optimize for downstream defect and metrology stability, not only mean Dice
- Target files: `wp5-seg/train.py`, `wp5-seg/pruning/finetune_pruned.py`, future experiment code
- Current behavior: optimization and evaluation emphasize standard segmentation metrics
- Proposed behavior: use class-aware or task-aware objectives and track defect-sensitive outputs
- How to test: compare average Dice, class 3/4 Dice, defect label flips, and metrology deltas
- Risk: more complex objective, more tuning effort
- Evidence: `docs/optimization_history/080_trt_accuracy_risk.md`
