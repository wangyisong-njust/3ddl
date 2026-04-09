# Publication Strategy

## Objective

Turn the current acceleration work into a write-up that is technically credible, experimentally traceable, and strong enough to support either:

1. a near-term engineering report, or
2. a later paper built around defect-aware co-optimization.

This note separates what is already supported by workspace evidence from what is still a proposal.

## Strongest Current Story

The strongest grounded narrative today is not "one model got faster." It is:

- end-to-end 3D semiconductor inspection is bottlenecked by both model runtime and pipeline engineering
- some accelerations are safe and worth keeping
- some accelerations are fast but unsafe for defect-sensitive outputs
- defect-aware metrics and metrology must be used to judge acceptability

This is stronger than a pure Dice-only or patch-only acceleration story because it connects speed to final inspection behavior.

## What Can Already Be Claimed

### Validated In This Workspace

| Layer | Claim | Why It Matters | Evidence |
| --- | --- | --- | --- |
| Engineering | Combined segmentation + metrology is safe and faster on validated `SN002` and `SN009` runs | Removes redundant reread/rewrite traffic without changing segmentation outputs | `docs/optimization_history/031_combined_seg_metrology_default.md` |
| Engineering | Report generation no longer needs `mmt/img`, and the final memory-reuse version is safe and faster on validated `SN002` and `SN009` runs | Removes one class of intermediate artifacts while preserving outputs | `docs/optimization_history/032_report_without_bbox_nifti.md` |
| Engineering | Combined-path bump GLTF export no longer needs per-bbox prediction NIfTIs, and isolated stage benchmarks show a `5.33%` to `8.81%` gain in the target stage | Removes another artifact class without changing validated outputs; useful example of why stage-isolated benchmarking matters | `docs/optimization_history/033_gltf_without_bbox_pred_nifti.md` |
| Runtime | Txt-quantized true in-memory detection plus `4`-worker CPU image preparation and intra-view batch prefetch keeps key outputs identical on all `8 / 8` validated local raw scans and further reduces the default front-end stage | This turns detection into a stronger default-safe engineering gain on the current dataset and shows that CPU preparation and batch serialization were both first-order bottlenecks even after file I/O was removed | `docs/optimization_history/059_detection_image_worker_parallelism.md`, `docs/optimization_history/061_detection_batch_prefetch_overlap.md` |
| Runtime | `torch.compile(reduce-overhead)` is a retained PyTorch fallback backend for the current `r=0.5` student, but TRT FP16 remains the preferred runtime backend overall | This adds a backend-level runtime result that is useful even when TensorRT deployment is unavailable, while still keeping the main deployment story grounded in the better FP16 backend | `wp5-seg/reports/paper_track/020_runtime_backend_calibration_compile_energy.md` |
| Runtime | Real-data INT8 calibration is now validated as a nuanced result rather than a future placeholder: fit-only calibration improves `r=0.375` INT8 quality relative to generic real calibration, but FP16 still wins overall | This strengthens the paper because calibration is now discussed with both a positive result and a negative result, instead of as an unsupported assumption | `wp5-seg/reports/paper_track/020_runtime_backend_calibration_compile_energy.md` |
| Runtime | Power / energy can now be reported alongside latency for PyTorch, compiled PyTorch, TRT FP16, and TRT INT8 | This supports a stronger latency-size-energy-quality narrative for deployment work | `wp5-seg/reports/paper_track/020_runtime_backend_calibration_compile_energy.md` |
| Deployment | Whole-pipeline ranking now shows that backend optimization does not materially change teacher-relative drift inside the same student family; backend and model ranking must be treated separately | This is a strong deployment-methodology result, and it prevents the paper from overclaiming that runtime tricks fix compressed-model quality | `wp5-seg/reports/paper_track/021_intelliscan_runtime_candidate_ranking.md` |
| Runtime | Same-GPU cross-view detection overlap is a validated negative result: it stays exact but slows down the target stage on both benchmarked samples | Useful because it shows that not all scheduling parallelism helps; resource contention can outweigh overlap when one YOLO model still serializes GPU execution | `docs/optimization_history/062_detection_view_prefetch_negative_result.md` |
| Systems | Multi-GPU batch dispatch improves batch wall clock without changing any per-sample outputs or code paths | This adds a practical deployment-throughput result and cleanly separates batch throughput gains from single-sample latency gains | `docs/optimization_history/063_multigpu_batch_throughput.md` |
| Runtime | Larger direct padded ROI is not a safe acceleration under the current model: same-crop runs still drift, and repeated stage-only timing on `SN010` is slower at `roi124` than baseline | Strong negative result showing padded-canvas size is a hidden parity-sensitive variable | `docs/optimization_history/034_larger_direct_roi_experiment.md` |
| Algorithm | Adaptive-margin drift is dominated by context shrinkage, not by the real fast-path inference mode itself | This is an actual causal finding, not just a speed observation | `docs/optimization_history/096_adaptive_margin_factor_ablation.md` |
| Metrics | Voxel accuracy alone is misleading; class `3`, class `4`, and metrology must drive acceptability decisions | Important methodological contribution for a defect-sensitive pipeline | `docs/optimization_history/080_trt_accuracy_risk.md`, `docs/metrics_definition.md` |
| Model | Structured pruning artifact reduces parameters from `5,749,509` to `1,438,853` (`74.97%` reduction) | Shows clear model-size compression already exists in code and artifacts | `wp5-seg/pruning/output/pruned_50pct.json` |
| Model | Current patch benchmark artifact shows pruned PyTorch FP32 inference faster than the baseline patch benchmark (`2.76x` in the saved artifact) | Gives a grounded model-level speed result even without a full TRT rerun in this pass | `wp5-seg/pruning/output/benchmark_test.json` |

### Reported But Not Revalidated In This Workspace Pass

| Claim | Why It Is Useful | Evidence |
| --- | --- | --- |
| Existing report states a two-stage acceleration story: model compression plus pipeline engineering | Useful as prior internal summary material for a future paper/report | `wp5-seg/reports/segmentation_acceleration_report.md` |
| Existing report states strong TensorRT patch-level speedups | Good motivation for further deployment work, but should not be treated as newly reproduced here | `wp5-seg/reports/segmentation_acceleration_report.md` |

## Innovation Map

### 1. Model-Level Innovation

Already present in code:

- structured pruning
- pruning-aware finetuning pipeline
- ONNX export
- TensorRT engine build scripts

Best next publishable angle:

- move from average-Dice-oriented compression to defect-aware compression
- keep model size, latency, class `3/4` Dice, and metrology stability in the same table

Status:

- compression tooling: implemented
- defect-aware compression objective: proposed, not yet validated
- real INT8 calibration: partially validated; fit-only selection is better than generic real calibration on `r=0.375`, but FP16 still wins as the retained backend

### 2. Algorithm-Level Innovation

Already supported by evidence:

- fast-path activation is not the real problem by itself
- crop/context geometry is the dominant risk when results drift

This is useful because it turns a vague optimization idea into a causal statement:

- unsafe idea: force more crops into ROI by trimming true context
- initially plausible but now disproven direction under the current model: enlarge the direct padded ROI without clipping true context
- safer direction: preserve both crop geometry and direct-path numerical behavior, not only the raw bbox extent

Status:

- factor-isolation ablation: validated
- larger direct-ROI study: validated negative result

### 3. Engineering / Systems Innovation

Already supported by evidence:

- combined segmentation + metrology
- report reconstruction without `mmt/img`
- partial in-memory execution
- explicit output-parity comparisons
- detection-stage parity depends on reproducing legacy serialization precision, not only on matching the image path

This is probably the strongest near-term report contribution because it directly affects end-to-end latency on real scans, not only patch benchmarks.

Status:

- combined default: validated
- reportless memory reuse: validated
- GLTF decoupling from `mmt/pred`: validated in code and output parity, but `.gltf` file creation itself still needs a PyVista-enabled recheck
- true in-memory detection with txt-equivalent quantization: validated on `8 / 8` local raw scans and now used as the default detection path, with `--file-detection` kept as a fallback
- pipeline parallelization: still proposed, but the new evidence now splits it into a rejected branch and a kept branch:
  - rejected: same-GPU cross-view overlap inside one sample
  - kept: multi-GPU dispatch across independent samples

### 4. Operator / Runtime Innovation

Already present or partially present:

- GPU normalization
- direct padded inference when crop fits ROI
- TensorRT FP16 integration

Best next runtime angles:

- profile sliding-window overhead versus direct padded inference on more real crops
- profile PyTorch path versus TensorRT path with defect-aware validation
- only revisit larger direct ROI after direct-path numerical alignment or retraining for the larger direct canvas

Proposed but not yet validated in this workspace:

- runtime compilation / graph capture for repeated crop inference
- operator-level profiling to identify whether memory movement, padding, or window orchestration dominates the remaining segmentation cost

Updated note:

- runtime compilation is no longer just proposed
- `torch.compile(reduce-overhead)` is now a retained fallback backend for `r=0.5`
- but it is not the strongest end state, because TRT FP16 still dominates the current patch-level latency-power tradeoff
- after whole-pipeline ranking, compile is also no longer a strong current single-scan deployment result because cold-start compile cost dominates the current CLI workflow

## Best Write-Up Options

### Option A: Engineering Report First

Best if the goal is a solid internal or thesis-stage report soon.

Suggested framing:

- baseline bottleneck breakdown
- safe optimizations that already work
- fast but unsafe optimizations and why they fail
- next defect-aware acceleration roadmap

Why this is strong now:

- most evidence already exists locally
- it can include both positive and negative ablations
- it does not need every future optimization to be finished first

### Option B: Paper Later

Best if the goal is a stronger publication after more experiments.

Suggested framing:

- defect-aware co-optimization of model compression and pipeline runtime for 3D semiconductor inspection

Minimum extra evidence still needed:

1. more multi-sample validation for the safe engineering path
2. a clearer decision on whether true in-memory detection can be made lossless
3. defect-aware compression results beyond average Dice
4. at least one geometry-preserving runtime optimization beyond the current safe I/O improvements

Note:

- The new larger-direct-ROI result narrows what “geometry-preserving” should mean.
- Preserving only the raw crop is insufficient; future work also has to preserve or explicitly retrain the direct-path canvas the network sees.

## Recommended Contribution Packaging

### Near-Term Claimable Contributions

1. End-to-end profiling shows that pipeline engineering, not just model forward time, dominates deployment latency.
2. Safe I/O refactors can deliver measurable gains without changing segmentation or metrology outputs.
3. No-write detector paths must reproduce legacy serialization precision; otherwise small bbox changes can propagate into 3D merge drift.
4. Defect-sensitive evaluation is necessary because voxel accuracy and even average Dice can hide practically important drift.
5. Controlled ablations can explain why a speedup is unsafe, instead of only reporting that it changed outputs.

### Mid-Term Contributions To Target

1. A geometry-preserving fast-path policy that accelerates inference without trimming true context.
2. A defect-aware compression objective or evaluation protocol for class `3/4` and metrology stability.
3. A unified Pareto view across latency, model size, class Dice, defect flips, and metrology deltas.

## Suggested Figures / Tables For A Future Write-Up

1. End-to-end baseline timing breakdown for one or more formal samples.
2. Safe engineering optimization table: baseline vs combined-default vs reportless memory reuse.
3. Unsafe optimization table: baseline vs TRT FP16 vs full adaptive margin.
4. Factor-isolation ablation: path-only vs context-only vs full adaptive.
5. Negative runtime ablation: larger direct ROI without crop trimming still drifts and fails to improve repeated stage timing.
6. Compression table: baseline model vs pruned model, with latency, size, and defect-sensitive metrics.
7. Final Pareto plot: latency versus class-3/class-4 or metrology stability.

## Recommended Next Execution Order

1. Finish the remaining safe engineering cleanup:
   - keep validating combined/reportless/no-`mmt/pred` paths on more raw scans
   - recheck actual GLTF output creation with PyVista installed
2. Use the now-stable detection baseline to pursue the next safe throughput layers:
   - multi-GPU batch scaling
   - multi-file stage overlap
   - heterogeneous-resource overlap
3. Upgrade model-compression evaluation:
   - real INT8 calibration
   - class `3/4` and metrology-aware comparison tables
4. Revisit runtime design only after the current direct-path parity risks are better understood:
   - larger direct ROI is now a documented negative result under the current model

## Bottom Line

If a report had to be written now, the best defensible angle is:

- a defect-aware acceleration study of a real 3D semiconductor inspection pipeline, showing both safe system-level gains and the risks of speed-first shortcuts.

If a paper is the target, the strongest next step is not another unconstrained speed hack. It is to add one more geometry-preserving runtime optimization and one more defect-aware compression study, then unify them under a single latency-versus-quality framework.
