# Roadmap

## Immediate Engineering Tasks

| Priority | Task | Expected Impact | Implementation Risk | Proof Required |
| --- | --- | --- | --- | --- |
| 1 | Validate the current quantized in-memory + `4`-worker + batch-prefetch detection default on any newly added or production-only scans | High; this is now the kept default front-end path | Low | Same-sample bbox / segmentation / metrology comparisons |
| 2 | Revalidate actual GLTF file generation in a PyVista-enabled environment now that bump GLTF export no longer depends on `mmt/pred` | Medium | Low to medium | Successful `.gltf` creation without per-bbox prediction NIfTIs |
| 3 | Extend the kept multi-GPU batch-throughput path from `2` GPUs to a broader scaling study | High for deployment throughput; this is the first validated higher-level orchestration win after the rejected same-GPU cross-view overlap | Medium | Same-sample output checks plus `1` vs `2` vs `3` GPU wall-clock comparisons |
| 4 | Acquire or expose more clipped validation cases for any future guardrail work | Medium | Medium | Additional raw `metrics.json`, `segmentation_stats.json`, and output comparisons |
| 5 | Only revisit the ramdisk detection harness if a future no-write detection regression needs factor isolation | Low current impact; quantized `--inmemory` is already the kept path | Low | A new regression that cannot be explained by the current quantization-compatible path |

## Validation / Benchmarking Tasks

| Priority | Task | Expected Impact | Risk | Proof Required |
| --- | --- | --- | --- | --- |
| 1 | Expand combined-default, reportless, and quantized true-inmemory validation on any future raw scans beyond the current local eight-sample set | Medium; the current local set is already exact, but future data should continue to agree | Low | Same-output checks plus stage timing on additional samples |
| 2 | Build a comparable sample set for baseline vs TRT FP16 | High | Medium | Same hardware, same samples, same bbox counts |
| 3 | Measure TRT drift on more than one sample | High | Medium | Class-level Dice plus metrology deltas across multiple cases |
| 4 | Track defect label flips, not only Dice | High | Low | CSV-level comparison script or documented manual process |
| 5 | Only revisit guarded adaptive margin if future work keeps the original crop context intact | Medium | Medium | New evidence showing a geometry-preserving policy rather than context shrinkage |

## Model Compression Research Tasks

| Priority | Task | Expected Impact | Risk | Proof Required |
| --- | --- | --- | --- | --- |
| 1 | Improve student training or selection under the whole-pipeline gate, using `r=0.375 + TRT FP16` as the quality runtime reference and `r=0.5 + TRT FP16` as the speed runtime reference | High; backend ranking is now done, and current evidence shows the remaining blocker is model-induced drift | Medium to high | Same-sample pipeline latency, class-3/class-4 stability, defect flips, and metrology deltas |
| 2 | Task-aware compression objective | High research value | Medium to high | Compare mean Dice, class-3/4 Dice, defect recall, metrology deltas |
| 3 | Upgrade INT8 from fit-only PTQ toward deployment-worthy INT8 only if FP16 is still too costly | Medium to high; fit-only calibration is the first positive result, but FP16 still wins overall | High | FP16 vs INT8 latency, power, class-level metrics, and downstream metrology |
| 4 | AMP/BF16 training efficiency study | Medium; may reduce training cost and energy, but is no longer the top deployment bottleneck | Medium | Training time, eval time, final Dice, class-level metrics |
| 5 | Hardware-aware sparsity / channel layouts | Medium | High | Measured latency on target GPU, not just parameter count |

## Operator / Runtime Acceleration Tasks

| Priority | Task | Expected Impact | Risk | Proof Required |
| --- | --- | --- | --- | --- |
| 1 | Treat backend selection as completed for the current retained students: use TRT FP16 as the main deployment backend and compile only as a fallback PyTorch runtime | High current impact; this is now evidenced at both patch and whole-pipeline levels | Low | Already established in `../wp5-seg/reports/paper_track/020_runtime_backend_calibration_compile_energy.md` and `../wp5-seg/reports/paper_track/021_intelliscan_runtime_candidate_ranking.md` |
| 2 | Profile segmentation runtime at the operator level for fit-crop versus sliding-window cases | Medium; helps decide whether remaining cost is window orchestration, padding, or memory movement | Medium | Reproducible profiler traces tied to the same sample set |
| 3 | Revisit larger direct ROI only if direct-path numerics are redesigned or the model is retrained for a larger direct canvas | Low current impact under the existing model; current evidence shows no worthwhile speedup and severe drift | High | Same-sample parity plus repeated stage-only benchmarks |
| 4 | Do not prioritize same-GPU cross-view detection overlap under the current single-model path | Low current impact; the repeated detection-only benchmark shows it is slower while adding complexity | Low | Already established in `062_detection_view_prefetch_negative_result.md` |
| 5 | Only revisit compile for service-style warm pipelines or model-server scenarios | Medium | Medium | A persistent-process benchmark that amortizes compile warmup |

## Long-Term Publication-Quality Directions

| Priority | Direction | Why It Matters |
| --- | --- | --- |
| 1 | Build a Pareto study across latency, model size, energy, class Dice, and metrology stability | Stronger than reporting average Dice alone |
| 2 | Standardize a defect-sensitive benchmark protocol | Needed to justify deployment-safe compression claims |
| 3 | Publish pipeline-level optimization separately from model-level compression, and distinguish single-sample latency from batch throughput | The current evidence now contains both kinds of engineering improvement |
| 4 | Extend evaluation beyond segmentation to final defect decisions | More aligned with semiconductor inspection outcomes |
| 5 | Frame the work as defect-aware co-optimization across model, algorithm, engineering, and runtime layers | This is the cleanest way to turn the current mixed evidence into a coherent paper or thesis chapter |
