# 3D-IntelliScan: Defect-Aware Acceleration for 3D Semiconductor Inspection

Resource-efficient deep learning via model compression and pruning, applied to a complete 3D X-ray semiconductor inspection pipeline.

**Author:** Wang Yisong, B.Eng. Computer Engineering, NUS  
**Main Supervisor:** Assoc Prof Bharadwaj Veeravalli (NUS)  
**Co-Supervisor:** Dr Xu Xun (I2R, A*STAR)

---

## Key Results

| Configuration | End-to-End Time | Speedup | Quality | Deployable? |
|---|---|---|---|---|
| Original pipeline | 93.23 s | 1.00x | Baseline | Yes |
| **Safe engineering path** | **48.15 s** | **1.94x** | **Exact parity** | **Yes** |
| Teacher TRT INT8 | 39.19 s | 2.38x | 7 pad flips | Near-pass |
| r=0.5 / TRT FP16 | 29.45 s | 3.17x | 71+141 flips | No |

The recommended deployment is the **safe engineering path**: 1.94x speedup with zero quality loss.

## Project Structure

```
3ddl/
├── intelliscan/          # End-to-end inference pipeline
│   ├── main.py           # Pipeline orchestrator
│   ├── detection.py      # YOLO 2D slice detection
│   ├── segmentation.py   # 3D semantic segmentation
│   ├── metrology.py      # Geometric measurement & defect flags
│   ├── report.py         # PDF report generation
│   └── models/           # Pre-trained model weights
│
├── wp5-seg/              # Model training & optimization
│   ├── train.py          # Baseline training (CE + Dice)
│   ├── eval.py           # Full-case evaluation
│   ├── pruning/          # Compression pipeline
│   │   ├── prune_basicunet.py    # Structured channel pruning
│   │   ├── finetune_pruned.py    # Post-pruning recovery with KD
│   │   ├── qat_finetune.py       # Quantization-aware training
│   │   ├── export_onnx.py        # PyTorch → ONNX
│   │   ├── build_trt_engine.py   # ONNX → TensorRT (FP16/INT8)
│   │   └── benchmark_trt.py      # TRT vs PyTorch comparison
│   └── runs/             # Training logs & configs
│
├── docs/                 # Documentation & report
│   ├── formal_report/    # LaTeX submission source
│   ├── figures/          # Report figures
│   └── optimization_history/  # Detailed optimization notes
│
└── scripts/              # Experiment & utility scripts
```

## Pipeline Overview

The inspection pipeline processes a raw 3D X-ray CT volume through six stages:

1. **Slice Extraction** — NIfTI volume → 2D images along two orthogonal views
2. **2D Detection** — YOLO detector identifies bump regions per slice
3. **3D BBox Merge** — 2D detections merged into 3D bounding boxes
4. **3D Segmentation** — Per-bump semantic segmentation (5 classes: background, copper pillar, solder, void, copper pad)
5. **Metrology** — BLT, pad misalignment, void ratio, solder extrusion measurements + defect flags
6. **Reporting** — CSV metrology table + PDF inspection report

## Documentation

### Report
- [**Formal LaTeX Report**](docs/formal_report/) — Full submission-ready report with all experiments and analysis
- [**Acceleration Report (Markdown)**](docs/3ddl_acceleration_report.md) — Working draft

### Optimization History
Detailed notes for each optimization step:

| # | Topic | Link |
|---|---|---|
| 000 | Baseline pipeline profiling | [000_baseline_pipeline.md](docs/optimization_history/000_baseline_pipeline.md) |
| 010 | Model compression summary | [010_model_compression_summary.md](docs/optimization_history/010_model_compression_summary.md) |
| 011 | Structured pruning | [011_structured_pruning.md](docs/optimization_history/011_structured_pruning.md) |
| 012 | Finetuning pruned model | [012_finetuning_pruned_model.md](docs/optimization_history/012_finetuning_pruned_model.md) |
| 013 | ONNX export & TRT engine build | [013_onnx_export_and_engine_build.md](docs/optimization_history/013_onnx_export_and_engine_build.md) |
| 014 | Single-patch benchmarks | [014_single_patch_benchmarks.md](docs/optimization_history/014_single_patch_benchmarks.md) |
| 020 | TRT FP16 integration | [020_trt_fp16_integration.md](docs/optimization_history/020_trt_fp16_integration.md) |
| 030 | Pipeline I/O optimization | [030_pipeline_io_optimization.md](docs/optimization_history/030_pipeline_io_optimization.md) |
| 031 | Combined seg+metrology | [031_combined_seg_metrology_default.md](docs/optimization_history/031_combined_seg_metrology_default.md) |
| 032 | Report without bbox NIfTI | [032_report_without_bbox_nifti.md](docs/optimization_history/032_report_without_bbox_nifti.md) |
| 033 | GLTF without bbox pred NIfTI | [033_gltf_without_bbox_pred_nifti.md](docs/optimization_history/033_gltf_without_bbox_pred_nifti.md) |
| 034 | Larger direct ROI experiment | [034_larger_direct_roi_experiment.md](docs/optimization_history/034_larger_direct_roi_experiment.md) |
| 040 | Batch inference | [040_batch_inference.md](docs/optimization_history/040_batch_inference.md) |
| 050 | Sliding window bypass | [050_sliding_window_bypass.md](docs/optimization_history/050_sliding_window_bypass.md) |
| 055 | In-memory path investigation | [055_inmemory_path_investigation.md](docs/optimization_history/055_inmemory_path_investigation.md) |
| 056 | True in-memory detection | [056_true_inmemory_detection.md](docs/optimization_history/056_true_inmemory_detection.md) |
| 057 | Detection parity quantization | [057_detection_parity_quantization.md](docs/optimization_history/057_detection_parity_quantization.md) |
| 058 | Multi-sample validation | [058_quantized_inmemory_multisample_validation.md](docs/optimization_history/058_quantized_inmemory_multisample_validation.md) |
| 059 | Detection worker parallelism | [059_detection_image_worker_parallelism.md](docs/optimization_history/059_detection_image_worker_parallelism.md) |
| 060 | SN002 benchmark | [060_sn002_reported_benchmark.md](docs/optimization_history/060_sn002_reported_benchmark.md) |
| 061 | Batch prefetch overlap | [061_detection_batch_prefetch_overlap.md](docs/optimization_history/061_detection_batch_prefetch_overlap.md) |
| 062 | View prefetch (negative) | [062_detection_view_prefetch_negative_result.md](docs/optimization_history/062_detection_view_prefetch_negative_result.md) |
| 070 | SN009 formal benchmark | [070_sn009_formal_benchmark.md](docs/optimization_history/070_sn009_formal_benchmark.md) |
| 080 | TRT accuracy risk | [080_trt_accuracy_risk.md](docs/optimization_history/080_trt_accuracy_risk.md) |
| 095 | Adaptive margin | [095_adaptive_margin_activation.md](docs/optimization_history/095_adaptive_margin_activation.md) |
| 096 | Adaptive margin ablation | [096_adaptive_margin_factor_ablation.md](docs/optimization_history/096_adaptive_margin_factor_ablation.md) |
| 097 | Guarded adaptive margin | [097_guarded_adaptive_margin_search.md](docs/optimization_history/097_guarded_adaptive_margin_search.md) |

### Other Documentation
- [Project Overview](docs/project_overview.md)
- [Metrics Definition](docs/metrics_definition.md)
- [Decision Log](docs/decision_log.md)
- [Experiment Registry](docs/experiment_registry.md)
- [Current Status](docs/current_status.md)

## Segmentation Classes

| Class | Label | Role in Metrology |
|---|---|---|
| 0 | Background | Non-structural material |
| 1 | Copper Pillar | Vertical interconnect; width/height measured |
| 2 | Solder | Bonding material; void ratio computed within |
| 3 | **Void** | **Primary defect indicator** (gas pocket in solder) |
| 4 | Copper Pad | Landing pad; misalignment measured vs. pillar |

## Quick Start

```bash
# Run inference on a raw scan
cd intelliscan
conda activate intelliscan
python main.py /path/to/scan.nii --output output/ --tag my_scan --force

# Train segmentation model
cd wp5-seg
conda activate 3d
python train.py --data_dir 3ddl-dataset/data --output_dir runs/my_train --epochs 30

# Run pruning pipeline
cd wp5-seg/pruning
bash run_pruning_pipeline.sh
```

## Hardware

All experiments were conducted on NVIDIA L40 GPUs (48 GB VRAM) with CUDA 12.2.

## License

Research use only. Contact supervisors for licensing inquiries.
