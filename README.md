# 3D-IntelliScan: Defect-Aware Acceleration for 3D Semiconductor Inspection

Resource-efficient deep learning via model compression and pruning, applied to a complete 3D X-ray semiconductor inspection pipeline. Achieves **1.94x end-to-end speedup with exact quality parity** on NVIDIA L40 hardware.

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
All benchmarks run on a single NVIDIA L40 (48 GB VRAM), CUDA 12.2, TensorRT 10.x.

---

## Environment Setup

Two separate conda environments are used: one for the inference pipeline (`intelliscan`) and one for model training/compression (`3d`).

### Inference pipeline (`intelliscan`)

```bash
conda create -n intelliscan python=3.10
conda activate intelliscan
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu122
pip install ultralytics nibabel numpy scipy matplotlib reportlab
# TensorRT is required for the optimised path:
pip install tensorrt  # or install via NVIDIA TensorRT wheel matching your CUDA version
```

### Training / compression (`3d`)

```bash
conda create -n 3d python=3.12
conda activate 3d
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu122
pip install monai nibabel numpy scipy matplotlib
pip install onnx onnxruntime tensorrt
```

> If you are working inside the repo, a local `.venv` under `wp5-seg/` contains the exact packages used during development (Python 3.12, PyTorch 2.9.1, MONAI 1.5.1, ONNX 1.21.0).

---

## Data Layout

The dataset is not distributed with this repo. The expected layout under `wp5-seg/3ddl-dataset/data/` is:

```
3ddl-dataset/data/
├── dataset_config.json          # train/test split (15 held-out test serials)
├── checksums.sha256             # integrity manifest
├── SN001/
│   ├── image.nii                # raw 3D X-ray CT volume (NIfTI)
│   └── label.nii                # voxel segmentation mask (5 classes)
├── SN002/
│   ├── image.nii
│   └── label.nii
└── ...                          # further serial numbers
```

Segmentation classes: `0` background · `1` copper pillar · `2` solder · `3` void (defect) · `4` copper pad.

Verify an existing download:

```bash
cd wp5-seg/3ddl-dataset
python verify_dataset.py data/
```

For the end-to-end inference pipeline, inputs are raw `.nii` volumes (one per scan):

```bash
# List the scans you want to process (one path per line):
cp intelliscan/files.example.txt intelliscan/files.txt
# then edit files.txt
```

---

## Reproducing Paper Results

### Table 1 — End-to-end pipeline benchmark (SN009)

```bash
cd intelliscan
conda activate intelliscan

# Baseline (original pipeline, no optimisations)
python main.py /path/to/SN009.nii --output output/baseline --tag SN009_baseline --force

# Safe engineering path (recommended deployment)
python main.py /path/to/SN009.nii --output output/safe --tag SN009_safe --force \
    --use_trt_detection --use_inmemory
```

Detailed per-stage timing and the metrology CSV are written to the output directory.  
Reference numbers: [070_sn009_formal_benchmark.md](docs/optimization_history/070_sn009_formal_benchmark.md).

### Table 2 — Segmentation model compression

```bash
cd wp5-seg
conda activate 3d

# 1. Train baseline segmentation model
python train.py \
    --data_dir 3ddl-dataset/data \
    --output_dir runs/baseline \
    --epochs 30 --batch_size 4 --seed 42 \
    --roi_x 112 --roi_y 112 --roi_z 80

# 2. Evaluate baseline
python eval.py --model_path runs/baseline/best.ckpt --data_dir 3ddl-dataset/data

# 3. Prune → finetune → benchmark (r = 0.3, safe path)
PRUNING_RATIO=0.3 MODEL_PATH=runs/baseline/best.ckpt \
    bash pruning/run_pruning_pipeline.sh

# 4. Export to TRT FP16 engine
conda activate 3d
python pruning/export_onnx.py   --model_path runs/pruning_r0.3/segmentation_model_pruned.ckpt \
                                 --output     runs/pruning_r0.3/model.onnx
python pruning/build_trt_engine.py --onnx runs/pruning_r0.3/model.onnx \
                                    --output runs/pruning_r0.3/model_fp16.trt --fp16

# 5. Single-patch TRT vs PyTorch benchmark
python pruning/benchmark_trt.py --trt_engine runs/pruning_r0.3/model_fp16.trt \
                                  --model_path runs/pruning_r0.3/segmentation_model_pruned.ckpt
```

Reference numbers: [011_structured_pruning.md](docs/optimization_history/011_structured_pruning.md) and [014_single_patch_benchmarks.md](docs/optimization_history/014_single_patch_benchmarks.md).

### Accuracy risk analysis (TRT INT8 vs FP16)

```bash
# Compare flip counts at different TRT precision modes
python pruning/benchmark_trt.py --accuracy_check --data_dir 3ddl-dataset/data \
    --trt_engine runs/pruning_r0.3/model_int8.trt
```

Reference: [080_trt_accuracy_risk.md](docs/optimization_history/080_trt_accuracy_risk.md).

---

## Smoke Test (no dataset required)

A lightweight check that the pipeline loads and runs without a real scan.  
`intelliscan/output_smoke/` contains pre-computed detection outputs that are replayed:

```bash
cd intelliscan
conda activate intelliscan
python main.py --smoke_test
```

Expected output: pipeline completes in <5 s, writes a CSV and PDF to `output_smoke/`, exits 0.

---

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
├── wp5-seg/              # Model training & optimisation
│   ├── train.py          # Baseline training (CE + Dice)
│   ├── eval.py           # Full-case evaluation
│   ├── pruning/          # Compression pipeline
│   │   ├── prune_basicunet.py    # Structured channel pruning
│   │   ├── finetune_pruned.py    # Post-pruning recovery with KD
│   │   ├── qat_finetune.py       # Quantization-aware training
│   │   ├── export_onnx.py        # PyTorch → ONNX
│   │   ├── build_trt_engine.py   # ONNX → TensorRT (FP16/INT8)
│   │   └── benchmark_trt.py      # TRT vs PyTorch comparison
│   └── 3ddl-dataset/     # Dataset loader & integrity tools
│
├── docs/                 # Documentation & report
│   ├── formal_report/    # LaTeX submission source
│   ├── figures/          # Report figures
│   └── optimization_history/  # Per-experiment notes (000–097)
│
└── scripts/              # Experiment & utility scripts
```

---

## Known Limitations

- **Hardware:** Optimised path (TRT engines) requires an NVIDIA GPU with CUDA 12.x and TensorRT 10.x. CPU-only inference is not supported.
- **L40-specific tuning:** TRT engine files (`.trt`) are device-specific and must be rebuilt on your target hardware.
- **Dataset:** The proprietary 3D X-ray CT dataset is not publicly released. Contact the supervisors for data access.
- **Smoke test:** The `--smoke_test` flag replays cached detections; it does not exercise GPU segmentation.
- **INT8 path:** Produces 7 metrology flip errors on SN009 and is not recommended for production use.

---

## Documentation

### Optimization History

| # | Topic | Link |
|---|---|---|
| 000 | Baseline pipeline profiling | [000_baseline_pipeline.md](docs/optimization_history/000_baseline_pipeline.md) |
| 010 | Model compression summary | [010_model_compression_summary.md](docs/optimization_history/010_model_compression_summary.md) |
| 011 | Structured pruning | [011_structured_pruning.md](docs/optimization_history/011_structured_pruning.md) |
| 012 | Finetuning pruned model | [012_finetuning_pruned_model.md](docs/optimization_history/012_finetuning_pruned_model.md) |
| 013 | ONNX export & TRT engine build | [013_onnx_export_and_engine_build.md](docs/optimization_history/013_onnx_export_and_engine_build.md) |
| 014 | Single-patch benchmarks | [014_single_patch_benchmarks.md](docs/optimization_history/014_single_patch_benchmarks.md) |
| 020 | TRT FP16 integration | [020_trt_fp16_integration.md](docs/optimization_history/020_trt_fp16_integration.md) |
| 030 | Pipeline I/O optimisation | [030_pipeline_io_optimization.md](docs/optimization_history/030_pipeline_io_optimization.md) |
| 050 | Sliding window bypass | [050_sliding_window_bypass.md](docs/optimization_history/050_sliding_window_bypass.md) |
| 056 | True in-memory detection | [056_true_inmemory_detection.md](docs/optimization_history/056_true_inmemory_detection.md) |
| 070 | SN009 formal benchmark | [070_sn009_formal_benchmark.md](docs/optimization_history/070_sn009_formal_benchmark.md) |
| 080 | TRT accuracy risk | [080_trt_accuracy_risk.md](docs/optimization_history/080_trt_accuracy_risk.md) |

### Other Documentation
- [Formal LaTeX Report](docs/formal_report/)
- [Project Overview](docs/project_overview.md)
- [Decision Log](docs/decision_log.md)
- [Experiment Registry](docs/experiment_registry.md)

---

## License

Research use only. Contact supervisors for licensing inquiries.
