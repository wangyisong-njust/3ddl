# 3D-IntelliScan

3D metrology and defect detection pipeline for semiconductor manufacturing. Processes NIfTI volumes through detection, segmentation, and metrology analysis.

## Usage

```bash
# Process files listed in files.txt
python main.py

# Process single file
python main.py /path/to/input.nii

# Force reprocess (ignore cache)
python main.py /path/to/input.nii --force

# List processed jobs
python main.py --list
```

## Input

- Copy `files.example.txt` to `files.txt` and add your NIfTI file paths (one per line)
- Or pass file path directly as argument

## Model Weights

There are two model weights needed: `detection_model.pt` and `segmentation_model.ckpt`. They are stored under `/nas/wangjie/wp5-model-weights`.

## Output

Default output directory: `output/`

```
output/
├── .pipeline_logbook.json    # Tracks processed files (skip duplicates)
└── SNxx/                     # Per-sample output folder
    ├── timing.log            # Processing time per stage
    ├── metrics.json          # Evaluation metrics
    ├── segmentation.nii.gz   # Full volume segmentation
    ├── bb3d.npy              # 3D bounding boxes
    ├── view1/, view2/        # 2D slices and detections
    ├── mmt/                  # Segmentation metadata and optional experiment traces
    │   ├── segmentation_regions.json  # Crop manifest for report reconstruction
    │   └── segmentation_stats.json    # Present on segmentation experiment runs
    └── metrology/
        ├── metrology.csv     # Measurement data
        └── metrology_report.pdf
```

## Project Structure

```
├── main.py          # Entry point - orchestrates full pipeline
├── detection.py     # YOLO-based 2D detection
├── merge.py         # 2D→3D bounding box generation
├── segmentation.py  # 3D segmentation inference
├── metrology.py     # Measurement computation (BLT, void ratio, etc.)
├── report.py        # PDF report generation (requires Claude API)
├── utils.py         # Logging, file ops, NII→JPG conversion
├── files.example.txt # Input file list template (copy to files.txt)
├── .env.example     # Environment config template (copy to .env)
├── models/          # Detection and segmentation model weights
└── pyproject.toml   # Dependencies
```

## Setup

It's recommended to use `uv` to maintain the project. It's required to have python>=3.12.
```bash
pip install -r pyproject.toml
# or with uv:
uv sync
```

**Models**: Place `detection_model.pt` and `segmentation_model.ckpt` in `models/`.

**Configuration**: Copy the example files before first run:
```bash
cp .env.example .env          # then fill in your CLAUDE_API_KEY
cp files.example.txt files.txt # then add your input file paths
```

**Report generation**: AI analysis requires a Claude API key in `.env`. Pass `--ai-analysis` to enable it. Without the flag, reports are generated without the AI summary page.

## Pipeline Stages

1. **NII→JPG** - Convert 3D volume to 2D slices (horizontal + vertical views)
2. **Detection** - YOLO inference on 2D slices
3. **3D BBox** - Merge 2D detections into 3D bounding boxes
4. **Segmentation** - Per-bbox 3D semantic segmentation
5. **Metrology** - Compute BLT, void ratio, pad misalignment, solder extrusion
6. **Report** - Generate PDF with visualizations and AI analysis

## Batch Throughput

For multi-sample throughput experiments, use `scripts/run_batch_multi_gpu.py`. It schedules independent `main.py` jobs across one or more GPUs without changing per-sample pipeline behavior.

## Configuration

Key options in `main.py` `PipelineConfig`:

- `output_base` - Output directory (default: `output`)
- `use_inmemory_detection` - Skip intermediate files for detection using the default quantized in-memory path; use `--file-detection` to fall back to the legacy JPG + detection txt workflow
- `detection_image_workers` - CPU worker count for in-memory slice preparation. The default is `4`, chosen from the validated `workers=1/4/8` sweep; override with `--detection-image-workers N`
- `detection_batch_prefetch` - Overlap next-batch image preparation with current YOLO inference. The default is ON; use `--no-detection-batch-prefetch` to disable it for A/B testing
- `use_combined_seg_metrology` - Process segmentation+metrology together (reduces I/O, default is ON)
- `clean_mask` - Apply morphological cleaning to masks (by default is OFF)
