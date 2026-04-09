# Repository Guidelines

## Project Structure & Module Organization
This workspace contains two Python 3.12 projects, each with its own Git history. Run Git commands from the relevant subdirectory, not the workspace root.

- `intelliscan/`: inference pipeline. `main.py` orchestrates detection, 3D box merging, segmentation, metrology, GLTF export, and PDF reporting. Shared helpers live in `utils.py`; model files are expected under `models/`.
- `wp5-seg/`: training, evaluation, and optimization for the segmentation model used by `intelliscan/`. Core entry points are `train.py`, `eval.py`, and scripts under `pruning/`.
- `wp5-seg/3ddl-dataset/`: dataset submodule expected by training code.
- `wp5-seg/reports/`: performance write-ups and benchmark notes.

## Build, Test, and Development Commands
Use `uv` where possible; both projects ship `uv.lock`.

- `cd intelliscan && uv sync`: install pipeline dependencies.
- `cd intelliscan && python main.py --list`: inspect cached jobs.
- `cd intelliscan && python main.py /path/to/input.nii --force`: run a pipeline smoke test.
- `cd intelliscan && uv run ruff check .`: run the configured linter.
- `cd wp5-seg && uv sync`: install training dependencies.
- `cd wp5-seg && bash run.sh`: launch baseline training with the checked-in defaults.
- `cd wp5-seg && bash run_eval.sh`: evaluate a trained checkpoint.
- `cd wp5-seg && bash pruning/run_pruning_pipeline.sh`: run pruning, fine-tuning, and benchmarking.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, snake_case for modules/functions, PascalCase for classes, and explicit type hints where practical. `intelliscan/pyproject.toml` sets Ruff with a 120-character line limit; use that standard across new files in both projects. Keep CLI wiring in top-level scripts and push reusable logic into importable modules.

## Testing Guidelines
There is no committed `tests/` package yet, so validation is workflow-based. For `intelliscan`, run a representative NIfTI through `main.py` and verify generated artifacts under `output/<sample>/`. For `wp5-seg`, rerun `eval.py` or the relevant pruning benchmark and record Dice/IoU or latency deltas. If you add automated tests, place them under `tests/test_<module>.py`.

## Commit & Pull Request Guidelines
Match the current history: short, imperative commit subjects such as `add TRT backend...` or `Add structured pruning...`. Keep each commit focused on one change. PRs should state which subproject was modified, list commands run, note any dataset/model assumptions, and include metrics, logs, or screenshots when reports, benchmarks, or visual outputs change.

## Security & Configuration Tips
Do not commit `.env`, local dataset paths, model weights, or generated `output/` artifacts. Start from `intelliscan/.env.example`, keep API keys local, and document any required external paths in the PR description.
