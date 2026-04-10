#!/usr/bin/env python3
"""Run repeated benchmarks for statistical robustness.
Focus on 3 key variants × SN009 × 3 runs each:
1. Safe default (teacher/eager) - the retained engineering path
2. TRT FP16 (teacher) - the unsafe-but-fast reference
3. Best compressed student r=0.375/TRT FP16 - retained speed point

Outputs metrics.json for each run, then computes mean ± std.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = ROOT / "intelliscan" / "main.py"
VENV_PYTHON = Path("/home/kaixin/anaconda3/envs/intelliscan/bin/python")
OUTPUT_BASE = ROOT / "intelliscan" / "output_repeated_benchmark"

SAMPLE = "SN009"
SAMPLE_PATH = Path("/home/kaixin/datasets/3DDL-WP5-Data/RawScans/NewData/SN009_3D_May24/2_die_interposer_3Drecon_txm.nii")

N_RUNS = 3
GPU_INDEX = 2

VARIANTS = [
    {
        "key": "teacher_eager",
        "label": "Teacher / Eager (safe default)",
        "model": ROOT / "intelliscan" / "models" / "segmentation_model.ckpt",
        "extra_args": [],
    },
    {
        "key": "r0p375_trt_fp16",
        "label": "r=0.375 / TRT FP16 (retained speed)",
        "model": ROOT / "wp5-seg/runs/paper_fulltrain_kd_r0p375_e10_b2/best.ckpt",
        "extra_args": ["--trt", "--trt-engine",
                       str(ROOT / "wp5-seg/runs/paper_runtime_formal/r0p375_fp16.engine")],
    },
]


def run_one(variant: dict, run_idx: int) -> dict | None:
    """Run one pipeline invocation and return parsed metrics."""
    tag = f"{SAMPLE}_{variant['key']}_run{run_idx}"
    out_dir = OUTPUT_BASE / tag

    cmd = [
        str(VENV_PYTHON), str(MAIN_PY),
        str(SAMPLE_PATH),
        "--output", str(OUTPUT_BASE),
        "--tag", tag,
        "--segmentation-model", str(variant["model"]),
        "--force",
        *variant["extra_args"],
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)

    print(f"\n{'='*60}")
    print(f"  {variant['label']}  run {run_idx}/{N_RUNS}")
    print(f"  tag: {tag}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT / "intelliscan"),
                           env=env, capture_output=True, text=True)
    wall = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode})")
        print(result.stderr[-500:] if result.stderr else "no stderr")
        return None

    # Parse metrics.json
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"  No metrics.json found at {metrics_path}")
        return None

    with open(metrics_path) as f:
        metrics = json.load(f)

    total = metrics.get("total_elapsed", wall)
    phases = {p["name"]: p["elapsed_seconds"] for p in metrics.get("phases", [])}

    print(f"  Total: {total:.2f}s  Wall: {wall:.2f}s")
    for name, t in phases.items():
        print(f"    {name}: {t:.2f}s")

    return {"run": run_idx, "total": total, "wall": wall, "phases": phases}


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for variant in VARIANTS:
        # Check if TRT engine exists for TRT variants
        if "--trt-engine" in variant.get("extra_args", []):
            idx = variant["extra_args"].index("--trt-engine")
            engine_path = Path(variant["extra_args"][idx + 1])
            if not engine_path.exists():
                print(f"\nSKIPPING {variant['label']}: engine not found at {engine_path}")
                continue

        # Check if model exists
        if not Path(variant["model"]).exists():
            print(f"\nSKIPPING {variant['label']}: model not found at {variant['model']}")
            continue

        runs = []
        for i in range(1, N_RUNS + 1):
            result = run_one(variant, i)
            if result:
                runs.append(result)

        if runs:
            all_results[variant["key"]] = {
                "label": variant["label"],
                "runs": runs,
            }

    # Compute summary statistics
    summary = {}
    for key, data in all_results.items():
        totals = [r["total"] for r in data["runs"]]
        import numpy as np
        arr = np.array(totals)
        summary[key] = {
            "label": data["label"],
            "n_runs": len(totals),
            "total_mean": float(np.mean(arr)),
            "total_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "total_min": float(np.min(arr)),
            "total_max": float(np.max(arr)),
            "runs": data["runs"],
        }

        # Per-phase stats
        all_phases = {}
        for r in data["runs"]:
            for pname, ptime in r["phases"].items():
                all_phases.setdefault(pname, []).append(ptime)
        phase_stats = {}
        for pname, times in all_phases.items():
            a = np.array(times)
            phase_stats[pname] = {
                "mean": float(np.mean(a)),
                "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
            }
        summary[key]["phase_stats"] = phase_stats

    # Save summary
    summary_path = OUTPUT_BASE / "repeated_benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print("  REPEATED BENCHMARK SUMMARY")
    print(f"{'='*70}")
    for key, s in summary.items():
        print(f"\n  {s['label']}  (n={s['n_runs']})")
        print(f"    Total: {s['total_mean']:.2f} ± {s['total_std']:.2f}s  "
              f"[{s['total_min']:.2f}, {s['total_max']:.2f}]")
        for pname, ps in s.get("phase_stats", {}).items():
            print(f"    {pname}: {ps['mean']:.2f} ± {ps['std']:.2f}s")

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
