#!/usr/bin/env python3
"""Run teacher FP16 vs r=0.375 FP16 fair comparison on SN009, 3 runs each."""
import json, os, subprocess, sys, time, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/kaixin/anaconda3/envs/intelliscan/bin/python")
MAIN_PY = ROOT / "intelliscan" / "main.py"
OUTPUT = ROOT / "intelliscan" / "output_fp16_comparison"
SAMPLE = Path("/home/kaixin/datasets/3DDL-WP5-Data/RawScans/NewData/SN009_3D_May24/2_die_interposer_3Drecon_txm.nii")
N = 3; GPU = 2

VARIANTS = [
    {"key": "teacher_trt_fp16", "label": "Teacher / TRT FP16",
     "model": ROOT / "intelliscan/models/segmentation_model.ckpt",
     "extra": ["--trt", "--trt-engine", str(ROOT / "wp5-seg/runs/paper_runtime_formal/teacher_fp16.engine")]},
    {"key": "r0p375_trt_fp16", "label": "r=0.375 / TRT FP16",
     "model": ROOT / "wp5-seg/runs/paper_fulltrain_kd_r0p375_e10_b2/best.ckpt",
     "extra": ["--trt", "--trt-engine", str(ROOT / "wp5-seg/runs/paper_runtime_formal/r0p375_fp16.engine")]},
]

OUTPUT.mkdir(parents=True, exist_ok=True)
env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(GPU)
results = {}

for v in VARIANTS:
    runs = []
    for i in range(1, N+1):
        tag = f"SN009_{v['key']}_run{i}"
        cmd = [str(PYTHON), str(MAIN_PY), str(SAMPLE), "--output", str(OUTPUT),
               "--tag", tag, "--segmentation-model", str(v["model"]), "--force", *v["extra"]]
        print(f"\n{'='*60}\n  {v['label']}  run {i}/{N}\n{'='*60}")
        r = subprocess.run(cmd, cwd=str(ROOT/"intelliscan"), env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAILED: {r.stderr[-300:]}")
            continue
        # Find metrics - handle double-prefix
        for prefix in [f"SN009_{tag}", tag]:
            mp = OUTPUT / prefix / "metrics.json"
            if mp.exists(): break
        if not mp.exists():
            mp = OUTPUT / tag / "metrics.json"
        if mp.exists():
            m = json.load(open(mp))
            t = m["total_elapsed"]
            seg = next((p["elapsed_seconds"] for p in m["phases"] if "Seg" in p["name"]), 0)
            print(f"  Total: {t:.2f}s  Seg+Met: {seg:.2f}s")
            runs.append({"run": i, "total": t, "seg_met": seg})
        else:
            print(f"  metrics.json not found")
    if runs:
        totals = np.array([r["total"] for r in runs])
        segs = np.array([r["seg_met"] for r in runs])
        results[v["key"]] = {"label": v["label"], "n": len(runs),
            "total_mean": float(np.mean(totals)), "total_std": float(np.std(totals, ddof=1)) if len(totals)>1 else 0,
            "seg_mean": float(np.mean(segs)), "seg_std": float(np.std(segs, ddof=1)) if len(segs)>1 else 0,
            "runs": runs}

json.dump(results, open(OUTPUT/"fp16_comparison_summary.json","w"), indent=2)
print(f"\n{'='*60}\n  FP16 COMPARISON SUMMARY\n{'='*60}")
for k,s in results.items():
    print(f"  {s['label']} (n={s['n']}): Total {s['total_mean']:.2f}±{s['total_std']:.2f}s  Seg+Met {s['seg_mean']:.2f}±{s['seg_std']:.2f}s")
