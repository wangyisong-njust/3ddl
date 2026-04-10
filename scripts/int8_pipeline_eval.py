#!/usr/bin/env python3
"""Run INT8 real-data calibrated engines through whole pipeline on SN009."""
import json, os, subprocess, sys, time, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/kaixin/anaconda3/envs/intelliscan/bin/python")
MAIN_PY = ROOT / "intelliscan" / "main.py"
OUTPUT = ROOT / "intelliscan" / "output_int8_eval"
SAMPLE = Path("/home/kaixin/datasets/3DDL-WP5-Data/RawScans/NewData/SN009_3D_May24/2_die_interposer_3Drecon_txm.nii")
GPU = 2

VARIANTS = [
    {"key": "r0p375_int8_real", "label": "r=0.375 / TRT INT8 (real calib)",
     "model": ROOT / "wp5-seg/runs/paper_fulltrain_kd_r0p375_e10_b2/best.ckpt",
     "engine": ROOT / "wp5-seg/runs/paper_runtime_formal/r0p375_int8_real.engine"},
    {"key": "r0p5_int8_real", "label": "r=0.5 / TRT INT8 (real calib)",
     "model": ROOT / "wp5-seg/runs/paper_fulltrain_kd_r0p375_e10_b2/best.ckpt",  # will use engine
     "engine": ROOT / "wp5-seg/runs/paper_runtime_formal/r0p5_int8_real.engine"},
    {"key": "teacher_int8_real", "label": "Teacher / TRT INT8 (real calib)",
     "model": ROOT / "intelliscan/models/segmentation_model.ckpt",
     "engine": None},  # Need to build first
]

OUTPUT.mkdir(parents=True, exist_ok=True)
env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(GPU)

for v in VARIANTS:
    if v["engine"] is None or not Path(v["engine"]).exists():
        print(f"SKIP {v['label']}: engine not found")
        continue
    tag = f"SN009_{v['key']}"
    cmd = [str(PYTHON), str(MAIN_PY), str(SAMPLE), "--output", str(OUTPUT),
           "--tag", tag, "--segmentation-model", str(v["model"]),
           "--trt", "--trt-engine", str(v["engine"]), "--force"]
    print(f"\n{'='*60}\n  {v['label']}\n{'='*60}")
    r = subprocess.run(cmd, cwd=str(ROOT/"intelliscan"), env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED: {r.stderr[-500:]}")
    else:
        # Find metrics
        for prefix in [f"SN009_{tag}", tag]:
            mp = OUTPUT / prefix / "metrics.json"
            if mp.exists(): break
        if mp.exists():
            m = json.load(open(mp))
            seg = next((p['elapsed_seconds'] for p in m['phases'] if 'Seg' in p['name']), 0)
            print(f"  Total: {m['total_elapsed']:.2f}s  Seg+Met: {seg:.2f}s")
        else:
            print(f"  metrics.json not found")

print("\nDone. Now run compare_ablation_outputs.py to get quality metrics.")
