#!/usr/bin/env python3
"""Benchmark retained runtime candidates inside the full intelliscan pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from compare_ablation_outputs import compare_variant


ROOT = Path(__file__).resolve().parents[2]
INTELLISCAN_DIR = ROOT / "intelliscan"
MAIN_PY = INTELLISCAN_DIR / "main.py"

SAMPLE_PATHS = {
    "SN002": Path("/home/kaixin/datasets/3DDL-WP5-Data/RawScans/NewData/SN002_3D_Feb24/4_die_stack_0.75um_3Drecon_txm.nii"),
    "SN009": Path("/home/kaixin/datasets/3DDL-WP5-Data/RawScans/NewData/SN009_3D_May24/2_die_interposer_3Drecon_txm.nii"),
    "SN010": Path("/home/kaixin/datasets/3DDL-WP5-Data/RawScans/NewData/SN010_3D_May24/Die_on_Interposer_3Drecon_txm.nii"),
}


VARIANTS = [
    {
        "key": "teacher_eager",
        "label": "teacher / eager",
        "status": "reference",
        "compression": "teacher",
        "backend": "pytorch_eager",
        "params": "teacher ckpt",
        "segmentation_model": ROOT / "intelliscan/models/segmentation_model.ckpt",
        "extra_args": [],
        "patch_latency_ms": 17.401992017403245,
        "patch_energy_j": 2.522571062498514,
    },
    {
        "key": "r0p375_eager",
        "label": "r=0.375 / eager",
        "status": "context_only",
        "compression": "pruned_r0.375",
        "backend": "pytorch_eager",
        "params": "KD c34=2,2",
        "segmentation_model": ROOT / "wp5-seg/runs/paper_fulltrain_kd_r0p375_e10_b2/best.ckpt",
        "extra_args": [],
        "patch_latency_ms": 16.468214336782694,
        "patch_energy_j": 3.2719128980937198,
    },
    {
        "key": "r0p375_trt_fp16",
        "label": "r=0.375 / TRT FP16",
        "status": "retained",
        "compression": "pruned_r0.375",
        "backend": "trt_fp16",
        "params": "KD c34=2,2",
        "segmentation_model": ROOT / "wp5-seg/runs/paper_fulltrain_kd_r0p375_e10_b2/best.ckpt",
        "extra_args": ["--trt", "--trt-engine", str(ROOT / "wp5-seg/runs/paper_runtime_formal/r0p375_fp16.engine")],
        "patch_latency_ms": 3.1533921137452126,
        "patch_energy_j": 0.3281508188322018,
    },
    {
        "key": "r0p375_trt_int8_fit",
        "label": "r=0.375 / TRT INT8 fit-only",
        "status": "validated_with_caveat",
        "compression": "pruned_r0.375",
        "backend": "trt_int8_fitonly64",
        "params": "KD c34=2,2; fit-only calibration",
        "segmentation_model": ROOT / "wp5-seg/runs/paper_fulltrain_kd_r0p375_e10_b2/best.ckpt",
        "extra_args": ["--trt", "--trt-engine", str(ROOT / "wp5-seg/runs/paper_runtime_formal/r0p375_int8_fitonly64.engine")],
        "patch_latency_ms": 2.7873074635863304,
        "patch_energy_j": 0.3234567436938752,
    },
    {
        "key": "r0p5_eager",
        "label": "r=0.5 / eager",
        "status": "context_only",
        "compression": "pruned_r0.5",
        "backend": "pytorch_eager",
        "params": "KD c34=2,2",
        "segmentation_model": ROOT / "wp5-seg/runs/paper_fulltrain_kd_c34_2_2_e10_b2/best.ckpt",
        "extra_args": [],
        "patch_latency_ms": 8.868635236285627,
        "patch_energy_j": 1.0756934575071604,
    },
    {
        "key": "r0p5_compile",
        "label": "r=0.5 / compile",
        "status": "retained",
        "compression": "pruned_r0.5",
        "backend": "pytorch_compile_reduce-overhead",
        "params": "KD c34=2,2; compile=reduce-overhead",
        "segmentation_model": ROOT / "wp5-seg/runs/paper_fulltrain_kd_c34_2_2_e10_b2/best.ckpt",
        "extra_args": ["--compile", "--compile-mode", "reduce-overhead"],
        "patch_latency_ms": 4.796485742554069,
        "patch_energy_j": 0.5386229046829332,
    },
    {
        "key": "r0p5_trt_fp16",
        "label": "r=0.5 / TRT FP16",
        "status": "retained",
        "compression": "pruned_r0.5",
        "backend": "trt_fp16",
        "params": "KD c34=2,2",
        "segmentation_model": ROOT / "wp5-seg/runs/paper_fulltrain_kd_c34_2_2_e10_b2/best.ckpt",
        "extra_args": ["--trt", "--trt-engine", str(ROOT / "wp5-seg/runs/paper_runtime_formal/r0p5_fp16.engine")],
        "patch_latency_ms": 1.6205270076170564,
        "patch_energy_j": 0.15916997601779084,
    },
]


class PowerMonitor:
    def __init__(self, gpu_index: int, poll_ms: int):
        self.gpu_index = gpu_index
        self.poll_s = poll_ms / 1000.0
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _query(self) -> dict[str, float] | None:
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=power.draw,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        try:
            raw = subprocess.check_output(cmd, text=True).strip()
        except subprocess.SubprocessError:
            return None
        if not raw:
            return None
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 3:
            return None
        try:
            return {
                "power_w": float(parts[0]),
                "utilization_pct": float(parts[1]),
                "memory_used_mb": float(parts[2]),
            }
        except ValueError:
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self._query()
            if sample is not None:
                self.samples.append(sample)
            self._stop.wait(self.poll_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


def choose_gpu() -> int:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,power.draw",
        "--format=csv,noheader,nounits",
    ]
    rows = subprocess.check_output(cmd, text=True).strip().splitlines()
    parsed: list[tuple[int, float, float]] = []
    for row in rows:
        index, memory_used, power_draw = [part.strip() for part in row.split(",")]
        parsed.append((int(index), float(memory_used), float(power_draw)))
    parsed.sort(key=lambda item: (item[1], item[2], item[0]))
    return parsed[0][0]


def sample_idle_power(gpu_index: int, poll_ms: int, samples: int = 5) -> float:
    values: list[float] = []
    for _ in range(samples):
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        values.append(float(raw))
        time.sleep(poll_ms / 1000.0)
    return sum(values) / len(values)


def load_metrics(output_dir: Path) -> dict:
    with open(output_dir / "metrics.json") as handle:
        return json.load(handle)


def phase_elapsed(metrics: dict, phase_name: str) -> float | None:
    for phase in metrics["phases"]:
        if phase["name"] == phase_name:
            return phase.get("elapsed") or phase.get("elapsed_seconds")
    return None


def summarize_energy(samples: list[dict[str, float]], wall_time_s: float, idle_power_w: float) -> dict[str, float | int]:
    if not samples:
        return {
            "sample_count": 0,
            "avg_power_w": 0.0,
            "max_power_w": 0.0,
            "avg_utilization_pct": 0.0,
            "avg_memory_used_mb": 0.0,
            "wall_time_s": wall_time_s,
            "energy_j": 0.0,
            "idle_power_w": idle_power_w,
            "dynamic_energy_j": 0.0,
        }

    avg_power = sum(sample["power_w"] for sample in samples) / len(samples)
    avg_util = sum(sample["utilization_pct"] for sample in samples) / len(samples)
    avg_mem = sum(sample["memory_used_mb"] for sample in samples) / len(samples)
    energy_j = avg_power * wall_time_s
    dynamic_energy_j = max(0.0, (avg_power - idle_power_w) * wall_time_s)
    return {
        "sample_count": len(samples),
        "avg_power_w": avg_power,
        "max_power_w": max(sample["power_w"] for sample in samples),
        "avg_utilization_pct": avg_util,
        "avg_memory_used_mb": avg_mem,
        "wall_time_s": wall_time_s,
        "energy_j": energy_j,
        "idle_power_w": idle_power_w,
        "dynamic_energy_j": dynamic_energy_j,
    }


def build_command(
    python_exe: str,
    input_file: Path,
    output_base: Path,
    tag: str,
    segmentation_model: Path,
    extra_args: list[str],
) -> list[str]:
    return [
        python_exe,
        str(MAIN_PY),
        str(input_file),
        "--force",
        "--quiet",
        "--output",
        str(output_base),
        "--tag",
        tag,
        "--segmentation-model",
        str(segmentation_model),
        *extra_args,
    ]


def output_dir_for(sample_name: str, output_base: Path, tag: str) -> Path:
    return output_base / f"{sample_name}_{tag}"


def aggregate_variant_row(variant_key: str, sample_payload: dict[str, dict], teacher_key: str = "teacher_eager") -> dict[str, object]:
    teacher_totals = []
    teacher_seg = []
    totals = []
    seg_times = []
    energies = []
    powers = []
    voxel = []
    c3 = []
    c4 = []
    total_items = 0
    pad_flips = 0
    solder_flips = 0
    weighted_blt_sum = 0.0
    weighted_pad_sum = 0.0
    patch_latency = None
    patch_energy = None
    status = None
    label = None
    compression = None
    backend = None
    params = None

    for sample_name, payload in sample_payload.items():
        variant = payload["variants"][variant_key]
        teacher = payload["variants"][teacher_key]
        teacher_totals.append(teacher["total_elapsed"])
        teacher_seg.append(teacher["segmentation_metrology_elapsed"])
        totals.append(variant["total_elapsed"])
        seg_times.append(variant["segmentation_metrology_elapsed"])
        energies.append(variant["power"]["energy_j"])
        powers.append(variant["power"]["avg_power_w"])
        patch_latency = variant["patch_latency_ms"]
        patch_energy = variant["patch_energy_j"]
        status = variant["status"]
        label = variant["label"]
        compression = variant["compression"]
        backend = variant["backend"]
        params = variant["params"]
        comparison = payload.get("comparisons_vs_teacher", {}).get(variant_key)
        if comparison:
            voxel.append(comparison["voxel_agreement"])
            c3.append(comparison["class_dice"]["3"])
            c4.append(comparison["class_dice"]["4"])
            total_items += comparison["metrology"]["count"]
            pad_flips += comparison["metrology"]["pad_flip_count"]
            solder_flips += comparison["metrology"]["solder_flip_count"]
            weighted_blt_sum += comparison["metrology"]["blt_mean_abs_delta"] * comparison["metrology"]["count"]
            weighted_pad_sum += (
                comparison["metrology"]["pad_misalignment_mean_abs_delta"] * comparison["metrology"]["count"]
            )

    teacher_total_sum = sum(teacher_totals)
    teacher_seg_sum = sum(teacher_seg)
    total_sum = sum(totals)
    seg_sum = sum(seg_times)
    result = {
        "label": label,
        "status": status,
        "compression": compression,
        "backend": backend,
        "params": params,
        "samples": len(totals),
        "patch_inference_ms": patch_latency,
        "patch_energy_j_per_iter": patch_energy,
        "mean_total_latency_s": sum(totals) / len(totals),
        "mean_seg_met_latency_s": sum(seg_times) / len(seg_times),
        "pooled_total_speedup_vs_teacher_pct": 100.0 * (teacher_total_sum - total_sum) / teacher_total_sum,
        "pooled_seg_met_speedup_vs_teacher_pct": 100.0 * (teacher_seg_sum - seg_sum) / teacher_seg_sum,
        "mean_pipeline_energy_j": sum(energies) / len(energies),
        "mean_pipeline_power_w": sum(powers) / len(powers),
    }
    if voxel:
        result.update(
            {
                "mean_voxel_agreement_vs_teacher": sum(voxel) / len(voxel),
                "mean_class3_dice_vs_teacher": sum(c3) / len(c3),
                "mean_class4_dice_vs_teacher": sum(c4) / len(c4),
                "pad_flip_count": pad_flips,
                "pad_flip_rate": pad_flips / total_items,
                "solder_flip_count": solder_flips,
                "solder_flip_rate": solder_flips / total_items,
                "blt_mean_abs_delta": weighted_blt_sum / total_items,
                "pad_misalignment_mean_abs_delta": weighted_pad_sum / total_items,
                "total_bumps_compared": total_items,
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable to use for intelliscan runs",
    )
    parser.add_argument(
        "--output-base",
        default=str(INTELLISCAN_DIR / "output_runtime_rank"),
        help="Where to write deployment benchmark outputs",
    )
    parser.add_argument(
        "--summary-json",
        default=str(INTELLISCAN_DIR / "output_runtime_rank/analysis/runtime_candidate_ranking_summary.json"),
        help="Where to save the final summary JSON",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=["SN002", "SN009", "SN010"],
        choices=sorted(SAMPLE_PATHS.keys()),
        help="Sample panel to benchmark",
    )
    parser.add_argument("--gpu-index", type=int, default=None, help="Physical GPU index for both execution and power sampling")
    parser.add_argument("--poll-ms", type=int, default=200, help="Power sampling interval in milliseconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_base = Path(args.output_base).resolve()
    summary_json = Path(args.summary_json).resolve()
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_base.mkdir(parents=True, exist_ok=True)

    gpu_index = args.gpu_index if args.gpu_index is not None else choose_gpu()
    idle_power_w = sample_idle_power(gpu_index, args.poll_ms)

    payload: dict[str, object] = {
        "date": time.strftime("%Y-%m-%d"),
        "gpu_index": gpu_index,
        "idle_power_w": idle_power_w,
        "variants": {variant["key"]: {k: v for k, v in variant.items() if k not in {"segmentation_model", "extra_args"}} for variant in VARIANTS},
        "samples": {},
    }

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    for sample_name in args.samples:
        input_file = SAMPLE_PATHS[sample_name]
        sample_result = {"input_file": str(input_file), "variants": {}, "comparisons_vs_teacher": {}}
        for variant in VARIANTS:
            tag = variant["key"]
            out_dir = output_dir_for(sample_name, output_base, tag)
            cmd = build_command(
                python_exe=args.python_exe,
                input_file=input_file,
                output_base=output_base,
                tag=tag,
                segmentation_model=variant["segmentation_model"],
                extra_args=variant["extra_args"],
            )
            start = time.perf_counter()
            monitor = PowerMonitor(gpu_index=gpu_index, poll_ms=args.poll_ms)
            monitor.start()
            completed = subprocess.run(
                cmd,
                cwd=INTELLISCAN_DIR,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            monitor.stop()
            wall_time_s = time.perf_counter() - start
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Variant {variant['key']} failed on {sample_name} with code {completed.returncode}\n{completed.stdout}"
                )
            (out_dir / "runner_stdout.log").write_text(completed.stdout, encoding="utf-8")
            metrics = load_metrics(out_dir)
            variant_result = {
                "path": str(out_dir),
                "label": variant["label"],
                "status": variant["status"],
                "compression": variant["compression"],
                "backend": variant["backend"],
                "params": variant["params"],
                "patch_latency_ms": variant["patch_latency_ms"],
                "patch_energy_j": variant["patch_energy_j"],
                "wall_clock_s": wall_time_s,
                "total_elapsed": metrics["total_elapsed"],
                "detection_elapsed": phase_elapsed(metrics, "2D Detection Inference"),
                "segmentation_metrology_elapsed": phase_elapsed(metrics, "3D Segmentation + Metrology"),
                "report_elapsed": phase_elapsed(metrics, "Report Generation"),
                "power": summarize_energy(monitor.samples, wall_time_s, idle_power_w),
            }
            sample_result["variants"][variant["key"]] = variant_result

        teacher_dir = Path(sample_result["variants"]["teacher_eager"]["path"])
        for variant in VARIANTS:
            if variant["key"] == "teacher_eager":
                continue
            candidate_dir = Path(sample_result["variants"][variant["key"]]["path"])
            sample_result["comparisons_vs_teacher"][variant["key"]] = compare_variant(
                teacher_dir,
                candidate_dir,
                [0, 1, 2, 3, 4],
            )
        payload["samples"][sample_name] = sample_result

    aggregate_rows = [
        aggregate_variant_row(variant["key"], payload["samples"])
        for variant in VARIANTS
    ]
    payload["aggregate_table"] = aggregate_rows
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(summary_json)


if __name__ == "__main__":
    main()
