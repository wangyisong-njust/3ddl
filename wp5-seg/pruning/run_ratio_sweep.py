#!/usr/bin/env python3
"""
Run a pruning-ratio screening sweep for BasicUNet.

This script is intentionally narrow:
- prune a set of ratios from the same teacher checkpoint
- benchmark patch latency for each pruned checkpoint
- collect parameter counts / features / benchmark metrics into one summary json

It does not run finetuning. The goal is to screen promising deployment candidates
before spending full-train recovery time.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pruning ratio screening sweep")
    parser.add_argument("--model_path", type=str, required=True, help="Teacher checkpoint path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for sweep artifacts")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        required=True,
        help="Pruning ratios to evaluate, e.g. 0.25 0.375 0.5 0.625",
    )
    parser.add_argument("--num_runs", type=int, default=50, help="Benchmark runs per candidate")
    parser.add_argument("--warmup", type=int, default=10, help="Benchmark warmup runs")
    parser.add_argument("--gpu", type=str, default=None, help="Optional CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--amp", action="store_true", help="Also run AMP benchmark")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    summary: list[dict[str, object]] = []

    for ratio in args.ratios:
        ratio_tag = str(ratio).replace(".", "p")
        pruned_path = out_dir / f"pruned_r{ratio_tag}.ckpt"
        info_path = pruned_path.with_suffix(".json")
        bench_path = out_dir / f"benchmark_r{ratio_tag}.json"

        run(
            [
                sys.executable,
                str(script_dir / "prune_basicunet.py"),
                "--model_path",
                args.model_path,
                "--pruning_ratio",
                str(ratio),
                "--output_path",
                str(pruned_path),
            ],
            env=env,
        )

        bench_cmd = [
            sys.executable,
            str(script_dir / "benchmark.py"),
            "--model_path",
            args.model_path,
            "--model_format",
            "state_dict",
            "--compare_path",
            str(pruned_path),
            "--compare_format",
            "pruned",
            "--num_runs",
            str(args.num_runs),
            "--warmup",
            str(args.warmup),
            "--output",
            str(bench_path),
        ]
        if args.amp:
            bench_cmd.append("--amp")
        run(bench_cmd, env=env)

        pruning_info = json.loads(info_path.read_text())
        bench_info = json.loads(bench_path.read_text())
        row = {
            "pruning_ratio": ratio,
            "pruned_model_path": str(pruned_path),
            "pruning_info_path": str(info_path),
            "benchmark_path": str(bench_path),
            "original_params": pruning_info["original_params"],
            "pruned_params": pruning_info["pruned_params"],
            "reduction_pct": pruning_info["reduction_pct"],
            "pruned_features": pruning_info["pruned_features"],
            "fp32_mean_ms": bench_info["model_2"]["fp32"]["mean_ms"],
            "fp32_speedup_vs_teacher": bench_info["speedup_fp32"],
        }
        if args.amp and "amp" in bench_info["model_2"]:
            row["amp_mean_ms"] = bench_info["model_2"]["amp"]["mean_ms"]
            row["amp_mean_ms_teacher"] = bench_info["model_1"]["amp"]["mean_ms"]
            row["amp_speedup_vs_teacher"] = (
                bench_info["model_1"]["amp"]["mean_ms"] / bench_info["model_2"]["amp"]["mean_ms"]
            )
        summary.append(row)

    summary_path = out_dir / "ratio_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
