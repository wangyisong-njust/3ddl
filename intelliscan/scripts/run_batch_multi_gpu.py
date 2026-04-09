#!/usr/bin/env python3
"""Run multiple intelliscan jobs across one or more GPUs and record throughput."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import PipelineLogbook  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="*", help="Input NIfTI files to process")
    parser.add_argument("--inputs-file", help="Optional text file with one input path per line")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU ids. If omitted, select the least-used GPUs.")
    parser.add_argument("--num-gpus", type=int, default=1, help="How many GPUs to select when --gpus is omitted")
    parser.add_argument("--output-base", required=True, help="Output base directory passed to main.py")
    parser.add_argument("--tag-prefix", required=True, help="Tag prefix appended with _g<gpu> for each launched job")
    parser.add_argument("--summary-json", required=True, help="Where to save the throughput summary JSON")
    parser.add_argument("--poll-seconds", type=float, default=0.5, help="Scheduler poll interval")
    parser.add_argument("--force", action="store_true", help="Pass --force to main.py")
    parser.add_argument("--quiet", action="store_true", help="Pass --quiet to main.py")
    return parser.parse_args()


def load_inputs(args: argparse.Namespace) -> list[str]:
    inputs: list[str] = []
    if args.inputs:
        inputs.extend(args.inputs)
    if args.inputs_file:
        with open(args.inputs_file, encoding="utf-8") as f:
            inputs.extend(line.strip() for line in f if line.strip())
    if not inputs:
        raise ValueError("No input files provided. Use --inputs or --inputs-file.")
    return [str(Path(item).resolve()) for item in inputs]


def select_gpus(args: argparse.Namespace) -> list[str]:
    if args.gpus:
        return [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[tuple[int, int]] = []
    for line in result.stdout.strip().splitlines():
        gpu_id, used_mem = [part.strip() for part in line.split(",")]
        rows.append((int(gpu_id), int(used_mem)))
    rows.sort(key=lambda item: (item[1], item[0]))
    selected = [str(gpu_id) for gpu_id, _ in rows[: max(1, args.num_gpus)]]
    if not selected:
        raise ValueError("Failed to select any GPUs.")
    return selected


def build_command(
    *,
    python_executable: str,
    main_script: Path,
    input_path: str,
    output_base: str,
    tag: str,
    force: bool,
    quiet: bool,
) -> list[str]:
    cmd = [python_executable, str(main_script), input_path, "--output", output_base, "--tag", tag]
    if force:
        cmd.append("--force")
    if quiet:
        cmd.append("--quiet")
    return cmd


def main() -> None:
    args = parse_args()
    inputs = load_inputs(args)
    gpus = select_gpus(args)

    output_base = str(Path(args.output_base).resolve())
    summary_path = Path(args.summary_json).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(output_base).resolve() / "batch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    pending = list(inputs)
    active: dict[str, dict] = {}
    jobs: list[dict] = []
    overall_start = time.time()

    def launch(gpu_id: str, input_path: str) -> None:
        sample_id = PipelineLogbook.extract_sample_id(input_path)
        tag = f"{args.tag_prefix}_g{gpu_id}"
        output_dir = str(Path(output_base) / f"{sample_id}_{tag}")
        log_path = log_dir / f"{sample_id}_{tag}.log"
        log_handle = open(log_path, "w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        cmd = build_command(
            python_executable=sys.executable,
            main_script=ROOT / "main.py",
            input_path=input_path,
            output_base=output_base,
            tag=tag,
            force=args.force,
            quiet=args.quiet,
        )
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        active[gpu_id] = {
            "process": proc,
            "input_path": input_path,
            "sample_id": sample_id,
            "gpu_id": gpu_id,
            "tag": tag,
            "output_dir": output_dir,
            "log_path": str(log_path),
            "started_at": time.time(),
            "log_handle": log_handle,
            "command": cmd,
        }

    while pending or active:
        for gpu_id in gpus:
            if gpu_id not in active and pending:
                launch(gpu_id, pending.pop(0))

        time.sleep(max(0.1, args.poll_seconds))

        completed: list[str] = []
        for gpu_id, job in active.items():
            proc = job["process"]
            return_code = proc.poll()
            if return_code is None:
                continue

            job["log_handle"].close()
            completed.append(gpu_id)
            ended_at = time.time()
            jobs.append(
                {
                    "input_path": job["input_path"],
                    "sample_id": job["sample_id"],
                    "gpu_id": job["gpu_id"],
                    "tag": job["tag"],
                    "output_dir": job["output_dir"],
                    "log_path": job["log_path"],
                    "command": job["command"],
                    "return_code": return_code,
                    "started_at_epoch": job["started_at"],
                    "ended_at_epoch": ended_at,
                    "elapsed_seconds": ended_at - job["started_at"],
                }
            )

        for gpu_id in completed:
            del active[gpu_id]

    overall_end = time.time()
    payload = {
        "inputs": inputs,
        "gpus": gpus,
        "output_base": output_base,
        "tag_prefix": args.tag_prefix,
        "job_count": len(jobs),
        "successful_job_count": sum(1 for job in jobs if job["return_code"] == 0),
        "failed_job_count": sum(1 for job in jobs if job["return_code"] != 0),
        "overall_started_at_epoch": overall_start,
        "overall_ended_at_epoch": overall_end,
        "overall_elapsed_seconds": overall_end - overall_start,
        "jobs": jobs,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
