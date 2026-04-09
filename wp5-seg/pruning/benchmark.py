#!/usr/bin/env python3
"""
Benchmark inference speed for original and pruned MONAI BasicUNet models.

Measures FP32 and AMP (mixed precision) latency with proper CUDA synchronization.
Adapted from the VNet pytorch_amp_benchmark.py.

Usage:
  # Benchmark original model (state_dict format)
  python benchmark.py --model_path best.ckpt --model_format state_dict

  # Benchmark pruned model (has features + state_dict)
  python benchmark.py --model_path pruned.ckpt --model_format pruned

  # Compare two models
  python benchmark.py --model_path best.ckpt --model_format state_dict \\
    --compare_path pruned.ckpt --compare_format pruned
"""

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from runtime_support import (
    NvidiaSMIPowerMonitor,
    load_basicunet_model,
    measure_idle_power,
    summarize_power_samples,
)


def benchmark_model(
    model,
    device,
    input_shape=(1, 1, 112, 112, 80),
    num_runs=100,
    warmup_runs=20,
    use_amp=False,
    compile_mode: str | None = None,
    gpu_index: int | None = None,
    power_sample_ms: int = 100,
):
    """
    Benchmark inference latency with proper CUDA synchronization.

    Returns dict with mean, std, min, max, median latency in ms.
    """
    dummy_input = torch.randn(*input_shape, device=device)
    run_model = model
    mode_label = "AMP" if use_amp else "FP32"
    if compile_mode:
        run_model = torch.compile(model, mode=compile_mode)
        mode_label = f"compile:{compile_mode}+{'AMP' if use_amp else 'FP32'}"
    make_ctx = (lambda: torch.amp.autocast("cuda")) if use_amp and device.type == "cuda" else nullcontext

    # First run captures compile overhead if compile is enabled.
    with torch.no_grad():
        with make_ctx():
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = run_model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            first_run_ms = (time.perf_counter() - t0) * 1000.0

            for _ in range(max(0, warmup_runs - 1)):
                _ = run_model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    idle_power_w = None
    if gpu_index is not None and device.type == "cuda":
        idle_power_w = measure_idle_power(gpu_index=gpu_index, sample_ms=power_sample_ms)

    # Timed runs
    latencies = []
    wall_t0 = time.perf_counter()
    monitor = NvidiaSMIPowerMonitor(gpu_index=gpu_index, sample_ms=power_sample_ms) if gpu_index is not None and device.type == "cuda" else None
    if monitor is not None:
        monitor.start()
    try:
        with torch.no_grad():
            with make_ctx():
                for _ in range(num_runs):
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _ = run_model(dummy_input)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    latencies.append((time.perf_counter() - t0) * 1000)
    finally:
        wall_time_s = time.perf_counter() - wall_t0
        if monitor is not None:
            monitor.stop()

    power = summarize_power_samples(
        samples=(monitor.samples if monitor is not None else []),
        wall_time_s=wall_time_s,
        iterations=num_runs,
        idle_power_w=idle_power_w,
    )
    return {
        "precision": mode_label,
        "compile_mode": compile_mode,
        "first_run_ms": float(first_run_ms),
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "median_ms": float(np.median(latencies)),
        "throughput_fps": 1000.0 / float(np.mean(latencies)),
        "power": power.as_dict(),
    }


def print_results(name, entries: list[tuple[str, dict]]):
    """Pretty-print benchmark results."""
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    print(f"  {'Mode':<30} {'Mean':>10} {'First':>10} {'Median':>10} {'FPS':>10} {'Power':>10}")
    print(f"  {'-' * 78}")
    baseline_ms = entries[0][1]["mean_ms"]
    for label, result in entries:
        power_text = "-"
        avg_power = result.get("power", {}).get("avg_power_w")
        if avg_power is not None:
            power_text = f"{avg_power:.1f}W"
        speedup = baseline_ms / result["mean_ms"]
        print(
            f"  {label:<30} {result['mean_ms']:>9.2f}ms {result['first_run_ms']:>9.2f}ms "
            f"{result['median_ms']:>9.2f}ms {result['throughput_fps']:>9.1f} {power_text:>10}"
            f"  ({speedup:.2f}x)"
        )
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MONAI BasicUNet inference speed")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_format", type=str, default="state_dict",
                        choices=["state_dict", "pruned"],
                        help="Model format: 'state_dict' for original, 'pruned' for pruned")
    parser.add_argument("--compare_path", type=str, default=None,
                        help="Second model to compare against")
    parser.add_argument("--compare_format", type=str, default="pruned",
                        choices=["state_dict", "pruned"])
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--amp", action="store_true", help="Also benchmark AMP mode")
    parser.add_argument("--compile-mode", type=str, default=None,
                        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                        help="Optional torch.compile mode for steady-state inference benchmarking")
    parser.add_argument("--benchmark-compile-amp", action="store_true",
                        help="When compile mode is enabled, also benchmark compile+AMP")
    parser.add_argument("--gpu-index", type=int, default=None,
                        help="Physical GPU index for power sampling via nvidia-smi")
    parser.add_argument("--power-sample-ms", type=int, default=100)
    parser.add_argument("--output", type=str, default=None, help="Save results as JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    input_shape = (1, 1, 112, 112, 80)
    print(f"Input shape: {input_shape}")
    print(f"Runs: {args.num_runs} (warmup: {args.warmup})")

    results = {}

    # Benchmark primary model
    print(f"\n--- Model 1: {args.model_path} ---")
    model1, features1 = load_basicunet_model(args.model_path, args.model_format, device)
    n_params1 = sum(p.numel() for p in model1.parameters())
    print(f"Loaded model 1: features={features1}")
    print(f"  Parameters: {n_params1:,} ({n_params1 / 1e6:.2f}M)")
    model1_results: list[tuple[str, dict]] = []
    fp32_1 = benchmark_model(
        model1, device, input_shape, args.num_runs, args.warmup, use_amp=False,
        gpu_index=args.gpu_index, power_sample_ms=args.power_sample_ms
    )
    model1_results.append(("fp32", fp32_1))
    if args.amp:
        amp_1 = benchmark_model(
            model1, device, input_shape, args.num_runs, args.warmup, use_amp=True,
            gpu_index=args.gpu_index, power_sample_ms=args.power_sample_ms
        )
        model1_results.append(("amp", amp_1))
    if args.compile_mode:
        compile_1 = benchmark_model(
            model1, device, input_shape, args.num_runs, args.warmup, use_amp=False,
            compile_mode=args.compile_mode, gpu_index=args.gpu_index, power_sample_ms=args.power_sample_ms
        )
        model1_results.append((f"compile_{args.compile_mode}_fp32", compile_1))
        if args.benchmark_compile_amp:
            compile_amp_1 = benchmark_model(
                model1, device, input_shape, args.num_runs, args.warmup, use_amp=True,
                compile_mode=args.compile_mode, gpu_index=args.gpu_index, power_sample_ms=args.power_sample_ms
            )
            model1_results.append((f"compile_{args.compile_mode}_amp", compile_amp_1))
    print_results("Model 1", model1_results)
    results["model_1"] = {"path": args.model_path, "features": list(features1)}
    for label, payload in model1_results:
        results["model_1"][label] = payload
    del model1
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Benchmark comparison model
    if args.compare_path:
        print(f"\n--- Model 2: {args.compare_path} ---")
        model2, features2 = load_basicunet_model(args.compare_path, args.compare_format, device)
        n_params2 = sum(p.numel() for p in model2.parameters())
        print(f"Loaded model 2: features={features2}")
        print(f"  Parameters: {n_params2:,} ({n_params2 / 1e6:.2f}M)")
        model2_results: list[tuple[str, dict]] = []
        fp32_2 = benchmark_model(
            model2, device, input_shape, args.num_runs, args.warmup, use_amp=False,
            gpu_index=args.gpu_index, power_sample_ms=args.power_sample_ms
        )
        model2_results.append(("fp32", fp32_2))
        if args.amp:
            amp_2 = benchmark_model(
                model2, device, input_shape, args.num_runs, args.warmup, use_amp=True,
                gpu_index=args.gpu_index, power_sample_ms=args.power_sample_ms
            )
            model2_results.append(("amp", amp_2))
        if args.compile_mode:
            compile_2 = benchmark_model(
                model2, device, input_shape, args.num_runs, args.warmup, use_amp=False,
                compile_mode=args.compile_mode, gpu_index=args.gpu_index, power_sample_ms=args.power_sample_ms
            )
            model2_results.append((f"compile_{args.compile_mode}_fp32", compile_2))
            if args.benchmark_compile_amp:
                compile_amp_2 = benchmark_model(
                    model2, device, input_shape, args.num_runs, args.warmup, use_amp=True,
                    compile_mode=args.compile_mode, gpu_index=args.gpu_index, power_sample_ms=args.power_sample_ms
                )
                model2_results.append((f"compile_{args.compile_mode}_amp", compile_amp_2))
        print_results("Model 2", model2_results)
        results["model_2"] = {"path": args.compare_path, "features": list(features2)}
        for label, payload in model2_results:
            results["model_2"][label] = payload

        # Comparison summary
        speedup_fp32 = fp32_1["mean_ms"] / fp32_2["mean_ms"]
        print(f"\n{'=' * 70}")
        print(f"  Comparison Summary")
        print(f"{'=' * 70}")
        print(f"  FP32 speedup: {speedup_fp32:.2f}x "
              f"({fp32_1['mean_ms']:.2f}ms -> {fp32_2['mean_ms']:.2f}ms)")
        if args.amp:
            amp_1 = next(payload for label, payload in model1_results if label == "amp")
            amp_2 = next(payload for label, payload in model2_results if label == "amp")
            speedup_amp = amp_1["mean_ms"] / amp_2["mean_ms"]
            print(f"  AMP speedup:  {speedup_amp:.2f}x "
                  f"({amp_1['mean_ms']:.2f}ms -> {amp_2['mean_ms']:.2f}ms)")
        if args.compile_mode:
            compile_1 = next(payload for label, payload in model1_results if label == f"compile_{args.compile_mode}_fp32")
            compile_2 = next(payload for label, payload in model2_results if label == f"compile_{args.compile_mode}_fp32")
            speedup_compile = compile_1["mean_ms"] / compile_2["mean_ms"]
            print(
                f"  Compile({args.compile_mode}) speedup: {speedup_compile:.2f}x "
                f"({compile_1['mean_ms']:.2f}ms -> {compile_2['mean_ms']:.2f}ms)"
            )
        print(f"{'=' * 70}")

        results["speedup_fp32"] = speedup_fp32
        del model2
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save results
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
