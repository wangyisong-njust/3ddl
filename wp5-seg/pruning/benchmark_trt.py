#!/usr/bin/env python3
"""
Benchmark TensorRT engines vs PyTorch models.

Compares: PyTorch FP32 vs TRT FP32 vs TRT FP16 vs TRT INT8
for both original and pruned models.

Usage:
  python benchmark_trt.py \
    --pytorch_model model.ckpt \
    --trt_engines model_fp32.engine model_fp16.engine model_int8.engine \
    --trt_labels FP32 FP16 INT8 \
    --num_runs 200
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

from runtime_support import (
    NvidiaSMIPowerMonitor,
    load_basicunet_model,
    measure_idle_power,
    summarize_power_samples,
)


def benchmark_pytorch(model, input_shape=(1, 1, 112, 112, 80),
                      num_runs=200, warmup=20, use_amp=False,
                      compile_mode: str | None = None,
                      gpu_index: int | None = None,
                      power_sample_ms: int = 100):
    """Benchmark PyTorch model."""
    device = torch.device("cuda")
    model = model.to(device).eval()
    run_model = torch.compile(model, mode=compile_mode) if compile_mode else model
    dummy = torch.randn(*input_shape, device=device)

    first_run_ms = None
    with torch.no_grad():
        if use_amp:
            ctx = lambda: torch.amp.autocast("cuda")
        else:
            from contextlib import nullcontext
            ctx = nullcontext

        with ctx():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_model(dummy)
            torch.cuda.synchronize()
            first_run_ms = (time.perf_counter() - t0) * 1000.0

        for _ in range(max(0, warmup - 1)):
            if use_amp:
                with torch.amp.autocast("cuda"):
                    run_model(dummy)
            else:
                run_model(dummy)
            torch.cuda.synchronize()

        idle_power_w = measure_idle_power(gpu_index=gpu_index, sample_ms=power_sample_ms) if gpu_index is not None else None
        monitor = NvidiaSMIPowerMonitor(gpu_index=gpu_index, sample_ms=power_sample_ms) if gpu_index is not None else None
        if monitor is not None:
            monitor.start()
        latencies = []
        wall_t0 = time.perf_counter()
        try:
            for _ in range(num_runs):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        run_model(dummy)
                else:
                    run_model(dummy)
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)
        finally:
            wall_time_s = time.perf_counter() - wall_t0
            if monitor is not None:
                monitor.stop()

    lat = np.array(latencies)
    power = summarize_power_samples(
        samples=(monitor.samples if monitor is not None else []),
        wall_time_s=wall_time_s,
        iterations=num_runs,
        idle_power_w=idle_power_w,
    )
    return {
        "compile_mode": compile_mode,
        "first_run_ms": float(first_run_ms) if first_run_ms is not None else None,
        "mean_ms": float(lat.mean()),
        "std_ms": float(lat.std()),
        "min_ms": float(lat.min()),
        "median_ms": float(np.median(lat)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "fps": float(1000.0 / lat.mean()),
        "power": power.as_dict(),
    }


def benchmark_trt_engine(engine_path, input_shape=(1, 1, 112, 112, 80),
                         num_runs=200, warmup=20,
                         gpu_index: int | None = None,
                         power_sample_ms: int = 100):
    """Benchmark TensorRT engine."""
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    stream = cuda.Stream()

    # Allocate I/O
    input_data = np.random.randn(*input_shape).astype(np.float32)
    d_input = cuda.mem_alloc(input_data.nbytes)

    # Find output tensors and allocate
    d_outputs = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            context.set_tensor_address(name, int(d_input))
        else:
            shape = engine.get_tensor_shape(name)
            size = int(np.prod(shape) * np.dtype(np.float32).itemsize)
            d_out = cuda.mem_alloc(size)
            d_outputs.append(d_out)
            context.set_tensor_address(name, int(d_out))

    # Warmup
    torch_first = None
    for warm_idx in range(warmup):
        cuda.memcpy_htod_async(d_input, input_data, stream)
        t0 = time.perf_counter()
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()
        if warm_idx == 0:
            torch_first = (time.perf_counter() - t0) * 1000.0

    # Benchmark
    latencies = []
    idle_power_w = measure_idle_power(gpu_index=gpu_index, sample_ms=power_sample_ms) if gpu_index is not None else None
    monitor = NvidiaSMIPowerMonitor(gpu_index=gpu_index, sample_ms=power_sample_ms) if gpu_index is not None else None
    if monitor is not None:
        monitor.start()
    wall_t0 = time.perf_counter()
    try:
        for _ in range(num_runs):
            cuda.memcpy_htod_async(d_input, input_data, stream)
            t0 = time.perf_counter()
            context.execute_async_v3(stream_handle=stream.handle)
            stream.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)
    finally:
        wall_time_s = time.perf_counter() - wall_t0
        if monitor is not None:
            monitor.stop()

    lat = np.array(latencies)
    power = summarize_power_samples(
        samples=(monitor.samples if monitor is not None else []),
        wall_time_s=wall_time_s,
        iterations=num_runs,
        idle_power_w=idle_power_w,
    )
    return {
        "first_run_ms": float(torch_first) if torch_first is not None else None,
        "mean_ms": float(lat.mean()),
        "std_ms": float(lat.std()),
        "min_ms": float(lat.min()),
        "median_ms": float(np.median(lat)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "fps": float(1000.0 / lat.mean()),
        "power": power.as_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark TRT vs PyTorch")
    parser.add_argument("--pytorch_model", type=str, help="PyTorch model path")
    parser.add_argument("--model_format", type=str, default="state_dict",
                        choices=["state_dict", "pruned"])
    parser.add_argument("--trt_engines", nargs="+", type=str, default=[],
                        help="TensorRT engine paths")
    parser.add_argument("--trt_labels", nargs="+", type=str, default=[],
                        help="Labels for TRT engines (e.g., FP32 FP16 INT8)")
    parser.add_argument("--num_runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--pytorch_amp", action="store_true")
    parser.add_argument("--pytorch_compile_mode", type=str, default=None,
                        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    parser.add_argument("--pytorch_compile_amp", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=None,
                        help="Physical GPU index for nvidia-smi power sampling")
    parser.add_argument("--power-sample-ms", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    input_shape = (1, 1, 112, 112, 80)
    results = {}

    print(f"Input shape: {input_shape}")
    print(f"Benchmark runs: {args.num_runs}")
    print()

    # PyTorch benchmark
    if args.pytorch_model:
        model, features = load_basicunet_model(args.pytorch_model, args.model_format, torch.device("cuda"))
        n_params = sum(p.numel() for p in model.parameters())
        print(f"PyTorch model: features={features}, params={n_params:,}")

        print("  Benchmarking PyTorch FP32...")
        results["pytorch_fp32"] = benchmark_pytorch(
            model, input_shape, args.num_runs, args.warmup,
            use_amp=False, gpu_index=args.gpu_index, power_sample_ms=args.power_sample_ms
        )
        print(f"    Mean: {results['pytorch_fp32']['mean_ms']:.2f}ms, "
              f"FPS: {results['pytorch_fp32']['fps']:.1f}")
        if args.pytorch_amp:
            print("  Benchmarking PyTorch AMP...")
            results["pytorch_amp"] = benchmark_pytorch(
                model, input_shape, args.num_runs, args.warmup,
                use_amp=True, gpu_index=args.gpu_index, power_sample_ms=args.power_sample_ms
            )
            print(f"    Mean: {results['pytorch_amp']['mean_ms']:.2f}ms, "
                  f"FPS: {results['pytorch_amp']['fps']:.1f}")
        if args.pytorch_compile_mode:
            print(f"  Benchmarking PyTorch compile ({args.pytorch_compile_mode})...")
            results["pytorch_compile_fp32"] = benchmark_pytorch(
                model, input_shape, args.num_runs, args.warmup,
                use_amp=False, compile_mode=args.pytorch_compile_mode,
                gpu_index=args.gpu_index, power_sample_ms=args.power_sample_ms
            )
            print(f"    Mean: {results['pytorch_compile_fp32']['mean_ms']:.2f}ms, "
                  f"FPS: {results['pytorch_compile_fp32']['fps']:.1f}")
            if args.pytorch_compile_amp:
                results["pytorch_compile_amp"] = benchmark_pytorch(
                    model, input_shape, args.num_runs, args.warmup,
                    use_amp=True, compile_mode=args.pytorch_compile_mode,
                    gpu_index=args.gpu_index, power_sample_ms=args.power_sample_ms
                )
                print(f"    Compile AMP mean: {results['pytorch_compile_amp']['mean_ms']:.2f}ms, "
                      f"FPS: {results['pytorch_compile_amp']['fps']:.1f}")

    # TRT benchmarks
    for i, engine_path in enumerate(args.trt_engines):
        label = args.trt_labels[i] if i < len(args.trt_labels) else f"TRT_{i}"
        key = f"trt_{label.lower()}"
        engine_size = Path(engine_path).stat().st_size / (1024 * 1024)
        print(f"\n  Benchmarking TRT {label} ({engine_size:.1f}MB)...")
        results[key] = benchmark_trt_engine(
            engine_path, input_shape, args.num_runs, args.warmup,
            gpu_index=args.gpu_index, power_sample_ms=args.power_sample_ms
        )
        print(f"    Mean: {results[key]['mean_ms']:.2f}ms, "
              f"FPS: {results[key]['fps']:.1f}")

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"{'Method':<25} {'Mean':>10} {'P95':>10} {'FPS':>10} {'Power':>10} {'Speedup':>10}")
    print(f"{'-' * 80}")

    baseline_ms = None
    for key, r in results.items():
        if baseline_ms is None:
            baseline_ms = r["mean_ms"]
        speedup = baseline_ms / r["mean_ms"]
        power_text = "-"
        avg_power = r.get("power", {}).get("avg_power_w")
        if avg_power is not None:
            power_text = f"{avg_power:.1f}W"
        print(f"{key:<25} {r['mean_ms']:>9.2f}ms {r['p95_ms']:>9.2f}ms "
              f"{r['fps']:>9.1f} {power_text:>10} {speedup:>9.2f}x")

    print(f"{'=' * 80}")

    # Save
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
