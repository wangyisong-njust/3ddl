#!/usr/bin/env python3
"""
Shared runtime helpers for wp5-seg deployment experiments.

This module intentionally keeps deployment-facing utilities in one place so
benchmark, calibration, and runtime-eval scripts stay numerically aligned.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from monai.networks.nets import BasicUNet


def load_basicunet_model(model_path: str, model_format: str, device: torch.device) -> tuple[BasicUNet, tuple[int, ...]]:
    """Load either a baseline checkpoint or a pruned checkpoint."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if model_format == "pruned":
        features = tuple(ckpt["features"])
        model = BasicUNet(spatial_dims=3, in_channels=1, out_channels=5, features=features)
        model.load_state_dict(ckpt["state_dict"])
    else:
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        features = (32, 32, 64, 128, 256, 32)
        model = BasicUNet(spatial_dims=3, in_channels=1, out_channels=5)
        model.load_state_dict(state_dict)
    return model.to(device).eval(), features


@dataclass
class PowerStats:
    sample_count: int
    avg_power_w: float | None
    max_power_w: float | None
    avg_utilization_pct: float | None
    wall_time_s: float
    energy_j: float | None
    idle_power_w: float | None
    dynamic_energy_j: float | None
    joules_per_iteration: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "sample_count": self.sample_count,
            "avg_power_w": self.avg_power_w,
            "max_power_w": self.max_power_w,
            "avg_utilization_pct": self.avg_utilization_pct,
            "wall_time_s": self.wall_time_s,
            "energy_j": self.energy_j,
            "idle_power_w": self.idle_power_w,
            "dynamic_energy_j": self.dynamic_energy_j,
            "joules_per_iteration": self.joules_per_iteration,
        }


class NvidiaSMIPowerMonitor:
    """
    Lightweight `nvidia-smi` based power sampler.

    `pynvml` is not available in the current environment, so this monitor uses
    `nvidia-smi -lms` in a background thread. It is coarse but adequate for
    repeated inference loops where we care about average/peak power and
    aggregate energy.
    """

    def __init__(self, gpu_index: int, sample_ms: int = 100):
        self.gpu_index = gpu_index
        self.sample_ms = max(20, int(sample_ms))
        self.samples: list[tuple[float, float]] = []
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def _reader(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        for raw_line in self._proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                power_text, util_text = [part.strip() for part in line.split(",")[:2]]
                power = float(power_text)
                util = float(util_text)
            except Exception:
                continue
            self.samples.append((power, util))

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("Power monitor already running")
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=power.draw,utilization.gpu",
            "--format=csv,noheader,nounits",
            "-lms",
            str(self.sample_ms),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=2.0)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._proc = None
        self._thread = None

    def __enter__(self) -> "NvidiaSMIPowerMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def summarize_power_samples(
    samples: Iterable[tuple[float, float]],
    wall_time_s: float,
    iterations: int,
    idle_power_w: float | None = None,
) -> PowerStats:
    sample_list = list(samples)
    if not sample_list:
        return PowerStats(
            sample_count=0,
            avg_power_w=None,
            max_power_w=None,
            avg_utilization_pct=None,
            wall_time_s=wall_time_s,
            energy_j=None,
            idle_power_w=idle_power_w,
            dynamic_energy_j=None,
            joules_per_iteration=None,
        )
    powers = np.array([p for p, _ in sample_list], dtype=np.float32)
    utils = np.array([u for _, u in sample_list], dtype=np.float32)
    avg_power = float(powers.mean())
    energy_j = avg_power * wall_time_s
    dynamic_energy_j = None
    if idle_power_w is not None:
        dynamic_energy_j = max(avg_power - idle_power_w, 0.0) * wall_time_s
    return PowerStats(
        sample_count=len(sample_list),
        avg_power_w=avg_power,
        max_power_w=float(powers.max()),
        avg_utilization_pct=float(utils.mean()),
        wall_time_s=wall_time_s,
        energy_j=energy_j,
        idle_power_w=idle_power_w,
        dynamic_energy_j=dynamic_energy_j,
        joules_per_iteration=(energy_j / iterations) if iterations > 0 else None,
    )


def measure_idle_power(gpu_index: int, sample_ms: int = 100, duration_s: float = 1.5) -> float | None:
    monitor = NvidiaSMIPowerMonitor(gpu_index=gpu_index, sample_ms=sample_ms)
    monitor.start()
    time.sleep(max(duration_s, sample_ms / 1000.0))
    monitor.stop()
    if not monitor.samples:
        return None
    return float(np.mean([power for power, _ in monitor.samples]))


class TensorRTPredictor:
    """
    TensorRT predictor wrapper for MONAI sliding-window inference.

    The implementation mirrors the production `intelliscan` backend closely but
    is kept local to `wp5-seg` to avoid cross-repo runtime dependencies.
    """

    def __init__(self, engine_path: str | Path, force_host_relay: bool = False):
        import tensorrt as trt
        import pycuda.driver as cuda

        self._trt = trt
        self._cuda = cuda
        self._ctx = None
        self._use_gpu_direct = False

        engine_path = Path(engine_path)
        if not force_host_relay:
            try:
                import pycuda.autoinit  # noqa: F401
                self._use_gpu_direct = True
            except Exception:
                cuda.init()
                self._ctx = cuda.Device(0).make_context()
        else:
            cuda.init()
            self._ctx = cuda.Device(0).make_context()

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self._input_name = None
        self._output_name = None
        self._input_shape = None
        self._output_shape = None
        self._d_input = None
        self._d_output = None

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            nbytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
            if mode == trt.TensorIOMode.INPUT:
                self._input_name = name
                self._input_shape = shape
                self._d_input = cuda.mem_alloc(nbytes)
                self.context.set_tensor_address(name, int(self._d_input))
            else:
                self._output_name = name
                self._output_shape = shape
                self._d_output = cuda.mem_alloc(nbytes)
                self.context.set_tensor_address(name, int(self._d_output))

        if self._use_gpu_direct:
            try:
                test_tensor = torch.zeros(self._input_shape, dtype=torch.float32, device="cuda")
                nbytes = test_tensor.nelement() * test_tensor.element_size()
                cuda.memcpy_dtod_async(self._d_input, test_tensor.data_ptr(), nbytes, self.stream)
                self.stream.synchronize()
                del test_tensor
            except Exception:
                self._use_gpu_direct = False
                cuda.init()
                self._ctx = cuda.Device(0).make_context()

        if not self._use_gpu_direct:
            self._h_input = cuda.pagelocked_empty(int(np.prod(self._input_shape)), dtype=np.float32)
            self._h_output = cuda.pagelocked_empty(int(np.prod(self._output_shape)), dtype=np.float32)
            if self._ctx is not None:
                self._ctx.pop()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_gpu_direct:
            return self._call_gpu_direct(x)
        return self._call_host_relay(x)

    def eval(self) -> "TensorRTPredictor":
        return self

    def _call_gpu_direct(self, x: torch.Tensor) -> torch.Tensor:
        cuda = self._cuda
        outputs = []
        for b in range(x.shape[0]):
            sample = x[b : b + 1].contiguous().float()
            output_tensor = torch.empty(self._output_shape, dtype=torch.float32, device=x.device)
            cuda.memcpy_dtod_async(self._d_input, sample.data_ptr(), sample.nelement() * sample.element_size(), self.stream)
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            cuda.memcpy_dtod_async(
                output_tensor.data_ptr(),
                self._d_output,
                output_tensor.nelement() * output_tensor.element_size(),
                self.stream,
            )
            self.stream.synchronize()
            outputs.append(output_tensor)
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)

    def _call_host_relay(self, x: torch.Tensor) -> torch.Tensor:
        cuda = self._cuda
        outputs = []
        for b in range(x.shape[0]):
            sample = x[b : b + 1].contiguous().float()
            np.copyto(self._h_input, sample.cpu().numpy().ravel())
            self._ctx.push()
            cuda.memcpy_htod(self._d_input, self._h_input)
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self._h_output, self._d_output, self.stream)
            self.stream.synchronize()
            self._ctx.pop()
            outputs.append(torch.from_numpy(self._h_output.reshape(self._output_shape).copy()).to(x.device))
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
