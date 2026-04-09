#!/usr/bin/env python3
"""
Build TensorRT engine from ONNX model.

Supports FP32, FP16, and INT8 precision.

Usage:
  python build_trt_engine.py --onnx_path model.onnx --engine_path model_fp16.engine --precision fp16
  python build_trt_engine.py --onnx_path model.onnx --engine_path model_int8.engine --precision int8
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import tensorrt as trt
from monai.data import Dataset

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False
    print("[WARNING] pycuda not available, INT8 calibration will not work")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibration using pre-loaded batches."""

    def __init__(self, batches: list[np.ndarray], cache_file="int8_cache.bin"):
        super().__init__()
        if not batches:
            raise ValueError("INT8 calibration requires at least one calibration batch")
        self.batches = batches
        self.current_batch = 0
        self.cache_file = cache_file
        self.batch_size = batches[0].shape[0]

        # Allocate device memory
        self.data = np.ascontiguousarray(batches[0].astype(np.float32))
        self.d_input = cuda.mem_alloc(self.data.nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_batch >= len(self.batches):
            return None
        self.data = np.ascontiguousarray(self.batches[self.current_batch].astype(np.float32))
        cuda.memcpy_htod(self.d_input, self.data)
        self.current_batch += 1
        return [int(self.d_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def adapt_image_to_input_shape(image: np.ndarray, spatial_shape: tuple[int, int, int]) -> tuple[np.ndarray, dict[str, int]]:
    """
    Center-crop or pad to the static TensorRT input shape.

    Calibration inputs must match the engine shape exactly. The dataset patches
    are close to the deployment ROI but not always identical, so we only apply
    deterministic center crop / symmetric zero-pad here.
    """

    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected image shape (1, X, Y, Z), got {image.shape}")

    current = image
    stats = {
        "cropped_dims": 0,
        "padded_dims": 0,
        "cropped_voxels_total": 0,
        "padded_voxels_total": 0,
    }

    for axis, target in enumerate(spatial_shape, start=1):
        size = current.shape[axis]
        if size > target:
            start = (size - target) // 2
            end = start + target
            slicer = [slice(None)] * current.ndim
            slicer[axis] = slice(start, end)
            current = current[tuple(slicer)]
            stats["cropped_dims"] += 1
            stats["cropped_voxels_total"] += size - target
        elif size < target:
            before = (target - size) // 2
            after = target - size - before
            pad_width = [(0, 0)] * current.ndim
            pad_width[axis] = (before, after)
            current = np.pad(current, pad_width, mode="constant")
            stats["padded_dims"] += 1
            stats["padded_voxels_total"] += target - size

    return current.astype(np.float32, copy=False), stats


def load_calibration_batches(
    data_dir: str,
    input_shape: tuple[int, int, int, int, int],
    split: str = "train",
    max_samples: int = 64,
    seed: int = 42,
    fit_only: bool = False,
) -> tuple[list[np.ndarray], dict[str, int | str]]:
    train_list, test_list = train.build_datalists(Path(data_dir))
    if split == "train":
        selected = list(train_list)
    elif split == "test":
        selected = list(test_list)
    else:
        selected = list(train_list) + list(test_list)

    rng = random.Random(seed)
    rng.shuffle(selected)
    if fit_only:
        spatial_shape = input_shape[2:]
        filtered = []
        for item in selected:
            shape = nib.load(item["image"]).shape
            if all(size <= target for size, target in zip(shape, spatial_shape)):
                filtered.append(item)
        selected = filtered
        rng.shuffle(selected)
    selected = selected[: max(1, min(max_samples, len(selected)))]

    _, val_tf = train.get_transforms(roi=input_shape[2:])
    dataset = Dataset(selected, transform=val_tf)

    batches: list[np.ndarray] = []
    summary = {
        "source_split": split,
        "fit_only": bool(fit_only),
        "requested_samples": int(max_samples),
        "selected_samples": int(len(selected)),
        "exact_shape_count": 0,
        "adapted_count": 0,
        "cropped_sample_count": 0,
        "padded_sample_count": 0,
        "cropped_voxels_total": 0,
        "padded_voxels_total": 0,
    }

    spatial_shape = input_shape[2:]
    for idx in range(len(dataset)):
        item = dataset[idx]
        image = item["image"]
        if hasattr(image, "numpy"):
            image_np = image.numpy()
        else:
            image_np = np.asarray(image)
        image_np = image_np.astype(np.float32, copy=False)
        if tuple(image_np.shape[1:]) == tuple(spatial_shape):
            summary["exact_shape_count"] += 1
            batch = image_np[None, ...]
        else:
            adapted, stats = adapt_image_to_input_shape(image_np, spatial_shape)
            summary["adapted_count"] += 1
            summary["cropped_voxels_total"] += stats["cropped_voxels_total"]
            summary["padded_voxels_total"] += stats["padded_voxels_total"]
            if stats["cropped_dims"] > 0:
                summary["cropped_sample_count"] += 1
            if stats["padded_dims"] > 0:
                summary["padded_sample_count"] += 1
            batch = adapted[None, ...]
        batches.append(np.ascontiguousarray(batch))
    return batches, summary


def build_engine(onnx_path: str, engine_path: str, precision: str = "fp16",
                 workspace_gb: int = 4, verbose: bool = False,
                 calibration_batches: list[np.ndarray] | None = None,
                 calibration_summary: dict[str, int | str] | None = None) -> bool:
    """Build TensorRT engine from ONNX model."""

    logger_level = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    logger = trt.Logger(logger_level)

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("Failed to parse ONNX model")
            for i in range(parser.num_errors):
                print(f"  Error {i}: {parser.get_error(i)}")
            return False

    print(f"ONNX parsed successfully")

    config = builder.create_builder_config()

    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    except AttributeError:
        config.max_workspace_size = workspace_gb * (1 << 30)

    # Set precision
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("Using FP16 precision")
        else:
            print("WARNING: FP16 not supported, falling back to FP32")
    elif precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            # Also enable FP16 as fallback for layers that don't support INT8
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            # Set calibrator
            cache_file = engine_path.replace(".engine", "_int8_cache.bin")
            if calibration_batches:
                calibrator = Int8Calibrator(
                    batches=calibration_batches,
                    cache_file=cache_file,
                )
                print(
                    f"Using INT8 precision with real calibration data "
                    f"({len(calibration_batches)} samples)"
                )
                if calibration_summary:
                    print(f"Calibration summary: {json.dumps(calibration_summary, indent=2)}")
            else:
                random_batches = [
                    np.random.randn(1, 1, 112, 112, 80).astype(np.float32)
                    for _ in range(100)
                ]
                calibrator = Int8Calibrator(
                    batches=random_batches,
                    cache_file=cache_file,
                )
                print("WARNING: Using INT8 precision with random calibration data (legacy smoke mode only)")
            config.int8_calibrator = calibrator
        else:
            print("WARNING: INT8 not supported, falling back to FP32")
    else:
        print("Using FP32 precision")

    print(f"Building TensorRT engine (workspace: {workspace_gb}GB)...")

    try:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("Failed to build engine")
            return False
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
    except AttributeError:
        engine = builder.build_engine(network, config)
        if engine is None:
            print("Failed to build engine")
            return False
        serialized_engine = engine.serialize()

    # Save engine
    os.makedirs(os.path.dirname(engine_path) if os.path.dirname(engine_path) else ".", exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    file_size = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"Engine saved: {engine_path} ({file_size:.1f} MB)")

    # Print engine info
    try:
        num_io = engine.num_io_tensors
        for i in range(num_io):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dtype = engine.get_tensor_dtype(name)
            mode = engine.get_tensor_mode(name)
            print(f"  {mode}: {name} {shape} ({dtype})")
    except AttributeError:
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            io = "INPUT" if engine.binding_is_input(i) else "OUTPUT"
            print(f"  {io}: {name} {shape}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    parser.add_argument("--onnx_path", required=True, help="ONNX model path")
    parser.add_argument("--engine_path", required=True, help="Output engine path")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--workspace-gb", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--calib-data-dir", type=str, default=None, help="Real calibration dataset directory")
    parser.add_argument("--calib-split", type=str, default="train", choices=["train", "test", "both"])
    parser.add_argument("--calib-max-samples", type=int, default=64)
    parser.add_argument("--calib-seed", type=int, default=42)
    parser.add_argument("--calib-fit-only", action="store_true",
                        help="Only use calibration samples that already fit the static ROI, avoiding center crops")
    parser.add_argument("--calib-summary", type=str, default=None, help="Optional JSON path for calibration summary")
    args = parser.parse_args()

    if not os.path.exists(args.onnx_path):
        print(f"ONNX file not found: {args.onnx_path}")
        sys.exit(1)

    calibration_batches = None
    calibration_summary = None
    if args.precision == "int8" and args.calib_data_dir:
        calibration_batches, calibration_summary = load_calibration_batches(
            data_dir=args.calib_data_dir,
            input_shape=(1, 1, 112, 112, 80),
            split=args.calib_split,
            max_samples=args.calib_max_samples,
            seed=args.calib_seed,
            fit_only=bool(args.calib_fit_only),
        )
        if args.calib_summary:
            out_path = Path(args.calib_summary)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(calibration_summary, indent=2))
        else:
            default_summary = Path(args.engine_path).with_suffix("").as_posix() + "_calibration_summary.json"
            Path(default_summary).write_text(json.dumps(calibration_summary, indent=2))

    success = build_engine(
        onnx_path=args.onnx_path,
        engine_path=args.engine_path,
        precision=args.precision,
        workspace_gb=getattr(args, "workspace_gb", 4),
        verbose=args.verbose,
        calibration_batches=calibration_batches,
        calibration_summary=calibration_summary,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
