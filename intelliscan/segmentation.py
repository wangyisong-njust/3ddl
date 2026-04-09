#!/usr/bin/env python3
"""
3D Segmentation Inference Module

Provides inference capabilities using MONAI BasicUNet for 3D semantic segmentation.
Designed for integration with detection + metrology pipelines.
Supports PyTorch and TensorRT backends.

Workflow:
1. Load trained BasicUNet checkpoint or TensorRT engine
2. For each detected 3D bounding box:
   - Expand bbox with margin (handles edge cases)
   - Extract crop from full volume
   - Apply ClipZScoreNormalize preprocessing
   - Run sliding window inference
   - Apply post-processing (void filling)
3. Assemble all predictions into full volume
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import BasicUNet

from utils import log

# =============================================================================
# TensorRT predictor wrapper
# =============================================================================


class TensorRTPredictor:
    """Wraps a TensorRT engine as a callable predictor for sliding_window_inference.

    Accepts PyTorch CUDA tensors and returns PyTorch CUDA tensors.
    Supports two transfer modes:
    - GPU direct: memcpy_dtod (fast, requires pycuda/PyTorch context compatibility)
    - Host relay: GPU→CPU→GPU via pinned memory (slower but universally compatible)

    Auto-detects the best mode at init time with a test transfer.
    """

    def __init__(self, engine_path: str | Path, num_classes: int = 5):
        import tensorrt as trt
        import pycuda.driver as cuda

        self._trt = trt
        self._cuda = cuda

        engine_path = Path(engine_path)
        log(f"Loading TensorRT engine: {engine_path}")

        # Try GPU direct mode first (pycuda.autoinit shares PyTorch's context)
        self._use_gpu_direct = False
        self._ctx = None
        try:
            import pycuda.autoinit  # noqa: F401
            self._use_gpu_direct = True
        except Exception:
            # autoinit failed, fall back to manual context
            cuda.init()
            self._ctx = cuda.Device(0).make_context()

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.num_classes = num_classes

        # Discover I/O tensor names, shapes, and allocate device buffers
        self._input_name = None
        self._output_name = None
        self._d_input = None
        self._d_output = None
        self._input_shape = None
        self._output_shape = None

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            nbytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
            if mode == trt.TensorIOMode.INPUT:
                self._input_name = name
                self._input_shape = tuple(shape)
                self._d_input = cuda.mem_alloc(nbytes)
                self.context.set_tensor_address(name, int(self._d_input))
                log(f"  TRT input '{name}': shape={tuple(shape)}")
            else:
                self._output_name = name
                self._output_shape = tuple(shape)
                self._d_output = cuda.mem_alloc(nbytes)
                self.context.set_tensor_address(name, int(self._d_output))
                log(f"  TRT output '{name}': shape={tuple(shape)}")

        # Validate GPU direct mode with a test transfer
        if self._use_gpu_direct:
            try:
                test_tensor = torch.zeros(self._input_shape, dtype=torch.float32, device="cuda")
                nbytes = test_tensor.nelement() * test_tensor.element_size()
                cuda.memcpy_dtod_async(self._d_input, test_tensor.data_ptr(), nbytes, self.stream)
                self.stream.synchronize()
                del test_tensor
            except Exception as e:
                log(f"GPU direct transfer failed ({e}), falling back to host relay mode")
                self._use_gpu_direct = False
                cuda.init()
                self._ctx = cuda.Device(0).make_context()

        # Pre-allocate pinned host buffers for host relay mode
        if not self._use_gpu_direct:
            self._h_input = cuda.pagelocked_empty(
                int(np.prod(self._input_shape)), dtype=np.float32
            )
            self._h_output = cuda.pagelocked_empty(
                int(np.prod(self._output_shape)), dtype=np.float32
            )
            # Pop manual context so PyTorch can use its own
            if self._ctx is not None:
                self._ctx.pop()

        mode_str = "GPU direct (dtod)" if self._use_gpu_direct else "host relay (htod/dtoh)"
        log(f"TensorRT engine loaded successfully, transfer mode: {mode_str}")

    def _call_gpu_direct(self, x: torch.Tensor) -> torch.Tensor:
        """Fast path: GPU-to-GPU memory transfer via data pointers."""
        cuda = self._cuda
        batch_size = x.shape[0]

        outputs = []
        for b in range(batch_size):
            sample = x[b : b + 1].contiguous().float()

            output_tensor = torch.empty(
                self._output_shape, dtype=torch.float32, device=x.device
            )

            input_nbytes = sample.nelement() * sample.element_size()
            output_nbytes = output_tensor.nelement() * output_tensor.element_size()

            cuda.memcpy_dtod_async(
                self._d_input, sample.data_ptr(), input_nbytes, self.stream
            )
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            cuda.memcpy_dtod_async(
                output_tensor.data_ptr(), self._d_output, output_nbytes, self.stream
            )
            self.stream.synchronize()

            outputs.append(output_tensor)

        if batch_size == 1:
            return outputs[0]
        return torch.cat(outputs, dim=0)

    def _call_host_relay(self, x: torch.Tensor) -> torch.Tensor:
        """Compatible path: GPU→CPU→TRT→CPU→GPU via pinned host memory."""
        cuda = self._cuda
        batch_size = x.shape[0]

        outputs = []
        for b in range(batch_size):
            sample = x[b : b + 1].contiguous().float()
            h_in = sample.cpu().numpy().ravel()
            np.copyto(self._h_input, h_in)

            self._ctx.push()

            cuda.memcpy_htod(self._d_input, self._h_input)
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self._h_output, self._d_output, self.stream)
            self.stream.synchronize()

            self._ctx.pop()

            output_tensor = torch.from_numpy(
                self._h_output.reshape(self._output_shape).copy()
            ).to(x.device)

            outputs.append(output_tensor)

        if batch_size == 1:
            return outputs[0]
        return torch.cat(outputs, dim=0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Run TRT inference on a PyTorch CUDA tensor.

        Automatically uses GPU direct or host relay depending on
        hardware compatibility detected at init time.

        Args:
            x: Input tensor of shape (B, 1, X, Y, Z) on CUDA

        Returns:
            Output tensor of shape (B, num_classes, X, Y, Z) on CUDA
        """
        if self._use_gpu_direct:
            return self._call_gpu_direct(x)
        return self._call_host_relay(x)

# =============================================================================
# Post-processing utilities
# =============================================================================


def fill_internal_voids(
    prediction: np.ndarray,
    background_class: int = 0,
    void_class: int = 3,
) -> np.ndarray:
    """Fill internal voids (holes) in a 3D segmentation volume.

    Finds all connected regions of background_class that do NOT touch
    the volume boundary and relabels them as void_class.

    Args:
        prediction: 3D integer segmentation array (D, H, W)
        background_class: Label considered as true background
        void_class: Label to assign to internal holes

    Returns:
        Modified prediction array with internal holes relabeled
    """
    # Lazy import - only needed when apply_void_fill is True
    from scipy.ndimage import label as scipy_label

    bg_mask = prediction == background_class
    bg_labels, n_labels = scipy_label(bg_mask)

    # Identify which labels touch the volume boundary (6 faces of cuboid)
    boundary = np.zeros_like(bg_mask, dtype=bool)
    boundary[0, :, :] = True
    boundary[-1, :, :] = True
    boundary[:, 0, :] = True
    boundary[:, -1, :] = True
    boundary[:, :, 0] = True
    boundary[:, :, -1] = True

    external_labels = set(np.unique(bg_labels[boundary & bg_mask]).tolist())
    external_labels.discard(0)

    # Relabel internal (non-boundary-touching) background as void
    for lab in range(1, n_labels + 1):
        if lab not in external_labels:
            prediction[bg_labels == lab] = void_class

    return prediction


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SegmentationConfig:
    """Configuration for segmentation inference pipeline."""

    roi_size: tuple[int, int, int] = (112, 112, 80)
    direct_roi_size: tuple[int, int, int] | None = None
    overlap: float = 0.5
    sw_batch_size: int = 1
    margin: int = 15
    use_adaptive_margin: bool = False
    use_guarded_adaptive_margin: bool = False
    guard_max_loss_xy: int = 4
    guard_max_loss_z: int = 8
    force_sliding_window: bool = False
    force_direct_full_crop: bool = False
    num_classes: int = 5
    apply_normalization: bool = True
    apply_void_fill: bool = False  # Disabled by default for performance
    device: str = "cuda"

    # Normalization parameters
    clip_percentile_low: float = 1.0
    clip_percentile_high: float = 99.0

    # Post-processing parameters
    void_class: int = 3
    background_class: int = 0

    # TensorRT backend
    use_trt: bool = False
    trt_engine_path: str | None = None

    # PyTorch runtime backend
    use_compile: bool = False
    compile_mode: str = "reduce-overhead"


# =============================================================================
# Core inference class
# =============================================================================


class SegmentationInference:
    """Segmentation inference engine using MONAI BasicUNet or TensorRT.

    Provides methods for:
    - Single crop inference (PyTorch or TensorRT backend)
    - Multi-bbox batch inference
    - Full volume assembly
    """

    def __init__(
        self,
        model_path: str | Path,
        config: SegmentationConfig | None = None,
    ):
        """Initialize inference engine.

        Args:
            model_path: Path to trained model checkpoint (.ckpt) or TRT engine (.engine)
            config: Configuration object (uses defaults if None)
        """
        self.model_path = Path(model_path)
        self.config = config or SegmentationConfig()
        device_name = self.config.device
        if device_name == "cuda" and not torch.cuda.is_available():
            log("CUDA is not available; falling back to CPU for segmentation inference.", level="warning")
            device_name = "cpu"
            self.config.device = device_name
        self.device = torch.device(device_name)
        self.model: BasicUNet | None = None
        self._predictor: TensorRTPredictor | BasicUNet | None = None

    def load_model(self) -> None:
        """Load model (PyTorch or TensorRT).

        Supports:
        - TensorRT engine (.engine) when config.use_trt is True
        - Original PyTorch state_dict (.ckpt)
        - Pruned PyTorch checkpoint (dict with 'features' key)
        """
        if self._predictor is not None:
            return  # Already loaded

        if self.config.use_trt:
            engine_path = self.config.trt_engine_path or str(self.model_path)
            self._predictor = TensorRTPredictor(engine_path, self.config.num_classes)
            log(f"Using TensorRT backend: {engine_path}")
            return

        raw_ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # Detect pruned model format (contains 'features' key)
        if isinstance(raw_ckpt, dict) and "features" in raw_ckpt:
            features = tuple(raw_ckpt["features"])
            self.model = BasicUNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=self.config.num_classes,
                features=features,
            ).to(self.device)
            state_dict = raw_ckpt["state_dict"]
            log(f"Loading pruned model with features={features}")
        else:
            self.model = BasicUNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=self.config.num_classes,
            ).to(self.device)
            state_dict = raw_ckpt
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        if self.config.use_compile:
            self.model = torch.compile(self.model, mode=self.config.compile_mode)
            log(f"Using PyTorch compile backend: mode={self.config.compile_mode}")
        self._predictor = self.model
        log(f"Loaded segmentation model from: {self.model_path}")

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply ClipZScoreNormalize preprocessing (CPU fallback)."""
        cfg = self.config
        flat = data.ravel()
        p_low = np.percentile(flat, cfg.clip_percentile_low)
        p_high = np.percentile(flat, cfg.clip_percentile_high)
        clipped = np.clip(data, p_low, p_high)
        mean, std = clipped.mean(), clipped.std()
        return ((clipped - mean) / (std + 1e-8)).astype(np.float32)

    def normalize_gpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply ClipZScoreNormalize on GPU.

        Args:
            tensor: 1-D or N-D float tensor on CUDA

        Returns:
            Normalized tensor (same shape)
        """
        cfg = self.config
        flat = tensor.reshape(-1)
        q_low = cfg.clip_percentile_low / 100.0
        q_high = cfg.clip_percentile_high / 100.0
        p_low = torch.quantile(flat, q_low)
        p_high = torch.quantile(flat, q_high)
        clipped = tensor.clamp(p_low, p_high)
        mean = clipped.mean()
        std = clipped.std()
        return (clipped - mean) / (std + 1e-8)

    def infer_crop(self, crop: np.ndarray) -> np.ndarray:
        """Run inference on a single 3D crop.

        Args:
            crop: 3D numpy array (X, Y, Z) - raw intensity values

        Returns:
            Prediction array (X, Y, Z) with class labels
        """
        if self._predictor is None:
            self.load_model()

        # To GPU tensor first, then normalize on GPU
        tensor = torch.from_numpy(crop.astype(np.float32)).to(self.device)

        if self.config.apply_normalization:
            tensor = self.normalize_gpu(tensor)

        # Shape to [1, 1, X, Y, Z]
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            orig_shape = tensor.shape[2:]
            direct_roi = get_direct_roi_size(self.config)
            mode = select_inference_mode(orig_shape, self.config)
            if mode == "direct_padded":
                # Symmetric pad to the direct-path ROI (needed for
                # InstanceNorm consistency while preserving crop geometry).
                pad = []
                for dim in reversed(range(3)):
                    diff = direct_roi[dim] - orig_shape[dim]
                    half = diff // 2
                    pad.extend([half, diff - half])
                padded = torch.nn.functional.pad(tensor, pad, mode="constant", value=0.0)
                logits = self._predictor(padded)
                slices = []
                for dim in range(3):
                    diff = direct_roi[dim] - orig_shape[dim]
                    half = diff // 2
                    slices.append(slice(half, half + orig_shape[dim]))
                logits = logits[:, :, slices[0], slices[1], slices[2]]
            elif mode == "direct_full":
                # Experimental path for factor ablations: one full-crop forward
                # without sliding-window tiling.
                logits = self._predictor(tensor)
            else:
                # Oversized: use sliding_window_inference
                logits = sliding_window_inference(
                    inputs=tensor,
                    roi_size=self.config.roi_size,
                    sw_batch_size=self.config.sw_batch_size,
                    predictor=self._predictor,
                    overlap=self.config.overlap,
                    mode="gaussian",
                )

        prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Post-process
        if self.config.apply_void_fill:
            prediction = fill_internal_voids(
                prediction,
                background_class=self.config.background_class,
                void_class=self.config.void_class,
            )

        return prediction


# =============================================================================
# Result dataclass and utility functions
# =============================================================================


@dataclass
class BBoxSegmentationResult:
    """Result from segmenting a single bounding box region."""

    bbox_index: int
    prediction: np.ndarray
    original_bbox: np.ndarray
    expanded_bbox: list[int]
    crop_shape: tuple[int, ...]


def bbox_shape(bbox: np.ndarray | list[int]) -> tuple[int, int, int]:
    """Return bbox size as (x, y, z)."""
    return (
        int(bbox[1]) - int(bbox[0]),
        int(bbox[3]) - int(bbox[2]),
        int(bbox[5]) - int(bbox[4]),
    )


def fits_in_roi(shape: tuple[int, int, int], roi_size: tuple[int, int, int]) -> bool:
    """Check whether a crop shape can use direct/batch inference."""
    return all(size <= limit for size, limit in zip(shape, roi_size))


def get_direct_roi_size(cfg: SegmentationConfig) -> tuple[int, int, int]:
    """Return the ROI used to decide direct padded inference.

    By default this matches the sliding-window ROI. Experiments can enlarge the
    direct ROI to admit more crops without trimming any real context.
    """
    return cfg.direct_roi_size or cfg.roi_size


def validate_inference_config(cfg: SegmentationConfig) -> None:
    """Validate experiment-only inference controls."""
    if cfg.use_adaptive_margin and cfg.use_guarded_adaptive_margin:
        msg = "use_adaptive_margin and use_guarded_adaptive_margin cannot both be enabled"
        raise ValueError(msg)
    if cfg.force_sliding_window and cfg.force_direct_full_crop:
        msg = "force_sliding_window and force_direct_full_crop cannot both be enabled"
        raise ValueError(msg)
    if cfg.use_trt and cfg.force_direct_full_crop:
        msg = "force_direct_full_crop is only supported with the PyTorch segmentation backend"
        raise ValueError(msg)
    direct_roi = get_direct_roi_size(cfg)
    if any(direct < base for direct, base in zip(direct_roi, cfg.roi_size)):
        msg = "direct_roi_size must be greater than or equal to roi_size in every dimension"
        raise ValueError(msg)
    if cfg.use_trt and direct_roi != cfg.roi_size:
        msg = "direct_roi_size experiments are only supported with the PyTorch segmentation backend"
        raise ValueError(msg)


def select_inference_mode(
    shape: tuple[int, int, int],
    cfg: SegmentationConfig,
) -> str:
    """Choose the inference path for a crop."""
    validate_inference_config(cfg)
    if cfg.force_sliding_window:
        return "sliding_window"
    if cfg.force_direct_full_crop:
        return "direct_full"
    if fits_in_roi(shape, get_direct_roi_size(cfg)):
        return "direct_padded"
    return "sliding_window"


def expand_bbox_fixed(
    bbox: np.ndarray,
    volume_shape: tuple[int, ...],
    margin: int,
) -> list[int]:
    """Expand bounding box with a fixed margin, clamped to volume boundaries.

    Args:
        bbox: [x_min, x_max, y_min, y_max, z_min, z_max]
        volume_shape: Shape of the full volume (X, Y, Z)
        margin: Expansion margin in voxels

    Returns:
        Expanded bbox as [x_min, x_max, y_min, y_max, z_min, z_max]
    """
    return [
        max(int(bbox[0]) - margin, 0),
        min(int(bbox[1]) + margin, volume_shape[0]),
        max(int(bbox[2]) - margin, 0),
        min(int(bbox[3]) + margin, volume_shape[1]),
        max(int(bbox[4]) - margin, 0),
        min(int(bbox[5]) + margin, volume_shape[2]),
    ]


def _expand_interval_adaptive(
    start: int,
    end: int,
    limit: int,
    margin: int,
    roi_limit: int,
) -> tuple[int, int]:
    """Expand one bbox dimension without exceeding the direct-path ROI.

    The rule is:
    - keep up to `margin` voxels of context on each side,
    - but never grow the crop beyond `roi_limit` in this dimension,
    - and if the raw object is already larger than `roi_limit`, do not add
      extra context on this axis and let sliding-window handle it.
    """
    size = end - start
    if margin <= 0 or size >= roi_limit:
        return start, end

    max_total_context = min(roi_limit - size, margin * 2)
    if max_total_context <= 0:
        return start, end

    left_room = start
    right_room = limit - end
    left_cap = min(margin, left_room)
    right_cap = min(margin, right_room)

    left = min(left_cap, max_total_context // 2)
    right = min(right_cap, max_total_context - left)
    remaining = max_total_context - left - right

    # If one side was clipped by the volume boundary, reuse the remaining
    # context budget on the opposite side, still capped by margin and ROI.
    if remaining > 0:
        extra_right = min(remaining, right_cap - right)
        right += extra_right
        remaining -= extra_right
    if remaining > 0:
        extra_left = min(remaining, left_cap - left)
        left += extra_left

    return start - left, end + right


def expand_bbox(
    bbox: np.ndarray,
    volume_shape: tuple[int, ...],
    margin: int,
    roi_size: tuple[int, int, int],
) -> list[int]:
    """Expand bbox with adaptive per-dimension margin.

    The expansion preserves as much configured context as possible while
    keeping each dimension within `roi_size` whenever the raw bbox already
    fits that ROI. Truly oversized objects still fall back to sliding-window.
    """
    x0, x1 = _expand_interval_adaptive(
        int(bbox[0]),
        int(bbox[1]),
        int(volume_shape[0]),
        int(margin),
        int(roi_size[0]),
    )
    y0, y1 = _expand_interval_adaptive(
        int(bbox[2]),
        int(bbox[3]),
        int(volume_shape[1]),
        int(margin),
        int(roi_size[1]),
    )
    z0, z1 = _expand_interval_adaptive(
        int(bbox[4]),
        int(bbox[5]),
        int(volume_shape[2]),
        int(margin),
        int(roi_size[2]),
    )
    return [x0, x1, y0, y1, z0, z1]


def select_margin_mode(cfg: SegmentationConfig) -> str:
    """Choose the bbox-margin policy for this run."""
    if cfg.use_guarded_adaptive_margin:
        return "guarded"
    if cfg.use_adaptive_margin:
        return "adaptive"
    return "fixed"


def select_inference_policy(cfg: SegmentationConfig) -> str:
    """Return a stable string label for the inference-path policy."""
    if cfg.force_sliding_window:
        return "force_sliding_window"
    if cfg.force_direct_full_crop:
        return "force_direct_full_crop"
    return "auto"


def select_guarded_bbox(
    raw_shape: tuple[int, int, int],
    fixed_expanded: list[int],
    adaptive_expanded: list[int],
    roi_size: tuple[int, int, int],
    max_loss_xy: int,
    max_loss_z: int,
) -> tuple[list[int], dict[str, object]]:
    """Apply a conservative guardrail before shrinking context.

    The guarded policy only reuses the adaptive crop on dimensions where:
    - the raw bbox already fits the ROI
    - the fixed-margin crop overflows only because of added context
    - the total context removed on that dimension stays within the configured cap

    If any overflowing dimension exceeds the guardrail, the bbox falls back to
    the original fixed-margin crop and stays on the baseline sliding-window path.
    """
    fixed_shape = bbox_shape(fixed_expanded)
    adaptive_shape = bbox_shape(adaptive_expanded)

    if not fits_in_roi(raw_shape, roi_size):
        return fixed_expanded, {
            "status": "raw_oversized",
            "loss_by_dim": [0, 0, 0],
            "overflow_dims": [],
            "rejected_xy": False,
            "rejected_z": False,
        }

    guarded_expanded = fixed_expanded.copy()
    loss_by_dim = [0, 0, 0]
    overflow_dims: list[str] = []
    rejected_xy = False
    rejected_z = False
    dim_labels = ("x", "y", "z")

    for dim, label in enumerate(dim_labels):
        if fixed_shape[dim] <= roi_size[dim]:
            continue
        if raw_shape[dim] > roi_size[dim]:
            return fixed_expanded, {
                "status": "raw_oversized",
                "loss_by_dim": loss_by_dim,
                "overflow_dims": overflow_dims,
                "rejected_xy": rejected_xy,
                "rejected_z": rejected_z,
            }

        overflow_dims.append(label)
        loss = fixed_shape[dim] - adaptive_shape[dim]
        loss_by_dim[dim] = int(loss)
        max_loss = max_loss_z if dim == 2 else max_loss_xy
        if loss > max_loss:
            if dim == 2:
                rejected_z = True
            else:
                rejected_xy = True

    if not overflow_dims:
        return fixed_expanded, {
            "status": "passthrough",
            "loss_by_dim": loss_by_dim,
            "overflow_dims": overflow_dims,
            "rejected_xy": rejected_xy,
            "rejected_z": rejected_z,
        }

    if rejected_xy or rejected_z:
        return fixed_expanded, {
            "status": "rejected",
            "loss_by_dim": loss_by_dim,
            "overflow_dims": overflow_dims,
            "rejected_xy": rejected_xy,
            "rejected_z": rejected_z,
        }

    for dim in range(3):
        if dim_labels[dim] not in overflow_dims:
            continue
        guarded_expanded[2 * dim] = adaptive_expanded[2 * dim]
        guarded_expanded[2 * dim + 1] = adaptive_expanded[2 * dim + 1]

    return guarded_expanded, {
        "status": "accepted",
        "loss_by_dim": loss_by_dim,
        "overflow_dims": overflow_dims,
        "rejected_xy": rejected_xy,
        "rejected_z": rejected_z,
    }


def build_bbox_expansion_plan(
    bbox: np.ndarray,
    volume_shape: tuple[int, ...],
    cfg: SegmentationConfig,
) -> dict[str, object]:
    """Build fixed/adaptive/selected crop plans for one bbox."""
    direct_roi = get_direct_roi_size(cfg)
    raw_shape = bbox_shape(bbox)
    fixed_expanded = expand_bbox_fixed(bbox, volume_shape, cfg.margin)
    fixed_shape = bbox_shape(fixed_expanded)
    adaptive_expanded = expand_bbox(bbox, volume_shape, cfg.margin, direct_roi)
    adaptive_shape = bbox_shape(adaptive_expanded)
    guard_info: dict[str, object] | None = None

    margin_mode = select_margin_mode(cfg)
    if margin_mode == "guarded":
        selected_expanded, guard_info = select_guarded_bbox(
            raw_shape=raw_shape,
            fixed_expanded=fixed_expanded,
            adaptive_expanded=adaptive_expanded,
            roi_size=direct_roi,
            max_loss_xy=cfg.guard_max_loss_xy,
            max_loss_z=cfg.guard_max_loss_z,
        )
    elif margin_mode == "adaptive":
        selected_expanded = adaptive_expanded
    else:
        selected_expanded = fixed_expanded

    return {
        "raw_shape": raw_shape,
        "fixed_expanded": fixed_expanded,
        "fixed_shape": fixed_shape,
        "adaptive_expanded": adaptive_expanded,
        "adaptive_shape": adaptive_shape,
        "selected_expanded": selected_expanded,
        "selected_shape": bbox_shape(selected_expanded),
        "guard_info": guard_info,
    }


def summarize_crop_shapes(shapes: list[tuple[int, int, int]]) -> dict[str, list[float] | list[int]]:
    """Summarize crop shapes for lightweight run-to-run traceability."""
    if not shapes:
        return {"min": [], "mean": [], "max": []}

    arr = np.asarray(shapes, dtype=np.float32)
    return {
        "min": arr.min(axis=0).astype(int).tolist(),
        "mean": np.round(arr.mean(axis=0), 2).tolist(),
        "max": arr.max(axis=0).astype(int).tolist(),
    }


def summarize_shape_deltas(
    baseline_shapes: list[tuple[int, int, int]],
    variant_shapes: list[tuple[int, int, int]],
) -> dict[str, list[float] | list[int]]:
    """Summarize per-dimension size deltas between two crop-shape lists."""
    if not baseline_shapes or not variant_shapes:
        return {"min": [], "mean": [], "max": []}

    base_arr = np.asarray(baseline_shapes, dtype=np.float32)
    variant_arr = np.asarray(variant_shapes, dtype=np.float32)
    delta = variant_arr - base_arr
    return {
        "min": delta.min(axis=0).astype(int).tolist(),
        "mean": np.round(delta.mean(axis=0), 2).tolist(),
        "max": delta.max(axis=0).astype(int).tolist(),
    }


# =============================================================================
# High-level inference functions
# =============================================================================


def segment_bboxes(
    engine: SegmentationInference,
    volume: np.ndarray,
    bboxes: np.ndarray,
    save_dir: Path | None = None,
    save_predictions: bool = True,
    save_raw_crops: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
    batch_size: int = 8,
) -> list[BBoxSegmentationResult]:
    """Segment multiple bounding box regions from a volume.

    Uses batched inference for crops that fit within roi_size to reduce
    per-crop overhead. Falls back to sliding_window_inference for
    crops larger than roi_size.

    Args:
        engine: Initialized SegmentationInference instance
        volume: Full 3D volume array
        bboxes: Array of bboxes [N, 6] format [x_min, x_max, y_min, y_max, z_min, z_max]
        save_dir: Optional directory for segmentation artifacts.
                  When provided, always writes segmentation_stats.json.
        save_predictions: Whether to save per-bbox prediction masks under
                  save_dir/pred/.
        save_raw_crops: Whether to save per-bbox raw crops under save_dir/img/.
        progress_callback: Optional callback(current_idx, total) for progress updates
        batch_size: Number of crops to batch together for inference

    Returns:
        List of BBoxSegmentationResult for each bbox
    """
    cfg = engine.config
    validate_inference_config(cfg)
    engine.load_model()
    device = engine.device
    sliding_roi = cfg.roi_size
    direct_roi = get_direct_roi_size(cfg)

    # Phase 1: extract & normalize all crops, classify into batchable vs oversized
    batch_items = []   # crops that fit in roi_size → batch inference
    direct_items = []  # crops that use single-pass full-crop inference
    large_items = []   # crops larger than roi_size → sliding_window
    raw_shapes: list[tuple[int, int, int]] = []
    fixed_shapes: list[tuple[int, int, int]] = []
    adaptive_shapes: list[tuple[int, int, int]] = []
    selected_shapes: list[tuple[int, int, int]] = []
    stats = {
        "roi_size": list(sliding_roi),
        "direct_roi_size": list(direct_roi),
        "configured_margin": int(cfg.margin),
        "margin_mode": select_margin_mode(cfg),
        "inference_policy": select_inference_policy(cfg),
        "total_bbox_count": int(len(bboxes)),
        "processed_bbox_count": 0,
        "skipped_small_count": 0,
        "original_fit_count": 0,
        "fixed_margin_fit_count": 0,
        "adaptive_margin_fit_count": 0,
        "selected_margin_fit_count": 0,
        "direct_path_count": 0,
        "direct_batch_count": 0,
        "direct_full_count": 0,
        "sliding_window_count": 0,
    }
    if cfg.use_guarded_adaptive_margin:
        stats["guard_policy"] = {
            "max_loss_xy": int(cfg.guard_max_loss_xy),
            "max_loss_z": int(cfg.guard_max_loss_z),
        }
        stats["guard_passthrough_count"] = 0
        stats["guard_candidate_count"] = 0
        stats["guard_accepted_count"] = 0
        stats["guard_rejected_count"] = 0
        stats["guard_raw_oversized_count"] = 0
        stats["guard_rejected_xy_count"] = 0
        stats["guard_rejected_z_count"] = 0

    for idx, bbox in enumerate(bboxes):
        plan = build_bbox_expansion_plan(bbox, volume.shape, cfg)
        raw_shape = plan["raw_shape"]
        fixed_shape = plan["fixed_shape"]
        adaptive_shape = plan["adaptive_shape"]
        expanded = plan["selected_expanded"]
        region_size = plan["selected_shape"]
        guard_info = plan["guard_info"]

        raw_shapes.append(raw_shape)
        fixed_shapes.append(fixed_shape)
        adaptive_shapes.append(adaptive_shape)
        selected_shapes.append(region_size)

        if fits_in_roi(raw_shape, direct_roi):
            stats["original_fit_count"] += 1
        if fits_in_roi(fixed_shape, direct_roi):
            stats["fixed_margin_fit_count"] += 1
        if fits_in_roi(adaptive_shape, direct_roi):
            stats["adaptive_margin_fit_count"] += 1
        if fits_in_roi(region_size, direct_roi):
            stats["selected_margin_fit_count"] += 1

        if cfg.use_guarded_adaptive_margin and guard_info is not None:
            guard_status = guard_info["status"]
            if guard_status == "passthrough":
                stats["guard_passthrough_count"] += 1
            elif guard_status == "raw_oversized":
                stats["guard_raw_oversized_count"] += 1
            else:
                stats["guard_candidate_count"] += 1
                if guard_status == "accepted":
                    stats["guard_accepted_count"] += 1
                else:
                    stats["guard_rejected_count"] += 1
                    if guard_info["rejected_xy"]:
                        stats["guard_rejected_xy_count"] += 1
                    if guard_info["rejected_z"]:
                        stats["guard_rejected_z_count"] += 1

        if any(s < 5 for s in region_size):
            log(f"Skipping bbox {idx}: region too small {region_size}")
            stats["skipped_small_count"] += 1
            continue

        crop = volume[
            expanded[0] : expanded[1],
            expanded[2] : expanded[3],
            expanded[4] : expanded[5],
        ]

        # Normalize on GPU
        tensor = torch.from_numpy(crop.astype(np.float32)).to(device)
        if cfg.apply_normalization:
            tensor = engine.normalize_gpu(tensor)

        inference_mode = select_inference_mode(region_size, cfg)

        item = {
            "idx": idx, "bbox": bbox, "expanded": expanded,
            "crop_shape": crop.shape, "tensor": tensor, "inference_mode": inference_mode,
        }

        if inference_mode == "direct_padded":
            batch_items.append(item)
        elif inference_mode == "direct_full":
            direct_items.append(item)
        else:
            large_items.append(item)

    stats["processed_bbox_count"] = int(len(batch_items) + len(direct_items) + len(large_items))
    stats["direct_batch_count"] = int(len(batch_items))
    stats["direct_full_count"] = int(len(direct_items))
    stats["direct_path_count"] = int(len(batch_items) + len(direct_items))
    stats["sliding_window_count"] = int(len(large_items))
    stats["shape_summary"] = {
        "raw_bbox": summarize_crop_shapes(raw_shapes),
        "fixed_margin_crop": summarize_crop_shapes(fixed_shapes),
        "adaptive_margin_crop": summarize_crop_shapes(adaptive_shapes),
        "selected_margin_crop": summarize_crop_shapes(selected_shapes),
    }
    stats["context_summary"] = {
        "fixed_margin_context": summarize_shape_deltas(raw_shapes, fixed_shapes),
        "adaptive_margin_context": summarize_shape_deltas(raw_shapes, adaptive_shapes),
        "selected_margin_context": summarize_shape_deltas(raw_shapes, selected_shapes),
    }
    engine.last_segmentation_stats = stats

    log(
        "Segmentation path stats: "
        f"sliding_roi={stats['roi_size']}, direct_roi={stats['direct_roi_size']}, "
        f"total={stats['total_bbox_count']}, processed={stats['processed_bbox_count']}, "
        f"raw_fit={stats['original_fit_count']}, fixed_margin_fit={stats['fixed_margin_fit_count']}, "
        f"adaptive_fit={stats['adaptive_margin_fit_count']}, selected_fit={stats['selected_margin_fit_count']}, "
        f"direct={stats['direct_path_count']} (batch={stats['direct_batch_count']}, full={stats['direct_full_count']}), "
        f"sliding={stats['sliding_window_count']}, skipped_small={stats['skipped_small_count']}"
    )
    log(
        "Crop size summary "
        f"raw_mean={stats['shape_summary']['raw_bbox']['mean']}, "
        f"fixed_mean={stats['shape_summary']['fixed_margin_crop']['mean']}, "
        f"adaptive_mean={stats['shape_summary']['adaptive_margin_crop']['mean']}, "
        f"selected_mean={stats['shape_summary']['selected_margin_crop']['mean']}"
    )
    log(
        "Inference groups: "
        f"direct_padded={len(batch_items)}, direct_full={len(direct_items)}, "
        f"sliding_window={len(large_items)}"
    )
    if cfg.use_guarded_adaptive_margin:
        log(
            "Guarded margin stats: "
            f"passthrough={stats['guard_passthrough_count']}, "
            f"candidates={stats['guard_candidate_count']}, "
            f"accepted={stats['guard_accepted_count']}, "
            f"rejected={stats['guard_rejected_count']} "
            f"(xy={stats['guard_rejected_xy_count']}, z={stats['guard_rejected_z_count']}), "
            f"raw_oversized={stats['guard_raw_oversized_count']}"
        )

    # Phase 2: batched inference for crops that fit in the direct-path ROI
    results_dict = {}

    if batch_items:
        # Pad each crop to the direct-path ROI and stack into batches.
        for start in range(0, len(batch_items), batch_size):
            batch = batch_items[start : start + batch_size]
            padded_tensors = []

            for item in batch:
                t = item["tensor"]  # shape (X, Y, Z)
                # Symmetric pad to the direct-path ROI.
                pad = []
                for dim in reversed(range(3)):
                    diff = direct_roi[dim] - t.shape[dim]
                    half = diff // 2
                    pad.extend([half, diff - half])
                padded = torch.nn.functional.pad(t, pad, mode="constant", value=0.0)
                padded_tensors.append(padded.unsqueeze(0).unsqueeze(0))  # [1, 1, X, Y, Z]

            batch_tensor = torch.cat(padded_tensors, dim=0)  # [B, 1, direct_roi...]

            with torch.no_grad():
                batch_logits = engine._predictor(batch_tensor)  # [B, C, direct_roi...]

            for i, item in enumerate(batch):
                logits_i = batch_logits[i : i + 1]  # [1, C, direct_roi...]
                # Symmetric unpad to original crop shape
                s = item["crop_shape"]
                slices = []
                for dim in range(3):
                    diff = direct_roi[dim] - s[dim]
                    half = diff // 2
                    slices.append(slice(half, half + s[dim]))
                logits_cropped = logits_i[:, :, slices[0], slices[1], slices[2]]
                pred = torch.argmax(logits_cropped, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                if cfg.apply_void_fill:
                    pred = fill_internal_voids(
                        pred, background_class=cfg.background_class,
                        void_class=cfg.void_class,
                    )

                results_dict[item["idx"]] = BBoxSegmentationResult(
                    bbox_index=item["idx"],
                    prediction=pred,
                    original_bbox=item["bbox"],
                    expanded_bbox=item["expanded"],
                    crop_shape=item["crop_shape"],
                )

            if progress_callback:
                progress_callback(min(start + batch_size, len(batch_items)), len(bboxes))

    # Phase 3: one-shot direct inference for full-crop experiments
    for item in direct_items:
        tensor = item["tensor"].unsqueeze(0).unsqueeze(0)  # [1, 1, X, Y, Z]
        with torch.no_grad():
            logits = engine._predictor(tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        if cfg.apply_void_fill:
            pred = fill_internal_voids(
                pred, background_class=cfg.background_class,
                void_class=cfg.void_class,
            )

        results_dict[item["idx"]] = BBoxSegmentationResult(
            bbox_index=item["idx"],
            prediction=pred,
            original_bbox=item["bbox"],
            expanded_bbox=item["expanded"],
            crop_shape=item["crop_shape"],
        )

    # Phase 4: sliding_window for oversized crops
    for item in large_items:
        tensor = item["tensor"].unsqueeze(0).unsqueeze(0)  # [1, 1, X, Y, Z]
        with torch.no_grad():
            logits = sliding_window_inference(
                inputs=tensor,
                roi_size=sliding_roi,
                sw_batch_size=cfg.sw_batch_size,
                predictor=engine._predictor,
                overlap=cfg.overlap,
                mode="gaussian",
            )
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        if cfg.apply_void_fill:
            pred = fill_internal_voids(
                pred, background_class=cfg.background_class,
                void_class=cfg.void_class,
            )

        results_dict[item["idx"]] = BBoxSegmentationResult(
            bbox_index=item["idx"],
            prediction=pred,
            original_bbox=item["bbox"],
            expanded_bbox=item["expanded"],
            crop_shape=item["crop_shape"],
        )

    # Phase 5: optional save + collect results in original order
    results = []
    for idx in sorted(results_dict.keys()):
        r = results_dict[idx]
        if save_dir is not None and (save_predictions or save_raw_crops):
            crop = volume[
                r.expanded_bbox[0] : r.expanded_bbox[1],
                r.expanded_bbox[2] : r.expanded_bbox[3],
                r.expanded_bbox[4] : r.expanded_bbox[5],
            ]
            if save_raw_crops:
                img_path = save_dir / "img" / f"img_{idx}.nii.gz"
                nib.save(nib.Nifti1Image(crop.astype(np.float32), np.eye(4)), str(img_path))
            if save_predictions:
                pred_path = save_dir / "pred" / f"pred_{idx}.nii.gz"
                nib.save(nib.Nifti1Image(r.prediction.astype(np.float32), np.eye(4)), str(pred_path))
        results.append(r)

    if save_dir is not None:
        stats_path = save_dir / "segmentation_stats.json"
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        log(f"Saved segmentation stats to: {stats_path}")

    return results


def assemble_full_volume(
    results: list[BBoxSegmentationResult],
    volume_shape: tuple[int, ...],
) -> np.ndarray:
    """Assemble per-bbox predictions into a full volume.

    Args:
        results: List of BBoxSegmentationResult from segment_bboxes
        volume_shape: Shape of the output volume

    Returns:
        Full volume with predictions placed at their bbox locations
    """
    output = np.zeros(volume_shape, dtype=np.float32)

    for result in results:
        bbox = result.expanded_bbox
        pred = result.prediction

        # Direct placement - prediction shape matches expanded bbox
        output[
            bbox[0] : bbox[0] + pred.shape[0],
            bbox[2] : bbox[2] + pred.shape[1],
            bbox[4] : bbox[4] + pred.shape[2],
        ] = pred

    return output


if __name__ == "__main__":
    log("3D Segmentation Inference Module")
    log("\nUsage:")
    log("  engine = SegmentationInference(model_path, config)")
    log("  results = segment_bboxes(engine, volume, bboxes, save_dir)")
    log("  full_vol = assemble_full_volume(results, volume.shape)")
