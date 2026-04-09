#!/usr/bin/env python3
"""
Finetune a pruned MONAI BasicUNet model.

Uses the same data pipeline, loss function, and evaluation as train.py
to ensure fair comparison with the baseline model.

Key design decisions (matching train.py exactly):
- Same transforms: LoadImaged + EnsureChannelFirstd + Orientationd + ClipZScoreNormalize
  + SpatialPadd + RandFlipd(3 axes) + RandSpatialCropd
- Same loss: 0.5 * CrossEntropy + 0.5 * Dice (ignoring class 6)
- Same evaluation: sliding_window_inference with roi=(112,112,80)

Usage:
  python finetune_pruned.py \\
    --pruned_model_path ../output/pruned_model.ckpt \\
    --data_dir /path/to/3ddl-dataset/data \\
    --output_dir ../runs/finetune_pruned \\
    --epochs 50 \\
    --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Callable
from contextlib import nullcontext, suppress
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.networks.nets import BasicUNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandFlipd,
    RandSpatialCropd,
    SpatialPadd,
)
from monai.utils import set_determinism

# Add 3ddl-dataset to path (same as train.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "3ddl-dataset"))
from dataset_loader import BumpDataset


# ============================================================
# Data Pipeline (identical to train.py)
# ============================================================

class ClipZScoreNormalizeD(MapTransform):
    """Per-sample robust normalization: clip to [p1, p99] then z-score."""
    def __init__(self, keys: list[str]):
        super().__init__(keys)
        self.eps = 1e-8

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            arr = d.get(key)
            if arr is None:
                continue
            flat = arr.reshape(-1) if arr.ndim == 3 else arr.reshape(arr.shape[0], -1).reshape(-1)
            p1 = np.percentile(flat, 1)
            p99 = np.percentile(flat, 99)
            clipped = np.clip(arr, p1, p99)
            mean = clipped.mean()
            std = clipped.std()
            d[key] = ((clipped - mean) / (std + self.eps)).astype(np.float32)
        return d


def build_datalists(data_dir: Path) -> tuple[list[dict], list[dict]]:
    """Build MONAI-style train/test datalists using BumpDataset."""
    ds = BumpDataset(data_dir=str(data_dir))
    cfg = ds.config
    splits = ds.split(test_serial_numbers=cfg.get("test_serial_numbers"))
    train_ds, test_ds = splits["train"], splits["test"]

    def to_monai_list(bump_ds) -> list[dict]:
        result = []
        for i in range(len(bump_ds)):
            meta = bump_ds.get_metadata(i)
            result.append({
                "image": meta["image_path"],
                "label": meta["label_path"],
                "id": meta["pair_id"],
            })
        return result

    return to_monai_list(train_ds), to_monai_list(test_ds)


def subset_datalist(datalist: list[dict], ratio: float, seed: int) -> list[dict]:
    """Subset a datalist to a given ratio of samples."""
    if ratio >= 0.999:
        return list(datalist)
    n = max(1, int(len(datalist) * ratio))
    rng = random.Random(seed)
    idxs = list(range(len(datalist)))
    rng.shuffle(idxs)
    idxs = sorted(idxs[:n])
    return [datalist[i] for i in idxs]


def get_transforms(roi: tuple[int, int, int] = (112, 112, 80)):
    """Build train and validation transforms (same as train.py)."""
    def build_seq(include_crop: bool, include_aug: bool, training: bool):
        seq = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ClipZScoreNormalizeD(keys=["image"]),
        ]
        if training:
            seq.append(SpatialPadd(keys=["image", "label"], spatial_size=roi))
        if include_aug:
            seq.extend([
                RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
                RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
                RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
            ])
        if include_crop:
            seq.append(RandSpatialCropd(keys=["image", "label"], roi_size=roi, random_size=False))
        return seq

    train_tf = Compose(build_seq(include_crop=True, include_aug=True, training=True))
    val_tf = Compose(build_seq(include_crop=False, include_aug=False, training=False))
    return train_tf, val_tf


# ============================================================
# Loss (identical to train.py)
# ============================================================

def dice_loss_masked(
    logits: torch.Tensor, target: torch.Tensor, ignore_mask: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Compute soft Dice loss over classes 0..4 on voxels where label != 6."""
    probs = F.softmax(logits, dim=1)
    target_clamped = torch.clamp(target, 0, 4).long()
    gt_oh = F.one_hot(target_clamped.squeeze(1), num_classes=5)
    gt_onehot = gt_oh.permute(0, 4, 1, 2, 3).to(probs.dtype)
    mask = ignore_mask.float().expand(-1, 5, -1, -1, -1)
    inter = torch.sum(probs * gt_onehot * mask, dim=(0, 2, 3, 4))
    denom = torch.sum(probs * mask + gt_onehot * mask, dim=(0, 2, 3, 4))
    dice_per_class = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice_per_class.mean()


def distillation_loss_masked(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    ignore_mask: torch.Tensor,
    target: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Masked KL distillation over valid voxels only."""
    t = max(float(temperature), 1e-6)
    student_log_probs = F.log_softmax(student_logits.float() / t, dim=1)
    teacher_probs = F.softmax(teacher_logits.float() / t, dim=1)
    # Sum KL over class dimension, then average over valid spatial voxels.
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=1, keepdim=True)
    mask = ignore_mask.float()
    if target is not None and class_weights is not None:
        target_clamped = torch.clamp(target.long(), 0, len(class_weights) - 1)
        voxel_weights = class_weights[target_clamped.squeeze(1)].unsqueeze(1).to(mask.dtype)
        weighted_mask = mask * voxel_weights
    else:
        weighted_mask = mask
    denom = weighted_mask.sum().clamp_min(1.0)
    return (kl * weighted_mask).sum() * (t * t) / denom


def feature_attention_loss_masked(
    student_feature: torch.Tensor,
    teacher_feature: torch.Tensor,
    ignore_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Attention-transfer style feature KD that is channel-count agnostic.

    The student and teacher widths differ after pruning, so we compare normalized
    spatial attention maps instead of raw feature tensors.
    """
    student_attention = student_feature.float().pow(2).mean(dim=1, keepdim=True)
    teacher_attention = teacher_feature.float().pow(2).mean(dim=1, keepdim=True)

    if teacher_attention.shape[2:] != student_attention.shape[2:]:
        teacher_attention = F.interpolate(teacher_attention, size=student_attention.shape[2:], mode="trilinear", align_corners=False)

    valid_mask = ignore_mask.float()
    if valid_mask.shape[2:] != student_attention.shape[2:]:
        valid_mask = F.interpolate(valid_mask, size=student_attention.shape[2:], mode="nearest")

    student_vec = student_attention.flatten(1)
    teacher_vec = teacher_attention.flatten(1)
    student_vec = F.normalize(student_vec, dim=1, eps=1e-6)
    teacher_vec = F.normalize(teacher_vec, dim=1, eps=1e-6)

    diff = (student_vec - teacher_vec).pow(2).view_as(student_attention)
    denom = valid_mask.sum().clamp_min(1.0)
    return (diff * valid_mask).sum() / denom


def resolve_amp_dtype(amp_dtype: str) -> torch.dtype:
    if amp_dtype == "bf16":
        return torch.bfloat16
    if amp_dtype == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported amp dtype: {amp_dtype}")


def make_autocast_context(device: torch.device, enabled: bool, amp_dtype: str):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=resolve_amp_dtype(amp_dtype))


def parse_class_weight_arg(class_weight_text: str) -> list[float]:
    values = [v.strip() for v in class_weight_text.split(",") if v.strip()]
    if len(values) != 5:
        raise ValueError("--kd-class-weights must contain exactly 5 comma-separated values")
    weights = [float(v) for v in values]
    if any(w <= 0 for w in weights):
        raise ValueError("--kd-class-weights values must be positive")
    return weights


def parse_feature_layers_arg(layer_text: str) -> list[str]:
    layers = [part.strip() for part in layer_text.split(",") if part.strip()]
    if not layers:
        raise ValueError("--feature-distill-layers must contain at least one module name when feature KD is enabled")
    return layers


class FeatureHookBank:
    def __init__(self, model: torch.nn.Module, layer_names: list[str]):
        self.layer_names = layer_names
        self.features: dict[str, torch.Tensor] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        modules = dict(model.named_modules())
        missing = [name for name in layer_names if name not in modules]
        if missing:
            raise ValueError(f"Unknown feature distill layers: {missing}")
        for name in layer_names:
            self.handles.append(modules[name].register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str) -> Callable:
        def hook(_module, _inputs, output):
            self.features[name] = output

        return hook

    def clear(self) -> None:
        self.features.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def feature_distillation_loss(
    student_features: dict[str, torch.Tensor],
    teacher_features: dict[str, torch.Tensor],
    ignore_mask: torch.Tensor,
    layer_names: list[str],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for layer_name in layer_names:
        student_feature = student_features.get(layer_name)
        teacher_feature = teacher_features.get(layer_name)
        if student_feature is None or teacher_feature is None:
            continue
        losses.append(feature_attention_loss_masked(student_feature, teacher_feature, ignore_mask))
    if not losses:
        return ignore_mask.new_tensor(0.0)
    return torch.stack(losses).mean()


def load_basicunet_model(model_path: str, model_format: str, device: torch.device) -> tuple[BasicUNet, tuple[int, ...]]:
    """Load either a baseline state_dict checkpoint or a pruned checkpoint."""
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
    return model.to(device), features


# ============================================================
# Evaluation (same sliding window as train.py)
# ============================================================

def compute_metrics(pred, gt) -> dict[int, dict[str, float]]:
    """Compute per-class Dice for classes 0..4; ignore class 6."""
    ignore_mask = gt != 6
    classes = [0, 1, 2, 3, 4]
    out = {}
    for cls in classes:
        pred_mask = pred == cls
        gt_mask = gt == cls
        pm = (pred_mask & ignore_mask).squeeze(1).cpu().numpy().astype(np.uint8)
        gm = (gt_mask & ignore_mask).squeeze(1).cpu().numpy().astype(np.uint8)
        inter = (pm & gm).sum(axis=(1, 2, 3))
        psum = pm.sum(axis=(1, 2, 3))
        gsum = gm.sum(axis=(1, 2, 3))
        both_empty = (psum + gsum) == 0
        valid = ~both_empty
        dice = np.full(pred.shape[0], np.nan, dtype=np.float32)
        dice[valid] = (2.0 * inter[valid]) / (psum[valid] + gsum[valid] + 1e-8)
        dice[both_empty] = 1.0
        out[cls] = {"dice": float(np.nanmean(dice)) if np.any(~np.isnan(dice)) else 0.0}
    return out


def evaluate(net, dl, device, roi=(112, 112, 80), max_cases: int | None = None):
    """Evaluate model using sliding window inference (same as train.py)."""
    net.eval()
    classes = [0, 1, 2, 3, 4]
    sums = {c: {"dice": 0.0, "n": 0} for c in classes}

    with torch.no_grad():
        for i, batch in enumerate(dl):
            if max_cases is not None and i >= max_cases:
                break
            img = batch["image"].to(device)
            gt = batch["label"].to(device)
            logits = sliding_window_inference(
                img, roi_size=roi, sw_batch_size=1, predictor=net,
                sw_device=device, device=device
            )
            pred = torch.argmax(logits, dim=1, keepdim=True)
            per_class = compute_metrics(pred.cpu(), gt.cpu())
            for c in classes:
                sums[c]["dice"] += per_class[c]["dice"]
                sums[c]["n"] += 1

    summary = {}
    for c in classes:
        n = max(sums[c]["n"], 1)
        summary[c] = sums[c]["dice"] / n

    avg_dice = float(np.mean(list(summary.values())))
    return avg_dice, summary


# ============================================================
# Training Loop
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Finetune pruned MONAI BasicUNet")
    parser.add_argument("--pruned_model_path", type=str, required=True,
                        help="Path to pruned model (output of prune_basicunet.py)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to 3ddl-dataset/data directory")
    parser.add_argument("--output_dir", type=str, default="runs/finetune_pruned",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (lower than training from scratch)")
    parser.add_argument("--subset_ratio", type=float, default=1.0, help="Proportion of train data to use (0.0-1.0)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_interval", type=int, default=5,
                        help="Evaluate every N epochs")
    parser.add_argument("--roi_x", type=int, default=112)
    parser.add_argument("--roi_y", type=int, default=112)
    parser.add_argument("--roi_z", type=int, default=80)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA AMP for finetuning")
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16"],
        help="Autocast dtype when --amp is enabled",
    )
    parser.add_argument("--max_eval_cases", type=int, default=None, help="Optional cap for quick validation runs")
    parser.add_argument("--teacher_model_path", type=str, default=None, help="Optional teacher checkpoint for KD")
    parser.add_argument(
        "--teacher_model_format",
        type=str,
        default="state_dict",
        choices=["state_dict", "pruned"],
        help="Teacher checkpoint format",
    )
    parser.add_argument(
        "--distill-weight",
        type=float,
        default=0.0,
        help="Weight on masked KL distillation loss; 0 disables KD",
    )
    parser.add_argument(
        "--distill-temperature",
        type=float,
        default=2.0,
        help="Temperature for teacher-student distillation",
    )
    parser.add_argument(
        "--distill-start-epoch",
        type=int,
        default=1,
        help="First epoch index (1-based) where KD becomes active; 1 means immediate KD",
    )
    parser.add_argument(
        "--kd-class-weights",
        type=str,
        default="1,1,1,1,1",
        help="Optional GT-class weights for KD as five comma-separated values for classes 0..4",
    )
    parser.add_argument(
        "--feature-distill-weight",
        type=float,
        default=0.0,
        help="Weight on feature-level attention KD; 0 disables feature KD",
    )
    parser.add_argument(
        "--feature-distill-layers",
        type=str,
        default="conv_0,down_1.convs,down_2.convs,down_3.convs,down_4.convs",
        help="Comma-separated module names used for feature KD",
    )
    parser.add_argument(
        "--save-eval-checkpoints",
        action="store_true",
        help="Save a checkpoint at every evaluation epoch for downstream checkpoint selection studies",
    )
    parser.add_argument("--no_timestamp", action="store_true",
                        help="Do not append timestamp to output_dir")
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    set_determinism(args.seed)

    if not args.no_timestamp:
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = Path(args.output_dir)
        args.output_dir = str(base.parent / f"{base.name}_{ts}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with suppress(Exception):
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    if args.distill_weight < 0:
        raise ValueError("--distill-weight must be non-negative")
    if args.feature_distill_weight < 0:
        raise ValueError("--feature-distill-weight must be non-negative")
    if args.distill_start_epoch < 1:
        raise ValueError("--distill-start-epoch must be >= 1")
    if (args.distill_weight > 0 or args.feature_distill_weight > 0) and not args.teacher_model_path:
        raise ValueError("--teacher_model_path is required when distillation is enabled")
    kd_class_weights = parse_class_weight_arg(args.kd_class_weights)
    feature_distill_layers = parse_feature_layers_arg(args.feature_distill_layers)

    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and args.amp_dtype == "fp16")
    kd_class_weights_tensor = torch.tensor(kd_class_weights, device=device, dtype=torch.float32)

    # Load pruned student model
    print(f"Loading pruned model: {args.pruned_model_path}")
    ckpt = torch.load(args.pruned_model_path, map_location="cpu", weights_only=False)
    features = tuple(ckpt["features"])
    print(f"Pruned architecture features: {features}")

    model = BasicUNet(spatial_dims=3, in_channels=1, out_channels=5, features=features)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,} ({n_params / 1e6:.2f}M)")

    teacher = None
    teacher_features = None
    student_feature_hooks: FeatureHookBank | None = None
    teacher_feature_hooks: FeatureHookBank | None = None
    if args.teacher_model_path:
        teacher, teacher_features = load_basicunet_model(args.teacher_model_path, args.teacher_model_format, device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        teacher_params = sum(p.numel() for p in teacher.parameters())
        print(f"Teacher model: {teacher_features}, params={teacher_params:,} ({teacher_params / 1e6:.2f}M)")
        if args.distill_weight <= 0 and args.feature_distill_weight <= 0:
            print("WARNING: teacher loaded but both logit and feature KD are disabled")
        if args.feature_distill_weight > 0:
            student_feature_hooks = FeatureHookBank(model, feature_distill_layers)
            teacher_feature_hooks = FeatureHookBank(teacher, feature_distill_layers)
            print(f"Feature KD layers: {feature_distill_layers}")

    if "pruning_info" in ckpt:
        info = ckpt["pruning_info"]
        print(f"Pruning ratio: {info.get('pruning_ratio', 'N/A')}")
        print(f"Original params: {info.get('original_params', 'N/A'):,}")
        print(f"Param reduction: {info.get('reduction_pct', 'N/A')}%")

    # Load data
    data_dir = Path(args.data_dir)
    print(f"\nLoading dataset from: {data_dir}")
    train_list, test_list = build_datalists(data_dir)
    train_list = subset_datalist(train_list, args.subset_ratio, args.seed)
    print(f"Train: {len(train_list)}, Test: {len(test_list)}")

    roi = (args.roi_x, args.roi_y, args.roi_z)
    tf_train, tf_val = get_transforms(roi=roi)

    ds_train = Dataset(train_list, transform=tf_train)
    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    ds_test = Dataset(test_list, transform=tf_val)
    dl_test = DataLoader(
        ds_test, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )

    # Optimizer: Adam with lower lr for finetuning
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Evaluate BEFORE finetuning (to see accuracy drop from pruning)
    print("\nEvaluating pruned model BEFORE finetuning...")
    pre_dice, pre_per_class = evaluate(model, dl_test, device, roi, max_cases=args.max_eval_cases)
    print(f"  Pre-finetune avg Dice: {pre_dice:.4f}")
    for c, d in pre_per_class.items():
        print(f"    Class {c}: {d:.4f}")

    # Training loop
    best_dice = pre_dice
    epoch_times = []
    eval_history: list[dict[str, object]] = []
    best_metrics_payload: dict[str, object] | None = None

    print(f"\n{'=' * 80}")
    print(f"Starting finetuning")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Batch size: {args.batch_size}")
    print(f"  Eval interval: every {args.eval_interval} epochs")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 80}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_supervised = 0.0
        epoch_kd = 0.0
        epoch_feature_kd = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in dl_train:
            img = batch["image"].to(device)
            lbl = batch["label"].long().to(device)
            ignore_mask = lbl != 6
            kd_active = teacher is not None and args.distill_weight > 0 and epoch >= args.distill_start_epoch
            feature_kd_active = teacher is not None and args.feature_distill_weight > 0 and epoch >= args.distill_start_epoch

            optimizer.zero_grad(set_to_none=True)
            if student_feature_hooks is not None:
                student_feature_hooks.clear()
            if teacher_feature_hooks is not None:
                teacher_feature_hooks.clear()
            with make_autocast_context(device, amp_enabled, args.amp_dtype):
                logits = model(img)

                # Same supervised loss as train.py: 0.5 * CE + 0.5 * Dice
                ce_target = lbl.squeeze(1).clone()
                ce_target[ce_target == 6] = 255
                ce = F.cross_entropy(logits, ce_target, ignore_index=255)
                dice = dice_loss_masked(logits, lbl, ignore_mask)
                supervised_loss = 0.5 * ce + 0.5 * dice

                kd_loss = logits.new_tensor(0.0)
                feature_kd_loss = logits.new_tensor(0.0)
                if kd_active:
                    with torch.no_grad():
                        teacher_logits = teacher(img)
                    kd_loss = distillation_loss_masked(
                        student_logits=logits,
                        teacher_logits=teacher_logits,
                        ignore_mask=ignore_mask,
                        target=lbl,
                        class_weights=kd_class_weights_tensor,
                        temperature=args.distill_temperature,
                    )
                elif feature_kd_active:
                    with torch.no_grad():
                        teacher(img)

                if feature_kd_active and student_feature_hooks is not None and teacher_feature_hooks is not None:
                    feature_kd_loss = feature_distillation_loss(
                        student_features=student_feature_hooks.features,
                        teacher_features=teacher_feature_hooks.features,
                        ignore_mask=ignore_mask,
                        layer_names=feature_distill_layers,
                    )

                loss = supervised_loss + args.distill_weight * kd_loss + args.feature_distill_weight * feature_kd_loss

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            epoch_supervised += supervised_loss.item()
            epoch_kd += kd_loss.item()
            epoch_feature_kd += feature_kd_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_supervised = epoch_supervised / max(n_batches, 1)
        avg_kd = epoch_kd / max(n_batches, 1)
        avg_feature_kd = epoch_feature_kd / max(n_batches, 1)
        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        print(
            f"Epoch {epoch}/{args.epochs} - loss {avg_loss:.4f} "
            f"(sup {avg_supervised:.4f}, kd {avg_kd:.4f}, feat {avg_feature_kd:.4f}) - "
            f"{elapsed:.1f}s - lr {optimizer.param_groups[0]['lr']:.2e} "
            f"- kd_active={epoch >= args.distill_start_epoch and (args.distill_weight > 0 or args.feature_distill_weight > 0)}"
        )

        # Evaluate periodically
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            avg_dice, per_class = evaluate(model, dl_test, device, roi, max_cases=args.max_eval_cases)
            pc_str = ", ".join(f"c{c}={d:.4f}" for c, d in per_class.items())
            print(f"  Test avg Dice: {avg_dice:.4f} [{pc_str}]")
            eval_payload = {
                "epoch": epoch,
                "avg_dice": avg_dice,
                "per_class_dice": {str(c): float(d) for c, d in per_class.items()},
                "train_loss": avg_loss,
                "supervised_loss": avg_supervised,
                "kd_loss": avg_kd,
                "feature_kd_loss": avg_feature_kd,
                "kd_active": bool(epoch >= args.distill_start_epoch and (args.distill_weight > 0 or args.feature_distill_weight > 0)),
                "distill_start_epoch": args.distill_start_epoch,
            }
            eval_history.append(eval_payload)

            if args.save_eval_checkpoints:
                eval_ckpt = {
                    "state_dict": model.state_dict(),
                    "features": list(features),
                    "best_dice": best_dice,
                    "epoch": epoch,
                    "eval_metrics": eval_payload,
                }
                if "pruning_info" in ckpt:
                    eval_ckpt["pruning_info"] = ckpt["pruning_info"]
                torch.save(eval_ckpt, out_dir / f"epoch_{epoch:03d}.ckpt")

            scheduler.step(avg_dice)

            if avg_dice > best_dice:
                best_dice = avg_dice
                best_metrics_payload = eval_payload
                save_dict = {
                    "state_dict": model.state_dict(),
                    "features": list(features),
                    "best_dice": best_dice,
                    "epoch": epoch,
                }
                if "pruning_info" in ckpt:
                    save_dict["pruning_info"] = ckpt["pruning_info"]
                torch.save(save_dict, out_dir / "best.ckpt")
                print(f"  -> New best! Saved to {out_dir / 'best.ckpt'}")

    # Save final model
    final_dict = {
        "state_dict": model.state_dict(),
        "features": list(features),
        "best_dice": best_dice,
        "epoch": args.epochs,
    }
    if "pruning_info" in ckpt:
        final_dict["pruning_info"] = ckpt["pruning_info"]
    torch.save(final_dict, out_dir / "last.ckpt")
    (out_dir / "eval_history.json").write_text(json.dumps(eval_history, indent=2))
    if best_metrics_payload is not None:
        (out_dir / "best_metrics.json").write_text(json.dumps(best_metrics_payload, indent=2))

    # Summary
    total_train_time = sum(epoch_times)
    print(f"\n{'=' * 80}")
    print(f"Finetuning complete!")
    print(f"  Pre-finetune Dice:  {pre_dice:.4f}")
    print(f"  Best finetune Dice: {best_dice:.4f}")
    print(f"  Recovery: {best_dice - pre_dice:+.4f}")
    print(f"  Total training time: {total_train_time:.1f}s ({total_train_time/60:.1f}min)")
    print(f"  Best model: {out_dir / 'best.ckpt'}")
    print(f"{'=' * 80}")

    # Save report
    report = {
        "pre_finetune_dice": pre_dice,
        "best_finetune_dice": best_dice,
        "recovery": best_dice - pre_dice,
        "features": list(features),
        "params": n_params,
        "epochs": args.epochs,
        "lr": args.lr,
        "subset_ratio": args.subset_ratio,
        "total_train_time_sec": total_train_time,
        "amp": amp_enabled,
        "amp_dtype": args.amp_dtype if amp_enabled else None,
        "teacher_model_path": args.teacher_model_path,
        "teacher_model_format": args.teacher_model_format if args.teacher_model_path else None,
        "teacher_features": list(teacher_features) if teacher_features is not None else None,
        "distill_weight": args.distill_weight,
        "distill_temperature": args.distill_temperature,
        "distill_start_epoch": args.distill_start_epoch,
        "kd_class_weights": kd_class_weights,
        "feature_distill_weight": args.feature_distill_weight,
        "feature_distill_layers": feature_distill_layers,
        "max_eval_cases": args.max_eval_cases,
    }
    if "pruning_info" in ckpt:
        report["pruning_info"] = {k: v for k, v in ckpt["pruning_info"].items()
                                  if k != "selected_indices"}
    (out_dir / "finetune_report.json").write_text(json.dumps(report, indent=2))
    print(f"Report saved to: {out_dir / 'finetune_report.json'}")

    if student_feature_hooks is not None:
        student_feature_hooks.close()
    if teacher_feature_hooks is not None:
        teacher_feature_hooks.close()


if __name__ == "__main__":
    main()
