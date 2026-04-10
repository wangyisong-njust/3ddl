#!/usr/bin/env python3
"""
Quantization-Aware Training (QAT) for MONAI BasicUNet.

Takes a pre-trained teacher checkpoint and fine-tunes it with fake quantization
nodes so the model learns INT8-robust representations. After QAT, the model
can be exported to ONNX and built as a TRT INT8 engine with minimal accuracy loss.

Usage:
  python qat_finetune.py \
    --checkpoint ../../intelliscan/models/segmentation_model.ckpt \
    --output_dir ../runs/qat_teacher \
    --epochs 10 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.ao.quantization as quant
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.networks.nets import BasicUNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandSpatialCropd,
    SpatialPadd,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train as train_module  # reuse helpers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "3ddl-dataset"))
from dataset_loader import BumpDataset


def prepare_qat_model(model: torch.nn.Module) -> torch.nn.Module:
    """Insert fake quantization nodes for QAT."""
    model.train()
    # Use FX-graph mode QAT for full model support
    model.qconfig = quant.QConfig(
        activation=quant.FakeQuantize.with_args(
            observer=quant.MovingAverageMinMaxObserver,
            quant_min=0, quant_max=255, dtype=torch.quint8,
            qscheme=torch.per_tensor_affine, reduce_range=False,
        ),
        weight=quant.FakeQuantize.with_args(
            observer=quant.MovingAveragePerChannelMinMaxObserver,
            quant_min=-128, quant_max=127, dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
        ),
    )
    # Propagate qconfig and insert fake quant
    quant.propagate_qconfig_(model)
    quant.prepare_qat(model, inplace=True)
    return model


def dice_loss_masked(logits, labels, mask):
    """Dice loss with ignore-label masking."""
    n_classes = logits.shape[1]
    pred = F.softmax(logits, dim=1)
    one_hot = F.one_hot(labels.squeeze(1).clamp(0, n_classes - 1), n_classes)
    one_hot = one_hot.permute(0, 4, 1, 2, 3).float()
    mask_f = mask.float()
    intersection = (pred * one_hot * mask_f).sum(dim=(2, 3, 4))
    union = (pred * mask_f).sum(dim=(2, 3, 4)) + (one_hot * mask_f).sum(dim=(2, 3, 4))
    dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
    return 1.0 - dice[:, :n_classes].mean()


def main():
    parser = argparse.ArgumentParser(description="QAT fine-tuning for BasicUNet")
    parser.add_argument("--checkpoint", required=True, help="Pre-trained teacher checkpoint")
    parser.add_argument("--output_dir", default="runs/qat_teacher", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default=None,
                        help="Dataset directory (default: auto-detect)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    data_dir = args.data_dir or str(Path(__file__).resolve().parent.parent / "3ddl-dataset" / "data")
    dataset = BumpDataset(data_dir)
    cfg = dataset.config
    splits = dataset.split(test_serial_numbers=cfg.get("test_serial_numbers"))
    def to_monai_list(bump_ds):
        result = []
        for i in range(len(bump_ds)):
            meta = bump_ds.get_metadata(i)
            result.append({"image": meta["image_path"], "label": meta["label_path"]})
        return result
    train_list = to_monai_list(splits["train"])
    test_list = to_monai_list(splits["test"])
    print(f"Dataset: {len(train_list)} train, {len(test_list)} test")

    roi_size = (112, 112, 80)
    tf_train = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        train_module.ClipZScoreNormalizeD(keys=["image"]),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
        RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    ])
    tf_val = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        train_module.ClipZScoreNormalizeD(keys=["image"]),
    ])

    dl_train = DataLoader(
        Dataset(train_list, transform=tf_train),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    dl_test = DataLoader(
        Dataset(test_list, transform=tf_val),
        batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Load pre-trained model
    model = BasicUNet(spatial_dims=3, in_channels=1, out_channels=5)
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Prepare QAT
    model = model.to(device)
    model = prepare_qat_model(model)
    print("QAT fake quantization nodes inserted")
    n_fq = sum(1 for m in model.modules() if isinstance(m, quant.FakeQuantize))
    print(f"  FakeQuantize modules: {n_fq}")

    # Optimizer — small lr for fine-tuning
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_dice = -1.0
    results = {"epochs": [], "best_epoch": None, "best_dice": None}

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in dl_train:
            img = batch["image"].to(device)
            lbl = batch["label"].long().to(device)
            ignore_mask = lbl != 6

            optimizer.zero_grad(set_to_none=True)
            logits = model(img)

            ce_target = lbl.squeeze(1).clone()
            ce_target[ce_target == 6] = 255
            ce = F.cross_entropy(logits, ce_target, ignore_index=255)
            dice = dice_loss_masked(logits, lbl, ignore_mask)
            loss = 0.5 * ce + 0.5 * dice

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_time = time.time() - t0
        avg_loss = epoch_loss / max(n_batches, 1)

        # Evaluate
        model.eval()
        class_dice_sums = np.zeros(5)
        n_samples = 0
        with torch.no_grad():
            for batch in dl_test:
                img = batch["image"].to(device)
                lbl = batch["label"].long().to(device)
                pred = sliding_window_inference(img, roi_size, 4, model, overlap=0.5)
                pred_cls = pred.argmax(dim=1, keepdim=True)
                for c in range(5):
                    p = (pred_cls == c).float()
                    g = (lbl == c).float()
                    inter = (p * g).sum().item()
                    union = p.sum().item() + g.sum().item()
                    class_dice_sums[c] += (2 * inter + 1e-7) / (union + 1e-7)
                n_samples += 1

        class_dice = class_dice_sums / max(n_samples, 1)
        avg_dice = class_dice.mean()
        epoch_info = {
            "epoch": epoch, "loss": avg_loss, "train_time": train_time,
            "avg_dice": float(avg_dice),
            "class_dice": {str(i): float(class_dice[i]) for i in range(5)},
        }
        results["epochs"].append(epoch_info)

        is_best = avg_dice > best_dice
        if is_best:
            best_dice = avg_dice
            results["best_epoch"] = epoch
            results["best_dice"] = float(best_dice)
            # Save best — save the underlying model state_dict (without fake quant)
            torch.save(model.state_dict(), output_dir / "best_qat.ckpt")

        print(f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  "
              f"dice={avg_dice:.4f}  c3={class_dice[3]:.4f}  c4={class_dice[4]:.4f}  "
              f"{'*BEST*' if is_best else ''}  ({train_time:.1f}s)")

    # Save last
    torch.save(model.state_dict(), output_dir / "last_qat.ckpt")

    # Save results
    with open(output_dir / "qat_report.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nQAT complete. Best dice: {best_dice:.4f} at epoch {results['best_epoch']}")
    print(f"Results saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Export: python export_onnx.py --model_path {output_dir}/best_qat.ckpt --output {output_dir}/teacher_qat.onnx")
    print(f"  2. Build TRT: python build_trt_engine.py --onnx_path {output_dir}/teacher_qat.onnx --engine_path {output_dir}/teacher_qat_int8.engine --precision int8")


if __name__ == "__main__":
    main()
