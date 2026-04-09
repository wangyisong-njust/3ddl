#!/usr/bin/env python3
"""
Evaluate PyTorch or TensorRT backends on the wp5-seg test set.

This keeps runtime-backend accuracy checks aligned with the standard evaluation
protocol used by `eval.py`, while allowing TensorRT engine validation before
anything is pushed into `intelliscan`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from monai.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train
from runtime_support import TensorRTPredictor, load_basicunet_model


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PyTorch checkpoint or TensorRT engine on wp5-seg")
    parser.add_argument("--backend", type=str, required=True, choices=["pytorch", "trt"])
    parser.add_argument("--model_path", type=str, required=True, help="Checkpoint path or TensorRT engine path")
    parser.add_argument("--model_format", type=str, default="state_dict", choices=["state_dict", "pruned"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--roi_x", type=int, default=112)
    parser.add_argument("--roi_y", type=int, default=112)
    parser.add_argument("--roi_z", type=int, default=80)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_cases", type=int, default=-1)
    parser.add_argument("--save_preds", action="store_true")
    parser.add_argument("--heavy", action="store_true", default=False)
    parser.add_argument("--hd_percentile", type=float, default=95.0)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    test_list = train.build_datalists(data_dir)[1]
    _, t_val = train.get_transforms(roi=(args.roi_x, args.roi_y, args.roi_z))
    ds_test = Dataset(test_list, transform=t_val)
    dl_test = DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and args.num_workers == 0,
    )

    if args.backend == "pytorch":
        predictor, features = load_basicunet_model(args.model_path, args.model_format, device)
        print(f"Loaded PyTorch model: features={features}")
    else:
        # Sliding-window evaluation prioritizes backend correctness over the
        # fastest transfer path. Host relay is slower but avoids CUDA context
        # issues seen with GPU-direct mode under repeated MONAI window calls.
        predictor = TensorRTPredictor(args.model_path, force_host_relay=True)
        print(f"Loaded TensorRT engine: {args.model_path}")

    metrics = train.evaluate(
        predictor,
        dl_test,
        device,
        out_dir,
        save_preds=bool(args.save_preds),
        max_cases=(None if args.max_cases < 0 else args.max_cases),
        heavy=bool(args.heavy),
        hd_percentile=float(args.hd_percentile),
    )
    payload = {
        "backend": args.backend,
        "model_path": args.model_path,
        "model_format": args.model_format,
        "metrics": metrics,
    }
    summary_path = out_dir / "metrics" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved metrics to: {summary_path}")
    print(json.dumps(metrics["average"], indent=2))


if __name__ == "__main__":
    main()
