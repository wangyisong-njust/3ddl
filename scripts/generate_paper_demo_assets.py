#!/usr/bin/env python3
"""Generate report/demo figures from validated 3DDL artifacts."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
FIG_DIR = DOCS_DIR / "figures"

SN002_RAW = Path(
    "/home/kaixin/datasets/3DDL-WP5-Data/RawScans/NewData/SN002_3D_Feb24/4_die_stack_0.75um_3Drecon_txm.nii"
)
SN002_OUTPUT = ROOT / "intelliscan/output_prefetch2/SN002_prefetch4"

RUNTIME_SUMMARY = ROOT / "intelliscan/output_runtime_rank/analysis/runtime_candidate_ranking_summary.json"
PRUNING_PARETO = ROOT / "wp5-seg/reports/paper_track/pruning_ratio_pareto_summary_20260408.json"
TEACHER_SUMMARY = ROOT / "wp5-seg/runs/paper_recovery_eval_teacher_full/metrics/summary.json"
R05_SUMMARY = ROOT / "wp5-seg/runs/paper_fulltrain_kd_c34_2_2_e10_b2_eval/metrics/summary.json"
CUMULATIVE_ENGINEERING = ROOT / "intelliscan/output_prefetch2/analysis/cumulative_engineering_gain_prefetch4_summary.json"
STUDENT_DEPLOY = ROOT / "intelliscan/output_student_deploy/analysis/student_deployment_summary_20260408.json"
SN009_ABLATION = ROOT / "intelliscan/output_ablation/analysis/SN009_reference_vs_variants.json"


def ensure_output_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.size == 0:
        return image
    lo, hi = np.percentile(image, [1, 99])
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return image


def load_demo_case():
    raw_img = nib.load(str(SN002_RAW))
    seg_img = nib.load(str(SN002_OUTPUT / "segmentation.nii.gz"))

    with open(SN002_OUTPUT / "metrology/metrology.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))

    demo_row = next(
        (
            row
            for row in rows
            if row["pad_misalignment_defect"] == "True"
            or row["solder_extrusion_defect"] == "True"
            or row["void_ratio_defect"] == "True"
        ),
        rows[0],
    )

    bb = [int(round(float(value))) for value in ast.literal_eval(demo_row["bb"])]
    x1, x2, y1, y2, z1, z2 = bb
    zmid = (z1 + z2) // 2

    raw_slice = np.asarray(raw_img.dataobj[:, :, zmid])
    seg_slice = np.asarray(seg_img.dataobj[:, :, zmid])
    raw_crop = np.asarray(raw_img.dataobj[x1:x2, y1:y2, zmid])
    seg_crop = np.asarray(seg_img.dataobj[x1:x2, y1:y2, zmid])

    with open(SN002_OUTPUT / "metrics.json") as handle:
        metrics = json.load(handle)

    phase_pairs: list[tuple[str, float]] = [
        (phase["name"], float(phase["elapsed_seconds"])) for phase in metrics["phases"]
    ]

    return {
        "raw_slice": raw_slice,
        "seg_slice": seg_slice,
        "raw_crop": raw_crop,
        "seg_crop": seg_crop,
        "bbox": bb,
        "zmid": zmid,
        "demo_row": demo_row,
        "phase_pairs": phase_pairs,
        "total_runtime": float(metrics["total_elapsed"]),
        "bbox_count": int(np.load(SN002_OUTPUT / "bb3d.npy").shape[0]),
        "metrology_rows": len(rows),
    }


def plot_pipeline_overview() -> Path:
    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.03, 0.30, 0.13, 0.38, "Input\nRaw 3D NIfTI\n`*.nii`"),
        (0.20, 0.30, 0.14, 0.38, "2D Detection\nTwo orthogonal views\nYOLO"),
        (0.38, 0.30, 0.14, 0.38, "3D Merge\nBounding boxes\n`bb3d.npy`"),
        (0.56, 0.30, 0.14, 0.38, "3D Segmentation\nFull volume mask\n`segmentation.nii.gz`"),
        (0.74, 0.30, 0.12, 0.38, "Metrology\nCSV + defect flags\n`metrology.csv`"),
        (0.89, 0.30, 0.08, 0.38, "Deliverables\n`final_report.pdf`\noptional `.gltf`\n`metrics.json`"),
    ]

    for x, y, w, h, label in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=2.0,
            edgecolor="#1f3b4d",
            facecolor="#e8f1f8",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)

    for left, right in zip(boxes[:-1], boxes[1:]):
        x1 = left[0] + left[2]
        x2 = right[0]
        arrow = FancyArrowPatch(
            (x1 + 0.01, 0.49),
            (x2 - 0.01, 0.49),
            arrowstyle="->",
            mutation_scale=16,
            linewidth=2.2,
            color="#355c7d",
        )
        ax.add_patch(arrow)

    ax.text(
        0.5,
        0.90,
        "End-to-end artifact flow of the current intelliscan default pipeline",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.11,
        "Safe retained path in this workspace: exact in-memory detection + 4 CPU image workers + intra-view batch prefetch",
        ha="center",
        va="center",
        fontsize=11,
        color="#37474f",
    )

    output = FIG_DIR / "pipeline_overview_demo.png"
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_metrology_cliff() -> Path:
    x = np.linspace(0.0, 0.20, 200)
    threshold = 0.10
    y = (x > threshold).astype(float)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(x, y, color="#c0392b", linewidth=3)
    ax.axvline(threshold, color="#2c3e50", linestyle="--", linewidth=1.8, label=r"$\tau_{pad}=0.10$")
    ax.scatter([0.095, 0.105], [0.0, 1.0], c=["#4c78a8", "#f58518"], s=90, zorder=5)
    ax.annotate("small boundary change\nstill below threshold", (0.095, 0.0), xytext=(0.03, 0.18), textcoords="data", arrowprops={"arrowstyle": "->"})
    ax.annotate("small boundary change\nflips defect state", (0.105, 1.0), xytext=(0.125, 0.72), textcoords="data", arrowprops={"arrowstyle": "->"})
    ax.set_ylim(-0.05, 1.10)
    ax.set_yticks([0, 1], ["non-defect", "defect"])
    ax.set_xlabel(r"Normalized pad shift $|\Delta_{pad}| / w_{pillar}$")
    ax.set_ylabel("Metrology decision")
    ax.set_title("Conceptual cliff effect of thresholded metrology")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")

    output = FIG_DIR / "metrology_cliff_effect.png"
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_two_level_gate() -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.04, 0.34, 0.18, 0.30, "Candidate method\n(model / runtime /\npipeline change)"),
        (0.30, 0.34, 0.18, 0.30, "Gate 1\nFull-case `wp5-seg`\nDice + class 3/4"),
        (0.56, 0.34, 0.18, 0.30, "Gate 2\nwhole `intelliscan`\nmetrology stability"),
        (0.82, 0.34, 0.14, 0.30, "Retain\nor reject"),
    ]
    colors = ["#e8f1f8", "#eef6ea", "#fff4d6", "#f6e6e8"]
    for (x, y, w, h, label), color in zip(boxes, colors):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.02",
                linewidth=2.0,
                edgecolor="#2f4858",
                facecolor=color,
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=12)

    for left, right in zip(boxes[:-1], boxes[1:]):
        x1 = left[0] + left[2]
        x2 = right[0]
        ax.add_patch(
            FancyArrowPatch(
                (x1 + 0.01, 0.49),
                (x2 - 0.01, 0.49),
                arrowstyle="->",
                mutation_scale=16,
                linewidth=2.2,
                color="#355c7d",
            )
        )

    ax.text(0.5, 0.86, "Two-Level Deployment-Safety Gate used throughout this report", ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0.26, 0.15, "Fail at Gate 1:\nreject before deployment", ha="center", va="center", fontsize=10)
    ax.text(0.69, 0.15, "Pass Gate 1 but fail Gate 2:\nvalidated negative result", ha="center", va="center", fontsize=10)

    output = FIG_DIR / "two_level_evaluation_gate.png"
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_combined_seg_met_flow() -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.0), gridspec_kw={"hspace": 0.30})

    rows = [
        (
            axes[0],
            "Old Separate Path",
            [
                "bbox crop",
                "3D segmentation",
                "save `pred_*.nii.gz`",
                "re-read prediction files",
                "metrology",
            ],
            "#f9e5e5",
            "#c44e52",
        ),
        (
            axes[1],
            "Retained Combined Path",
            [
                "bbox crop",
                "3D segmentation",
                "keep prediction in memory",
                "direct metrology",
                "shared final artifacts",
            ],
            "#e8f1f8",
            "#4c78a8",
        ),
    ]

    for ax, title, labels, facecolor, edgecolor in rows:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.02, 0.88, title, fontsize=14, fontweight="bold", ha="left", va="center")

        x_positions = [0.05, 0.25, 0.45, 0.67, 0.84]
        widths = [0.13, 0.13, 0.16, 0.16, 0.12]
        for x, w, label in zip(x_positions, widths, labels):
            ax.add_patch(
                FancyBboxPatch(
                    (x, 0.28),
                    w,
                    0.38,
                    boxstyle="round,pad=0.02,rounding_size=0.02",
                    linewidth=2.0,
                    edgecolor=edgecolor,
                    facecolor=facecolor,
                )
            )
            ax.text(x + w / 2, 0.47, label, ha="center", va="center", fontsize=11)

        for (x, w), next_x in zip(zip(x_positions[:-1], widths[:-1]), x_positions[1:]):
            ax.add_patch(
                FancyArrowPatch(
                    (x + w + 0.01, 0.47),
                    (next_x - 0.01, 0.47),
                    arrowstyle="->",
                    mutation_scale=15,
                    linewidth=2.1,
                    color="#355c7d",
                )
            )

    axes[0].text(
        0.5,
        0.08,
        "Extra artifact write/read cycle increases I/O and keeps metrology dependent on per-bbox prediction files.",
        ha="center",
        va="center",
        fontsize=10,
        color="#6b2d34",
    )
    axes[1].text(
        0.5,
        0.08,
        "Segmentation geometry is unchanged; only redundant artifact flow is removed before metrology.",
        ha="center",
        va="center",
        fontsize=10,
        color="#264653",
    )

    fig.suptitle("Engineering redesign of segmentation-to-metrology data flow", fontsize=16, fontweight="bold")
    output = FIG_DIR / "combined_seg_metrology_flow.png"
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_sn002_demo_walkthrough() -> Path:
    case = load_demo_case()

    raw_slice = normalize_image(case["raw_slice"]).T
    raw_crop = normalize_image(case["raw_crop"]).T
    seg_crop = case["seg_crop"].T

    x1, x2, y1, y2, *_ = case["bbox"]
    demo_row = case["demo_row"]

    seg_cmap = ListedColormap(
        [
            (0.0, 0.0, 0.0, 0.0),
            (0.89, 0.29, 0.20, 0.55),
            (0.20, 0.53, 0.96, 0.55),
            (0.15, 0.69, 0.38, 0.65),
            (0.96, 0.74, 0.10, 0.65),
        ]
    )

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.92], width_ratios=[1.1, 1.0, 1.1], hspace=0.28, wspace=0.20)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(raw_slice, cmap="gray", origin="lower")
    ax0.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#ff4d4f", linewidth=2.0))
    ax0.set_title("Whole raw-scan slice with one detected bump bbox")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(raw_crop, cmap="gray", origin="lower")
    ax1.set_title("Representative bump crop from the raw volume")
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(raw_crop, cmap="gray", origin="lower")
    ax2.imshow(seg_crop, cmap=seg_cmap, origin="lower", vmin=0, vmax=4)
    ax2.set_title("Same crop with predicted segmentation overlay")
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3 = fig.add_subplot(gs[1, 0])
    phase_names = [name.replace("Generation", "Gen.") for name, _ in case["phase_pairs"]]
    phase_times = [value for _, value in case["phase_pairs"]]
    y_pos = np.arange(len(phase_names))
    ax3.barh(y_pos, phase_times, color="#4c78a8")
    ax3.set_yticks(y_pos, phase_names)
    ax3.invert_yaxis()
    ax3.set_xlabel("Seconds")
    ax3.set_title(f"Stage timing for SN002 (total {case['total_runtime']:.2f}s)")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    metrology_lines = [
        "Representative bump metrology",
        f"filename: {demo_row['filename']}",
        f"bbox: {demo_row['bb']}",
        f"BLT: {demo_row['BLT']}",
        f"Pad misalignment: {demo_row['Pad_misalignment']}",
        f"Pillar width: {demo_row['pillar_width']}",
        f"void_ratio_defect: {demo_row['void_ratio_defect']}",
        f"solder_extrusion_defect: {demo_row['solder_extrusion_defect']}",
        f"pad_misalignment_defect: {demo_row['pad_misalignment_defect']}",
    ]
    ax4.text(
        0.02,
        0.98,
        "\n".join(metrology_lines),
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "#f7f7f7", "edgecolor": "#999999"},
    )

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    artifact_lines = [
        "Input -> output artifact trail",
        f"input: {SN002_RAW.name}",
        f"output dir: {SN002_OUTPUT.name}/",
        f"3D boxes: bb3d.npy ({case['bbox_count']} rows)",
        "full segmentation: segmentation.nii.gz",
        f"metrology rows: {case['metrology_rows']}",
        "tabular output: metrology/metrology.csv",
        "human-readable report: metrology/final_report.pdf",
        "profiling log: metrics.json",
    ]
    ax5.text(
        0.02,
        0.98,
        "\n".join(artifact_lines),
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "#eef6ea", "edgecolor": "#6b8e23"},
    )

    fig.suptitle("Representative end-to-end demo on SN002 using the current safe default pipeline", fontsize=16, fontweight="bold")

    output = FIG_DIR / "sn002_pipeline_demo.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_sn009_stage_breakdown() -> Path:
    variants = [
        ("PyTorch baseline", ROOT / "intelliscan/output_formal/SN009_current_gpu/metrics.json"),
        ("Earlier --inmemory", ROOT / "intelliscan/output_formal/SN009_current_gpu_inmemory/metrics.json"),
        ("TRT FP16", ROOT / "intelliscan/output_formal/SN009_current_gpu_trt/metrics.json"),
    ]
    labels = []
    totals = []
    frontend = []
    segmentation = []
    metrology = []
    report = []
    for label, path in variants:
        labels.append(label)
        with open(path) as handle:
            payload = json.load(handle)
        phase_map = {phase["name"]: float(phase["elapsed_seconds"]) for phase in payload["phases"]}
        totals.append(float(payload["total_elapsed"]))
        frontend.append(
            phase_map.get("NII to JPG Conversion", 0.0)
            + phase_map.get("2D Detection Inference", 0.0)
            + phase_map.get("3D Bounding Box Generation", 0.0)
        )
        segmentation.append(phase_map.get("3D Segmentation", 0.0))
        metrology.append(phase_map.get("Metrology", 0.0))
        report.append(phase_map.get("Report Generation", 0.0))

    baseline_total = totals[0]
    baseline_seg = segmentation[0]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), gridspec_kw={"width_ratios": [1.0, 1.25]})

    total_colors = ["#4c78a8", "#f58518", "#54a24b"]
    x = np.arange(len(labels))
    bars = axes[0].bar(x, totals, color=total_colors, width=0.62)
    axes[0].set_xticks(x, labels, rotation=15)
    axes[0].set_ylabel("Seconds")
    axes[0].set_title("Whole-pipeline total runtime")
    axes[0].grid(True, axis="y", alpha=0.25)
    for idx, (bar, total) in enumerate(zip(bars, totals)):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            total + 1.2,
            f"{total:.2f}s",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold" if idx == 0 else None,
        )
        if idx > 0:
            delta = (baseline_total - total) / baseline_total * 100.0
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                total * 0.55,
                f"{delta:.1f}%\nfaster",
                ha="center",
                va="center",
                fontsize=10,
                color="#ffffff",
                fontweight="bold",
            )

    stage_names = ["Front-end", "Segmentation", "Metrology"]
    baseline_stage = [frontend[0], segmentation[0], metrology[0]]
    inmemory_stage = [frontend[1], segmentation[1], metrology[1]]
    trt_stage = [frontend[2], segmentation[2], metrology[2]]
    stage_x = np.arange(len(stage_names))
    width = 0.24
    axes[1].bar(stage_x - width, baseline_stage, width, color="#4c78a8", label="PyTorch baseline")
    axes[1].bar(stage_x, inmemory_stage, width, color="#f58518", label="Earlier --inmemory")
    axes[1].bar(stage_x + width, trt_stage, width, color="#54a24b", label="TRT FP16")
    axes[1].set_xticks(stage_x, stage_names)
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Key stage comparison")
    axes[1].set_ylim(0, 54)
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=9)

    frontend_delta = (baseline_stage[0] - inmemory_stage[0]) / baseline_stage[0] * 100.0
    seg_inmemory_delta = (inmemory_stage[1] - baseline_stage[1]) / baseline_stage[1] * 100.0
    seg_trt_delta = (baseline_stage[1] - trt_stage[1]) / baseline_stage[1] * 100.0
    axes[1].text(
        stage_x[0],
        inmemory_stage[0] + 1.2,
        f"-{frontend_delta:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#b36b00",
        fontweight="bold",
    )
    axes[1].text(
        stage_x[1],
        inmemory_stage[1] + 1.2,
        f"+{seg_inmemory_delta:.1f}%\nslower",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#b36b00",
        fontweight="bold",
    )
    axes[1].text(
        stage_x[1] + width,
        trt_stage[1] + 1.2,
        f"-{seg_trt_delta:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#2f6b2f",
        fontweight="bold",
    )

    output = FIG_DIR / "sn009_formal_stage_breakdown.png"
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_engineering_reduction() -> Path:
    with open(CUMULATIVE_ENGINEERING) as handle:
        payload = json.load(handle)
    samples = payload["samples"]
    names = [entry["sample"] for entry in samples]
    baseline = [entry["file_default_total"] for entry in samples]
    final = [entry["prefetch4_total"] for entry in samples]
    reductions = [entry["total_reduction_ratio"] * 100.0 for entry in samples]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.4, 1.0]})

    x = np.arange(len(names))
    width = 0.38
    axes[0].bar(x - width / 2, baseline, width, label="older file-based baseline", color="#c44e52")
    axes[0].bar(x + width / 2, final, width, label="current safe default path", color="#4c78a8")
    axes[0].set_xticks(x, names, rotation=30)
    axes[0].set_ylabel("Total runtime (s)")
    axes[0].set_title("Eight-sample whole-pipeline runtime before and after retained engineering optimizations")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(names, reductions, color="#54a24b")
    axes[1].axhline(payload["mean_total_reduction_ratio"] * 100.0, linestyle="--", color="#2f4858", linewidth=1.6, label="mean reduction")
    axes[1].set_ylabel("Total runtime reduction (%)")
    axes[1].set_title("Per-sample total runtime reduction")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(fontsize=9)

    output = FIG_DIR / "engineering_cumulative_runtime_reduction.png"
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_pruning_pareto() -> Path:
    with open(PRUNING_PARETO) as handle:
        payload = json.load(handle)

    entries = payload["entries"]
    teacher_dice = float(payload["teacher_full_avg_dice"])
    teacher_latency = 17.401992017403245

    fig, ax = plt.subplots(figsize=(8.5, 6))
    x = [entry["fp32_mean_ms"] for entry in entries]
    y = [entry["avg_dice"] for entry in entries]
    labels = [f"r={entry['pruning_ratio']}" for entry in entries]
    sizes = [max(entry["reduction_pct"], 1.0) * 10 for entry in entries]

    ax.scatter([teacher_latency], [teacher_dice], marker="*", s=260, color="#d62728", label="teacher")
    scatter = ax.scatter(x, y, s=sizes, c=["#4c78a8", "#f58518", "#54a24b"], alpha=0.9, edgecolors="black")

    for x_i, y_i, label in zip(x, y, labels):
        ax.annotate(label, (x_i, y_i), xytext=(8, 6), textcoords="offset points", fontsize=10)
    ax.annotate("teacher", (teacher_latency, teacher_dice), xytext=(8, 6), textcoords="offset points", fontsize=10)

    ax.set_xlabel("Patch inference latency (ms)")
    ax.set_ylabel("Full-case average Dice")
    ax.set_title("Pruning-ratio Pareto front for the retained KD recipe")
    ax.grid(True, alpha=0.3)

    output = FIG_DIR / "pruning_ratio_pareto.png"
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_deployment_gap() -> Path:
    with open(TEACHER_SUMMARY) as handle:
        teacher = json.load(handle)
    with open(STUDENT_DEPLOY) as handle:
        deploy = json.load(handle)

    points = [
        {
            "label": "teacher",
            "avg_dice": float(teacher["average"]["dice"]),
            "flip_rate": 0.0,
            "color": "#d62728",
            "size": 260,
        }
    ]
    model_metrics = {
        "r=0.375": (0.8951335147108839, 0.7869013851915283, 0.8716918322546728),
        "r=0.5": (0.889527044574805, 0.7545048112381549, 0.8749368314085335),
    }
    for key, label, color in [("r0p375", "r=0.375", "#4c78a8"), ("r0p5", "r=0.5", "#f58518")]:
        total_flips = 0
        total_decisions = 0
        for sample_data in deploy.values():
            comp = sample_data["comparisons_vs_teacher"][key]["metrology"]
            total_flips += int(comp["pad_flip_count"]) + int(comp["solder_flip_count"])
            total_decisions += 2 * int(comp["count"])
        avg_dice, _, _ = model_metrics[label]
        points.append(
            {
                "label": label,
                "avg_dice": avg_dice,
                "flip_rate": total_flips / max(total_decisions, 1),
                "color": color,
                "size": 220,
            }
        )

    fig, ax = plt.subplots(figsize=(8.2, 6))
    for point in points:
        ax.scatter(point["avg_dice"], point["flip_rate"], s=point["size"], color=point["color"], edgecolors="black")
        ax.annotate(point["label"], (point["avg_dice"], point["flip_rate"]), xytext=(8, 6), textcoords="offset points", fontsize=10)
    ax.set_xlabel("Full-case average Dice")
    ax.set_ylabel("Whole-pipeline defect-flip rate")
    ax.set_title("Deployment gap: high full-case Dice does not guarantee stable whole-pipeline behavior")
    ax.grid(True, alpha=0.25)

    output = FIG_DIR / "deployment_gap_fullcase_vs_pipeline.png"
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_adaptive_margin_ablation() -> Path:
    with open(SN009_ABLATION) as handle:
        payload = json.load(handle)

    labels = ["A baseline", "B path-only", "C context-only", "D full adaptive"]
    seg_times = [33.36, 24.35, 20.24, 20.77]
    pad_flips = [0, payload["SN009_B_fixed_direct_full"]["metrology"]["pad_flip_count"], payload["SN009_C_adaptive_sliding"]["metrology"]["pad_flip_count"], payload["SN009_D_adaptive_auto"]["metrology"]["pad_flip_count"]]
    c3 = [1.0, payload["SN009_B_fixed_direct_full"]["class_dice"]["3"], payload["SN009_C_adaptive_sliding"]["class_dice"]["3"], payload["SN009_D_adaptive_auto"]["class_dice"]["3"]]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    axes[0].bar(labels, seg_times, color="#4c78a8")
    axes[0].set_title("Segmentation time on SN009")
    axes[0].set_ylabel("Seconds")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(labels, pad_flips, color="#e45756")
    axes[1].set_title("Pad defect flips vs baseline")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, axis="y", alpha=0.25)

    axes[2].bar(labels, c3, color="#54a24b")
    axes[2].set_title("Class-3 Dice vs baseline")
    axes[2].set_ylabel("Dice")
    axes[2].set_ylim(0, 1.05)
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].grid(True, axis="y", alpha=0.25)

    fig.suptitle("Adaptive-margin ablation on SN009: context shrinkage causes most of the drift")
    output = FIG_DIR / "adaptive_margin_ablation_sn009.png"
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_case_study_teacher_vs_student() -> Path:
    raw_path = SN002_RAW
    teacher_dir = ROOT / "intelliscan/output_runtime_rank/SN002_teacher_eager"
    student_dir = ROOT / "intelliscan/output_runtime_rank/SN002_r0p5_eager"

    with open(teacher_dir / "metrology/metrology.csv", newline="") as handle:
        teacher_rows = {row["filename"]: row for row in csv.DictReader(handle)}
    with open(student_dir / "metrology/metrology.csv", newline="") as handle:
        student_rows = {row["filename"]: row for row in csv.DictReader(handle)}

    target_name = "pred_0.nii.gz"
    teacher_row = teacher_rows[target_name]
    student_row = student_rows[target_name]
    bb = [int(round(float(value))) for value in ast.literal_eval(teacher_row["bb"])]
    x1, x2, y1, y2, z1, z2 = bb
    zmid = (z1 + z2) // 2

    raw = nib.load(str(raw_path))
    teacher_seg = nib.load(str(teacher_dir / "segmentation.nii.gz"))
    student_seg = nib.load(str(student_dir / "segmentation.nii.gz"))

    raw_crop = normalize_image(np.asarray(raw.dataobj[x1:x2, y1:y2, zmid])).T
    teacher_crop = np.asarray(teacher_seg.dataobj[x1:x2, y1:y2, zmid]).T
    student_crop = np.asarray(student_seg.dataobj[x1:x2, y1:y2, zmid]).T

    seg_cmap = ListedColormap(
        [
            (0.0, 0.0, 0.0, 0.0),
            (0.89, 0.29, 0.20, 0.55),
            (0.20, 0.53, 0.96, 0.55),
            (0.15, 0.69, 0.38, 0.65),
            (0.96, 0.74, 0.10, 0.65),
        ]
    )

    fig = plt.figure(figsize=(12.2, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.92], wspace=0.12)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    for ax, overlay, title in [
        (ax0, teacher_crop, "Teacher mask"),
        (ax1, student_crop, "Compressed student mask (r=0.5)"),
    ]:
        ax.imshow(raw_crop, cmap="gray", origin="lower")
        ax.imshow(overlay, cmap=seg_cmap, origin="lower", vmin=0, vmax=4)
        ax.set_title(title, fontsize=14, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_edgecolor("#333333")

    ax2.axis("off")
    lines = [
        ("Case", "SN002 / pred_0.nii.gz"),
        ("Teacher BLT", teacher_row["BLT"]),
        ("Student BLT", student_row["BLT"]),
        ("Teacher pad misalignment", teacher_row["Pad_misalignment"]),
        ("Student pad misalignment", student_row["Pad_misalignment"]),
        ("Teacher solder defect", teacher_row["solder_extrusion_defect"]),
        ("Student solder defect", student_row["solder_extrusion_defect"]),
        ("Teacher pad defect", teacher_row["pad_misalignment_defect"]),
        ("Student pad defect", student_row["pad_misalignment_defect"]),
    ]
    text = "\n".join(f"{label}: {value}" for label, value in lines)
    ax2.text(
        0.02,
        0.96,
        text,
        ha="left",
        va="top",
        fontsize=11.2,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f8f8f8", "edgecolor": "#888888"},
    )
    ax2.text(
        0.02,
        0.17,
        "Observed change:\n"
        "solder defect flips from True to False\n"
        "while the crop remains visually similar.",
        ha="left",
        va="top",
        fontsize=10.8,
        color="#7a1f1f",
    )
    output = FIG_DIR / "case_study_teacher_vs_student_sn002.png"
    fig.tight_layout(pad=0.6)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_runtime_ranking() -> Path:
    with open(RUNTIME_SUMMARY) as handle:
        payload = json.load(handle)

    rows = payload["aggregate_table"]
    colors = {
        "reference": "#d62728",
        "retained": "#4c78a8",
        "validated_with_caveat": "#f2a104",
        "context_only": "#9c755f",
    }
    markers = {
        "pytorch_eager": "o",
        "trt_fp16": "s",
        "trt_int8_fitonly64": "^",
        "pytorch_compile_reduce-overhead": "D",
    }

    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    for row in rows:
        size = max(float(row["mean_pipeline_energy_j"]) / 18.0, 80.0)
        label = row["label"]
        if label == "r=0.5 / compile":
            label = "r=0.5 / compile*"
        ax.scatter(
            float(row["mean_total_latency_s"]),
            float(row.get("mean_class3_dice_vs_teacher") or 1.0),
            s=size,
            c=colors.get(row["status"], "#777777"),
            marker=markers.get(row["backend"], "o"),
            edgecolors="black",
            alpha=0.90,
        )
        ax.annotate(label, (float(row["mean_total_latency_s"]), float(row.get("mean_class3_dice_vs_teacher") or 1.0)), xytext=(8, 6), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Mean whole-pipeline latency (s)")
    ax.set_ylabel("Whole-pipeline class-3 Dice vs teacher")
    ax.set_title("Whole-pipeline runtime ranking: latency, teacher-relative quality, and energy")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02,
        0.02,
        "* compile point uses the same student checkpoint as eager;\nfull-case compile quality was not rerun separately.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "#ffffff", "edgecolor": "#999999", "alpha": 0.85},
    )

    output = FIG_DIR / "runtime_candidate_ranking_latency_quality.png"
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    ensure_output_dir()
    outputs = [
        plot_metrology_cliff(),
        plot_pipeline_overview(),
        plot_two_level_gate(),
        plot_combined_seg_met_flow(),
        plot_sn002_demo_walkthrough(),
        plot_sn009_stage_breakdown(),
        plot_engineering_reduction(),
        plot_pruning_pareto(),
        plot_deployment_gap(),
        plot_adaptive_margin_ablation(),
        plot_case_study_teacher_vs_student(),
        plot_runtime_ranking(),
    ]
    print("Generated figures:")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
