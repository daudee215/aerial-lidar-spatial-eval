# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Daud Tasleem
"""Report generation: JSON summary and matplotlib visualisations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from aerial_lidar_spatial_eval.metrics import EvalResult


def generate_report(
    result: EvalResult,
    output_dir: str | Path,
    model_name: str = "model",
) -> Path:
    """
    Write a JSON summary and PNG figures to *output_dir*.

    Files written
    -------------
    - summary.json          : structured metrics
    - confusion_heatmap.png : distance-weighted confusion matrix heatmap
    - stratified_iou.png    : mIoU vs distance band bar chart
    - mde_per_class.png     : mean distance error per class

    Parameters
    ----------
    result : EvalResult
        Output of :meth:`SpatialEvaluator.evaluate`.
    output_dir : str | Path
        Directory to write output files.
    model_name : str
        Model identifier used in figure titles.

    Returns
    -------
    Path to summary.json
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- JSON summary ---
    mde = result.dw_confusion.mean_distance_error().tolist()
    band_miou = result.stratified_iou.mean_iou_per_band()
    summary: dict[str, Any] = {
        "model": model_name,
        "n_points": result.n_points,
        "n_classes": result.n_classes,
        "overall_accuracy": round(result.overall_accuracy, 6),
        "miou": round(result.miou, 6),
        "per_class_iou": {
            name: round(float(iou), 6)
            for name, iou in zip(result.class_names, result.per_class_iou, strict=False)
            if not np.isnan(iou)
        },
        "mean_distance_error_m": {
            name: round(float(d), 4)
            for name, d in zip(result.class_names, mde, strict=False)
        },
        "spatially_stratified_miou": {
            f"band_{i}": round(float(m), 6)
            for i, m in enumerate(band_miou)
        },
        "band_point_counts": result.stratified_iou.band_point_counts,
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # --- Confusion heatmap ---
    _plot_confusion_heatmap(result, out / "confusion_heatmap.png", model_name)

    # --- Stratified IoU bar chart ---
    _plot_stratified_iou(result, out / "stratified_iou.png", model_name)

    # --- MDE per class ---
    _plot_mde(result, mde, out / "mde_per_class.png", model_name)

    return summary_path


def _plot_confusion_heatmap(result: EvalResult, path: Path, model_name: str) -> None:
    mat = result.dw_confusion.count_matrix.astype(float)
    row_sums = mat.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm_mat = np.where(row_sums > 0, mat / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(max(6, result.n_classes), max(5, result.n_classes - 1)))
    im = ax.imshow(norm_mat, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Fraction")
    ax.set_xticks(range(result.n_classes))
    ax.set_yticks(range(result.n_classes))
    ax.set_xticklabels(result.class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(result.class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} — Normalised Confusion Matrix (count-based)")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_stratified_iou(result: EvalResult, path: Path, model_name: str) -> None:
    band_miou = result.stratified_iou.mean_iou_per_band()
    counts = result.stratified_iou.band_point_counts
    bands = [0.0] + result.stratified_iou.distance_bands

    import math
    labels = []
    for i in range(len(bands)):
        lo = bands[i]
        hi = bands[i + 1] if i + 1 < len(bands) else float("inf")
        if math.isinf(hi):
            labels.append(f">{lo:.1f}m\n(interior)")
        else:
            labels.append(f"{lo:.1f}–{hi:.1f}m")

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(band_miou)))  # type: ignore[attr-defined]
    fig, ax = plt.subplots(figsize=(max(7, len(band_miou) * 1.2), 4))
    bars = ax.bar(range(len(band_miou)), band_miou, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(band_miou)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean IoU")
    ax.set_ylim(0, 1)
    ax.set_title(f"{model_name} — Spatially Stratified mIoU by Distance to Boundary")
    for bar, count in zip(bars, counts, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"n={count:,}",
            ha="center",
            fontsize=7,
        )
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_mde(result: EvalResult, mde: list[float], path: Path, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(max(6, result.n_classes * 0.9), 4))
    x = range(result.n_classes)
    colors = ["#e74c3c" if d > 1.0 else "#3498db" for d in mde]
    ax.bar(x, mde, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(result.class_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Distance Error (m)")
    ax.set_title(f"{model_name} — Mean Distance Error per Class (misclassified points)")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="1 m threshold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
