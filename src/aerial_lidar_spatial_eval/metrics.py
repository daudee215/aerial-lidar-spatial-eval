# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Daud Tasleem
"""
Core spatially-aware evaluation metrics.

All public classes operate on pre-loaded numpy arrays (xyz, true_labels,
pred_labels) so callers can use any I/O backend.  For LAS/LAZ files use
:func:`aerial_lidar_spatial_eval.io.load_las_predictions`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_inputs(
    xyz: NDArray[np.float64],
    true_labels: NDArray[np.int32],
    pred_labels: NDArray[np.int32],
) -> None:
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be shape (N, 3), got {xyz.shape}")
    n = len(xyz)
    if len(true_labels) != n or len(pred_labels) != n:
        raise ValueError("xyz, true_labels and pred_labels must have the same length")


def _chunk_iterator(n: int, chunk_size: int) -> Iterator[tuple[int, int]]:
    """Yield (start, end) index pairs for chunked processing."""
    for start in range(0, n, chunk_size):
        yield start, min(start + chunk_size, n)


# ---------------------------------------------------------------------------
# DistanceWeightedConfusionMatrix
# ---------------------------------------------------------------------------

@dataclass
class DistanceWeightedConfusionMatrix:
    """
    Confusion matrix where each off-diagonal entry is weighted by the
    3-D distance from the misclassified point to the nearest point of
    the ground-truth class.

    Parameters
    ----------
    n_classes : int
        Number of semantic classes (labels 0 … n_classes-1).
    chunk_size : int
        Number of points processed per KD-tree batch. Reduce if OOM.
    verbose : bool
        Show tqdm progress bar during KD-tree queries.
    """

    n_classes: int
    chunk_size: int = 500_000
    verbose: bool = True

    # filled by fit()
    _matrix: NDArray[np.float64] = field(init=False)
    _count_matrix: NDArray[np.int64] = field(init=False)

    def __post_init__(self) -> None:
        self._matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.float64)
        self._count_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

    # ------------------------------------------------------------------
    def fit(
        self,
        xyz: NDArray[np.float64],
        true_labels: NDArray[np.int32],
        pred_labels: NDArray[np.int32],
    ) -> "DistanceWeightedConfusionMatrix":
        """
        Build the distance-weighted confusion matrix.

        For correctly classified points the weight is 0 (they are placed on
        the diagonal with weight = 0, i.e. the diagonal accumulates *counts*
        only, not distances).  For misclassified points the weight is the
        Euclidean distance (metres) to the nearest point of the ground-truth
        class in 3-D space.

        Parameters
        ----------
        xyz : (N, 3) float64
            3-D coordinates (X, Y, Z) of every point.
        true_labels : (N,) int32
            Ground-truth semantic class labels.
        pred_labels : (N,) int32
            Predicted semantic class labels.

        Returns
        -------
        self
        """
        _validate_inputs(xyz, true_labels, pred_labels)
        n = len(xyz)

        # Reset
        self._matrix[:] = 0.0
        self._count_matrix[:] = 0

        # Accumulate diagonal (correct) in one pass — no KD-tree needed
        correct_mask = true_labels == pred_labels
        for c in range(self.n_classes):
            self._count_matrix[c, c] = int(np.sum(correct_mask & (true_labels == c)))

        # Build per-class KD-trees over GT points
        class_trees: dict[int, KDTree | None] = {}
        for c in range(self.n_classes):
            idx = np.where(true_labels == c)[0]
            class_trees[c] = KDTree(xyz[idx]) if len(idx) > 0 else None

        # Process misclassified points in chunks
        wrong_idx = np.where(~correct_mask)[0]
        if len(wrong_idx) == 0:
            return self

        chunks = list(_chunk_iterator(len(wrong_idx), self.chunk_size))
        iterator = tqdm(chunks, desc="Distance-weighted confusion", disable=not self.verbose)

        for start, end in iterator:
            chunk = wrong_idx[start:end]
            chunk_xyz = xyz[chunk]
            chunk_true = true_labels[chunk]
            chunk_pred = pred_labels[chunk]

            # For each misclassified point query distance to nearest GT point
            # of its true class
            unique_true = np.unique(chunk_true)
            dist_cache: dict[int, NDArray[np.float64]] = {}

            for c in unique_true:
                c_mask = chunk_true == c
                if class_trees[c] is None:
                    dist_cache[c] = np.zeros(np.sum(c_mask), dtype=np.float64)
                    continue
                dists, _ = class_trees[c].query(chunk_xyz[c_mask], workers=-1)
                dist_cache[c] = dists.astype(np.float64)

            # Accumulate into matrix
            for i, (t, p) in enumerate(zip(chunk_true, chunk_pred)):
                # Find position of this point within its true-class group
                c_positions = np.where(chunk_true[:i+1] == t)[0]
                pos = len(c_positions) - 1
                d = dist_cache[t][pos] if t in dist_cache else 0.0
                self._matrix[t, p] += d
                self._count_matrix[t, p] += 1

        return self

    # ------------------------------------------------------------------
    @property
    def matrix(self) -> NDArray[np.float64]:
        """Distance-weighted confusion matrix (n_classes × n_classes)."""
        return self._matrix.copy()

    @property
    def count_matrix(self) -> NDArray[np.int64]:
        """Raw count confusion matrix (n_classes × n_classes)."""
        return self._count_matrix.copy()

    def mean_distance_error(self) -> NDArray[np.float64]:
        """
        Per-class mean distance error: average displacement (m) for
        misclassified points, per true class.
        """
        off_diag_counts = self._count_matrix.copy()
        np.fill_diagonal(off_diag_counts, 0)
        row_counts = off_diag_counts.sum(axis=1).astype(np.float64)
        off_diag_dist = self._matrix.copy()
        np.fill_diagonal(off_diag_dist, 0.0)
        row_dist = off_diag_dist.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mde = np.where(row_counts > 0, row_dist / row_counts, 0.0)
        return mde


# ---------------------------------------------------------------------------
# SpatiallyStratifiedIoU
# ---------------------------------------------------------------------------

@dataclass
class SpatiallyStratifiedIoU:
    """
    Intersection-over-Union computed separately in distance bands from
    class boundaries.  Reveals how model quality degrades near boundaries.

    Parameters
    ----------
    n_classes : int
        Number of semantic classes.
    distance_bands : list[float]
        Monotonically increasing band edges in metres, e.g. [0.5, 1.0, 2.0, 5.0].
        Points within [0, bands[0]) form band 0, [bands[0], bands[1]) form band 1,
        and so on.  Points beyond the last edge form an "interior" band.
    chunk_size : int
        Batch size for KD-tree queries.
    verbose : bool
        Show progress.
    """

    n_classes: int
    distance_bands: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0])
    chunk_size: int = 500_000
    verbose: bool = True

    # filled by fit()
    _band_iou: list[NDArray[np.float64]] = field(init=False, default_factory=list)
    _band_point_counts: list[int] = field(init=False, default_factory=list)

    # ------------------------------------------------------------------
    def fit(
        self,
        xyz: NDArray[np.float64],
        true_labels: NDArray[np.int32],
        pred_labels: NDArray[np.int32],
    ) -> "SpatiallyStratifiedIoU":
        """Compute spatially stratified IoU."""
        _validate_inputs(xyz, true_labels, pred_labels)
        n = len(xyz)

        # Build KD-tree over all boundary points (where adjacent points differ in class)
        # Boundary detection: for each point find nearest neighbour of different class
        boundary_xyz = self._detect_boundaries(xyz, true_labels)

        if len(boundary_xyz) == 0:
            # Degenerate: no boundaries — all points in one class
            iou = self._iou_from_masks(true_labels, pred_labels, np.ones(n, dtype=bool))
            self._band_iou = [iou] * (len(self.distance_bands) + 1)
            self._band_point_counts = [n] + [0] * len(self.distance_bands)
            return self

        boundary_tree = KDTree(boundary_xyz)

        # Compute per-point distance to nearest boundary
        dist_to_boundary = np.empty(n, dtype=np.float64)
        for start, end in tqdm(
            list(_chunk_iterator(n, self.chunk_size)),
            desc="Boundary distance query",
            disable=not self.verbose,
        ):
            dists, _ = boundary_tree.query(xyz[start:end], workers=-1)
            dist_to_boundary[start:end] = dists

        # Assign points to bands and compute per-band IoU
        bands = [0.0] + list(self.distance_bands)
        self._band_iou = []
        self._band_point_counts = []

        for i in range(len(bands)):
            lo = bands[i]
            hi = bands[i + 1] if i + 1 < len(bands) else float("inf")
            mask = (dist_to_boundary >= lo) & (dist_to_boundary < hi)
            count = int(mask.sum())
            self._band_point_counts.append(count)
            if count == 0:
                self._band_iou.append(np.full(self.n_classes, float("nan")))
            else:
                self._band_iou.append(self._iou_from_masks(true_labels, pred_labels, mask))

        return self

    def _detect_boundaries(
        self, xyz: NDArray[np.float64], labels: NDArray[np.int32]
    ) -> NDArray[np.float64]:
        """Return XYZ of points that lie on class boundaries (k-NN check)."""
        if len(xyz) < 2:
            return np.empty((0, 3), dtype=np.float64)
        tree = KDTree(xyz)
        k = min(9, len(xyz))
        _, nn_idx = tree.query(xyz, k=k, workers=-1)
        nn_labels = labels[nn_idx]  # (N, k)
        is_boundary = np.any(nn_labels != labels[:, None], axis=1)
        return xyz[is_boundary]

    def _iou_from_masks(
        self,
        true: NDArray[np.int32],
        pred: NDArray[np.int32],
        mask: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        t = true[mask]
        p = pred[mask]
        ious = np.full(self.n_classes, float("nan"))
        for c in range(self.n_classes):
            tp = int(np.sum((t == c) & (p == c)))
            fp = int(np.sum((t != c) & (p == c)))
            fn = int(np.sum((t == c) & (p != c)))
            denom = tp + fp + fn
            ious[c] = tp / denom if denom > 0 else float("nan")
        return ious

    # ------------------------------------------------------------------
    @property
    def band_iou(self) -> list[NDArray[np.float64]]:
        """Per-band IoU arrays, length = len(distance_bands) + 1."""
        return [a.copy() for a in self._band_iou]

    @property
    def band_point_counts(self) -> list[int]:
        """Number of points in each distance band."""
        return list(self._band_point_counts)

    def mean_iou_per_band(self) -> list[float]:
        """Macro-averaged (nanmean) IoU per distance band."""
        return [float(np.nanmean(b)) for b in self._band_iou]


# ---------------------------------------------------------------------------
# HardPointDetector
# ---------------------------------------------------------------------------

@dataclass
class HardPointDetector:
    """
    Identifies 'hard' points — those misclassified by at least one model
    in a collection of predictions — and computes focused evaluation metrics
    on this challenging subset, per arxiv:2603.22420 §3.2.

    Parameters
    ----------
    n_classes : int
        Number of semantic classes.
    """

    n_classes: int
    _hard_mask: NDArray[np.bool_] | None = field(init=False, default=None)

    def fit(
        self,
        true_labels: NDArray[np.int32],
        predictions: list[NDArray[np.int32]],
    ) -> "HardPointDetector":
        """
        Compute hard-point mask from multiple model predictions.

        Parameters
        ----------
        true_labels : (N,) int32
        predictions : list of (N,) int32 arrays, one per model
        """
        n = len(true_labels)
        hard = np.zeros(n, dtype=bool)
        for pred in predictions:
            hard |= pred != true_labels
        self._hard_mask = hard
        return self

    @property
    def hard_mask(self) -> NDArray[np.bool_]:
        if self._hard_mask is None:
            raise RuntimeError("Call fit() first")
        return self._hard_mask.copy()

    @property
    def hard_point_ratio(self) -> float:
        """Fraction of total points that are 'hard'."""
        if self._hard_mask is None:
            raise RuntimeError("Call fit() first")
        return float(self._hard_mask.mean())

    def iou_on_hard_points(
        self,
        true_labels: NDArray[np.int32],
        pred_labels: NDArray[np.int32],
    ) -> NDArray[np.float64]:
        """Compute per-class IoU restricted to hard points only."""
        if self._hard_mask is None:
            raise RuntimeError("Call fit() first")
        t = true_labels[self._hard_mask]
        p = pred_labels[self._hard_mask]
        ious = np.full(self.n_classes, float("nan"))
        for c in range(self.n_classes):
            tp = int(np.sum((t == c) & (p == c)))
            fp = int(np.sum((t != c) & (p == c)))
            fn = int(np.sum((t == c) & (p != c)))
            denom = tp + fp + fn
            ious[c] = tp / denom if denom > 0 else float("nan")
        return ious


# ---------------------------------------------------------------------------
# SpatialEvaluator — high-level convenience wrapper
# ---------------------------------------------------------------------------

class SpatialEvaluator:
    """
    High-level interface that runs all spatial metrics in one call.

    Parameters
    ----------
    n_classes : int
        Number of semantic classes.
    class_names : list[str] | None
        Optional human-readable class names (length n_classes).
    distance_bands : list[float]
        Band edges in metres for stratified IoU. Default: [0.5, 1.0, 2.0, 5.0].
    chunk_size : int
        Batch size for KD-tree operations. Reduce if OOM.
    verbose : bool
        Show progress bars.
    """

    def __init__(
        self,
        n_classes: int,
        class_names: list[str] | None = None,
        distance_bands: list[float] | None = None,
        chunk_size: int = 500_000,
        verbose: bool = True,
    ) -> None:
        self.n_classes = n_classes
        self.class_names = class_names or [str(i) for i in range(n_classes)]
        self.distance_bands = distance_bands or [0.5, 1.0, 2.0, 5.0]
        self.chunk_size = chunk_size
        self.verbose = verbose

        self.dw_confusion = DistanceWeightedConfusionMatrix(n_classes, chunk_size, verbose)
        self.stratified_iou = SpatiallyStratifiedIoU(
            n_classes, self.distance_bands, chunk_size, verbose
        )

    def evaluate(
        self,
        xyz: NDArray[np.float64],
        true_labels: NDArray[np.int32],
        pred_labels: NDArray[np.int32],
    ) -> "EvalResult":
        """
        Run all spatial metrics and return a structured result.

        Parameters
        ----------
        xyz : (N, 3) float64 — georeferenced point coordinates
        true_labels : (N,) int32 — ground-truth labels
        pred_labels : (N,) int32 — predicted labels

        Returns
        -------
        EvalResult
        """
        _validate_inputs(xyz, true_labels, pred_labels)

        self.dw_confusion.fit(xyz, true_labels, pred_labels)
        self.stratified_iou.fit(xyz, true_labels, pred_labels)

        # Standard metrics for comparison
        n = len(xyz)
        correct = (true_labels == pred_labels).sum()
        oa = float(correct / n)
        per_class_iou = self.stratified_iou._iou_from_masks(
            true_labels, pred_labels, np.ones(n, dtype=bool)
        )
        miou = float(np.nanmean(per_class_iou))

        return EvalResult(
            n_points=n,
            n_classes=self.n_classes,
            class_names=self.class_names,
            overall_accuracy=oa,
            miou=miou,
            per_class_iou=per_class_iou,
            dw_confusion=self.dw_confusion,
            stratified_iou=self.stratified_iou,
        )


@dataclass
class EvalResult:
    """Container for all spatial evaluation results."""
    n_points: int
    n_classes: int
    class_names: list[str]
    overall_accuracy: float
    miou: float
    per_class_iou: NDArray[np.float64]
    dw_confusion: DistanceWeightedConfusionMatrix
    stratified_iou: SpatiallyStratifiedIoU

    def summary(self) -> str:
        lines = [
            f"Points evaluated : {self.n_points:,}",
            f"Classes          : {self.n_classes}",
            f"Overall Accuracy : {self.overall_accuracy:.4f}",
            f"Standard mIoU    : {self.miou:.4f}",
            "",
            "Per-class IoU (standard):",
        ]
        for i, (name, iou) in enumerate(zip(self.class_names, self.per_class_iou)):
            lines.append(f"  {name:<20s} {iou:.4f}")
        lines += [
            "",
            "Spatially-stratified mIoU (by distance to boundary):",
        ]
        bands = [0.0] + self.stratified_iou.distance_bands
        for i, (miou_b, count) in enumerate(
            zip(
                self.stratified_iou.mean_iou_per_band(),
                self.stratified_iou.band_point_counts,
            )
        ):
            lo = bands[i]
            hi = bands[i + 1] if i + 1 < len(bands) else float("inf")
            tag = f"[{lo:.1f}m, {'∞' if math.isinf(hi) else f'{hi:.1f}m'})"
            lines.append(f"  {tag:<16s}  mIoU={miou_b:.4f}  n={count:,}")
        lines += [
            "",
            "Mean distance error per class (misclassified points only, metres):",
        ]
        mde = self.dw_confusion.mean_distance_error()
        for name, d in zip(self.class_names, mde):
            lines.append(f"  {name:<20s} {d:.3f} m")
        return "\n".join(lines)
