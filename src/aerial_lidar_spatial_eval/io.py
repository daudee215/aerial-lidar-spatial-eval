# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Daud Tasleem
"""I/O helpers for LAS/LAZ files with prediction labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

try:
    import laspy  # type: ignore[import-untyped]
    _HAS_LASPY = True
except ImportError:
    _HAS_LASPY = False


def load_las_predictions(
    las_path: str | Path,
    pred_field: str = "user_data",
    gt_field: str = "classification",
    xyz_scale: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int32]]:
    """
    Load a LAS/LAZ file and extract xyz, ground-truth labels, and
    predicted labels.

    Parameters
    ----------
    las_path : str | Path
        Path to .las or .laz file.
    pred_field : str
        Extra-byte field containing predicted class labels. Default: "user_data".
    gt_field : str
        Standard LAS field containing ground-truth classification.
        Default: "classification".
    xyz_scale : float
        Scale factor applied to coordinates (e.g. 0.001 to convert mm → m).

    Returns
    -------
    xyz : (N, 3) float64
    true_labels : (N,) int32
    pred_labels : (N,) int32
    """
    if not _HAS_LASPY:
        raise ImportError("laspy is required: pip install laspy")

    las = laspy.read(str(las_path))
    xyz = np.column_stack([
        np.asarray(las.x, dtype=np.float64),
        np.asarray(las.y, dtype=np.float64),
        np.asarray(las.z, dtype=np.float64),
    ]) * xyz_scale

    true_labels = np.asarray(getattr(las, gt_field), dtype=np.int32)

    try:
        pred_labels = np.asarray(getattr(las, pred_field), dtype=np.int32)
    except AttributeError as exc:
        available = [str(d.name) for d in las.point_format.extra_dims]
        raise AttributeError(
            f"Field '{pred_field}' not found. Available extra fields: {available}"
        ) from exc

    return xyz, true_labels, pred_labels


def generate_synthetic_las(
    n_points: int,
    n_classes: int,
    noise_fraction: float = 0.05,
    seed: int = 42,
    output_path: str | Path | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int32]]:
    """
    Generate synthetic labelled point cloud data for testing and benchmarking.

    Creates spatially coherent class regions with configurable boundary
    noise to simulate real segmentation outputs.

    Parameters
    ----------
    n_points : int
        Total number of points.
    n_classes : int
        Number of distinct semantic classes.
    noise_fraction : float
        Fraction of boundary-region points with incorrect predictions.
    seed : int
        Random seed.
    output_path : str | Path | None
        If provided, write to a LAS file (requires laspy).

    Returns
    -------
    xyz, true_labels, pred_labels
    """
    rng = np.random.default_rng(seed)

    # Spatially coherent: tile the 1km × 1km area into a grid of class regions
    side = 1000.0  # metres
    grid_size = int(np.ceil(np.sqrt(n_classes)))
    cell = side / grid_size

    xyz = rng.uniform([0, 0, 0], [side, side, 30], size=(n_points, 3)).astype(np.float64)

    # Assign labels by spatial grid cell
    gx = np.clip((xyz[:, 0] / cell).astype(np.int32), 0, grid_size - 1)
    gy = np.clip((xyz[:, 1] / cell).astype(np.int32), 0, grid_size - 1)
    true_labels = ((gx + gy * grid_size) % n_classes).astype(np.int32)

    # Perfect prediction initially
    pred_labels = true_labels.copy()

    # Inject noise at boundaries
    from scipy.spatial import KDTree
    tree = KDTree(xyz)
    _, nn_idx = tree.query(xyz, k=5, workers=-1)
    nn_labels = true_labels[nn_idx]
    is_boundary = np.any(nn_labels != true_labels[:, None], axis=1)
    boundary_idx = np.where(is_boundary)[0]
    n_noise = max(1, int(noise_fraction * len(boundary_idx)))
    noise_idx = rng.choice(boundary_idx, size=n_noise, replace=False)
    pred_labels[noise_idx] = rng.integers(0, n_classes, size=n_noise, dtype=np.int32)

    if output_path is not None:
        if not _HAS_LASPY:
            raise ImportError("laspy is required to write LAS files")
        header = laspy.LasHeader(point_format=6, version="1.4")
        header.add_extra_dim(laspy.ExtraBytesParams(name="user_data", type=np.int32))
        las = laspy.LasData(header=header)
        las.x = xyz[:, 0]
        las.y = xyz[:, 1]
        las.z = xyz[:, 2]
        las.classification = true_labels.astype(np.uint8)
        las.user_data = pred_labels.astype(np.int32)
        las.write(str(output_path))

    return xyz, true_labels, pred_labels
