# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Daud Tasleem
"""
Benchmark: throughput (points/second) for distance-weighted metrics.
Run with:  pytest tests/test_benchmark.py --benchmark-only
"""

import pytest

from aerial_lidar_spatial_eval.io import generate_synthetic_las
from aerial_lidar_spatial_eval.metrics import (
    DistanceWeightedConfusionMatrix,
    SpatiallyStratifiedIoU,
)


@pytest.fixture(scope="module")
def large_dataset():
    """~200k points — realistic tile size for ALS data."""
    return generate_synthetic_las(200_000, n_classes=5, noise_fraction=0.10, seed=42)


def _run_dwcm(xyz, true, pred):
    dwcm = DistanceWeightedConfusionMatrix(n_classes=5, chunk_size=50_000, verbose=False)
    dwcm.fit(xyz, true, pred)
    return dwcm


def _run_stratified_iou(xyz, true, pred):
    siou = SpatiallyStratifiedIoU(n_classes=5, distance_bands=[0.5, 1.0, 2.0, 5.0], verbose=False)
    siou.fit(xyz, true, pred)
    return siou


def test_dwcm_throughput(benchmark, large_dataset):
    xyz, true, pred = large_dataset
    benchmark(_run_dwcm, xyz, true, pred)
    n = len(xyz)
    rate = n / benchmark.stats["mean"]
    print(f"\nDWCM throughput: {rate/1e6:.2f} M pts/s")


def test_stratified_iou_throughput(benchmark, large_dataset):
    xyz, true, pred = large_dataset
    benchmark(_run_stratified_iou, xyz, true, pred)
    n = len(xyz)
    rate = n / benchmark.stats["mean"]
    print(f"\nStratified IoU throughput: {rate/1e6:.2f} M pts/s")
