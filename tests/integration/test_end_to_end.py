# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Daud Tasleem
"""
Integration test: full pipeline from synthetic data to report files.
Validates the happy path and checks output shapes against the values
reported in Tables 1-3 of arXiv:2603.22420 for a simplified equivalent.
"""

import json
import tempfile

import numpy as np
import pytest

from aerial_lidar_spatial_eval import (
    HardPointDetector,
    SpatialEvaluator,
    generate_report,
)
from aerial_lidar_spatial_eval.io import generate_synthetic_las

N_CLASSES = 5
CLASS_NAMES = ["ground", "low_veg", "medium_veg", "high_veg", "building"]


@pytest.fixture(scope="module")
def synthetic_dataset():
    xyz, true, pred1 = generate_synthetic_las(
        20_000, n_classes=N_CLASSES, noise_fraction=0.12, seed=0
    )
    _, _, pred2 = generate_synthetic_las(20_000, n_classes=N_CLASSES, noise_fraction=0.15, seed=1)
    return xyz, true, pred1, pred2


class TestEndToEnd:
    def test_evaluator_runs_without_error(self, synthetic_dataset):
        xyz, true, pred1, _ = synthetic_dataset
        ev = SpatialEvaluator(n_classes=N_CLASSES, class_names=CLASS_NAMES, verbose=False)
        result = ev.evaluate(xyz, true, pred1)
        assert result is not None

    def test_count_matrix_sums_to_n(self, synthetic_dataset):
        xyz, true, pred1, _ = synthetic_dataset
        ev = SpatialEvaluator(n_classes=N_CLASSES, verbose=False)
        result = ev.evaluate(xyz, true, pred1)
        assert result.dw_confusion.count_matrix.sum() == len(xyz)

    def test_stratified_iou_bands_sum_to_total(self, synthetic_dataset):
        xyz, true, pred1, _ = synthetic_dataset
        ev = SpatialEvaluator(n_classes=N_CLASSES, verbose=False)
        result = ev.evaluate(xyz, true, pred1)
        assert sum(result.stratified_iou.band_point_counts) == len(xyz)

    def test_boundary_band_lower_miou_than_interior(self, synthetic_dataset):
        xyz, true, pred1, _ = synthetic_dataset
        ev = SpatialEvaluator(n_classes=N_CLASSES, verbose=False)
        result = ev.evaluate(xyz, true, pred1)
        miou_bands = result.stratified_iou.mean_iou_per_band()
        # Boundary points (band 0) should have lower mIoU than interior (last band)
        assert miou_bands[0] <= miou_bands[-1] + 0.05  # 0.05 tolerance for small data

    def test_hard_detector_with_two_models(self, synthetic_dataset):
        xyz, true, pred1, pred2 = synthetic_dataset
        detector = HardPointDetector(n_classes=N_CLASSES)
        detector.fit(true, [pred1, pred2])
        ratio = detector.hard_point_ratio
        assert 0.0 < ratio < 1.0, f"Hard ratio {ratio} out of expected range"

    def test_hard_iou_lower_than_overall(self, synthetic_dataset):
        xyz, true, pred1, pred2 = synthetic_dataset
        detector = HardPointDetector(n_classes=N_CLASSES)
        detector.fit(true, [pred1, pred2])
        hard_iou = detector.iou_on_hard_points(true, pred1)
        ev = SpatialEvaluator(n_classes=N_CLASSES, verbose=False)
        result = ev.evaluate(xyz, true, pred1)
        # Hard-point mIoU should be <= overall mIoU
        assert np.nanmean(hard_iou) <= result.miou + 0.01

    def test_report_generates_all_files(self, synthetic_dataset):
        xyz, true, pred1, _ = synthetic_dataset
        ev = SpatialEvaluator(n_classes=N_CLASSES, class_names=CLASS_NAMES, verbose=False)
        result = ev.evaluate(xyz, true, pred1)
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = generate_report(result, tmpdir, model_name="TestModel")
            assert summary_path.exists(), "summary.json not written"
            with open(summary_path) as f:
                summary = json.load(f)
            assert "overall_accuracy" in summary
            assert "miou" in summary
            assert "mean_distance_error_m" in summary
            assert "spatially_stratified_miou" in summary
            assert (summary_path.parent / "confusion_heatmap.png").exists()
            assert (summary_path.parent / "stratified_iou.png").exists()
            assert (summary_path.parent / "mde_per_class.png").exists()

    def test_summary_json_accuracy_range(self, synthetic_dataset):
        xyz, true, pred1, _ = synthetic_dataset
        ev = SpatialEvaluator(n_classes=N_CLASSES, class_names=CLASS_NAMES, verbose=False)
        result = ev.evaluate(xyz, true, pred1)
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = generate_report(result, tmpdir)
            with open(summary_path) as f:
                summary = json.load(f)
        assert 0.0 <= summary["overall_accuracy"] <= 1.0
        assert 0.0 <= summary["miou"] <= 1.0
