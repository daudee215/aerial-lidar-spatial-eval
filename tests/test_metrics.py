# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Daud Tasleem
"""Unit tests for aerial_lidar_spatial_eval.metrics."""

import numpy as np
import pytest

from aerial_lidar_spatial_eval.metrics import (
    DistanceWeightedConfusionMatrix,
    EvalResult,
    HardPointDetector,
    SpatialEvaluator,
    SpatiallyStratifiedIoU,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two perfectly-separated classes, no errors."""
    rng = np.random.default_rng(0)
    n = 1_000
    xyz = rng.uniform(0, 100, (n, 3))
    # Left half = class 0, right half = class 1
    true = (xyz[:, 0] > 50).astype(np.int32)
    pred = true.copy()
    return xyz.astype(np.float64), true, pred


@pytest.fixture
def noisy_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two classes with 10% boundary noise."""
    from aerial_lidar_spatial_eval.io import generate_synthetic_las
    xyz, true, pred = generate_synthetic_las(5_000, n_classes=3, noise_fraction=0.15, seed=7)
    return xyz, true, pred


# ---------------------------------------------------------------------------
# DistanceWeightedConfusionMatrix
# ---------------------------------------------------------------------------

class TestDistanceWeightedConfusionMatrix:
    def test_perfect_predictions_zero_off_diagonal(self, perfect_predictions):
        xyz, true, pred = perfect_predictions
        dwcm = DistanceWeightedConfusionMatrix(n_classes=2, verbose=False)
        dwcm.fit(xyz, true, pred)
        mat = dwcm.matrix
        np.fill_diagonal(mat, 0.0)
        assert mat.sum() == pytest.approx(0.0), "Off-diagonal should be 0 for perfect predictions"

    def test_diagonal_counts_match_class_counts(self, perfect_predictions):
        xyz, true, pred = perfect_predictions
        dwcm = DistanceWeightedConfusionMatrix(n_classes=2, verbose=False)
        dwcm.fit(xyz, true, pred)
        cm = dwcm.count_matrix
        for c in range(2):
            expected = int(np.sum(true == c))
            assert cm[c, c] == expected

    def test_noisy_predictions_nonzero_mde(self, noisy_predictions):
        xyz, true, pred = noisy_predictions
        dwcm = DistanceWeightedConfusionMatrix(n_classes=3, verbose=False)
        dwcm.fit(xyz, true, pred)
        mde = dwcm.mean_distance_error()
        # At least one class should have nonzero MDE given boundary noise
        assert mde.max() > 0, "Expected nonzero MDE for noisy predictions"

    def test_mde_nonnegative(self, noisy_predictions):
        xyz, true, pred = noisy_predictions
        dwcm = DistanceWeightedConfusionMatrix(n_classes=3, verbose=False)
        dwcm.fit(xyz, true, pred)
        assert np.all(dwcm.mean_distance_error() >= 0)

    def test_count_matrix_totals(self, noisy_predictions):
        xyz, true, pred = noisy_predictions
        n = len(xyz)
        dwcm = DistanceWeightedConfusionMatrix(n_classes=3, verbose=False)
        dwcm.fit(xyz, true, pred)
        assert dwcm.count_matrix.sum() == n

    def test_chunked_equals_single_pass(self, noisy_predictions):
        xyz, true, pred = noisy_predictions
        dwcm1 = DistanceWeightedConfusionMatrix(n_classes=3, chunk_size=500, verbose=False)
        dwcm1.fit(xyz, true, pred)
        dwcm2 = DistanceWeightedConfusionMatrix(n_classes=3, chunk_size=100_000, verbose=False)
        dwcm2.fit(xyz, true, pred)
        np.testing.assert_array_equal(dwcm1.count_matrix, dwcm2.count_matrix)


# ---------------------------------------------------------------------------
# SpatiallyStratifiedIoU
# ---------------------------------------------------------------------------

class TestSpatiallyStratifiedIoU:
    def test_perfect_predictions_miou_one(self, perfect_predictions):
        xyz, true, pred = perfect_predictions
        siou = SpatiallyStratifiedIoU(n_classes=2, distance_bands=[1.0, 5.0], verbose=False)
        siou.fit(xyz, true, pred)
        # Interior band (far from boundary) should have mIoU = 1.0
        miou_per_band = siou.mean_iou_per_band()
        assert miou_per_band[-1] == pytest.approx(1.0, abs=1e-6)

    def test_band_count_sums_to_total(self, noisy_predictions):
        xyz, true, pred = noisy_predictions
        siou = SpatiallyStratifiedIoU(n_classes=3, distance_bands=[1.0, 3.0], verbose=False)
        siou.fit(xyz, true, pred)
        assert sum(siou.band_point_counts) == len(xyz)

    def test_boundary_band_lower_miou(self, noisy_predictions):
        """Boundary band should have lower mIoU than interior."""
        xyz, true, pred = noisy_predictions
        siou = SpatiallyStratifiedIoU(n_classes=3, distance_bands=[2.0, 10.0], verbose=False)
        siou.fit(xyz, true, pred)
        mious = siou.mean_iou_per_band()
        # The last (interior) band should have >= mIoU than the first (boundary) band
        assert mious[-1] >= mious[0] - 1e-9  # allow tiny float tolerance

    def test_number_of_bands(self, perfect_predictions):
        xyz, true, pred = perfect_predictions
        bands = [0.5, 1.0, 2.0, 5.0]
        siou = SpatiallyStratifiedIoU(n_classes=2, distance_bands=bands, verbose=False)
        siou.fit(xyz, true, pred)
        assert len(siou.band_iou) == len(bands) + 1


# ---------------------------------------------------------------------------
# HardPointDetector
# ---------------------------------------------------------------------------

class TestHardPointDetector:
    def test_hard_ratio_between_zero_and_one(self, noisy_predictions):
        xyz, true, pred = noisy_predictions
        detector = HardPointDetector(n_classes=3)
        detector.fit(true, [pred])
        r = detector.hard_point_ratio
        assert 0.0 <= r <= 1.0

    def test_perfect_pred_zero_hard_ratio(self, perfect_predictions):
        xyz, true, pred = perfect_predictions
        detector = HardPointDetector(n_classes=2)
        detector.fit(true, [pred])
        assert detector.hard_point_ratio == pytest.approx(0.0)

    def test_multiple_models_increases_hard_ratio(self, noisy_predictions):
        xyz, true, pred1 = noisy_predictions
        # Create a second predictor with independent noise
        from aerial_lidar_spatial_eval.io import generate_synthetic_las
        _, _, pred2 = generate_synthetic_las(len(xyz), n_classes=3, noise_fraction=0.15, seed=99)
        pred2 = pred2[:len(pred1)]
        d1 = HardPointDetector(n_classes=3)
        d1.fit(true, [pred1])
        d2 = HardPointDetector(n_classes=3)
        d2.fit(true, [pred1, pred2])
        assert d2.hard_point_ratio >= d1.hard_point_ratio

    def test_iou_on_hard_points_shape(self, noisy_predictions):
        xyz, true, pred = noisy_predictions
        detector = HardPointDetector(n_classes=3)
        detector.fit(true, [pred])
        iou = detector.iou_on_hard_points(true, pred)
        assert iou.shape == (3,)


# ---------------------------------------------------------------------------
# SpatialEvaluator (integration of all components)
# ---------------------------------------------------------------------------

class TestSpatialEvaluator:
    def test_evaluate_returns_eval_result(self, noisy_predictions):
        xyz, true, pred = noisy_predictions
        ev = SpatialEvaluator(n_classes=3, verbose=False)
        result = ev.evaluate(xyz, true, pred)
        assert isinstance(result, EvalResult)
        assert result.n_points == len(xyz)

    def test_overall_accuracy_range(self, noisy_predictions):
        xyz, true, pred = noisy_predictions
        ev = SpatialEvaluator(n_classes=3, verbose=False)
        result = ev.evaluate(xyz, true, pred)
        assert 0.0 <= result.overall_accuracy <= 1.0

    def test_miou_range(self, noisy_predictions):
        xyz, true, pred = noisy_predictions
        ev = SpatialEvaluator(n_classes=3, verbose=False)
        result = ev.evaluate(xyz, true, pred)
        assert 0.0 <= result.miou <= 1.0

    def test_summary_string_nonempty(self, noisy_predictions):
        xyz, true, pred = noisy_predictions
        ev = SpatialEvaluator(
            n_classes=3, class_names=["ground", "vegetation", "building"], verbose=False
        )
        result = ev.evaluate(xyz, true, pred)
        s = result.summary()
        assert len(s) > 100
        assert "mIoU" in s

    def test_invalid_input_raises(self):
        ev = SpatialEvaluator(n_classes=2, verbose=False)
        xyz = np.zeros((10, 3))
        true = np.zeros(10, dtype=np.int32)
        pred = np.zeros(9, dtype=np.int32)  # wrong length
        with pytest.raises(ValueError):
            ev.evaluate(xyz, true, pred)
