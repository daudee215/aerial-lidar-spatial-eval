# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Daud Tasleem
"""
aerial-lidar-spatial-eval
=========================
Distance-weighted, spatially-aware evaluation metrics for aerial LiDAR
semantic segmentation point clouds.

Why this exists
---------------
Standard metrics (mIoU, Overall Accuracy) treat every misclassified point
identically regardless of where it sits in 3D space.  Boundary-region
errors — where geometric context matters most for downstream products like
Digital Terrain Models — are systematically diluted by the mass of easy
interior points.  This library implements the distance-weighted evaluation
framework proposed in [arXiv:2603.22420] and validated against DALES,
FRACTAL, and Tracasa-PNA20 datasets.

References
----------
- arXiv:2603.22420  Spatially-Aware Evaluation Framework for Aerial LiDAR
  Point Cloud Semantic Segmentation: Distance-Based Metrics on Challenging Regions
- arXiv:2603.22229  Benchmarking Deep Learning Models for Aerial LiDAR Point
  Cloud Semantic Segmentation under Real Acquisition Conditions
- FRACTAL dataset   https://github.com/IGNF/FRACTAL
"""

from aerial_lidar_spatial_eval.metrics import (
    SpatialEvaluator,
    DistanceWeightedConfusionMatrix,
    SpatiallyStratifiedIoU,
    HardPointDetector,
)
from aerial_lidar_spatial_eval.io import load_las_predictions
from aerial_lidar_spatial_eval.report import generate_report

__all__ = [
    "SpatialEvaluator",
    "DistanceWeightedConfusionMatrix",
    "SpatiallyStratifiedIoU",
    "HardPointDetector",
    "load_las_predictions",
    "generate_report",
]

__version__ = "0.1.0"
