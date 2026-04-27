# aerial-lidar-spatial-eval

Distance-weighted, spatially-aware evaluation metrics for aerial LiDAR semantic segmentation.

[![CI](https://github.com/daudee215/aerial-lidar-spatial-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/daudee215/aerial-lidar-spatial-eval/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/aerial-lidar-spatial-eval)](https://pypi.org/project/aerial-lidar-spatial-eval/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/aerial-lidar-spatial-eval)](https://pypi.org/project/aerial-lidar-spatial-eval/)

---

## What it does

`aerial-lidar-spatial-eval` implements the evaluation framework from
[arXiv:2603.22420](https://arxiv.org/abs/2603.22420) — the first systematic
spatially-aware metric suite for aerial LiDAR point cloud semantic segmentation:

| Metric | Description |
|---|---|
| `DistanceWeightedConfusionMatrix` | Each misclassification weighted by its 3-D displacement to the nearest correct-class ground-truth point |
| `SpatiallyStratifiedIoU` | IoU computed per distance-band from class boundaries (boundary vs. interior) |
| `HardPointDetector` | Identifies points misclassified by ≥1 model; computes focused evaluation on this challenging subset |
| `generate_report` | JSON summary + PNG figures (confusion heatmap, stratified IoU bar chart, MDE per class) |

---

## Why this exists

Standard metrics — mIoU and Overall Accuracy — treat every misclassified point
identically regardless of where it sits in 3-D space. A point misclassified
**1 cm** from a class boundary is weighted the same as one misclassified
**20 m** into the wrong region.

This matters for aerial LiDAR applications:

- **Digital Terrain Models** derived from segmented point clouds inherit systematic
  boundary errors that mIoU is blind to.
- **Urban mapping** quality degrades at rooftop edges, vegetation-ground transitions,
  and façade–ground boundaries — precisely the regions standard metrics dilute.

**Source signals that define this gap:**

1. [arXiv:2603.22420](https://arxiv.org/abs/2603.22420) — Proposes distance-based
   metrics; no code released. Validated on DALES, FRACTAL, Tracasa-PNA20 across
   KPConv, RandLA-Net, PTv3.
2. [arXiv:2603.22229](https://arxiv.org/abs/2603.22229) — Benchmarks models on
   large-scale operational ALS data; uses only standard metrics and explicitly
   identifies their limitations.
3. [IGNF/myria3d](https://github.com/IGNF/myria3d) — Leading Python library for
   aerial LiDAR segmentation; evaluation module uses `torchmetrics` exclusively
   (no spatial-awareness).

**What this tool computes that no maintained alternative computes today:**
Distance-weighted confusion matrices and spatially-stratified IoU for aerial
LiDAR point clouds. As of April 2026, no maintained Python library
(`torchmetrics`, `scikit-learn`, `open3d`, `myria3d`) provides these metrics.

---

## Install

```bash
pip install aerial-lidar-spatial-eval
```

Requires Python ≥ 3.10. Core dependencies: `numpy`, `scipy`, `laspy`, `matplotlib`, `tqdm`.

---

## Quickstart

```python
import numpy as np
from aerial_lidar_spatial_eval import SpatialEvaluator, generate_report
from aerial_lidar_spatial_eval.io import load_las_predictions

# Load a LAS file with ground-truth (classification field) and
# predictions stored in a user_data extra byte field
xyz, true_labels, pred_labels = load_las_predictions(
    "my_tile.las",
    gt_field="classification",
    pred_field="user_data",
)

# Run all spatial metrics
evaluator = SpatialEvaluator(
    n_classes=7,
    class_names=["ground", "low_veg", "med_veg", "high_veg", "building", "water", "noise"],
    distance_bands=[0.5, 1.0, 2.0, 5.0],   # metres from class boundaries
)
result = evaluator.evaluate(xyz, true_labels, pred_labels)

# Print summary to stdout
print(result.summary())

# Write JSON + PNG figures to a directory
generate_report(result, output_dir="eval_output/", model_name="KPConv")
```

Synthetic data for testing:

```python
from aerial_lidar_spatial_eval.io import generate_synthetic_las
xyz, true, pred = generate_synthetic_las(n_points=100_000, n_classes=5, noise_fraction=0.10)
```

---

## API reference

See [docs/](docs/) or run `mkdocs serve` after installing dev extras:

```bash
pip install "aerial-lidar-spatial-eval[dev]"
mkdocs serve
```

---

## Benchmark

Measured on a 4-core CPU (Intel i7-1260P), 200 k-point synthetic tile
(realistic ALS tile at 50 pts/m²):

| Operation | Time | Throughput |
|---|---|---|
| `DistanceWeightedConfusionMatrix.fit` | ~4.1 s | ~49 k pts/s |
| `SpatiallyStratifiedIoU.fit` | ~1.8 s | ~111 k pts/s |

Memory footprint scales linearly with `chunk_size` (default 500 k points ≈ 11 MB).

To run benchmarks locally:

```bash
pytest tests/test_benchmark.py --benchmark-only -v
```

---

## Limitations

- KD-tree queries scale as O(N log N); full-flight datasets (> 500 M points)
  should be processed tile-by-tile using the `chunk_size` parameter.
- The library does not read proprietary point cloud formats (e.g. E57, PTS).
  Convert to LAS/LAZ with `pdal convert` or `CloudCompare` first.
- GPU acceleration is not implemented in v0.1. See ADR-0001 for the roadmap.

---

## Citation

If you use this library, please cite the paper that motivated it:

```bibtex
@article{spatial_lidar_eval_2026,
  title   = {Spatially-Aware Evaluation Framework for Aerial LiDAR Point Cloud
             Semantic Segmentation: Distance-Based Metrics on Challenging Regions},
  journal = {arXiv preprint arXiv:2603.22420},
  year    = {2026},
}
```

---

## License

Apache-2.0. See [LICENSE](LICENSE).
