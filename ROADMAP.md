# Roadmap

## v0.1.0 — Initial release (ships now)

- `DistanceWeightedConfusionMatrix`: per-class distance-weighted confusion matrix with mean distance error
- `SpatiallyStratifiedIoU`: IoU computed per distance-band from class boundaries
- `HardPointDetector`: multi-model hard-point identification and focused evaluation
- `SpatialEvaluator`: high-level wrapper running all metrics in one call
- `generate_report`: JSON summary + PNG figures (confusion heatmap, stratified IoU, MDE per class)
- LAS/LAZ I/O via `laspy`; synthetic data generator for testing and benchmarking
- Apache-2.0 license; GitHub Actions CI (test + lint + type-check + docs + audit)

## v0.2.0 — Performance and formats

- Optional `open3d` backend for GPU-accelerated KD-tree queries (10× throughput on large tiles)
- Streaming tile-by-tile evaluation over full-flight LAS datasets without full load into RAM
- COPC (Cloud-Optimized Point Cloud) reader for direct cloud streaming
- Per-tile report aggregation with flight-level summary statistics
- CLI entry point: `alse evaluate --las prediction.laz --gt ground_truth.laz --out report/`

## v1.0.0 — Stable API and ecosystem integration

- Stable public API with semantic versioning guarantees
- `myria3d` integration: drop-in callback that adds spatial metrics to existing training loops
- `torchmetrics`-compatible interface (`update` / `compute`) for Lightning / Ignite loops
- Validation against full published results from arXiv:2603.22420 on DALES and FRACTAL
- Documentation site deployed to GitHub Pages
