# aerial-lidar-spatial-eval

Distance-weighted, spatially-aware evaluation metrics for aerial LiDAR semantic segmentation.

See the [README](https://github.com/daudee215/aerial-lidar-spatial-eval#readme) for quickstart and installation instructions.

## Why this exists

Standard metrics (mIoU, OA) treat every misclassified point identically regardless of spatial position.
This library implements the distance-weighted evaluation framework from [arXiv:2603.22420](https://arxiv.org/abs/2603.22420).
