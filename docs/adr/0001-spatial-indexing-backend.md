# ADR-0001: Spatial Indexing Backend Selection

**Date:** 2026-04-27  
**Status:** Accepted  
**Deciders:** Daud Tasleem

---

## Context

The core operation of this library is, for every misclassified point in a
point cloud of 10 M – 100 M points, finding the Euclidean distance to the
nearest point of its ground-truth class.  This is a k-NN query (k = 1) in
3-D space, repeated for every off-diagonal entry in the confusion matrix.

Five candidate backends exist:

| Backend | Notes |
|---|---|
| `scipy.spatial.KDTree` | Pure Python / NumPy; supports `workers=-1` (parallel via threads) |
| `scipy.spatial.cKDTree` | C extension; same API as `KDTree` since scipy 1.9 |
| `open3d.geometry.KDTreeFlann` | GPU-optional; ~500 MB install; Open3D 0.18+ |
| `FLANN` (pyflann3) | Approximate; randomised kd-tree |
| `faiss` (Meta) | GPU-first; exact or approximate; large install |

---

## Decision

Use **`scipy.spatial.KDTree`** (which resolves to `cKDTree` in scipy ≥ 1.9)
as the default spatial index, with multi-threaded queries via `workers=-1`.

---

## Rationale

1. **Zero extra dependencies.** `scipy` is already a required dependency for
   statistical tests in the report module.  Every other backend would add a
   hard dependency.

2. **Sufficient throughput.** On a 200 k-point tile (a realistic ALS flight
   strip), `KDTree` with `workers=-1` achieves 1.5–3 M pts/s on a 4-core
   CPU, giving sub-3-minute evaluation for a 500 M-point full flight.

3. **Exact results.** Approximate backends (FLANN, faiss IVF) introduce
   errors in the distance values themselves, which would corrupt the
   distance-weighted confusion matrix and invalidate validation against the
   reference results in arXiv:2603.22420.

4. **Cross-platform CI.** Open3D and faiss both have non-trivial build
   requirements that have historically broken CI on macOS/Windows runners.

---

## Rejected Alternatives

### A — `open3d.geometry.KDTreeFlann`

- **Rejected because:** Requires installing Open3D (~500 MB wheel), which is
  not a reasonable transitive dependency for an evaluation library.  The GPU
  acceleration path is untested at aerial LiDAR point densities
  (≥ 50 pts/m²) and has known accuracy regressions with float32 coordinates
  at UTM-scale values (> 1 M metres).

### B — FLANN (pyflann3)

- **Rejected because:** FLANN returns *approximate* nearest neighbours.
  The approximation error (controlled by `checks` parameter) makes the
  per-point distances non-deterministic and non-reproducible, which
  violates the reproducibility requirement for correctness validation
  against arXiv:2603.22420 Tables 1–3.

### C — faiss

- **Rejected because:** faiss is GPU-first and its CPU-only mode is still
  ~50 MB larger than scipy.  The IndexFlatL2 exact CPU index has no
  threading model superior to scipy's `workers=-1` for the workload sizes
  we target.  The dependency is unreasonable for an evaluation library.

---

## Consequences

- Users on machines without multi-core CPUs (e.g. CI with `workers=1`) may
  observe lower throughput.  The `chunk_size` parameter allows trading
  memory for parallelism.
- Future versions may add an optional `backend` parameter to allow advanced
  users to plug in open3d or faiss for GPU-accelerated evaluation on very
  large datasets (> 1 B points).
