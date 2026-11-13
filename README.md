# gyre-tracking

Eddy tracking on an analytic **double-gyre** flow, built around a per-frame
**linear assignment** core whose output carries a *verified optimality
certificate*: the solver returns dual potentials, and three O(n^2) checks
(dual feasibility, complementary slackness, strong duality) prove each
frame's matching optimal. **Trust the certificate, not the implementation.**

## Pipeline

```
analytic double-gyre flow  ->  closed-form vortex-center trajectories (ground truth)
        ↓  Gaussian noise / missed detections / Poisson clutter
per-frame detections (no identity)
        ↓  constant-velocity prediction + gate  →  padded cost matrix (dummy columns)
Hungarian method (from scratch, O(n^3), dual potentials)  →  per-frame assignment
        ↓  track lifecycle (birth / coast / death)  →  trajectories
optional: fragment stitching (same solver, one level up)
        ↓  compare against closed-form truth
association accuracy / track fragmentation / ID switches
```

## Status

Work in progress.

- **Implemented**: analytic flow, closed-form ground-truth centers, detection
  simulator, optimality certificates, full test suite.
- **In progress**: `tracking.hungarian.solve` (shortest augmenting path with
  dual potentials) — the test files under `tests/test_hungarian_*.py` are its
  acceptance criteria; association & track lifecycle; stitching; metrics;
  experiment sweeps.

## Quickstart

```bash
pip install -e ".[dev]"
pytest
```

`tests/test_flow_truth.py` passes today (analytic checks). The Hungarian test
files fail until `src/tracking/hungarian.py::solve` is implemented — by
design: tests first, solver second.

## Layout

```
src/tracking/
  flow.py           analytic double gyre: stream function, velocity, RK4
  truth.py          closed-form vortex-center trajectories (ground truth)
  detector_sim.py   noisy detections: misses + clutter
  hungarian.py      from-scratch assignment solver  <- the piece to implement
  certificates.py   dual feasibility / complementary slackness / strong duality
  associate.py      gating, padded cost matrix, track lifecycle
  stitch.py         fragment stitching (reuse of the same solver)
  metrics.py        association accuracy, fragmentation, ID switches
  viz.py            animations and summary plots
tests/              acceptance criteria (brute force, certificates, invariance)
experiments/        greedy-vs-Hungarian sweeps
```

## Notes

- The double gyre (Shadden, Lekien & Marsden 2005) has *closed-form* vortex
  centers, so every tracking error is measurable against exact truth.
- Deliberate simplifications (see README of the final report): constant
  velocity instead of Kalman, isotropic noise instead of Mahalanobis gating,
  two-pass stitching instead of multi-frame min-cost flow.
