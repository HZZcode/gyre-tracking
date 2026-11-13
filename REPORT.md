# Gyre-Tracking Experiment Report

Status: in progress
Date: -

## 1. Test objectives

Does an optimal per-frame assignment (Hungarian, with a verified dual
certificate) produce measurably better trajectories than a greedy matcher as
noise, missed-detection rate and clutter grow?

## 2. Scenarios and ground truth

- S1 `double-gyre`: detections sampled around the closed-form centers of the
  analytic double gyre (Shadden et al. 2005).
- S2 `crossing`: two constant-velocity targets crossing at a shallow angle —
  the pathological case for greedy matching.

Ground truth is exact (closed form), so every association error is measurable.

## 3. Conditions

- sigma in {0.01, 0.03, 0.06, 0.10}
- p_detect in {0.95, 0.85, 0.70}
- clutter rate in {0, 1, 3} per frame
- 20 Monte-Carlo seeds per cell
- algorithms: {greedy, hungarian}

## 4. Results

(tbd — filled after `experiments/sweep.py` runs)

## 5. Known limitations

(tbd)
