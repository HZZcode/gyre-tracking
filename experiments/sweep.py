"""Greedy-vs-Hungarian association sweeps (experiment driver). TODO

Scenarios:
- S1 `double-gyre`: detections from tracking.detector_sim on the analytic flow;
- S2 `crossing`: two constant-velocity targets crossing at a shallow angle,
  the pathological case for greedy matching.

Conditions: sigma in {0.01, 0.03, 0.06, 0.10}, p_detect in {0.95, 0.85, 0.70},
clutter in {0, 1, 3}, 20 seeds per cell, algorithms {greedy, hungarian}.

Output: experiments/results/sweep.csv with columns
[scenario, algo, sigma, p_detect, clutter, seed, assoc_acc, frag_mean, id_switches]
"""

from __future__ import annotations


def run_condition(
    scenario: str,
    algo: str,
    sigma: float,
    p_detect: float,
    clutter: float,
    seed: int,
) -> dict:
    """Run one (scenario, algorithm, noise cell) and return one metric row."""
    raise NotImplementedError


def main() -> None:
    """Run the full sweep and write experiments/results/sweep.csv."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
