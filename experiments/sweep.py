"""Solver comparison sweep: certified Hungarian vs greedy baseline.

Scenarios x sigma x p_detect x clutter x 20 seeds x 2 solvers, 200 frames
per run. S1 = double-gyre centers; S2 = crossing constant-velocity objects
(see scenarios.py). Writes experiments/results/sweep.csv, one row per run;
experiments/plot_sweep.py aggregates into per-scenario figures.
"""

from __future__ import annotations

import csv
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tracking import associate, detector_sim, flow, metrics, scenarios  # noqa: E402
from tracking.hungarian import greedy_solve  # noqa: E402

N_FRAMES = 200
DT = 0.1
SEEDS = range(20)
SIGMAS = [0.01, 0.03, 0.06, 0.10]
P_DETECTS = [0.95, 0.85, 0.70]
CLUTTERS = [0.0, 1.0, 3.0]
SCENARIOS = ["S1", "S2"]


def greedy_solver(C):
    """Greedy baseline as a drop-in solver: same (sigma, u, v) shape as
    hungarian.solve, with fake duals (u = v = 0). Run with verify=False —
    greedy is not optimal and has no certificate to check."""
    return greedy_solve(C), np.zeros(len(C)), np.zeros(len(C))


def truth_for(scenario: str, times: np.ndarray) -> np.ndarray:
    if scenario == "S1":
        return scenarios.s1_gyre(flow.DoubleGyre(), times)
    if scenario == "S2":
        return scenarios.s2_crossing(times)
    raise ValueError(scenario)


def run_one(scenario, sigma, p_detect, clutter, seed, solver_name):
    gate = 6.8 * sigma + 0.02
    times = np.arange(N_FRAMES) * DT
    truth = truth_for(scenario, times)
    rng = np.random.default_rng(seed)
    frames = detector_sim.simulate_tracks(truth, sigma, p_detect, clutter, rng)
    if solver_name == "hungarian":
        tm = associate.TrackManager(gate_sq=gate**2, dt=DT)
    elif solver_name == "greedy":
        tm = associate.TrackManager(gate_sq=gate**2, dt=DT,
                                    solver=greedy_solver, verify=False)
    else:
        raise ValueError(solver_name)
    gt, matches = [], []
    for f in frames:
        matches.append(tm.update(f.points))
        gt.append(f.source)
    frag = metrics.fragmentation_counts(gt, matches)
    return {
        "assoc": metrics.association_accuracy(gt, matches),
        "frag": sum(frag.values()) / len(frag) if frag else float("nan"),
        "ids": metrics.id_switches(gt, matches),
    }


def main():
    out = pathlib.Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    csv_path = out / "sweep.csv"
    t0 = time.time()
    n = 0
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "sigma", "p_detect", "clutter", "seed",
                    "solver", "assoc", "frag", "ids"])
        for scenario in SCENARIOS:
            for sigma in SIGMAS:
                for p in P_DETECTS:
                    for c in CLUTTERS:
                        for seed in SEEDS:
                            for name in ("hungarian", "greedy"):
                                r = run_one(scenario, sigma, p, c, seed, name)
                                w.writerow([scenario, sigma, p, c, seed, name,
                                            f"{r['assoc']:.4f}",
                                            f"{r['frag']:.4f}", r["ids"]])
                                n += 1
                        fh.flush()
                    print(f"[{time.time() - t0:6.0f}s] {scenario} sigma={sigma} "
                          f"p={p}: {c:.0f} clutter done ({n} runs)", flush=True)
    print(f"{n} runs -> {csv_path}  ({time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
