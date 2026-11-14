"""Aggregate sweep.csv into per-scenario solver-comparison figures.

Reads experiments/results/sweep.csv, writes:
- experiments/results/sweep_summary.csv (mean/std per condition, all of it)
- figures/sweep_s1.png: association accuracy + fragments per object
- figures/sweep_s2.png: association accuracy + ID switches (the crossing
  scenario's headline metric)
"""

from __future__ import annotations

import collections
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
P_SLICE = 0.85              # which p_detect slice the figures show
CLUTTER_COLS = [0.0, 1.0, 3.0]
METRIC_LABEL = {"assoc": "association accuracy",
                "frag": "fragments per object",
                "ids": "ID switches per run"}
SCENARIOS = {
    "S1": {"metrics": ("assoc", "frag"),
           "title": "S1 double gyre"},
    "S2": {"metrics": ("assoc", "ids"),
           "title": "S2 crossing (constant-velocity)"},
}
STYLE = {"hungarian": dict(color="tab:blue", marker="o", ls="-",
                           label="Hungarian (certified optimum)"),
         "greedy": dict(color="tab:red", marker="s", ls="--",
                        label="greedy (no certificate)")}


def main():
    rows = list(csv.DictReader(open(REPO / "experiments" / "results" / "sweep.csv")))
    n_seeds = len({r["seed"] for r in rows})

    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        key = (r["scenario"], float(r["sigma"]), float(r["p_detect"]),
               float(r["clutter"]), r["solver"])
        for m in ("assoc", "frag", "ids"):
            g[key][m].append(float(r[m]))

    summary = REPO / "experiments" / "results" / "sweep_summary.csv"
    with open(summary, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "sigma", "p_detect", "clutter", "solver",
                    "assoc_mean", "assoc_std", "frag_mean", "frag_std",
                    "ids_mean", "ids_std", "n"])
        for key in sorted(g, key=lambda k: (k[0], k[1], k[2], k[3], k[4])):
            sc, sigma, p, c, solver = key
            v = g[key]
            w.writerow([sc, sigma, p, c, solver,
                        f"{np.mean(v['assoc']):.4f}", f"{np.std(v['assoc']):.4f}",
                        f"{np.mean(v['frag']):.3f}", f"{np.std(v['frag']):.3f}",
                        f"{np.mean(v['ids']):.1f}", f"{np.std(v['ids']):.1f}",
                        len(v["assoc"])])
    print("wrote", summary)

    sigmas = sorted({k[1] for k in g})
    for scenario, spec in SCENARIOS.items():
        metrics = spec["metrics"]
        fig, axes = plt.subplots(len(metrics), 3,
                                 figsize=(12.5, 3.3 * len(metrics) + 0.9),
                                 sharex=True, constrained_layout=True)
        axes = np.atleast_2d(axes)
        for ci, c in enumerate(CLUTTER_COLS):
            for ri, metric in enumerate(metrics):
                ax = axes[ri, ci]
                for solver in ("hungarian", "greedy"):
                    vals = [g[(scenario, s, P_SLICE, c, solver)][metric]
                            for s in sigmas]
                    means = [np.mean(v) for v in vals]
                    stds = [np.std(v) for v in vals]
                    ax.errorbar(sigmas, means, yerr=stds, capsize=3, lw=1.8,
                                markersize=5, **STYLE[solver])
                ax.set_xscale("log")
                ax.set_xticks(sigmas)
                ax.set_xticklabels([f"{s:g}" for s in sigmas])
                ax.grid(alpha=0.3)
                if ri == len(metrics) - 1:
                    ax.set_xlabel(r"detection noise $\sigma$")
                if ci == 0:
                    ax.set_ylabel(METRIC_LABEL[metric])
                if ri == 0:
                    ax.set_title(f"clutter = {c:.0f} / frame")
        axes[0, 0].legend(fontsize=8, loc="lower left")
        fig.suptitle(f"Solver comparison, {spec['title']}  "
                     f"(p_detect = {P_SLICE}, {n_seeds} seeds, 200 frames/run)",
                     fontsize=12)
        out = REPO / "figures" / f"sweep_{scenario.lower()}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
