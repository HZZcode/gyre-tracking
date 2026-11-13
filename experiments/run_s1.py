"""First end-to-end run: S1 double gyre, Hungarian solver, metrics vs truth.

Gate heuristic: the gate must cover the ONE-STEP PREDICTION error, whose
per-coordinate variance is ~5*sigma^2 (position noise compounds through the
finite-difference velocity: 2*eps_k - eps_{k-1}). The 99% chi-square(2)
quantile is 9.21, so gate ~ sqrt(5*9.21)*sigma ~ 6.8*sigma, plus a floor for
per-frame center motion.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tracking import associate, detector_sim, flow, metrics  # noqa: E402


def run(sigma: float, p_detect: float, clutter: float, seed: int = 0,
        n_frames: int = 200, dt: float = 0.1) -> dict:
    gate = 6.8 * sigma + 0.02
    gyre = flow.DoubleGyre()
    times = np.arange(n_frames) * dt
    rng = np.random.default_rng(seed)
    frames = detector_sim.simulate(gyre, times, sigma, p_detect, clutter, rng)

    tm = associate.TrackManager(gate_sq=gate**2, dt=dt)
    gt, matches = [], []
    for f in frames:
        matches.append(tm.update(f.points))
        gt.append(f.source)

    frag = metrics.fragmentation_counts(gt, matches)
    return {
        "sigma": sigma, "p": p_detect, "clutter": clutter, "gate": gate,
        "assoc": metrics.association_accuracy(gt, matches),
        "frag": (sum(frag.values()) / len(frag)) if frag else float("nan"),
        "ids": metrics.id_switches(gt, matches),
        "finished": len(tm.finished),
    }


if __name__ == "__main__":
    hdr = f"{'sigma':>6} {'p_det':>6} {'clut':>5} {'gate':>6} {'assoc':>7} {'frag':>6} {'ids':>4} {'fin':>4}"
    print(hdr)
    print("-" * len(hdr))
    for sigma, p, c in [(0.01, 0.95, 0), (0.03, 0.90, 1), (0.06, 0.85, 3), (0.10, 0.70, 3)]:
        r = run(sigma, p, c)
        print(f"{r['sigma']:>6.2f} {r['p']:>6.2f} {r['clutter']:>5.0f} {r['gate']:>6.2f} "
              f"{r['assoc']:>7.3f} {r['frag']:>6.2f} {r['ids']:>4d} {r['finished']:>4d}")
