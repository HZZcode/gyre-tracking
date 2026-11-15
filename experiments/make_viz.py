"""Produce figures/overview.png and figures/tracking.gif.

Scenario: S1 double gyre, sigma=0.03, p_detect=0.90, clutter=1/frame.
N_FRAMES = 100 at dt = 0.1 covers exactly one forcing period (T = 10), so the
GIF shows the gyres swap left-right and back.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tracking import associate, detector_sim, flow, metrics, viz  # noqa: E402
from tracking.truth import true_centers  # noqa: E402

N_FRAMES = 100
DT = 0.1
SIGMA, P_DETECT, CLUTTER, SEED = 0.03, 0.90, 1.0, 0


def build_states():
    """Run the tracker once, recording everything the visuals need per frame."""
    gyre = flow.DoubleGyre()
    times = np.arange(N_FRAMES) * DT
    rng = np.random.default_rng(SEED)
    frames = detector_sim.simulate(gyre, times, SIGMA, P_DETECT, CLUTTER, rng)
    gate = 6.8 * SIGMA + 0.02
    tm = associate.TrackManager(gate_sq=gate**2, dt=DT)

    states, matches, gt = [], [], []
    for k, f in enumerate(frames):
        matches.append(tm.update(f.points))
        gt.append(f.source)
        snap = {tr.track_id: (tr.pos.copy(), tr.misses > 0, tr.hits < tm.min_hits)
                for tr in tm.tracks}
        states.append({
            "k": k, "K": N_FRAMES, "t": times[k],
            "points": f.points, "source": f.source,
            "truth": true_centers(gyre, [times[k]], (0.5, 1.5))[:, 0, :],
            "snapshot": snap,
        })

    # Trail histories: a track is alive on a contiguous frame range, so the
    # prefix up to frame k is just hist[tid][:k - born + 1].
    hist: dict = {}
    born: dict = {}
    for s in states:
        k = s["k"]
        for tid, (pos, *_rest) in s["snapshot"].items():
            if tid not in hist:
                hist[tid] = []
                born[tid] = k
            hist[tid].append(pos)
        s["history"] = {tid: np.asarray(hist[tid][: k - born[tid] + 1])
                        for tid in s["snapshot"]}
    return gyre, states, gt, matches


def main():
    gyre, states, gt, matches = build_states()
    acc = metrics.association_accuracy(gt, matches)
    n_ids = len({tid for s in states for tid in s["snapshot"]})
    n_conf = len({tid for s in states
                  for tid, (_p, _c, prov) in s["snapshot"].items() if not prov})
    title = (f"double gyre tracking  (sigma={SIGMA}, p_detect={P_DETECT}, "
             f"clutter={CLUTTER:.0f}/frame)   association accuracy = {acc:.3f}"
             f"   {n_ids} ids born -> {n_conf} confirmed")

    out = pathlib.Path(__file__).resolve().parents[1] / "figures"
    out.mkdir(exist_ok=True)
    viz.overview(gyre, states, out / "overview.png", title=title)
    print("wrote", out / "overview.png")
    viz.animate(gyre, states, out / "tracking.gif", trail=30, fps=10)
    print(f"wrote {out / 'tracking.gif'}  (assoc={acc:.3f}, {n_ids} track ids)")


if __name__ == "__main__":
    main()
