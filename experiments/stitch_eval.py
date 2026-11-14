"""Before/after stitching: does post-hoc linking repair the fragments?

For a few S1 conditions (p_detect = 0.85, 20 seeds, 200 frames), runs the
tracker, builds per-tid TrackSpans from the live-track stream, applies
stitch.stitch_plan, relabels the matches, and reports metrics before vs
after. Also renders the before/after trail figure for one representative
run (sigma=0.03, clutter=1, seed 0).

Writes:
- experiments/results/stitch_eval.csv (one row per seed x condition)
- figures/stitch_before_after.png
"""

from __future__ import annotations

import csv
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tracking import associate, detector_sim, flow, metrics, scenarios  # noqa: E402
from tracking import stitch, viz  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
N_FRAMES = 200
DT = 0.1
P_DETECT = 0.85
SEEDS = range(20)
CONDITIONS = [(0.03, 0.0), (0.03, 1.0), (0.03, 3.0), (0.10, 0.0)]
FIG_CONDITION = (0.03, 0.0)
FIG_SEED = 0


def run_one(sigma, clutter, seed, collect=False):
    """Run the tracker once; optionally collect per-tid trajectory data.

    Returns (gt, matches, spans, full, confirmed) where spans maps tid to
    stitch.TrackSpan, full maps tid to its claimed positions in frame order
    (one per ALIVE frame), and confirmed is the set of tids that reached
    min_hits at some point.
    """
    gate = 6.8 * sigma + 0.02
    gyre = flow.DoubleGyre()
    times = np.arange(N_FRAMES) * DT
    truth = scenarios.s1_gyre(gyre, times)
    frames = detector_sim.simulate_tracks(
        truth, sigma, P_DETECT, clutter, np.random.default_rng(seed))
    tm = associate.TrackManager(gate_sq=gate**2, dt=DT)
    gt, matches = [], []
    raw: dict[int, dict] = {}
    full: dict[int, list] = {}
    confirmed: set[int] = set()
    for k, f in enumerate(frames):
        fm = tm.update(f.points)
        gt.append(f.source)
        matches.append(fm)
        if not collect:
            continue
        for tr in tm.tracks:
            pos = tr.positions[-1]           # last claim (frozen while coasting)
            tid = tr.track_id
            sp = raw.get(tid)
            if sp is None:
                raw[tid] = {"f0": k, "f1": k, "p0": pos.copy(), "p1": pos.copy(),
                            "prev_k": k, "prev_p": pos.copy(),
                            "vel": np.zeros(2)}
            else:
                sp["vel"] = (pos - sp["prev_p"]) / (DT * (k - sp["prev_k"]))
                sp["f1"] = k
                sp["p1"] = pos.copy()
                sp["prev_k"] = k
                sp["prev_p"] = pos.copy()
            full.setdefault(tid, []).append(pos.copy())
            if tr.hits >= tm.min_hits:
                confirmed.add(tid)
    if not collect:
        return gt, matches, None, None, None
    spans = {tid: stitch.TrackSpan(sp["f0"], sp["f1"], sp["p0"], sp["p1"], sp["vel"])
             for tid, sp in raw.items()}
    return gt, matches, spans, full, confirmed


def relabel(matches: list, id_map: dict[int, int]) -> list:
    return [np.array([id_map.get(int(t), int(t)) if t >= 0 else -1 for t in fm],
                     dtype=np.int64)
            for fm in matches]


def score(gt, matches):
    frag = metrics.fragmentation_counts(gt, matches)
    return (metrics.association_accuracy(gt, matches),
            sum(frag.values()) / len(frag) if frag else float("nan"),
            metrics.id_switches(gt, matches))


def main():
    out = REPO / "experiments" / "results"
    out.mkdir(exist_ok=True)
    csv_path = out / "stitch_eval.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sigma", "clutter", "seed",
                    "assoc_before", "assoc_after",
                    "frag_before", "frag_after",
                    "ids_before", "ids_after"])
        for sigma, clutter in CONDITIONS:
            acc_b, acc_a, fr_b, fr_a, id_b, id_a = [], [], [], [], [], []
            for seed in SEEDS:
                gt, matches, spans, _full, _conf = run_one(sigma, clutter, seed,
                                                           collect=True)
                id_map = stitch.stitch_plan(spans, gate_sq=(6.8 * sigma + 0.02) ** 2,
                                            dt=DT)
                ab, fb, ib = score(gt, matches)
                aa, fa, ia = score(gt, relabel(matches, id_map))
                w.writerow([sigma, clutter, seed, f"{ab:.4f}", f"{aa:.4f}",
                            f"{fb:.3f}", f"{fa:.3f}", ib, ia])
                acc_b.append(ab); acc_a.append(aa)
                fr_b.append(fb); fr_a.append(fa)
                id_b.append(ib); id_a.append(ia)
            print(f"sigma={sigma:.2f} clutter={clutter:.0f}  "
                  f"assoc {np.mean(acc_b):.3f} -> {np.mean(acc_a):.3f}   "
                  f"frag {np.mean(fr_b):5.2f} -> {np.mean(fr_a):5.2f}   "
                  f"ids {np.mean(id_b):6.1f} -> {np.mean(id_a):6.1f}", flush=True)
    print("wrote", csv_path)

    # ---- representative before/after figure ------------------------------
    sigma, clutter = FIG_CONDITION
    gt, matches, spans, full, confirmed = run_one(sigma, clutter, FIG_SEED,
                                                  collect=True)
    gate = 6.8 * sigma + 0.02
    id_map = stitch.stitch_plan(spans, gate_sq=gate**2, dt=DT)
    fb, fa = score(gt, matches)[1], score(gt, relabel(matches, id_map))[1]

    times = np.arange(N_FRAMES) * DT
    truth = scenarios.s1_gyre(flow.DoubleGyre(), times)   # (2, K, 2)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), sharex=True,
                             sharey=True, constrained_layout=True)
    for ax, use_map in zip(axes, (False, True)):
        viz.draw_field(ax, flow.DoubleGyre(), 0.0, alpha=0.45)
        for j in range(truth.shape[1]):
            ax.plot(truth[:, j, 0], truth[:, j, 1], "--", color="black",
                    lw=1.4, alpha=0.55, zorder=5)
        if not use_map:
            for tid in sorted(confirmed):
                pts = np.asarray(full[tid])
                ax.plot(pts[:, 0], pts[:, 1], color=viz.track_color(tid),
                        lw=1.7, alpha=0.9, zorder=6)
            ax.set_title(f"before stitching: {len(confirmed)} confirmed ids, "
                         f"frag = {fb:.1f}")
        else:
            chains: dict[int, list] = {}
            for tid in confirmed:
                chains.setdefault(id_map[tid], []).append(
                    (spans[tid].first_frame, full[tid]))
            n_chains = 0
            for root, members in sorted(chains.items()):
                members.sort(key=lambda m: m[0])
                pts = np.concatenate([np.asarray(p) for _, p in members])
                if len(pts) < 3:      # min_hits: skip sub-confirmation stubs
                    continue
                n_chains += 1
                ax.plot(pts[:, 0], pts[:, 1], color=viz.track_color(root),
                        lw=1.7, alpha=0.9, zorder=6)
            ax.set_title(f"after stitching: {n_chains} chains, frag = {fa:.1f}")
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks(())
        ax.set_yticks(())
    fig.suptitle(f"Fragment stitching, S1 double gyre  "
                 f"(sigma={sigma}, clutter={clutter:.0f}/frame, "
                 f"p_detect={P_DETECT}, seed {FIG_SEED})", fontsize=11)
    out_fig = REPO / "figures" / "stitch_before_after.png"
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)
    print("wrote", out_fig)


if __name__ == "__main__":
    main()
