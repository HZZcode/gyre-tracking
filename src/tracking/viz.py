"""Static and animated visuals for the gyre-tracking pipeline.

Two artifacts (see experiments/make_viz.py for the harness):
- overview(...): one static figure, sensor view vs tracker view;
- animate(...):  per-frame GIF — flow field + truth + detections + trails.

Visual grammar: tracks are tab10(tid % 10); true centers are gold stars;
real detections are dark dots; clutter is light gray crosses. A coasting
track is drawn hollow, a provisional (unconfirmed) track dashed.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402


def psi_field(gyre, t: float, nx: int = 181, ny: int = 91):
    """Meshgrid (X, Y) and streamfunction PSI over the domain at time t."""
    x = np.linspace(0.0, 2.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)
    return X, Y, gyre.streamfunction(X, Y, t)


def track_color(tid: int):
    """Consistent color per track id (cycles through tab10)."""
    return plt.get_cmap("tab10")(tid % 10)


def draw_field(ax, gyre, t: float, alpha: float = 1.0):
    X, Y, PSI = psi_field(gyre, t)
    return ax.contourf(X, Y, PSI, levels=np.linspace(-0.1, 0.1, 21),
                       cmap="RdBu_r", alpha=alpha, extend="both")


def draw_detections(ax, points, source) -> None:
    points = np.asarray(points)
    source = np.asarray(source)
    clutter = source == -1
    if clutter.any():
        ax.scatter(points[clutter, 0], points[clutter, 1], s=16, c="0.72",
                   marker="x", linewidths=0.9, zorder=3)
    real = ~clutter
    if real.any():
        ax.scatter(points[real, 0], points[real, 1], s=13, c="0.10", zorder=4)


def draw_truth(ax, centers) -> None:
    centers = np.asarray(centers)
    ax.scatter(centers[:, 0], centers[:, 1], marker="*", s=230, c="gold",
               edgecolors="black", linewidths=0.8, zorder=7)


def draw_tracks(ax, snapshot: dict, history: dict, trail: int = 30) -> None:
    """snapshot: {tid: (pos, coasting, provisional)}; history: {tid: (T, 2)}."""
    for tid, (pos, coasting, provisional) in snapshot.items():
        pts = history[tid][-trail:]
        c = track_color(tid)
        ax.plot(pts[:, 0], pts[:, 1], color=c, lw=2.0,
                ls="--" if provisional else "-", alpha=0.9, zorder=5)
        ax.scatter([pos[0]], [pos[1]], s=48,
                   facecolor="none" if coasting else c,
                   edgecolor=c, linewidths=1.8, zorder=6)


def _legend_handles() -> list:
    return [
        Line2D([], [], marker="*", ls="", color="gold", markeredgecolor="black",
               markersize=13, label="true vortex centers"),
        Line2D([], [], marker="o", ls="", color="0.10", markersize=5,
               label="real detection"),
        Line2D([], [], marker="x", ls="", color="0.55", markersize=6,
               label="clutter"),
        Line2D([], [], color="tab:blue", lw=2,
               label="track (dashed = provisional, hollow = coasting)"),
    ]


def draw_frame(ax, gyre, s: dict, trail: int = 30) -> None:
    """Draw one animation frame (state dict) onto ax; clears first."""
    ax.clear()
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks(())
    ax.set_yticks(())
    draw_field(ax, gyre, s["t"])
    draw_detections(ax, s["points"], s["source"])
    draw_tracks(ax, s["snapshot"], s["history"], trail)
    draw_truth(ax, s["truth"])
    ax.set_title(f"double gyre   t = {s['t']:5.2f}   frame {s['k'] + 1}/{s['K']}"
                 f"   live tracks = {len(s['snapshot'])}", fontsize=10)
    ax.legend(handles=_legend_handles(), loc="upper left", fontsize=7,
              framealpha=0.85)


def animate(gyre, states: list, path, trail: int = 30, fps: float = 10,
            dpi: int = 110) -> None:
    """Render the state list to an animated GIF at `path` (PillowWriter)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.1))

    def _update(i: int):
        draw_frame(ax, gyre, states[i], trail)
        return []

    anim = FuncAnimation(fig, _update, frames=len(states), blit=False)
    anim.save(str(path), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def overview(gyre, states: list, path, title: str = "") -> None:
    """Static two-panel figure for README/REPORT: sensor view vs tracker view.

    The tracker panel shows only tracks that were CONFIRMED at some point
    (hits >= min_hits in some frame): short-lived provisional clutter tracks
    would otherwise render as straight lines crossing the whole domain.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), sharex=True,
                             sharey=True, constrained_layout=True)
    truth = np.stack([s["truth"] for s in states])            # (K, n_fc, 2)
    allpts = np.concatenate([np.asarray(s["points"]) for s in states])
    allsrc = np.concatenate([np.asarray(s["source"]) for s in states])
    confirmed = {tid for s in states
                 for tid, (_p, _c, prov) in s["snapshot"].items() if not prov}

    ax = axes[0]
    cf = draw_field(ax, gyre, 0.0)
    for j in range(truth.shape[1]):
        ax.plot(truth[:, j, 0], truth[:, j, 1], "--", color="black",
                lw=1.4, alpha=0.55, zorder=5)
    draw_detections(ax, allpts, allsrc)
    ax.set_title("sensor view: detections + true center paths")
    ax.legend(handles=_legend_handles(), loc="upper left", fontsize=7,
              framealpha=0.85)

    ax = axes[1]
    draw_field(ax, gyre, 0.0, alpha=0.45)
    for j in range(truth.shape[1]):
        ax.plot(truth[:, j, 0], truth[:, j, 1], "--", color="black",
                lw=1.4, alpha=0.55, zorder=5)
    full: dict = {}
    for s in states:
        for tid, (pos, *_rest) in s["snapshot"].items():
            if tid in confirmed:
                full.setdefault(tid, []).append(np.asarray(pos))
    for tid, pts in full.items():
        pts = np.asarray(pts)
        ax.plot(pts[:, 0], pts[:, 1], color=track_color(tid), lw=1.7,
                alpha=0.9, zorder=6)
    n_conf = len(confirmed)
    ax.set_title(f"tracker view: confirmed trajectories ({n_conf} ids)")

    for a in axes:
        a.set_xlim(0, 2)
        a.set_ylim(0, 1)
        a.set_aspect("equal")
        a.set_xticks(())
        a.set_yticks(())
    fig.colorbar(cf, ax=axes[0], shrink=0.85, label=r"streamfunction $\psi$")
    fig.suptitle(title, fontsize=11)
    fig.savefig(str(path), dpi=160)
    plt.close(fig)
