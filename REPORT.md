# Gyre-Tracking Experiment Report

Multi-target tracking on an analytic double-gyre flow, built around a
**certified per-frame linear assignment** core. This report states what was
built, what it measures, and — the part we care most about — the failure
modes found on the way, how each was diagnosed, and what fixed it.

Everything here is reproducible from the repository; commands are in §8.

---

## 1. Objectives

1. Build a from-scratch Hungarian assignment solver (JV shortest augmenting
   path with dual potentials) whose output carries a **verified optimality
   certificate**, and track vortex centers of the double gyre with it.
2. Measure how association quality degrades with detection noise, missed
   detections, and clutter — and whether the certified optimum measurably
   beats a greedy matcher.
3. Characterize the *structural* failure modes of per-frame association
   (fragmentation, identity switches at crossings) and repair what is
   repairable post hoc.

The application context: ground truth is **closed form** (the double gyre's
vortex centers have an exact analytic trajectory), so every tracking error
is measurable against exact truth — no annotation noise, no ambiguity.

## 2. System

```
analytic double-gyre flow  ->  closed-form vortex-center trajectories (truth)
        |  Gaussian noise (sigma) / misses (p_detect) / Poisson clutter
        v
per-frame detections, no identity
        |  constant-velocity prediction, gate r = 6.8*sigma + 0.02
        |  padded cost matrix, S = n_tracks + m  (dummy rows & columns)
        v
Hungarian method (from scratch, O(n^3), dual potentials u, v)
        |  certificate check: dual feasibility + complementary slackness
        |  + strong duality, on EVERY solve (tests + runtime)
        v
track lifecycle: two-pass confirmation, coast, birth gating
        |
        +-- fragment stitching (the same certified solver, one level up)
        v
metrics vs exact truth: association accuracy / fragments per object / ID switches
```

Key design decisions:

- **Gate radius `6.8 * sigma + 0.02`.** With finite-difference velocities, the
  one-step prediction error has variance ≈ 5σ² per coordinate, so the squared
  gate distance follows 5σ²·χ²₂. The 99% quantile of χ²₂ is 9.21, giving
  r² = 9.21·5σ², i.e. r ≈ 6.8σ; the additive floor covers the dt-discretization
  error at turning points.
- **Velocity cap `max_step = 0.05`/frame** — a poor man's process-noise bound
  that keeps a coasting track's prediction from flying off at a turning point
  (case study A).
- **Padding to S = n + m.** Pad with dummy columns (one per track) and dummy
  rows (one per detection) filled with the gate cost. Then
  `total = gate_sq·(n+m) + Σ(d² − gate_sq)`, so each real pair decides
  *independently*: claim ⟺ d² < gate_sq. Padding only to max(n, m) — the
  usual textbook shortcut — leaves no dummy at m == n_tracks and forces
  detections onto tracks across the whole domain (case study C).
- **Two-pass lifecycle.** Confirmed tracks (hits ≥ 3) compete in pass 1;
  provisional tracks may only claim pass-2 leftovers and die on their first
  miss. A confirmed track coasts up to 5 consecutive misses before death.
- **Birth gating.** An unclaimed detection *inside some confirmed track's
  gate* is treated as clutter or a duplicate, not as a new object — it does
  not spawn a track. This one rule removed most clutter-driven identity
  churn (case study B).
- **Certified solver as an interface.** `TrackManager(solver=..., verify=...)`
  accepts any callable returning `(sigma, u, v)`; the greedy baseline plugs in
  as a three-line wrapper with fake duals. The comparison in §4 is therefore
  a one-line swap, and the certificate check (`verify=False` for greedy)
  documents exactly what greedy does not have.

## 3. Scenarios and conditions

- **S1 `double-gyre`** — detections around the two analytic vortex centers;
  the centers sweep horizontally with period T = 10 and never approach within
  ~0.54, so per-frame association is well-posed except for clutter.
- **S2 `crossing`** — two constant-velocity objects on mirror-image lines,
  crossing at (1.0, 0.5) mid-run: the pathological geometry for sequential
  matching.

Conditions: σ ∈ {0.01, 0.03, 0.06, 0.10}, p_detect ∈ {0.95, 0.85, 0.70},
clutter ∈ {0, 1, 3} per frame, 20 seeds per cell, 200 frames per run,
solvers {hungarian, greedy} — **2880 runs** per full sweep (~3 min).

## 4. Results

### 4.1 S1 performance envelope

Hungarian, p_detect = 0.85, 20 seeds (full grid in
`experiments/results/sweep_summary.csv`; figure `figures/sweep_s1.png`):

| condition | assoc | frag / object | ids / run |
|---|---|---|---|
| clutter = 0, any σ | **0.986** | 1.65 | 1.3 |
| clutter = 1, σ = 0.03 | 0.762 | 4.08 | 12.8 |
| clutter = 3, σ = 0.03 | 0.567 | 7.50 | 33.8 |
| clutter = 3, σ = 0.10 | 0.332 | 8.4 | 146.7 |

Three observations that shape everything below:

1. **At clutter = 0, noise barely matters.** Association is 0.986 *flat across
   σ from 0.01 to 0.10*. With two well-separated objects and a correctly
   scaled gate, per-frame assignment essentially never errs; the residual
   fragmentation (1.65) is σ-independent and comes entirely from
   **miss-driven birth churn** — a newborn (unconfirmed) track that meets one
   miss dies and its object restarts under a new id. This is exactly what
   stitching (§4.4) repairs.
2. **Clutter, not noise, is the enemy.** Going 0 → 3 clutter/frame costs four
   times more association accuracy than going σ 0.01 → 0.10. In-gate clutter
   competes for claims, occasionally wins, drags a track off its object for a
   few frames, and the churn cascades through the confirmation machinery.
3. **The σ = 0.10, clutter = 3 corner is the architecture's ceiling.** The
   gate grows as 6.8σ + 0.02 ≈ 0.70, i.e. a quarter of the domain area is
   in-gate for every track, and 3 clutter/frame land inside it constantly.
   Per-frame LAP with isotropic gating cannot be saved by tuning here —
   the gate-width ablation in §5.5 shows why.

### 4.2 Hungarian vs greedy

`figures/sweep_s1.png`, `figures/sweep_s2.png`. On S1, greedy tracks the
certified optimum closely while objects are well separated — mean association
differences of +0.02–0.04 for Hungarian at clutter ≥ 1, consistent across
seeds (e.g. σ = 0.1, p = 0.7, clutter = 3: **0.340 vs 0.292**). Greedy's
sequential commitment costs little when the cost matrix is nearly diagonal.

S2 is where the difference becomes structural. ID switches per run at
clutter = 0 (Hungarian vs greedy):

| σ | 0.01 | 0.03 | 0.06 | 0.10 |
|---|---|---|---|---|
| Hungarian | 6.0 | **10.4** | **16.5** | **24.8** |
| greedy | 7.4 | 14.9 | 21.8 | 31.0 |

Greedy swaps identities **20–35% more often** at every σ; with clutter added,
Hungarian's association accuracy stays ahead at all σ (clutter = 1:
0.77/0.64/0.53/0.45 vs 0.71/0.57/0.52/0.42). Around the crossing the four
points (two predictions + two detections) cluster within a few σ; a globally
consistent assignment preserves identity through the ambiguity window better
than locally cheapest choices. Note also that **even the optimum switches**
(6–25 times per run): per-frame association has a geometric limit at
crossings, which is what multi-frame methods address (§6).

### 4.3 What the certificate buys — honestly

The certificate is not magic: in easy cells greedy ≈ optimal, and the numbers
above are modest. What the certificate *does* buy:

- **Per-frame guarantees.** Every accepted matching is provably
  minimum-cost (dual feasibility, complementary slackness, strong duality —
  checked in tests on random matrices, brute-force comparisons, and at
  runtime for every solve). Debugging can therefore assume: if the tracking
  misbehaves, the *inputs* (gating, prediction, bookkeeping) are wrong, not
  the assignment. Both critical bugs in §5 were found this way.
- **A fair, minimal baseline.** The greedy wrapper returns fake duals
  (u = v = 0) and runs with `verify=False`; the asymmetric treatment is the
  honest statement of the difference: greedy has nothing to verify.
- **Failure attribution.** S2's residual switches are *not* solver
  suboptimality — the certificate rules that out per frame — so they must be
  geometric, which §4.2 confirms and §6 discusses.

### 4.4 Fragment stitching

`stitch.stitch_plan` links a dead track to a later-born one when the newborn's
first position lies in the gate of the dead track's constant-velocity
extrapolation across the gap (≤ 3 frames). Competing candidates are resolved
by **the same certified solver** — one more assignment problem, this time
between track endpoints — then merged with union-find.

Before → after (20 seeds, p_detect = 0.85):

| condition | assoc | frag / object | ids / run |
|---|---|---|---|
| σ = 0.03, clutter = 0 | 0.986 → **1.000** | 1.65 → **1.02** | 1.3 → **0.1** |
| σ = 0.10, clutter = 0 | 0.986 → 0.999 | 1.65 → 1.07 | 1.3 → 0.1 |
| σ = 0.03, clutter = 1 | 0.762 → 0.783 | 4.08 → 3.33 | 12.8 → 11.2 |
| σ = 0.03, clutter = 3 | 0.567 → **0.654** | 7.50 → 5.03 | 33.8 → 28.8 |

Birth-churn fragments — gap 1, continuation exactly on the extrapolation —
are repaired *completely*: frag → 1.02 (the theoretical floor is 1.0),
association → 1.000. High-clutter chaos is repaired only partially (+0.087
association at clutter = 3): there, tracks die after long clutter-occupation
battles and their successors are born several frames later, outside
`max_gap`, so the linking hypothesis never fires. Stitching repairs
**fragments**; it cannot repair **confusions** — after a crossing swap both
tracks are alive and continuous, with no dead/young pair to link.

Figure: `figures/stitch_before_after.png`.

## 5. Failure-mode case studies

The debugging log, ordered by when each bit us. Each case: symptom →
diagnosis → fix.

### A. Velocity blowup at turning points

*Symptom:* tracks overshot wildly whenever a vortex center decelerated to
reverse. *Diagnosis:* the finite-difference velocity, frozen during coasting,
predicted straight through the turn. *Fix:* cap the per-frame prediction
displacement (`max_step = 0.05`) — a crude process-noise bound that bounds
how far a prediction may run from the last claim.

### B. Junk competing for confirmation

*Symptom:* in clutter, two detections that happened to persist for a few
frames confirmed into a "track" and then out-competed the *real* object's
young track for detections. *Fix:* two passes (confirmed tracks claim first;
provisional tracks see only leftovers) plus **birth gating** — an unclaimed
detection inside a confirmed track's gate never spawns a track. The rule
encodes an assumption: inside-gate detections are clutter or duplicates;
new-object evidence must appear *outside* every confirmed gate.

### C. Forced matching at equal counts

*Symptom:* straight tracks jumping across the domain in the overview figure.
*Diagnosis:* padding the cost matrix to max(n, m) leaves **no dummy** when
m == n_tracks; every detection must claim a track and vice versa, so a
detection 1.5 domain-units away was "matched" to the only free track.
*Fix:* pad to S = n + m; the dummy-fill decomposition (§2) makes every pair's
threshold decision independent. Lesson: *assignment feasibility is part of
the model* — "matching" a detection to a far track is worse than leaving both
unmatched, and the cost matrix must be able to say so.

### D. The lost frame bookkeeping

*Symptom:* association accuracy ≈ 0.97 that did not respond to *any*
parameter change. *Diagnosis:* the two-pass refactor moved claim recording
into `_match_pass` but dropped the write `frame_match[j] = track_id` — every
confirmed claim was recorded as −1, so metrics were scoring birth patterns,
not identity. Caught by a new unit test (`fm == [-1, -1]` while hit counts
grew correctly). *Fix:* thread `frame_match` through both passes. Lesson:
**an end-to-end number that looks plausible is not evidence** — this bug made
every metric since the refactor meaningless while tests of internals stayed
green.

### E. The subset-index bookkeeping (caught by the sweep)

*Symptom:* the 20-seed sweep reported fragmentation ≈ 4.5 (± 5.6) at
clutter = 0, σ = 0.01 — physically impossible; misses alone cannot fragment a
clean scene. *Diagnosis:* a frame-by-frame autopsy of the worst seed showed
birth/death churn every frame for ~15 frames. Pass 2 received
`detections[free]` (a subset) but wrote `claimed[j]` and `frame_match[j]` at
*subset-relative* indices — the track claimed the right detection while the
bookkeeping marked the wrong one, so the same detection spawned a new track
every frame. *Fix:* `_match_pass` now receives the original index vector
explicitly; a regression test (`test_two_pass_bookkeeping_uses_original_indices`)
pins the contract with a mixed confirmed/provisional frame. Lesson, stated
bluntly: **the sweep produced self-consistent garbage twice**; only a
hand-built scenario test plus a per-frame autopsy could see it. Large-scale
experiments validate distributions; unit tests validate contracts — and a
wrong contract poisons every distribution built on it.

### F. The gate-width hypothesis, refuted

*Hypothesis:* clutter steals happen because gates are too loose; tightening
the gate multiplier should recover accuracy. *Ablation:* gate multiplier 3 → 9
at σ = 0.06, clutter = 3 moved association only between 0.25 and 0.32 —
*wider was, if anything, slightly better* (recovery capture outweighs steal
cost). *Conclusion:* the collapse is architectural (per-frame LAP + static
gating), not a tuning error. The effective fixes were birth gating (§B) and,
later, stitching — and the honest negative result redirected effort away from
tuning. Lesson: **ablate before attributing.**

## 6. Known limitations and extensions

1. **Crossing identity switches are structural.** Certified per-frame
   optimality still swaps at S2 crossings, more as σ (hence the gate, hence
   the ambiguity window) grows. The named fix is **multi-frame association**
   (min-cost-flow over a sliding window), which buys identity continuity at
   the price of latency and the loss of the per-frame certificate's simple
   story.
2. **Constant-velocity + isotropic gate.** A Kalman filter with an adaptive
   (Mahalanobis, χ²-gated) ellipsoid would shrink gates in well-observed
   directions and is the standard next step; `max_step` is its poor cousin.
3. **Static confirmation parameters.** `min_hits = 3`, `max_coast = 5`,
   `max_gap = 3` are fixed; a clutter-adaptive confirmation (M-of-N logic)
   would trade latency for robustness at high clutter.
4. **Stitching's `max_gap` is a hard bound** — exactly why high-clutter
   repair is partial (§4.4).
5. **Two objects only.** The S2 grid uses 2 targets; the solver is O(n³) and
   the padding generalizes, but clutter density per track is what drives the
   envelope, and denser scenes will cross the architecture's ceiling sooner.

## 7. Methodology notes

- **Tests as executable specs.** The solver's acceptance criteria
  (brute-force equivalence on random matrices, dual certificates, invariance
  to permutations and cost scaling) were written before the implementation;
  the same discipline caught both integration bugs above. Final suite:
  **69 tests, all green.**
- **Autopsy over intuition.** Every behavioral anomaly was traced to a
  frame-by-frame event log of the worst seed before any fix was attempted.
- **Honest metrics.** Association accuracy uses majority-vote identity
  resolution; fragmentation counts distinct ids per object; ID switches count
  adjacent-frame changes — all computed against closed-form truth, on real
  detections only.

## 8. Reproducing

```bash
pip install -e ".[dev]"
pytest                                        # 69 tests

python experiments/run_s1.py                  # 4-row baseline table
python experiments/sweep.py                   # 2880 runs -> results/sweep.csv (~3 min)
python experiments/plot_sweep.py              # figures/sweep_s1.png, sweep_s2.png
python experiments/stitch_eval.py             # before/after stitching + figure
python experiments/make_viz.py                # figures/overview.png, tracking.gif
```

Aggregated results are committed (`sweep_summary.csv`, `stitch_eval.csv`);
raw per-run CSVs are regenerable and gitignored.
