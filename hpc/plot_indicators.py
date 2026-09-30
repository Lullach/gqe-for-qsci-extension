"""
Where would the subspace-size indicators have stopped the runs?

Redraws the error-vs-epoch figure from plot_results.py (same curves: mean over
seeds of the best-so-far error) and marks, per family, the epoch at which a
live rule would have said "the policy has saturated BECAUSE OF the subspace":

    plateau   the best-so-far error improved by less than --delta mHa over the
              last --window epochs (read from the logged curve), AND
    hot       the indicator, computed on the best-so-far circuit's QSCI
              subspace, says the subspace is still limiting.

Indicator values come from hpc/replay_indicators.py. "Hot" thresholds are taken
from the exact-vector calibration (hpc/calibrate_indicators.py), where error
<= 0.2 mHa gave tail weight <= 1e-7 and boundary ~0, and error 0.2-2 mHa gave
tail weight >= 4e-5 and boundary >= 0.11:

    tail weight    > 1e-5
    boundary       > 0.1 mHa
    ENPT2          |E_PT2| > 1.6 mHa  (more than chemical accuracy lies just
                                       outside the subspace)

The marker sits at the MEDIAN over seeds of each seed's stop epoch, on the mean
curve. Seeds on which a rule never fires are counted in the legend rather than
silently dropped.

Only the GQE-optimized panel can be marked. The Global-refined subspace is
accumulated over every circuit of every epoch and has no sequence of its own,
so it cannot be replayed without re-running training.

Plain Python with matplotlib (not the container):

    python hpc/plot_indicators.py
"""

import argparse
import csv
import glob
import os
import statistics as stats
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_results import (CHEMICAL_ACCURACY_MHA, STAGES, _series,  # noqa: E402
                          family, load_rows)

GQE_STAGE = "GQE-optimized(best_so_far)"

RULES = [
    # name, csv column, hot(value), marker, legend label
    ("tail", "tail_weight", lambda v: v > 1e-5, "x",
     "tail weight > 1e-5"),
    ("boundary", "boundary_mha", lambda v: v > 0.1, "+",
     "boundary energy > 0.1 mHa"),
    ("pt2", "pt2_mha", lambda v: abs(v) > CHEMICAL_ACCURACY_MHA, "v",
     "ENPT2 > 1.6 mHa"),
]


def load_indicators(pattern):
    """{exp_tag: [(epoch, {column: value}), ...]} sorted by epoch."""
    out = defaultdict(list)
    paths = sorted(glob.glob(pattern))
    if not paths:
        sys.exit(f"no indicator CSVs match {pattern!r}; run "
                 "hpc/replay_indicators.py first.")
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                vals = {c: float(r[c]) for _n, c, *_ in RULES if r.get(c)}
                out[r["exp_tag"]].append((int(r["epoch"]), vals))
    for tag in out:
        out[tag].sort(key=lambda x: x[0])
    print(f"indicator replays: {len(out)} run(s) from {len(paths)} file(s)")
    return out


def per_run_curve(rows, stage):
    """{exp_tag: {epoch: err_mha}} for one stage."""
    out = defaultdict(dict)
    for r in rows:
        if r["stage"] == stage and r["err_mha"] is not None:
            out[r["exp_tag"]][r["epoch"]] = r["err_mha"]
    return out


def stop_epoch(curve, replays, column, hot, window, delta):
    """
    First epoch at which the run has plateaued AND the indicator on the current
    best-so-far circuit is hot. None if that never happens.

    The indicator is a step function: it only changes when the best-so-far
    circuit does, which is exactly where the replay evaluated it.
    """
    epochs = sorted(curve)
    j, current = 0, None
    for i, ep in enumerate(epochs):
        while j < len(replays) and replays[j][0] <= ep:
            current = replays[j][1].get(column)
            j += 1
        if ep < window or current is None:
            continue
        past = curve.get(ep - window)
        if past is None:
            continue
        plateau = (past - curve[ep]) < delta
        if plateau and hot(current):
            return ep
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results",
                    default="outputs/gqe-for-qsci/ArchitectureComparison/results.csv")
    ap.add_argument("--indicators",
                    default="data/indicators/*.csv")
    ap.add_argument("--out",
                    default="outputs/gqe-for-qsci/ArchitectureComparison/figures/indicators")
    ap.add_argument("--window", type=int, default=25,
                    help="plateau window in epochs")
    ap.add_argument("--delta", type=float, default=1.0,
                    help="an improvement below this over the window (mHa) "
                         "counts as a plateau")
    args = ap.parse_args()

    replays = load_indicators(args.indicators)
    # Only the replayed families. The merged results.csv also carries runs on
    # other molecules (N2) and smoke runs; mixing them onto one error axis would
    # compare errors against different CASCI references.
    replayed = {family({"exp_tag": t}) for t in replays}
    rows = [r for r in load_rows(args.results)
            if r["stage"] in STAGES and family(r) in replayed]
    curves = per_run_curve(rows, GQE_STAGE)

    # stop epochs per family and rule
    stops = defaultdict(lambda: defaultdict(list))
    never = defaultdict(lambda: defaultdict(int))
    for tag, curve in curves.items():
        if tag not in replays:
            continue
        fam = family({"exp_tag": tag})
        for name, column, hot, *_ in RULES:
            ep = stop_epoch(curve, replays[tag], column, hot,
                            args.window, args.delta)
            if ep is None:
                never[fam][name] += 1
            else:
                stops[fam][name].append(ep)

    print(f"\nstop epoch (plateau of < {args.delta} mHa over {args.window} "
          f"epochs AND indicator hot), median over seeds:")
    print(f"{'family':<22}" + "".join(f"{n:>16}" for n, *_ in RULES))
    fams = sorted({family({"exp_tag": t}) for t in replays})
    for fam in fams:
        cells = []
        for name, *_ in RULES:
            eps, miss = stops[fam][name], never[fam][name]
            cell = f"{int(stats.median(eps))}" if eps else "never"
            if miss and eps:
                cell += f" ({miss} never)"
            cells.append(f"{cell:>16}")
        print(f"{fam:<22}" + "".join(cells))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    colors = {}
    for ax, stage in zip(axes, STAGES):
        for fam, pts in sorted(_series(rows, stage, "err_mha").items()):
            line, = ax.plot(list(pts), list(pts.values()), lw=1.5, label=fam)
            colors[fam] = line.get_color()
            if stage != GQE_STAGE:
                continue
            for name, _c, _h, marker, _lab in RULES:
                eps = stops[fam][name]
                if not eps:
                    continue
                ep = int(stats.median(eps))
                nearest = min(pts, key=lambda e: abs(e - ep))
                ax.plot(ep, pts[nearest], marker=marker, ms=10, mew=2,
                        color=colors[fam], ls="none", zorder=5)
        ax.set(xlabel="epoch", title=stage, yscale="log")
        ax.axhline(CHEMICAL_ACCURACY_MHA, ls="--", c="k", lw=0.8)
        ax.grid(alpha=0.25, lw=0.5)

    axes[1].text(0.5, 0.2, "no markers: not replayable. The refined subspace\n"
                 "is built from every circuit of every epoch.",
                 transform=axes[1].transAxes, ha="center", va="center",
                 fontsize=8, color="0.35")

    # legend 1: families. legend 2: what each marker means and how often it fired
    axes[1].legend(fontsize=7, loc="upper right")
    handles = []
    for name, _c, _h, marker, label in RULES:
        n_fired = sum(len(stops[f][name]) for f in fams)
        n_total = n_fired + sum(never[f][name] for f in fams)
        handles.append(Line2D([], [], marker=marker, ls="none", color="k",
                              ms=8, mew=2,
                              label=f"{label}: fired in {n_fired}/{n_total} runs"))
    axes[0].legend(handles=handles, fontsize=7, loc="upper right",
                   title=f"stop = plateau ({args.window} ep) + indicator hot",
                   title_fontsize=7)
    axes[0].set_ylabel("error vs CASCI (mHa)")
    fig.suptitle("where the subspace-size indicators would have stopped the runs "
                 "(mean over seeds; marker = median stop epoch)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=150)
    plt.close(fig)
    print(f"\nwrote {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
