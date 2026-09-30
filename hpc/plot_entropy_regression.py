"""
Which cheap axis linearises the subspace requirement?

Scores every candidate predictor from hpc/entropy_scan.py against two targets:

    log2(n_needed)    the determinant COUNT   -- the cost that actually binds
    -log2(fraction)   the entropy DEFICIT     -- "does the fraction shrink"

by ordinary least squares, reporting R^2 and the RMSE in bits (a RMSE of 1 bit
means "predicts the count to within a factor of 2"). Also splits on regime,
because the requirement rises to a peak near where the bond breaks and collapses
after it: a weight-based predictor can only be expected to work on the rising
branch, since past dissociation the extra determinants are near-degenerate spin
couplings that carry weight but almost no energy.

Runs in plain Python (numpy + matplotlib), NOT in the container: pyci is only
needed to produce the CSV.

    python3 hpc/plot_entropy_regression.py
    python3 hpc/plot_entropy_regression.py --csv data/subspace/entropy.csv --out figures

Figures are drawn in black and grey with marker SHAPE carrying the regime, so
they survive greyscale printing and colour-vision deficiency.
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np

# predictor -> human-readable label. "_cisd" needs no exact solve and is
# therefore the only kind that could ever be used predictively; "_fci" columns
# are diagnostics that say how much of the error is the CISD surrogate and how
# much is the descriptor itself.
PREDICTORS = [
    ("R0.1_cisd",  "Renyi 0.10 (CISD)"),
    ("R0.15_cisd", "Renyi 0.15 (CISD)"),
    ("R0.25_cisd", "Renyi 1/4 (CISD)"),
    ("R0.5_cisd",  "Renyi 1/2 (CISD)"),
    ("R0.75_cisd", "Renyi 3/4 (CISD)"),
    ("H_cisd",     "Shannon (CISD)"),
    ("R2_cisd",    "Renyi 2 (CISD)"),
    ("S_cisd",     "marginal sum (CISD)"),
    ("c_hf2_cisd", "|c_HF|^2 (CISD)"),
    ("R0.1_fci",   "Renyi 0.10 (exact)"),
    ("R0.15_fci",  "Renyi 0.15 (exact)"),
    ("R0.25_fci",  "Renyi 1/4 (exact)"),
    ("R0.5_fci",   "Renyi 1/2 (exact)"),
    ("R0.75_fci",  "Renyi 3/4 (exact)"),
    ("H_fci",      "Shannon (exact)"),
    ("R2_fci",     "Renyi 2 (exact)"),
    ("S_fci",      "marginal sum (exact)"),
    ("log2_n_fci", "log2 CI dimension"),
]

TARGETS = [
    ("log2_n_needed", "log2 n_needed"),
    ("deficit_bits",  "-log2 fraction"),
]


def load(path):
    if not os.path.exists(path):
        sys.exit(f"{path} not found -- run hpc/entropy_scan.py first (in the "
                 f"container: pyci is not installed outside it).")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                r["n_needed"] = int(r["n_needed"])
                r["n_fci"] = int(r["n_fci"])
                r["bond_length"] = float(r["bond_length"])
                r["log2_n_needed"] = math.log2(r["n_needed"])
                r["log2_n_fci"] = math.log2(r["n_fci"])
                r["deficit_bits"] = float(r["deficit_bits"])
            except (KeyError, ValueError):
                continue
            for key, _ in PREDICTORS:
                if key in ("log2_n_fci",):
                    continue
                try:
                    r[key] = float(r[key])
                except (KeyError, ValueError):
                    r[key] = float("nan")
            rows.append(r)
    if not rows:
        sys.exit(f"{path} has no usable rows.")
    print(f"loaded {len(rows)} configuration(s) from {path}")
    return rows


def tag_regimes(rows, min_points=4):
    """
    Mark each configuration 'rising' or 'past-peak' relative to its family's
    most demanding geometry. Families with too few geometries to locate a peak
    are left 'rising', which is the branch they are presumed to be on.
    """
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    for fam, rs in by_fam.items():
        if len(rs) < min_points:
            for r in rs:
                r["regime"] = "rising"
            continue
        peak = max(rs, key=lambda r: r["n_needed"])["bond_length"]
        for r in rs:
            r["regime"] = "rising" if r["bond_length"] <= peak else "past-peak"
    return by_fam


def fit(xs, ys):
    """(slope, intercept, R^2, rmse). None if there is nothing to fit."""
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.ptp(x) == 0:
        return None
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    resid = y - pred
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return b, a, r2, float(np.sqrt((resid ** 2).mean())), x.size


def score_table(rows, target_key, target_label, subsets):
    print(f"\n\n=== predicting {target_label} ===")
    head = f"{'predictor':<22}" + "".join(f"{name:>26}" for name, _ in subsets)
    print(head)
    print(f"{'':<22}" + "".join(f"{'R2      rmse   slope':>26}" for _ in subsets))
    print("-" * len(head))
    ranked = []
    for key, label in PREDICTORS:
        cells, primary = [], None
        for i, (_, subset) in enumerate(subsets):
            res = fit([r[key] for r in subset], [r[target_key] for r in subset])
            if res is None:
                cells.append(f"{'-':>26}")
                continue
            b, a, r2, rmse, n = res
            cells.append(f"{r2:>14.3f}{rmse:>7.2f}{b:>7.2f}")
            if i == 0:
                primary = r2
        ranked.append((primary if primary is not None else -9, label, key, cells))
    for _, label, key, cells in sorted(ranked, reverse=True):
        print(f"{label:<22}" + "".join(cells))

    ordering = [(lab, k) for _, lab, k, _ in sorted(ranked, reverse=True)]

    # The slope alone is not enough to claim a law: state the intercept too, so
    # "slope 1" can be read as n_needed = 2^(x + a) rather than just a good fit.
    print()
    for tag, want_cheap in (("best overall", False), ("best without an exact solve", True)):
        pick = next((x for x in ordering
                     if x[1].endswith("_cisd") == want_cheap), None)
        if pick is None:
            continue
        label, key = pick
        res = fit([r[key] for r in subsets[0][1]], [r[target_key] for r in subsets[0][1]])
        if res is None:
            continue
        b, a, r2, rmse, n = res
        print(f"  {tag:<28} {label:<20} "
              f"{target_label} = {b:+.3f}*x {a:+.3f}   "
              f"R2 {r2:.3f}, {rmse:.2f} bits (x{2**rmse:.2f}), n={n}")
    return ordering


def fig_scatter(rows, key, label, target_key, target_label, path, plt):
    """
    Black filled circles for the rising branch, grey open squares past the peak.
    Regime is carried by SHAPE and FILL, not hue, so the figure reads in
    greyscale and with any colour vision.
    """
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    styles = {"rising":    dict(marker="o", mfc="k",    mec="k", label="rising branch"),
              "past-peak": dict(marker="s", mfc="none", mec="0.45",
                                label="past the peak")}
    for regime, style in styles.items():
        rs = [r for r in rows if r.get("regime") == regime
              and np.isfinite(r[key])]
        if not rs:
            continue
        ax.plot([r[key] for r in rs], [r[target_key] for r in rs],
                ls="none", ms=5.5, mew=1.1, **style)

    rising = [r for r in rows if r.get("regime") == "rising" and np.isfinite(r[key])]
    res = fit([r[key] for r in rising], [r[target_key] for r in rising])
    if res:
        b, a, r2, rmse, n = res
        xs = np.array([min(r[key] for r in rising), max(r[key] for r in rising)])
        ax.plot(xs, a + b * xs, "k--", lw=1.1,
                label=f"rising fit: R$^2$={r2:.2f}, {rmse:.2f} bits")

    ax.set(xlabel=f"{label}  [bits]", ylabel=f"{target_label}  [bits]",
           title=f"{target_label} against {label}")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=150)
    plt.close(fig)


def fig_regime(by_fam, plt, path):
    """Why the split exists: requirement against bond length, families with
    enough geometries to show the rise and the collapse. Greyscale, labelled at
    the line end instead of by colour."""
    fams = [(f, rs) for f, rs in by_fam.items() if len(rs) >= 8]
    fig, ax = plt.subplots(figsize=(7.6, 5))
    greys = np.linspace(0.0, 0.7, max(len(fams), 1))
    for (fam, rs), g in zip(sorted(fams), greys):
        rs = sorted(rs, key=lambda r: r["bond_length"])
        x = [r["bond_length"] for r in rs]
        y = [r["log2_n_needed"] for r in rs]
        ax.plot(x, y, marker="o", ms=3, lw=1.2, color=str(g))
        ax.annotate(fam, (x[-1], y[-1]), textcoords="offset points",
                    xytext=(4, 0), fontsize=7, color=str(g), va="center")
        peak = max(rs, key=lambda r: r["n_needed"])
        ax.plot([peak["bond_length"]], [peak["log2_n_needed"]], marker="x",
                ms=8, mew=1.6, color=str(g), ls="none")
    ax.set(xlabel="bond length [A]", ylabel="log2 n_needed  [bits]",
           title="the requirement peaks near bond breaking (x) and collapses after")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/subspace/entropy.csv")
    ap.add_argument("--out", default="figures_entropy")
    ap.add_argument("--min-ci", type=int, default=1000,
                    help="drop CI spaces smaller than this: 'h2 needs 50%%' is "
                         "2 determinants of 4 and no asymptotic law applies")
    args = ap.parse_args()

    rows = load(args.csv)
    kept = [r for r in rows if r["n_fci"] >= args.min_ci]
    print(f"{len(kept)} of {len(rows)} have >= {args.min_ci:,} determinants "
          f"(the rest are too small for an asymptotic law)")
    by_fam = tag_regimes(kept)

    rising = [r for r in kept if r["regime"] == "rising"]
    past = [r for r in kept if r["regime"] == "past-peak"]
    subsets = [("rising branch", rising), ("all configurations", kept)]
    print(f"regimes: {len(rising)} rising, {len(past)} past-peak")

    best = {}
    for tkey, tlabel in TARGETS:
        ordering = score_table(kept, tkey, tlabel, subsets)
        best[tkey] = ordering

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("\nmatplotlib not available here; the tables above are the result.")

    os.makedirs(args.out, exist_ok=True)
    fig_regime(by_fam, plt, os.path.join(args.out, "regimes"))

    # the best cheap predictor, and the best one overall, for each target
    for tkey, tlabel in TARGETS:
        ordering = best[tkey]
        cheap = next((x for x in ordering if x[1].endswith("_cisd")), None)
        exact = next((x for x in ordering if x[1].endswith("_fci")), None)
        for tag, pick in (("cheap", cheap), ("exact", exact)):
            if pick is None:
                continue
            label, key = pick
            name = f"{tkey}_vs_{tag}"
            fig_scatter(kept, key, label, tkey, tlabel,
                        os.path.join(args.out, name), plt)
    print(f"\nfigures in {args.out}/")


if __name__ == "__main__":
    main()
