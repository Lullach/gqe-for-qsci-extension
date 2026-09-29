"""
Summaries and figures from the per-circuit CSVs, without re-running anything.

Every run writes `<output>/results.csv`, one row per evaluated circuit (see
gqe_qsci/results_log.py). This globs them together and either prints a text
summary or draws the default figures.

DEPENDENCIES, deliberately minimal: loading and `--summary` use only the
standard library, because the ABCI-Q container has numpy/scipy/torch but NOT
pandas or matplotlib, and rebuilding it just to read a CSV would be absurd.
Figures need matplotlib, so draw those on your laptop — copy the CSVs over, or
`--merge` them into one file first.

    # on the cluster (no extra deps):
    python3 hpc/plot_results.py --summary
    python3 hpc/plot_results.py --merge merged.csv

    # on your laptop (matplotlib available):
    python3 hpc/plot_results.py --glob "outputs/gqe-for-qsci/archL15-*"

Columns: exp_tag, model, n_params, seed, epoch, molecule, split, stage,
sample_idx, energy, R-CASCI, R-CCSD, subspace_dim, num_sampled_basis,
num_symmetry_preserving_basis, cx_count, total_gates, seq.

`stage` is "GQE-optimized" for every sampled circuit of a rollout
(sample_idx >= 0) and "<...>(best_so_far)" for the running bests
(sample_idx == -1). Errors are not precomputed: the reference energies are
columns, so pick your own convention.
"""

import argparse
import csv
import glob
import os
import statistics as stats
import sys
from collections import defaultdict

HA_TO_MHA = 1000.0
CHEMICAL_ACCURACY_MHA = 1.6

NUMERIC = {
    "n_params", "seed", "epoch", "sample_idx", "energy", "R-CASCI", "R-CCSD",
    "subspace_dim", "num_sampled_basis", "num_symmetry_preserving_basis",
    "cx_count", "total_gates",
}


def _num(value):
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except ValueError:
        return None
    return int(f) if f.is_integer() else f


def load_rows(pattern):
    """List of dicts, stdlib only. Adds err_mha where a CASCI reference exists."""
    paths = sorted(glob.glob(os.path.join(pattern, "results.csv")))
    if not paths:
        paths = [p for p in sorted(glob.glob(pattern)) if p.endswith(".csv")]
    if not paths:
        sys.exit(f"no results.csv found under {pattern!r}. Runs that finished "
                 "before results_log.py landed do not have one.")

    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for k in NUMERIC:
                    if k in row:
                        row[k] = _num(row[k])
                row["run_dir"] = os.path.dirname(path)
                if row.get("energy") is not None and row.get("R-CASCI") is not None:
                    row["err_mha"] = (row["energy"] - row["R-CASCI"]) * HA_TO_MHA
                else:
                    row["err_mha"] = None
                rows.append(row)

    tags = {r["exp_tag"] for r in rows}
    models = {r["model"] for r in rows}
    print(f"loaded {len(paths)} run(s), {len(rows):,} rows, "
          f"{len(tags)} exp_tag(s), {len(models)} model(s)")
    return rows


def group_mean(rows, key_fields, value_field):
    """{key tuple: {x: mean over rows}} — the one bit of pandas actually used."""
    buckets = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = r.get(value_field)
        x = r.get("epoch")
        if v is None or x is None:
            continue
        buckets[tuple(r.get(k) for k in key_fields)][x].append(v)
    return {
        key: {x: stats.mean(vs) for x, vs in sorted(series.items())}
        for key, series in buckets.items()
    }


# --------------------------------------------------------------------------
# text summary — works with zero non-stdlib dependencies
# --------------------------------------------------------------------------

def summarize(rows):
    best = [r for r in rows
            if r["sample_idx"] == -1 and "best_so_far" in (r["stage"] or "")]
    if not best:
        print("\nno best_so_far rows found.")
        return

    # final (= largest epoch) error per run, per stage
    final = {}
    for r in best:
        if r["err_mha"] is None:
            continue
        key = (r["model"], r["n_params"], r["stage"], r["exp_tag"], r["seed"])
        prev = final.get(key)
        if prev is None or r["epoch"] > prev[0]:
            final[key] = (r["epoch"], r["err_mha"])

    agg = defaultdict(list)
    for (model, n_params, stage, _tag, _seed), (_ep, err) in final.items():
        agg[(model, n_params, stage)].append(err)

    print()
    print("=" * 86)
    print("FINAL best-so-far error vs CASCI, mHa (lower is better)")
    print("=" * 86)
    print(f"{'model':<24} {'params':>11}  {'stage':<28} {'error':>18}")
    print("-" * 86)
    for (model, n_params, stage), errs in sorted(agg.items()):
        if len(errs) == 1:
            cell = f"{errs[0]:.2f}  (1 seed)"
        else:
            cell = f"{stats.mean(errs):.2f} +/- {stats.stdev(errs):.2f} ({len(errs)})"
        params = f"{n_params:,}" if isinstance(n_params, int) else "?"
        print(f"{model:<24} {params:>11}  {stage:<28} {cell:>18}")

    print(f"\n  chemical accuracy = {CHEMICAL_ACCURACY_MHA} mHa")
    if len({p for _m, p, _s in agg}) > 1:
        print("  NOTE: parameter counts differ between models. A ranking that")
        print("        tracks `params` is a capacity result, not an architecture one.")

    # did the policy stall?
    print()
    print("=" * 86)
    print("STALL CHECK - last epoch at which a new best was found")
    print("=" * 86)
    last_improve = {}
    seen = {}
    for r in sorted(best, key=lambda r: (r["exp_tag"], r["stage"], r["epoch"])):
        if r["err_mha"] is None:
            continue
        key = (r["exp_tag"], r["stage"])
        if key not in seen or r["err_mha"] < seen[key] - 1e-9:
            seen[key] = r["err_mha"]
            last_improve[key] = r["epoch"]
    max_epoch = defaultdict(int)
    for r in best:
        max_epoch[(r["exp_tag"], r["stage"])] = max(
            max_epoch[(r["exp_tag"], r["stage"])], r["epoch"] or 0)
    tag_to_model = {r["exp_tag"]: r["model"] for r in best}
    by_model = defaultdict(list)
    for key, epoch in last_improve.items():
        total = max_epoch[key] or 1
        by_model[(tag_to_model.get(key[0], "?"), key[1])].append(
            100.0 * epoch / total)
    print(f"{'model':<24} {'stage':<28} {'last gain (% of run)':>26}")
    print("-" * 86)
    for (model, stage), pcts in sorted(by_model.items()):
        if len(pcts) == 1:
            cell = f"{pcts[0]:.0f}%  (1 seed)"
        else:
            cell = f"{stats.mean(pcts):.0f}% +/- {stats.stdev(pcts):.0f} ({len(pcts)})"
        print(f"{model:<24} {stage:<28} {cell:>26}")
    print("\n  A low percentage means training continued long after the policy")
    print("  stopped improving: wasted compute, or exploration that died early.")


# --------------------------------------------------------------------------
# figures — matplotlib only
# --------------------------------------------------------------------------

def fig_best_vs_epoch(rows, ax):
    series = group_mean([r for r in rows if r["sample_idx"] == -1],
                        ("model", "stage"), "err_mha")
    if not series:
        return False
    for (model, stage), pts in sorted(series.items()):
        ax.plot(list(pts), list(pts.values()), label=f"{model} / {stage}", lw=1.4)
    ax.axhline(CHEMICAL_ACCURACY_MHA, ls="--", c="k", lw=0.8)
    ax.set(xlabel="epoch", ylabel="error vs CASCI (mHa)", yscale="log",
           title="best-so-far error (mean over seeds)")
    ax.legend(fontsize=6)
    return True


def fig_subspace_vs_epoch(rows, ax):
    series = group_mean([r for r in rows if r["sample_idx"] == -1],
                        ("model", "stage"), "subspace_dim")
    if not series:
        return False
    for (model, stage), pts in sorted(series.items()):
        ax.plot(list(pts), list(pts.values()), label=f"{model} / {stage}", lw=1.4)
    ax.set(xlabel="epoch", ylabel="subspace dim",
           title="subspace size (flat = saturated cap, or no new best)")
    ax.legend(fontsize=6)
    return True


def fig_sample_spread(rows, ax):
    batch = [r for r in rows if (r["sample_idx"] or -1) >= 0 and r["err_mha"] is not None]
    if not batch:
        return False
    per_epoch = defaultdict(list)
    for r in batch:
        per_epoch[r["epoch"]].append(r["err_mha"])
    xs = sorted(per_epoch)
    lo = [min(per_epoch[x]) for x in xs]
    hi = [max(per_epoch[x]) for x in xs]
    mid = [stats.median(per_epoch[x]) for x in xs]
    ax.fill_between(xs, lo, hi, alpha=0.25, label="min-max")
    ax.plot(xs, mid, lw=1.4, label="median")
    ax.axhline(CHEMICAL_ACCURACY_MHA, ls="--", c="k", lw=0.8)
    ax.set(xlabel="epoch", ylabel="error vs CASCI (mHa)", yscale="log",
           title="rollout batch spread (collapse = exploration died)")
    ax.legend(fontsize=6)
    return True


def fig_final_by_model(rows, ax):
    best = [r for r in rows
            if r["sample_idx"] == -1 and "best_so_far" in (r["stage"] or "")
            and r["err_mha"] is not None]
    if not best:
        return False
    final = {}
    for r in best:
        key = (r["model"], r["stage"], r["exp_tag"], r["seed"])
        if key not in final or r["epoch"] > final[key][0]:
            final[key] = (r["epoch"], r["err_mha"])
    grouped = defaultdict(list)
    for (model, stage, _t, _s), (_e, err) in final.items():
        grouped[(model, stage)].append(err)
    labels, positions = [], []
    for i, (key, errs) in enumerate(sorted(grouped.items())):
        ax.scatter([i] * len(errs), errs, s=18)
        labels.append(f"{key[0]}\n{key[1]}")
        positions.append(i)
    ax.axhline(CHEMICAL_ACCURACY_MHA, ls="--", c="k", lw=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=5, rotation=30, ha="right")
    ax.set(ylabel="final error vs CASCI (mHa)", yscale="log",
           title="final error per model (one point per seed)")
    return True


FIGURES = [
    ("best_vs_epoch", fig_best_vs_epoch),
    ("subspace_vs_epoch", fig_subspace_vs_epoch),
    ("sample_spread", fig_sample_spread),
    ("final_by_model", fig_final_by_model),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/gqe-for-qsci/*")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--summary", action="store_true",
                    help="text summary only; needs no third-party packages, so "
                         "it works inside the ABCI-Q container")
    ap.add_argument("--merge", metavar="PATH",
                    help="write every row to one CSV and exit — copy that to a "
                         "machine with matplotlib to draw figures")
    args = ap.parse_args()

    rows = load_rows(args.glob)

    if args.merge:
        fields = [k for k in rows[0] if k not in ("err_mha",)]
        with open(args.merge, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.merge} ({len(rows):,} rows)")
        return

    if args.summary:
        summarize(rows)
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib is not installed here (the ABCI-Q container has "
              "numpy/scipy/torch but not matplotlib).")
        print("Falling back to the text summary. For figures, either run this on "
              "your laptop, or:")
        print("    python3 hpc/plot_results.py --merge merged.csv")
        print("and copy merged.csv somewhere with matplotlib.")
        summarize(rows)
        return

    os.makedirs(args.out, exist_ok=True)
    for name, fn in FIGURES:
        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            ok = fn(rows, ax)
        except Exception as exc:                                  # noqa: BLE001
            print(f"  skip {name}: {type(exc).__name__}: {exc}")
            plt.close(fig)
            continue
        if not ok:
            print(f"  skip {name}: no matching rows")
            plt.close(fig)
            continue
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(args.out, f"{name}.{ext}"), dpi=150)
        plt.close(fig)
        print(f"  wrote {name}.pdf / .png")
    print(f"\nfigures in {args.out}/")


if __name__ == "__main__":
    main()
