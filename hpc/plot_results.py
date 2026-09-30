"""
Summaries and figures from the per-circuit CSVs, without re-running anything.

Every run writes `<output>/results.csv`, one row per evaluated circuit (see
gqe_qsci/results_log.py). This globs them together and either prints a text
summary or draws figures.

TWO GROUPINGS MATTER, and getting either wrong silently averages unrelated runs:

  family   the exp_tag minus its -s<seed> suffix. Grouping by `model` is wrong:
           CircuitDAGGNNPolicy covers archL15-dag-gnn, L12-dag-gnn, L20-dag-gnn
           and allpauli-h10 -- four different experiments.
  molecule identified by the CASCI reference, because single-molecule runs all
           log molecule="target" and that cannot tell H10 (-4.9237) from N2
           (-107.44). Runs with different references never share an axis; their
           errors are not comparable.

DEPENDENCIES are deliberately minimal: loading, `--summary` and `--merge` use
only the standard library, because the ABCI-Q container has numpy/scipy/torch
but NOT pandas or matplotlib. Only figures need matplotlib.

    # on the cluster:
    python3 hpc/plot_results.py --summary
    python3 hpc/plot_results.py --merge merged.csv

    # anywhere with matplotlib:
    python3 hpc/plot_results.py --glob merged.csv --out figures/

Columns: exp_tag, model, n_params, seed, epoch, molecule, split, stage,
sample_idx, energy, R-CASCI, R-CCSD, subspace_dim, num_sampled_basis,
num_symmetry_preserving_basis, cx_count, total_gates, seq.
"""

import argparse
import csv
import glob
import os
import re
import statistics as stats
import sys
from collections import defaultdict

HA_TO_MHA = 1000.0
CHEMICAL_ACCURACY_MHA = 1.6
STAGES = ["GQE-optimized(best_so_far)", "Global-refined(best_so_far)"]

NUMERIC = {
    "n_params", "seed", "epoch", "sample_idx", "energy", "R-CASCI", "R-CCSD",
    "subspace_dim", "num_sampled_basis", "num_symmetry_preserving_basis",
    "cx_count", "total_gates", "tail_weight", "boundary_mha", "pt2_mha",
}

# Subspace-size indicators (gqe_qsci/qsci/diagnostics.py). "Hot" means the
# indicator says the subspace is still limiting the energy. Thresholds: ENPT2 at
# chemical accuracy; the two boundary ones from the exact-vector calibration.
# They are LOGGED, never acted on — pt2 is the reliable one, the boundary pair
# flickers on sampled subspaces. See NOTES.md, "Live subspace-size indicators".
INDICATORS = [
    ("pt2", "pt2_mha", lambda v: abs(v) > CHEMICAL_ACCURACY_MHA),
    ("tail", "tail_weight", lambda v: v > 1e-5),
    ("boundary", "boundary_mha", lambda v: v > 0.1),
]
PLATEAU_WINDOW = 25      # epochs
PLATEAU_DELTA = 1.0      # mHa: less improvement than this over the window

SEED_SUFFIX = re.compile(r"-s\d+$")


def family(row):
    """exp_tag without its -s<seed> suffix: the unit a sweep varies."""
    return SEED_SUFFIX.sub("", row.get("exp_tag") or "")


def molecule_key(row):
    """The system, identified by its CASCI reference (see module docstring)."""
    ref = row.get("R-CASCI")
    return round(ref, 3) if ref is not None else None


def _num(value):
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except ValueError:
        return None
    return int(f) if f.is_integer() else f


def load_rows(pattern):
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
                e, ref = row.get("energy"), row.get("R-CASCI")
                row["err_mha"] = (e - ref) * HA_TO_MHA if (e is not None and ref is not None) else None
                rows.append(row)

    print(f"loaded {len(paths)} file(s), {len(rows):,} rows, "
          f"{len({r['exp_tag'] for r in rows})} run(s), "
          f"{len({family(r) for r in rows})} famil(ies)")
    return rows


# --------------------------------------------------------------------------
# text summary - standard library only
# --------------------------------------------------------------------------

def _final_per_seed(rows, stage):
    """{family: [final error, one per seed]}"""
    latest = {}
    for r in rows:
        if r["stage"] != stage or r["err_mha"] is None:
            continue
        key = (family(r), r["exp_tag"], r["seed"])
        if key not in latest or r["epoch"] > latest[key][0]:
            latest[key] = (r["epoch"], r["err_mha"])
    out = defaultdict(list)
    for (fam, _t, _s), (_e, err) in latest.items():
        out[fam].append(err)
    return out


def summarize(rows):
    by_mol = defaultdict(list)
    for r in rows:
        by_mol[molecule_key(r)].append(r)

    for ref, mol_rows in sorted(by_mol.items(), key=lambda kv: (kv[0] is None, kv[0])):
        models = {family(r): (r["model"], r["n_params"]) for r in mol_rows}
        print()
        print("=" * 92)
        print(f"SYSTEM with CASCI = {ref}      ({len(models)} run families)")
        print("=" * 92)
        print(f"{'family':<26} {'model':<32} {'params':>11} "
              f"{'GQE-optimized':>16} {'Global-refined':>16}")
        print("-" * 92)
        finals = {st: _final_per_seed(mol_rows, st) for st in STAGES}
        for fam in sorted(models):
            model, n_params = models[fam]
            cells = []
            for st in STAGES:
                errs = finals[st].get(fam, [])
                if not errs:
                    cells.append("-")
                elif len(errs) == 1:
                    cells.append(f"{errs[0]:.2f}")
                else:
                    cells.append(f"{stats.mean(errs):.2f}+/-{stats.stdev(errs):.2f}")
            params = f"{n_params:,}" if isinstance(n_params, int) else "?"
            print(f"{fam:<26} {model:<32} {params:>11} {cells[0]:>16} {cells[1]:>16}")

        print(f"\n  errors in mHa vs this system's CASCI; chemical accuracy = "
              f"{CHEMICAL_ACCURACY_MHA} mHa")
        if len({p for _m, p in models.values()}) > 1:
            print("  NOTE: parameter counts differ. A ranking that tracks `params`")
            print("        is a capacity result, not an architecture one.")

        # stall check
        seen, last_gain, max_ep = {}, {}, defaultdict(int)
        for r in sorted(mol_rows, key=lambda r: (r["exp_tag"], r["stage"], r["epoch"] or 0)):
            if r["stage"] not in STAGES or r["err_mha"] is None:
                continue
            key = (r["exp_tag"], r["stage"])
            max_ep[key] = max(max_ep[key], r["epoch"] or 0)
            if key not in seen or r["err_mha"] < seen[key] - 1e-9:
                seen[key] = r["err_mha"]
                last_gain[key] = r["epoch"]
        tag_fam = {r["exp_tag"]: family(r) for r in mol_rows}
        pct = defaultdict(list)
        for key, ep in last_gain.items():
            pct[(tag_fam[key[0]], key[1])].append(100.0 * ep / (max_ep[key] or 1))
        if pct:
            print(f"\n  STALL: last epoch with a new best, as % of the run")
            for (fam, stage), ps in sorted(pct.items()):
                cell = (f"{ps[0]:.0f}%" if len(ps) == 1
                        else f"{stats.mean(ps):.0f}% +/- {stats.stdev(ps):.0f}")
                print(f"    {fam:<26} {stage:<30} {cell:>14}")
            print("    low % = training continued long after the policy stopped improving")

        _indicator_report(mol_rows)


def _indicator_report(rows):
    """
    Did the subspace-size indicators fire, and when?

    "fired" = the first epoch at which the best-so-far error had plateaued
    (improved by less than PLATEAU_DELTA mHa over PLATEAU_WINDOW epochs) WHILE
    the indicator was hot. That is the point where a run could have been told
    "this plateau is the subspace, not the policy". Nothing acts on it.

    Runs from before the indicator columns existed are skipped silently.
    """
    runs = defaultdict(list)          # (family, stage, exp_tag) -> rows
    for r in rows:
        if r["stage"] in STAGES and r["err_mha"] is not None \
                and r.get("pt2_mha") is not None:
            runs[(family(r), r["stage"], r["exp_tag"])].append(r)
    if not runs:
        return

    fired = defaultdict(lambda: defaultdict(list))   # (fam, stage) -> name -> [epoch|None]
    last = defaultdict(list)                         # (fam, stage) -> [final pt2]
    for (fam, stage, _tag), rs in runs.items():
        rs.sort(key=lambda r: r["epoch"])
        err = {r["epoch"]: r["err_mha"] for r in rs}
        last[(fam, stage)].append(abs(rs[-1]["pt2_mha"]))
        for name, column, hot in INDICATORS:
            epoch = None
            for r in rs:
                past = err.get(r["epoch"] - PLATEAU_WINDOW)
                value = r.get(column)
                if past is None or value is None:
                    continue
                if past - r["err_mha"] < PLATEAU_DELTA and hot(value):
                    epoch = r["epoch"]
                    break
            fired[(fam, stage)][name].append(epoch)

    print(f"\n  SUBSPACE INDICATORS: epoch at which the run had plateaued "
          f"(< {PLATEAU_DELTA:g} mHa over {PLATEAU_WINDOW} epochs)")
    print("  while the indicator was hot. Median over seeds; (k/n) = seeds "
          "on which it fired.")
    print(f"    {'family':<26} {'stage':<28} {'pt2':>12} {'tail':>12} "
          f"{'boundary':>12} {'final |pt2|':>12}")
    for (fam, stage) in sorted(fired):
        cells = []
        for name, _c, _h in INDICATORS:
            eps = fired[(fam, stage)][name]
            hit = [e for e in eps if e is not None]
            cells.append(f"{int(stats.median(hit))} ({len(hit)}/{len(eps)})"
                         if hit else f"- (0/{len(eps)})")
        print(f"    {fam:<26} {stage:<28} {cells[0]:>12} {cells[1]:>12} "
              f"{cells[2]:>12} {stats.mean(last[(fam, stage)]):>10.2f} mHa")
    print("    pt2 (ENPT2) is the one to trust: it estimates the energy just "
          "outside the subspace.")
    print("    '-' = never fired: the plateau, if any, is not the subspace size.")


# --------------------------------------------------------------------------
# figures - one set per system, GQE-optimized and Global-refined side by side
# --------------------------------------------------------------------------

def _series(rows, stage, value):
    buckets = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["stage"] != stage or r.get(value) is None:
            continue
        buckets[family(r)][r["epoch"]].append(r[value])
    return {f: {e: stats.mean(v) for e, v in sorted(s.items())}
            for f, s in buckets.items()}


def paired_curve(rows, plt, value, ylabel, title, logy, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    drew = False
    for ax, stage in zip(axes, STAGES):
        for fam, pts in sorted(_series(rows, stage, value).items()):
            ax.plot(list(pts), list(pts.values()), lw=1.5, label=fam)
            drew = True
        ax.set(xlabel="epoch", title=stage)
        if logy:
            ax.set_yscale("log")
        if value == "err_mha":
            ax.axhline(CHEMICAL_ACCURACY_MHA, ls="--", c="k", lw=0.8)
        ax.grid(alpha=0.25, lw=0.5)
    if not drew:
        plt.close(fig)
        return False
    axes[0].set_ylabel(ylabel)
    axes[1].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=150)
    plt.close(fig)
    return True


def final_scatter(rows, plt, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    drew = False
    for ax, stage in zip(axes, STAGES):
        finals = _final_per_seed(rows, stage)
        names = sorted(finals)
        for i, fam in enumerate(names):
            ax.scatter([i] * len(finals[fam]), finals[fam], s=26, zorder=3)
            ax.scatter([i], [stats.mean(finals[fam])], marker="_", s=420, c="k", zorder=4)
            drew = True
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=6, rotation=35, ha="right")
        ax.set(title=stage, yscale="log")
        ax.axhline(CHEMICAL_ACCURACY_MHA, ls="--", c="k", lw=0.8)
        ax.grid(alpha=0.25, lw=0.5, axis="y")
    if not drew:
        plt.close(fig)
        return False
    axes[0].set_ylabel("final error vs CASCI (mHa)")
    fig.suptitle(title + "    points = seeds, bar = mean, dashed = chemical accuracy")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=150)
    plt.close(fig)
    return True


def cost_vs_error(rows, plt, title, path):
    """Accuracy against compiled circuit cost, which is what changing L buys."""
    fig, ax = plt.subplots(figsize=(7, 4.8))
    finals = _final_per_seed(rows, "Global-refined(best_so_far)")
    costs = defaultdict(list)
    for r in rows:
        if r["stage"] == "GQE-optimized(best_so_far)" and r.get("total_gates"):
            costs[family(r)].append(r["total_gates"])
    drew = False
    for fam in sorted(finals):
        if fam not in costs:
            continue
        x, y = stats.mean(costs[fam]), stats.mean(finals[fam])
        ax.scatter(x, y, s=52, zorder=3)
        ax.annotate(fam, (x, y), fontsize=6, xytext=(5, 4), textcoords="offset points")
        drew = True
    if not drew:
        plt.close(fig)
        return False
    ax.axhline(CHEMICAL_ACCURACY_MHA, ls="--", c="k", lw=0.8)
    ax.set(xlabel="compiled gates in the best circuit",
           ylabel="final error vs CASCI (mHa)", yscale="log", title=title)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=150)
    plt.close(fig)
    return True


def make_figures(rows, out_dir, plt):
    os.makedirs(out_dir, exist_ok=True)
    by_mol = defaultdict(list)
    for r in rows:
        by_mol[molecule_key(r)].append(r)

    for ref, mol_rows in sorted(by_mol.items(), key=lambda kv: (kv[0] is None, kv[0])):
        fams = {family(r): r["model"] for r in mol_rows}
        label = f"CASCI={ref}"
        slug = "casci" + str(ref).replace(".", "p").replace("-", "m")
        print(f"\n{label}: {len(fams)} run families")
        for fam, model in sorted(fams.items()):
            print(f"    {fam:<28} {model}")

        for value, ylabel, name, logy in [
            ("err_mha", "error vs CASCI (mHa)", "error", True),
            ("subspace_dim", "subspace dim", "subspace", False),
            ("num_sampled_basis", "distinct determinants sampled", "sampled_basis", False),
        ]:
            path = os.path.join(out_dir, f"{name}_{slug}")
            if paired_curve(mol_rows, plt, value, ylabel,
                            f"{name} ({label}, mean over seeds)", logy, path):
                print(f"  wrote {name}_{slug}")
        if final_scatter(mol_rows, plt, f"final error ({label})",
                         os.path.join(out_dir, f"final_{slug}")):
            print(f"  wrote final_{slug}")
        if cost_vs_error(mol_rows, plt, f"cost vs accuracy ({label})",
                         os.path.join(out_dir, f"cost_{slug}")):
            print(f"  wrote cost_{slug}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/gqe-for-qsci/*")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--summary", action="store_true",
                    help="text only; no third-party packages, so it works in "
                         "the ABCI-Q container")
    ap.add_argument("--merge", metavar="PATH",
                    help="write every row to one CSV and exit")
    ap.add_argument("--exclude", default="smoke",
                    help="regex; run families matching it are dropped. Defaults "
                         "to 'smoke' because probe runs inherit the experiment's "
                         "W&B group and output tree, and a 2-iteration probe in "
                         "the tables looks like a catastrophically bad model. "
                         "Pass '' to keep everything.")
    args = ap.parse_args()

    rows = load_rows(args.glob)

    if args.exclude:
        pattern = re.compile(args.exclude)
        dropped = sorted({family(r) for r in rows if pattern.search(family(r))})
        if dropped:
            rows = [r for r in rows if not pattern.search(family(r))]
            print(f"excluded {len(dropped)} famil(ies) matching "
                  f"{args.exclude!r}: {', '.join(dropped)}")

    if args.merge:
        # Union over ALL rows, first-seen order: runs from before a column
        # existed would otherwise decide the header and silently drop it.
        fields = list(dict.fromkeys(
            k for r in rows for k in r if k != "err_mha"))
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
        print("Use --merge and draw the figures on a machine that has it.")
        summarize(rows)
        return

    make_figures(rows, args.out, plt)
    print(f"\nfigures in {args.out}/")


if __name__ == "__main__":
    main()
