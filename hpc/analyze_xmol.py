"""
Summarise a cross-molecule / transfer experiment across seeds.

Answers the one question these experiments exist to answer:

    how much of trained-on-target performance does a policy that never saw the
    target recover, and how much does that vary across seeds?

Reads a W&B group and splits runs into
  zero-shot : trained elsewhere, evaluated via zeroshot/<molecule>/...
  baseline  : trained directly on the target, so no zeroshot/ keys

Two refinement stages are reported, because they tell different stories:

  GQE-optimized    the circuit's own QSCI energy — raw policy quality
  Global-refined   after classical subspace refinement — the deployed number

Global refinement enlarges the determinant subspace up to qsci.max_dim, so when
it saturates that cap the classical step is doing part of the work and compresses
the difference between a good and a mediocre circuit. Reporting only the refined
number flatters transfer; only the raw one ignores the pipeline actually used.

A run may hold out SEVERAL geometries of the target (the hydrogen-chain set
evaluates H10 at four bond lengths), so zero-shot results are reported per
held-out molecule rather than collapsed — a mean over geometries of differing
difficulty is not a meaningful number.

The runs log `<metric> - R-CASCI` in Hartree, so no reference lookup is needed.

Usage (needs `wandb login` and the runs SYNCED):
    python3 hpc/analyze_xmol.py
    python3 hpc/analyze_xmol.py --group hchain-h4h6h8-to-h10
    python3 hpc/analyze_xmol.py --min-iters 0          # keep probe runs too
"""

import argparse
import re
import statistics as stats
import sys
from collections import defaultdict

HA_TO_MHA = 1000.0
CHEMICAL_ACCURACY_MHA = 1.6

STAGES = ["GQE-optimized", "Global-refined"]
ZS_KEY = "zeroshot/{mol}/{stage}(best_so_far)/energy - R-CASCI"
BL_KEY = "{stage}(best_so_far)/energy - R-CASCI"

ZS_PATTERN = re.compile(r"^zeroshot/([^/]+)/")


def mha(summary, key):
    v = summary.get(key)
    if v is None:
        return None
    try:
        return float(v) * HA_TO_MHA
    except (TypeError, ValueError):
        return None


def eval_molecules(summary):
    """Every held-out molecule this run reported, from its zeroshot/<mol>/ keys."""
    found = set()
    for k in summary:
        m = ZS_PATTERN.match(k)
        if m:
            found.add(m.group(1))
    return sorted(found)


def run_iters(run):
    """max_iters as logged in wandb.config, or None."""
    try:
        return int(run.config["trainer"]["max_iters"])
    except (KeyError, TypeError, ValueError):
        return None


def fmt(vals, width=9):
    if not vals:
        return "n/a".rjust(width)
    if len(vals) == 1:
        return f"{vals[0]:{width}.2f}"
    return f"{stats.mean(vals):{width}.2f} +/- {stats.stdev(vals):6.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="xmol-lih-h2o-to-n2")
    ap.add_argument("--project", default="gqe-for-qsci")
    ap.add_argument("--entity", default=None)
    ap.add_argument(
        "--min-iters", type=int, default=10,
        help="drop runs with fewer training iterations than this — smoke tests "
             "inherit the experiment's W&B group and would otherwise be "
             "averaged in as extra seeds. 0 keeps everything.",
    )
    args = ap.parse_args()

    try:
        import wandb
    except ImportError:
        sys.exit("wandb not installed - run this inside the container, or "
                 "`pip install wandb`.")

    api = wandb.Api()
    path = f"{args.entity + '/' if args.entity else ''}{args.project}"
    all_runs = list(api.runs(path, filters={"group": args.group}))
    if not all_runs:
        sys.exit(f"No runs found in group '{args.group}' of project '{path}'. "
                 "Did you sync the offline runs?")

    runs, dropped = [], []
    for r in all_runs:
        it = run_iters(r)
        if it is not None and it < args.min_iters:
            dropped.append((r.name, it))
        else:
            runs.append(r)

    print(f"\ngroup: {args.group}   ({len(runs)} runs"
          f"{f', {len(dropped)} dropped' if dropped else ''})")
    if dropped:
        for name, it in dropped:
            print(f"  dropped {name}  ({it} iters < --min-iters {args.min_iters})")

    # zs[molecule][stage] -> errors in mHa across seeds;  bl[stage] -> same
    zs = defaultdict(lambda: defaultdict(list))
    bl = defaultdict(list)
    rows = []

    for r in sorted(runs, key=lambda r: r.name):
        s = dict(r.summary)
        mols = eval_molecules(s)
        kind = "zero-shot" if mols else "baseline"
        it = run_iters(r)

        if mols:
            for mol in mols:
                for stage in STAGES:
                    e = mha(s, ZS_KEY.format(mol=mol, stage=stage))
                    if e is not None:
                        zs[mol][stage].append(e)
            detail = ", ".join(mols)
        else:
            for stage in STAGES:
                e = mha(s, BL_KEY.format(stage=stage))
                if e is not None:
                    bl[stage].append(e)
            detail = "target"
        rows.append((r.name, kind, it, r.state, detail))

    print(f"\n{'run':<26}{'kind':<11}{'iters':>7}{'state':<11}  evaluated on")
    print("-" * 78)
    for name, kind, it, state, detail in rows:
        print(f"{name:<26}{kind:<11}{(it if it is not None else '?'):>7}"
              f"  {state:<11}{detail}")

    print("\n" + "=" * 78)
    print("RESULT — error vs the target's own CASCI, in mHa (lower is better)")
    print("=" * 78)

    print(f"\n  baseline (trained directly on the target)")
    for stage in STAGES:
        n = len(bl[stage])
        print(f"    {stage:<16}{fmt(bl[stage])}   "
              f"({n} seed{'s' if n != 1 else ''})")

    if zs:
        print(f"\n  zero-shot, per held-out molecule")
        head = f"    {'molecule':<16}" + "".join(f"{s:>26}" for s in STAGES)
        print(head)
        for mol in sorted(zs):
            line = f"    {mol:<16}"
            for stage in STAGES:
                line += f"{fmt(zs[mol][stage], width=17)}"
            print(line)

        print(f"\n  generalization gap (zero-shot minus baseline; <= 0 means "
              "transfer matched\n  training on the target)")
        for mol in sorted(zs):
            line = f"    {mol:<16}"
            for stage in STAGES:
                if zs[mol][stage] and bl[stage]:
                    gap = stats.mean(zs[mol][stage]) - stats.mean(bl[stage])
                    line += f"{gap:>+17.2f}"
                else:
                    line += f"{'n/a':>17}"
            print(line)

    print(f"\n  chemical accuracy = {CHEMICAL_ACCURACY_MHA} mHa")
    single = [k for k, v in bl.items() if len(v) == 1]
    single += [f"{m}/{s}" for m in zs for s, v in zs[m].items() if len(v) == 1]
    if single:
        print("  NOTE: single-seed entries carry no error bar — directional only.")
    print("  NOTE: Global-refined enlarges the subspace up to qsci.max_dim. If it\n"
          "        saturates that cap, the classical refinement is doing part of\n"
          "        the work; read it alongside GQE-optimized, not instead of it.")
    print("  NOTE: a baseline error far above chemical accuracy usually means the\n"
          "        subspace cap, not the policy, is the limit — check what fraction\n"
          "        of the target's determinant space qsci.max_dim covers.")


if __name__ == "__main__":
    main()
