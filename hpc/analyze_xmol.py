"""
Summarise the Phase 4 cross-molecule experiment across seeds.

Answers the one question the experiment exists to answer:

    how much of trained-on-N2 performance does a policy that never saw N2
    recover, and how much does that vary across seeds?

Reads the W&B group (default: xmol-lih-h2o-to-n2), splits runs into
  zero-shot : xmol_dag_gnn        -> zeroshot/n2/... (N2 never trained on)
  baseline  : xmol_baseline_n2    -> trained directly on N2

and reports TWO refinement stages, because they tell different stories:

  GQE-optimized    the circuit's own QSCI energy — raw policy quality
  Global-refined   after classical subspace refinement — the deployed number

The gap between the two matters. Global refinement enlarges the determinant
subspace up to qsci.max_dim, so when it saturates the cap it compresses the
difference between a good and a mediocre circuit. Reporting only the refined
number flatters zero-shot; reporting only the raw one ignores the pipeline
actually used. So both are printed.

The runs already log `<metric> - R-CASCI` in Hartree, so no reference lookup is
needed — those keys are read directly and converted to mHa.

Usage (needs `wandb login` and the runs SYNCED):
    python3 hpc/analyze_xmol.py
    python3 hpc/analyze_xmol.py --group xmol-lih-h2o-to-n2 --entity <you>
"""

import argparse
import statistics as stats
import sys

HA_TO_MHA = 1000.0
CHEMICAL_ACCURACY_MHA = 1.6

# (label, zero-shot key, baseline key). Zero-shot metrics are namespaced by the
# held-out molecule; baseline runs are single-molecule so carry no prefix.
STAGES = [
    ("GQE-optimized",
     "zeroshot/{mol}/GQE-optimized(best_so_far)/energy - R-CASCI",
     "GQE-optimized(best_so_far)/energy - R-CASCI"),
    ("Global-refined",
     "zeroshot/{mol}/Global-refined(best_so_far)/energy - R-CASCI",
     "Global-refined(best_so_far)/energy - R-CASCI"),
]


def eval_molecule(summary):
    """Held-out molecule name, discovered from the zeroshot/<mol>/... keys."""
    for k in summary:
        if k.startswith("zeroshot/"):
            parts = k.split("/")
            if len(parts) > 2:
                return parts[1]
    return None


def mha(summary, key):
    v = summary.get(key)
    if v is None:
        return None
    try:
        return float(v) * HA_TO_MHA
    except (TypeError, ValueError):
        return None


def fmt(vals):
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:8.2f}          (1 seed)"
    return (f"{stats.mean(vals):8.2f} +/- {stats.stdev(vals):5.2f} "
            f"({len(vals)} seeds)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="xmol-lih-h2o-to-n2")
    ap.add_argument("--project", default="gqe-for-qsci")
    ap.add_argument("--entity", default=None)
    args = ap.parse_args()

    try:
        import wandb
    except ImportError:
        sys.exit("wandb not installed - run this inside the container, or "
                 "`pip install wandb`.")

    api = wandb.Api()
    path = f"{args.entity + '/' if args.entity else ''}{args.project}"
    runs = list(api.runs(path, filters={"group": args.group}))
    if not runs:
        sys.exit(f"No runs found in group '{args.group}' of project '{path}'. "
                 "Did you sync the offline runs?")

    # collected[stage][kind] -> list of errors in mHa
    collected = {label: {"zero-shot": [], "baseline": []} for label, _, _ in STAGES}
    rows = []

    for r in sorted(runs, key=lambda r: r.name):
        s = dict(r.summary)
        mol = eval_molecule(s)
        is_zeroshot = mol is not None
        kind = "zero-shot" if is_zeroshot else "baseline"

        errs = {}
        for label, zs_key, bl_key in STAGES:
            key = zs_key.format(mol=mol) if is_zeroshot else bl_key
            e = mha(s, key)
            errs[label] = e
            if e is not None:
                collected[label][kind].append(e)
        rows.append((r.name, kind, r.state, errs))

    print(f"\ngroup: {args.group}   ({len(runs)} runs)\n")
    header = f"{'run':<26}{'kind':<11}{'state':<10}"
    header += "".join(f"{label:>17}" for label, _, _ in STAGES)
    print(header)
    print("-" * len(header))
    for name, kind, state, errs in rows:
        line = f"{name:<26}{kind:<11}{state:<10}"
        for label, _, _ in STAGES:
            e = errs[label]
            line += f"{(f'{e:.2f} mHa' if e is not None else '-'):>17}"
        print(line)

    print("\n" + "=" * len(header))
    print("RESULT — error vs N2's CASCI, mHa (lower is better)")
    print("=" * len(header))
    for label, _, _ in STAGES:
        zs, bl = collected[label]["zero-shot"], collected[label]["baseline"]
        print(f"\n  {label}")
        print(f"    baseline  (trained on N2) : {fmt(bl)}")
        print(f"    zero-shot (never saw N2)  : {fmt(zs)}")
        if zs and bl:
            gap = stats.mean(zs) - stats.mean(bl)
            print(f"    generalization gap        : {gap:+8.2f}   "
                  "(zero-shot minus baseline; <= 0 means transfer matched "
                  "training on N2)")

    print(f"\n  chemical accuracy = {CHEMICAL_ACCURACY_MHA} mHa")
    if any(len(v["zero-shot"]) + len(v["baseline"]) <= 2
           for v in collected.values()):
        print("\n  NOTE: single-seed numbers carry no error bar and should be "
              "read as directional only.")
    print("  NOTE: Global-refined enlarges the subspace up to qsci.max_dim. "
          "When it\n        saturates that cap the classical refinement is "
          "doing part of the work,\n        so read it alongside the "
          "GQE-optimized row rather than instead of it.")


if __name__ == "__main__":
    main()
