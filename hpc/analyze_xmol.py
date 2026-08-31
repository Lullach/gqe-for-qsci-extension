"""
Summarise the Phase 4 cross-molecule experiment across seeds.

Answers the one question the experiment exists to answer:

    how much of trained-on-N2 performance does a policy that never saw N2
    recover, and how much does that vary across seeds?

Reads the W&B group (default: xmol-lih-h2o-to-n2), splits runs into
  zero-shot : xmol_dag_gnn        -> zeroshot/n2/... (N2 never trained on)
  baseline  : xmol_baseline_n2    -> trained directly on N2
and reports each as an error against N2's own CASCI, in mHa, mean +/- sd.

Usage (needs `wandb login` and the runs SYNCED):
    python3 hpc/analyze_xmol.py
    python3 hpc/analyze_xmol.py --group xmol-lih-h2o-to-n2 --entity <you>
"""

import argparse
import statistics as stats
import sys

HA_TO_MHA = 1000.0

# Metric candidates, best first — the exact key depends on which refinement
# stage logged last, so fall back rather than silently reporting nothing.
ZEROSHOT_KEYS = [
    "zeroshot/n2/Global-refined(best_so_far)/energy",
    "zeroshot/n2/GQE-optimized(best_so_far)/energy",
    "zeroshot/n2/GQE-optimized/energy/min",
]
BASELINE_KEYS = [
    "Global-refined(best_so_far)/energy",
    "GQE-optimized(best_so_far)/energy",
    "GQE-optimized/energy/min",
]
CASCI_KEYS = ["R-CASCI", "reference/R-CASCI"]


def pick(summary, keys):
    for k in keys:
        v = summary.get(k)
        if v is not None:
            try:
                return float(v), k
            except (TypeError, ValueError):
                continue
    return None, None


def fmt(vals, unit="mHa"):
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.2f} {unit}  (1 seed)"
    return (f"{stats.mean(vals):.2f} +/- {stats.stdev(vals):.2f} {unit}"
            f"  ({len(vals)} seeds)")


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
    runs = [r for r in api.runs(path, filters={"group": args.group})]
    if not runs:
        sys.exit(f"No runs found in group '{args.group}' of project '{path}'. "
                 "Did you sync the offline runs?")

    print(f"\ngroup: {args.group}   ({len(runs)} runs)\n")
    print(f"{'run':<28} {'kind':<10} {'state':<10} {'err vs CASCI':>14}")
    print("-" * 66)

    zs, bl, casci_seen = [], [], set()
    for r in sorted(runs, key=lambda r: r.name):
        s = dict(r.summary)
        is_zeroshot = "zeroshot" in r.tags or r.name.startswith("xmol-dag-gnn")
        energy, _ = pick(s, ZEROSHOT_KEYS if is_zeroshot else BASELINE_KEYS)
        casci, _ = pick(s, CASCI_KEYS)

        err = None
        if energy is not None and casci is not None:
            err = (energy - casci) * HA_TO_MHA
            (zs if is_zeroshot else bl).append(err)
            casci_seen.add(round(casci, 6))

        kind = "zero-shot" if is_zeroshot else "baseline"
        print(f"{r.name:<28} {kind:<10} {r.state:<10} "
              f"{(f'{err:.2f} mHa' if err is not None else '-'):>14}")

    print("\n" + "=" * 66)
    print("RESULT")
    print("=" * 66)
    print(f"  baseline  (trained on N2) : {fmt(bl)}")
    print(f"  zero-shot (never saw N2)  : {fmt(zs)}")
    if zs and bl:
        gap = stats.mean(zs) - stats.mean(bl)
        print(f"\n  generalization gap        : {gap:+.2f} mHa")
        print("    (zero-shot minus baseline; smaller is better transfer,")
        print("     <= 0 would mean transfer matched or beat training on N2)")
    if len(casci_seen) > 1:
        print(f"\n  WARNING: runs disagree on N2 CASCI {sorted(casci_seen)} - "
              "are they really the same molecule/active space?")
    print("\n  chemical accuracy = 1.6 mHa")


if __name__ == "__main__":
    main()
