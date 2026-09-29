"""
Figures from hpc/subspace_requirement.py.

Three views of one number -- the smallest fraction of the CI space that reaches
chemical accuracy -- against the three things that might explain it:

  1. ranked        every configuration, sorted by requirement. The spectrum:
                   how wide is the range, and which systems sit where.
  2. bond_length   requirement against geometry, one line per molecule. This is
                   where multireference character should show up: stretched
                   bonds need a larger share of the space.
  3. ci_dimension  requirement against full CI size (log x). The scaling
                   question: does the required FRACTION shrink as systems grow,
                   which is what would make selected CI worthwhile, or not?

Needs matplotlib, so run it wherever you have it (not the ABCI-Q container).

    python3 hpc/plot_subspace_requirement.py
    python3 hpc/plot_subspace_requirement.py --csv subspace_requirement.csv --out figures/
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

CHEMICAL_ACCURACY_MHA = 1.6


# The writer's columns changed while the first scan was running, so a CSV can
# hold more than one layout. Keyed by field count; the label is always
# "<family>_r<bond length>", so family and bond_length are recoverable even from
# the oldest rows.
SCHEMAS = {
    12: ["config", "basis", "nelecas", "norbcas", "qubits", "n_fci", "n_needed",
         "fraction", "err_mha", "target_mha", "fci_energy", "seconds"],
    15: ["config", "family", "atoms", "bond_length", "basis", "nelecas",
         "norbcas", "qubits", "n_fci", "n_needed", "fraction", "err_mha",
         "target_mha", "fci_energy", "seconds"],
    16: ["config", "family", "atoms", "bond_length", "basis", "nelecas",
         "norbcas", "qubits", "n_fci", "n_needed", "fraction", "err_mha",
         "target_mha", "fci_energy", "selfchecked", "seconds"],
}


def load(pattern):
    """
    Load one CSV or a glob of them. The cluster writes one file per family
    (concurrent appends to a shared file would interleave rows mid-line), so
    merging here is the normal case, not a special one.
    """
    paths = sorted(glob.glob(pattern)) or ([pattern] if os.path.exists(pattern) else [])
    if not paths:
        sys.exit(f"no CSV matched {pattern!r} -- run hpc/subspace_requirement.py first.")

    rows, skipped = [], defaultdict(int)
    raw_lines = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            raw_lines.extend(csv.reader(f))
    if True:
        for raw in raw_lines:
            if not raw or raw[0] in ("config", "system"):
                continue                                  # header, any version
            fields = SCHEMAS.get(len(raw))
            if fields is None:
                skipped[f"unknown layout ({len(raw)} fields)"] += 1
                continue
            row = dict(zip(fields, raw))

            # family / bond length are recoverable from the label in every layout
            label = row.get("config", "")
            if "_r" in label:
                fam, _, blen = label.rpartition("_r")
                row.setdefault("family", fam)
                row.setdefault("bond_length", blen)
            try:
                row["fraction"] = float(row["fraction"])
                row["pct"] = 100.0 * row["fraction"]
                row["bond_length"] = float(row["bond_length"])
                row["n_fci"] = int(row["n_fci"])
                row["n_needed"] = int(row["n_needed"])
                row["qubits"] = int(row["qubits"])
            except (KeyError, ValueError, TypeError) as exc:
                skipped[f"{type(exc).__name__}: {exc}"] += 1
                continue
            rows.append(row)

    if skipped:
        print("skipped rows:")
        for reason, n in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>4}  {reason}")
    if not rows:
        sys.exit(f"{path} produced no usable rows (see the reasons above).")

    # a label can appear under two layouts if a scan was rerun; keep the last
    dedup = {r["config"]: r for r in rows}
    if len(dedup) != len(rows):
        print(f"note: {len(rows) - len(dedup)} duplicate config label(s); kept the last")
    rows = list(dedup.values())

    fams = sorted({r["family"] for r in rows})
    print(f"loaded {len(rows)} configuration(s) from {len(paths)} file(s), "
          f"{len(fams)} famil(ies): {', '.join(fams)}")
    return rows



def _colors(plt, families):
    cmap = plt.get_cmap("tab20")
    return {fam: cmap(i % 20) for i, fam in enumerate(sorted(families))}


def fig_ranked(rows, plt, colors, path):
    """Requirement spectrum. Sorted, because an unsorted index conveys nothing."""
    ordered = sorted(rows, key=lambda r: r["pct"])
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, r in enumerate(ordered):
        ax.scatter(i, r["pct"], color=colors[r["family"]], s=26, zorder=3)
    seen = set()
    handles = []
    for r in ordered:
        if r["family"] not in seen:
            seen.add(r["family"])
            handles.append(plt.Line2D([], [], marker="o", ls="", ms=5,
                                      color=colors[r["family"]], label=r["family"]))
    ax.set(xlabel="configuration (sorted by requirement)",
           ylabel="% of CI space for chemical accuracy", yscale="log",
           title="how much of the CI space chemical accuracy needs")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(handles=handles, fontsize=6, ncol=2, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=150)
    plt.close(fig)


def fig_bond_length(rows, plt, colors, path):
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for fam, rs in sorted(by_fam.items()):
        rs = sorted(rs, key=lambda r: r["bond_length"])
        ax.plot([r["bond_length"] for r in rs], [r["pct"] for r in rs],
                marker="o", ms=3.5, lw=1.3, color=colors[fam], label=fam)
    ax.set(xlabel="bond length (A)", ylabel="% of CI space for chemical accuracy",
           yscale="log",
           title="requirement vs geometry (stretching = multireference character)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=150)
    plt.close(fig)


def fig_ci_dimension(rows, plt, colors, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for fam in sorted({r["family"] for r in rows}):
        rs = [r for r in rows if r["family"] == fam]
        axes[0].scatter([r["n_fci"] for r in rs], [r["pct"] for r in rs],
                        color=colors[fam], s=26, label=fam, zorder=3)
        axes[1].scatter([r["n_fci"] for r in rs], [r["n_needed"] for r in rs],
                        color=colors[fam], s=26, label=fam, zorder=3)
    axes[0].set(xlabel="full CI dimension", xscale="log", yscale="log",
                ylabel="% of CI space needed",
                title="required FRACTION vs system size")
    axes[1].set(xlabel="full CI dimension", xscale="log", yscale="log",
                ylabel="determinants needed",
                title="required COUNT vs system size")
    lims = [min(r["n_fci"] for r in rows), max(r["n_fci"] for r in rows)]
    axes[1].plot(lims, lims, ls="--", c="k", lw=0.8, label="= full CI")
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
    axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="subspace_*.csv",
                    help="one CSV or a glob; the cluster writes one "
                         "file per family, so a glob is normal")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    rows = load(args.csv)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not available here; run this where it is installed.")

    os.makedirs(args.out, exist_ok=True)
    colors = _colors(plt, {r["family"] for r in rows})

    fig_ranked(rows, plt, colors, os.path.join(args.out, "requirement_ranked"))
    fig_bond_length(rows, plt, colors, os.path.join(args.out, "requirement_vs_bond"))
    fig_ci_dimension(rows, plt, colors, os.path.join(args.out, "requirement_vs_cidim"))

    pcts = sorted(r["pct"] for r in rows)
    print(f"\nrequirement: min {pcts[0]:.3f}%  median {pcts[len(pcts)//2]:.2f}%  "
          f"max {pcts[-1]:.2f}%")
    worst = max(rows, key=lambda r: r["pct"])
    best = min(rows, key=lambda r: r["pct"])
    print(f"  easiest: {best['config']}  ({best['pct']:.3f}% = "
          f"{best['n_needed']:,} of {best['n_fci']:,})")
    print(f"  hardest: {worst['config']} ({worst['pct']:.2f}% = "
          f"{worst['n_needed']:,} of {worst['n_fci']:,})")
    print(f"\nfigures in {args.out}/")


if __name__ == "__main__":
    main()
