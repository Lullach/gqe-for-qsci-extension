"""
Figures from the per-circuit CSVs, without re-running anything.

Every run writes `<output>/results.csv`, one row per evaluated circuit (see
gqe_qsci/results_log.py). This globs them into one dataframe and emits the
default figures. Adding a new figure is a function here plus one line in FIGURES
— no job submission, no W&B panel fiddling, and the output is PDF, so it goes
straight into the thesis.

    python3 hpc/plot_results.py                         # every run under outputs/
    python3 hpc/plot_results.py --glob "outputs/gqe-for-qsci/hchain-*"
    python3 hpc/plot_results.py --csv        # just dump the merged table
    python3 hpc/plot_results.py --out figures/

Columns: exp_tag, model, seed, epoch, molecule, split, stage, sample_idx,
energy, R-CASCI, R-CCSD, subspace_dim, num_sampled_basis,
num_symmetry_preserving_basis, cx_count, total_gates, seq.

`stage` is "GQE-optimized" for every sampled circuit of a rollout
(sample_idx >= 0) and "<...>(best_so_far)" for the running bests
(sample_idx == -1). Errors are not precomputed: the reference energies are
columns, so pick your own convention.
"""

import argparse
import glob
import os
import sys

HA_TO_MHA = 1000.0
CHEMICAL_ACCURACY_MHA = 1.6


def load(pattern):
    import pandas as pd

    paths = sorted(glob.glob(os.path.join(pattern, "results.csv")))
    if not paths:
        paths = sorted(glob.glob(pattern))
    paths = [p for p in paths if p.endswith(".csv")]
    if not paths:
        sys.exit(f"no results.csv found under {pattern!r}. Runs started before "
                 "results_log.py landed will not have one.")

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["run_dir"] = os.path.dirname(p)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["err_mha"] = (df["energy"] - df["R-CASCI"]) * HA_TO_MHA
    print(f"loaded {len(paths)} run(s), {len(df):,} rows, "
          f"{df['exp_tag'].nunique()} exp_tag(s), {df['model'].nunique()} model(s)")
    return df


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def fig_best_vs_epoch(df, ax):
    """Best-so-far error against epoch, one line per (model, stage), across seeds.

    This is the figure that shows whether a policy is still improving. On the
    H10 baseline the best-so-far metrics flattened around step 143 of 540, which
    is how the stall was spotted."""
    best = df[df["sample_idx"] == -1]
    if best.empty:
        return False
    for (model, stage), g in best.groupby(["model", "stage"]):
        m = g.groupby("epoch")["err_mha"].mean()
        ax.plot(m.index, m.values, label=f"{model} / {stage}", lw=1.4)
    ax.axhline(CHEMICAL_ACCURACY_MHA, ls="--", c="k", lw=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("error vs CASCI (mHa)")
    ax.set_yscale("log")
    ax.set_title("best-so-far error (mean over seeds)")
    ax.legend(fontsize=6)
    return True


def fig_subspace_vs_epoch(df, ax):
    """Subspace dimension against epoch.

    Shows two things at once: whether the classical refinement is saturating the
    qsci.coverage cap, and whether the raw subspace is bounded by 2^L (a circuit
    of L gates can populate at most 2^L determinants)."""
    best = df[df["sample_idx"] == -1]
    if best.empty or best["subspace_dim"].isna().all():
        return False
    for (model, stage), g in best.groupby(["model", "stage"]):
        m = g.groupby("epoch")["subspace_dim"].mean()
        ax.plot(m.index, m.values, label=f"{model} / {stage}", lw=1.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("subspace dim")
    ax.set_title("subspace size (flat = saturated cap, or no new best)")
    ax.legend(fontsize=6)
    return True


def fig_sample_spread(df, ax):
    """Spread of the rollout batch per epoch — is the policy still exploring?

    Only possible because every sampled circuit is written, not just the best."""
    batch = df[df["sample_idx"] >= 0]
    if batch.empty:
        return False
    g = batch.groupby("epoch")["err_mha"]
    lo, mid, hi = g.min(), g.median(), g.max()
    ax.fill_between(mid.index, lo.values, hi.values, alpha=0.25, label="min-max")
    ax.plot(mid.index, mid.values, lw=1.4, label="median")
    ax.axhline(CHEMICAL_ACCURACY_MHA, ls="--", c="k", lw=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("error vs CASCI (mHa)")
    ax.set_yscale("log")
    ax.set_title("rollout batch spread (collapse = exploration died)")
    ax.legend(fontsize=6)
    return True


def fig_final_by_model(df, ax):
    """Final best error per model, points = seeds. The architecture comparison."""
    best = df[(df["sample_idx"] == -1) & df["stage"].str.contains("best_so_far")]
    if best.empty:
        return False
    final = (best.sort_values("epoch")
                 .groupby(["model", "stage", "seed"])["err_mha"].last()
                 .reset_index())
    labels, positions = [], []
    for i, ((model, stage), g) in enumerate(final.groupby(["model", "stage"])):
        ax.scatter([i] * len(g), g["err_mha"], s=18)
        labels.append(f"{model}\n{stage}")
        positions.append(i)
    ax.axhline(CHEMICAL_ACCURACY_MHA, ls="--", c="k", lw=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=5, rotation=30, ha="right")
    ax.set_ylabel("final error vs CASCI (mHa)")
    ax.set_yscale("log")
    ax.set_title("final error per model (one point per seed)")
    return True


FIGURES = [
    ("best_vs_epoch", fig_best_vs_epoch),
    ("subspace_vs_epoch", fig_subspace_vs_epoch),
    ("sample_spread", fig_sample_spread),
    ("final_by_model", fig_final_by_model),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/gqe-for-qsci/*",
                    help="run directories (each containing results.csv), or a "
                         "glob matching csv files directly")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--csv", action="store_true",
                    help="write the merged table and exit, for your own analysis")
    args = ap.parse_args()

    df = load(args.glob)

    if args.csv:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "merged_results.csv")
        df.to_csv(path, index=False)
        print(f"wrote {path}")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    for name, fn in FIGURES:
        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            ok = fn(df, ax)
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
