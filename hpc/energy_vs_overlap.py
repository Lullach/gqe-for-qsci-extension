"""
How much does selecting determinants by OVERLAP cost against selecting by ENERGY?

Every requirement number in data/subspace/ comes from top-|c| selection: keep the
determinants with the largest coefficients. That is overlap-optimal — it provably
maximises the retained sum of |c|^2 — and it is the right model for QSCI, whose
sampled determinants are |c|^2-distributed. But it is NOT energy-optimal, and the
energy is what we actually report. Epstein-Nesbet puts the energy a determinant
contributes at

    e_D  ~=  |c_D|^2 * (H_DD - E)

so a determinant with a large coefficient sitting almost at the correlated energy
buys little, while a smaller coefficient high above it can buy a lot. That extra
gap factor is exactly what heat-bath CI targets.

This script runs the SAME bisection twice on the same exact vector, once with
determinants ordered by |c| and once by e_D, and reports the ratio. The ratio
answers a question the rest of the study leans on: is the oracle a tight bound,
so we can keep quoting it, or is it loose enough that "the requirement" depends
on who is selecting?

Both orderings use the exact FCI vector. Neither is a practical method — the
point is to bound the gap between selection RULES, with everything else held
fixed. The true optimum is combinatorial over C(N, n) subsets and is not
computed here, so this measures a gap, not the best possible one.

Cost is ~2 * log2(N_FCI) diagonalizations per configuration, so --max-fci is
deliberately low. CPU only (pyscf + pyci); locally that means the container:

    docker run --rm --entrypoint /bin/bash -v "${PWD}:/workspace" -w /workspace \
        gqe_qsci_cpu -lc "python3 hpc/energy_vs_overlap.py"

Resumable: appended and flushed per configuration, finished ones are skipped.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

REPO = os.environ.get("REPO") or (
    "/workspace" if os.path.isdir("/workspace/gqe_qsci")
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "hpc"))

import pyci                                                       # noqa: E402

from gqe_qsci.molecule import PySCFMolecule                       # noqa: E402
from gqe_qsci.qsci.baseline import (                              # noqa: E402
    _determinants_of, bisect_minimal, build_pyci_hamiltonian, diagonalize,
)
from entropy_scan import diagonal                                 # noqa: E402
from subspace_requirement import G, configurations                # noqa: E402


COLUMNS = ["config", "family", "bond_length", "basis", "nelecas", "norbcas",
           "n_fci", "n_overlap", "n_energy", "ratio", "frac_overlap",
           "frac_energy", "err_overlap", "err_energy", "fci_energy", "seconds"]


# CLAUDE
def already_done(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {r["config"] for r in csv.DictReader(f) if r.get("config")}


# CLAUDE
def compare(ham, nelec, target_mha, max_cycle=1000):
    """(n_overlap, err_o, n_energy, err_e, fci_energy, n_fci) for one system."""
    wfn = pyci.fullci_wfn(ham.nbasis, *nelec)
    wfn.add_all_dets()
    op = pyci.sparse_op(ham, wfn)
    energies, coeffs = op.solve(maxiter=max_cycle)
    fci_energy = float(energies[0])

    dets = _determinants_of(wfn)
    c = np.abs(np.asarray(coeffs[0]).ravel())
    if len(dets) != c.size:
        raise RuntimeError(f"{len(dets)} determinants but {c.size} coefficients")

    diag = diagonal(op, len(dets))
    if fci_energy > diag.min() + 1e-6:
        raise RuntimeError("E above min(H_DD); the diagonal is wrong")

    # overlap ranking: |c|. energy ranking: Epstein-Nesbet contribution.
    gap = np.clip(diag - fci_energy, 0.0, None)
    order_overlap = np.argsort(-c)
    order_energy = np.argsort(-(c ** 2 * gap))

    out = []
    for order in (order_overlap, order_energy):
        ordered = [dets[i] for i in order]
        out.append(bisect_minimal(ham, ordered, nelec, fci_energy,
                                  target_mha, max_cycle))
    (n_o, e_o), (n_e, e_e) = out
    return n_o, e_o, n_e, e_e, fci_energy, len(dets)


# CLAUDE
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=1.6)
    ap.add_argument("--families", default=None)
    ap.add_argument("--max-fci", type=int, default=20_000,
                    help="this runs the bisection TWICE per configuration, so "
                         "the ceiling is lower than the main scan's")
    ap.add_argument("--out", default="data/subspace/energy_vs_overlap.csv")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    fams = {s.strip() for s in args.families.split(",")} if args.families else None
    configs = configurations(fams)
    done = set() if args.redo else already_done(args.out)
    todo = [c for c in configs if c[0] not in done]

    print(f"target {args.target} mHa | {len(configs)} configuration(s), "
          f"{len(done)} done, {len(todo)} to run, CI spaces <= {args.max_fci:,}")
    print(f"writing {args.out}\n")
    hdr = (f"{'config':<14}{'CI space':>10}{'overlap':>9}{'energy':>9}"
           f"{'ratio':>8}{'saving':>9}{'secs':>7}")
    print(hdr)
    print("-" * len(hdr))

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    mode = "w" if args.redo else "a"
    new = (args.redo or not os.path.exists(args.out)
           or os.path.getsize(args.out) == 0)

    ratios = []
    with open(args.out, mode, newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(COLUMNS)
            fh.flush()

        for label, fam, atoms, ne, no, r, basis in todo:
            t0 = time.time()
            try:
                mol = PySCFMolecule(
                    geometry=G(type="linear_chain", atoms=atoms, bond_length=r),
                    basis=basis, nelecas=ne, norbcas=no, spin=0, charge=0)
                if mol.n_determinants > args.max_fci:
                    continue
                ham = build_pyci_hamiltonian(mol)
                nelec = tuple(int(x) for x in mol.nelec)
                n_o, e_o, n_e, e_e, fci_e, n_fci = compare(
                    ham, nelec, args.target)
            except Exception as exc:                              # noqa: BLE001
                print(f"{label:<14}  FAILED: {type(exc).__name__}: {exc}")
                continue

            ratio = n_o / n_e if n_e else float("nan")
            secs = time.time() - t0
            ratios.append(ratio)
            print(f"{label:<14}{n_fci:>10,}{n_o:>9,}{n_e:>9,}{ratio:>8.2f}"
                  f"{100 * (1 - n_e / n_o):>8.0f}%{secs:>7.0f}")

            writer.writerow([label, fam, f"{r:.2f}", basis, ne, no, n_fci,
                             n_o, n_e, f"{ratio:.4f}", f"{n_o / n_fci:.8f}",
                             f"{n_e / n_fci:.8f}", f"{e_o:.4f}", f"{e_e:.4f}",
                             f"{fci_e:.10f}", f"{secs:.1f}"])
            fh.flush()

    if ratios:
        rs = sorted(ratios)
        print(f"\nratio n_overlap / n_energy over {len(rs)} configurations:")
        print(f"  min {rs[0]:.2f}   median {rs[len(rs) // 2]:.2f}   "
              f"max {rs[-1]:.2f}   geometric mean "
              f"{float(np.exp(np.mean(np.log(rs)))):.2f}")
        print("\n  > 1 means energy-ranked selection needs FEWER determinants,")
        print("  i.e. the top-|c| oracle overstates the requirement by that factor.")


if __name__ == "__main__":
    main()
