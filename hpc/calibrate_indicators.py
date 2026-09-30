"""
Calibrate the subspace-size indicators against exact answers.

For each system, take the exact FCI vector, form the top-n subspace by |c| for a
log-spaced range of n, re-diagonalize, and record three things:

    err_mha        E(n) - E_fci, the TRUE truncation error (needs the exact answer)
    boundary_mha   energy lost dropping the bottom 10% of the subspace (live)
    tail_weight    |c|^2 carried by that bottom 10% (live, free)

The question is whether the two live quantities track the true error well
enough to say "this subspace is too small for chemical accuracy" without
knowing the exact answer — and if so, at what threshold.

Top-|c| subspaces are the best case of what QSCI builds (its samples are
|c|^2-distributed), so this is the right regime to calibrate in; the replay of
actual training runs (hpc/replay_indicators.py) then checks the calibration on
real QSCI subspaces.

CPU only. In the container:

    docker run --rm --entrypoint /bin/bash -v "${PWD}:/workspace" -w /workspace \
        gqe_qsci_cpu -lc "python3 hpc/calibrate_indicators.py"
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

from gqe_qsci.molecule import PySCFMolecule                       # noqa: E402
from gqe_qsci.qsci.baseline import (                              # noqa: E402
    _full_ci_ordered, build_pyci_hamiltonian,
)
from gqe_qsci.qsci.diagnostics import boundary_indicators         # noqa: E402
from subspace_requirement import G                                # noqa: E402

import pyci                                                       # noqa: E402

# (label, atoms, (nelecas, norbcas), bond length, basis). h10_r1.60 is the
# molecule every architecture-comparison run trained on, so it comes first.
SYSTEMS = [
    ("h10_r1.60", ["H"] * 10, (10, 10), 1.6, "sto-3g"),
    ("h8_r1.60",  ["H"] * 8,  (8, 8),   1.6, "sto-3g"),
    ("h8_r0.90",  ["H"] * 8,  (8, 8),   0.9, "sto-3g"),
    ("n2_r1.10",  ["N", "N"], (10, 8),  1.1, "sto-3g"),
    ("n2_r2.00",  ["N", "N"], (10, 8),  2.0, "sto-3g"),
    ("co_r1.13",  ["C", "O"], (10, 8),  1.13, "sto-3g"),
    ("lif_r1.60", ["Li", "F"], (8, 8),  1.6, "sto-3g"),
    ("c2_r1.25",  ["C", "C"], (8, 8),   1.25, "sto-3g"),
]


def solve_with_vector(ham, dets, nelec, max_cycle):
    wfn = pyci.fullci_wfn(ham.nbasis, *nelec)
    for d in dets:
        wfn.add_det(d)
    energies, coeffs = pyci.sparse_op(ham, wfn).solve(maxiter=max_cycle)
    return float(energies[0]), coeffs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=24,
                    help="subspace sizes per system, log-spaced")
    ap.add_argument("--out", default="data/subspace/indicator_calibration.csv")
    ap.add_argument("--systems", default=None, help="comma-separated labels")
    ap.add_argument("--max-cycle", type=int, default=1000)
    args = ap.parse_args()

    want = {s.strip() for s in args.systems.split(",")} if args.systems else None
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["system", "n_fci", "n", "fraction", "err_mha",
                         "boundary_mha", "tail_weight", "seconds"])
        for label, atoms, (ne, no), r, basis in SYSTEMS:
            if want and label not in want:
                continue
            t0 = time.time()
            mol = PySCFMolecule(
                geometry=G(type="linear_chain", atoms=atoms, bond_length=r),
                basis=basis, nelecas=ne, norbcas=no, spin=0, charge=0)
            nelec = tuple(int(x) for x in mol.nelec)
            ham = build_pyci_hamiltonian(mol)
            ordered, e_fci = _full_ci_ordered(ham, nelec, args.max_cycle,
                                              selfcheck=False)
            n_fci = len(ordered)
            sizes = np.unique(np.round(np.geomspace(
                20, n_fci, args.points)).astype(int))

            print(f"\n{label}  ({n_fci:,} determinants)")
            print(f"{'n':>8}{'frac':>8}{'err':>10}{'boundary':>10}"
                  f"{'tail_w':>11}")
            for n in sizes:
                subset = ordered[:n]
                e, c = solve_with_vector(ham, subset, nelec, args.max_cycle)
                ind = boundary_indicators(ham, subset, c, e, nelec,
                                          max_cycle=args.max_cycle)
                err = (e - e_fci) * 1000.0
                print(f"{n:>8,}{n / n_fci:>8.3f}{err:>10.3f}"
                      f"{ind.boundary_mha:>10.3f}{ind.tail_weight:>11.2e}")
                writer.writerow([label, n_fci, n, f"{n / n_fci:.6f}",
                                 f"{err:.6f}", f"{ind.boundary_mha:.6f}",
                                 f"{ind.tail_weight:.6e}",
                                 f"{time.time() - t0:.1f}"])
                fh.flush()


if __name__ == "__main__":
    main()
