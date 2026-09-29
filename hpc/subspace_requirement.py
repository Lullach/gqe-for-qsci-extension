"""
How much of the CI space does chemical accuracy actually need?

For each system, solve the full CI problem, sort determinants by |c|, and bisect
for the smallest subspace whose energy is within 1.6 mHa of exact. That is the
subspace size QSCI would need IN THE BEST CASE, because QSCI's sampled
determinants are essentially |c|^2-distributed -- top-n by |c| is what sampling
converges to.

This is the question the H10 result raised. Everything there landed at ~30 mHa
regardless of architecture, circuit length or Pauli convention, and the oracle
and HCI sat at ~30-40 mHa too, so the 5% cap was the limit rather than the
policy. The useful follow-up is not "make the policy better" but "for which
systems is a tractable subspace enough at all?"

Two caveats that must travel with the numbers:

  * The requirement is an UPPER bound. Top-n by |c| maximises OVERLAP with the
    exact state, not energy; the energy value of a determinant is roughly
    |c_D|^2 * (H_DD - E), so a cleverer selection (HCI targets exactly that) may
    reach the same accuracy with fewer determinants.
  * It needs the exact answer, so it is a characterisation of the systems, not a
    method. Its value is telling you where QSCI can work before spending GPU
    hours finding out.

CPU only (pyscf + pyci, no cudaq). Runs on a laptop; the cost is one full CI
solve plus ~log2(N) diagonalizations per system, so keep active spaces modest.

    python3 hpc/subspace_requirement.py
    python3 hpc/subspace_requirement.py --systems h4,h6,h8 --target 1.6
    python3 hpc/subspace_requirement.py --out subspace_requirement.csv
"""

import argparse
import csv
import os
import sys
import time

REPO = os.environ.get("REPO") or (
    "/workspace" if os.path.isdir("/workspace/gqe_qsci")
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, REPO)

from gqe_qsci.molecule import PySCFMolecule                      # noqa: E402
from gqe_qsci.qsci.baseline import (                             # noqa: E402
    build_pyci_hamiltonian, minimal_subspace,
)


class G:
    """Minimal stand-in for the Hydra geometry node PySCFMolecule expects."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def chain(n, r):
    return G(type="linear_chain", atoms=["H"] * n, bond_length=r)


# (label, geometry, nelecas, norbcas). Chain length varies the size, bond length
# varies the correlation strength -- the two axes that should drive the answer.
def default_systems():
    out = []
    for n in (4, 6, 8, 10):
        for r in (0.7, 1.0, 1.4, 1.8, 2.4, 3.0):
            out.append((f"h{n}_r{r:.1f}", chain(n, r), n, n))
    out.append(("lih_r1.6", G(type="linear_chain", atoms=["Li", "H"],
                              bond_length=1.6), 2, 5))
    out.append(("n2_r1.1", G(type="linear_chain", atoms=["N", "N"],
                             bond_length=1.1), 10, 8))
    out.append(("n2_r2.5", G(type="linear_chain", atoms=["N", "N"],
                             bond_length=2.5), 10, 8))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=1.6,
                    help="accuracy to reach, mHa (default: chemical accuracy)")
    ap.add_argument("--systems", default=None,
                    help="comma-separated labels to keep, e.g. h4,h6,n2_r2.5")
    ap.add_argument("--max-fci", type=int, default=250_000,
                    help="skip systems whose CI space exceeds this; the full "
                         "solve is the expensive step")
    ap.add_argument("--out", default="subspace_requirement.csv")
    ap.add_argument("--basis", default="sto-3g")
    args = ap.parse_args()

    systems = default_systems()
    if args.systems:
        keep = {s.strip() for s in args.systems.split(",")}
        systems = [s for s in systems
                   if s[0] in keep or s[0].split("_")[0] in keep]
        if not systems:
            sys.exit(f"no systems matched {args.systems!r}")

    print(f"target accuracy: {args.target} mHa")
    print(f"{len(systems)} system(s); writing {args.out}")
    print()
    header = (f"{'system':<12}{'qubits':>7}{'CI space':>11}{'n needed':>10}"
              f"{'fraction':>10}{'err(mHa)':>10}{'secs':>7}")
    print(header)
    print("-" * len(header))

    new = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    with open(args.out, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(["system", "basis", "nelecas", "norbcas", "qubits",
                             "n_fci", "n_needed", "fraction", "err_mha",
                             "target_mha", "fci_energy", "seconds"])

        for label, geom, nelecas, norbcas in systems:
            t0 = time.time()
            try:
                mol = PySCFMolecule(geometry=geom, basis=args.basis,
                                    nelecas=nelecas, norbcas=norbcas,
                                    spin=0, charge=0)
                n_fci = mol.n_determinants
                if n_fci > args.max_fci:
                    print(f"{label:<12}{norbcas*2:>7}{n_fci:>11,}"
                          f"{'skipped (> --max-fci)':>37}")
                    continue
                ham = build_pyci_hamiltonian(mol)
                n, frac, err, fci_e, n_fci = minimal_subspace(
                    ham, tuple(int(x) for x in mol.nelec), target_mha=args.target)
            except Exception as exc:                              # noqa: BLE001
                print(f"{label:<12}  FAILED: {type(exc).__name__}: {exc}")
                continue

            secs = time.time() - t0
            print(f"{label:<12}{norbcas*2:>7}{n_fci:>11,}{n:>10,}"
                  f"{100*frac:>9.2f}%{err:>10.3f}{secs:>7.0f}")
            writer.writerow([label, args.basis, nelecas, norbcas, norbcas * 2,
                             n_fci, n, f"{frac:.6f}", f"{err:.4f}",
                             args.target, f"{fci_e:.10f}", f"{secs:.1f}"])
            fh.flush()          # a killed run keeps every system it finished

    print()
    print("  fraction = smallest top-|c| subspace reaching the target, as a")
    print("  share of the full CI space. It is an UPPER bound on the requirement:")
    print("  top-|c| is overlap-optimal, not energy-optimal, so a cleverer")
    print("  selection might need fewer. It is the right number for QSCI anyway,")
    print("  since |c|^2-distributed sampling is what QSCI converges to.")


if __name__ == "__main__":
    main()
