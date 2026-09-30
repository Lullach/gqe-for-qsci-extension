"""
How much of the CI space does chemical accuracy actually need?

For each configuration, solve the full CI problem, sort determinants by |c|, and
bisect for the smallest subspace whose energy is within 1.6 mHa of exact. That is
the subspace size QSCI would need IN THE BEST CASE, because QSCI's sampled
determinants are essentially |c|^2-distributed -- top-n by |c| is what sampling
converges to.

This is the question the H10 architecture comparison raised. Everything there
landed at ~30 mHa regardless of architecture, circuit length or Pauli convention,
with the oracle and HCI at ~30-40 too, so the 5% subspace cap was the limit
rather than the policy. The useful follow-up is not "make the policy better" but
"for which systems is a tractable subspace enough at all?"

Two caveats that must travel with the numbers:

  * The requirement is an UPPER bound. Top-n by |c| maximises OVERLAP with the
    exact state, not energy; the energy value of a determinant is roughly
    |c_D|^2 * (H_DD - E), so a cleverer selection (HCI targets exactly that) may
    reach the same accuracy with fewer determinants.
  * It needs the exact answer, so it characterises systems rather than being a
    method. Its value is telling you where QSCI can work before spending GPU
    hours finding out.

Active spaces are FULL VALENCE unless a system specifies otherwise, so "fraction
of the CI space" means the same thing across systems.

CPU ONLY (pyci + pyscf, no cudaq, no GPU). On ABCI-Q submit to rt_QC, not rt_QG:
a GPU queue would bill GPU rates for an idle GPU. See hpc/jobs/subspace.sh.

Resumable: the CSV is appended and flushed per configuration, and configurations
already present are skipped.

    python3 hpc/subspace_requirement.py --max-fci 20000     # the quick 155
    python3 hpc/subspace_requirement.py --families h10
    python3 hpc/subspace_requirement.py --families h12,c2h2,co2,n2b --max-fci 1000000
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


def _r(lo, hi, n):
    step = (hi - lo) / (n - 1)
    return [round(lo + i * step, 2) for i in range(n)]


# (family, atoms, (nelecas, norbcas), bond lengths, basis)
#
# Full valence in STO-3G. Per atom, valence electrons / valence orbitals:
#   H 1/1, Li 1/4, Be 2/4, B 3/4, C 4/4, N 5/4, O 6/4, F 7/4.
SYSTEMS = [
    # --- hydrogen chains: size and correlation on one clean axis -----------
    ("h4",   ["H"] * 4,             (4, 4),   _r(0.6, 3.4, 12), "sto-3g"),
    ("h6",   ["H"] * 6,             (6, 6),   _r(0.6, 3.4, 12), "sto-3g"),
    ("h8",   ["H"] * 8,             (8, 8),   _r(0.6, 3.4, 10), "sto-3g"),
    ("h10",  ["H"] * 10,            (10, 10), _r(0.7, 3.1, 5),  "sto-3g"),
    # --- diatomics, weakly to strongly correlated --------------------------
    ("h2",   ["H", "H"],            (2, 2),   _r(0.5, 3.0, 10), "sto-3g"),
    ("lih",  ["Li", "H"],           (2, 5),   _r(1.0, 4.0, 10), "sto-3g"),
    ("bh",   ["B", "H"],            (4, 5),   _r(0.8, 3.2, 10), "sto-3g"),
    ("hf",   ["H", "F"],            (8, 5),   _r(0.7, 2.8, 10), "sto-3g"),
    ("li2",  ["Li", "Li"],          (2, 8),   _r(2.0, 6.0, 9),  "sto-3g"),
    ("f2",   ["F", "F"],            (14, 8),  _r(1.0, 3.2, 9),  "sto-3g"),
    ("n2",   ["N", "N"],            (10, 8),  _r(0.8, 3.2, 10), "sto-3g"),
    ("co",   ["C", "O"],            (10, 8),  _r(0.9, 3.0, 9),  "sto-3g"),
    ("c2",   ["C", "C"],            (8, 8),   _r(1.0, 3.2, 9),  "sto-3g"),
    ("lif",  ["Li", "F"],           (8, 8),   _r(1.2, 4.0, 9),  "sto-3g"),
    ("beo",  ["Be", "O"],           (8, 8),   _r(1.0, 3.0, 9),  "sto-3g"),
    ("bn",   ["B", "N"],            (8, 8),   _r(1.0, 3.0, 9),  "sto-3g"),
    ("beh2", ["H", "Be", "H"],      (4, 6),   _r(1.0, 2.8, 8),  "sto-3g"),

    # --- LARGER: cluster only (raise --max-fci). Cost tracks the CI
    #     DIMENSION, not the qubit count.
    #
    #   h12   extends the H-chain series to 24 qubits: the SIZE axis, on the
    #         family that already has four points.
    #   c2h2  another 20-qubit system with completely different chemistry from
    #         H10 - does the requirement follow size, or bonding?
    #   co2   a real polyatomic, 24 qubits, multiple bonds.
    #   n2b   N2 again, but 6-31G with a LARGER active space. Same chemistry,
    #         bigger space: isolates "does the required fraction shrink with
    #         dimension" from "does chemistry matter".
    ("h12",  ["H"] * 12,            (12, 12), [1.0, 1.6, 2.4],  "sto-3g"),
    ("c2h2", ["H", "C", "C", "H"],  (10, 10), _r(1.0, 2.4, 5),  "sto-3g"),
    ("co2",  ["O", "C", "O"],       (16, 12), _r(1.0, 2.4, 4),  "sto-3g"),
    ("n2b",  ["N", "N"],            (10, 12), [1.1, 1.6, 2.5],  "6-31g"),
]


def configurations(families=None):
    out = []
    for fam, atoms, (ne, no), lengths, basis in SYSTEMS:
        if families and fam not in families:
            continue
        for r in lengths:
            out.append((f"{fam}_r{r:.2f}", fam, atoms, ne, no, r, basis))
    return out


def already_done(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {row["config"] for row in csv.DictReader(f) if row.get("config")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=1.6,
                    help="accuracy to reach, mHa (default: chemical accuracy)")
    ap.add_argument("--families", default=None,
                    help="comma-separated families, e.g. h12,c2h2,co2,n2b")
    ap.add_argument("--max-fci", type=int, default=100_000,
                    help="skip configurations whose CI space exceeds this")
    ap.add_argument("--max-selfcheck", type=int, default=100_000,
                    help="skip the all-determinant self-check above this CI "
                         "size. It is a SECOND full CI solve, and the code path "
                         "is identical to the small systems where it passed.")
    ap.add_argument("--out", default="data/subspace/subspace_requirement.csv",
                    help="results CSV; data/subspace/ is tracked in git, "
                         "unlike outputs/")
    ap.add_argument("--redo", action="store_true",
                    help="recompute configurations already in the CSV")
    args = ap.parse_args()

    fams = {s.strip() for s in args.families.split(",")} if args.families else None
    configs = configurations(fams)
    done = set() if args.redo else already_done(args.out)
    todo = [c for c in configs if c[0] not in done]

    print(f"target {args.target} mHa | {len(configs)} configuration(s), "
          f"{len(done)} already done, {len(todo)} to run")
    print(f"writing {args.out}")
    print()
    header = (f"{'config':<14}{'basis':>8}{'qubits':>7}{'CI space':>11}"
              f"{'n needed':>10}{'fraction':>10}{'err(mHa)':>10}{'secs':>8}")
    print(header)
    print("-" * len(header))

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    new = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    with open(args.out, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(["config", "family", "atoms", "bond_length", "basis",
                             "nelecas", "norbcas", "qubits", "n_fci", "n_needed",
                             "fraction", "err_mha", "target_mha", "fci_energy",
                             "selfchecked", "seconds"])
            fh.flush()

        for label, fam, atoms, ne, no, r, basis in todo:
            t0 = time.time()
            try:
                mol = PySCFMolecule(
                    geometry=G(type="linear_chain", atoms=atoms, bond_length=r),
                    basis=basis, nelecas=ne, norbcas=no, spin=0, charge=0)
                n_fci = mol.n_determinants
                if n_fci > args.max_fci:
                    print(f"{label:<14}{basis:>8}{no*2:>7}{n_fci:>11,}"
                          f"   skipped (> --max-fci)")
                    continue
                selfcheck = n_fci <= args.max_selfcheck
                if not selfcheck:
                    print(f"{label:<14}{basis:>8}{no*2:>7}{n_fci:>11,}"
                          f"   (self-check skipped, > --max-selfcheck)")
                ham = build_pyci_hamiltonian(mol)
                n, frac, err, fci_e, n_fci = minimal_subspace(
                    ham, tuple(int(x) for x in mol.nelec),
                    target_mha=args.target, selfcheck=selfcheck)
            except Exception as exc:                              # noqa: BLE001
                print(f"{label:<14}  FAILED: {type(exc).__name__}: {exc}")
                continue

            secs = time.time() - t0
            print(f"{label:<14}{basis:>8}{no*2:>7}{n_fci:>11,}{n:>10,}"
                  f"{100*frac:>9.2f}%{err:>10.3f}{secs:>8.0f}")
            writer.writerow([label, fam, "".join(atoms), f"{r:.2f}", basis,
                             ne, no, no * 2, n_fci, n, f"{frac:.8f}",
                             f"{err:.4f}", args.target, f"{fci_e:.10f}",
                             int(selfcheck), f"{secs:.1f}"])
            fh.flush()          # a killed run keeps every configuration finished

    print()
    print("  fraction = smallest top-|c| subspace reaching the target, as a share")
    print("  of the full CI space. UPPER bound on the requirement: top-|c| is")
    print("  overlap-optimal, not energy-optimal, so a cleverer selection might")
    print("  need fewer. It is the right number for QSCI anyway, since sampling")
    print("  converges to the |c|^2 distribution.")
    print()
    print("  plot it:  python3 hpc/plot_subspace_requirement.py")


if __name__ == "__main__":
    main()
