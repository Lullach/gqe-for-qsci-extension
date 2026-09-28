"""
Fair classical baseline for QSCI: energy vs subspace size for
  - HCI    (heat-bath CI — the classical competitor QSCI must beat)
  - random (number/spin-conserving determinants — the floor)
  - FCI    (exact, the ceiling; == CASCI here since the active space is full CI)

Compare against your QSCI runs: read the QSCI best energy at a given
subspace_dim (e.g. 170) from W&B, and see whether it sits BELOW the HCI curve at
the same size. If not, the quantum sampling isn't selecting better determinants
than a classical heuristic. See NOTES.md, "Value proposition".

Run inside Docker (needs pyscf + pyci; NOT cudaq). Local is fine - no GPU:
    python3 baseline_selection.py molecule=n2
    python3 baseline_selection.py molecule=h10 dims=500,1000,3175,6350,12700,31752

`dims` overrides the default subspace sizes; it is consumed here and NOT passed
to Hydra. The sizes matter more than they look. The H-chain experiment ran H10 at
qsci.coverage=0.05, i.e. 3175 of 63504 determinants, so HCI's error at 3175 is the
FLOOR that experiment could possibly have reached. If the QSCI baseline (32.65 mHa
over 5 seeds) sits close to it, the subspace cap is the limit and the policy is
already near optimal; if HCI reaches single-digit mHa there, the policy is leaving
most of it on the table and raising coverage is not the fix.
"""

import csv
import os
import sys

import numpy as np
from hydra import compose, initialize
from hydra.utils import instantiate

from gqe_qsci.qsci.baseline import (
    build_pyci_hamiltonian, hci_curve, random_curve, fci_dimension,
)

DIMS = [10, 20, 50, 100, 170, 300, 500, 1000]   # 170 = the paper's d_max;
#                                                override with dims=a,b,c


def main():
    args = [a for a in sys.argv[1:] if "=" in a]
    # dims=... is ours, not Hydra's - pull it out before composing.
    dims_arg = [a for a in args if a.startswith("dims=")]
    overrides = [a for a in args if not a.startswith("dims=")]
    requested = DIMS
    if dims_arg:
        requested = sorted({int(x) for x in dims_arg[-1].split("=", 1)[1].split(",")})

    with initialize(version_base="1.3", config_path="configs"):
        cfg = compose(config_name="default", overrides=overrides)

    molecule = instantiate(cfg.molecule)      # pyscf only — no cudaq / factory
    norb = molecule.norb
    nelec = tuple(int(x) for x in molecule.nelec)
    fci_dim = fci_dimension(norb, nelec)
    fci_energy = float(molecule.compute_casci())   # exact in this active space

    print(f"\nmolecule active space: {sum(nelec)}e, {norb}o  ->  "
          f"{norb*2} qubits, FCI dim = {fci_dim}")
    print(f"FCI (CASCI) energy = {fci_energy:.8f} Ha\n")

    dims = [d for d in requested if d <= fci_dim]
    if not dims:
        sys.exit(f"every requested dim exceeds the FCI dimension {fci_dim}")
    print(f"subspace sizes: {dims}")
    rng = np.random.default_rng(0)

    print("running HCI (classical selected-CI) ...")
    hci = hci_curve(build_pyci_hamiltonian(molecule), nelec, max_det=max(dims))
    print("running random selection ...")
    rnd = random_curve(build_pyci_hamiltonian(molecule), norb, nelec, dims, rng, n_seeds=3)

    def mha(e):
        return (e - fci_energy) * 1000.0

    print("\n=== HCI (heat-bath CI) convergence ===")
    print(f"{'ndet':>7} {'energy (Ha)':>15} {'err vs FCI (mHa)':>18}")
    for ndet, e in hci:
        print(f"{ndet:>7} {e:>15.8f} {mha(e):>18.4f}")

    print("\n=== random selection (mean +/- std over 3 seeds) ===")
    print(f"{'ndet':>7} {'energy (Ha)':>15} {'err vs FCI (mHa)':>18} {'std (mHa)':>12}")
    for ndet, e, s in rnd:
        print(f"{ndet:>7} {e:>15.8f} {mha(e):>18.4f} {s*1000:>12.4f}")

    # Headline: HCI's error at the LARGEST size requested, i.e. the floor a
    # QSCI run with that subspace cap could possibly reach.
    largest = max(hci, key=lambda t: t[0])
    print()
    print(f">>> HCI reaches {mha(largest[1]):.3f} mHa vs FCI at "
          f"{largest[0]} determinants "
          f"({100.0 * largest[0] / fci_dim:.2f}% of the CI space).")
    print(">>> HCI is a strong HEURISTIC reference, not a bound: it is greedy and")
    print(">>> its criterion is first-order PT. The real bound is the oracle (top-N")
    print(">>> by |c| from the exact FCI vector), which is not measured here.")
    print(">>> QSCI far above HCI -> the policy is selecting poorly. Close to or")
    print(">>> below it -> the quantum sampling is competitive with classical")
    print(">>> selection, which is the value-proposition claim.")

    out_dir = cfg.get("output", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "baseline_selection.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "ndet", "energy_Ha", "err_vs_fci_mHa", "std_mHa"])
        for ndet, e in hci:
            w.writerow(["hci", ndet, e, mha(e), ""])
        for ndet, e, s in rnd:
            w.writerow(["random", ndet, e, mha(e), s * 1000])
        w.writerow(["fci", fci_dim, fci_energy, 0.0, ""])
    print(f"\nsaved: {csv_path}")


if __name__ == "__main__":
    main()
