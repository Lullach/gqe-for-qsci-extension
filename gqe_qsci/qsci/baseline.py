"""
Classical determinant-selection baselines for QSCI (value-proposition test).

QSCI selects a determinant subspace by *quantum sampling*, then diagonalizes the
Hamiltonian in that subspace classically. The honest question is whether the
quantum sampling picks better determinants than a *classical* selection at the
same subspace size. This module provides the two classical competitors:

  * heat-bath CI (HCI)   — the standard classical selected-CI heuristic (the
                           fair competitor QSCI must beat), via pyci.add_hci.
  * random selection     — number/spin-conserving random determinants (the floor:
                           how much does *any* non-trivial selection buy?).

Only the SELECTION differs; the pyci Hamiltonian build and the diagonalization
are identical to QSCIPipeline (see qsci/pipeline.py). Deliberately imports
neither cudaq nor the training stack, so it runs with just pyscf + pyci.

See NOTES.md, "Value proposition: fair classical baseline".
"""

import math

import numpy as np
import pyci

from gqe_qsci.qsci.determinant import Determinant


def _ndet(wfn) -> int:
    """Number of determinants in a pyci wavefunction (API varies by version)."""
    try:
        return int(len(wfn))
    except TypeError:
        return int(wfn.ndet)


def build_pyci_hamiltonian(molecule):
    """pyci.hamiltonian for a molecule's active space — same recipe as
    QSCIPipeline.__init__ (h2 transposed to physicist ordering)."""
    ham = molecule.cas_hamiltonian
    h2 = np.asarray(ham.h2.transpose(0, 2, 1, 3), order="C")
    return pyci.hamiltonian(ham.e_core, ham.h1, h2)


def fci_dimension(norb, nelec):
    na, nb = nelec
    return math.comb(norb, na) * math.comb(norb, nb)


def hf_determinant(nelec):
    """Hartree-Fock reference: the lowest na (nb) orbitals occupied."""
    na, nb = nelec
    return Determinant([np.uint64((1 << na) - 1), np.uint64((1 << nb) - 1)])


def diagonalize(pyci_ham, determinants, nelec, max_cycle=1000):
    """Ground-state energy of the Hamiltonian projected onto `determinants`.
    Same path as QSCIPipeline.diagonalize (fullci_wfn + sparse_op.solve)."""
    wfn = pyci.fullci_wfn(pyci_ham.nbasis, *nelec)
    for det in determinants:
        wfn.add_det(det)
    op = pyci.sparse_op(pyci_ham, wfn)
    energies, _ = op.solve(maxiter=max_cycle)
    return float(energies[0])


# ---------------------------------------------------------------------------
# Random selection (the floor)
# ---------------------------------------------------------------------------

def random_determinants(norb, nelec, n_det, rng):
    """
    n_det distinct number/spin-conserving determinants, HF always included.
    Occupation masks are built directly (bit i = orbital i), the same
    convention Determinant.from_interleaved_bitstring uses, so they diagonalize
    correctly without relying on pyscf address conventions.
    """
    na, nb = nelec
    n_det = min(n_det, fci_dimension(norb, nelec))
    seen: set[tuple[int, int]] = set()

    def mask(orbs):
        m = 0
        for o in orbs:
            m |= (1 << int(o))
        return m

    hf = ((1 << na) - 1, (1 << nb) - 1)
    seen.add(hf)
    while len(seen) < n_det:
        a = mask(rng.choice(norb, na, replace=False))
        b = mask(rng.choice(norb, nb, replace=False))
        seen.add((a, b))
    return [Determinant([np.uint64(a), np.uint64(b)]) for a, b in seen]


def random_curve(pyci_ham, norb, nelec, dims, rng, n_seeds=3, max_cycle=1000):
    """
    (ndet, mean_energy, std_energy) for random selection at each size in `dims`,
    averaged over n_seeds independent draws.
    """
    out = []
    for d in dims:
        energies = [
            diagonalize(pyci_ham, random_determinants(norb, nelec, d, rng), nelec, max_cycle)
            for _ in range(n_seeds)
        ]
        out.append((min(d, fci_dimension(norb, nelec)), float(np.mean(energies)), float(np.std(energies))))
    return out


# ---------------------------------------------------------------------------
# Heat-bath CI (the fair classical competitor)
# ---------------------------------------------------------------------------

def hci_curve(pyci_ham, nelec, max_det, eps_start=1e-2, eps_min=1e-8, max_cycle=1000):
    """
    Classical heat-bath CI convergence: (ndet, energy) recorded after each HCI
    growth step, from HF up to ~max_det determinants.

    Each step adds determinants D connected to the current wavefunction with
    |H[D, j] * c_j| > eps (pyci.add_hci), then re-solves; eps is halved until the
    subspace reaches max_det. Recording the natural (ndet, energy) sequence
    avoids any determinant-array extraction from pyci.
    """
    wfn = pyci.fullci_wfn(pyci_ham.nbasis, *nelec)
    wfn.add_det(hf_determinant(nelec))

    op = pyci.sparse_op(pyci_ham, wfn)
    energies, coeffs = op.solve(maxiter=max_cycle)
    curve = [(_ndet(wfn), float(energies[0]))]

    eps = eps_start
    while _ndet(wfn) < max_det and eps >= eps_min:
        n_added = pyci.add_hci(pyci_ham, wfn, coeffs[0], eps=eps)
        if n_added > 0:
            op = pyci.sparse_op(pyci_ham, wfn)
            energies, coeffs = op.solve(maxiter=max_cycle)
            curve.append((_ndet(wfn), float(energies[0])))
        eps *= 0.5
    return curve


# ---------------------------------------------------------------------------
# Oracle selection (the bound: what NO method can beat at a given size)
# ---------------------------------------------------------------------------

def _determinants_of(wfn):
    """
    Every determinant in `wfn`, as Determinant objects, in the wavefunction's
    own order — so index i here matches coefficient i from op.solve().

    pyci's accessor has moved between versions, hence the fallbacks. The caller
    verifies the result rather than trusting it: see oracle_curve's self-check.
    """
    occs = None
    for attr in ("to_occ_array", "to_occs_array"):
        if hasattr(wfn, attr):
            occs = np.asarray(getattr(wfn, attr)())
            break
    if occs is None:
        raise RuntimeError(
            "this pyci build exposes neither to_occ_array nor to_occs_array; "
            "the oracle needs the determinant list to sort it by coefficient."
        )

    dets = []
    for row in occs:                      # row: (2, nocc) occupied orbital indices
        masks = []
        for spin in (0, 1):
            m = 0
            for orbital in row[spin]:
                m |= (1 << int(orbital))
            masks.append(np.uint64(m))
        dets.append(Determinant(masks))
    return dets


def oracle_curve(pyci_ham, nelec, dims, max_cycle=1000):
    """
    (ndet, energy) for the BEST subspace of each size: take the exact FCI vector,
    sort determinants by |c|, keep the top n, re-diagonalize.

    This is what every selection heuristic — heat-bath CI, and QSCI's quantum
    sampling — is approximating. It is not a method: it needs the answer first.
    It is the bound that says whether a given subspace size CAN reach a given
    accuracy at all, which is the difference between "the policy is the limit"
    and "the cap is the limit".

    Strictly it is a near-optimal bound rather than a provable one: a subspace's
    variational energy is not a sum of independent per-determinant contributions,
    so some other n-determinant set could in principle do slightly better. In
    practice top-n by |c| is very close, and far tighter than HCI.

    Self-checked: re-diagonalizing over ALL extracted determinants must reproduce
    the full CI energy. If the extraction or the coefficient ordering were wrong,
    that check fails loudly instead of returning a plausible bad curve.
    """
    wfn = pyci.fullci_wfn(pyci_ham.nbasis, *nelec)
    wfn.add_all_dets()
    op = pyci.sparse_op(pyci_ham, wfn)
    energies, coeffs = op.solve(maxiter=max_cycle)
    fci_energy = float(energies[0])
    c = np.abs(np.asarray(coeffs[0]).ravel())

    dets = _determinants_of(wfn)
    if len(dets) != c.size:
        raise RuntimeError(
            f"extracted {len(dets)} determinants but got {c.size} coefficients; "
            "pyci's determinant order does not match its coefficient order."
        )

    check = diagonalize(pyci_ham, dets, nelec, max_cycle)
    if abs(check - fci_energy) > 1e-8:
        raise RuntimeError(
            f"oracle self-check failed: re-diagonalizing all {len(dets)} "
            f"extracted determinants gave {check:.10f}, full CI is "
            f"{fci_energy:.10f}. The determinant extraction is wrong, so any "
            "oracle curve from it would be meaningless."
        )

    order = np.argsort(-c)                 # most important determinant first
    curve = []
    for n in dims:
        n = min(int(n), len(dets))
        subset = [dets[i] for i in order[:n]]
        curve.append((n, diagonalize(pyci_ham, subset, nelec, max_cycle)))
    return curve, fci_energy
