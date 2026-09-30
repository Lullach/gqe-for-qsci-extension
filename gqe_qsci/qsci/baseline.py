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

    This is what QSCI's quantum sampling approximates: the sampled
    determinants are essentially |c|^2-distributed, so top-n by |c| is the
    best that sampling could ever converge to.

    IT IS NOT A LOWER BOUND ON THE ACHIEVABLE ERROR. Top-n by |c| provably
    maximises OVERLAP with the exact state (it maximises the retained
    sum of |c_i|^2), but energy is a different objective: a subspace's
    variational energy is not a sum of per-determinant contributions, and
    perturbation theory puts the energy value of determinant D at roughly
    |c_D|^2 * (H_DD - E) — the extra gap factor is exactly what HCI's
    criterion targets. So some other n-subset can have a LOWER energy, and
    this curve is an UPPER bound on the error achievable at each size.
    Finding the true optimum is combinatorial over C(N, n) subsets.

    In practice it is close, and it beat HCI on H10, but do not report it as
    a proven floor.

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

    if selfcheck:
        # Re-diagonalizing over ALL determinants must reproduce full CI. This
        # is a SECOND full solve, so it is skippable for large spaces once the
        # same code path has been verified on a small one.
        check = diagonalize(pyci_ham, dets, nelec, max_cycle)
        if abs(check - fci_energy) > 1e-8:
            raise RuntimeError(
                f"self-check failed: all-determinant diagonalization gave "
                f"{check:.10f}, full CI is {fci_energy:.10f}; extraction is wrong.")
    order = np.argsort(-c)                 # most important determinant first
    curve = []
    for n in dims:
        n = min(int(n), len(dets))
        subset = [dets[i] for i in order[:n]]
        curve.append((n, diagonalize(pyci_ham, subset, nelec, max_cycle)))
    return curve, fci_energy


def _full_ci_ordered(pyci_ham, nelec, max_cycle=1000, selfcheck=True):
    """(determinants sorted by |c| descending, fci_energy). Shared setup."""
    wfn = pyci.fullci_wfn(pyci_ham.nbasis, *nelec)
    wfn.add_all_dets()
    op = pyci.sparse_op(pyci_ham, wfn)
    energies, coeffs = op.solve(maxiter=max_cycle)
    fci_energy = float(energies[0])
    c = np.abs(np.asarray(coeffs[0]).ravel())

    dets = _determinants_of(wfn)
    if len(dets) != c.size:
        raise RuntimeError(
            f"extracted {len(dets)} determinants but got {c.size} coefficients."
        )
    check = diagonalize(pyci_ham, dets, nelec, max_cycle)
    if abs(check - fci_energy) > 1e-8:
        raise RuntimeError(
            f"self-check failed: all-determinant diagonalization gave {check:.10f}, "
            f"full CI is {fci_energy:.10f}; the extraction is wrong."
        )
    order = np.argsort(-c)
    return [dets[i] for i in order], fci_energy


def bisect_minimal(pyci_ham, ordered, nelec, fci_energy, target_mha=1.6,
                   max_cycle=1000, progress=None):
    """
    Smallest PREFIX of `ordered` whose energy is within `target_mha` of
    `fci_energy`. Returns (n, error_mha).

    Bisection, so ~log2(N) diagonalizations rather than a full curve. It assumes
    the error is monotone in prefix length; that holds for any sensible ranking
    but is not guaranteed determinant by determinant, so treat a result as the
    smallest prefix the bisection FOUND rather than a proven minimum.

    Split out of minimal_subspace so that a different ranking — by energy
    contribution rather than by |c| — measures the same thing through the same
    code. See hpc/energy_vs_overlap.py.
    """
    n_total = len(ordered)

    def err_mha(n):
        e = diagonalize(pyci_ham, ordered[:n], nelec, max_cycle)
        return (e - fci_energy) * 1000.0

    # the full space always qualifies, so the search is well posed
    lo, hi = 1, n_total
    hi_err = err_mha(hi)
    if hi_err > target_mha:                       # only via a solver failure
        return n_total, hi_err

    best = (hi, hi_err)
    while lo < hi:
        mid = (lo + hi) // 2
        e = err_mha(mid)
        if progress:
            progress(mid, e)
        if e <= target_mha:
            best, hi = (mid, e), mid
        else:
            lo = mid + 1
    return best


def minimal_subspace(pyci_ham, nelec, target_mha=1.6, max_cycle=1000,
                     progress=None, selfcheck=True):
    """
    Smallest top-|c| subspace whose energy is within `target_mha` of full CI.

    Returns (n, fraction_of_ci_space, error_mha, fci_energy, n_fci).

    The answer is an UPPER bound on the requirement: top-|c| is overlap-optimal,
    not energy-optimal (see oracle_curve), so some cleverer selection might reach
    the same accuracy with fewer determinants. It is the right question to ask of
    QSCI regardless, because |c|^2-distributed sampling is precisely what QSCI
    does — this is the subspace size QSCI would need in the best case.
    hpc/energy_vs_overlap.py measures how loose the bound actually is.
    """
    ordered, fci_energy = _full_ci_ordered(pyci_ham, nelec, max_cycle, selfcheck)
    n_fci = len(ordered)
    n, err = bisect_minimal(pyci_ham, ordered, nelec, fci_energy, target_mha,
                            max_cycle, progress)
    return n, n / n_fci, err, fci_energy, n_fci
