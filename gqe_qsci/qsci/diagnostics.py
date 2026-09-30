"""
Live indicators of whether a QSCI energy is limited by the SIZE of its subspace.

A plateau in the best-so-far energy has two possible causes, and they call for
opposite responses:

  * the policy has stopped finding better determinants  -> a policy problem
  * the subspace is too small to represent the state    -> raise coverage (if
    the subspace is at its cap) or L / shots (if the circuit cannot populate
    more determinants than it does)

Both indicators here look at the BOUNDARY of the subspace: the determinants with
the smallest coefficients in the QSCI eigenvector. If those still matter, the
subspace was truncated in the middle of the distribution, and more determinants
would help. If they carry nothing, the subspace has captured what it can, and a
remaining error is about WHICH determinants were chosen, not how many.

  tail_weight   sum of |c|^2 over the bottom `frac` of determinants by |c|.
                Free: it only reads the eigenvector already computed.
  boundary_mha  energy lost by dropping that same bottom slice and
                re-diagonalizing, in mHa. One extra diagonalization.

They measure the same slice in weight and in energy respectively. The
energy-vs-overlap study (hpc/energy_vs_overlap.py) found the two rankings pick
the same determinants to within ~1% (geometric mean ratio 1.01 over 165
configurations), so the free indicator should carry most of the information of
the paid one. Neither needs the exact answer.

The marginal value of the LAST 10% is used as a proxy for the marginal value of
the NEXT 10%. That assumes the value per determinant varies smoothly across the
cut, which holds for a subspace ordered by importance and is what the
calibration in hpc/calibrate_indicators.py checks against exact vectors.
"""

from dataclasses import dataclass

import numpy as np
import pyci


@dataclass(frozen=True, slots=True)
class BoundaryIndicators:
    tail_weight: float         # sum |c|^2 over the bottom slice
    boundary_mha: float | None  # E(without slice) - E(full), mHa; None if not computed
    ndet: int
    frac: float


def _energy(pyci_ham, determinants, nelec, max_cycle):
    wfn = pyci.fullci_wfn(pyci_ham.nbasis, *nelec)
    for det in determinants:
        wfn.add_det(det)
    energies, _ = pyci.sparse_op(pyci_ham, wfn).solve(maxiter=max_cycle)
    return float(energies[0])


def enpt2_mha(pyci_ham, wfn, coeffs, energy, eps: float = 1e-5) -> float:
    """
    Epstein-Nesbet second-order estimate of the energy lying OUTSIDE the
    subspace, in mHa (negative: expanding the subspace would lower E by about
    this much).

    This is the indicator that works on real QSCI subspaces where the two
    boundary indicators do not. It sums |<a|H|psi>|^2 / (E - H_aa) over
    determinants a NOT in the subspace but connected to it, so it measures what
    is missing instead of what is at the edge. Replaying the L=15 H10 runs, it
    recovered 83% / 46% / 28% of the true error at 370 / 102 / 63 mHa — an
    underestimate that grows as the subspace improves (it only sees one
    excitation step past the subspace), but a GAUGE that scales with the error,
    where the boundary indicators read "converged" at 370 mHa.

    pyci returns E_var + E_PT2; this returns the correction alone.
    """
    total = float(pyci.compute_enpt2(pyci_ham, wfn, coeffs, float(energy), eps))
    return (total - float(energy)) * 1000.0


def boundary_indicators(pyci_ham, determinants, coeffs, energy, nelec,
                        frac: float = 0.1, with_energy: bool = True,
                        max_cycle: int = 200,
                        min_ndet: int = 20) -> BoundaryIndicators:
    """
    Both indicators for one solved subspace.

    `determinants` and `coeffs` must be in the same order (as returned by the
    diagonalization); `energy` is that diagonalization's ground-state energy.
    Subspaces smaller than `min_ndet` are too small for a "bottom 10%" to mean
    anything and return NaN.
    """
    c = np.abs(np.asarray(coeffs, dtype=float).ravel())
    n = c.size
    if n != len(determinants):
        raise ValueError(f"{len(determinants)} determinants but {n} coefficients")
    if n < min_ndet:
        return BoundaryIndicators(float("nan"), None if not with_energy
                                  else float("nan"), n, frac)

    w = c ** 2
    w = w / w.sum()
    k = max(1, int(round(frac * n)))
    order = np.argsort(c)                    # ascending: the tail first
    tail, keep = order[:k], order[k:]
    tail_weight = float(w[tail].sum())

    boundary = None
    if with_energy:
        kept = [determinants[i] for i in keep]
        e_kept = _energy(pyci_ham, kept, nelec, max_cycle)
        boundary = (e_kept - float(energy)) * 1000.0

    return BoundaryIndicators(tail_weight, boundary, n, frac)
