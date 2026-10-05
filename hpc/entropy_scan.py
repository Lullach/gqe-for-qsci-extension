"""
Is there a cheap axis on which the subspace requirement is a simple function?

`subspace_requirement.py` measured, for 196 configurations, the smallest top-|c|
subspace reaching chemical accuracy. That is descriptive: it needs the exact
answer, so it cannot tell you in advance whether QSCI can work on a molecule.
This script computes candidate PREDICTORS for that number and lets
plot_entropy_regression.py fit them.

WHERE THE LAW COMES FROM
------------------------
The CI coefficients define a probability distribution over determinants,
p(D) = |c_D|^2. "How many determinants capture all but eps of the weight" is the
smallest-eps-typical-set problem, and its answer to leading exponential order is
2^H, where H = -sum_D p log2 p is the Shannon entropy of that distribution
(asymptotic equipartition). So

    log2 n_needed  ~=  H          ... (1)

is theory, up to an eps-dependent offset and O(sqrt) corrections — which is also
why the tiny systems in the scan do not obey it (h2 needing "2 of 4" is not an
asymptotic statement about anything). Equivalently, since a wavefunction spread
uniformly over the whole space has H = log2 n_fci,

    log2(fraction)  ~=  H - log2 n_fci  =  -KL(p || uniform)      ... (1')

so the FRACTION the other script reports is an entropy deficit in disguise. That
is why its plots wanted a log y axis.

(1) still needs the exact vector. The usable step: a determinant IS an occupation
pattern, so p's marginals are the per-spin-orbital occupation probabilities
p_i = <n_i>, whose binary entropies h(p_i) are computable from a 1-RDM. By
subadditivity, exactly,

    S := sum_i h(p_i)  =  H  +  I,    I = total correlation >= 0    ... (2)

S is cheap and H is not, so the working model is log2 n_needed ~= a + b*S with
b < 1 absorbing I. THAT part is a fit, not a theorem: it is linear only insofar
as I is roughly proportional to S across the systems compared. Note also that
some of I is trivial — fixed particle number alone makes the marginals dependent
— so do not read b as a pure measure of chemistry.

WHAT THIS SCRIPT COMPUTES, per configuration
--------------------------------------------
  S_cisd   summed spin-orbital marginal entropy from a CISD vector. The actual
           candidate predictor: no FCI anywhere, O(n^4) space.
  S_fci    the same marginals from the exact vector. Isolates "is CISD a good
           enough surrogate" from "are marginals a good enough surrogate".
  H_fci    the exact joint entropy. This is the falsifiable test of (1): plotted
           against log2 n_needed the slope should be ~1.
  S_hf     sanity check, must be 0 (a single determinant has no entropy).

S_fci and H_fci need a full CI solve, so they are capped by --max-exact; S_cisd
is always computed. Marginals are taken in the SAME canonical active-space MO
basis the scan's determinants live in — determinant counts are basis dependent,
so the predictor has to be too.

CPU only: pyscf + pyci, no cudaq, no GPU. Locally that means the container:

    docker run --rm --entrypoint /bin/bash -v "${PWD}:/workspace" -w /workspace \
        gqe_qsci_cpu -lc "python3 hpc/entropy_scan.py"

Resumable: the CSV is appended and flushed per configuration, and configurations
already in it are skipped.
"""

import argparse
import csv
import math
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
from gqe_qsci.qsci.baseline import build_pyci_hamiltonian         # noqa: E402
from subspace_requirement import G, configurations                # noqa: E402


LOG2E = 1.0 / math.log(2.0)


# ---------------------------------------------------------------------------
# entropies
# ---------------------------------------------------------------------------

# CLAUDE
def _binary_entropy_bits(p):
    """h(p) in bits, elementwise, with h(0) = h(1) = 0."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    q = 1.0 - p
    out = np.zeros_like(p)
    for x in (p, q):
        nz = x > 0.0
        out[nz] -= x[nz] * np.log2(x[nz])
    return out


# CLAUDE
def marginal_occupations(wfn, weights, nbasis):
    """
    Per-spin-orbital occupation probability <n_i> = sum over determinants that
    occupy orbital i of |c_D|^2.

    `to_occ_array` gives (ndet, 2, nocc) arrays of occupied orbital indices, so
    this is a weighted bincount. Returns shape (2, nbasis).
    """
    occs = np.asarray(wfn.to_occ_array())
    if occs.ndim != 3 or occs.shape[1] != 2:
        raise RuntimeError(f"unexpected occupation array shape {occs.shape}")

    p = np.zeros((2, nbasis), dtype=float)
    for spin in (0, 1):
        idx = occs[:, spin, :]                       # (ndet, nocc)
        np.add.at(p[spin], idx.ravel(),
                  np.repeat(weights, idx.shape[1]))
    return p


# Renyi orders. alpha < 1 weights the TAIL of the spectrum, alpha > 1 the mode.
# Shannon (alpha -> 1) fails badly at weak correlation: HF alone holds ~98% of
# the weight, so H ~ 0, while the correlation energy lives entirely in the tail
# that H ignores. alpha < 1 is also exactly the condition under which MPS
# truncation error is guaranteed to decay fast, so this is the principled
# generalisation rather than a fishing expedition.
# alpha -> 0 degenerates to log2(support size), which is log2 n_fci for any
# full-support vector and therefore carries no information, so the useful order
# is an interior optimum. 0.1 and 0.15 bracket it from below.
RENYI_ORDERS = (0.1, 0.15, 0.25, 0.5, 0.75, 2.0)


# CLAUDE
def _renyi_bits(w, alpha):
    """R_alpha = log2(sum p^alpha) / (1 - alpha), in bits."""
    nz = w[w > 0.0]
    if abs(alpha - 1.0) < 1e-12:
        return float(-(nz * np.log2(nz)).sum())
    return float(np.log2((nz ** alpha).sum()) / (1.0 - alpha))


# CLAUDE
def _spectral(w, prefix=""):
    """Every entropy of one normalised weight vector, in bits."""
    out = {f"{prefix}H": _renyi_bits(w, 1.0), f"{prefix}c_hf2": float(w.max())}
    for alpha in RENYI_ORDERS:
        out[f"{prefix}R{alpha:g}"] = _renyi_bits(w, alpha)
    return out


# CLAUDE
def entropy_bits(wfn, coeffs, nbasis, nelec, gap=None):
    """
    Spectral and marginal entropies of one solved wavefunction, all in bits.

    Returns (S, spectral) where
      S            = sum over spin-orbitals of h(<n_i>) -- the CHEAP quantity,
                     obtainable from a 1-RDM without any exact solve
      spectral["H"]      = Shannon entropy of |c|^2
      spectral["R<a>"]   = Renyi entropy of order a
      spectral["c_hf2"]  = largest single weight, i.e. |c_HF|^2 for these states

    With `gap` = H_DD - E, the same quantities are also computed for the
    ENERGY-weighted distribution, under the "E" prefix. Epstein-Nesbet puts the
    energy a determinant contributes at |c_D|^2 * (H_DD - E), so that is the
    distribution an energy criterion is actually sensitive to — as against
    |c_D|^2, which is what OVERLAP is sensitive to. Past dissociation the two
    come apart badly: the extra determinants are near-degenerate spin couplings
    with real weight and almost no energy, so the plain entropies stay high
    while the requirement collapses.
    """
    w = np.asarray(coeffs, dtype=float).ravel() ** 2
    w /= w.sum()                                  # solver normalisation is not exact

    spectral = _spectral(w)
    if gap is not None:
        we = w * np.asarray(gap, dtype=float)
        we = np.clip(we, 0.0, None)      # H_DD >= E variationally; guard rounding
        total = we.sum()
        spectral.update(_spectral(we / total, prefix="E") if total > 0
                        else {k: float("nan") for k in _spectral(w, "E")})

    p = marginal_occupations(wfn, w, nbasis)
    got = p.sum(axis=1)
    if not np.allclose(got, nelec, atol=1e-6):
        raise RuntimeError(
            f"marginal occupations sum to {got}, expected {nelec}; the "
            "occupation array and the coefficient order disagree."
        )
    S = float(_binary_entropy_bits(p).sum())
    return S, spectral


# ---------------------------------------------------------------------------
# wavefunctions
# ---------------------------------------------------------------------------

# CLAUDE
def _solve(ham, wfn, max_cycle):
    op = pyci.sparse_op(ham, wfn)
    energies, coeffs = op.solve(maxiter=max_cycle)
    return float(energies[0]), coeffs[0], op


# CLAUDE
def diagonal(op, n):
    """
    H_DD for every determinant, from the operator already built for the solve.

    pyci exposes `data`/`indices`/`indptr` as opaque scalars rather than CSR
    arrays, so `get_element` is the way in. It is fast (microseconds per call)
    and, unlike the CSR route, documented. `ecore` is stored separately and must
    be added back: without it the diagonal sits below the variational minimum.
    """
    return np.fromiter((op.get_element(i, i) for i in range(n)),
                       dtype=float, count=n) + op.ecore


# CLAUDE
def _entropies(ham, wfn, nelec, max_cycle):
    """Solve, take the diagonal, and return both weight- and energy-weighted
    entropies. The variational check E <= min(H_DD) catches an ecore slip."""
    energy, c, op = _solve(ham, wfn, max_cycle)
    n = _ndet(wfn)
    diag = diagonal(op, n)
    if energy > diag.min() + 1e-6:
        raise RuntimeError(
            f"E = {energy:.10f} is above min(H_DD) = {diag.min():.10f}; the "
            "diagonal is wrong (ecore?), so the energy weighting would be too."
        )
    S, spectral = entropy_bits(wfn, c, ham.nbasis, nelec, gap=diag - energy)
    return S, spectral, energy, n


# CLAUDE
def cisd_entropy(ham, nelec, max_cycle=1000):
    """Entropies of a CISD vector — the predictors that need no exact solve."""
    wfn = pyci.fullci_wfn(ham.nbasis, *nelec)
    wfn.add_hartreefock_det()
    for order in (1, 2):
        wfn.add_excited_dets(order)
    return _entropies(ham, wfn, nelec, max_cycle)


# CLAUDE
def fci_entropy(ham, nelec, max_cycle=1000):
    """Entropies of the exact vector — the surrogate check and the theory check."""
    wfn = pyci.fullci_wfn(ham.nbasis, *nelec)
    wfn.add_all_dets()
    S, spectral, energy, _ = _entropies(ham, wfn, nelec, max_cycle)
    return S, spectral, energy


# CLAUDE
def hf_entropy(ham, nelec):
    """Must be exactly 0 — one determinant carries no entropy. Cheap guard
    against a sign or ordering error in the marginal bookkeeping."""
    wfn = pyci.fullci_wfn(ham.nbasis, *nelec)
    wfn.add_hartreefock_det()
    S, _ = entropy_bits(wfn, np.array([1.0]), ham.nbasis, nelec)
    return S


# every spectral quantity, in the order the CSV stores them. The "E" copies are
# the same entropies of the ENERGY-weighted distribution |c_D|^2 (H_DD - E).
BASE_KEYS = ["H"] + [f"R{a:g}" for a in RENYI_ORDERS] + ["c_hf2"]
SPECTRAL_KEYS = BASE_KEYS + [f"E{k}" for k in BASE_KEYS]


# CLAUDE
def _ndet(wfn):
    try:
        return int(len(wfn))
    except TypeError:
        return int(wfn.ndet)


# ---------------------------------------------------------------------------
# the measured requirement, from the scan
# ---------------------------------------------------------------------------

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


# CLAUDE
def measured_requirements(pattern):
    """{config label: (n_needed, n_fci, fraction)} from the scan CSVs. Keyed by
    field count because the writer's columns changed mid-scan."""
    import glob
    out = {}
    for path in sorted(glob.glob(pattern)):
        with open(path, newline="", encoding="utf-8") as f:
            for raw in csv.reader(f):
                if not raw or raw[0] in ("config", "system"):
                    continue
                fields = SCHEMAS.get(len(raw))
                if fields is None:
                    continue
                r = dict(zip(fields, raw))
                try:
                    out[r["config"]] = (int(r["n_needed"]), int(r["n_fci"]),
                                        float(r["fraction"]))
                except (KeyError, ValueError):
                    continue
    return out


# CLAUDE
def already_done(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {r["config"] for r in csv.DictReader(f) if r.get("config")}


COLUMNS = (["config", "family", "bond_length", "basis", "nelecas", "norbcas",
            "n_fci", "n_needed", "fraction", "deficit_bits", "n_cisd",
            "e_cisd", "e_fci", "S_cisd", "S_fci"]
           + [f"{k}_cisd" for k in SPECTRAL_KEYS]
           + [f"{k}_fci" for k in SPECTRAL_KEYS]
           + ["S_hf", "seconds"])


# CLAUDE
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="data/subspace/subspace_*.csv",
                    help="the measured requirements to predict")
    ap.add_argument("--out", default="data/subspace/entropy.csv")
    ap.add_argument("--families", default=None, help="comma-separated subset")
    ap.add_argument("--max-exact", type=int, default=100_000,
                    help="skip S_fci/H_fci above this CI dimension (a full CI "
                         "solve). S_cisd is always computed.")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    measured = measured_requirements(args.scan)
    if not measured:
        sys.exit(f"no measured requirements matched {args.scan!r}; "
                 "run hpc/subspace_requirement.py first.")

    fams = {s.strip() for s in args.families.split(",")} if args.families else None
    configs = [c for c in configurations(fams) if c[0] in measured]
    done = set() if args.redo else already_done(args.out)
    todo = [c for c in configs if c[0] not in done]

    print(f"{len(measured)} measured configuration(s); {len(configs)} matched to "
          f"the systems table, {len(done)} already done, {len(todo)} to run")
    print(f"exact entropies for CI spaces <= {args.max_exact:,}")
    print(f"writing {args.out}\n")

    hdr = (f"{'config':<14}{'CI space':>10}{'needed':>8}{'log2 n':>8}"
           f"{'R.25cis':>9}{'ER.25cis':>10}{'R.25fci':>9}{'ER.25fci':>10}"
           f"{'secs':>7}")
    print(hdr)
    print("-" * len(hdr))

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # --redo TRUNCATES. Appending recomputed rows to a file written by an older
    # version of this script silently mixes column layouts behind one header,
    # which the reader cannot detect.
    mode = "w" if args.redo else "a"
    new = (args.redo or not os.path.exists(args.out)
           or os.path.getsize(args.out) == 0)

    with open(args.out, mode, newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(COLUMNS)
            fh.flush()

        for label, fam, atoms, ne, no, r, basis in todo:
            t0 = time.time()
            n_needed, n_fci, fraction = measured[label]
            try:
                mol = PySCFMolecule(
                    geometry=G(type="linear_chain", atoms=atoms, bond_length=r),
                    basis=basis, nelecas=ne, norbcas=no, spin=0, charge=0)
                nelec = tuple(int(x) for x in mol.nelec)
                ham = build_pyci_hamiltonian(mol)

                s_hf = hf_entropy(ham, nelec)
                if abs(s_hf) > 1e-9:
                    raise RuntimeError(f"HF entropy is {s_hf}, must be 0")

                s_cisd, sp_cisd, e_cisd, n_cisd = cisd_entropy(ham, nelec)

                if n_fci <= args.max_exact:
                    s_fci, sp_fci, e_fci = fci_entropy(ham, nelec)
                else:
                    nan = float("nan")
                    s_fci, e_fci = nan, nan
                    sp_fci = {k: nan for k in SPECTRAL_KEYS}
            except Exception as exc:                              # noqa: BLE001
                print(f"{label:<14}  FAILED: {type(exc).__name__}: {exc}")
                continue

            # the quantity to be predicted: -log2(fraction), i.e. KL from uniform
            deficit = -math.log2(fraction) if fraction > 0 else float("nan")
            secs = time.time() - t0

            def f(x, w=9):
                return " " * (w - 2) + "-" + " " if math.isnan(x) else f"{x:>{w}.2f}"

            print(f"{label:<14}{n_fci:>10,}{n_needed:>8,}"
                  f"{math.log2(n_needed):>8.2f}{sp_cisd['R0.25']:>9.2f}"
                  f"{sp_cisd['ER0.25']:>10.2f}{f(sp_fci['R0.25'])}"
                  f"{f(sp_fci['ER0.25'], 10)}{secs:>7.1f}")

            writer.writerow(
                [label, fam, f"{r:.2f}", basis, ne, no, n_fci, n_needed,
                 f"{fraction:.8f}", f"{deficit:.6f}", n_cisd,
                 f"{e_cisd:.10f}", f"{e_fci:.10f}",
                 f"{s_cisd:.6f}", f"{s_fci:.6f}"]
                + [f"{sp_cisd[k]:.6f}" for k in SPECTRAL_KEYS]
                + [f"{sp_fci[k]:.6f}" for k in SPECTRAL_KEYS]
                + [f"{s_hf:.2e}", f"{secs:.1f}"])
            fh.flush()

    print("\n  S       summed spin-orbital marginal entropy, bits — cheap, 1-RDM only")
    print("  H       joint Shannon entropy of |c|^2. log2(n_needed) ~= H is the")
    print("          prediction of (1); it FAILS at weak correlation, where HF holds")
    print("          almost all the weight but the energy lives in the tail.")
    print("  R_alpha Renyi entropy. alpha < 1 weights the tail instead of the mode,")
    print("          which is what an energy criterion is actually sensitive to.")
    print("  deficit -log2(fraction) = KL(|c|^2 || uniform), bits.")
    print("\n  fit it:  python3 hpc/plot_entropy_regression.py")


if __name__ == "__main__":
    main()
