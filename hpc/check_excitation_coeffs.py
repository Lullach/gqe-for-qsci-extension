"""
Does ExcitationPool build a faithful excitation generator?

NOTES.md ("prefer `spec: excitation`") flags a suspected bug. The pool sums a
gate's Pauli strings like this:

    operator = None
    for p in g.generator.paulistrings:
        term = convert_pauli_to_cudaq_spin(p)
        operator = term if operator is None else (operator + term * p._coeff)

The FIRST string is added with no coefficient; every later one is scaled by
`p._coeff`. `convert_pauli_to_cudaq_spin` builds the Pauli WORD only and drops the
coefficient, so nothing restores it. If a JW excitation's terms all have
coefficients of equal magnitude (±1/8 for a double), the first term ends up with
~8x the weight of the others and loses its sign — and the operator is not the
generator it claims to be.

This decides whether `spec: excitation` can become the default, so it is worth
having as a script rather than a one-off paste.

Run (no GPU needed):
    python3 hpc/check_excitation_coeffs.py
    python3 hpc/check_excitation_coeffs.py molecule=h2o
"""

import os
import sys

REPO = os.environ.get("REPO") or (
    "/workspace" if os.path.isdir("/workspace/gqe_qsci")
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, REPO)

from hydra import compose, initialize_config_dir       # noqa: E402
from hydra.utils import instantiate                    # noqa: E402

from gqe_qsci.gqe.operator_pool import ExcitationPool  # noqa: E402
from gqe_qsci.gqe.utils import convert_pauli_to_cudaq_spin  # noqa: E402


def coeff_of(pauli_string):
    """tequila keeps this private and the attribute name has moved around."""
    for attr in ("_coeff", "coeff", "coefficient"):
        if hasattr(pauli_string, attr):
            value = getattr(pauli_string, attr)
            return value() if callable(value) else value
    return None


def main():
    overrides = [a for a in sys.argv[1:] if "=" in a]
    if not any(o.startswith("molecule=") for o in overrides):
        overrides.append("molecule=h2o")

    with initialize_config_dir(version_base="1.3",
                               config_dir=os.path.join(REPO, "configs")):
        cfg = compose(config_name="default", overrides=overrides)

    molecule = instantiate(cfg.molecule)
    pool = ExcitationPool(molecule, params=None,
                          threshold=cfg.operator_pool.ccsd_threshold)
    ansatz = pool.make_uccsd_ansatz(threshold=cfg.operator_pool.ccsd_threshold)

    print(f"\nmolecule: {overrides}")
    print(f"excitation gates in the UCCSD ansatz: {len(ansatz.gates)}\n")

    for gate_index in (0, 1):
        if gate_index >= len(ansatz.gates):
            break
        g = ansatz.gates[gate_index]
        strings = list(g.generator.paulistrings)
        print("=" * 70)
        print(f"gate {gate_index}:  amplitude = {float(g.parameter):+.8f}, "
              f"{len(strings)} Pauli strings")
        print("=" * 70)

        coeffs = []
        for p in strings:
            c = coeff_of(p)
            coeffs.append(c)
            word = convert_pauli_to_cudaq_spin(p)
            print(f"  coeff = {str(c):>24}   word = {word}")

        numeric = [complex(c) for c in coeffs if c is not None]
        if len(numeric) != len(strings):
            print("\n  !! could not read a coefficient from every string - the "
                  "attribute name may have changed in this tequila version.")
            continue

        mags = sorted({round(abs(c), 10) for c in numeric})
        print(f"\n  distinct |coeff| values: {mags}")
        if len(mags) == 1 and len(numeric) > 1:
            print(f"  -> all {len(numeric)} terms share magnitude {mags[0]}.")
            print(f"  -> dropping the FIRST one's coefficient gives it "
                  f"{1.0 / mags[0]:.1f}x the weight of the others: THIS IS A BUG.")
        else:
            print("  -> magnitudes differ; the first-term asymmetry still looks "
                  "wrong, but check against the JW decomposition by hand.")

        # What the pool builds today vs. what scaling every term would give.
        as_built = None
        corrected = None
        for p in strings:
            term = convert_pauli_to_cudaq_spin(p)
            c = coeff_of(p)
            as_built = term if as_built is None else (as_built + term * c)
            scaled = term * c
            corrected = scaled if corrected is None else (corrected + scaled)

        print(f"\n  as built by ExcitationPool : {as_built}")
        print(f"  every term scaled          : {corrected}")
        print("\n  If those differ only in the first term's weight, the fix is to "
              "scale\n  every term including the first.")


if __name__ == "__main__":
    main()
