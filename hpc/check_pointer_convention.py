"""
Does `only_use_first_pauli` reach the pointer's make_excitation_operator?

The pointer policy must mean the same thing by "an action" as the pool does:

    only_use_first_pauli=True   one Pauli fragment, 1 rotation
    only_use_first_pauli=False  the FULL excitation generator, ~8 rotations

The compiled gate count is a deterministic function of the operator, so this can
be settled directly — no training run, no W&B sync, no throwaway smoke runs
cluttering the project.

Builds an empty pool exactly as `n2_pointer` does (ccsd_screening: false, so the
pool starts at the identity and no CCSD runs), picks a real same-spin double from
the molecule's own orbital occupations, and builds the operator both ways.

Run (no GPU needed):
    python3 hpc/check_pointer_convention.py
    python3 hpc/check_pointer_convention.py molecule=n2
"""

import os
import sys

REPO = os.environ.get("REPO") or (
    "/workspace" if os.path.isdir("/workspace/gqe_qsci")
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, REPO)

from hydra import compose, initialize_config_dir               # noqa: E402
from hydra.utils import instantiate                            # noqa: E402

from gqe_qsci.gqe.operator_pool import PauliEvolutionPool      # noqa: E402
from gqe_qsci.gqe.utils import get_pauli_evolution_gate_count  # noqa: E402


def main():
    overrides = [a for a in sys.argv[1:] if "=" in a]
    if not any(o.startswith("molecule=") for o in overrides):
        overrides.append("molecule=h2o")

    with initialize_config_dir(version_base="1.3",
                               config_dir=os.path.join(REPO, "configs")):
        cfg = compose(config_name="default", overrides=overrides)

    molecule = instantiate(cfg.molecule)

    # Empty pool, no CCSD — exactly what experiment=n2_pointer builds.
    pool = PauliEvolutionPool(
        molecule, params=None,
        threshold=cfg.operator_pool.ccsd_threshold,
        remove_z_ladder=cfg.operator_pool.remove_z_ladder,
        only_use_first_pauli=True,
        ccsd_screening=False,
    )

    print()
    print("=" * 70)
    print("POINTER: make_excitation_operator under both conventions")
    print("=" * 70)
    print(f"  molecule overrides : {overrides}")
    print(f"  remove_z_ladder    : {cfg.operator_pool.remove_z_ladder}")
    print(f"  pool size          : {len(pool)}   (must be 1: identity only)")

    orb = pool.get_orbital_features()
    n_so = orb.shape[0]
    occ_alpha = [q for q in range(n_so) if orb[q, 1] > 0 and q % 2 == 0]
    virt_alpha = [q for q in range(n_so) if orb[q, 1] == 0 and q % 2 == 0]
    if len(occ_alpha) < 2 or len(virt_alpha) < 2:
        sys.exit("not enough alpha orbitals for a same-spin double in this "
                 "active space; try molecule=n2")

    i, j = occ_alpha[0], occ_alpha[1]
    a, b = virt_alpha[0], virt_alpha[1]
    pairs = [(a, i), (b, j)]
    print(f"  excitation         : occupied {i},{j} -> virtual {a},{b}")

    def measure(operator):
        """(number of Pauli terms, compiled gate count) — same recipe as
        PauliEvolutionPool.get_gate_count."""
        terms = 0
        total = 0
        for term in operator:
            total += get_pauli_evolution_gate_count(
                term.get_pauli_word(pool.n_qubits))["total"]
            terms += 1
        return terms, total

    results = {}
    for first_only in (True, False):
        pool._only_use_first_pauli = first_only
        operator = pool.make_excitation_operator(pairs, angle=0.1)
        terms, total = measure(operator)
        results[first_only] = (terms, total)
        label = "first fragment" if first_only else "full generator"
        print()
        print(f"  only_use_first_pauli = {first_only}   ({label})")
        print(f"    Pauli terms     : {terms}")
        print(f"    compiled gates  : {total}")

    terms_true, gates_true = results[True]
    terms_false, gates_false = results[False]

    print()
    print("-" * 70)
    if terms_false > terms_true and gates_false > gates_true:
        ratio = gates_false / gates_true if gates_true else float("inf")
        print(f"  PASS: full generator has {terms_false} terms vs {terms_true}, "
              f"and {ratio:.1f}x the compiled gates.")
        print("  The flag reaches make_excitation_operator, and an excitation")
        print("  action really is ~8x deeper than a fragment action — which is")
        print("  why switching the default breaks comparability at fixed L.")
        return 0

    print("  FAIL: both conventions produced the same operator. The flag is not")
    print("  reaching make_excitation_operator.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
