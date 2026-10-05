"""
Real objects for tiny systems, for reading and testing the core code.

Instead of guessing what a function takes, build the objects a real run uses,
at the smallest size, and pass them in. Everything here is constructed by the
project's own Factory from the project's own configs, exactly as train.py
does, so nothing is mocked. Wiring only: no physics lives in this file.

Run it INSIDE the dev container (.devcontainer/), where pyscf, pyci and cudaq
exist. From the VS Code interactive window, the debugger or a test:

    from tests.fixtures import h2, lih, n2, example_seq, describe

    s = h2()                 # H2/STO-3G: 2 orbitals, 4 qubits, 4 determinants
    describe(s)              # sizes and one example of every input type
    seq = example_seq(s)     # a random circuit, as logged in results.csv
    s.pipeline.process({"idx": torch.tensor([seq])})   # one real QSCI step

    from numpy_sampler import circuit_terms, numpy_statevector
    coeffs, words = circuit_terms(s.sampler, seq)       # the real inputs ...
    psi = numpy_statevector(s.pool.n_qubits, s.pool.n_electrons, coeffs, words)

A routine for any core function:
  1. breakpoint inside the function,
  2. call it through one of these systems (or run a 2-epoch training),
  3. read the actual arguments in the debugger's Variables pane,
  4. shrink them to something you can check by hand, predict, run, compare,
  5. keep what you learned as a test in tests/invariants/.

To watch a function run line by line, with every variable's value and every
array's shape, wrap it with snoop (installed in the dev container):

    import snoop
    snoop(numpy_statevector)(2, 1, [np.pi / 4], ["YX"])

Systems are cached: the first call per system takes a few seconds (RHF, CCSD
and the pool; PySCF results are also cached on disk in .cache/pyscf), later
calls are instant.

The systems use the BASE config (configs/default.yaml): 100k shots, subspace
cap 2000, no excitation dedup. For an experiment's exact settings, or any other
change, pass overrides as on the command line:

    n2("experiment=n2_pool_matched")          # 10k shots, cap 170, dedup on
    n2("operator_pool.only_use_first_pauli=false")
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class System:
    name: str
    cfg: Any          # the composed Hydra config
    molecule: Any     # gqe_qsci.molecule.PySCFMolecule
    pool: Any         # the operator pool (indices in a sequence refer to it)
    sampler: Any      # gqe_qsci.gqe.sampler.Sampler (cudaq)
    pipeline: Any     # gqe_qsci.qsci.pipeline.QSCIPipeline


@lru_cache(maxsize=None)
def system(molecule: str, *overrides: str) -> System:
    """Any molecule config in configs/molecule/, plus command-line style
    overrides."""
    from hydra import compose, initialize_config_dir
    from gqe_qsci.factory import Factory

    with initialize_config_dir(version_base="1.3",
                               config_dir=os.path.join(REPO, "configs")):
        cfg = compose(config_name="default",
                      overrides=[f"molecule={molecule}", *overrides])
    pipe = Factory().create_qsci_pipeline(cfg)
    return System(molecule, cfg, pipe.mol, pipe.operator_pool,
                  pipe.sampler, pipe)


def h2(*overrides: str) -> System:
    """H2, STO-3G: 4 qubits. Small enough to check every number by hand."""
    return system("h2", *overrides)


def lih(*overrides: str) -> System:
    """LiH: the next step up."""
    return system("lih", *overrides)


def n2(*overrides: str) -> System:
    """N2 at the bond length of configs/molecule/n2.yaml: 16 qubits, the
    system of the pointer-vs-pool runs."""
    return system("n2", *overrides)


def example_seq(s: System, n_gates: int | None = None, seed: int = 0) -> list[int]:
    """A random circuit in the form training uses and results.csv logs: pool
    index 0 first (the identity every logged sequence starts with), then
    n_gates indices into s.pool (default: the config's ngates)."""
    rng = np.random.default_rng(seed)
    n = int(s.cfg.ngates) if n_gates is None else n_gates
    return [0] + [int(i) for i in rng.integers(1, len(s.pool), n)]


def describe(s: System) -> None:
    """Sizes, and one concrete example of each input type, for system s."""
    m, p = s.molecule, s.pool
    print(f"{s.name}: {m.norb} active orbitals, nelec {tuple(m.nelec)}, "
          f"{m.n_determinants} determinants")
    print(f"  qubits {p.n_qubits}, electrons {p.n_electrons}, "
          f"pool size {len(p)}, subspace cap {s.pipeline.max_dim}, "
          f"shots {s.sampler.shots_count}")
    seq = example_seq(s, n_gates=min(3, int(s.cfg.ngates)))
    try:
        from numpy_sampler import circuit_terms
    except ImportError:                       # hpc/ not on sys.path
        import sys
        sys.path.insert(0, os.path.join(REPO, "hpc"))
        from numpy_sampler import circuit_terms
    coeffs, words = circuit_terms(s.sampler, seq)
    print(f"  example seq      {seq}")
    print(f"  -> coeffs        {[round(c, 4) for c in coeffs]}")
    print(f"  -> Pauli words   {words}")
