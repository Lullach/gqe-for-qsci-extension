# Invariant tests

Physics invariants of the core code (see `.githooks/core_paths`). The
pre-commit hook runs `pytest tests/invariants -q` whenever a commit touches
core, and blocks the commit if a test fails. While no tests exist, pytest
collects nothing (exit code 5) and the hook lets the commit through.

The tests are written by Lukas.

## Planned invariants

> DRAFT proposed by Claude, to be edited, reordered or deleted. The most useful
> invariants compare two independent routes to the same number, because those
> catch convention errors (bit order, signs, coefficients) that look plausible.

1. **Hamiltonian symmetry.** `h1` is symmetric and `h2` has the 8-fold
   permutation symmetry of real orbitals.
2. **Full space reproduces CASCI.** Diagonalizing in the full determinant space
   gives the CASCI energy of the same active space.
3. **Variational bound.** Any subspace energy is >= the full CI energy, and
   adding determinants never raises it.
4. **Particle number.** Every determinant that survives post-selection has
   exactly (n_alpha, n_beta) electrons.
5. **Determinant <-> bitstring round trip.** Interleaved bitstring to
   `Determinant` and back is the identity, in the qubit order `cudaq.sample`
   uses (qubit 0 printed first).
6. **Operator pool coefficients.** A full excitation generator carries every
   Pauli string with its Jordan-Wigner coefficient (the ExcitationPool bug of
   2026-09-28 gave the first string 8x weight).
7. **Spin completion is closed.** Symmetry completion returns a set closed
   under alpha/beta exchange.
8. **Numpy sampler = CUDA-Q.** Fidelity 1 between `hpc/numpy_sampler.py` and
   `cudaq.get_state` on a random circuit. Needs CUDA-Q (Docker), so probably a
   manual check rather than a hook test.

## Environment

The hook runs the tests in the `gqe_qsci_cpu` Docker container, the same
environment as local runs, so tests can import the physics code (pyscf, pyci,
cudaq are all there; no Windows Python here has them). Docker Desktop must be
running when you commit core; a run takes ~10 s plus the tests themselves.
`PYTHONPATH` is the repo root, so `import gqe_qsci...` works.

To run them by hand (Git Bash):

    MSYS_NO_PATHCONV=1 docker run --rm --entrypoint /bin/bash -v "$PWD:/workspace" -w /workspace -e PYTHONPATH=/workspace gqe_qsci_cpu -c "python3 -m pytest tests/invariants -q"

`INVARIANTS_CMD` overrides the command the hook uses, e.g. on a machine
without Docker.
