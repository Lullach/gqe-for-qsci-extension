"""
The GQE sampler's circuits, simulated in numpy instead of cudaq.

Used by hpc/train_fast_sampler.py (and through it hpc/jobs/packed_cpu.sh) on
any machine without a GPU. On the CPU target cudaq takes ~55 s to simulate one
20-qubit circuit (sample and get_state alike); this takes well under a second.

Every gate the sampler applies is a Pauli exponential, which is a permutation
plus phases:

    exp(i theta P) psi = cos(theta) psi + i sin(theta) P psi
    P |b>  =  i^(#Y) * (-1)^popcount(b & zmask) |b XOR xmask>

with Y = i X Z (Z acts first).

Two cudaq conventions this depends on, both MEASURED, not assumed:
  * cudaq.get_state is little endian: qubit q sits at bit q of the index. Of the
    eight sign / word-order / bit-order combinations only this one gives
    fidelity 1 against cudaq.
  * cudaq.sample prints QUBIT 0 FIRST, i.e. the index's bits least-significant
    first — the reverse of format(index, "b"). Getting this wrong still passes
    the particle-number post-selection (it swaps alpha/beta and reverses the
    orbitals) and silently gives energies ~800 mHa too high.

Verified at fidelity 1.000000000000 against cudaq.get_state on every pool it
has been used with (the eight H10 architecture-comparison pools and N2). Check
a NEW pool before trusting it, in the container:

    python3 hpc/numpy_sampler.py experiment=<name>

Originally part of hpc/replay_indicators.py; that and the rest of the
subspace-indicator investigation are preserved at git tag
thesis/subspace-indicators.
"""

import os
import sys

import numpy as np


def circuit_terms(sampler, seq):
    """The (theta, Pauli word) list the sampler's kernel would receive."""
    coeffs, words = [], []
    for op in [sampler.pool[j] for j in seq]:
        coeffs += [c.real for c in sampler.term_coefficients(op)]
        words += [term.get_pauli_word(sampler.pool.n_qubits) for term in op]
    return coeffs, words


def numpy_statevector(n_qubits, n_electrons, coeffs, words):
    """The sampler's kernel: X on the first n_electrons qubits, then
    exp(i theta P) for each Pauli word. Little endian, as cudaq.get_state."""
    dim = 1 << n_qubits
    idx = np.arange(dim, dtype=np.int64)
    psi = np.zeros(dim, dtype=np.complex128)
    occ = 0
    for q in range(n_electrons):
        occ |= 1 << q
    psi[occ] = 1.0

    for theta, word in zip(coeffs, words):
        xmask = zmask = ny = 0
        for q, ch in enumerate(word):
            bit = 1 << q
            if ch in "XY":
                xmask |= bit
            if ch in "ZY":
                zmask |= bit
            ny += ch == "Y"
        if xmask == 0 and zmask == 0:
            psi *= np.exp(1j * theta)               # identity term: global phase
            continue
        src = idx ^ xmask                         # (P psi)[c] = phase(c^x) psi[c^x]
        parity = np.zeros(dim, dtype=np.int64)
        z = zmask
        while z:
            low = z & -z
            parity ^= (src & low) != 0
            z ^= low
        phase = (1j ** ny) * (1 - 2 * parity)
        p_psi = phase * psi[src]
        psi = np.cos(theta) * psi + 1j * np.sin(theta) * p_psi
    return psi


def sample_counts(sampler, seq, shots, rng):
    """cudaq-compatible {bitstring: count}, qubit 0 first as cudaq.sample
    prints it."""
    coeffs, words = circuit_terms(sampler, seq)
    n = sampler.pool.n_qubits
    psi = numpy_statevector(n, sampler.pool.n_electrons, coeffs, words)
    p = np.abs(psi) ** 2
    p /= p.sum()
    support = np.flatnonzero(p > 1e-16)
    draws = rng.multinomial(shots, p[support])
    return {format(int(i), f"0{n}b")[::-1]: int(c)
            for i, c in zip(support, draws) if c > 0}


def verify(sampler, seq):
    """Fidelity of the numpy statevector against cudaq.get_state for one
    circuit. Slow: the cudaq half takes about a minute at 20 qubits."""
    import cudaq
    coeffs, words = circuit_terms(sampler, seq)
    ours = numpy_statevector(sampler.pool.n_qubits, sampler.pool.n_electrons,
                             coeffs, words)
    theirs = np.asarray(cudaq.get_state(
        sampler.kernel, sampler.pool.n_qubits, sampler.pool.n_electrons,
        coeffs, [cudaq.pauli_word(w) for w in words]))
    return abs(np.vdot(theirs, ours)) ** 2


if __name__ == "__main__":
    REPO = "/workspace" if os.path.isdir("/workspace/gqe_qsci") else \
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, REPO)
    from hydra import compose, initialize_config_dir
    from gqe_qsci.factory import Factory

    with initialize_config_dir(version_base="1.3",
                               config_dir=os.path.join(REPO, "configs")):
        cfg = compose(config_name="default", overrides=sys.argv[1:])
    pipe = Factory().create_qsci_pipeline(cfg)
    rng = np.random.default_rng(0)
    seq = [0] + [int(x) for x in rng.integers(1, len(pipe.operator_pool),
                                             int(cfg.ngates))]
    f = verify(pipe.sampler, seq)
    print(f"fidelity numpy vs cudaq on a random {len(seq) - 1}-gate circuit: "
          f"{f:.12f}")
    sys.exit(0 if f > 1 - 1e-9 else "MISMATCH: do not use the numpy sampler "
                                    "with this pool.")
