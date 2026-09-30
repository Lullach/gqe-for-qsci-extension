"""
Replay finished training runs to see when the subspace-size indicators would
have fired.

results.csv logs the gate sequence of every circuit, so the best-so-far circuit
of the GQE-optimized stage can be rebuilt exactly and pushed back through the
same QSCI steps the pipeline uses (sample -> post-select -> symmetry completion
up to max_dim -> diagonalize). The indicators (gqe_qsci/qsci/diagnostics.py)
are then computed on that real QSCI subspace.

Only epochs where the best-so-far CIRCUIT changes are replayed: between them the
indicator cannot change, so the rest would repeat identical work.

What is NOT replayed: the Global-refined stage. Its subspace is an accumulated
state built from every circuit of every earlier epoch, with no sequence of its
own, so reproducing it means re-running the whole of training. The plot script
treats that stage separately and says so.

Shot noise: the replay re-samples with its own shots, so a replayed energy
differs from the logged one by sampling noise. `energy_logged` and
`energy_replayed` are both written so the size of that difference is visible
rather than assumed.

CPU only. In the container, one process per family so no two writers share a
file:

    docker run --rm --entrypoint /bin/bash -v "${PWD}:/workspace" -w /workspace \
        -e OMP_NUM_THREADS=2 gqe_qsci_cpu -lc \
        "python3 hpc/replay_indicators.py --family archL15-dag-gnn"
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np

REPO = os.environ.get("REPO") or (
    "/workspace" if os.path.isdir("/workspace/gqe_qsci")
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, REPO)

csv.field_size_limit(10_000_000)

# exp_tag family -> experiment config that produced it
EXPERIMENTS = {
    "archL15-dag-gnn":     "archL15_dag_gnn",
    "archL15-diff-1shot":  "archL15_diff_1shot",
    "archL15-diff-absorb": "archL15_diff_absorb",
    "archL15-diff-gnn":    "archL15_diff_gnn",
    "archL15-gpt2":        "archL15_gpt2",
    "L12-dag-gnn":         "Lsweep_dag_gnn_L12",
    "L20-dag-gnn":         "Lsweep_dag_gnn_L20",
    "allpauli-h10":        "ablation_allpauli_h10",
}

STAGE = "GQE-optimized(best_so_far)"


def family_of(tag):
    head, sep, tail = tag.rpartition("-s")
    return head if sep and tail.isdigit() else tag


def best_so_far_changes(path, family):
    """
    {exp_tag: [(epoch, energy, subspace_dim, seq), ...]} for the epochs where
    the best-so-far circuit of `family` changes. Streams the CSV: it is ~70 MB,
    and all but ~8% of it is per-sample rows this does not need.
    """
    runs = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["stage"] != STAGE or family_of(r["exp_tag"]) != family:
                continue
            seq = tuple(int(x) for x in r["seq"].split()) if r["seq"] else None
            runs[r["exp_tag"]].append(
                (int(r["epoch"]), float(r["energy"]),
                 int(r["subspace_dim"] or 0), seq, float(r["R-CASCI"])))
    out = {}
    for tag, rows in runs.items():
        rows.sort()
        changes, last = [], None
        for epoch, energy, dim, seq, ref in rows:
            if seq is not None and seq != last:
                changes.append((epoch, energy, dim, seq, ref))
                last = seq
        out[tag] = changes
    return out


def build_pipeline(experiment):
    """The same molecule, pool and QSCI settings the run trained with."""
    from hydra import compose, initialize_config_dir
    from gqe_qsci.factory import Factory

    with initialize_config_dir(version_base="1.3",
                               config_dir=os.path.join(REPO, "configs")):
        cfg = compose(config_name="default",
                      overrides=[f"experiment={experiment}"])
    return Factory().create_qsci_pipeline(cfg), cfg


def circuit_terms(sampler, seq):
    """The (theta, Pauli word) list the sampler's kernel would receive."""
    coeffs, words = [], []
    for op in [sampler.pool[j] for j in seq]:
        coeffs += [c.real for c in sampler.term_coefficients(op)]
        words += [term.get_pauli_word(sampler.pool.n_qubits) for term in op]
    return coeffs, words


def numpy_statevector(n_qubits, n_electrons, coeffs, words):
    """
    The sampler's kernel, in numpy: X on the first n_electrons qubits, then
    exp(i theta P) for each Pauli word.

    Why not cudaq: on the CPU target one 20-qubit circuit takes ~55 s in cudaq
    (get_state and sample alike), and a replay needs hundreds of them. Every
    gate here is a Pauli exponential, which is a permutation plus phases:

        exp(i theta P) psi = cos(theta) psi + i sin(theta) P psi
        P |b>  =  i^(#Y) * (-1)^popcount(b & zmask) |b XOR xmask>

    with Y = i X Z (Z acts first). Qubit q sits at bit q of the index (little
    endian), as in cudaq.get_state. Measured, not assumed: of the eight
    sign / word-order / bit-order combinations only this one gives fidelity 1
    against cudaq. --verify re-checks it; run it on any new pool.
    """
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
    """
    cudaq-compatible {bitstring: count} from the numpy statevector.

    cudaq.sample prints QUBIT 0 FIRST, i.e. the index's bits least-significant
    first — the reverse of format(index, "b"). Getting this wrong still passes
    the particle-number post-selection (it swaps alpha/beta and reverses the
    orbitals) and silently gives energies ~800 mHa too high, so it was checked
    against logged energies: 370.6 vs 372.0, 353.5 vs 358.2, 355.1 vs 355.5 mHa
    on the first three archL15-dag-gnn-s1 circuits, the gaps being shot noise.
    """
    coeffs, words = circuit_terms(sampler, seq)
    n = sampler.pool.n_qubits
    psi = numpy_statevector(n, sampler.pool.n_electrons, coeffs, words)
    p = np.abs(psi) ** 2
    p /= p.sum()
    support = np.flatnonzero(p > 1e-16)
    draws = rng.multinomial(shots, p[support])
    return {format(int(i), f"0{n}b")[::-1]: int(c)
            for i, c in zip(support, draws) if c > 0}


def replay(pipe, seq, rng, with_pt2=True):
    """(energy, indicators, subspace_dim, pt2_mha) for one circuit, through the
    pipeline's own subspace construction."""
    import pyci
    from gqe_qsci.qsci.diagnostics import boundary_indicators
    from gqe_qsci.qsci.subspace import DeterminantSubspace

    counts = sample_counts(pipe.sampler, seq, pipe.sampler.shots_count, rng)
    sub = DeterminantSubspace.from_cudaq_sample_result(counts)
    sub = sub.post_select_by_nelec(pipe.nelec)
    sub = sub.enlarge(max_dim=pipe.max_dim, method=pipe.enlarge_method)

    wfn = pyci.fullci_wfn(pipe._pyci_ham.nbasis, *pipe.nelec)
    for det in sub.determinants:
        wfn.add_det(det)
    energies, coeffs = pyci.sparse_op(pipe._pyci_ham, wfn).solve(
        maxiter=pipe.max_cycle)
    energy = float(energies[0])
    ind = boundary_indicators(pipe._pyci_ham, sub.determinants, coeffs[0],
                              energy, pipe.nelec, max_cycle=pipe.max_cycle)
    pt2 = float("nan")
    if with_pt2:
        from gqe_qsci.qsci.diagnostics import enpt2_mha
        pt2 = enpt2_mha(pipe._pyci_ham, wfn, coeffs[0], energy)
    return energy, ind, sub.ndet, pt2


def verify(pipe, seq):
    """numpy statevector against cudaq.get_state for one circuit. Slow (the
    cudaq half takes about a minute) but it is the only thing licensing the
    fast path."""
    import cudaq
    s = pipe.sampler
    coeffs, words = circuit_terms(s, seq)
    ours = numpy_statevector(s.pool.n_qubits, s.pool.n_electrons, coeffs, words)
    theirs = np.asarray(cudaq.get_state(
        s.kernel, s.pool.n_qubits, s.pool.n_electrons, coeffs,
        [cudaq.pauli_word(w) for w in words]))
    fidelity = abs(np.vdot(theirs, ours)) ** 2
    diff = np.abs(np.abs(ours) ** 2 - np.abs(theirs) ** 2).max()
    print(f"verify: fidelity {fidelity:.12f}, max |prob diff| {diff:.2e}")
    if fidelity < 1 - 1e-9:
        sys.exit("numpy statevector disagrees with cudaq; not replaying.")


COLUMNS = ["exp_tag", "family", "seed", "epoch", "energy_logged",
           "energy_replayed", "casci", "err_logged_mha", "subspace_dim",
           "max_dim", "tail_weight", "boundary_mha", "pt2_mha", "seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",
                    default="outputs/gqe-for-qsci/ArchitectureComparison/results.csv")
    ap.add_argument("--family", required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument("--out-dir",
                    default="data/indicators")
    ap.add_argument("--max-epoch", type=int, default=None,
                    help="stop replaying past this epoch")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds to replay (default: all)")
    ap.add_argument("--verify", action="store_true",
                    help="check the numpy simulator against cudaq on the first "
                         "circuit before replaying")
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args()

    changes = best_so_far_changes(args.csv, args.family)
    if not changes:
        sys.exit(f"no {STAGE} rows for family {args.family!r} in {args.csv}")
    total = sum(len(v) for v in changes.values())
    print(f"{args.family}: {len(changes)} run(s), {total} best-so-far circuit(s)")

    if args.seeds:
        keep = {s.strip() for s in args.seeds.split(",")}
        changes = {t: v for t, v in changes.items()
                   if t.rpartition("-s")[2] in keep}

    pipe, _ = build_pipeline(EXPERIMENTS[args.family])
    print(f"max_dim {pipe.max_dim}, shots {pipe.sampler.shots_count}")
    if args.verify:
        verify(pipe, next(iter(changes.values()))[-1][3])
    rng = np.random.default_rng(args.rng_seed)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.family}.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for tag in sorted(changes):
            seed = tag.rpartition("-s")[2]
            for epoch, e_log, dim, seq, ref in changes[tag]:
                if args.max_epoch is not None and epoch > args.max_epoch:
                    break
                t0 = time.time()
                e_rep, ind, ndet, pt2 = replay(pipe, seq, rng)
                secs = time.time() - t0
                print(f"{tag:<24} ep {epoch:>4}  err {1000 * (e_log - ref):7.2f}"
                      f" (replay {1000 * (e_rep - ref):7.2f}) mHa  dim {ndet:>5}"
                      f"  tail {ind.tail_weight:.1e}  bound "
                      f"{ind.boundary_mha:6.3f}  pt2 {pt2:8.2f}  {secs:5.1f}s",
                      flush=True)
                writer.writerow([tag, args.family, seed, epoch, f"{e_log:.10f}",
                                 f"{e_rep:.10f}", f"{ref:.10f}",
                                 f"{1000 * (e_log - ref):.4f}", ndet,
                                 pipe.max_dim, f"{ind.tail_weight:.6e}",
                                 f"{ind.boundary_mha:.6f}", f"{pt2:.4f}",
                                 f"{secs:.1f}"])
                fh.flush()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
