"""
Gate for the pointer action space (design option "C"): can a policy build valid
excitations by pointing at ORBITALS, with no operator pool as input?

The whole design rests on masking being exactly right. An off-by-one in the Sz
bookkeeping does not crash — it silently emits an excitation that breaks
particle number or spin, which QSCI would happily diagonalize into a wrong
answer. So the masks are checked in both directions against a brute-force
enumeration of the excitation space (cheap at these active-space sizes, and the
same strategy used to verify the DAG canonical mask):

    SOUND     nothing invalid can be sampled      (sample many, validate each)
    COMPLETE  nothing valid is masked out         (replay every valid excitation)

Together those two make the closed-form O(n) masks provably equivalent to the
O(n^4) enumeration they replace.

Also checked:
  - sampling and replay agree exactly (GRPO's importance ratio depends on it)
  - gradients flow through encoder + pointer
  - ONE module instance serves 10q / 12q / 16q molecules with no resizing
  - the seam with CircuitDAGGNNPolicy: its pooled frontier IS the pointer query

Runs on a laptop, CPU only, in about a minute. torch_geometric is needed only
for the last section; everything else runs without it.

    docker run --rm --entrypoint /bin/bash \
      -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp \
      -e OMPI_MCA_opal_warn_on_missing_libcuda=0 \
      -v "${workdir}:/workspace" -w /workspace gqe_qsci_cpu \
      -lc "python3 hpc/smoke_pointer.py"
"""

import os
import sys
import traceback

import numpy
import torch

REPO = os.environ.get("REPO") or (
    "/workspace" if os.path.isdir("/workspace/gqe_qsci")
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, REPO)

from hydra import compose, initialize_config_dir            # noqa: E402
from gqe_qsci.factory import Factory                        # noqa: E402
from gqe_qsci.gqe.models.pointer import (                   # noqa: E402
    ExcitationPointer, ExcitationRules, OrbitalEncoder,
    build_orbital_inputs, excitation_qubits, gate_embedding,
)

HIDDEN = 64
NGATES = 10
N_SAMPLES = 2000

torch.manual_seed(0)
results = []


def record(name, ok, msg=""):
    results.append((name, ok, msg))
    print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f"  ({msg})" if msg else ""),
          flush=True)


# ---------------------------------------------------------------------------
# Ground truth, independent of the code under test
# ---------------------------------------------------------------------------

def enumerate_excitations(orb):
    """Every Sz-conserving single and double, in canonical spelling (j>i, b>a)."""
    n = orb.shape[0]
    occ = orb[:, 1] > 0
    spin = orb[:, 2].astype(int)
    occs = [p for p in range(n) if occ[p]]
    virts = [p for p in range(n) if not occ[p]]

    singles = [(i, n, a, n) for i in occs for a in virts if spin[a] == spin[i]]
    doubles = []
    for x, i in enumerate(occs):
        for j in occs[x + 1:]:
            n_beta = spin[i] + spin[j]
            for y, a in enumerate(virts):
                for b in virts[y + 1:]:
                    if spin[a] + spin[b] == n_beta:
                        doubles.append((i, j, a, b))
    return singles, doubles


def violations(row, orb):
    """
    Physics check written from scratch, NOT by consulting the enumeration —
    so a shared misconception cannot make both agree.
    """
    n = orb.shape[0]
    occ = orb[:, 1] > 0
    spin = orb[:, 2].astype(int)
    i, j, a, b = (int(v) for v in row)
    bad = []

    if not (0 <= i < n and occ[i]):
        bad.append("i is not an occupied orbital")
    if not (0 <= a < n and not occ[a]):
        bad.append("a is not a virtual orbital")

    single = (j == n)
    if single != (b == n):
        bad.append("j and b disagree about arity")

    if single:
        if 0 <= i < n and 0 <= a < n and spin[a] != spin[i]:
            bad.append("single excitation flips spin")
    else:
        if not (0 <= j < n and occ[j]):
            bad.append("j is not an occupied orbital")
        elif j <= i:
            bad.append("j <= i (non-canonical spelling)")
        if not (0 <= b < n and not occ[b]):
            bad.append("b is not a virtual orbital")
        elif b <= a:
            bad.append("b <= a (non-canonical spelling)")
        if 0 <= j < n and 0 <= b < n and 0 <= i < n and 0 <= a < n:
            if spin[i] + spin[j] != spin[a] + spin[b]:
                bad.append("Sz not conserved")
    return bad


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------

print("=" * 72)
print("Pointer action space: can we generate excitations without an operator pool?")
print("=" * 72)
print("\n=== building molecules (LiH 10q / H2O 12q / N2 16q) ===", flush=True)

with initialize_config_dir(version_base="1.3",
                           config_dir=os.path.join(REPO, "configs")):
    cfg = compose(
        config_name="default",
        overrides=["molecule_set=cross_molecule", "model=dag_gnn_features",
                   "operator_pool.dedup_excitations=true"],
    )

bundles = Factory().create_molecule_bundles(cfg)
inputs = {}
for name, b in bundles.items():
    orb, pair = build_orbital_inputs(b.pool)
    inputs[name] = (orb, pair)
    print(f"  {name:<6} n_qubits={b.n_qubits:<3} pool V={b.vocab_size:<5} "
          f"orb={orb.shape} pair={pair.shape}", flush=True)

record("orbital inputs built for every molecule", len(inputs) == len(bundles))
record("all molecules share pair_dim",
       len({p.shape[-1] for _, p in inputs.values()}) == 1)

# ---------------------------------------------------------------------------
# ONE module instance for every molecule — size independence is the point
# ---------------------------------------------------------------------------

orb_dim = next(iter(inputs.values()))[0].shape[1]
pair_dim = next(iter(inputs.values()))[1].shape[-1]

encoder = OrbitalEncoder(orb_dim, pair_dim, HIDDEN, num_layers=2, num_heads=4)
pointer = ExcitationPointer(HIDDEN, pair_dim)
encoder.eval()
pointer.eval()

# Frozen normalization pooled over all molecules, so the same physical orbital
# lands at the same coordinates everywhere (OperatorScorer's rationale).
encoder.set_normalization(
    numpy.concatenate([o for o, _ in inputs.values()], axis=0),
    numpy.concatenate([p.reshape(-1, pair_dim) for _, p in inputs.values()],
                      axis=0).reshape(1, -1, pair_dim),
)
record("one encoder/pointer pair constructed for all molecules", True,
       f"hidden={HIDDEN}, params="
       f"{sum(p.numel() for p in list(encoder.parameters()) + list(pointer.parameters())):,}")

scaling_rows = []

for name, b in bundles.items():
    print(f"\n=== {name}  ({b.n_qubits} qubits) ===", flush=True)
    orb, pair = inputs[name]
    n = orb.shape[0]

    orb_t = torch.from_numpy(orb)
    pair_t = torch.from_numpy(pair)
    rules = ExcitationRules(orb)

    singles, doubles = enumerate_excitations(orb)
    valid = singles + doubles
    valid_set = set(valid)
    print(f"  excitation space: {len(singles)} singles + {len(doubles)} doubles "
          f"= {len(valid)}", flush=True)
    scaling_rows.append((name, b.n_qubits, b.vocab_size, len(valid), 4 * (n + 1)))

    try:
        with torch.no_grad():
            keys = encoder(orb_t, pair_t)
        record(f"{name}: encoder -> orbital keys", tuple(keys.shape) == (n, HIDDEN),
               f"shape={tuple(keys.shape)}")
    except Exception as exc:                                        # noqa: BLE001
        record(f"{name}: encoder -> orbital keys", False,
               f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        continue

    # -- masks are never empty ---------------------------------------------
    try:
        empty = []
        probe = torch.randn(8, HIDDEN)
        with torch.no_grad():
            picks_probe, _ = pointer(probe, keys, pair_t, rules, 1.0)
        for s in range(4):
            m = rules.step_mask(s, [picks_probe[:, k] for k in range(s)], 8)
            if not bool(m.any(dim=-1).all()):
                empty.append(s)
        record(f"{name}: no step ever masks out every candidate", not empty,
               "" if not empty else f"empty at steps {empty}")
    except Exception as exc:                                        # noqa: BLE001
        record(f"{name}: no step ever masks out every candidate", False,
               f"{type(exc).__name__}: {exc}")

    # -- SOUND: nothing invalid can be sampled ------------------------------
    query = torch.randn(N_SAMPLES, HIDDEN)
    with torch.no_grad():
        picks, step_logp = pointer(query, keys, pair_t, rules, 1.0)

    rows = [tuple(int(v) for v in r) for r in picks]
    bad_physics = [(r, violations(r, orb)) for r in rows]
    bad_physics = [(r, v) for r, v in bad_physics if v]
    record(f"{name}: every sample obeys occupancy/Sz/ordering "
           f"({N_SAMPLES} samples)", not bad_physics,
           "" if not bad_physics else f"{len(bad_physics)} bad, e.g. "
                                      f"{bad_physics[0][0]} -> {bad_physics[0][1]}")

    outside = [r for r in rows if r not in valid_set]
    record(f"{name}: every sample is in the enumerated space", not outside,
           "" if not outside else f"{len(outside)} outside, e.g. {outside[0]}")

    distinct = len(set(rows))
    record(f"{name}: sampling reaches most of the space",
           distinct >= 0.5 * len(valid),
           f"{distinct}/{len(valid)} distinct = {100*distinct/len(valid):.0f}%")

    # -- COMPLETE: nothing valid is masked out ------------------------------
    try:
        forced = torch.tensor(valid, dtype=torch.long)
        q_all = torch.randn(len(valid), HIDDEN)
        with torch.no_grad():
            back, logp_all = pointer(q_all, keys, pair_t, rules, 1.0, forced=forced)
        finite = bool(torch.isfinite(logp_all).all())
        echoed = bool(torch.equal(back, forced))
        record(f"{name}: every valid excitation has finite log-prob "
               f"({len(valid)} replayed)", finite and echoed,
               "" if (finite and echoed)
               else f"finite={finite} picks_echoed={echoed}")
    except Exception as exc:                                        # noqa: BLE001
        record(f"{name}: every valid excitation has finite log-prob", False,
               f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    # -- sampling and replay must agree exactly -----------------------------
    with torch.no_grad():
        _, replay_logp = pointer(query, keys, pair_t, rules, 1.0, forced=picks)
    delta = (replay_logp - step_logp).abs().max().item()
    record(f"{name}: replay reproduces the sampling log-prob", delta < 1e-5,
           f"max|delta|={delta:.2e}")

    # -- log-probs are usable by GRPO --------------------------------------
    total = step_logp.sum(-1)
    record(f"{name}: gate log-probs finite and <= 0",
           bool(torch.isfinite(total).all() and (total <= 1e-6).all()),
           f"mean={total.mean().item():.3f}")

    # -- footprints look like real gates ------------------------------------
    fps = [excitation_qubits(r, n) for r in rows[:200]]
    sizes = {len(f) for f in fps}
    record(f"{name}: footprints are 2 (single) or 4 (double) orbitals",
           sizes <= {2, 4}, f"sizes seen: {sorted(sizes)}")
    record(f"{name}: footprints have no repeated qubit",
           all(len(set(f)) == len(f) for f in fps))

# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

print("\n=== gradients ===", flush=True)
try:
    encoder.train()
    pointer.train()
    name = list(bundles)[-1]
    orb, pair = inputs[name]
    rules = ExcitationRules(orb)
    keys = encoder(torch.from_numpy(orb), torch.from_numpy(pair))
    q = torch.randn(16, HIDDEN)
    _, logp = pointer(q, keys, torch.from_numpy(pair), rules, 1.0)
    logp.sum().backward()

    params = list(encoder.parameters()) + list(pointer.parameters())
    grads = [p.grad for p in params if p.grad is not None]
    ok = (bool(grads)
          and all(torch.isfinite(g).all() for g in grads)
          and any(g.abs().sum() > 0 for g in grads))
    record(f"gradients flow through encoder + pointer (on {name})", ok,
           f"{len(grads)}/{len(params)} tensors have grad")
except Exception as exc:                                            # noqa: BLE001
    record("gradients flow through encoder + pointer", False,
           f"{type(exc).__name__}: {exc}")
    traceback.print_exc()
encoder.eval()
pointer.eval()

# ---------------------------------------------------------------------------
# Seam with the DAG GNN
# ---------------------------------------------------------------------------

print("\n=== DAG GNN seam ===", flush=True)
try:
    # dag_gnn imports torch_geometric lazily: the module import SUCCEEDS with
    # GATConv = None and only _require_pyg() raises, at construction time. So
    # guarding the import alone would report a spurious FAIL instead of a skip.
    from gqe_qsci.gqe.models import dag_gnn
    CircuitDAGGNNPolicy = dag_gnn.CircuitDAGGNNPolicy
    have_pyg = dag_gnn.GATConv is not None
except Exception as exc:                                            # noqa: BLE001
    have_pyg = False
    print(f"  dag_gnn unavailable ({type(exc).__name__}: {exc})", flush=True)

if not have_pyg:
    print("  torch_geometric not installed; SKIPPING the pooled-frontier check "
          "(not a failure — re-run with `pip install torch_geometric` to cover it).",
          flush=True)

if have_pyg:
    try:
        name = "n2" if "n2" in bundles else list(bundles)[-1]
        b = bundles[name]
        orb, pair = inputs[name]
        rules = ExcitationRules(orb)
        keys = encoder(torch.from_numpy(orb), torch.from_numpy(pair))

        policy = CircuitDAGGNNPolicy(
            vocab_size=b.vocab_size, ngates=NGATES, n_qubits=b.n_qubits,
            qubit_footprints=b.qubit_footprints, hidden_size=HIDDEN,
            num_layers=2, num_heads=4, dropout=0.0,
            commutation_matrix=b.commutation_matrix, canonical_masking=True,
            feature_scorer=True, operator_features=b.operator_features,
            orbital_features=b.orbital_features,
        )
        policy.eval()

        B, device = 4, torch.device("cpu")
        node_tokens, frontier, esrc, edst = policy._init_dag_state(B, device)
        with torch.no_grad():
            pooled = policy._step_forward(node_tokens, frontier, esrc, edst, device)

        record("DAG pooled frontier has the pointer's query shape",
               tuple(pooled.shape) == (B, HIDDEN), f"shape={tuple(pooled.shape)}")

        with torch.no_grad():
            picks, _ = pointer(pooled, keys, torch.from_numpy(pair), rules, 1.0)
        bad = [violations(tuple(int(v) for v in r), orb) for r in picks]
        record("DAG frontier drives the pointer to valid excitations",
               not any(bad), "" if not any(bad) else str([x for x in bad if x][0]))
    except Exception as exc:                                        # noqa: BLE001
        record("DAG pooled frontier drives the pointer", False,
               f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

# DAG bookkeeping from excitation footprints — no operator ids anywhere.
try:
    name = "n2" if "n2" in bundles else list(bundles)[-1]
    b = bundles[name]
    orb, pair = inputs[name]
    n = orb.shape[0]
    rules = ExcitationRules(orb)
    keys = encoder(torch.from_numpy(orb), torch.from_numpy(pair))

    B = 4
    frontier = [list(range(b.n_qubits)) for _ in range(B)]
    in_degree = [dict() for _ in range(B)]
    expected_edges = [0] * B

    for step in range(NGATES):
        gate_node = b.n_qubits + step
        with torch.no_grad():
            picks, _ = pointer(torch.randn(B, HIDDEN), keys,
                               torch.from_numpy(pair), rules, 1.0)
        for s in range(B):
            fp = excitation_qubits(tuple(int(v) for v in picks[s]), n)
            for q in fp:
                in_degree[s][gate_node] = in_degree[s].get(gate_node, 0) + 1
                frontier[s][q] = gate_node
            expected_edges[s] += len(fp)

    degree_ok = all(
        sum(in_degree[s].values()) == expected_edges[s] for s in range(B)
    )
    frontier_ok = all(
        all(0 <= v < b.n_qubits + NGATES for v in frontier[s]) for s in range(B)
    )
    record(f"DAG built for {NGATES} gates from footprints alone",
           degree_ok and frontier_ok,
           f"edges={expected_edges[0]} (sample 0), frontier in range={frontier_ok}")
except Exception as exc:                                            # noqa: BLE001
    record("DAG built from footprints alone", False,
           f"{type(exc).__name__}: {exc}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# End to end: the DAG GNN driven entirely by orbital pointers
# ---------------------------------------------------------------------------

if have_pyg:
    print("\n=== end to end: DAG GNN + pointer, no operator ids ===", flush=True)
    try:
        name = "n2" if "n2" in bundles else list(bundles)[-1]
        b = bundles[name]
        orb, pair = inputs[name]
        n = orb.shape[0]
        pair_t = torch.from_numpy(pair)
        rules = ExcitationRules(orb)

        policy = CircuitDAGGNNPolicy(
            vocab_size=b.vocab_size, ngates=NGATES, n_qubits=b.n_qubits,
            qubit_footprints=b.qubit_footprints, hidden_size=HIDDEN,
            num_layers=2, num_heads=4, dropout=0.0,
            commutation_matrix=b.commutation_matrix, canonical_masking=True,
            feature_scorer=True, operator_features=b.operator_features,
            orbital_features=b.orbital_features,
        )
        encoder.train()
        pointer.train()
        policy.train()

        B, device = 4, torch.device("cpu")
        keys = encoder(torch.from_numpy(orb), pair_t)
        _, frontier, esrc, edst = policy._init_dag_state(B, device)

        gate_embs, all_picks, all_logp = [], [], []
        for step in range(NGATES):
            gate_node = b.n_qubits + step
            node_embs = policy._assemble_node_embeddings(gate_embs, B, device)
            pooled = policy._step_forward(None, frontier, esrc, edst, device,
                                          node_embs=node_embs)          # (B, H)
            picks, logp = pointer(pooled, keys, pair_t, rules, 1.0)
            gate_embs.append(gate_embedding(picks, keys))
            all_picks.append(picks)
            all_logp.append(logp.sum(-1))

            # Wire the new gate to the frontier of each orbital it touches.
            # The footprint comes from the excitation itself — no operator table.
            for s in range(B):
                for q in excitation_qubits(tuple(int(v) for v in picks[s]), n):
                    esrc[s].append(int(frontier[s, q]))
                    edst[s].append(gate_node)
                    frontier[s, q] = gate_node

        record("end to end: node embeddings assembled without operator ids",
               tuple(node_embs.shape) == (B, b.n_qubits + NGATES, HIDDEN),
               f"shape={tuple(node_embs.shape)}")

        bad = [v for p in all_picks for r in p
               for v in [violations(tuple(int(x) for x in r), orb)] if v]
        record(f"end to end: all {B * NGATES} generated gates are valid",
               not bad, "" if not bad else str(bad[0]))

        circuit_logp = torch.stack(all_logp, dim=1).sum(-1)              # (B,)
        record("end to end: circuit log-prob finite and <= 0",
               bool(torch.isfinite(circuit_logp).all()
                    and (circuit_logp <= 1e-6).all()),
               f"mean={circuit_logp.mean().item():.2f} over {NGATES} gates")

        circuit_logp.sum().backward()
        mods = {"encoder": encoder, "pointer": pointer, "dag": policy}
        missing = [
            tag for tag, m in mods.items()
            if not any(p.grad is not None and torch.isfinite(p.grad).all()
                       and p.grad.abs().sum() > 0 for p in m.parameters())
        ]
        record("end to end: gradients reach encoder, pointer AND the DAG GNN",
               not missing, "" if not missing else f"no gradient in: {missing}")

        encoder.eval()
        pointer.eval()
    except Exception as exc:                                        # noqa: BLE001
        record("end to end: DAG GNN + pointer", False,
               f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

# ---------------------------------------------------------------------------
# What this buys — the reason for the whole exercise
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("SCALING: what the policy has to score per gate")
print("=" * 72)
print(f"{'molecule':<10}{'qubits':>7}{'pool V':>9}{'V^2 commut.':>14}"
      f"{'|S+D|':>8}{'pointer/gate':>14}")
for name, nq, V, n_sd, per_gate in scaling_rows:
    print(f"{name:<10}{nq:>7}{V:>9}{V*V:>14,}{n_sd:>8}{per_gate:>14}")
print("\n  pool V         operators the current policy scores, and the side of")
print("                 the (V,V) commutation matrix the DAG mask needs")
print("  |S+D|          the full excitation space the pointer can reach")
print("  pointer/gate   candidates actually scored: 4 steps x (n_orbitals + 1)")

# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
n_fail = sum(1 for _, ok, _ in results if not ok)
for name, ok, msg in results:
    if not ok:
        print(f"  FAIL  {name}  ({msg})")
print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
sys.exit(1 if n_fail else 0)
