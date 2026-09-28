"""
Gate for PointerDAGGNNPolicy — the POLICY, against a real OperatorPool.

hpc/smoke_pointer.py covers the action-space primitives (masks, encoder,
pointer) and drives them by hand. It never constructs the policy, so it would
stay green through a broken policy. This script covers the other half:

    factory -> policy -> sample_sequence -> pool indices -> log_prob

The load-bearing check is that the log-probability computed WHILE SAMPLING
equals the one log_prob() recovers from the stored integer indices. That path
runs through OperatorPool.ensure_excitation (which appends to the pool and
memoizes the key) and back out through excitation_keys, and if it is wrong the
GRPO importance ratio is computed against a different distribution than the one
that was sampled — silently, with no crash and no obviously wrong energy.

A stub pool cannot test that: the real pool assigns indices as sampling visits
excitations, deduplicates repeats, and computes MP2 angles on the way.

Also checked: the policy carries no operator menu / footprint table /
commutation matrix; one instance serves 10, 12 and 16 qubits with no parameter
resize; gradients reach the encoder, the pointer AND the GAT layers; and both
guardrails fire (a stale index raises, act() raises).

Needs torch_geometric (the policy is a GNN), unlike smoke_pointer.py:

    docker run --rm --entrypoint /bin/bash \
      -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp \
      -e OMPI_MCA_opal_warn_on_missing_libcuda=0 \
      -v "${PWD}:/workspace" -w /workspace gqe_qsci_cpu \
      -lc "pip install torch_geometric && python3 hpc/smoke_pointer_dag.py"
"""

import os
import sys
import traceback

import torch

REPO = os.environ.get("REPO") or (
    "/workspace" if os.path.isdir("/workspace/gqe_qsci")
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, REPO)

from hydra import compose, initialize_config_dir            # noqa: E402
from gqe_qsci.factory import Factory                        # noqa: E402
from gqe_qsci.gqe.models.pointer_dag import PointerDAGGNNPolicy  # noqa: E402

NGATES = 4
BATCH = 3
BETA = 1.0

torch.manual_seed(0)
results = []


def record(name, ok, msg=""):
    results.append((name, ok, msg))
    print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f"  ({msg})" if msg else ""),
          flush=True)


def violations(row, orb):
    """
    Physics check for one (i, j, a, b) tuple, written from the rules rather than
    from the code under test. Deliberately a second copy of the one in
    smoke_pointer.py: a shared helper could drift into agreeing with a bug.
    """
    n = orb.shape[0]
    occ = orb[:, 1] > 0
    spin = orb[:, 2].astype(int)
    i, j, a, b = (int(v) for v in row)
    bad = []

    if not (0 <= i < n and occ[i]):
        bad.append("i is not occupied")
    if not (0 <= a < n and not occ[a]):
        bad.append("a is not virtual")

    single = (j == n)
    if single != (b == n):
        bad.append("j and b disagree about arity")

    if single:
        if 0 <= i < n and 0 <= a < n and spin[a] != spin[i]:
            bad.append("single flips spin")
    else:
        if not (0 <= j < n and occ[j]):
            bad.append("j is not occupied")
        elif j <= i:
            bad.append("j <= i (non-canonical)")
        if not (0 <= b < n and not occ[b]):
            bad.append("b is not virtual")
        elif b <= a:
            bad.append("b <= a (non-canonical)")
        if 0 <= j < n and 0 <= b < n and 0 <= i < n and 0 <= a < n:
            if spin[i] + spin[j] != spin[a] + spin[b]:
                bad.append("Sz not conserved")
    return bad


def fresh_state(batch=BATCH):
    return {"idx": torch.zeros((batch, 1), dtype=torch.long)}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

print("=" * 72)
print("PointerDAGGNNPolicy against a real OperatorPool")
print("=" * 72)
print("\n=== building molecules (LiH 10q / H2O 12q / N2 16q) ===", flush=True)

with initialize_config_dir(version_base="1.3",
                           config_dir=os.path.join(REPO, "configs")):
    cfg = compose(
        config_name="default",
        overrides=[
            "molecule_set=cross_molecule",
            "model=pointer_dag",
            f"ngates={NGATES}",
            "model.hidden_size=64",
            "model.num_layers=2",
            "model.dropout=0.0",        # determinism checks below
            "operator_pool.dedup_excitations=true",
        ],
    )

factory = Factory()
bundles = factory.create_molecule_bundles(cfg)
names = list(bundles)
for name, b in bundles.items():
    print(f"  {name:<6} n_qubits={b.n_qubits:<3} initial pool V={b.vocab_size}",
          flush=True)

first = bundles[names[0]]
policy = factory.create_model(cfg, op_pool=first.pool)
policy.eval()

record("factory builds a PointerDAGGNNPolicy from model=pointer_dag",
       isinstance(policy, PointerDAGGNNPolicy), type(policy).__name__)

n_params = sum(p.numel() for p in policy.parameters())
record("policy constructed", True, f"{n_params:,} parameters")

# The claim the action space exists to make: no per-operator tables.
keys = set(policy.state_dict())
menu_like = sorted(k for k in keys
                   if any(t in k for t in ("operator_head", "commutes",
                                           "_fp_flat", "qubit_encoder")))
record("no operator menu / footprint / commutation buffers on the policy",
       not menu_like, "" if not menu_like else f"found {menu_like}")
record("orbital table is where the molecule enters",
       "space.orb_feats" in keys and "space.pair_feats" in keys,
       f"{len([k for k in keys if k.startswith('space.')])} space.* entries")

# ---------------------------------------------------------------------------
# Sampling, on each molecule in turn — ONE policy instance throughout
# ---------------------------------------------------------------------------

for name in names:
    b = bundles[name]
    print(f"\n=== {name}  ({b.n_qubits} qubits) ===", flush=True)
    if name != names[0]:
        policy.set_molecule(b)
    orb = b.pool.get_orbital_features()

    record(f"{name}: n_qubits tracks the molecule",
           policy.n_qubits == b.n_qubits,
           f"policy={policy.n_qubits} bundle={b.n_qubits}")
    record(f"{name}: parameter count unchanged by the swap",
           sum(p.numel() for p in policy.parameters()) == n_params)

    before = len(b.pool)
    torch.manual_seed(11)
    with torch.no_grad():
        state = policy.sample_sequence(fresh_state(), BETA)
    idx = state["idx"]

    record(f"{name}: sample_sequence shape and BOS column",
           tuple(idx.shape) == (BATCH, NGATES + 1) and bool(idx[:, 0].eq(0).all()),
           f"shape={tuple(idx.shape)}")

    ops = idx[:, 1:]
    in_range = bool((ops >= 0).all() and (ops < len(b.pool)).all())
    record(f"{name}: every emitted index is a real pool index", in_range,
           f"max={int(ops.max())} pool={len(b.pool)}")
    record(f"{name}: the pool grew as the policy explored",
           len(b.pool) >= before,
           f"{before} -> {len(b.pool)} operators")

    ks = b.pool.excitation_keys
    unknown = [int(k) for k in ops.reshape(-1) if int(k) not in ks]
    record(f"{name}: every index is invertible to its excitation", not unknown,
           "" if not unknown else f"{len(unknown)} unknown, e.g. {unknown[0]}")

    bad = [(int(k), violations(ks[int(k)], orb)) for k in ops.reshape(-1)
           if int(k) in ks]
    bad = [(k, v) for k, v in bad if v]
    record(f"{name}: every generated gate is a valid excitation", not bad,
           "" if not bad else f"{len(bad)} bad, e.g. {ks[bad[0][0]]} -> {bad[0][1]}")

    # -- the load-bearing check ---------------------------------------------
    # Sampling log-prob vs the one log_prob() recovers from stored INDICES.
    # _rollout is internal, but it is the only place the sampling log-prob is
    # visible; sample_sequence drops it on the floor.
    try:
        torch.manual_seed(23)
        with torch.no_grad():
            picks, sampled_logp, _ = policy._rollout(BATCH, BETA, torch.device("cpu"))
            ops2 = policy.space.to_indices(picks)
            idx2 = torch.cat([torch.zeros(BATCH, 1, dtype=torch.long), ops2], dim=1)
            replay_logp = policy.log_prob(idx2, BETA)
        delta = (replay_logp - sampled_logp).abs().max().item()
        record(f"{name}: replay through the POOL reproduces the sampling log-prob",
               delta < 1e-5, f"max|delta|={delta:.2e}")

        round_trip = policy.space.to_picks(ops2)
        record(f"{name}: indices round-trip back to the same pointers",
               bool(torch.equal(round_trip, picks)))
    except Exception as exc:                                        # noqa: BLE001
        record(f"{name}: replay through the POOL reproduces the sampling log-prob",
               False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    # -- log_prob is well-behaved -------------------------------------------
    with torch.no_grad():
        lp1, ent = policy.log_prob(idx, BETA, return_entropy=True)
        lp2 = policy.log_prob(idx, BETA)

    record(f"{name}: log_prob shape and sign",
           tuple(lp1.shape) == (BATCH, NGATES)
           and bool(torch.isfinite(lp1).all() and (lp1 <= 1e-6).all()),
           f"shape={tuple(lp1.shape)} mean={lp1.mean().item():.3f}")
    record(f"{name}: log_prob is deterministic",
           bool(torch.equal(lp1, lp2)))
    record(f"{name}: entropy finite and non-negative",
           bool(torch.isfinite(ent).all() and (ent >= -1e-6).all()),
           f"mean={ent.mean().item():.3f}")

# ---------------------------------------------------------------------------
# Gradients — all three parameter groups must move
# ---------------------------------------------------------------------------

print("\n=== gradients ===", flush=True)
try:
    policy.train()
    policy.zero_grad(set_to_none=True)
    state = policy.sample_sequence(fresh_state(), BETA)
    logp = policy.log_prob(state["idx"], BETA)
    logp.sum().backward()

    groups = {
        "orbital encoder": policy.space.encoder,
        "pointer": policy.space.pointer,
        "DAG GNN layers": policy.gnn_layers,
    }
    missing = [
        tag for tag, m in groups.items()
        if not any(p.grad is not None and torch.isfinite(p.grad).all()
                   and p.grad.abs().sum() > 0 for p in m.parameters())
    ]
    record("gradients reach encoder, pointer AND the DAG GNN", not missing,
           "" if not missing else f"no gradient in: {missing}")
    policy.eval()
except Exception as exc:                                            # noqa: BLE001
    record("gradients reach encoder, pointer AND the DAG GNN", False,
           f"{type(exc).__name__}: {exc}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# Guardrails — these must FAIL LOUDLY, not silently mis-decode
# ---------------------------------------------------------------------------

print("\n=== guardrails ===", flush=True)

stale = torch.full((BATCH, NGATES + 1), 10 ** 6, dtype=torch.long)
stale[:, 0] = 0
try:
    policy.log_prob(stale, BETA)
    record("a stale index raises instead of mis-decoding", False,
           "log_prob accepted an index this run never produced")
except RuntimeError as exc:
    record("a stale index raises instead of mis-decoding", True,
           f"RuntimeError: {str(exc)[:60]}...")
except Exception as exc:                                            # noqa: BLE001
    record("a stale index raises instead of mis-decoding", False,
           f"raised {type(exc).__name__}, expected RuntimeError")

try:
    policy.act(fresh_state(), BETA)
    record("act() refuses (whole-sequence policy)", False, "act() returned")
except RuntimeError:
    record("act() refuses (whole-sequence policy)", True)
except Exception as exc:                                            # noqa: BLE001
    record("act() refuses (whole-sequence policy)", False,
           f"raised {type(exc).__name__}, expected RuntimeError")

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
