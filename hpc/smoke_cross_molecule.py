"""
Phase 4 gate: does a feature-based policy survive a molecule swap when the
QUBIT COUNT changes?

Everything cross-molecule so far (Phase 3, the N2 bond scan) kept n_qubits
fixed at 16 — only the feature VALUES changed. Here LiH (10q), H2O (12q) and
N2 (16q) have different qubit counts, different active spaces and different
vocabulary sizes, which exercises code paths that have never run:

  - DAG GNN: node count is n_qubits + ngates, so the graph itself resizes;
    _fp_flat, commutes and orbital_features all change shape on set_molecule
  - OperatorScorer: the menu changes V, so the buffer must be re-registered
  - frozen normalization: stats pooled over molecules of different sizes

Checks, per model:
  1. bundles build (pool + features) for all three molecules
  2. set_molecule swaps cleanly between every pair, in both directions
  3. after each swap, sample_sequence produces in-range operator ids
  4. log_prob on those samples is finite (this is what GRPO consumes)
  5. gradients flow after a swap (weights are shared across molecules)

Run inside the container on a GPU node: see hpc/jobs/smoke_cross_molecule.sh
"""

import os
import sys
import traceback

import torch
from hydra import compose, initialize_config_dir

REPO = os.environ.get("REPO", "/workspace")
sys.path.insert(0, REPO)

from gqe_qsci.factory import Factory  # noqa: E402

MODELS = ["dag_gnn_features", "diffusion_singleshot_features",
          "diffusion_absorbing_matched_features"]

results = []


def record(name, ok, msg=""):
    results.append((name, ok, msg))
    print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f"  ({msg})" if msg else ""), flush=True)


print("=" * 66)
print("Phase 4 gate: molecule swap across DIFFERENT qubit counts")
print("=" * 66)

# ---------------------------------------------------------------- bundles
print("\n=== building bundles (LiH 10q / H2O 12q / N2 16q) ===", flush=True)
with initialize_config_dir(version_base="1.3", config_dir=os.path.join(REPO, "configs")):
    cfg = compose(
        config_name="default",
        overrides=["molecule_set=cross_molecule", "model=dag_gnn_features",
                   "operator_pool.dedup_excitations=true"],
    )

factory = Factory()
bundles = factory.create_molecule_bundles(cfg)
for n, b in bundles.items():
    print(f"  {n:<6} split={b.split:<5} n_qubits={b.n_qubits:<3} "
          f"V={b.vocab_size:<5} feat_dim={b.feat_dim}")

qubit_counts = {b.n_qubits for b in bundles.values()}
record("bundles have DIFFERENT qubit counts", len(qubit_counts) > 1,
       f"n_qubits seen: {sorted(qubit_counts)}")
record("all bundles share feat_dim", len({b.feat_dim for b in bundles.values()}) == 1)

names = list(bundles)

# ---------------------------------------------------------------- per model
for model_name in MODELS:
    print(f"\n=== model: {model_name} ===", flush=True)
    try:
        with initialize_config_dir(version_base="1.3",
                                   config_dir=os.path.join(REPO, "configs")):
            mcfg = compose(
                config_name="default",
                overrides=["molecule_set=cross_molecule", f"model={model_name}",
                           "operator_pool.dedup_excitations=true"],
            )
        first = bundles[names[0]]
        model = factory.create_model(mcfg, op_pool=first.pool)
        model.eval()
        record(f"{model_name}: constructed", True)
    except Exception as exc:  # noqa: BLE001
        record(f"{model_name}: constructed", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        continue

    # swap between every ordered pair, so both grow and shrink are covered
    for src in names:
        for dst in names:
            if src == dst:
                continue
            tag = f"{model_name}: {src}({bundles[src].n_qubits}q) -> {dst}({bundles[dst].n_qubits}q)"
            try:
                model.set_molecule(bundles[src])
                model.set_molecule(bundles[dst])
                b = bundles[dst]

                state = {"idx": torch.zeros(2, 1, dtype=torch.long)}
                with torch.no_grad():
                    out = model.sample_sequence(state, 1.0)
                gen = out["idx"][:, 1:]

                in_range = bool((gen >= 0).all() and (gen < b.vocab_size).all())
                shape_ok = gen.shape == (2, cfg.ngates)

                lp_kwargs = {}
                if "reveal_step" in out:
                    lp_kwargs["reveal_step"] = out["reveal_step"]
                with torch.no_grad():
                    lp = model.log_prob(out["idx"], 1.0, **lp_kwargs)
                finite = bool(torch.isfinite(lp).all())

                record(tag, in_range and shape_ok and finite,
                       "" if (in_range and shape_ok and finite)
                       else f"in_range={in_range} shape={tuple(gen.shape)} finite={finite}")
            except Exception as exc:  # noqa: BLE001
                record(tag, False, f"{type(exc).__name__}: {exc}")
                traceback.print_exc()

    # gradients must still flow on the molecule we ended on
    try:
        model.train()
        b = bundles[names[-1]]
        model.set_molecule(b)
        state = {"idx": torch.zeros(2, 1, dtype=torch.long)}
        with torch.no_grad():
            out = model.sample_sequence(state, 1.0)
        lp_kwargs = {"reveal_step": out["reveal_step"]} if "reveal_step" in out else {}
        lp = model.log_prob(out["idx"], 1.0, **lp_kwargs)
        lp.sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        ok = bool(grads) and all(torch.isfinite(g).all() for g in grads) \
            and any(g.abs().sum() > 0 for g in grads)
        record(f"{model_name}: gradients flow after swap", ok)
    except Exception as exc:  # noqa: BLE001
        record(f"{model_name}: gradients flow after swap", False,
               f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

# ---------------------------------------------------------------- summary
print("\n" + "=" * 66)
print("SUMMARY")
print("=" * 66)
n_fail = sum(1 for _, ok, _ in results if not ok)
for name, ok, msg in results:
    if not ok:
        print(f"  FAIL  {name}  ({msg})")
print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
sys.exit(1 if n_fail else 0)
