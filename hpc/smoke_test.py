"""
ABCI-Q smoke test: verify the container, the GPU, and the project stack.

Run inside the Singularity image on a GPU node (see hpc/jobs/smoke_gpu.sh).
Checks, in dependency order, so the first failure tells you what to fix:

  1. torch sees a CUDA device (proves --nv passed the GPU into the container)
  2. CUDA-Q imports, the `nvidia` GPU target works, a tiny kernel samples
  3. project dependencies import (pyscf, pyci, tequila, torch_geometric, ...)
  4. gqe_qsci imports from the bind-mounted /workspace
  5. a real (tiny) molecule + operator pool builds end to end

Exit code 0 = everything works.
"""

import sys
import traceback

results = []


def check(name):
    """Decorator: run a check, record pass/fail, never abort the whole script."""
    def wrap(fn):
        print(f"\n=== {name} ===", flush=True)
        try:
            fn()
            results.append((name, True, ""))
            print(f"  -> OK", flush=True)
        except Exception as exc:  # noqa: BLE001 - we want every failure reported
            results.append((name, False, f"{type(exc).__name__}: {exc}"))
            print(f"  -> FAILED: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
        return fn
    return wrap


@check("1. torch + CUDA")
def _torch():
    import torch
    print(f"  torch {torch.__version__}")
    print(f"  cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "no CUDA device visible - did the job request a GPU resource "
            "(-l rt_QG=1) and did singularity get --nv ?"
        )
    print(f"  device count : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"    [{i}] {torch.cuda.get_device_name(i)}")
    # Actually touch the GPU - availability alone doesn't prove it computes.
    x = torch.randn(1000, 1000, device="cuda")
    _ = (x @ x).sum().item()
    print("  matmul on GPU: OK")


@check("2. CUDA-Q GPU target")
def _cudaq():
    import cudaq
    print(f"  cudaq {cudaq.__version__}")
    targets = [t.name for t in cudaq.get_targets()]
    print(f"  targets: {sorted(set(targets))}")

    # nvidia = GPU statevector simulator; this is what makes ABCI-Q worthwhile.
    cudaq.set_target("nvidia")
    print(f"  set_target('nvidia'): OK")

    @cudaq.kernel
    def bell():
        q = cudaq.qvector(2)
        h(q[0])
        x.ctrl(q[0], q[1])

    counts = cudaq.sample(bell, shots_count=1000)
    d = {k: counts[k] for k in counts}
    print(f"  bell state counts: {d}")
    if set(d) - {"00", "11"}:
        raise RuntimeError(f"unexpected bitstrings in Bell state: {sorted(d)}")


@check("3. project dependencies")
def _deps():
    import importlib
    mods = [
        "numpy", "scipy", "pyscf", "pyci", "tequila",
        "pytorch_lightning", "transformers", "hydra", "omegaconf", "wandb",
        "torch_geometric",
    ]
    for m in mods:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "?")
        print(f"  {m:<20} {ver}")


@check("4. gqe_qsci import")
def _project():
    import gqe_qsci
    from gqe_qsci.factory import Factory
    from gqe_qsci.molecule import PySCFMolecule
    print(f"  gqe_qsci from: {gqe_qsci.__file__}")
    print(f"  Factory, PySCFMolecule: OK")


@check("5. molecule + operator pool (H2, cheapest real case)")
def _molecule():
    from hydra import compose, initialize_config_dir
    import os

    cfg_dir = os.path.join(os.environ.get("REPO", "/workspace"), "configs")
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg = compose(config_name="default", overrides=["molecule=h2"])

    from gqe_qsci.factory import Factory
    factory = Factory()
    mol = factory.create_molecule(cfg)
    pool = factory.create_operator_pool(cfg)
    print(f"  H2: n_qubits={pool.n_qubits}, vocab_size={pool.get_vocab_size()}")
    print(f"  HF energy   : {mol.hf.e_tot:.8f} Ha")
    feats = pool.get_operator_features()
    print(f"  feature menu: {feats.shape}")


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for name, ok, msg in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({msg})" if msg else ""))

n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
sys.exit(1 if n_fail else 0)
