"""
Re-evaluate a trained multi-molecule checkpoint zero-shot at chosen inverse
temperatures beta, without training.

Why: in the small-molecule transfer runs (docs/run_notes/smallmol-transfer.md)
zero-shot got worse over training for gpt2-small and dag-gnn, while beta ramped
from 0.5 to 13.3 over 640 epochs. Sampling the FINAL policy at low and high beta
separates "the policy learned something worse" from "the policy is sampled too
sharply".

Same sampling as TrainPipeline._zeroshot_eval (eval mode, num_samples circuits
per batch, QSCI per circuit), with circuits simulated by the numpy sampler as in
hpc/train_fast_sampler.py. Writes one CSV row per circuit.

    python3 hpc/eval_checkpoint_beta.py experiment=smallmol_diff_gnn trainer.seed=1 \
        --ckpt outputs/gqe-for-qsci/smallmol-diff-gnn-s1/models/last.ckpt \
        --betas 0.5,1,2,4,8,13.3 --batches 5 --out scratch/beta_eval/diff-gnn-s1.csv
"""
import argparse
import csv
import os
import sys
import tempfile

import numpy as np
import torch
from hydra import compose, initialize_config_dir

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "hpc")]

from gqe_qsci.gqe.sampler import Sampler                          # noqa: E402
from gqe_qsci.factory import Factory                              # noqa: E402
from gqe_qsci.train_pipeline import TrainPipeline                 # noqa: E402
from numpy_sampler import sample_counts                           # noqa: E402

_rng = np.random.default_rng(int(os.environ.get("FAST_SAMPLER_SEED", "0")))


def _run(self, state):
    return [sample_counts(self, [int(j) for j in row], self.shots_count, _rng)
            for row in state["idx"].tolist()]


Sampler.run = _run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("overrides", nargs="+", help="hydra overrides, e.g. experiment=... trainer.seed=1")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--betas", default="0.5,1,2,4,8,13.3")
    ap.add_argument("--batches", type=int, default=5, help="batches of num_samples circuits per beta")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    with initialize_config_dir(version_base="1.3", config_dir=os.path.join(REPO, "configs")):
        cfg = compose(config_name="default",
                      overrides=[*args.overrides, f"output={tempfile.mkdtemp()}"])
    pipe = TrainPipeline(Factory(), cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # The checkpoint holds the per-molecule tensors (e.g. the operator-feature
    # table) of the LAST molecule trained on, whose vocab size differs from the
    # model's freshly built one. set_molecule() replaces them for every molecule
    # anyway, so skip exactly the shape-mismatched entries and say which.
    own = pipe.state_dict()
    sd = {k: v for k, v in ckpt["state_dict"].items()
          if k not in own or own[k].shape == v.shape}
    skipped = sorted(set(ckpt["state_dict"]) - set(sd))
    print(f"skipped (per-molecule, shape differs): {skipped}")
    missing, unexpected = pipe.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k not in skipped]
    if missing or unexpected:
        print(f"load_state_dict: missing={missing} unexpected={unexpected}")
    print(f"checkpoint beta at save: {ckpt.get('inverse_temperature', ckpt.get('extra_info'))}")
    pipe.model.eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exp_tag", "molecule", "beta", "batch", "sample_idx", "energy",
                    "R-CASCI", "err_mha", "subspace_dim", "num_sampled_basis", "seq"])
        for bundle in pipe.eval_bundles:
            pipe._activate_bundle(bundle)
            ref = float(pipe.metric_logger.reference_energies["R-CASCI"])
            for beta in [float(b) for b in args.betas.split(",")]:
                for batch in range(args.batches):
                    state = {"idx": torch.zeros((pipe.num_samples, 1), dtype=torch.long)}
                    with torch.no_grad():
                        if hasattr(pipe.model, "sample_sequence"):
                            state = pipe.model.sample_sequence(state, beta)
                        else:
                            for _ in range(pipe.ngates):
                                state = pipe.update_state(state, pipe.model.act(state, beta))
                    result = pipe.qsci_pipeline.process(state)
                    for i, s in enumerate(result.samples):
                        w.writerow([cfg.exp_tag, bundle.name, beta, batch, i, s.energy, ref,
                                    1000 * (s.energy - ref), s.subspace_dim,
                                    s.num_sampled_basis, " ".join(map(str, s.seq))])
                f.flush()
                print(f"{bundle.name} beta={beta} done", flush=True)


if __name__ == "__main__":
    main()
