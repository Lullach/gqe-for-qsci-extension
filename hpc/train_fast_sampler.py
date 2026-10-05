"""
train.py, with the numpy statevector sampler in place of cudaq's.

For anywhere WITHOUT a GPU: the laptop, and rt_QC on ABCI-Q (via
hpc/jobs/packed_cpu.sh). On the CPU target cudaq takes ~55 s to simulate one
20-qubit circuit; the numpy simulator in hpc/numpy_sampler.py does the same
circuit in well under a second and was checked against cudaq.get_state at
fidelity 1.000000000000 on every pool it has been used with. Everything else —
the policy, the QSCI pipeline, refinement, logging — is the unmodified training
code.

Why not just use the GPU: runs packed onto one GPU queue for it (8 runs took
67 min each instead of ~9), while rt_QC costs half of rt_QG per hour. See
hpc/jobs/packed.sh.

Shots are drawn with numpy's generator instead of cudaq's, so a run here is a
statistically equivalent run, not a bit-identical one.

Arguments are passed straight to train.py:

    docker run --rm --entrypoint /bin/bash -v "${PWD}:/workspace" -w /workspace \
        -e WANDB_MODE=offline -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp \
        gqe_qsci_cpu -lc "python3 hpc/train_fast_sampler.py experiment=n2_pool_matched"
"""

import os
import runpy
import sys

import numpy as np

REPO = os.environ.get("REPO") or (
    "/workspace" if os.path.isdir("/workspace/gqe_qsci")
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "hpc"))

from gqe_qsci.gqe.sampler import Sampler                          # noqa: E402
from numpy_sampler import sample_counts                           # noqa: E402

_rng = np.random.default_rng(int(os.environ.get("FAST_SAMPLER_SEED", "0")))


def _run(self, state):
    return [sample_counts(self, [int(j) for j in row], self.shots_count, _rng)
            for row in state["idx"].tolist()]


Sampler.run = _run
print("train_fast_sampler: numpy statevector sampler active (local runs only)")

sys.argv = [os.path.join(REPO, "train.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
