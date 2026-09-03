#!/bin/bash
#PBS -N gqe_train
#PBS -l rt_QG=1
#PBS -l walltime=24:00:00
#PBS -j oe
#
# Training run on 1 H100 (rt_QG = 32 cores + 1 GPU).
#
# -W group_list is PERSONAL and NOT baked into this script (see smoke_gpu.sh
# for why) — pass it at submit time:
#     source hpc/env.local.sh
#
# Submit with the experiment name (and optionally extra hydra overrides):
#     qsub -W group_list=$ABCIQ_GROUP -v EXPERIMENT=n2_bond_scan_dag_gnn hpc/jobs/train.sh
#     qsub -W group_list=$ABCIQ_GROUP -v EXPERIMENT=n2_l10_gpt2_paper,EXTRA="trainer.max_iters=50" hpc/jobs/train.sh
#
# Override walltime per submission if needed:
#     qsub -W group_list=$ABCIQ_GROUP -l walltime=4:00:00 -v EXPERIMENT=... hpc/jobs/train.sh
#
# For a full node (4 GPUs, needed only for MPI multi-QPU sampling):
#     change to  #PBS -l rt_QF=1  and see hpc/README.md
#
# W&B runs OFFLINE (compute nodes have no outbound internet); sync afterwards
# from the login node - the command is printed at the end of this script.

set -euo pipefail

cd "${PBS_O_WORKDIR:-$PWD}"

REPO="${REPO:-$PWD}"
SIF="${SIF:-$HOME/images/cudaq_sandbox}"
EXPERIMENT="${EXPERIMENT:?set EXPERIMENT, e.g. qsub -v EXPERIMENT=n2_bond_scan_dag_gnn hpc/jobs/train.sh}"
EXTRA="${EXTRA:-}"

echo "=================================================="
echo "job id     : ${PBS_JOBID:-<none>}"
echo "node       : $(hostname)"
echo "repo       : $REPO"
echo "image      : $SIF"
echo "experiment : $EXPERIMENT"
echo "extra      : ${EXTRA:-<none>}"
echo "date       : $(date)"
echo "=================================================="

if [ ! -e "$SIF" ]; then
    echo "ERROR: image not found: $SIF   (build it with hpc/build_image.sh)" >&2
    exit 1
fi

nvidia-smi || echo "WARNING: nvidia-smi failed on the host"

echo
# Copy the sandbox to node-local scratch: Lustre cannot back the overlay that
# --nv requires. See _stage_container.sh.
. "$REPO/hpc/jobs/_stage_container.sh"

echo "=== training ==="
# Environment is set EXPLICITLY here rather than relying on the def file's
# %environment: converting a .sif to a sandbox regenerates .singularity.d/ and
# drops it, so PYTHONPATH silently vanished and `import gqe_qsci` failed.
singularity exec --nv \
    --bind "$REPO:/workspace" \
    --env PYTHONPATH=/workspace \
    --env PYTHONUNBUFFERED=1 \
    --env OMP_NUM_THREADS=1 \
    --env OMPI_MCA_pml=ob1 \
    --env OMPI_MCA_btl=self,tcp \
    --env OMPI_MCA_opal_warn_on_missing_libcuda=0 \
    --env WANDB_MODE=offline \
    --env WANDB_DIR=/workspace/outputs \
    --env HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}" \
    --workdir /workspace \
    "$SIF" \
    python3 /workspace/train.py experiment="$EXPERIMENT" $EXTRA

echo
echo "=== training finished ==="
echo
echo "W&B ran offline. Sync from the LOGIN node with:"
echo "  singularity exec $SIF wandb sync $REPO/outputs/gqe-for-qsci/*/wandb/offline-run-*"
