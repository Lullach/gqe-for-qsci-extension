#!/bin/bash
#PBS -N gqe_smoke
#PBS -l rt_QG=1
#PBS -l walltime=0:30:00
#PBS -j oe
#
# Smoke test: 1 GPU, verifies the container + GPU + project stack on ABCI-Q.
# Nothing here trains anything; it only proves the environment works.
#
# -W group_list is PERSONAL (your account's allocation group) and is NOT baked
# into this script — PBS #PBS directives are parsed literally, before any
# shell variable exists, so it must be passed at submit time instead:
#
#     source hpc/env.local.sh   # sets ABCIQ_GROUP (see env.local.sh.example)
#     qsub -W group_list=$ABCIQ_GROUP hpc/jobs/smoke_gpu.sh
#
# Watch it:      qstat
# Read output:   cat gqe_smoke.o<jobid>

set -euo pipefail

cd "${PBS_O_WORKDIR:-$PWD}"

REPO="${REPO:-$PWD}"
SIF="${SIF:-$HOME/images/cudaq_sandbox}"

echo "=================================================="
echo "job id   : ${PBS_JOBID:-<none>}"
echo "node     : $(hostname)"
echo "workdir  : $PWD"
echo "repo     : $REPO"
echo "image    : $SIF"
echo "date     : $(date)"
echo "=================================================="

if [ ! -e "$SIF" ]; then
    echo "ERROR: image not found: $SIF   (build it with hpc/build_image.sh)" >&2
    exit 1
fi

echo
echo "=== nvidia-smi (host) ==="
nvidia-smi || echo "WARNING: nvidia-smi failed on the host"

echo
# Copy the sandbox to node-local scratch: Lustre cannot back the overlay that
# --nv requires. See _stage_container.sh.
. "$REPO/hpc/jobs/_stage_container.sh"

echo
echo "=== running smoke test inside the container ==="
# --nv  : expose the NVIDIA GPU + driver into the container
# --bind: mount the repo at /workspace (the image contains no project code)
singularity exec --nv \
    --bind "$REPO:/workspace" \
    --env REPO=/workspace \
    "$SIF" \
    python3 /workspace/hpc/smoke_test.py

echo
echo "=== smoke test finished (exit $?) ==="
