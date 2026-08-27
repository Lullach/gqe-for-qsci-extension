#!/bin/bash
#PBS -N gqe_xmol_smoke
#PBS -l rt_QG=1
#PBS -l walltime=1:00:00
#PBS -j oe
#
# Phase 4 gate: verify a feature-based policy survives molecule swaps when the
# QUBIT COUNT changes (LiH 10q / H2O 12q / N2 16q). Nothing is trained here.
#
#     source hpc/env.local.sh
#     qsub -W group_list=$ABCIQ_GROUP hpc/jobs/smoke_cross_molecule.sh
#
# Read output:  cat gqe_xmol_smoke.o<jobid>

set -euo pipefail

cd "${PBS_O_WORKDIR:-$PWD}"

REPO="${REPO:-$PWD}"
SIF="${SIF:-$HOME/images/cudaq_sandbox}"

echo "=================================================="
echo "job id   : ${PBS_JOBID:-<none>}"
echo "node     : $(hostname)"
echo "repo     : $REPO"
echo "image    : $SIF"
echo "date     : $(date)"
echo "=================================================="

if [ ! -e "$SIF" ]; then
    echo "ERROR: image not found: $SIF   (build it with hpc/build_image.sh)" >&2
    exit 1
fi

echo
. "$REPO/hpc/jobs/_stage_container.sh"

echo "=== cross-molecule gate ==="
singularity exec --nv \
    --bind "$REPO:/workspace" \
    --env REPO=/workspace \
    --env PYTHONPATH=/workspace \
    --env PYTHONUNBUFFERED=1 \
    --env OMP_NUM_THREADS=1 \
    --env OMPI_MCA_pml=ob1 \
    --env OMPI_MCA_btl=self,tcp \
    --env OMPI_MCA_opal_warn_on_missing_libcuda=0 \
    "$SIF" \
    python3 /workspace/hpc/smoke_cross_molecule.py

echo
echo "=== gate finished (exit $?) ==="
