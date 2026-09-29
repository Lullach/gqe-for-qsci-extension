#!/bin/bash
#PBS -N gqe_subspace
#PBS -l rt_QC=1
#PBS -l walltime=24:00:00
#PBS -j oe
#
# Subspace-requirement scan on the LARGE systems (hpc/subspace_requirement.py).
#
# rt_QC, NOT rt_QG: this is pyci Davidson on the CPU. There is no GPU work at
# all, so a GPU queue would bill GPU rates for an idle card.
#
#     source hpc/env.local.sh
#     qsub -W group_list=$ABCIQ_GROUP -v FAMILIES=h12,c2h2,co2,n2b hpc/jobs/subspace.sh
#
# Optional:
#     -v FAMILIES=...,MAXFCI=1000000,TARGET=1.6,OUT=subspace_large.csv
#
# Resumable by design: the CSV is appended and flushed per configuration, and
# anything already in it is skipped. If this hits the walltime, resubmit the
# same command and it picks up where it stopped.
#
# Read output:  cat gqe_subspace.o<jobid>

set -euo pipefail

cd "${PBS_O_WORKDIR:-$PWD}"

REPO="${REPO:-$PWD}"
SIF="${SIF:-$HOME/images/cudaq_sandbox}"

FAMILIES="${FAMILIES:-h12,c2h2,co2,n2b}"
MAXFCI="${MAXFCI:-1000000}"
TARGET="${TARGET:-1.6}"
OUT="${OUT:-subspace_large.csv}"

echo "=================================================="
echo "job id   : ${PBS_JOBID:-<none>}"
echo "node     : $(hostname)"
echo "families : $FAMILIES"
echo "max_fci  : $MAXFCI"
echo "target   : $TARGET mHa"
echo "out      : $OUT"
echo "date     : $(date)"
echo "=================================================="

if [ ! -e "$SIF" ]; then
    echo "container not found at $SIF" >&2
    exit 1
fi

. "$REPO/hpc/jobs/_stage_container.sh"

# No --nv: there is no GPU in rt_QC and nothing here uses one.
singularity exec \
    --bind "$REPO:/workspace" \
    --env PYTHONPATH=/workspace \
    --env PYTHONUNBUFFERED=1 \
    --env OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}" \
    --workdir /workspace \
    "$SIF" \
    python3 /workspace/hpc/subspace_requirement.py \
        --families "$FAMILIES" \
        --max-fci "$MAXFCI" \
        --target "$TARGET" \
        --out "/workspace/$OUT"

echo
echo "=== finished ==="
echo "Resubmit the same command to continue; finished configurations are skipped."
