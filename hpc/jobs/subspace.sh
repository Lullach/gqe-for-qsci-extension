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
#     for f in h12 c2h2 co2 n2b; do
#       qsub -W group_list=$ABCIQ_GROUP -v FAMILIES=$f hpc/jobs/subspace.sh
#     done
#
# ONE JOB PER FAMILY, deliberately. PBS reads -v as a comma-separated list of
# VARIABLES, so `-v FAMILIES=h12,c2h2` means "FAMILIES=h12, plus export c2h2
# from my shell" and fails with "cannot send environment with the job". Separate
# jobs also queue in parallel and stop a slow family blocking the rest.
#
# To pass several families to one job anyway, separate them with + or a space:
#     -v FAMILIES=h12+c2h2
#
# Optional:  -v FAMILIES=...,MAXFCI=1000000,TARGET=1.6,OUT=my.csv
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

FAMILIES="${FAMILIES:-h12}"
# Accept + or spaces as separators, since PBS cannot carry a comma in -v.
FAMILIES="$(echo "$FAMILIES" | tr '+ ' ',,' | sed 's/,,*/,/g; s/^,//; s/,$//')"
MAXFCI="${MAXFCI:-1000000}"
TARGET="${TARGET:-1.6}"
# One CSV per family by default. Several jobs appending to one file would
# interleave rows mid-line; merge them afterwards instead, which the plotting
# script does anyway (it takes a glob).
OUT="${OUT:-subspace_$(echo "$FAMILIES" | tr ',' '_').csv}"

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
echo
echo "Merge and plot (needs matplotlib, so do this off the cluster):"
echo "  python3 hpc/plot_subspace_requirement.py --csv subspace_requirement.csv"
