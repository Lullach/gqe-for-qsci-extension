#!/bin/bash
#PBS -N gqe_packed
#PBS -l rt_QG=1
#PBS -l walltime=24:00:00
#PBS -j oe
#
# MANY training runs in ONE job, sharing the node's single GPU.
#
# Why: rt_QG is a fixed bundle of 32 cores + 1 GPU, billed per hour whatever is
# used, and one training run uses ONE core (every job in the PBS history shows
# cput/walltime = 0.97-0.99). A 5-seed sweep submitted as a job array therefore
# rents five bundles to use five cores. Packed, it rents one.
#
# It pays off as long as packing N runs slows each of them by less than N x:
# points = 2/hour x walltime, so N runs in one job that takes k times longer
# cost k/N of what N separate jobs cost.
#
#     source hpc/env.local.sh
#     qsub -W group_list=$ABCIQ_GROUP \
#          -v EXPERIMENTS=archL15_dag_gnn+archL15_gpt2,SEEDS=1-5 hpc/jobs/packed.sh
#
#   EXPERIMENTS  experiment names separated by + (PBS cannot carry a comma
#                inside a -v value: it reads commas as separating variables)
#   SEEDS        a range "1-5" or a list "1+3+5". Default: 1
#   MAXPAR       runs alive at once. Default: 32 (one per core). More runs than
#                this queue up inside the job and start as others finish.
#   EXTRA        hydra overrides applied to every run, e.g. EXTRA="trainer.max_iters=50"
#
# Runs = EXPERIMENTS x SEEDS. Each gets its own log, output directory and W&B
# run, exactly as if it had been submitted alone.
#
# WHAT LIMITS MAXPAR is memory, and it has to be measured: single-run jobs
# reported 16-29 GB against this resource type's 200 GB, but part of that is
# the one-off container copy, not the run. resources.log (next to the run logs)
# samples total RAM and GPU memory every two minutes; read it after a first
# small job before asking for 32.
#
# Read output:  cat gqe_packed.o<jobid>      (summary table at the end)
#               ls  logs/packed_<jobid>/      (one log per run + resources.log)

# No -e: one failed run must not take the others down. No -u either: bash
# treats an EMPTY array as unset under -u, which broke the run bookkeeping.
# The variables that must be set are checked explicitly below.
set -o pipefail

cd "${PBS_O_WORKDIR:-$PWD}"

REPO="${REPO:-$PWD}"
SIF="${SIF:-$HOME/images/cudaq_sandbox}"
EXPERIMENTS="${EXPERIMENTS:?set EXPERIMENTS, e.g. -v EXPERIMENTS=archL15_dag_gnn+archL15_gpt2}"
SEEDS="${SEEDS:-1}"
MAXPAR="${MAXPAR:-32}"
EXTRA="${EXTRA:-}"

# "1-5" -> 1 2 3 4 5 ; "1+3+5" -> 1 3 5
if [[ "$SEEDS" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    SEED_LIST="$(seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}")"
else
    SEED_LIST="$(echo "$SEEDS" | tr '+' ' ')"
fi
EXP_LIST="$(echo "$EXPERIMENTS" | tr '+' ' ')"

JOBTAG="${PBS_JOBID:-local$$}"
LOGDIR="$REPO/logs/packed_${JOBTAG%%.*}"
mkdir -p "$LOGDIR"

n_runs=0
for e in $EXP_LIST; do for s in $SEED_LIST; do n_runs=$((n_runs + 1)); done; done

echo "=================================================="
echo "job id      : ${PBS_JOBID:-<none>}"
echo "node        : $(hostname)"
echo "experiments : $EXP_LIST"
echo "seeds       : $(echo $SEED_LIST)"
echo "runs        : $n_runs   (at most $MAXPAR at once)"
echo "extra       : ${EXTRA:-<none>}"
echo "logs        : $LOGDIR"
echo "date        : $(date)"
echo "=================================================="

if [ ! -e "$SIF" ]; then
    echo "ERROR: image not found: $SIF   (build it with hpc/build_image.sh)" >&2
    exit 1
fi
nvidia-smi || echo "WARNING: nvidia-smi failed on the host"
echo

# One container copy for all runs.
. "$REPO/hpc/jobs/_stage_container.sh"

# --- resource sampler: the evidence for how many runs actually fit ----------
(
    echo "time,runs_alive,ram_used_gb,ram_total_gb,gpu_util_pct,gpu_mem_used_mb,gpu_mem_total_mb"
    while true; do
        # anchored on python3: the singularity launcher carries the same
        # arguments and would otherwise count every run twice
        alive=$(pgrep -u "$(id -u)" -f "^python3 /workspace/train.py" | wc -l)
        ram=$(free -g | awk '/^Mem:/ {print $3 "," $2}')
        gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
                         --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        echo "$(date +%H:%M:%S),$alive,$ram,$gpu"
        sleep 120
    done
) > "$LOGDIR/resources.log" 2>&1 &
SAMPLER=$!

# --- launch ------------------------------------------------------------------
declare -A NAME=() START=()
launch() {
    local exp="$1" seed="$2" name="$1-s$2"
    # Environment set explicitly, as in train.sh: a sandbox drops %environment.
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
        --env HYDRA_FULL_ERROR=1 \
        --workdir /workspace \
        "$SIF" \
        python3 /workspace/train.py experiment="$exp" trainer.seed="$seed" $EXTRA \
        > "$LOGDIR/$name.log" 2>&1 &
    NAME[$!]="$name"
    START[$!]=$(date +%s)
    echo "started  $name  (pid $!)"
}

declare -A STATUS=() SECS=()
reap() {                       # collect whichever runs have finished
    local pid
    for pid in "${!NAME[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid"; STATUS[${NAME[$pid]}]=$?
            SECS[${NAME[$pid]}]=$(( $(date +%s) - START[$pid] ))
            echo "finished ${NAME[$pid]}  exit ${STATUS[${NAME[$pid]}]}  after $(( SECS[${NAME[$pid]}] / 60 )) min"
            unset "NAME[$pid]"
        fi
    done
}

for exp in $EXP_LIST; do
    for seed in $SEED_LIST; do
        while [ "${#NAME[@]}" -ge "$MAXPAR" ]; do sleep 15; reap; done
        launch "$exp" "$seed"
        sleep 3            # spread the start-up load (imports, SCF) a little
    done
done
while [ "${#NAME[@]}" -gt 0 ]; do sleep 15; reap; done

kill "$SAMPLER" 2>/dev/null

# --- summary -----------------------------------------------------------------
echo
echo "=== summary ==="
failed=0
for name in $(printf '%s\n' "${!STATUS[@]}" | sort); do
    if [ "${STATUS[$name]}" -eq 0 ]; then state="ok"; else state="FAILED (exit ${STATUS[$name]})"; failed=$((failed + 1)); fi
    printf '  %-34s %4d min  %s\n' "$name" "$(( SECS[$name] / 60 ))" "$state"
done
echo
echo "peak of the sampled resources (see $LOGDIR/resources.log):"
awk -F, 'NR>1 { if ($2>a) a=$2; if ($3>r) r=$3; t=$4; if ($6>g) g=$6; gt=$7 }
         END { printf "  runs alive %d | RAM %d of %d GB | GPU memory %d of %d MB\n", a, r, t, g, gt }' \
    "$LOGDIR/resources.log"
echo
echo "W&B ran offline. Sync from the LOGIN node with:"
echo "  singularity exec $SIF wandb sync $REPO/outputs/gqe-for-qsci/*/wandb/offline-run-*"

if [ "$failed" -gt 0 ]; then
    echo
    echo "$failed run(s) FAILED - read their logs in $LOGDIR"
    exit 1
fi
