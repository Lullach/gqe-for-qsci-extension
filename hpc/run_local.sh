#!/bin/bash
# packed_cpu.sh for the laptop: many training runs side by side, one Docker
# container (gqe_qsci_cpu) per run, numpy sampler (hpc/train_fast_sampler.py).
#
# From the repo root, in Git Bash:
#
#     hpc/run_local.sh "n2L15_gpt2_small_lr8 lifL15_gpt2_small_lr8" 1-5
#
#   $1  experiment names, space-separated, in quotes
#   $2  seeds, e.g. 1-5 or "1 3 5" (default 1-5)
#   $3  how many runs at once (default 8; Docker here has 14 CPUs and 16 GB,
#       and one small run needs ~1 GB)
#
# EXTRA="..." adds Hydra overrides to every run, e.g. EXTRA="trainer.max_iters=1"
# for a smoke test.
#
# torch_geometric (needed by the GNN policies) is not in the local image, unlike
# the cluster's: each container installs it if missing, from a pip cache kept
# in the Docker volume gqe_pip_cache, so only the first run downloads it.
#
# Same layout as on ABCI-Q: results in outputs/gqe-for-qsci/<exp_tag>/, one log
# per run in logs/local_<timestamp>/, and a final ok/FAILED list. W&B is offline.

set -u
EXPERIMENTS="${1:?usage: hpc/run_local.sh \"exp1 exp2\" [seeds] [maxpar]}"
SEEDS="${2:-1-5}"
MAXPAR="${3:-8}"

case "$SEEDS" in
    *-*) SEEDS="$(seq "${SEEDS%-*}" "${SEEDS#*-}")" ;;
esac

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="$REPO/logs/local_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOGDIR"
echo "logs: $LOGDIR"

run() {   # $1 experiment, $2 seed
    MSYS_NO_PATHCONV=1 docker run --rm --entrypoint /bin/bash \
        -v "$REPO:/workspace" -v gqe_pip_cache:/root/.cache/pip -w /workspace \
        -e PYTHONPATH=/workspace:/workspace/hpc -e PYTHONUNBUFFERED=1 \
        -e OMP_NUM_THREADS=1 -e WANDB_MODE=offline -e HYDRA_FULL_ERROR=1 \
        -e OMPI_MCA_pml=ob1 -e OMPI_MCA_btl=self,tcp \
        -e OMPI_MCA_opal_warn_on_missing_libcuda=0 \
        gqe_qsci_cpu -c "python3 -c 'import torch_geometric' 2>/dev/null || pip install -q torch_geometric; python3 hpc/train_fast_sampler.py experiment=$1 trainer.seed=$2 ${EXTRA:-}" \
        > "$LOGDIR/$1-s$2.log" 2>&1
}

pids=() names=()
for exp in $EXPERIMENTS; do
    for seed in $SEEDS; do
        while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ]; do sleep 5; done
        run "$exp" "$seed" &
        pids+=($!) names+=("$exp-s$seed")
        echo "started  $exp-s$seed"
    done
done

failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "  ok      ${names[$i]}"
    else echo "  FAILED  ${names[$i]}  (see $LOGDIR/${names[$i]}.log)"; failed=$((failed + 1)); fi
done
echo "done: ${#pids[@]} run(s), $failed failed"
