#!/bin/bash
#PBS -N gqe_packed_cpu
#PBS -l rt_QC=1
#PBS -l walltime=24:00:00
#PBS -j oe
#
# packed.sh on rt_QC: many training runs in one CPU-only job, one core each.
#
# Why: rt_QC costs half of rt_QG per hour in every category, and packing on
# rt_QG does not pay because the runs queue for the shared GPU (job 220006:
# 8 runs took 67 min each instead of ~9). Here circuits are simulated in numpy
# by hpc/train_fast_sampler.py, verified at fidelity 1 against cudaq, so the
# runs share nothing and 32 can run side by side.
#
#     source hpc/env.local.sh
#     qsub -W group_list=$ABCIQ_GROUP \
#          -v EXPERIMENTS=n2_pool_matched,SEEDS=1-8 hpc/jobs/packed_cpu.sh
#
# Same variables as packed.sh (EXPERIMENTS, SEEDS, MAXPAR, EXTRA).
#
# Caveats:
#   * shots come from numpy's generator, not cudaq's: statistically
#     equivalent runs, not bit-identical ones;
#   * the policy trains on the CPU. Small models (DAG GNN, 437K parameters)
#     do not notice; GPT-2 (~43M) will be markedly slower per step;
#   * how much memory rt_QC grants per job is not measured yet -
#     resources.log reports what the runs use.

export PACKED_MODE=cpu
. "${PBS_O_WORKDIR:-$PWD}/hpc/jobs/packed.sh"
