# Running GQE-QSCI on ABCI-Q (System H)

Setup and job scripts for AIST G-QuAT's **ABCI-Q System H** — 505 nodes, each
with 4× NVIDIA H100 (80 GB). Docs: <https://g-quat-abciq.github.io/abciq-docs/en/>

Your username and allocation group are **personal account info and are not
committed to this repo**. Copy `hpc/env.local.sh.example` to `hpc/env.local.sh`
(gitignored) and fill them in:

```bash
cp hpc/env.local.sh.example hpc/env.local.sh
# edit hpc/env.local.sh: set ABCIQ_USER and ABCIQ_GROUP
source hpc/env.local.sh
```

Every PBS job **must** carry `-W group_list=$ABCIQ_GROUP` on the `qsub` command
line (PBS `#PBS` directives are parsed literally before any shell variable
exists, so the group can't be baked into the job scripts themselves — see the
comments in `jobs/*.sh`).

## What is different from the local Docker workflow

| | Local | ABCI-Q |
|---|---|---|
| Access | — | 2-hop SSH: `qas.q.abci.ai` → `qes` |
| Run | `docker run` | PBS Pro batch jobs (`qsub`) |
| Container | Docker | SingularityCE 4.1.5 (`.sif`) |
| Hardware | CPU | H100 GPU (`nvidia` CUDA-Q target) |
| Code in image | baked in (`COPY . .`) | **bind-mounted** at `/workspace` |
| W&B | online | **offline**, synced afterwards from the login node |

The bind-mount is the important one: edit code locally, `git pull` on the
cluster, re-run — no image rebuild.

## Files

```
hpc/
├── README.md              this file
├── env.local.sh.example   template for your PERSONAL username/group (copy, fill in, never commit)
├── cudaq_qsci.def         Singularity definition (mirrors ../dockerfile)
├── build_image.sh         builds the .sif on the login node
├── smoke_test.py          verifies GPU + CUDA-Q + project stack
├── ssh_config.example     ~/.ssh/config entry for `ssh abciq` (generic, no real username)
└── jobs/
    ├── smoke_gpu.sh       1 GPU, 30 min — environment check
    └── train.sh           1 GPU, 24 h  — a training run
```

## One-time setup

**1. SSH access.** Register an Ed25519 public key at the ABCI-Q User Portal.
Copy `ssh_config.example` into `~/.ssh/config`, replace `<ABCIQ_USERNAME>` with
your own username, adjust the `IdentityFile` path if needed. Test:

```bash
ssh abciq
```

**2. Clone the repo on the cluster.**

```bash
ssh abciq
git clone <repo-url> ~/gqe-for-qsci
```

**3. Set your personal account info** (never committed — see top of this file):

```bash
cd ~/gqe-for-qsci
cp hpc/env.local.sh.example hpc/env.local.sh
# edit hpc/env.local.sh: ABCIQ_USER, ABCIQ_GROUP
source hpc/env.local.sh
```

**4. Build the container** (login node — compute nodes have no internet):

```bash
cd hpc && bash build_image.sh
```

Writes `~/images/cudaq_qsci.sif`. Takes a while: multi-GB base image plus a
PyCI source build. The script probes `--fakeroot` first and explains what to do
if it is not permitted for your account.

## Running

Always `source hpc/env.local.sh` first in a fresh shell (sets `$ABCIQ_GROUP`).

**Smoke test first** — proves GPU, CUDA-Q and the project stack work:

```bash
cd ~/gqe-for-qsci
qsub -W group_list=$ABCIQ_GROUP hpc/jobs/smoke_gpu.sh
qstat                       # watch it
cat gqe_smoke.o<jobid>      # read the output
```

All 5 checks should report PASS.

**Then a training run:**

```bash
qsub -W group_list=$ABCIQ_GROUP -v EXPERIMENT=n2_bond_scan_dag_gnn hpc/jobs/train.sh
qsub -W group_list=$ABCIQ_GROUP -v EXPERIMENT=n2_l10_gpt2_paper,EXTRA="trainer.max_iters=50" hpc/jobs/train.sh
```

Useful PBS commands:

| | |
|---|---|
| `qstat` | your jobs |
| `qstat -f <jobid>` | full detail on one job |
| `qdel <jobid>` | cancel |
| `qstat -q` | queues |

## Resource types

From the ABCI-Q docs (`-l rt_XX=1`):

| Type | Cores | GPUs | Use |
|---|---|---|---|
| `rt_QG` | 32 | 1 | **default here** — one training run |
| `rt_QF` | 192 | 4 | full node; required for multi-node / MPI multi-QPU |
| `rt_QC` | 32 | 0 | CPU-only (e.g. pool inspection, `smoke_features.py`) |

The models are small and the wall-clock is dominated by CUDA-Q simulation plus
QSCI diagonalisation, so **1 GPU (`rt_QG`) is the right default**. `rt_QF` only
pays off with `sampler.mpi=true` across 4 QPUs.

## Weights & Biases

Compute nodes have no outbound internet, so `train.sh` sets
`WANDB_MODE=offline`. After a job finishes, sync from the **login node**:

```bash
singularity exec ~/images/cudaq_qsci.sif \
    wandb sync ~/gqe-for-qsci/outputs/gqe-for-qsci/<exp_tag>/wandb/offline-run-*
```

You will need `wandb login` (once, on the login node) before the first sync.

## Notes / gotchas

- **The container is a SANDBOX directory, not a `.sif`.** On this system a `.sif`
  larger than ~4 GB fails to mount with
  `kernel reported a bad superblock for squashfs image partition`. The image is
  not corrupt — the superblock is valid, compression is plain gzip, and
  `unsquashfs -l` lists all ~107k files. Bisection (busybox 2 MB ✓, tiny
  fakeroot build 2.2 MB ✓, cuda-quantum base 3.0 GB ✓, full 7.0 GB image ✗)
  rules out compression, Lustre, fakeroot and corruption, leaving size. Our
  stack is ~7 GB (3 GB base + ~4 GB torch/CUDA) and will not fit under 4 GB, so
  `build_image.sh` builds a sandbox directory instead. Uses ~14 GB and many
  inodes; start-up cost is irrelevant next to a multi-hour run. `SANDBOX=0`
  builds a `.sif` if the stack ever shrinks.
- **Build on the login node, not in a job.** The `docker pull` and `pip install`
  steps need internet.
- **`.sif` location.** Default `~/images/` (home has a 500 GB quota). To share
  one image with your group use `/groups/$ABCIQ_GROUP/` and set `SIF=...`.
- **PySCF cache.** `train.py` writes `.cache/pyscf/` next to the repo; it is
  bind-mounted, so CASCI/CCSD references are computed once and reused.
- **`qstat -u` does not work** on this PBS build (that is a Slurm flag); plain
  `qstat` lists your jobs.
- **Local scratch** `$PBS_LOCALDIR` (3.5 TB NVMe) is available per job if a run
  ever becomes I/O bound on Lustre. Not needed at current scale.
