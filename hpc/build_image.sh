#!/bin/bash
# Build the GQE-QSCI Singularity container on the ABCI-Q login node (qes).
#
#   cd ~/gqe-for-qsci/hpc && bash build_image.sh
#
# Takes a while (pulls a multi-GB base image, compiles PyCI) and writes a
# SANDBOX directory at ~/images/cudaq_sandbox. Run it on the LOGIN node, not in
# a job: compute nodes have no outbound internet for the docker pull / pip
# installs.
#
# WHY A SANDBOX AND NOT A .sif
# ----------------------------
# On this system a .sif larger than ~4 GB fails to mount:
#     FATAL: ... kernel reported a bad superblock for squashfs image partition
# The image is NOT corrupt - `unsquashfs -l` lists all ~107k files fine, the
# superblock is valid, compression is plain gzip. Verified by bisection:
#     busybox 2 MB (pull)                -> mounts OK
#     tiny 2.2 MB (--fakeroot + %post)   -> mounts OK
#     cuda-quantum base 3.0 GB (pull)    -> mounts OK
#     this image 7.0 GB (--fakeroot)     -> FAILS
# So it is neither compression, Lustre, fakeroot nor corruption - it is size
# (the classic 4 GiB boundary). Our stack is ~7 GB (base 3 GB + torch/CUDA
# ~4 GB) and cannot realistically be squeezed under 4 GB, so we skip squashfs
# entirely and use a sandbox directory. Costs ~14 GB of disk and many inodes,
# but container start-up is irrelevant next to a multi-hour training run.
#
# Set SANDBOX=0 to build a .sif instead (fine if the stack ever shrinks).
set -euo pipefail

DEF="$(cd "$(dirname "$0")" && pwd)/cudaq_qsci.def"
IMAGE_DIR="${IMAGE_DIR:-$HOME/images}"
SANDBOX="${SANDBOX:-1}"
if [ "$SANDBOX" = "1" ]; then
    SIF="${SIF:-$IMAGE_DIR/cudaq_sandbox}"
    BUILD_ARGS="--sandbox"
else
    SIF="${SIF:-$IMAGE_DIR/cudaq_qsci.sif}"
    BUILD_ARGS=""
fi

mkdir -p "$IMAGE_DIR"

# Keep Singularity's scratch off /tmp: the base image layers are several GB and
# a small /tmp is a common cause of "no space left on device" mid-build.
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-$IMAGE_DIR/.tmp}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-$IMAGE_DIR/.cache}"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

echo "def file : $DEF"
echo "output   : $SIF"
echo "tmpdir   : $SINGULARITY_TMPDIR"
echo

if [ -e "$SIF" ]; then
    echo "ERROR: $SIF already exists. Remove it or set SIF=... to build elsewhere." >&2
    exit 1
fi

# Unprivileged users need --fakeroot to run %post. Some sites disable it; check
# first so the failure is a clear message rather than a confusing build error.
echo "== checking --fakeroot =="
if singularity build --fakeroot "$SINGULARITY_TMPDIR/_fakeroot_probe.sif" \
        docker://busybox:latest >/dev/null 2>&1; then
    echo "fakeroot: OK"
    rm -f "$SINGULARITY_TMPDIR/_fakeroot_probe.sif"
else
    rm -f "$SINGULARITY_TMPDIR/_fakeroot_probe.sif"
    cat >&2 <<'EOF'

ERROR: `singularity build --fakeroot` does not work for this account.

Without fakeroot the %post section cannot run, so the extra Python packages
(PyCI, torch, ...) cannot be baked in. Options:

  1. Ask the ABCI-Q helpdesk to enable fakeroot for your account.
  2. Pull the bare base image and layer the deps into a venv in $HOME:
         singularity pull $SIF docker://ghcr.io/nvidia/cudaqx:0.4.0
     then create a venv on a login node and point PYTHONPATH at it in the
     job scripts. (More moving parts; only do this if 1 is refused.)

EOF
    exit 1
fi

echo
echo "== building (this takes a while) =="
time singularity build --fakeroot $BUILD_ARGS "$SIF" "$DEF"

echo
echo "== done =="
ls -lh "$SIF"
echo
echo "Smoke-test it with:"
echo "  source hpc/env.local.sh   # sets ABCIQ_GROUP (see env.local.sh.example)"
echo "  qsub -W group_list=\$ABCIQ_GROUP hpc/jobs/smoke_gpu.sh"
