#!/bin/bash
# Build the GQE-QSCI Singularity image on the ABCI-Q login node (qes).
#
#   cd ~/gqe-for-qsci/hpc && bash build_image.sh
#
# Takes a while (pulls a multi-GB base image, compiles PyCI) and writes
# ~/images/cudaq_qsci.sif. Run it on the LOGIN node, not in a job: compute
# nodes typically have no outbound internet for the docker pull / pip installs.
set -euo pipefail

DEF="$(cd "$(dirname "$0")" && pwd)/cudaq_qsci.def"
IMAGE_DIR="${IMAGE_DIR:-$HOME/images}"
SIF="${SIF:-$IMAGE_DIR/cudaq_qsci.sif}"

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
time singularity build --fakeroot "$SIF" "$DEF"

echo
echo "== done =="
ls -lh "$SIF"
echo
echo "Smoke-test it with:"
echo "  source hpc/env.local.sh   # sets ABCIQ_GROUP (see env.local.sh.example)"
echo "  qsub -W group_list=\$ABCIQ_GROUP hpc/jobs/smoke_gpu.sh"
