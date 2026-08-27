# Stage the container sandbox onto node-local scratch. Sourced by the job
# scripts; expects $SIF to be set and reassigns it to the local copy.
#
# WHY
# ---
# The container is a sandbox directory (a >4 GB .sif will not mount here — see
# build_image.sh). But `--nv` injects GPU binaries into the container, which
# needs a writable overlay, and Lustre cannot be an overlay lower directory:
#     WARNING: ... /home/... is located on a LUSTRE filesystem incompatible as
#              overlay lower directory
#     FATAL: nvidia-container-cli ... /usr/bin/nvidia-smi: read-only file system
#
# $PBS_LOCALDIR is node-local NVMe (XFS, 3.5 TB), which does support overlays.
# Copying costs a few minutes once per job — negligible against a training run,
# and it also takes the container's I/O off the shared filesystem.
#
# Set STAGE_LOCAL=0 to skip (e.g. debugging on the login node).

if [ "${STAGE_LOCAL:-1}" = "1" ] && [ -n "${PBS_LOCALDIR:-}" ] && [ -d "$SIF" ]; then
    _local_sif="$PBS_LOCALDIR/$(basename "$SIF")"
    echo "=== staging container to node-local scratch ==="
    echo "  from : $SIF"
    echo "  to   : $_local_sif"
    df -h "$PBS_LOCALDIR" | tail -1
    # -a preserves symlinks/permissions/times, which a container rootfs needs.
    time cp -a "$SIF" "$_local_sif"
    SIF="$_local_sif"
    echo "  staged OK -> $SIF"
    echo
elif [ -n "${PBS_LOCALDIR:-}" ] && [ ! -d "$SIF" ]; then
    echo "NOTE: \$SIF is not a directory (.sif file?) - skipping local staging."
    echo
fi
