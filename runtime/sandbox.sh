#!/usr/bin/env bash
# runtime/sandbox.sh — Phase D OS sandbox for the untrusted-kernel scoring child.
#
# PURPOSE: RCE / filesystem / network CONTAINMENT. If a submitted kernel reaches arbitrary code execution
# through some not-yet-closed in-process escape, this jail stops it from touching the harness, the host
# filesystem (champions/, results, other submissions), or the network. It is NOT GPU isolation: CUDA
# requires writable /dev/nvidia*, so a kernel can still issue raw ioctls to the device — GPU-side
# confidentiality is delivered by the no-secret-in-child design (the secret schedule, seed, and clock
# never enter this process), not by this sandbox.
#
# INVOKED by cco/isolate.py via $CCO_SANDBOX as:   runtime/sandbox.sh <scratch_dir> <argv...>
# It runs <argv...> under bubblewrap: read-only root, the per-run <scratch_dir> the ONLY writable host
# path, a private /tmp + /dev/shm, NO network, dropped capabilities, die-with-parent. LD_PRELOAD and the
# rest of the environment pass through (the Tier-2 vendor trap still loads inside the jail).
#
# *** MUST BE VALIDATED + TUNED ON THE PRODUCTION HOST ***  The exact /dev/nvidia* node set and any extra
# library paths CUDA/Triton need are host-specific. Before enabling (export CCO_SANDBOX=.../sandbox.sh),
# confirm a CLEAN champion kernel runs to completion under this jail on that host. The dev box (WSL2) has
# no bubblewrap, so this is never exercised there — isolate.py runs the child unsandboxed when CCO_SANDBOX
# is unset, which is sound (the load-bearing anti-cheat does not depend on the jail).
set -euo pipefail

SCRATCH="$1"; shift
command -v bwrap >/dev/null 2>&1 || { echo "sandbox.sh: bubblewrap (bwrap) not found on host" >&2; exit 127; }

# GPU device nodes CUDA needs (only those that exist on this host).
DEV_BINDS=()
for d in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset \
         /dev/nvidia0 /dev/nvidia1 /dev/nvidia2 /dev/nvidia3 /dev/nvidia4 /dev/nvidia5 /dev/nvidia6 /dev/nvidia7; do
  [[ -e "$d" ]] && DEV_BINDS+=(--dev-bind "$d" "$d")
done

exec bwrap \
  --ro-bind / / \
  --dev /dev \
  "${DEV_BINDS[@]}" \
  --tmpfs /tmp \
  --tmpfs /dev/shm \
  --proc /proc \
  --bind "$SCRATCH" "$SCRATCH" \
  --chdir "$SCRATCH" \
  --unshare-all \
  --new-session \
  --die-with-parent \
  --cap-drop ALL \
  -- "$@"
