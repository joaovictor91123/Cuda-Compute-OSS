#!/usr/bin/env bash
# CCO canonical-rerun entrypoint — runtime/start.sh.
#
# Runs INSIDE the pinned cco-runtime image on the trusted GPU box. Job: reproduce the score for ONE
# kernel (the mounted /cco/kernel.py) under controlled conditions and emit the bound score blob
# (cco/blob.py). The image + this script + the harness are byte-locked; only kernel.py varies.
#
# The champion-vs-challenger WIN decision is NOT made here: the harness scores one kernel per run.
# The pipeline runs this once for the challenger and once for the champion, then compares the two
# samples with cco/significance.py.
#
# Inputs (env):
#   PR_HEAD_SHA  - PR HEAD commit SHA; seeds input generation (unpredictable to the miner). REQUIRED.
#   KERNEL_TYPE  - optional; otherwise read from the mounted kernel.py's KERNEL_TYPE.
#   GPU_INDEX    - optional; overrides CCO_GPU_INDEX from gpu.config.
#   OUT          - optional; blob output path (default: /out/score-blob.json).
#
# Egress: the canonical rerun runs with the network DENIED (cco.config.json gate4.egress=deny_all),
# enforced by the orchestration layer (docker --network=none / firewalled pod), NOT here.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
# shellcheck source=/dev/null
source "${HERE}/gpu.config"

GPU_INDEX="${GPU_INDEX:-${CCO_GPU_INDEX:-0}}"
OUT="${OUT:-/out/score-blob.json}"
LOG="$(mktemp)"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1

: "${PR_HEAD_SHA:?set PR_HEAD_SHA to the PR HEAD commit SHA}"

echo "== CCO canonical rerun =="
echo "pinned SKU : ${CCO_GPU_NAME} (cc ${CCO_GPU_CC})"
echo "gpu index  : ${GPU_INDEX}"
echo "PR_HEAD_SHA: ${PR_HEAD_SHA}"

# --- 1. Assert the pinned SKU is actually present (a swap is a competition reset) ---
ACTUAL_NAME="$(nvidia-smi --id="${GPU_INDEX}" --query-gpu=name --format=csv,noheader)"
if [[ "${ACTUAL_NAME}" != "${CCO_GPU_NAME}" ]]; then
  echo "FATAL: GPU SKU mismatch: got '${ACTUAL_NAME}', pinned '${CCO_GPU_NAME}'" >&2
  exit 3
fi

# --- 2. Assert exclusive GPU (no other compute processes) ---
NPROC="$(nvidia-smi --id="${GPU_INDEX}" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)"
if [[ "${NPROC}" -ne 0 ]]; then
  echo "FATAL: GPU ${GPU_INDEX} is not exclusive (${NPROC} compute process(es) running)" >&2
  exit 4
fi

# --- 3. Lock clocks (best-effort; needs a privileged container). Stable clocks => stable timing ---
if [[ "${CCO_LOCK_CLOCKS:-1}" == "1" ]]; then
  nvidia-smi --id="${GPU_INDEX}" -pm 1 >/dev/null 2>&1 || echo "warn: could not enable persistence mode" >&2
  [[ -n "${CCO_LOCK_GRAPHICS_CLOCK:-}" ]] && nvidia-smi --id="${GPU_INDEX}" -lgc "${CCO_LOCK_GRAPHICS_CLOCK}" >/dev/null 2>&1 || true
  [[ -n "${CCO_LOCK_MEMORY_CLOCK:-}" ]]   && nvidia-smi --id="${GPU_INDEX}" -lmc "${CCO_LOCK_MEMORY_CLOCK}" >/dev/null 2>&1 || true
fi

# --- 4. Derive the per-PR input seed from the PR HEAD SHA ---
SEED="$(python3 "${REPO_ROOT}/cco/seed.py" "${PR_HEAD_SHA}")"
echo "input_seed : ${SEED} (derived from PR_HEAD_SHA)"

# --- 5. Run the locked harness; capture output and extract the bound score blob ---
mkdir -p "$(dirname "${OUT}")"
KARG=()
[[ -n "${KERNEL_TYPE:-}" ]] && KARG=(--kernel "${KERNEL_TYPE}")
( cd "${REPO_ROOT}" && python3 benchmark.py --blob --seed "${SEED}" "${KARG[@]}" ) | tee "${LOG}"

# The blob prints between these markers; extract just the JSON object.
awk '/^=== SCORE BLOB ===$/{f=1;next} /^=== END SCORE BLOB ===$/{f=0} f' "${LOG}" > "${OUT}"
if [[ ! -s "${OUT}" ]]; then
  echo "FATAL: no score blob produced (harness failed?). See log above." >&2
  exit 5
fi
echo "== blob written: ${OUT} =="
