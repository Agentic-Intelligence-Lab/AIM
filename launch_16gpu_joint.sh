#!/usr/bin/bash
# 16-GPU (2×8) joint-diffusion mask training.
# SSH-launches node 1 (remote) then starts node 0 (local) in foreground.
#
# Usage:
#   bash launch_16gpu_joint.sh                          # 2-node 16 GPU
#   NNODES=1 bash launch_16gpu_joint.sh                 # single-node 8 GPU
#   RESUME_FROM=/path/to/ckpt bash launch_16gpu_joint.sh

set -euo pipefail

WORK_DIR="${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SAVE_ROOT="${SAVE_ROOT:-${WORK_DIR}/joint_mask_train_output}"
NNODES="${NNODES:-2}"
MASTER_ADDR="${MASTER_ADDR:-10.11.12.12}"
MASTER_PORT="${MASTER_PORT:-29701}"
REMOTE_HOST="${REMOTE_HOST:-root@139.196.6.208}"
REMOTE_PORT="${REMOTE_PORT:-1023}"
LOG_FILE="${WORK_DIR}/joint_train.log"
REMOTE_LOG="/tmp/node1_joint_train.log"
RESUME_FROM="${RESUME_FROM:-}"

mkdir -p "${SAVE_ROOT}/checkpoints"

RESUME_ARG=""
if [[ -n "${RESUME_FROM}" ]]; then
  RESUME_ARG="--resume-from ${RESUME_FROM}"
fi

echo "=== 16-GPU joint-diffusion mask training ==="
echo "SAVE_ROOT=${SAVE_ROOT}"
echo "NNODES=${NNODES}  MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"
if [[ -n "${RESUME_FROM}" ]]; then echo "Resuming from: ${RESUME_FROM}"; fi

if [[ "${NNODES}" == "2" ]]; then
  echo "=== Kill stale processes on remote node ==="
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
    "${REMOTE_HOST}" -p "${REMOTE_PORT}" \
    "pkill -f 'train_mask_joint' 2>/dev/null || true; sleep 1; true" \
    || true

  echo "=== Launch node 1 (remote) ==="
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 \
    "${REMOTE_HOST}" -p "${REMOTE_PORT}" bash -s <<EOF
set -e
cd "${WORK_DIR}"
export TOKENIZERS_PARALLELISM=false
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  python -m torch.distributed.run \\
    --nproc_per_node=8 --nnodes=2 --node_rank=1 \\
    --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \\
    --tee 3 \\
    -m wan_va.train_mask_joint \\
    --config-name robotwin_mask_joint \\
    --save-root "${SAVE_ROOT}" \\
    ${RESUME_ARG} \\
  > "${REMOTE_LOG}" 2>&1 &
echo "Remote PID \$!  log: ${REMOTE_LOG}"
EOF
  sleep 5
  LOCAL_ARGS=(--nnodes=2 --node_rank=0)
else
  LOCAL_ARGS=(--nnodes=1 --node_rank=0)
fi

echo "=== Launch node 0 (local) ==="
cd "${WORK_DIR}"
export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m torch.distributed.run \
  --nproc_per_node=8 "${LOCAL_ARGS[@]}" \
  --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  --tee 3 \
  -m wan_va.train_mask_joint \
  --config-name robotwin_mask_joint \
  --save-root "${SAVE_ROOT}" \
  ${RESUME_ARG} \
  2>&1 | tee "${LOG_FILE}"
