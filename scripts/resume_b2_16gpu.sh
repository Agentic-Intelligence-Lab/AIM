#!/usr/bin/bash
# Resume B2 full_train (train_mask_joint) on 2×8 GPUs from latest checkpoint.
# Node 0: run this script on the machine whose net0 is MASTER_ADDR (e.g. 10.11.12.12).
# Node 1: started via SSH to REMOTE_HOST.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SAVE_ROOT="${SAVE_ROOT:-${WORK_DIR}/full_train_b2_output}"
CKPT="${RESUME_FROM:-${SAVE_ROOT}/checkpoints/checkpoint_step_1200}"
MASTER_ADDR="${MASTER_ADDR:-10.11.12.12}"
MASTER_PORT="${MASTER_PORT:-29803}"
REMOTE_HOST="${REMOTE_HOST:-root@139.196.6.208}"
REMOTE_PORT="${REMOTE_PORT:-1023}"
LOG_LOCAL="${WORK_DIR}/full_train_b2_node0_resume.log"
REMOTE_LOG="/tmp/full_train_b2_node1_resume.log"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=net0
export GLOO_SOCKET_IFNAME=net0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

echo "=== Resume B2 16-GPU ==="
echo "SAVE_ROOT=${SAVE_ROOT}"
echo "RESUME_FROM=${CKPT}"
echo "MASTER=${MASTER_ADDR}:${MASTER_PORT}"

echo "=== Kill stale train_mask_joint on remote ==="
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
  "${REMOTE_HOST}" -p "${REMOTE_PORT}" \
  "pkill -f 'train_mask_joint' 2>/dev/null || true; sleep 2; true" || true

echo "=== Launch node 1 (remote) ==="
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 \
  "${REMOTE_HOST}" -p "${REMOTE_PORT}" bash -s <<EOF
set -e
cd "${WORK_DIR}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=net0
export GLOO_SOCKET_IFNAME=net0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
nohup python -m torch.distributed.run \\
  --nproc_per_node=8 --nnodes=2 --node_rank=1 \\
  --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \\
  --tee 3 \\
  -m wan_va.train_mask_joint \\
  --config-name robotwin_mask_joint \\
  --save-root "${SAVE_ROOT}" \\
  --resume-from "${CKPT}" \\
  > "${REMOTE_LOG}" 2>&1 &
echo "Remote started PID=\$! log=${REMOTE_LOG}"
EOF

sleep 5

echo "=== Launch node 0 (local, background) ==="
cd "${WORK_DIR}"
nohup python -m torch.distributed.run \
  --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  --tee 3 \
  -m wan_va.train_mask_joint \
  --config-name robotwin_mask_joint \
  --save-root "${SAVE_ROOT}" \
  --resume-from "${CKPT}" \
  > "${LOG_LOCAL}" 2>&1 &
echo "Local node0 PID=$! log=${LOG_LOCAL}"
echo "Done. tail -f ${LOG_LOCAL}"
