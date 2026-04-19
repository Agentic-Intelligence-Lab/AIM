#!/bin/bash
cd /szeluresearch/fly/lingbot-va-uni

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL: force use of the inter-node interface net0 (10.11.12.12)
export NCCL_SOCKET_IFNAME=net0
export GLOO_SOCKET_IFNAME=net0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

python -m torch.distributed.run \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=10.11.12.12 \
  --master_port=29803 \
  --tee 3 \
  -m wan_va.train_mask_joint \
  --config-name robotwin_mask_joint \
  --save-root /szeluresearch/fly/lingbot-va-uni/full_train_b2_output \
  > /szeluresearch/fly/lingbot-va-uni/full_train_b2_node0.log 2>&1
