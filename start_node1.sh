#!/bin/bash
# Run on node 1 (remote). Log goes to shared storage.
cd /szeluresearch/fly/unified-model
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m torch.distributed.run \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr=10.11.12.12 \
  --master_port=29713 \
  --tee 3 \
  -m wan_va.train_mask_joint \
  --config-name robotwin_mask_joint_overfit \
  --save-root /szeluresearch/fly/unified-model/joint_overfit_v4_output \
  >> /szeluresearch/fly/unified-model/joint_overfit_v4b_node1.log 2>&1
