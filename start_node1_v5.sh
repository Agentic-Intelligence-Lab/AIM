#!/bin/bash
cd /szeluresearch/fly/AIM
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m torch.distributed.run \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr=10.11.12.12 \
  --master_port=29801 \
  --tee 3 \
  -m wan_va.train_mask_joint \
  --config-name robotwin_mask_joint_overfit \
  --save-root /szeluresearch/fly/AIM/joint_overfit_v5_output \
  > /szeluresearch/fly/AIM/joint_overfit_v5_node1.log 2>&1
