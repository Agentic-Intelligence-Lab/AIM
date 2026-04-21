# AIM

`AIM` is a research codebase for training, serving, and evaluating a LingBot-VA style world-action model built on top of a Wan 2.2 video-diffusion backbone. This repository is organized as a source-code release for the code paths that are actually present here: model training, mask/value-branch finetuning, online inference serving, offline debugging, and RoboTwin evaluation.

**Quick links:** [Technical report](https://arxiv.org/abs/2604.11135) 

> This repository is intentionally code-only. Large checkpoints, generated artifacts, latent datasets, and machine-specific outputs are kept outside Git history and should be distributed through external storage such as Hugging Face.

## Overview

- Related technical report: `AIM: Intent-Aware Unified World Action Modeling with Spatial Value Maps` (`arXiv:2604.11135`)
- Main training entrypoints: `wan_va/train.py`, `wan_va/train_mask_joint.py`
- Online serving entrypoint: `wan_va/wan_va_server.py`
- Evaluation code: `evaluation/robotwin/`
- Lightweight dependency file: `requirements/lerobot.txt`

## What Is In This Repository

- Training code for the base RGB-plus-action post-training pipeline
- Joint video-plus-mask finetuning code
- Online inference server and deployment helpers
- Offline inference and debugging scripts
- RoboTwin evaluation clients and launch scripts
- Small example images for sanity checks and demos

## What Is Not In This Repository

- Large training checkpoints and output directories
- Generated videos, logs, and experiment outputs
- Full auxiliary pipelines described outside the current code snapshot

## External Assets

Public checkpoint mirror:

- `https://huggingface.co/AUTMOEN999/AIM`

## Related Technical Report

This repository is related to the following technical report:

- `AIM: Intent-Aware Unified World Action Modeling with Spatial Value Maps`
- Authors: Liaoyuan Fan, Zetian Xu, Chen Cao, Wenyao Zhang, Mingqi Yuan, and Jiayu Chen
- arXiv: `2604.11135`

Important scope note:

- The current repository reflects the code snapshot available here
- It should not be interpreted as a full artifact release of every auxiliary pipeline described in the report
- In particular, large-scale data generation, external checkpoints, and some training-stage dependencies remain external to the repository

## Repository Structure

```text
.
├── evaluation/                  # RoboTwin evaluation client code
├── example/                     # Small sample observation images
├── requirements/                # Dependency files
├── script/                      # Legacy launch helpers
├── scripts/                     # Inference, resume, and evaluation scripts
├── wan_va/
│   ├── configs/                 # Experiment and deployment configs
│   ├── dataset/                 # LeRobot latent dataset loader
│   ├── distributed/             # FSDP and distributed helpers
│   ├── modules/                 # Transformer / VAE loading utilities
│   ├── utils/                   # Logging, scheduling, remote infer helpers
│   ├── train.py                 # Base post-training entrypoint
│   ├── train_mask_joint.py      # Joint video+mask training entrypoint
│   └── wan_va_server.py         # Online inference server
├── launch_16gpu_joint.sh        # 2-node joint finetuning launcher
└── README.md
```

## Installation

The repository does not yet ship with one fully pinned environment export, but it includes a LeRobot dependency file at `requirements/lerobot.txt`. A practical starting point is:

```bash
conda create -n aim python=3.11 -y
conda activate aim

# Install the correct PyTorch build for your CUDA version first.
# Example only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install \
  diffusers transformers safetensors easydict einops numpy scipy pillow \
  opencv-python tqdm packaging pyyaml wandb websockets msgpack \
  matplotlib imageio toppra transforms3d typing_extensions pyav

pip install --no-deps lerobot==0.3.2
pip install -r requirements/lerobot.txt
```

Additional evaluation dependencies:

- `sapien` for RoboTwin simulation and rendering

Some experiments require additional local runtime assets and environment setup beyond a fresh clone. The public checkpoint mirror linked above is provided separately from this GitHub repository.

## Configuration

Several path settings can be overridden with environment variables:

| Variable | Purpose |
| --- | --- |
| `WAN22_PRETRAINED_MODEL_PATH` | Base Wan 2.2 model path |
| `ROBOTWIN_LEROBOT_PATH` | Root folder of the LeRobot latent dataset |
| `WAN_VA_B2_TRANSFORMER_PATH` | Finetuned B2 transformer checkpoint path |
| `ROBOWIN_ROOT` | Path to the local RoboTwin repository |
| `WORK_DIR` | Override repo root in launch scripts |
| `ROBOTWIN_POLICY_CONFIG` | Evaluation policy config path for `scripts/run_eval_10tasks.sh` |

Before running on a new machine, review the scripts that still carry experiment-specific defaults:

- `scripts/infer_demo.py`
- `scripts/infer_mask_v2.py`
- `scripts/infer_joint_overfit.py`
- `evaluation/robotwin/eval_polict_client_openpi.py`
- `launch_16gpu_joint.sh`
- `scripts/resume_b2_16gpu.sh`

## Quick Start

### 1. Base Post-Training

```bash
export WAN22_PRETRAINED_MODEL_PATH=/path/to/base/model
export ROBOTWIN_LEROBOT_PATH=/path/to/robotwin_lerobot

NGPU=8 CONFIG_NAME=robotwin_train \
  bash script/run_va_posttrain.sh --save-root ./train_out
```

### 2. Joint Video+Mask Finetuning

```bash
export WAN22_PRETRAINED_MODEL_PATH=/path/to/base/model
export ROBOTWIN_LEROBOT_PATH=/path/to/robotwin_lerobot

bash launch_16gpu_joint.sh
```

### 3. Resume Joint Finetuning

```bash
export RESUME_FROM=/path/to/checkpoint_step_xxxx
bash scripts/resume_b2_16gpu.sh
```

### 4. Start the Online Inference Server

```bash
export WAN22_PRETRAINED_MODEL_PATH=/path/to/base/model
export WAN_VA_B2_TRANSFORMER_PATH=/path/to/checkpoint/transformer

NGPU=1 CONFIG_NAME=robotwin_b2 \
  bash script/run_launch_va_server_sync.sh --port 29056 --save_root ./visualization
```

### 5. Offline Demo and Debug Inference

Use one of the task scripts under `scripts/`:

- `scripts/infer_demo.py`
- `scripts/infer_mask_v2.py`
- `scripts/infer_joint_overfit.py`

These scripts are intentionally experiment-oriented. Review their task assumptions and path defaults before running them outside the original environment.

### 6. RoboTwin Evaluation

Typical evaluation flow:

1. Start one or more model servers
2. Point `ROBOWIN_ROOT` to your local RoboTwin checkout
3. Set `ROBOTWIN_POLICY_CONFIG` if your deployment config is not at the default path
4. Run:

```bash
bash scripts/run_eval_10tasks.sh
```

## Main Entry Points

The main runnable entry points in the current codebase are:

- `python -m wan_va.train --config-name robotwin_train`
- `python -m wan_va.train_mask_joint --config-name robotwin_mask_joint`
- `python -m wan_va.wan_va_server --config-name robotwin`
- `python -m wan_va.wan_va_server --config-name robotwin_b2`
- `python -u -m evaluation.robotwin.eval_polict_client_openpi ...`

## Reproducibility Notes

- The repository is designed around latent datasets and external checkpoints; it will not run end-to-end from a fresh clone without those assets
- Evaluation requires more dependencies and local setup than training-only inspection
- Some launch scripts still contain machine-specific defaults such as IPs, hostnames, ports, and network interface names
- The current repository reflects the code snapshot available here and should be understood with that scope in mind

## Citation

If you use this repository in academic work, please cite the related technical report:

```bibtex
@article{aim2026,
  title   = {AIM: Intent-Aware Unified World Action Modeling with Spatial Value Maps},
  author  = {Liaoyuan Fan and Zetian Xu and Chen Cao and Wenyao Zhang and Mingqi Yuan and Jiayu Chen},
  journal = {arXiv preprint arXiv:2604.11135},
  year    = {2026}
}
```
