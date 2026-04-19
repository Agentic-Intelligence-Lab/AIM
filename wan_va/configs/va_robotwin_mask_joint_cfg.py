# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os
from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg

# ── Full joint-diffusion mask training ──────────────────────────────────────
va_robotwin_mask_joint_cfg = EasyDict(__name__='Config: VA robotwin mask joint train')
va_robotwin_mask_joint_cfg.update(va_robotwin_cfg)

_ROBOTWIN_LEROBOT_ROOT = os.environ.get(
    "ROBOTWIN_LEROBOT_PATH",
    "/szeluresearch/fly/robotwin_lerobot",
)

va_robotwin_mask_joint_cfg.dataset_path = _ROBOTWIN_LEROBOT_ROOT
va_robotwin_mask_joint_cfg.empty_emb_path = os.path.join(
    _ROBOTWIN_LEROBOT_ROOT,
    "empty_emb.pt",
)
va_robotwin_mask_joint_cfg.enable_wandb  = False
va_robotwin_mask_joint_cfg.load_worker   = 8
va_robotwin_mask_joint_cfg.save_interval = 100
va_robotwin_mask_joint_cfg.gc_interval   = 50
va_robotwin_mask_joint_cfg.cfg_prob      = 0.1

va_robotwin_mask_joint_cfg.mask_cam_keys = [
    'observation.masks.cam_high',
    'observation.masks.cam_left_wrist',
    'observation.masks.cam_right_wrist',
]

va_robotwin_mask_joint_cfg.learning_rate = 1e-4
va_robotwin_mask_joint_cfg.beta1         = 0.9
va_robotwin_mask_joint_cfg.beta2         = 0.95
va_robotwin_mask_joint_cfg.weight_decay  = 0.1
va_robotwin_mask_joint_cfg.warmup_steps  = 500
va_robotwin_mask_joint_cfg.max_episode_frames = 500
va_robotwin_mask_joint_cfg.batch_size    = 8    # B2 joint seq; BS=8 safe for 95GB GPU
va_robotwin_mask_joint_cfg.gradient_accumulation_steps = 2  # effective BS=16/GPU, 256 global
va_robotwin_mask_joint_cfg.num_steps     = 100000
va_robotwin_mask_joint_cfg.max_episodes  = None   # None = use all episodes
va_robotwin_mask_joint_cfg.save_interval = 100


# ── Overfitting variant (B2 validation, 2 episodes) ─────────────────────────
va_robotwin_mask_joint_overfit_cfg = EasyDict(__name__='Config: VA robotwin mask joint overfit')
va_robotwin_mask_joint_overfit_cfg.update(va_robotwin_mask_joint_cfg)

va_robotwin_mask_joint_overfit_cfg.dataset_path = os.path.join(
    _ROBOTWIN_LEROBOT_ROOT,
    "adjust_bottle-aloha-agilex_randomized_500-1000",
)
va_robotwin_mask_joint_overfit_cfg.max_episodes   = 2    # 2 real episodes from this single task
va_robotwin_mask_joint_overfit_cfg.dataset_repeat = 128  # 2 × 128 = 256 items → 16 per GPU (16 GPUs)
va_robotwin_mask_joint_overfit_cfg.batch_size     = 16   # B2 uses ~2x mem; start conservative
va_robotwin_mask_joint_overfit_cfg.num_steps      = 2000
va_robotwin_mask_joint_overfit_cfg.save_interval  = 100
va_robotwin_mask_joint_overfit_cfg.warmup_steps   = 50
va_robotwin_mask_joint_overfit_cfg.load_worker    = 2
