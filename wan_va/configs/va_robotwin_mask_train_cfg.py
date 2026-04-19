# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg
import os

va_robotwin_mask_train_cfg = EasyDict(__name__='Config: VA robotwin mask train')
va_robotwin_mask_train_cfg.update(va_robotwin_cfg)

_ROBOTWIN_LEROBOT_ROOT = os.environ.get(
    "ROBOTWIN_LEROBOT_PATH",
    "/szeluresearch/fly/robotwin_lerobot",
)

va_robotwin_mask_train_cfg.dataset_path = _ROBOTWIN_LEROBOT_ROOT
va_robotwin_mask_train_cfg.empty_emb_path = os.path.join(
    _ROBOTWIN_LEROBOT_ROOT,
    "empty_emb.pt",
)
va_robotwin_mask_train_cfg.enable_wandb = False
va_robotwin_mask_train_cfg.load_worker = 8
va_robotwin_mask_train_cfg.save_interval = 100
va_robotwin_mask_train_cfg.gc_interval = 50
va_robotwin_mask_train_cfg.cfg_prob = 0.1

va_robotwin_mask_train_cfg.mask_cam_keys = [
    'observation.masks.cam_high',
    'observation.masks.cam_left_wrist',
    'observation.masks.cam_right_wrist',
]

va_robotwin_mask_train_cfg.learning_rate = 1e-4
va_robotwin_mask_train_cfg.beta1 = 0.9
va_robotwin_mask_train_cfg.beta2 = 0.95
va_robotwin_mask_train_cfg.weight_decay = 0.1
va_robotwin_mask_train_cfg.warmup_steps = 500
va_robotwin_mask_train_cfg.max_episode_frames = 500
va_robotwin_mask_train_cfg.batch_size = 16
va_robotwin_mask_train_cfg.gradient_accumulation_steps = 1
va_robotwin_mask_train_cfg.num_steps = 50000
