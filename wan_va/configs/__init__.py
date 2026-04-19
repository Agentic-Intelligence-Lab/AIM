# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_demo_train_cfg import va_demo_train_cfg
from .va_demo_cfg import va_demo_cfg
from .va_demo_i2va import va_demo_i2va_cfg
from .va_robotwin_mask_train_cfg import va_robotwin_mask_train_cfg
from .va_robotwin_mask_joint_cfg import (
    va_robotwin_mask_joint_cfg,
    va_robotwin_mask_joint_overfit_cfg,
)
from .va_robotwin_b2_cfg import va_robotwin_b2_cfg

VA_CONFIGS = {
    'robotwin': va_robotwin_cfg,
    'franka': va_franka_cfg,
    'robotwin_i2av': va_robotwin_i2va_cfg,
    'franka_i2av': va_franka_i2va_cfg,
    'robotwin_train': va_robotwin_train_cfg,
    'demo': va_demo_cfg,
    'demo_train': va_demo_train_cfg,
    'demo_i2av': va_demo_i2va_cfg,
    'robotwin_mask_train': va_robotwin_mask_train_cfg,
    'robotwin_mask_joint': va_robotwin_mask_joint_cfg,
    'robotwin_mask_joint_overfit': va_robotwin_mask_joint_overfit_cfg,
    'robotwin_b2': va_robotwin_b2_cfg,
}