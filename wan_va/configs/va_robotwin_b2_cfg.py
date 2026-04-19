# Server config for the B2 finetuned model (mask head checkpoint)
import os
from pathlib import Path
from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg

va_robotwin_b2_cfg = EasyDict(__name__='Config: VA robotwin B2 finetuned')
va_robotwin_b2_cfg.update(va_robotwin_cfg)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Override transformer path to load our B2 checkpoint
# VAE / tokenizer / text_encoder still come from the original pretrained model
va_robotwin_b2_cfg.transformer_path = os.environ.get(
    "WAN_VA_B2_TRANSFORMER_PATH",
    str(
        _REPO_ROOT
        / "full_train_b2_output"
        / "checkpoints"
        / "checkpoint_step_1200"
        / "transformer"
    ),
)

va_robotwin_b2_cfg.infer_mode = 'server'
