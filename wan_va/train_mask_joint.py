# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Joint-diffusion mask training: video + mask tokens processed together through
the frozen backbone, enabling mask tokens to attend to video tokens directly.

Architecture (v4):
    [vid_noisy | vid_cond | mask_noisy | act_noisy | act_cond]  →  DiT (SDPA, frozen)
                                                                        ↓ split
                                    vid_feat              mask_feat
                                       ↓                     ↓
                                   proj_out (frozen)   mask_proj_out (trainable)
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from safetensors.torch import save_file

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model, apply_ac
from distributed.util import (
    _configure_model,
    init_distributed,
    dist_mean,
    dist_max,
)
from einops import rearrange
from modules.utils import load_transformer
from utils import (
    init_logger,
    logger,
    get_mesh_id,
    sample_timestep_id,
    data_seq_to_patch,
    warmup_constant_lambda,
    FlowMatchScheduler,
)
from dataset import MultiLatentLeRobotDataset


def pad_collate_fn(batch):
    """Pad variable-length latent tensors to the max length in the batch."""
    max_lat_f = max(b['latents'].shape[1] for b in batch)
    max_act_f = max(b['actions'].shape[1] for b in batch)

    def pad_dim1(t, target, val=0.0):
        diff = target - t.shape[1]
        if diff == 0:
            return t
        return F.pad(t, [0] * (2 * (t.ndim - 2)) + [0, diff], value=val)

    out = {}
    for key in batch[0]:
        tensors = [b[key] for b in batch]
        if key in ('latents', 'mask_latents'):
            tensors = [pad_dim1(t, max_lat_f) for t in tensors]
        elif key in ('actions', 'actions_mask'):
            tensors = [pad_dim1(t, max_act_f) for t in tensors]
        out[key] = torch.stack(tensors)

    lat_masks = []
    for b in batch:
        real_f = b['latents'].shape[1]
        m = torch.zeros(max_lat_f, dtype=torch.bool)
        m[:real_f] = True
        lat_masks.append(m)
    out['valid_frames'] = torch.stack(lat_masks)
    return out


class JointMaskTrainer:
    def __init__(self, config):
        self.step = 0
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size

        logger.info("Loading transformer...")

        if hasattr(config, 'resume_from') and config.resume_from:
            transformer_path = os.path.join(config.resume_from, 'transformer')
            if config.rank == 0:
                logger.info(f"Resuming from: {transformer_path}")
        else:
            transformer_path = os.path.join(
                config.wan22_pretrained_model_name_or_path, 'transformer')

        # Load with the checkpoint's original attn_mode (flex).
        # The joint forward pass uses _use_sdpa_fallback=True to bypass
        # FlexAttn's T-shape block mask during backbone processing, so
        # Triton/FlexAttn is never actually invoked during training.
        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device='cpu',
        )

        logger.info("Applying activation checkpointing...")
        apply_ac(self.transformer)

        logger.info("Setting up FSDP...")
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_model,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )

        # B2: backbone and all pre-existing heads remain frozen.
        # Only mask_proj_out is trained — the mask stream features come from the frozen
        # backbone itself (mask tokens are embedded via the shared patch_embedding_mlp
        # and processed through all 40 transformer blocks alongside video tokens).
        self.transformer.requires_grad_(False)
        for p in self.transformer.mask_proj_out.parameters():
            p.requires_grad_(True)
        self.transformer.train()

        trainable = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.transformer.parameters())
        if config.rank == 0:
            logger.info(f"Trainable params: {trainable:,} / {total:,} "
                        f"({trainable / total * 100:.4f}%)")

        self.optimizer = torch.optim.AdamW(
            [p for p in self.transformer.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps),
        )

        logger.info("Setting up datasets...")
        train_dataset = MultiLatentLeRobotDataset(config=config)
        if config.rank == 0:
            logger.info(f"Dataset size: {len(train_dataset)}")

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=True,
            seed=42,
        ) if config.world_size > 1 else None

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None),
            num_workers=config.load_worker,
            sampler=train_sampler,
            collate_fn=pad_collate_fn,
            pin_memory=True,
        )

        self.train_scheduler_latent = FlowMatchScheduler(
            shift=config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(
            shift=config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        self.train_loader_iter = None

    # ── Batch management ─────────────────────────────────────────────────────

    def _get_next_batch(self):
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        try:
            return next(self.train_loader_iter)
        except StopIteration:
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(
                    self.train_loader.sampler.epoch + 1)
            self.train_loader_iter = iter(self.train_loader)
            return next(self.train_loader_iter)

    # ── Noise addition (identical to train_mask.py) ──────────────────────────

    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False,
                   action_mode=False, noisy_cond_prob=0.):
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(
            batch_size=F,
            num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents = train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets = train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1

        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,
            latent.shape[-2] // patch_h,
            latent.shape[-1] // patch_w,
            t=1 if action_mode else 0,
            f_w=1, f_shift=0, action=action_mode,
        ).to(self.device)
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                batch_size=F, min_timestep_bd=0.5, max_timestep_bd=1.0,
                num_train_timesteps=train_scheduler.num_train_timesteps)
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(
                device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets        *= action_mask.float()
            latent         *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        latent_dict = self._add_noise(
            latent=batch_dict['latents'],
            train_scheduler=self.train_scheduler_latent,
            action_mask=None, action_mode=False, noisy_cond_prob=0.5)

        action_dict = self._add_noise(
            latent=batch_dict['actions'],
            train_scheduler=self.train_scheduler_action,
            action_mask=batch_dict['actions_mask'],
            action_mode=True, noisy_cond_prob=0.0)

        latent_dict['text_emb'] = batch_dict['text_emb']
        action_dict['text_emb'] = batch_dict['text_emb']
        action_dict['actions_mask'] = batch_dict['actions_mask']

        # B2 mask dict: same noise schedule and timesteps as the video.
        # noisy_cond_prob=0 for mask (clean GT mask always used as conditioning,
        # matching inference where mask frame-0 is pinned to clean GT).
        video_timesteps    = latent_dict['timesteps']      # [B, F]
        video_cond_ts      = latent_dict['cond_timesteps'] # [B, F]
        mask_noise         = torch.zeros_like(batch_dict['mask_latents']).normal_()
        noisy_mask_latents = self.train_scheduler_latent.add_noise(
            batch_dict['mask_latents'], mask_noise, video_timesteps[0], t_dim=2)
        mask_targets = self.train_scheduler_latent.training_target(
            batch_dict['mask_latents'], mask_noise, video_timesteps[0])

        mask_dict = dict(
            noisy_latents  = noisy_mask_latents,
            latent         = batch_dict['mask_latents'],  # clean GT mask (conditioning)
            timesteps      = video_timesteps,
            cond_timesteps = video_cond_ts,
            grid_id        = latent_dict['grid_id'],      # same spatial grid as video
            targets        = mask_targets,
        )

        return {
            'latent_dict': latent_dict,
            'mask_dict':   mask_dict,
            'action_dict': action_dict,
            'mask_targets': mask_targets,
            'chunk_size':  torch.randint(1, 5, (1,)).item(),
            'window_size': torch.randint(4, 65, (1,)).item(),
        }

    def _to_device(self, input_dict):
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.to(self.device)
            elif isinstance(value, dict):
                self._to_device(value)
        return input_dict

    # ── Loss ─────────────────────────────────────────────────────────────────

    def compute_mask_loss(self, mask_pred, mask_targets, timesteps, valid_frames=None):
        """Flow-matching MSE loss, identical to the video latent loss."""
        mask_pred = data_seq_to_patch(
            self.patch_size, mask_pred,
            mask_targets.shape[-3], mask_targets.shape[-2], mask_targets.shape[-1],
            batch_size=mask_pred.shape[0])

        Bn, Fn = timesteps.shape
        loss_weight = self.train_scheduler_latent.training_weight(
            timesteps.flatten()).reshape(Bn, Fn)

        loss = F.mse_loss(mask_pred.float(), mask_targets.float().detach(),
                          reduction='none')
        loss = loss * loss_weight[:, None, :, None, None]

        loss = loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)  # (B*F, H*W*C)
        loss_per_frame = loss.sum(dim=1)
        elems_per_frame = torch.ones_like(loss).sum(dim=1)
        per_frame_loss  = loss_per_frame / (elems_per_frame + 1e-6)

        if valid_frames is not None:
            fm = valid_frames.float().flatten()
            scalar = (per_frame_loss * fm).sum() / (fm.sum() + 1e-6)
        else:
            scalar = per_frame_loss.mean()

        return scalar / self.gradient_accumulation_steps

    # ── Training step ────────────────────────────────────────────────────────

    def _train_step(self, batch, batch_idx):
        batch = self._to_device(batch)
        input_dict = self._prepare_input_dict(batch)

        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        if not should_sync:
            self.transformer.set_requires_gradient_sync(False)
        else:
            self.transformer.set_requires_gradient_sync(True)

        # Parallel heads: backbone (frozen, FlexAttn unchanged) → vid features →
        # proj_out (frozen) for video, mask_proj_out (trainable) for mask.
        _, _, mask_pred = self.transformer(input_dict, train_mode=True)

        valid_frames = batch.get('valid_frames', None)
        loss = self.compute_mask_loss(
            mask_pred, input_dict['mask_targets'],
            input_dict['latent_dict']['timesteps'], valid_frames)

        loss.backward()
        losses = {'mask_loss': loss.detach()}

        if should_sync:
            total_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.transformer.parameters() if p.requires_grad], 2.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            losses['total_norm'] = total_norm
            losses['should_log'] = True
        else:
            losses['should_log'] = False

        return losses

    # ── Checkpoint save/load ─────────────────────────────────────────────────

    def save_checkpoint(self):
        try:
            state_dict = get_model_state_dict(
                self.transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            optim_state = get_optimizer_state_dict(
                self.transformer, self.optimizer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )

            if self.config.rank == 0:
                ckpt_dir = self.save_dir / f"checkpoint_step_{self.step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)

                transformer_dir = ckpt_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)

                sd_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
                save_file(sd_bf16,
                          transformer_dir / "diffusion_pytorch_model.safetensors")

                cfg = dict(self.transformer.config)
                cfg.pop('_name_or_path', None)
                with open(transformer_dir / "config.json", 'w') as f:
                    json.dump(cfg, f, indent=2)

                mask_sd = {k: v.to(torch.bfloat16) for k, v in state_dict.items()
                           if 'mask_proj_out' in k}
                save_file(mask_sd, ckpt_dir / "mask_head.safetensors")

                torch.save({
                    'step': self.step,
                    'optimizer_state_dict': optim_state,
                    'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                }, ckpt_dir / "training_state.pt")

                logger.info(f"Checkpoint saved: {ckpt_dir}")

            if dist.is_initialized():
                dist.barrier()

        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Checkpoint save failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
            if dist.is_initialized():
                dist.barrier()

    def _load_training_state(self, checkpoint_path):
        state_path = Path(checkpoint_path) / "training_state.pt"
        if not state_path.exists():
            if self.config.rank == 0:
                logger.warning(f"training_state.pt not found in {checkpoint_path}")
            return

        state = torch.load(state_path, map_location='cpu', weights_only=False)
        set_optimizer_state_dict(
            self.transformer, self.optimizer,
            optim_state_dict=state['optimizer_state_dict'],
            options=StateDictOptions(full_state_dict=True, strict=False),
        )
        if 'lr_scheduler_state_dict' in state:
            self.lr_scheduler.load_state_dict(state['lr_scheduler_state_dict'])
        self.step = state.get('step', 0)
        if self.config.rank == 0:
            logger.info(f"Resumed from step {self.step}")
        if dist.is_initialized():
            dist.barrier()

    # ── Main training loop ───────────────────────────────────────────────────

    def train(self):
        logger.info(f"Starting joint-diffusion mask training for {self.config.num_steps} steps")
        self.transformer.train()

        pbar = tqdm(
            total=self.config.num_steps,
            desc="Joint Mask Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step,
        )

        self.optimizer.zero_grad()
        acc_losses = []
        step_in_acc = 0

        while self.step < self.config.num_steps:
            batch  = self._get_next_batch()
            losses = self._train_step(batch, step_in_acc)
            acc_losses.append(losses['mask_loss'])
            step_in_acc += 1

            if losses['should_log']:
                lr = self.lr_scheduler.get_last_lr()[0]
                loss_val = dist_mean(
                    torch.stack(acc_losses).sum()).detach().cpu().item()
                acc_losses  = []
                step_in_acc = 0

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    pbar.n += self.gradient_accumulation_steps
                    pbar.set_postfix({
                        'loss': f'{loss_val:.4f}',
                        'step': self.step,
                        'grad': f'{losses["total_norm"].item():.2f}',
                        'lr':   f'{lr:.2e}',
                    })

                self.step += 1

                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Saving checkpoint at step {self.step}")
                    self.save_checkpoint()

            if dist.is_initialized():
                dist.barrier()

        pbar.close()
        logger.info("Joint mask training complete!")


def run(args):
    config = VA_CONFIGS[args.config_name]

    rank       = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    config.rank       = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.save_root is not None:
        config.save_root = args.save_root
    if args.resume_from is not None:
        config.resume_from = args.resume_from

    if rank == 0:
        logger.info(f"Config: {args.config_name} | world_size={world_size}")

    trainer = JointMaskTrainer(config)

    if hasattr(config, 'resume_from') and config.resume_from:
        trainer._load_training_state(config.resume_from)

    trainer.train()


def main():
    parser = argparse.ArgumentParser(
        description="Joint-diffusion mask head training for WAN-VA")
    parser.add_argument("--config-name", type=str,
                        default='robotwin_mask_joint',
                        choices=['robotwin_mask_joint', 'robotwin_mask_joint_overfit'])
    parser.add_argument("--save-root", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Checkpoint dir, e.g. checkpoints/checkpoint_step_100")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
