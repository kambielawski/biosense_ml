#!/usr/bin/env python3
"""Training script for gaze-based dynamics model.

Loads the gaze HDF5 dataset, trains the GazeDynamics model with
crop reconstruction + position delta loss.

Usage:
    python scripts/train_gaze.py \
        model=gaze_dynamics training=gaze \
        +training.gaze_h5=data/gaze/gaze_dataset.h5
"""

import logging
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import wandb
from biosense_ml.models import build_model
from biosense_ml.pipeline.gaze_dataset import GazeCropDataset
from biosense_ml.utils.checkpoint import manage_top_k_checkpoints, save_checkpoint
from biosense_ml.utils.logging import finish_wandb

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_pos: float = 1.0,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """Train for one epoch.

    Returns dict with train_loss, train_crop_loss, train_pos_loss.
    """
    model.train()
    total_loss = 0.0
    total_crop_loss = 0.0
    total_pos_loss = 0.0
    n_batches = 0

    for batch in loader:
        context_crops = batch["context_crops"].to(device)  # (B, K, 3, 32, 32)
        target_crop = batch["target_crop"].to(device)  # (B, 3, 32, 32)
        target_delta = batch["target_delta"].to(device)  # (B, 2)

        optimizer.zero_grad()

        pred_crop, pred_delta = model(context_crops)

        crop_loss = nn.functional.l1_loss(pred_crop, target_crop)
        pos_loss = nn.functional.l1_loss(pred_delta, target_delta)
        loss = crop_loss + lambda_pos * pos_loss

        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        total_crop_loss += crop_loss.item()
        total_pos_loss += pos_loss.item()
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "train_loss": total_loss / n,
        "train_crop_loss": total_crop_loss / n,
        "train_pos_loss": total_pos_loss / n,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lambda_pos: float = 1.0,
) -> dict[str, float]:
    """Compute validation loss."""
    model.eval()
    total_loss = 0.0
    total_crop_loss = 0.0
    total_pos_loss = 0.0
    n_batches = 0

    for batch in loader:
        context_crops = batch["context_crops"].to(device)
        target_crop = batch["target_crop"].to(device)
        target_delta = batch["target_delta"].to(device)

        pred_crop, pred_delta = model(context_crops)

        crop_loss = nn.functional.l1_loss(pred_crop, target_crop)
        pos_loss = nn.functional.l1_loss(pred_delta, target_delta)
        loss = crop_loss + lambda_pos * pos_loss

        total_loss += loss.item()
        total_crop_loss += crop_loss.item()
        total_pos_loss += pos_loss.item()
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "val_loss": total_loss / n,
        "val_crop_loss": total_crop_loss / n,
        "val_pos_loss": total_pos_loss / n,
    }


@torch.no_grad()
def compute_baselines(
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Compute copy-last-crop and zero-delta baselines.

    Returns:
        copy_crop_l1: L1 of using the last context crop as prediction.
        zero_delta_l1: L1 of predicting zero position delta.
    """
    total_crop_l1 = 0.0
    total_delta_l1 = 0.0
    n_batches = 0

    for batch in loader:
        context_crops = batch["context_crops"].to(device)  # (B, K, 3, 32, 32)
        target_crop = batch["target_crop"].to(device)  # (B, 3, 32, 32)
        target_delta = batch["target_delta"].to(device)  # (B, 2)

        # Baseline 1: copy last crop
        last_crop = context_crops[:, -1]  # (B, 3, 32, 32)
        total_crop_l1 += nn.functional.l1_loss(last_crop, target_crop).item()

        # Baseline 2: predict zero delta
        zero_delta = torch.zeros_like(target_delta)
        total_delta_l1 += nn.functional.l1_loss(zero_delta, target_delta).item()

        n_batches += 1

    n = max(n_batches, 1)
    return {
        "baseline_copy_crop_l1": total_crop_l1 / n,
        "baseline_zero_delta_l1": total_delta_l1 / n,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.training.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    h5_path = cfg.training.gaze_h5
    context_len = cfg.model.context_len
    lambda_pos = cfg.training.loss.lambda_pos
    checkpoint_dir = Path(cfg.training.get(
        "checkpoint_dir", "outputs/gaze_dynamics/checkpoints"
    ))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Init W&B
    run_name = f"gaze_dynamics_{time.strftime('%Y%m%d_%H%M%S')}"
    try:
        wandb.init(
            project=cfg.project_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name,
            tags=["gaze_dynamics", "gaze"],
        )
    except Exception as e:
        logger.warning("W&B init failed: %s. Continuing without logging.", e)

    # Datasets
    train_ds = GazeCropDataset(h5_path, split="train", context_len=context_len)
    val_ds = GazeCropDataset(h5_path, split="val", context_len=context_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    # Compute baselines on val set
    baseline_metrics = compute_baselines(val_loader, device)
    logger.info("Baselines: %s", baseline_metrics)
    if wandb.run:
        wandb.log(baseline_metrics, step=0)

    # Model
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: gaze_dynamics (%d params)", n_params)
    if wandb.run:
        wandb.config.update({"n_params": n_params})

    # Optimizer & scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        betas=tuple(cfg.training.optimizer.betas),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs,
        eta_min=cfg.training.scheduler.min_lr,
    )

    # Training loop
    best_val_loss = float("inf")
    epochs = cfg.training.epochs

    for epoch in range(1, epochs + 1):
        t0 = time.monotonic()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            lambda_pos=lambda_pos,
            grad_clip=cfg.training.gradient_clip,
        )
        val_metrics = validate(model, val_loader, device, lambda_pos=lambda_pos)
        scheduler.step()

        elapsed = time.monotonic() - t0
        lr = optimizer.param_groups[0]["lr"]

        log_dict = {
            **train_metrics,
            **val_metrics,
            "lr": lr,
            "epoch": epoch,
            "epoch_time": elapsed,
        }
        if wandb.run:
            wandb.log(log_dict, step=epoch)

        logger.info(
            "Epoch %3d/%d | train=%.4f (crop=%.4f pos=%.4f) | "
            "val=%.4f (crop=%.4f pos=%.4f) | lr=%.2e | %.1fs",
            epoch, epochs,
            train_metrics["train_loss"],
            train_metrics["train_crop_loss"],
            train_metrics["train_pos_loss"],
            val_metrics["val_loss"],
            val_metrics["val_crop_loss"],
            val_metrics["val_pos_loss"],
            lr, elapsed,
        )

        # Checkpointing
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_checkpoint(
                checkpoint_dir / "checkpoint_best.pt",
                model, optimizer, scheduler, epoch,
                best_val_loss, cfg,
            )

        if epoch % cfg.training.checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_epoch{epoch:04d}.pt"
            save_checkpoint(
                ckpt_path, model, optimizer, scheduler, epoch,
                val_metrics["val_loss"], cfg,
            )
            manage_top_k_checkpoints(checkpoint_dir, cfg.training.keep_top_k)

    # Log final comparison vs baselines
    logger.info(
        "Final val crop L1: %.4f vs copy-last baseline: %.4f",
        val_metrics["val_crop_loss"],
        baseline_metrics["baseline_copy_crop_l1"],
    )
    logger.info(
        "Final val pos L1: %.4f vs zero-delta baseline: %.4f",
        val_metrics["val_pos_loss"],
        baseline_metrics["baseline_zero_delta_l1"],
    )

    finish_wandb()
    logger.info("Training complete. Best val_loss=%.4f", best_val_loss)


if __name__ == "__main__":
    main()
