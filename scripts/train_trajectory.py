#!/usr/bin/env python3
"""Training script for trajectory prediction models (MLP baseline, GRU).

Loads the trajectory HDF5 dataset, trains the selected model with next-step
prediction loss, and evaluates with autoregressive rollout.

Usage:
    python scripts/train_trajectory.py \
        model=trajectory_gru training=trajectory \
        +training.trajectory_h5=data/trajectory_dataset.h5

    python scripts/train_trajectory.py \
        model=trajectory_mlp training=trajectory \
        +training.trajectory_h5=data/trajectory_dataset.h5
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
from tqdm import tqdm

import wandb
from biosense_ml.evaluation.trajectory_eval import evaluate_dataset
from biosense_ml.models import build_model
from biosense_ml.pipeline.trajectory_dataset import (
    TrajectoryDataset,
    TrajectorySequenceDataset,
)
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
    model_type: str,
    context_len: int,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """Train for one epoch with next-step prediction loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        features = batch["features"].to(device)  # (B, T, 10)
        targets = batch["targets"].to(device)  # (B, T, 2)

        optimizer.zero_grad()

        if model_type == "gru":
            pred_xy, _ = model(features)  # (B, T, 2)
        else:
            # MLP: slide context_len window over sequence
            B, T, D = features.shape
            if T <= context_len:
                continue
            # Gather all windows
            windows = []
            window_targets = []
            for t in range(context_len, T):
                windows.append(features[:, t - context_len : t])  # (B, K, 10)
                window_targets.append(targets[:, t - 1])  # (B, 2) — target is t-th pos
            windows = torch.stack(windows, dim=1)  # (B, T-K, K, 10)
            window_targets = torch.stack(window_targets, dim=1)  # (B, T-K, 2)
            Bw, Tw, K, D = windows.shape
            pred_xy = model(windows.reshape(Bw * Tw, K, D))  # (B*(T-K), 2)
            pred_xy = pred_xy.reshape(Bw, Tw, 2)
            targets = window_targets

        loss = nn.functional.mse_loss(pred_xy, targets)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return {"train_loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_type: str,
    context_len: int,
) -> dict[str, float]:
    """Compute validation loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)

        if model_type == "gru":
            pred_xy, _ = model(features)
        else:
            B, T, D = features.shape
            if T <= context_len:
                continue
            windows = []
            window_targets = []
            for t in range(context_len, T):
                windows.append(features[:, t - context_len : t])
                window_targets.append(targets[:, t - 1])
            windows = torch.stack(windows, dim=1)
            window_targets = torch.stack(window_targets, dim=1)
            Bw, Tw, K, D = windows.shape
            pred_xy = model(windows.reshape(Bw * Tw, K, D))
            pred_xy = pred_xy.reshape(Bw, Tw, 2)
            targets = window_targets

        loss = nn.functional.mse_loss(pred_xy, targets)
        total_loss += loss.item()
        n_batches += 1

    return {"val_loss": total_loss / max(n_batches, 1)}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    h5_path = cfg.training.trajectory_h5
    model_type = cfg.model.name  # "trajectory_gru" or "trajectory_mlp"
    seq_len = cfg.training.seq_len
    context_len = cfg.model.get("context_len", 10)
    checkpoint_dir = Path(cfg.training.get(
        "checkpoint_dir", f"outputs/{model_type}/checkpoints"
    ))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Init W&B (bypass init_wandb which expects data.preprocessing.mode)
    run_name = f"{model_type}_{time.strftime('%Y%m%d_%H%M%S')}"
    try:
        wandb.init(
            project=cfg.project_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name,
            tags=[model_type, "trajectory"],
        )
    except Exception as e:
        logger.warning("W&B init failed: %s. Continuing without logging.", e)

    # Datasets
    train_ds = TrajectoryDataset(h5_path, split="train", seq_len=seq_len, context_len=context_len)
    val_ds = TrajectoryDataset(h5_path, split="val", seq_len=seq_len, context_len=context_len)

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

    # Model
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s (%d params)", model_type, n_params)
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
            model_type=model_type.replace("trajectory_", ""),
            context_len=context_len,
            grad_clip=cfg.training.gradient_clip,
        )
        val_metrics = validate(
            model, val_loader, device,
            model_type=model_type.replace("trajectory_", ""),
            context_len=context_len,
        )
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
            "Epoch %3d/%d | train_loss=%.6f | val_loss=%.6f | lr=%.2e | %.1fs",
            epoch, epochs, train_metrics["train_loss"],
            val_metrics["val_loss"], lr, elapsed,
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

    # Final evaluation with autoregressive rollout
    logger.info("Running final evaluation with autoregressive rollout...")
    eval_ds = TrajectorySequenceDataset(h5_path, split="val")
    if len(eval_ds) > 0:
        # Load best checkpoint
        best_ckpt = checkpoint_dir / "checkpoint_best.pt"
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            logger.info("Loaded best checkpoint (epoch %d, best_metric=%.6f)",
                        ckpt["epoch"], ckpt["best_metric"])

        eval_results = evaluate_dataset(
            model=model,
            sequences=[eval_ds[i] for i in range(len(eval_ds))],
            burn_in_frac=cfg.training.burn_in_frac,
            time_horizons=list(cfg.training.eval_time_horizons),
            model_type=model_type.replace("trajectory_", ""),
            context_len=context_len,
        )
        logger.info("Eval results: %s", eval_results)
        if wandb.run:
            wandb.log({f"eval/{k}": v for k, v in eval_results.items()
                       if isinstance(v, (int, float))})

    finish_wandb()
    logger.info("Training complete. Best val_loss=%.6f", best_val_loss)


if __name__ == "__main__":
    main()
