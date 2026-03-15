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


def get_sampling_rate(epoch: int, cfg: DictConfig) -> float:
    """Compute scheduled sampling rate for the current epoch."""
    if not cfg.training.get("scheduled_sampling", False):
        return 0.0
    ss_start = cfg.training.get("ss_start", 0.0)
    ss_end = cfg.training.get("ss_end", 0.5)
    ss_warmup = cfg.training.get("ss_warmup_epochs", 50)
    if ss_warmup <= 0:
        return ss_end
    progress = min(epoch / ss_warmup, 1.0)
    return ss_start + (ss_end - ss_start) * progress


def get_rollout_steps(epoch: int, cfg: DictConfig) -> int:
    """Compute multi-step rollout length K for the current epoch."""
    if not cfg.training.get("multistep_rollout", False):
        return 1
    max_k = cfg.training.get("max_rollout_steps", 8)
    warmup = cfg.training.get("rollout_warmup_epochs", 40)
    if warmup <= 0:
        return max_k
    # Start at K=1, linearly increase to max_k over warmup epochs
    progress = min(epoch / warmup, 1.0)
    return max(1, int(1 + (max_k - 1) * progress))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_type: str,
    context_len: int,
    grad_clip: float = 1.0,
    sampling_rate: float = 0.0,
    rollout_steps: int = 1,
    rollout_detach: bool = False,
) -> dict[str, float]:
    """Train for one epoch with optional scheduled sampling and multi-step rollout.

    Args:
        model: Trajectory model (GRU or MLP).
        loader: Training data loader.
        optimizer: Optimizer.
        device: Torch device.
        model_type: "gru" or "mlp".
        context_len: Context window size for MLP (K).
        grad_clip: Max gradient norm.
        sampling_rate: Probability of using model's own prediction as input (0=teacher forcing).
        rollout_steps: Number of autoregressive steps to unroll during training.
        rollout_detach: If True, detach gradients between rollout steps.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    use_rollout = sampling_rate > 0.0 or rollout_steps > 1

    for batch in loader:
        features = batch["features"].to(device)  # (B, T, 10)
        targets = batch["targets"].to(device)  # (B, T, 2)
        B, T, D = features.shape

        optimizer.zero_grad()

        if not use_rollout:
            # Original one-step teacher forcing (unchanged for backward compatibility)
            if model_type == "gru":
                pred_xy, _ = model(features)  # (B, T, 2)
            else:
                if T <= context_len:
                    continue
                windows = []
                window_targets = []
                for t in range(context_len, T):
                    windows.append(features[:, t - context_len : t])
                    window_targets.append(targets[:, t - 1])
                windows = torch.stack(windows, dim=1)
                window_targets = torch.stack(window_targets, dim=1)
                Bw, Tw, K, Dw = windows.shape
                pred_xy = model(windows.reshape(Bw * Tw, K, Dw))
                pred_xy = pred_xy.reshape(Bw, Tw, 2)
                targets = window_targets

            loss = nn.functional.mse_loss(pred_xy, targets)
        else:
            # Multi-step rollout with scheduled sampling
            loss = _rollout_loss(
                model, features, targets, model_type, context_len,
                sampling_rate, rollout_steps, rollout_detach,
            )
            if loss is None:
                continue

        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return {"train_loss": total_loss / max(n_batches, 1)}


def _rollout_loss(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    model_type: str,
    context_len: int,
    sampling_rate: float,
    rollout_steps: int,
    rollout_detach: bool,
) -> torch.Tensor | None:
    """Compute loss with multi-step rollout and scheduled sampling.

    Unrolls the model for `rollout_steps` steps, optionally mixing in
    the model's own predictions at each step (scheduled sampling).

    Args:
        model: Trajectory model.
        features: (B, T, 10) full sequence features.
        targets: (B, T, 2) target positions.
        model_type: "gru" or "mlp".
        context_len: Context window for MLP.
        sampling_rate: Probability of using model prediction instead of GT.
        rollout_steps: Number of steps to unroll.
        rollout_detach: Detach between rollout steps if True.

    Returns:
        Scalar loss tensor, or None if the sequence is too short.
    """
    B, T, D = features.shape
    # Need at least context + rollout_steps + 1 timesteps
    min_ctx = context_len if model_type == "mlp" else 2
    if T < min_ctx + rollout_steps + 1:
        return None

    # Pick a random start point for the rollout within the sequence
    # Context ends at `ctx_end`, rollout runs from ctx_end to ctx_end + rollout_steps
    max_start = T - rollout_steps - 1
    ctx_end = random.randint(min_ctx, max(min_ctx, max_start))
    actual_steps = min(rollout_steps, T - ctx_end - 1)
    if actual_steps < 1:
        return None

    all_preds = []
    all_targets = []

    if model_type == "gru":
        # Burn in: process context to build hidden state
        context = features[:, :ctx_end]  # (B, ctx_end, 10)
        _, hidden = model(context)  # hidden: (num_layers, B, hidden_dim)

        # Previous positions for constructing features
        prev_prev_xy = features[:, max(0, ctx_end - 2), :2]  # (B, 2)
        prev_xy = features[:, ctx_end - 1, :2]  # (B, 2) ground truth

        for k in range(actual_steps):
            t = ctx_end + k

            # Decide: use GT or model's prediction for input
            if k == 0 or random.random() >= sampling_rate:
                # Teacher forcing: use ground-truth features
                feat_k = features[:, t]  # (B, 10)
            else:
                # Scheduled sampling: construct features from model's prediction
                dt_k = features[:, t, 4:5]  # (B, 1) — always GT
                actions_k = features[:, t, 5:10]  # (B, 5) — always GT
                safe_dt = dt_k.clamp(min=1e-6)
                vel_k = (prev_xy - prev_prev_xy) / safe_dt  # (B, 2)
                feat_k = torch.cat([prev_xy, vel_k, dt_k, actions_k], dim=-1)  # (B, 10)

            # Forward step
            pred_xy, hidden = model.step(feat_k, hidden)
            pred_xy = torch.clamp(pred_xy, 0.0, 1.0)

            if rollout_detach:
                hidden = hidden.detach()

            all_preds.append(pred_xy)
            all_targets.append(targets[:, t])  # (B, 2)

            # Track positions for next step's velocity computation
            prev_prev_xy = prev_xy
            if random.random() < sampling_rate:
                prev_xy = pred_xy  # use prediction
            else:
                prev_xy = targets[:, t]  # use GT position

    else:  # MLP
        # Initialize sliding window from ground truth
        window = features[:, ctx_end - context_len : ctx_end].clone()  # (B, K, 10)

        for k in range(actual_steps):
            t = ctx_end + k

            # Predict from current window
            pred_xy = model(window)  # (B, 2)
            pred_xy = torch.clamp(pred_xy, 0.0, 1.0)

            all_preds.append(pred_xy)
            all_targets.append(targets[:, t])  # (B, 2)

            # Build next feature to slide into window
            if random.random() < sampling_rate and k > 0:
                # Use model's prediction
                prev_xy_for_vel = window[:, -1, :2]
                dt_k = features[:, t, 4:5]  # always GT
                actions_k = features[:, t, 5:10]  # always GT
                safe_dt = dt_k.clamp(min=1e-6)
                vel_k = (pred_xy - prev_xy_for_vel) / safe_dt
                next_feat = torch.cat([pred_xy, vel_k, dt_k, actions_k], dim=-1)
            else:
                # Teacher forcing: use ground-truth feature at this timestep
                next_feat = features[:, t]

            if rollout_detach:
                next_feat = next_feat.detach()

            # Slide window
            window = torch.cat([window[:, 1:], next_feat.unsqueeze(1)], dim=1)

    preds = torch.stack(all_preds, dim=1)  # (B, K, 2)
    tgts = torch.stack(all_targets, dim=1)  # (B, K, 2)
    return nn.functional.mse_loss(preds, tgts)


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

    rollout_detach = cfg.training.get("rollout_detach", False)
    use_ss = cfg.training.get("scheduled_sampling", False)
    use_ms = cfg.training.get("multistep_rollout", False)
    if use_ss or use_ms:
        logger.info(
            "Rollout training enabled: scheduled_sampling=%s (%.1f->%.1f over %d ep), "
            "multistep=%s (max_K=%d over %d ep), detach=%s",
            use_ss, cfg.training.get("ss_start", 0.0), cfg.training.get("ss_end", 0.5),
            cfg.training.get("ss_warmup_epochs", 50),
            use_ms, cfg.training.get("max_rollout_steps", 8),
            cfg.training.get("rollout_warmup_epochs", 40),
            rollout_detach,
        )

    for epoch in range(1, epochs + 1):
        t0 = time.monotonic()

        sampling_rate = get_sampling_rate(epoch, cfg)
        rollout_steps = get_rollout_steps(epoch, cfg)

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            model_type=model_type.replace("trajectory_", ""),
            context_len=context_len,
            grad_clip=cfg.training.gradient_clip,
            sampling_rate=sampling_rate,
            rollout_steps=rollout_steps,
            rollout_detach=rollout_detach,
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
            "sampling_rate": sampling_rate,
            "rollout_steps": rollout_steps,
        }
        if wandb.run:
            wandb.log(log_dict, step=epoch)

        logger.info(
            "Epoch %3d/%d | train=%.6f | val=%.6f | lr=%.2e | sr=%.2f | K=%d | %.1fs",
            epoch, epochs, train_metrics["train_loss"],
            val_metrics["val_loss"], lr, sampling_rate, rollout_steps, elapsed,
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
