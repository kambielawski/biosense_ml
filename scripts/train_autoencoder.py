"""Standalone training script for the convolutional autoencoder.

Loads shards directly from data/processed/manifest.json, streams them via
WebDataset, and trains ConvAutoencoder with MSE reconstruction loss.

Usage:
    python scripts/train_autoencoder.py model=autoencoder [overrides...]
"""

import json
import logging
import random
import time
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import webdataset as wds
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from biosense_ml.models import build_model
from biosense_ml.pipeline.transforms import get_transforms
from biosense_ml.utils.checkpoint import manage_top_k_checkpoints, save_checkpoint
from biosense_ml.utils.logging import finish_wandb, init_wandb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_shard_paths(manifest_path: Path) -> list[str]:
    """Read shard paths from manifest.json.

    Args:
        manifest_path: Absolute path to manifest.json.

    Returns:
        List of shard paths as strings (relative to project root).
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    return manifest["shard_paths"]


def make_decode_fn(transform):
    """Return a sample-decode function that applies *transform* to the PIL image.

    WebDataset's .decode("pil") already converts the JPEG bytes to a PIL Image
    and stores it under the "jpg" key (no leading dot — the dot-prefix
    convention is for raw/undecoded samples only). This function picks up that
    PIL image, applies the torchvision transform, and returns a dict so that
    wds.batched() can properly stack the tensors across samples.

    Note: when .map() returns a plain tensor (not a dict), wds 1.0.x's
    .batched() wraps items in a dict with integer keys rather than stacking
    them. Returning {"image": tensor} ensures .batched() produces
    {"image": (B, C, H, W)} which can be indexed cleanly in the training loop.

    Args:
        transform: A torchvision Compose transform (output of get_transforms).

    Returns:
        Callable that maps a wds sample dict to {"image": float32 tensor}.
    """
    def decode_fn(sample: dict) -> dict:
        img = sample["jpg"]           # PIL Image (already decoded by wds)
        return {"image": transform(img)}   # -> {"image": (C, H, W) float tensor}
    return decode_fn


def build_loader(
    shard_paths: list[str],
    project_root: Path,
    transform,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    shuffle_buffer: int,
) -> DataLoader:
    """Build a DataLoader backed by a WebDataset shard list.

    Args:
        shard_paths: Relative shard paths from the manifest.
        project_root: Absolute path to the project root (prepended to each shard path).
        transform: Torchvision transform pipeline to apply to each image.
        batch_size: Batch size.
        num_workers: DataLoader worker count.
        shuffle: If True, shuffle shards and apply an in-memory shuffle buffer.
        shuffle_buffer: Size of the WebDataset shuffle buffer (used when shuffle=True).

    Returns:
        A DataLoader yielding {"image": tensor} dicts where the tensor has
        shape (B, C, H, W).
    """
    abs_paths = [str(project_root / p) for p in shard_paths]

    dataset = wds.WebDataset(abs_paths, shardshuffle=shuffle)
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = (
        dataset
        .decode("pil")
        .map(make_decode_fn(transform))
        .batched(batch_size, partial=True)
    )

    loader = DataLoader(
        dataset,
        batch_size=None,   # batching is handled by wds .batched()
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler | None,
    device: torch.device,
    use_amp: bool,
    train: bool,
) -> float:
    """Run one full epoch (train or val).

    Args:
        model: The ConvAutoencoder.
        loader: DataLoader yielding image tensors.
        optimizer: Optimizer (None during validation).
        scaler: GradScaler for AMP (None when AMP disabled).
        device: Target device.
        use_amp: Whether to use automatic mixed precision.
        train: If True, run in training mode with gradient updates.

    Returns:
        Mean MSE loss over the epoch.
    """
    model.train(train)
    criterion = nn.MSELoss()
    total_loss = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            # batch is a dict {"image": (B, C, H, W)} from wds .batched()
            images = batch["image"].to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                reconstruction, _ = model(images)
                loss = criterion(reconstruction, images)

            if train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train the ConvAutoencoder.

    Args:
        cfg: Full Hydra config (model=autoencoder must be selected).
    """
    # ---- device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.training.mixed_precision and torch.cuda.is_available()
    logger.info("Device: %s  AMP: %s", device, use_amp)

    # ---- reproducibility ----
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # ---- manifest -> shards ----
    # Hydra changes cwd to outputs dir; use the original cwd stored by Hydra.
    project_root = Path(hydra.utils.get_original_cwd())
    manifest_path = project_root / "data" / "processed" / "manifest.json"
    logger.info("Loading manifest from %s", manifest_path)
    all_shards = load_shard_paths(manifest_path)
    logger.info("Total shards in manifest: %d", len(all_shards))

    # Optional shard cap for fast experiments (sanity-check, dev runs).
    # Pass via CLI: +training.max_shards=1  (or any integer N)
    max_shards = cfg.training.get("max_shards", None)
    if max_shards is not None:
        all_shards = all_shards[:int(max_shards)]
        logger.info("max_shards=%d applied — using %d shards", max_shards, len(all_shards))

    logger.info("Total shards: %d", len(all_shards))

    # Deterministic 90/10 train/val split on shards.
    # Guard: with very few shards (e.g. max_shards=1), floor(n*0.9) can be 0,
    # leaving train empty.  In that case use all shards for both splits so that
    # sanity-check and tiny-dataset runs always have a non-empty train loader.
    random.shuffle(all_shards)
    split_idx = int(len(all_shards) * 0.9)
    if split_idx == 0:
        logger.warning(
            "split_idx=0 with %d total shards — using all shards for both "
            "train and val (overfitting mode intended for sanity-check runs)",
            len(all_shards),
        )
        train_shards = all_shards
        val_shards = all_shards
    else:
        train_shards = all_shards[:split_idx]
        val_shards = all_shards[split_idx:]
    logger.info("Train shards: %d  Val shards: %d", len(train_shards), len(val_shards))

    # ---- transforms ----
    train_transform = get_transforms(cfg, split="train")
    val_transform = get_transforms(cfg, split="val")

    # ---- data loaders ----
    batch_size: int = cfg.data.batch_size
    num_workers: int = cfg.data.num_workers
    shuffle_buffer: int = cfg.data.shuffle_buffer

    train_loader = build_loader(
        train_shards, project_root, train_transform,
        batch_size, num_workers, shuffle=True, shuffle_buffer=shuffle_buffer,
    )
    val_loader = build_loader(
        val_shards, project_root, val_transform,
        batch_size, num_workers, shuffle=False, shuffle_buffer=shuffle_buffer,
    )

    # ---- model ----
    model = build_model(cfg).to(device)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # ---- optimizer / scheduler / scaler ----
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
    scaler = GradScaler() if use_amp else None

    # ---- checkpointing ----
    ckpt_dir = project_root / "outputs" / "autoencoder" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    # ---- W&B ----
    run = init_wandb(cfg)

    # ---- training loop ----
    epochs: int = cfg.training.epochs
    epoch_durations: list[float] = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = run_epoch(
            model, train_loader, optimizer, scaler, device, use_amp, train=True
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Validate
        val_loss = run_epoch(
            model, val_loader, None, None, device, use_amp, train=False
        )

        epoch_duration = time.time() - epoch_start
        epoch_durations.append(epoch_duration)
        avg_epoch_s = sum(epoch_durations) / len(epoch_durations)
        remaining_h = avg_epoch_s * (epochs - epoch) / 3600
        total_h = avg_epoch_s * epochs / 3600

        # Log
        wandb.log(
            {
                "train/loss": train_loss,
                "val/loss": val_loss,
                "lr": current_lr,
                "perf/epoch_time_min": epoch_duration / 60,
                "perf/estimated_total_h": total_h,
                "perf/estimated_remaining_h": remaining_h,
            },
            step=epoch,
        )
        logger.info(
            "Epoch %d/%d  train_loss=%.6f  val_loss=%.6f  lr=%.2e"
            "  epoch_time=%.1fmin  remaining=%.1fh",
            epoch, epochs, train_loss, val_loss, current_lr,
            epoch_duration / 60, remaining_h,
        )

        # Checkpoint every N epochs or when val improves
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if epoch % cfg.training.checkpoint_every == 0 or is_best:
            tag = "best" if is_best else f"epoch{epoch:04d}"
            ckpt_path = ckpt_dir / f"checkpoint_{tag}.pt"
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, best_val_loss, cfg)
            manage_top_k_checkpoints(ckpt_dir, cfg.training.keep_top_k)

    logger.info("Training complete. Best val loss: %.6f", best_val_loss)
    finish_wandb()


if __name__ == "__main__":
    main()
