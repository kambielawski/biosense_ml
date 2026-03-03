"""Checkpoint save/load utilities with top-K management."""

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    epoch: int,
    best_metric: float,
    cfg: DictConfig,
) -> None:
    """Save a training checkpoint.

    Args:
        path: File path to save the checkpoint.
        model: The model (handles DDP unwrapping).
        optimizer: The optimizer.
        scheduler: The LR scheduler.
        epoch: Current epoch number.
        best_metric: Best validation metric so far.
        cfg: Full Hydra config for reproducibility.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Unwrap DDP if needed
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
        "best_metric": best_metric,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    torch.save(checkpoint, path)
    logger.info("Saved checkpoint to %s (epoch %d, metric %.4f)", path, epoch, best_metric)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
) -> dict:
    """Load a training checkpoint.

    Args:
        path: Path to the checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional scheduler to restore state.

    Returns:
        Dict with "epoch" and "best_metric" keys.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.info(
        "Loaded checkpoint from %s (epoch %d, metric %.4f)",
        path,
        checkpoint["epoch"],
        checkpoint["best_metric"],
    )
    return {"epoch": checkpoint["epoch"], "best_metric": checkpoint["best_metric"]}


def manage_top_k_checkpoints(checkpoint_dir: Path, keep_top_k: int) -> None:
    """Keep only the top K most recent checkpoints, deleting older ones.

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        keep_top_k: Number of checkpoints to retain.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"), key=lambda p: p.stat().st_mtime)

    while len(checkpoints) > keep_top_k:
        oldest = checkpoints.pop(0)
        oldest.unlink()
        logger.info("Removed old checkpoint: %s", oldest)
