"""Standalone training script for the RSSM world model.

Loads pre-encoded latent sequences from HDF5 (produced by encode_latents.py),
trains the RSSM with teacher forcing, and logs to W&B.

Usage:
    python scripts/train_rssm.py \
        model=rssm training=rssm \
        +training.latent_h5=data/rssm/latents.h5 \
        [overrides...]
"""

import logging
import random
import time
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb
from biosense_ml.models import build_model
from biosense_ml.training.metrics import MetricTracker
from biosense_ml.utils.checkpoint import manage_top_k_checkpoints, save_checkpoint
from biosense_ml.utils.logging import finish_wandb, init_wandb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RSSMSequenceDataset(Dataset):
    """Dataset that yields fixed-length sub-sequences from HDF5 latent data.

    Samples random contiguous sub-sequences of length seq_len from temporal
    sequences grouped by batch. If a sequence is shorter than seq_len,
    the entire sequence is returned (padded with zeros and masked).
    """

    def __init__(self, h5_path: str, seq_len: int, sequence_indices: list[int]) -> None:
        """Initialize dataset.

        Args:
            h5_path: Path to HDF5 file from encode_latents.py.
            seq_len: Target sub-sequence length.
            sequence_indices: Indices into sequence_starts/lengths to use
                (allows train/val splitting at the sequence level).
        """
        self.h5_path = h5_path
        self.seq_len = seq_len
        self.sequence_indices = sequence_indices
        self._file = None

        # Read sequence metadata (small, fine to do eagerly)
        with h5py.File(h5_path, "r") as f:
            self.seq_starts = np.array(f["sequence_starts"])
            self.seq_lengths = np.array(f["sequence_lengths"])
            self.ae_latent_dim = int(f.attrs["ae_latent_dim"])
            self.action_dim = int(f.attrs["action_dim"])

        # Build index: each entry is (seq_idx, start_offset_within_seq)
        # For sequences longer than seq_len, we create multiple windows
        self._windows: list[tuple[int, int]] = []
        for si in self.sequence_indices:
            slen = int(self.seq_lengths[si])
            if slen <= seq_len:
                self._windows.append((si, 0))
            else:
                # Sliding windows with stride = seq_len // 2
                stride = max(1, seq_len // 2)
                for offset in range(0, slen - seq_len + 1, stride):
                    self._windows.append((si, offset))
                # Ensure the last window is included
                if (slen - seq_len) % stride != 0:
                    self._windows.append((si, slen - seq_len))

    def _open(self) -> None:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self._open()
        si, offset = self._windows[idx]
        start = int(self.seq_starts[si]) + offset
        slen = min(self.seq_len, int(self.seq_lengths[si]) - offset)

        latents = np.array(self._file["latents"][start:start + slen], dtype=np.float32)
        actions = np.array(self._file["actions"][start:start + slen], dtype=np.float32)

        # Pad if shorter than seq_len
        actual_len = slen
        if slen < self.seq_len:
            pad_len = self.seq_len - slen
            latents = np.pad(latents, ((0, pad_len), (0, 0)), mode="constant")
            actions = np.pad(actions, ((0, pad_len), (0, 0)), mode="constant")

        # Mask: 1 for valid timesteps, 0 for padding
        mask = np.zeros(self.seq_len, dtype=np.float32)
        mask[:actual_len] = 1.0

        return {
            "latents": torch.from_numpy(latents),
            "actions": torch.from_numpy(actions),
            "mask": torch.from_numpy(mask),
        }


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_rssm_loss(
    outputs: dict[str, torch.Tensor],
    mask: torch.Tensor,
    beta_kl: float = 1.0,
    kl_balance_alpha: float = 0.8,
    free_bits: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute RSSM loss with reconstruction, KL balancing, and free bits.

    Args:
        outputs: Dict from RSSM.forward() with prior/posterior params and predictions.
        mask: Valid timestep mask, shape (B, T).
        beta_kl: Weight for KL loss.
        kl_balance_alpha: KL balancing coefficient (0.8 = push prior toward posterior).
        free_bits: Per-dimension KL floor in nats.

    Returns:
        Dict with loss, recon_loss, kl_loss, kl_dyn, kl_rep (all scalar tensors).
    """
    from biosense_ml.models.rssm import RSSM

    obs_pred = outputs["obs_pred"]       # (B, T, ae_latent_dim)
    obs_target = outputs["obs_target"]   # (B, T, ae_latent_dim)
    post_mu = outputs["post_mu"]         # (B, T, z_dim)
    post_sigma = outputs["post_sigma"]
    prior_mu = outputs["prior_mu"]
    prior_sigma = outputs["prior_sigma"]

    # Expand mask for broadcasting: (B, T) -> (B, T, 1)
    mask_3d = mask.unsqueeze(-1)

    # --- Reconstruction loss: MSE on projected latents ---
    recon_sq = (obs_pred - obs_target) ** 2  # (B, T, ae_latent_dim)
    recon_loss = (recon_sq * mask_3d).sum() / mask.sum()

    # --- KL loss with balancing (DreamerV3 Eq. 3) ---
    # L_dyn: train sequence model, sg(posterior) || prior
    # L_rep: train encoder, posterior || sg(prior)
    kl_per_dim = RSSM.kl_divergence(post_mu, post_sigma, prior_mu, prior_sigma)  # (B, T, z_dim)

    # Dynamic loss: sg(q) || p — stop gradient on posterior
    kl_dyn_per_dim = RSSM.kl_divergence(
        post_mu.detach(), post_sigma.detach(), prior_mu, prior_sigma
    )
    kl_dyn_clamped = torch.clamp(kl_dyn_per_dim, min=free_bits)
    kl_dyn = (kl_dyn_clamped * mask_3d).sum() / mask.sum()

    # Representation loss: q || sg(p) — stop gradient on prior
    kl_rep_per_dim = RSSM.kl_divergence(
        post_mu, post_sigma, prior_mu.detach(), prior_sigma.detach()
    )
    kl_rep_clamped = torch.clamp(kl_rep_per_dim, min=free_bits)
    kl_rep = (kl_rep_clamped * mask_3d).sum() / mask.sum()

    # Balanced KL
    kl_loss = kl_balance_alpha * kl_dyn + (1.0 - kl_balance_alpha) * kl_rep

    # Total loss
    total_loss = recon_loss + beta_kl * kl_loss

    # Unbalanced KL for monitoring
    kl_raw = (kl_per_dim * mask_3d).sum() / mask.sum()

    return {
        "loss": total_loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
        "kl_raw": kl_raw,
        "kl_dyn": kl_dyn,
        "kl_rep": kl_rep,
    }


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
    cfg: DictConfig,
    noise_std: float = 0.0,
) -> dict[str, float]:
    """Run one epoch of RSSM training or validation.

    Args:
        model: The RSSM model.
        loader: DataLoader yielding sequence batches.
        optimizer: Optimizer (None during validation).
        scaler: GradScaler for AMP.
        device: Target device.
        use_amp: Whether to use AMP.
        train: If True, run in training mode.
        cfg: Training config (for loss hyperparams).
        noise_std: Noise injection std on projected latents.

    Returns:
        Dict of averaged metrics.
    """
    model.train(train)
    tracker = MetricTracker()

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            latents = batch["latents"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            # Optional noise injection on latents
            if train and noise_std > 0:
                noise = torch.randn_like(latents) * noise_std
                latents = latents + noise

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(latents, actions)
                losses = compute_rssm_loss(
                    outputs, mask,
                    beta_kl=cfg.training.loss.beta_kl,
                    kl_balance_alpha=cfg.training.kl_balance_alpha,
                    free_bits=cfg.training.free_bits,
                )
                loss = losses["loss"]

            if train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if cfg.training.gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.training.gradient_clip
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if cfg.training.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.training.gradient_clip
                        )
                    optimizer.step()

            B = latents.size(0)
            for k, v in losses.items():
                tracker.update(k, v.item(), n=B)

    return tracker.all_averages()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train the RSSM world model.

    Args:
        cfg: Full Hydra config (model=rssm, training=rssm).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.training.mixed_precision and torch.cuda.is_available()
    logger.info("Device: %s  AMP: %s", device, use_amp)

    # Reproducibility
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Paths
    project_root = Path(hydra.utils.get_original_cwd())
    h5_path = str(project_root / cfg.training.latent_h5)
    logger.info("Loading latent data from %s", h5_path)

    # Read dataset metadata
    with h5py.File(h5_path, "r") as f:
        num_sequences = int(f.attrs["num_sequences"])
        ae_latent_dim = int(f.attrs["ae_latent_dim"])
        action_dim = int(f.attrs["action_dim"])
        num_samples = int(f.attrs["num_samples"])
    logger.info("Dataset: %d samples, %d sequences, ae_dim=%d, action_dim=%d",
                num_samples, num_sequences, ae_latent_dim, action_dim)

    # Verify config matches data
    if cfg.model.ae_latent_dim != ae_latent_dim:
        logger.warning(
            "Config ae_latent_dim=%d != HDF5 ae_latent_dim=%d. Using HDF5 value.",
            cfg.model.ae_latent_dim, ae_latent_dim,
        )
        cfg.model.ae_latent_dim = ae_latent_dim
    if cfg.model.action_dim != action_dim:
        logger.warning(
            "Config action_dim=%d != HDF5 action_dim=%d. Using HDF5 value.",
            cfg.model.action_dim, action_dim,
        )
        cfg.model.action_dim = action_dim

    # Train/val split at sequence level
    seq_indices = list(range(num_sequences))
    random.shuffle(seq_indices)
    split_idx = int(len(seq_indices) * cfg.training.train_ratio)
    if split_idx == 0:
        train_indices = seq_indices
        val_indices = seq_indices
    else:
        train_indices = seq_indices[:split_idx]
        val_indices = seq_indices[split_idx:]
    logger.info("Train sequences: %d, Val sequences: %d", len(train_indices), len(val_indices))

    # Datasets & loaders
    seq_len = cfg.training.seq_len
    train_dataset = RSSMSequenceDataset(h5_path, seq_len, train_indices)
    val_dataset = RSSMSequenceDataset(h5_path, seq_len, val_indices)

    batch_size = cfg.data.batch_size
    num_workers = min(cfg.data.num_workers, 4)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    logger.info("Train windows: %d, Val windows: %d", len(train_dataset), len(val_dataset))

    # Model
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("RSSM parameters: %d (%.2fM)", n_params, n_params / 1e6)

    # Optimizer / scheduler / scaler
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

    # Checkpointing
    ckpt_subdir = cfg.training.get("checkpoint_dir", "outputs/rssm/checkpoints")
    ckpt_dir = project_root / ckpt_subdir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    # W&B
    run = init_wandb(cfg)

    # Training loop
    epochs = cfg.training.epochs
    epoch_durations = []

    # Noise annealing schedule
    noise_init = cfg.training.noise_std_init
    noise_final = cfg.training.noise_std_final
    noise_anneal_epochs = int(epochs * cfg.training.noise_anneal_fraction)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Compute current noise std
        if noise_anneal_epochs > 0 and epoch <= noise_anneal_epochs:
            frac = (epoch - 1) / noise_anneal_epochs
            noise_std = noise_init + (noise_final - noise_init) * frac
        else:
            noise_std = noise_final

        # Train
        train_metrics = run_epoch(
            model, train_loader, optimizer, scaler, device, use_amp,
            train=True, cfg=cfg, noise_std=noise_std,
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Validate
        val_metrics = run_epoch(
            model, val_loader, None, None, device, use_amp,
            train=False, cfg=cfg, noise_std=0.0,
        )

        epoch_duration = time.time() - epoch_start
        epoch_durations.append(epoch_duration)
        avg_epoch_s = sum(epoch_durations) / len(epoch_durations)
        remaining_h = avg_epoch_s * (epochs - epoch) / 3600

        # Log
        log_dict = {
            "lr": current_lr,
            "noise_std": noise_std,
            "perf/epoch_time_min": epoch_duration / 60,
            "perf/estimated_remaining_h": remaining_h,
        }
        for k, v in train_metrics.items():
            log_dict[f"train/{k}"] = v
        for k, v in val_metrics.items():
            log_dict[f"val/{k}"] = v
        wandb.log(log_dict, step=epoch)

        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  recon=%.4f  kl=%.4f  "
            "lr=%.2e  noise=%.4f  time=%.1fmin  remaining=%.1fh",
            epoch, epochs,
            train_metrics["loss"], val_metrics["loss"],
            val_metrics["recon_loss"], val_metrics["kl_raw"],
            current_lr, noise_std,
            epoch_duration / 60, remaining_h,
        )

        # Checkpoint
        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]

        if epoch % cfg.training.checkpoint_every == 0 or is_best:
            tag = "best" if is_best else f"epoch{epoch:04d}"
            ckpt_path = ckpt_dir / f"checkpoint_{tag}.pt"
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, best_val_loss, cfg)
            manage_top_k_checkpoints(ckpt_dir, cfg.training.keep_top_k)

    logger.info("Training complete. Best val loss: %.6f", best_val_loss)
    finish_wandb()


if __name__ == "__main__":
    main()
