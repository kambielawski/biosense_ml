"""Training loop with mixed precision, checkpointing, wandb logging, and DDP support."""

import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from biosense_ml.data.dataset import make_dataloader
from biosense_ml.models import build_model
from biosense_ml.training.metrics import MetricTracker
from biosense_ml.utils.checkpoint import load_checkpoint, manage_top_k_checkpoints, save_checkpoint
from biosense_ml.utils.distributed import cleanup_distributed, is_main_process, setup_distributed
from biosense_ml.utils.logging import finish_wandb, init_wandb, log_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """Manages the full training lifecycle: setup, training loop, validation, checkpointing."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._set_seed(cfg.seed)

        # Distributed setup
        self.rank, self.world_size, self.local_rank = setup_distributed()

        # Device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")
        logger.info("Using device: %s", self.device)

        # Model
        self.model = build_model(cfg).to(self.device)
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])

        # Optimizer
        self.optimizer = self._build_optimizer(cfg)

        # Scheduler
        self.scheduler = self._build_scheduler(cfg)

        # Loss
        self.criterion = self._build_loss(cfg)

        # Mixed precision
        self.use_amp = cfg.training.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Data
        self.train_loader = make_dataloader(cfg, split="train")
        self.val_loader = make_dataloader(cfg, split="val")

        # Checkpoint directory
        self.checkpoint_dir = Path(cfg.output_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Tracking
        self.start_epoch = 0
        self.best_metric = float("inf")  # Lower is better (loss); flip for accuracy
        self.global_step = 0

        # Resume from checkpoint if specified
        if cfg.get("resume_from"):
            info = load_checkpoint(
                Path(cfg.resume_from), self.model, self.optimizer, self.scheduler
            )
            self.start_epoch = info["epoch"] + 1
            self.best_metric = info["best_metric"]
            logger.info("Resuming from epoch %d", self.start_epoch)

        # Wandb (main process only)
        if is_main_process():
            init_wandb(cfg)

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_optimizer(self, cfg: DictConfig) -> torch.optim.Optimizer:
        """Build optimizer from config."""
        opt_cfg = cfg.training.optimizer
        if opt_cfg.name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                betas=tuple(opt_cfg.betas),
            )
        elif opt_cfg.name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_cfg.name}")

    def _build_scheduler(self, cfg: DictConfig) -> torch.optim.lr_scheduler.LRScheduler:
        """Build LR scheduler from config."""
        sched_cfg = cfg.training.scheduler
        if sched_cfg.name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.training.epochs - sched_cfg.warmup_epochs,
                eta_min=sched_cfg.min_lr,
            )
        elif sched_cfg.name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_cfg.name}")

    def _build_loss(self, cfg: DictConfig) -> nn.Module:
        """Build loss function from config."""
        loss_name = cfg.training.loss.name
        if loss_name == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif loss_name == "mse":
            return nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss: {loss_name}")

    def fit(self) -> None:
        """Run the full training loop."""
        logger.info("Starting training for epochs %d-%d", self.start_epoch, self.cfg.training.epochs - 1)

        try:
            for epoch in range(self.start_epoch, self.cfg.training.epochs):
                train_metrics = self._train_one_epoch(epoch)

                if epoch % self.cfg.training.val_every == 0:
                    val_metrics = self._validate(epoch)

                    if is_main_process():
                        log_metrics(val_metrics, step=epoch, prefix="val/")
                        log_metrics({"lr": self.optimizer.param_groups[0]["lr"]}, step=epoch)

                        # Checkpoint if improved
                        current_metric = val_metrics.get("loss", float("inf"))
                        if current_metric < self.best_metric:
                            self.best_metric = current_metric
                            save_checkpoint(
                                self.checkpoint_dir / f"checkpoint_best.pt",
                                self.model, self.optimizer, self.scheduler,
                                epoch, self.best_metric, self.cfg,
                            )

                if epoch % self.cfg.training.checkpoint_every == 0 and is_main_process():
                    save_checkpoint(
                        self.checkpoint_dir / f"checkpoint_{epoch:04d}.pt",
                        self.model, self.optimizer, self.scheduler,
                        epoch, self.best_metric, self.cfg,
                    )
                    manage_top_k_checkpoints(
                        self.checkpoint_dir, self.cfg.training.keep_top_k
                    )

                self.scheduler.step()

        finally:
            if is_main_process():
                finish_wandb()
            cleanup_distributed()

        logger.info("Training complete. Best metric: %.4f", self.best_metric)

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Dict of averaged training metrics.
        """
        self.model.train()
        tracker = MetricTracker()
        log_every = self.cfg.training.log_every

        pbar = tqdm(self.train_loader, desc=f"Train epoch {epoch}", disable=not is_main_process())
        for batch_idx, (inputs, metadata) in enumerate(pbar):
            inputs = inputs.to(self.device, non_blocking=True)

            # Extract targets from metadata — stub: assumes "label" key exists
            targets = torch.tensor(
                [m.get("label", 0) for m in metadata], dtype=torch.long, device=self.device
            )

            # Forward
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            # Backward
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.cfg.training.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.training.gradient_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Track metrics
            batch_size = inputs.size(0)
            tracker.update("loss", loss.item(), n=batch_size)

            # Log
            self.global_step += 1
            if self.global_step % log_every == 0 and is_main_process():
                batch_metrics = tracker.all_averages()
                log_metrics(batch_metrics, step=self.global_step, prefix="train/")
                pbar.set_postfix(loss=f"{batch_metrics['loss']:.4f}")

        train_metrics = tracker.all_averages()
        if is_main_process():
            log_metrics(train_metrics, step=epoch, prefix="train_epoch/")
        return train_metrics

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict[str, float]:
        """Run validation.

        Args:
            epoch: Current epoch number.

        Returns:
            Dict of averaged validation metrics.
        """
        self.model.eval()
        tracker = MetricTracker()

        pbar = tqdm(self.val_loader, desc=f"Val epoch {epoch}", disable=not is_main_process())
        for inputs, metadata in pbar:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = torch.tensor(
                [m.get("label", 0) for m in metadata], dtype=torch.long, device=self.device
            )

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            tracker.update("loss", loss.item(), n=inputs.size(0))

        val_metrics = tracker.all_averages()
        logger.info("Epoch %d val metrics: %s", epoch, val_metrics)
        return val_metrics
