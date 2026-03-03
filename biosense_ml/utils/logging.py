"""Weights & Biases integration helpers."""

import logging
from datetime import datetime, timezone

import wandb
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def init_wandb(cfg: DictConfig) -> wandb.sdk.wandb_run.Run:
    """Initialize a wandb run with the full Hydra config.

    Args:
        cfg: Full Hydra config.

    Returns:
        The wandb Run object.
    """
    run = wandb.init(
        project=cfg.project_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=f"{cfg.model.name}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}",
        tags=[cfg.model.name, cfg.data.preprocessing.mode],
    )
    logger.info("Initialized wandb run: %s", run.name)
    return run


def log_metrics(metrics: dict[str, float], step: int, prefix: str = "") -> None:
    """Log a dict of metrics to wandb with optional prefix.

    Args:
        metrics: Dict of metric name -> value.
        step: Global step number.
        prefix: Optional prefix like "train/" or "val/".
    """
    wandb.log({f"{prefix}{k}": v for k, v in metrics.items()}, step=step)


def log_config_artifact(cfg: DictConfig) -> None:
    """Save the full config as a wandb artifact for reproducibility."""
    artifact = wandb.Artifact("config", type="config")
    with artifact.new_file("config.yaml") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    wandb.log_artifact(artifact)


def finish_wandb() -> None:
    """Finish the current wandb run."""
    wandb.finish()
    logger.info("Finished wandb run")
