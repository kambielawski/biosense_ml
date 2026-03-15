"""Model registry and factory."""

import torch.nn as nn
from omegaconf import DictConfig

from biosense_ml.models.autoencoder import ConvAutoencoder
from biosense_ml.models.baseline import BaselineModel
from biosense_ml.models.conv_rssm import ConvRSSM
from biosense_ml.models.rssm import RSSM
from biosense_ml.models.trajectory_gru import TrajectoryGRU
from biosense_ml.models.trajectory_mlp import TrajectoryMLP

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "autoencoder": ConvAutoencoder,
    "baseline": BaselineModel,
    "conv_rssm": ConvRSSM,
    "rssm": RSSM,
    "trajectory_gru": TrajectoryGRU,
    "trajectory_mlp": TrajectoryMLP,
}


def build_model(cfg: DictConfig) -> nn.Module:
    """Build a model from config.

    Args:
        cfg: Full Hydra config (uses cfg.model).

    Returns:
        Instantiated model.

    Raises:
        KeyError: If model name is not in the registry.
    """
    name = cfg.model.name
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name](cfg.model)
