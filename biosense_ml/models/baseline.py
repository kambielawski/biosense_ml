"""Baseline model: simple CNN for images, simple MLP for latent vectors."""

import torch
import torch.nn as nn
from omegaconf import DictConfig


class BaselineModel(nn.Module):
    """A simple, runnable baseline model for end-to-end pipeline verification.

    Supports two input modes:
        - image: A small CNN with global average pooling and a classification head.
        - latent: An MLP that maps latent vectors to class predictions.
    """

    def __init__(self, model_cfg: DictConfig) -> None:
        super().__init__()
        self.input_type = model_cfg.input_type

        if self.input_type == "image":
            self._build_cnn(model_cfg)
        elif self.input_type == "latent":
            self._build_mlp(model_cfg)
        else:
            raise ValueError(f"Unknown input_type: {self.input_type}")

    def _build_cnn(self, cfg: DictConfig) -> None:
        """Build a simple CNN: conv blocks -> global avg pool -> FC head."""
        hidden = cfg.hidden_dim
        in_channels = cfg.input_channels
        num_classes = cfg.num_classes
        dropout = cfg.dropout

        # Build conv blocks: progressively increase channels
        layers = []
        channels = [in_channels, hidden // 2, hidden, hidden * 2, hidden * 2]
        for i in range(min(cfg.num_layers, len(channels) - 1)):
            layers.extend([
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[min(cfg.num_layers, len(channels) - 1)], num_classes),
        )

    def _build_mlp(self, cfg: DictConfig) -> None:
        """Build a simple MLP for latent vector inputs."""
        hidden = cfg.hidden_dim
        latent_dim = cfg.latent_dim
        num_classes = cfg.num_classes
        dropout = cfg.dropout
        num_layers = cfg.num_layers

        layers = [nn.Linear(latent_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(hidden, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor. Shape depends on input_type:
                - image: (B, C, H, W)
                - latent: (B, latent_dim)

        Returns:
            Logits of shape (B, num_classes).
        """
        if self.input_type == "image":
            x = self.features(x)
            x = self.pool(x).flatten(1)
            return self.head(x)
        else:
            return self.mlp(x)
