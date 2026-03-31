"""Gaze-based dynamics model for organoid video prediction.

Conv Encoder + MLP Predictor + Conv Decoder architecture that operates on
32×32 crops centered on the organoid. Predicts the next crop appearance
and the position delta (dy, dx) for gaze tracking.

Input:  (B, K, 3, 32, 32) — last K crops from the gaze pipeline
Output: (B, 3, 32, 32) predicted next crop + (B, 2) predicted (dy, dx)
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig


class GazeDynamics(nn.Module):
    """Conv encoder + MLP predictor + Conv decoder for crop-based prediction."""

    def __init__(self, model_cfg: DictConfig) -> None:
        super().__init__()
        self.context_len = model_cfg.context_len  # K
        self.latent_dim = model_cfg.latent_dim  # 128
        self.hidden_dim = model_cfg.hidden_dim  # 256
        self.dropout = model_cfg.get("dropout", 0.0)

        # --- Conv Encoder: 32×32×3 → latent_dim ---
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.latent_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # --- MLP Predictor: K * latent_dim → latent_dim + 2 ---
        mlp_input_dim = self.context_len * self.latent_dim
        mlp_layers = [
            nn.Linear(mlp_input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if self.dropout > 0:
            mlp_layers.append(nn.Dropout(self.dropout))
        mlp_layers.append(nn.Linear(self.hidden_dim, self.latent_dim + 2))
        self.predictor = nn.Sequential(*mlp_layers)

        # --- Conv Decoder: latent_dim → 32×32×3 ---
        self.decoder_fc = nn.Linear(self.latent_dim, self.latent_dim * 4 * 4)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            # No activation — ImageNet-normalized range
        )

    def encode(self, crops: torch.Tensor) -> torch.Tensor:
        """Encode a batch of crops to latent vectors.

        Args:
            crops: (B, 3, 32, 32) or (B*K, 3, 32, 32).

        Returns:
            (B, latent_dim) or (B*K, latent_dim) latent vectors.
        """
        return self.encoder(crops)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to crops.

        Args:
            z: (B, latent_dim) latent vectors.

        Returns:
            (B, 3, 32, 32) reconstructed crops.
        """
        x = self.decoder_fc(z)
        x = x.view(-1, self.latent_dim, 4, 4)
        return self.decoder_conv(x)

    def forward(self, crops: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next crop and position delta from context window.

        Args:
            crops: (B, K, 3, 32, 32) — last K crops.

        Returns:
            pred_crop: (B, 3, 32, 32) — predicted next crop.
            pred_delta: (B, 2) — predicted (dy, dx) position shift.
        """
        B, K, C, H, W = crops.shape

        # Encode all K crops
        flat_crops = crops.reshape(B * K, C, H, W)
        flat_z = self.encode(flat_crops)  # (B*K, latent_dim)
        z_seq = flat_z.reshape(B, K * self.latent_dim)  # (B, K*latent_dim)

        # Predict next latent + delta
        pred = self.predictor(z_seq)  # (B, latent_dim + 2)
        pred_z = pred[:, : self.latent_dim]  # (B, latent_dim)
        pred_delta = pred[:, self.latent_dim :]  # (B, 2)

        # Decode to crop
        pred_crop = self.decode(pred_z)  # (B, 3, 32, 32)

        return pred_crop, pred_delta

    def rollout(
        self,
        context_crops: torch.Tensor,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive rollout for evaluation.

        Args:
            context_crops: (B, K, 3, 32, 32) — burn-in context.
            horizon: Number of future steps to predict.

        Returns:
            pred_crops: (B, H, 3, 32, 32) — predicted crops.
            pred_deltas: (B, H, 2) — predicted position deltas.
        """
        B, K, C, H, W = context_crops.shape
        window = context_crops.clone()

        all_crops = []
        all_deltas = []

        for _ in range(horizon):
            pred_crop, pred_delta = self.forward(window)
            all_crops.append(pred_crop)
            all_deltas.append(pred_delta)

            # Slide window: drop oldest, append predicted crop
            window = torch.cat([window[:, 1:], pred_crop.unsqueeze(1)], dim=1)

        return torch.stack(all_crops, dim=1), torch.stack(all_deltas, dim=1)
