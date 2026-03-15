"""MLP baseline for trajectory prediction.

Takes the last K timesteps' feature vectors and predicts the next (x, y) position.
This establishes whether sequential memory (GRU) actually helps beyond a simple
feedforward approach.

Feature vector per timestep (10-dim):
    centroid_xy (2) + velocity_xy (2) + dt (1) + actions (5)

Input: (B, K, 10) — last K timesteps' features
Output: (B, 2) — predicted next (x, y) position
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig


class TrajectoryMLP(nn.Module):
    """MLP baseline for next-step trajectory prediction.

    Flattens the last K timesteps into a single vector and passes through
    a small MLP to predict the next centroid position.
    """

    def __init__(self, model_cfg: DictConfig) -> None:
        super().__init__()
        self.context_len = model_cfg.context_len  # K
        self.input_dim = model_cfg.input_dim  # 10
        self.hidden_dim = model_cfg.hidden_dim
        self.output_dim = model_cfg.output_dim  # 2 (x, y)
        self.num_layers = model_cfg.num_layers
        self.dropout = model_cfg.get("dropout", 0.0)

        flat_dim = self.context_len * self.input_dim

        layers = [
            nn.Linear(flat_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))

        for _ in range(self.num_layers - 1):
            layers.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
            ])
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))

        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next position from context window.

        Args:
            x: (B, K, input_dim) — last K timesteps' feature vectors.

        Returns:
            (B, 2) — predicted next (x, y) position.
        """
        B = x.shape[0]
        return self.mlp(x.reshape(B, -1))

    def rollout(
        self,
        context: torch.Tensor,
        future_actions: torch.Tensor,
        future_dt: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressive rollout for evaluation.

        Args:
            context: (B, K, input_dim) — initial context window (ground truth).
            future_actions: (B, H, 5) — known future action vectors for H steps.
            future_dt: (B, H) — known future dt values for H steps.

        Returns:
            predictions: (B, H, 2) — predicted (x, y) positions for H steps.
        """
        B, K, D = context.shape
        H = future_actions.shape[1]

        # Working copy of the sliding context window
        window = context.clone()  # (B, K, D)
        predictions = []

        for t in range(H):
            # Predict next position
            pred_xy = self.forward(window)  # (B, 2)
            predictions.append(pred_xy)

            # Build next feature vector from prediction
            # velocity = (pred_xy - prev_xy) / dt
            prev_xy = window[:, -1, :2]  # (B, 2)
            dt_t = future_dt[:, t : t + 1]  # (B, 1)
            safe_dt = dt_t.clamp(min=1e-6)
            velocity = (pred_xy - prev_xy) / safe_dt  # (B, 2)
            actions_t = future_actions[:, t]  # (B, 5)

            # Assemble: [centroid_xy, velocity_xy, dt, actions]
            next_feat = torch.cat([pred_xy, velocity, dt_t, actions_t], dim=-1)  # (B, 10)

            # Slide window: drop oldest, append new
            window = torch.cat([window[:, 1:, :], next_feat.unsqueeze(1)], dim=1)

        return torch.stack(predictions, dim=1)  # (B, H, 2)
