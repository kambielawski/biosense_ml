"""GRU model for trajectory prediction with dt-as-input.

A compact GRU that processes the 10-dim feature vector at each timestep
and predicts the next (x, y) centroid position. Handles irregular time
sampling by including dt as an input feature.

Feature vector per timestep (10-dim):
    centroid_xy (2) + velocity_xy (2) + dt (1) + actions (5)

Architecture:
    Linear(input_dim, hidden_dim) → GRU(hidden_dim, hidden_dim) → Linear(hidden_dim, 2)
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig


class TrajectoryGRU(nn.Module):
    """GRU for next-step trajectory prediction with action conditioning.

    Maintains a hidden state that captures the organoid's motion history.
    At each step, predicts the next centroid position.
    """

    def __init__(self, model_cfg: DictConfig) -> None:
        super().__init__()
        self.input_dim = model_cfg.input_dim  # 10
        self.hidden_dim = model_cfg.hidden_dim  # 32–64
        self.output_dim = model_cfg.output_dim  # 2 (x, y)
        self.num_layers = model_cfg.get("num_layers", 1)
        self.dropout = model_cfg.get("dropout", 0.0)

        # Input projection
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)

        # GRU backbone
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )

        # Output head: predict next (x, y)
        self.output_head = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass over a sequence.

        Args:
            x: (B, T, input_dim) — sequence of feature vectors.
            hidden: (num_layers, B, hidden_dim) — initial hidden state.
                If None, initializes to zeros.

        Returns:
            pred_xy: (B, T, 2) — predicted next (x, y) at each step.
            hidden: (num_layers, B, hidden_dim) — final hidden state.
        """
        B, T, _ = x.shape

        if hidden is None:
            hidden = torch.zeros(
                self.num_layers, B, self.hidden_dim,
                device=x.device, dtype=x.dtype,
            )

        projected = self.input_proj(x)  # (B, T, hidden_dim)
        gru_out, hidden = self.gru(projected, hidden)  # (B, T, hidden_dim)
        pred_xy = self.output_head(gru_out)  # (B, T, 2)

        return pred_xy, hidden

    def step(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step forward (for autoregressive rollout).

        Args:
            x: (B, input_dim) — single timestep feature vector.
            hidden: (num_layers, B, hidden_dim) — current hidden state.

        Returns:
            pred_xy: (B, 2) — predicted next (x, y).
            hidden: (num_layers, B, hidden_dim) — updated hidden state.
        """
        pred_xy, hidden = self.forward(x.unsqueeze(1), hidden)
        return pred_xy.squeeze(1), hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create zero-initialized hidden state."""
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_dim,
            device=device,
        )

    def rollout(
        self,
        context: torch.Tensor,
        future_actions: torch.Tensor,
        future_dt: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Autoregressive rollout for evaluation.

        Burns in on the context sequence, then predicts autoregressively
        using ground-truth future actions and dt.

        Args:
            context: (B, T_ctx, input_dim) — burn-in context (ground truth).
            future_actions: (B, H, 5) — known future action vectors.
            future_dt: (B, H) — known future dt values.
            hidden: Optional initial hidden state.

        Returns:
            predictions: (B, H, 2) — predicted (x, y) for each future step.
        """
        B = context.shape[0]
        H = future_actions.shape[1]
        device = context.device

        if hidden is None:
            hidden = self.init_hidden(B, device)

        # Burn-in: process context to build up hidden state
        # The model predicts next-step at each position, but we only need
        # the hidden state and the last prediction for rollout
        _, hidden = self.forward(context, hidden)

        # Last known position (from context)
        prev_xy = context[:, -1, :2]  # (B, 2)

        predictions = []
        for t in range(H):
            dt_t = future_dt[:, t : t + 1]  # (B, 1)
            actions_t = future_actions[:, t]  # (B, 5)

            # Compute velocity from last step
            # On the first rollout step, use the context's last velocity
            if t == 0:
                prev_vel = context[:, -1, 2:4]  # (B, 2)
            else:
                safe_dt = dt_t.clamp(min=1e-6)
                prev_vel = (prev_xy - prev_prev_xy) / safe_dt  # (B, 2)

            # Assemble feature vector: [centroid_xy, velocity_xy, dt, actions]
            feat = torch.cat([prev_xy, prev_vel, dt_t, actions_t], dim=-1)  # (B, 10)

            # Single-step prediction
            pred_xy, hidden = self.step(feat, hidden)
            pred_xy = torch.clamp(pred_xy, 0.0, 1.0)
            predictions.append(pred_xy)

            prev_prev_xy = prev_xy
            prev_xy = pred_xy

        return torch.stack(predictions, dim=1)  # (B, H, 2)
