"""Convolutional RSSM with ConvGRU backbone for spatially-structured latent dynamics.

Preserves the 16x16 spatial structure of the AE bottleneck throughout.
All components use 3x3 convolutions instead of MLPs.

Architecture:
  - ConvGRU deterministic backbone: h_t (B, C_h, 16, 16)
  - Per-pixel Gaussian stochastic variable: z_t (B, C_z, 16, 16)
  - Action embedding: Linear(3, C_a) broadcast to (B, C_a, 16, 16)
  - Prior: Conv3x3 + Conv1x1 on h_t -> per-pixel Gaussian
  - Posterior: Conv3x3 + Conv1x1 on (h_t, delta_t) -> per-pixel Gaussian
  - Obs predictor: Conv3x3 x2 on (h_t, z_t) -> predicted delta (B, 32, 16, 16)
  - Residual prediction: predicts delta_t = x_t - x_{t-1} in spatial bottleneck space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class ConvGRUCell(nn.Module):
    """Convolutional GRU cell operating on spatial feature maps.

    Standard GRU equations with 3x3 convolutions replacing linear transforms.
    """

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels

        # Reset gate
        self.conv_reset = nn.Conv2d(
            input_channels + hidden_channels, hidden_channels,
            kernel_size=3, padding=1,
        )
        # Update gate
        self.conv_update = nn.Conv2d(
            input_channels + hidden_channels, hidden_channels,
            kernel_size=3, padding=1,
        )
        # Candidate hidden state
        self.conv_candidate = nn.Conv2d(
            input_channels + hidden_channels, hidden_channels,
            kernel_size=3, padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> torch.Tensor:
        """One step of ConvGRU.

        Args:
            x: Input features (B, C_in, H, W).
            h_prev: Previous hidden state (B, C_h, H, W).

        Returns:
            h_t: Updated hidden state (B, C_h, H, W).
        """
        combined = torch.cat([x, h_prev], dim=1)
        r = torch.sigmoid(self.conv_reset(combined))
        u = torch.sigmoid(self.conv_update(combined))

        combined_r = torch.cat([x, r * h_prev], dim=1)
        h_cand = torch.tanh(self.conv_candidate(combined_r))

        h_t = (1 - u) * h_prev + u * h_cand
        return h_t


class ConvGaussianHead(nn.Module):
    """Convolutional Gaussian head: Conv3x3 + ReLU + Conv1x1 -> (mu, sigma)."""

    def __init__(self, in_channels: int, hidden_channels: int, z_channels: int, min_std: float = 0.1) -> None:
        super().__init__()
        self.min_std = min_std
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, 2 * z_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, sigma) each of shape (B, C_z, H, W)."""
        h = F.relu(self.conv1(x))
        out = self.conv2(h)
        mu, raw_std = out.chunk(2, dim=1)
        sigma = F.softplus(raw_std) + self.min_std
        return mu, sigma


class ConvRSSM(nn.Module):
    """Convolutional Recurrent State-Space Model.

    Preserves 16x16 spatial structure. Uses ConvGRU for deterministic state
    and per-pixel Gaussian stochastic variables.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        # Dimensions
        self.ae_latent_channels: int = getattr(cfg, "ae_latent_channels", 32)
        self.spatial_size: int = getattr(cfg, "spatial_size", 16)
        self.h_channels: int = getattr(cfg, "h_channels", 64)
        self.z_channels: int = getattr(cfg, "z_channels", 16)
        self.action_channels: int = getattr(cfg, "action_channels", 16)
        self.action_dim: int = getattr(cfg, "action_dim", 3)
        min_std: float = getattr(cfg, "min_std", 0.1)

        # For compatibility with flat-vector HDF5 data
        self.ae_latent_dim = self.ae_latent_channels * self.spatial_size * self.spatial_size

        # Action embedding: Linear(3, C_a) then broadcast to (C_a, 16, 16)
        self.action_embed = nn.Linear(self.action_dim, self.action_channels)

        # ConvGRU: input is concat(z_{t-1}, action_embed)
        gru_input_channels = self.z_channels + self.action_channels
        self.conv_gru = ConvGRUCell(gru_input_channels, self.h_channels)

        # Prior: p(z_t | h_t)
        self.prior_head = ConvGaussianHead(
            self.h_channels, self.h_channels, self.z_channels, min_std=min_std,
        )

        # Posterior: q(z_t | h_t, delta_t)
        self.posterior_head = ConvGaussianHead(
            self.h_channels + self.ae_latent_channels, self.h_channels, self.z_channels, min_std=min_std,
        )

        # Obs predictor: predict delta_t from (h_t, z_t)
        obs_in = self.h_channels + self.z_channels
        self.obs_predictor = nn.Sequential(
            nn.Conv2d(obs_in, self.h_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.h_channels, self.ae_latent_channels, kernel_size=3, padding=1),
        )

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialized (h_0, z_0) as spatial tensors."""
        S = self.spatial_size
        h = torch.zeros(batch_size, self.h_channels, S, S, device=device)
        z = torch.zeros(batch_size, self.z_channels, S, S, device=device)
        return h, z

    def embed_action(self, action: torch.Tensor) -> torch.Tensor:
        """Embed action vector and broadcast to spatial feature map.

        Args:
            action: (B, action_dim)

        Returns:
            (B, C_a, spatial_size, spatial_size)
        """
        embedded = self.action_embed(action)  # (B, C_a)
        S = self.spatial_size
        return embedded.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, S, S)

    def step_deterministic(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        a_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Advance deterministic state by one step.

        Args:
            h_prev: (B, C_h, H, W)
            z_prev: (B, C_z, H, W)
            a_prev: (B, action_dim) — raw action vector

        Returns:
            h_t: (B, C_h, H, W)
        """
        action_embed = self.embed_action(a_prev)  # (B, C_a, H, W)
        gru_input = torch.cat([z_prev, action_embed], dim=1)  # (B, C_z + C_a, H, W)
        return self.conv_gru(gru_input, h_prev)

    def compute_prior(self, h_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute prior p(z_t | h_t)."""
        return self.prior_head(h_t)

    def compute_posterior(
        self,
        h_t: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior q(z_t | h_t, delta_t).

        Args:
            h_t: (B, C_h, H, W)
            delta_t: (B, C_ae, H, W) — frame-to-frame delta in spatial bottleneck space
        """
        inp = torch.cat([h_t, delta_t], dim=1)
        return self.posterior_head(inp)

    def predict_observation(self, h_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """Predict frame-to-frame delta from model state.

        Returns:
            Predicted delta (B, C_ae, H, W).
        """
        inp = torch.cat([h_t, z_t], dim=1)
        return self.obs_predictor(inp)

    @staticmethod
    def sample_gaussian(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Sample from diagonal Gaussian using reparameterization trick."""
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    @staticmethod
    def kl_divergence(
        mu_q: torch.Tensor,
        sigma_q: torch.Tensor,
        mu_p: torch.Tensor,
        sigma_p: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL(q || p) for diagonal Gaussians, per element."""
        var_q = sigma_q ** 2
        var_p = sigma_p ** 2
        return 0.5 * (
            torch.log(var_p / var_q)
            + var_q / var_p
            + (mu_q - mu_p) ** 2 / var_p
            - 1.0
        )

    def forward_single_step(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        a_prev: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """One step of teacher-forced ConvRSSM.

        Args:
            h_prev: (B, C_h, H, W)
            z_prev: (B, C_z, H, W)
            a_prev: (B, action_dim) — raw action vector
            delta_t: (B, C_ae, H, W) — current frame-to-frame delta

        Returns:
            Dict with h_t, prior/posterior params, z_t, obs_pred.
        """
        h_t = self.step_deterministic(h_prev, z_prev, a_prev)

        prior_mu, prior_sigma = self.compute_prior(h_t)
        post_mu, post_sigma = self.compute_posterior(h_t, delta_t)

        z_t = self.sample_gaussian(post_mu, post_sigma)
        obs_pred = self.predict_observation(h_t, z_t)

        return {
            "h_t": h_t,
            "prior_mu": prior_mu,
            "prior_sigma": prior_sigma,
            "post_mu": post_mu,
            "post_sigma": post_sigma,
            "z_t": z_t,
            "obs_pred": obs_pred,
        }

    def forward(
        self,
        ae_latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Teacher-forced forward pass over a sequence.

        Accepts flat or spatial AE latents. If flat (B, T, D), reshapes to spatial.
        Computes deltas internally. Posterior receives deltas, reconstruction target is deltas.

        Args:
            ae_latents: AE bottlenecks, shape (B, T, D) flat or (B, T, C, H, W) spatial.
            actions: Action vectors, shape (B, T, action_dim).

        Returns:
            Dict with stacked tensors (B, T, C, H, W) for spatial outputs:
                prior_mu, prior_sigma, post_mu, post_sigma: (B, T, C_z, H, W)
                z_t: Sampled posterior states (B, T, C_z, H, W)
                obs_pred: Predicted deltas (B, T, C_ae, H, W)
                obs_target: Actual deltas (B, T, C_ae, H, W)
                h_final: Final deterministic state (B, C_h, H, W)
        """
        # Reshape flat latents to spatial if needed
        if ae_latents.dim() == 3:
            B, T, D = ae_latents.shape
            S = self.spatial_size
            C = self.ae_latent_channels
            ae_latents = ae_latents.view(B, T, C, S, S)

        B, T, C, H, W = ae_latents.shape
        device = ae_latents.device

        # Compute frame-to-frame deltas: delta_0 = zeros, delta_t = x_t - x_{t-1}
        zeros = torch.zeros(B, 1, C, H, W, device=device)
        deltas = torch.cat([zeros, ae_latents[:, 1:] - ae_latents[:, :-1]], dim=1)

        # Initialize states
        h_t, z_t = self.initial_state(B, device)

        # Collect outputs
        all_prior_mu, all_prior_sigma = [], []
        all_post_mu, all_post_sigma = [], []
        all_z, all_obs_pred = [], []

        for t in range(T):
            if t == 0:
                a_prev = torch.zeros(B, self.action_dim, device=device)
            else:
                a_prev = actions[:, t - 1]

            delta_t = deltas[:, t]
            step = self.forward_single_step(h_t, z_t, a_prev, delta_t)

            h_t = step["h_t"]
            z_t = step["z_t"]

            all_prior_mu.append(step["prior_mu"])
            all_prior_sigma.append(step["prior_sigma"])
            all_post_mu.append(step["post_mu"])
            all_post_sigma.append(step["post_sigma"])
            all_z.append(step["z_t"])
            all_obs_pred.append(step["obs_pred"])

        return {
            "prior_mu": torch.stack(all_prior_mu, dim=1),
            "prior_sigma": torch.stack(all_prior_sigma, dim=1),
            "post_mu": torch.stack(all_post_mu, dim=1),
            "post_sigma": torch.stack(all_post_sigma, dim=1),
            "z_t": torch.stack(all_z, dim=1),
            "obs_pred": torch.stack(all_obs_pred, dim=1),
            "obs_target": deltas,
            "h_final": h_t,
        }

    def imagine(
        self,
        h_0: torch.Tensor,
        z_0: torch.Tensor,
        actions: torch.Tensor,
        x_last: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Open-loop imagination with residual accumulation in spatial space.

        Args:
            h_0: Initial deterministic state (B, C_h, H, W).
            z_0: Initial stochastic state (B, C_z, H, W).
            actions: Action sequence (B, T, action_dim).
            x_last: Last known absolute AE latent (B, C_ae, H, W) or (B, D) flat.

        Returns:
            Dict with spatial tensors:
                prior_mu, prior_sigma: (B, T, C_z, H, W)
                z_t: (B, T, C_z, H, W)
                obs_pred: Accumulated absolute AE latents (B, T, C_ae, H, W)
                delta_pred: Raw predicted deltas (B, T, C_ae, H, W)
        """
        # Reshape flat x_last to spatial if needed
        if x_last.dim() == 2:
            S = self.spatial_size
            C = self.ae_latent_channels
            x_last = x_last.view(-1, C, S, S)

        B, T, _ = actions.shape
        h_t, z_t = h_0, z_0
        x_prev = x_last

        all_prior_mu, all_prior_sigma = [], []
        all_z, all_obs_pred, all_delta_pred = [], [], []

        for t in range(T):
            a_prev = actions[:, t]
            h_t = self.step_deterministic(h_t, z_t, a_prev)

            prior_mu, prior_sigma = self.compute_prior(h_t)
            z_t = self.sample_gaussian(prior_mu, prior_sigma)
            delta_pred = self.predict_observation(h_t, z_t)

            x_t = x_prev + delta_pred
            x_prev = x_t

            all_prior_mu.append(prior_mu)
            all_prior_sigma.append(prior_sigma)
            all_z.append(z_t)
            all_obs_pred.append(x_t)
            all_delta_pred.append(delta_pred)

        return {
            "prior_mu": torch.stack(all_prior_mu, dim=1),
            "prior_sigma": torch.stack(all_prior_sigma, dim=1),
            "z_t": torch.stack(all_z, dim=1),
            "obs_pred": torch.stack(all_obs_pred, dim=1),
            "delta_pred": torch.stack(all_delta_pred, dim=1),
        }
