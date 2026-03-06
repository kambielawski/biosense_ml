"""Recurrent State-Space Model (RSSM) for action-conditioned latent dynamics.

Architecture follows DreamerV3 (Hafner et al., 2023) adapted for BIOSENSE:
  - Continuous Gaussian latents (diagonal) instead of categorical
  - GRU deterministic backbone
  - Operates directly on flattened AE bottleneck vectors (no projector)
  - Residual prediction: obs_predictor predicts δ_t = x_t - x_{t-1}

RSSM equations:
  Deterministic:  h_t = GRU(h_{t-1}, concat(z_{t-1}, a_{t-1}))
  Prior:          p(z_t | h_t) = N(mu_prior(h_t), sigma_prior(h_t))
  Posterior:      q(z_t | h_t, δ_t) = N(mu_post(h_t, δ_t), sigma_post(h_t, δ_t))
  Observation:    δ̂_t = obs_predictor(h_t, z_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class MLP(nn.Module):
    """Simple feedforward network with ReLU activations."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(prev, hidden_dim), nn.ReLU(inplace=True)])
            prev = hidden_dim
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianHead(nn.Module):
    """Predicts mean and log-std of a diagonal Gaussian distribution.

    sigma is computed as softplus(raw) + min_std to ensure positivity and
    prevent collapse.
    """

    def __init__(self, in_dim: int, hidden_dim: int, z_dim: int, min_std: float = 0.1) -> None:
        super().__init__()
        self.min_std = min_std
        self.fc_hidden = nn.Linear(in_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_raw_std = nn.Linear(hidden_dim, z_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, sigma) each of shape (..., z_dim)."""
        h = F.relu(self.fc_hidden(x))
        mu = self.fc_mu(h)
        sigma = F.softplus(self.fc_raw_std(h)) + self.min_std
        return mu, sigma


class RSSM(nn.Module):
    """Recurrent State-Space Model with continuous Gaussian latents.

    Operates directly on flattened AE bottleneck vectors (ae_latent_dim).
    Uses residual prediction: predicts frame-to-frame deltas δ_t = x_t - x_{t-1}.

    Components:
      - gru: GRU cell for deterministic state h_t
      - prior_head: GaussianHead predicting p(z_t | h_t)
      - posterior_head: GaussianHead predicting q(z_t | h_t, δ_t)
      - obs_predictor: MLP predicting δ_t from (h_t, z_t)
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        ae_latent_dim: int = cfg.ae_latent_dim
        h_dim: int = getattr(cfg, "h_dim", 512)
        z_dim: int = getattr(cfg, "z_dim", 256)
        action_dim: int = getattr(cfg, "action_dim", 3)
        min_std: float = getattr(cfg, "min_std", 0.1)

        self.ae_latent_dim = ae_latent_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.action_dim = action_dim

        # GRU: input is concat(z_{t-1}, a_{t-1})
        self.gru = nn.GRUCell(z_dim + action_dim, h_dim)

        # Prior: p(z_t | h_t)
        self.prior_head = GaussianHead(h_dim, h_dim, z_dim, min_std=min_std)

        # Posterior: q(z_t | h_t, δ_t) where δ_t is the frame-to-frame delta
        self.posterior_head = GaussianHead(h_dim + ae_latent_dim, h_dim, z_dim, min_std=min_std)

        # Observation predictor: predict δ_t from (h_t, z_t)
        self.obs_predictor = MLP(h_dim + z_dim, h_dim, ae_latent_dim, num_layers=1)

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialized (h_0, z_0)."""
        h = torch.zeros(batch_size, self.h_dim, device=device)
        z = torch.zeros(batch_size, self.z_dim, device=device)
        return h, z

    def step_deterministic(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        a_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the deterministic state by one step.

        h_t = GRU(h_{t-1}, concat(z_{t-1}, a_{t-1}))
        """
        gru_input = torch.cat([z_prev, a_prev], dim=-1)
        h_t = self.gru(gru_input, h_prev)
        return h_t

    def compute_prior(self, h_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute prior distribution p(z_t | h_t)."""
        return self.prior_head(h_t)

    def compute_posterior(
        self,
        h_t: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior distribution q(z_t | h_t, δ_t).

        Args:
            h_t: Deterministic state, shape (B, h_dim).
            delta_t: Frame-to-frame delta, shape (B, ae_latent_dim).
        """
        inp = torch.cat([h_t, delta_t], dim=-1)
        return self.posterior_head(inp)

    def predict_observation(self, h_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """Predict frame-to-frame delta from model state.

        Returns:
            Predicted delta δ̂_t, shape (B, ae_latent_dim).
        """
        inp = torch.cat([h_t, z_t], dim=-1)
        return self.obs_predictor(inp)

    @staticmethod
    def sample_gaussian(
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
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
        """Compute KL(q || p) for diagonal Gaussians, per-dimension."""
        var_q = sigma_q ** 2
        var_p = sigma_p ** 2
        kl = 0.5 * (
            torch.log(var_p / var_q)
            + var_q / var_p
            + (mu_q - mu_p) ** 2 / var_p
            - 1.0
        )
        return kl

    def forward_single_step(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        a_prev: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """One step of teacher-forced RSSM with residual prediction.

        Args:
            h_prev: Previous deterministic state (B, h_dim).
            z_prev: Previous stochastic state (B, z_dim).
            a_prev: Previous action (B, action_dim).
            delta_t: Current frame-to-frame delta (B, ae_latent_dim).

        Returns:
            Dict with h_t, prior/posterior params, z_t, obs_pred (predicted delta).
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
        """Teacher-forced forward pass over a sequence with residual prediction.

        Computes deltas δ_t = x_t - x_{t-1} internally. The posterior receives
        deltas, and the reconstruction target is deltas.

        Args:
            ae_latents: Flattened AE bottlenecks, shape (B, T, ae_latent_dim).
            actions: Action vectors, shape (B, T, action_dim).

        Returns:
            Dict with stacked tensors over time dimension (B, T, ...):
                prior_mu, prior_sigma, post_mu, post_sigma: (B, T, z_dim)
                z_t: Sampled posterior states (B, T, z_dim)
                obs_pred: Predicted deltas (B, T, ae_latent_dim)
                obs_target: Actual deltas (B, T, ae_latent_dim)
                h_final: Final deterministic state (B, h_dim)
        """
        B, T, _ = ae_latents.shape
        device = ae_latents.device

        # Compute frame-to-frame deltas
        # δ_0 = zeros, δ_t = x_t - x_{t-1} for t > 0
        zeros = torch.zeros(B, 1, self.ae_latent_dim, device=device)
        deltas = torch.cat([zeros, ae_latents[:, 1:] - ae_latents[:, :-1]], dim=1)

        # Initialize states
        h_t, z_t = self.initial_state(B, device)

        # Collect outputs
        all_prior_mu, all_prior_sigma = [], []
        all_post_mu, all_post_sigma = [], []
        all_z, all_obs_pred = [], []

        for t in range(T):
            # Action from previous timestep (zero for t=0)
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
        """Open-loop imagination with residual accumulation.

        Rolls out using prior only (no observations). Predicts deltas and
        accumulates them to produce absolute AE latents for decoding.

        Args:
            h_0: Initial deterministic state (B, h_dim).
            z_0: Initial stochastic state (B, z_dim).
            actions: Action sequence (B, T, action_dim).
            x_last: Last known absolute AE latent (B, ae_latent_dim).

        Returns:
            Dict with:
                prior_mu, prior_sigma: (B, T, z_dim)
                z_t: Sampled from prior (B, T, z_dim)
                obs_pred: Accumulated absolute AE latents (B, T, ae_latent_dim)
                delta_pred: Raw predicted deltas (B, T, ae_latent_dim)
        """
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

            # Accumulate: x̂_t = x_prev + δ̂_t
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
