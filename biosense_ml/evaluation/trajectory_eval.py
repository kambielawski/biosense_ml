"""Evaluation harness for trajectory prediction models.

Implements the evaluation protocol from the model selection doc:
- Autoregressive rollout with burn-in
- ADE, FDE, HDE metrics
- Velocity error
- Stimulus-stratified evaluation (future extension)

All metrics operate in [0,1] normalized coordinate space.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class RolloutMetrics:
    """Metrics from a single sequence rollout."""
    ade: float = 0.0
    fde: float = 0.0
    hde: dict[float, float] = field(default_factory=dict)
    velocity_error: float = 0.0
    rollout_length: int = 0


def evaluate_rollout(
    model: torch.nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    dt: torch.Tensor,
    burn_in_frac: float = 0.2,
    time_horizons: list[float] | None = None,
    model_type: str = "gru",
    context_len: int = 10,
) -> RolloutMetrics:
    """Evaluate a single sequence with autoregressive rollout.

    Args:
        model: Trajectory prediction model (TrajectoryGRU or TrajectoryMLP).
        features: (T, 10) full sequence features.
        targets: (T, 2) full sequence target positions.
        dt: (T,) inter-frame time deltas.
        burn_in_frac: Fraction of sequence for burn-in.
        time_horizons: Wall-clock time horizons for HDE (seconds).
        model_type: "gru" or "mlp".
        context_len: Context window size for MLP.

    Returns:
        RolloutMetrics with computed metrics.
    """
    if time_horizons is None:
        time_horizons = [10.0, 30.0, 60.0, 120.0, 300.0]

    T = features.shape[0]
    B_idx = max(int(T * burn_in_frac), context_len if model_type == "mlp" else 2)
    H = T - B_idx - 1  # number of rollout steps

    if H < 1:
        return RolloutMetrics()

    device = next(model.parameters()).device
    features = features.unsqueeze(0).to(device)  # (1, T, 10)
    targets = targets.to(device)
    dt = dt.to(device)

    model.eval()
    with torch.no_grad():
        context = features[:, :B_idx]  # (1, B, 10)
        future_actions = features[:, B_idx : B_idx + H, 5:10]  # (1, H, 5)
        future_dt = dt[B_idx + 1 : B_idx + 1 + H].unsqueeze(0)  # (1, H)

        if model_type == "gru":
            predictions = model.rollout(context, future_actions, future_dt)
        else:
            # MLP: use last context_len frames as initial window
            mlp_context = features[:, B_idx - context_len : B_idx]
            predictions = model.rollout(mlp_context, future_actions, future_dt)

    # predictions: (1, H, 2)
    preds = predictions.squeeze(0).cpu().numpy()  # (H, 2)
    gt = targets[B_idx + 1 : B_idx + 1 + H].cpu().numpy()  # (H, 2)

    actual_H = min(preds.shape[0], gt.shape[0])
    if actual_H < 1:
        return RolloutMetrics()
    preds = preds[:actual_H]
    gt = gt[:actual_H]

    # ADE: average L2 displacement error
    displacements = np.linalg.norm(preds - gt, axis=1)
    ade = float(displacements.mean())

    # FDE: final displacement error
    fde = float(displacements[-1])

    # HDE: horizon-stratified displacement error
    cum_time = np.cumsum(dt[B_idx + 1 : B_idx + 1 + actual_H].cpu().numpy())
    hde = {}
    for tau in time_horizons:
        idx = np.searchsorted(cum_time, tau)
        if idx < actual_H:
            hde[tau] = float(displacements[idx])

    # Velocity error
    if actual_H > 1:
        pred_vel = np.diff(preds, axis=0)
        gt_vel = np.diff(gt, axis=0)
        vel_err = float(np.linalg.norm(pred_vel - gt_vel, axis=1).mean())
    else:
        vel_err = 0.0

    return RolloutMetrics(
        ade=ade,
        fde=fde,
        hde=hde,
        velocity_error=vel_err,
        rollout_length=actual_H,
    )


def evaluate_dataset(
    model: torch.nn.Module,
    sequences: list[dict[str, torch.Tensor]],
    burn_in_frac: float = 0.2,
    time_horizons: list[float] | None = None,
    model_type: str = "gru",
    context_len: int = 10,
) -> dict:
    """Evaluate model across all sequences in a dataset split.

    Args:
        model: Trained trajectory prediction model.
        sequences: List of sequence dicts from TrajectorySequenceDataset.
        burn_in_frac: Fraction for burn-in.
        time_horizons: Wall-clock horizons for HDE.
        model_type: "gru" or "mlp".
        context_len: Context window for MLP.

    Returns:
        Dict with aggregated metrics (median, IQR, per-sequence).
    """
    if time_horizons is None:
        time_horizons = [10.0, 30.0, 60.0, 120.0, 300.0]

    all_metrics: list[RolloutMetrics] = []

    for seq in sequences:
        metrics = evaluate_rollout(
            model=model,
            features=seq["features"],
            targets=seq["targets"],
            dt=seq["dt"],
            burn_in_frac=burn_in_frac,
            time_horizons=time_horizons,
            model_type=model_type,
            context_len=context_len,
        )
        if metrics.rollout_length > 0:
            all_metrics.append(metrics)

    if not all_metrics:
        logger.warning("No valid rollouts produced")
        return {"num_sequences": 0}

    # Filter out NaN/Inf metrics (can happen if rollout diverges)
    valid_metrics = [m for m in all_metrics
                     if np.isfinite(m.ade) and np.isfinite(m.fde)]
    if len(valid_metrics) < len(all_metrics):
        logger.warning(
            "Filtered %d/%d sequences with NaN/Inf rollout metrics",
            len(all_metrics) - len(valid_metrics), len(all_metrics),
        )
    if not valid_metrics:
        logger.warning("All rollouts produced NaN/Inf")
        return {"num_sequences": 0}

    all_metrics = valid_metrics
    ades = np.array([m.ade for m in all_metrics])
    fdes = np.array([m.fde for m in all_metrics])
    vel_errs = np.array([m.velocity_error for m in all_metrics])

    results = {
        "num_sequences": len(all_metrics),
        "ade_median": float(np.median(ades)),
        "ade_mean": float(np.mean(ades)),
        "ade_iqr": (float(np.percentile(ades, 25)), float(np.percentile(ades, 75))),
        "fde_median": float(np.median(fdes)),
        "fde_mean": float(np.mean(fdes)),
        "fde_iqr": (float(np.percentile(fdes, 25)), float(np.percentile(fdes, 75))),
        "velocity_error_median": float(np.median(vel_errs)),
        "velocity_error_mean": float(np.mean(vel_errs)),
    }

    # HDE at each horizon
    for tau in time_horizons:
        horizon_vals = [m.hde[tau] for m in all_metrics if tau in m.hde]
        if horizon_vals:
            arr = np.array(horizon_vals)
            results[f"hde_{int(tau)}s_median"] = float(np.median(arr))
            results[f"hde_{int(tau)}s_mean"] = float(np.mean(arr))
            results[f"hde_{int(tau)}s_count"] = len(horizon_vals)

    # Length-weighted ADE
    lengths = np.array([m.rollout_length for m in all_metrics])
    results["ade_weighted"] = float(np.sum(ades * lengths) / np.sum(lengths))

    logger.info(
        "Evaluation: %d sequences | ADE=%.6f (median) | FDE=%.6f (median) | VelErr=%.6f",
        len(all_metrics), results["ade_median"], results["fde_median"],
        results["velocity_error_median"],
    )

    return results
