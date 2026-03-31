"""Evaluation harness for gaze-based dynamics models.

Implements rollout evaluation with:
- Crop L1 (appearance prediction quality)
- ADE, FDE, HDE (position prediction quality via cumulative deltas)
- Copy-last-crop baseline comparison
- Divergence rate detection

All position metrics operate in 512×512 pixel coordinate space.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class GazeRolloutMetrics:
    """Metrics from a single gaze sequence rollout."""

    crop_l1: float = 0.0
    baseline_crop_l1: float = 0.0
    ade: float = 0.0
    fde: float = 0.0
    hde: dict[float, float] = field(default_factory=dict)
    velocity_error: float = 0.0
    rollout_length: int = 0
    diverged: bool = False


def evaluate_rollout(
    model: torch.nn.Module,
    sequence: dict[str, torch.Tensor],
    context_len: int = 10,
    burn_in_frac: float = 0.2,
    time_horizons: list[float] | None = None,
) -> GazeRolloutMetrics:
    """Evaluate a single sequence with autoregressive rollout.

    Args:
        model: GazeDynamics model.
        sequence: Dict from GazeSequenceDataset with crops, centers, deltas,
            has_motion, timestamps.
        context_len: Context window size (K).
        burn_in_frac: Fraction of sequence used for burn-in.
        time_horizons: Wall-clock horizons in seconds for HDE.

    Returns:
        GazeRolloutMetrics with computed metrics.
    """
    if time_horizons is None:
        time_horizons = [10.0, 30.0, 60.0, 120.0, 300.0]

    crops = sequence["crops"]  # (T, 3, 32, 32)
    centers = sequence["centers"]  # (T, 2)
    deltas = sequence["deltas"]  # (T, 2)
    timestamps = sequence["timestamps"]  # (T,)
    T = crops.shape[0]

    burn_in = max(int(T * burn_in_frac), context_len)
    H = T - burn_in - 1  # rollout steps

    if H < 1:
        return GazeRolloutMetrics()

    device = next(model.parameters()).device
    context = crops[burn_in - context_len : burn_in].unsqueeze(0).to(device)  # (1, K, 3, 32, 32)

    model.eval()
    with torch.no_grad():
        pred_crops, pred_deltas = model.rollout(context, horizon=H)

    # Move to numpy
    pred_crops_np = pred_crops.squeeze(0).cpu().numpy()  # (H, 3, 32, 32)
    pred_deltas_np = pred_deltas.squeeze(0).cpu().numpy()  # (H, 2)
    gt_crops_np = crops[burn_in + 1 : burn_in + 1 + H].numpy()  # (H, 3, 32, 32)
    gt_deltas_np = deltas[burn_in + 1 : burn_in + 1 + H].numpy()  # (H, 2)
    gt_centers_np = centers[burn_in + 1 : burn_in + 1 + H].numpy()  # (H, 2)

    actual_H = min(pred_crops_np.shape[0], gt_crops_np.shape[0])
    if actual_H < 1:
        return GazeRolloutMetrics()

    pred_crops_np = pred_crops_np[:actual_H]
    pred_deltas_np = pred_deltas_np[:actual_H]
    gt_crops_np = gt_crops_np[:actual_H]
    gt_deltas_np = gt_deltas_np[:actual_H]
    gt_centers_np = gt_centers_np[:actual_H]

    # --- Crop L1 ---
    crop_l1 = float(np.abs(pred_crops_np - gt_crops_np).mean())

    # --- Baseline: copy last context crop ---
    last_crop = crops[burn_in].numpy()  # (3, 32, 32)
    baseline_crop_l1 = float(np.abs(last_crop[None] - gt_crops_np).mean())

    # --- Position metrics via cumulative deltas ---
    start_center = centers[burn_in].numpy()  # (2,)
    pred_centers = start_center + np.cumsum(pred_deltas_np, axis=0)  # (H, 2)

    displacements = np.linalg.norm(pred_centers - gt_centers_np, axis=1)  # (H,)

    # Check for divergence (NaN/Inf or extreme error)
    if not np.all(np.isfinite(displacements)):
        return GazeRolloutMetrics(
            crop_l1=crop_l1,
            baseline_crop_l1=baseline_crop_l1,
            rollout_length=actual_H,
            diverged=True,
        )

    ade = float(displacements.mean())
    fde = float(displacements[-1])

    # --- HDE: horizon-stratified displacement error ---
    ts = timestamps.numpy()
    cum_time = ts[burn_in + 1 : burn_in + 1 + actual_H] - ts[burn_in]
    hde = {}
    for tau in time_horizons:
        idx = np.searchsorted(cum_time, tau)
        if idx < actual_H:
            hde[tau] = float(displacements[idx])

    # --- Velocity error ---
    if actual_H > 1:
        pred_vel = np.diff(pred_centers, axis=0)
        gt_vel = np.diff(gt_centers_np, axis=0)
        vel_err = float(np.linalg.norm(pred_vel - gt_vel, axis=1).mean())
    else:
        vel_err = 0.0

    return GazeRolloutMetrics(
        crop_l1=crop_l1,
        baseline_crop_l1=baseline_crop_l1,
        ade=ade,
        fde=fde,
        hde=hde,
        velocity_error=vel_err,
        rollout_length=actual_H,
        diverged=False,
    )


def evaluate_dataset(
    model: torch.nn.Module,
    sequences: list[dict[str, torch.Tensor]],
    context_len: int = 10,
    burn_in_frac: float = 0.2,
    time_horizons: list[float] | None = None,
) -> dict:
    """Evaluate model across all sequences in a dataset split.

    Args:
        model: Trained GazeDynamics model.
        sequences: List of sequence dicts from GazeSequenceDataset.
        context_len: Context window size (K).
        burn_in_frac: Fraction for burn-in.
        time_horizons: Wall-clock horizons for HDE.

    Returns:
        Dict with aggregated metrics.
    """
    if time_horizons is None:
        time_horizons = [10.0, 30.0, 60.0, 120.0, 300.0]

    all_metrics: list[GazeRolloutMetrics] = []

    for seq in sequences:
        metrics = evaluate_rollout(
            model=model,
            sequence=seq,
            context_len=context_len,
            burn_in_frac=burn_in_frac,
            time_horizons=time_horizons,
        )
        if metrics.rollout_length > 0:
            all_metrics.append(metrics)

    if not all_metrics:
        logger.warning("No valid rollouts produced")
        return {"num_sequences": 0}

    # Filter out NaN/Inf metrics
    valid_metrics = [
        m for m in all_metrics
        if np.isfinite(m.ade) and np.isfinite(m.fde) and not m.diverged
    ]
    diverged_count = sum(1 for m in all_metrics if m.diverged)
    nan_count = len(all_metrics) - len(valid_metrics) - diverged_count

    if nan_count > 0:
        logger.warning(
            "Filtered %d/%d sequences with NaN/Inf metrics",
            nan_count, len(all_metrics),
        )

    # Divergence rate includes both explicit divergence and NaN
    total_evaluated = len(all_metrics)
    total_invalid = total_evaluated - len(valid_metrics)
    divergence_rate = total_invalid / total_evaluated if total_evaluated > 0 else 0.0

    if not valid_metrics:
        logger.warning("All rollouts produced NaN/Inf or diverged")
        return {"num_sequences": 0, "divergence_rate": divergence_rate}

    # Aggregate position metrics
    ades = np.array([m.ade for m in valid_metrics])
    fdes = np.array([m.fde for m in valid_metrics])
    vel_errs = np.array([m.velocity_error for m in valid_metrics])
    crop_l1s = np.array([m.crop_l1 for m in valid_metrics])
    baseline_l1s = np.array([m.baseline_crop_l1 for m in valid_metrics])
    lengths = np.array([m.rollout_length for m in valid_metrics])

    # Crop L1 improvement over baseline
    mean_crop_l1 = float(np.mean(crop_l1s))
    mean_baseline_l1 = float(np.mean(baseline_l1s))
    crop_improvement = (
        (mean_baseline_l1 - mean_crop_l1) / mean_baseline_l1 * 100.0
        if mean_baseline_l1 > 0 else 0.0
    )

    results = {
        "num_sequences": len(valid_metrics),
        "divergence_rate": divergence_rate,
        "diverged_count": total_invalid,
        # Crop reconstruction
        "crop_l1_median": float(np.median(crop_l1s)),
        "crop_l1_mean": mean_crop_l1,
        "baseline_crop_l1_mean": mean_baseline_l1,
        "crop_improvement_pct": crop_improvement,
        # Position: ADE
        "ade_median": float(np.median(ades)),
        "ade_mean": float(np.mean(ades)),
        "ade_iqr": (float(np.percentile(ades, 25)), float(np.percentile(ades, 75))),
        # Position: FDE
        "fde_median": float(np.median(fdes)),
        "fde_mean": float(np.mean(fdes)),
        "fde_iqr": (float(np.percentile(fdes, 25)), float(np.percentile(fdes, 75))),
        # Velocity error
        "velocity_error_median": float(np.median(vel_errs)),
        "velocity_error_mean": float(np.mean(vel_errs)),
        # Length-weighted ADE
        "ade_weighted": float(np.sum(ades * lengths) / np.sum(lengths)),
    }

    # HDE at each horizon
    for tau in time_horizons:
        horizon_vals = [m.hde[tau] for m in valid_metrics if tau in m.hde]
        if horizon_vals:
            arr = np.array(horizon_vals)
            results[f"hde_{int(tau)}s_median"] = float(np.median(arr))
            results[f"hde_{int(tau)}s_mean"] = float(np.mean(arr))
            results[f"hde_{int(tau)}s_count"] = len(horizon_vals)

    logger.info(
        "Evaluation: %d sequences | ADE=%.4f (median) | FDE=%.4f (median) | "
        "CropL1=%.4f (%.1f%% vs baseline) | DivRate=%.1f%%",
        len(valid_metrics), results["ade_median"], results["fde_median"],
        results["crop_l1_mean"], crop_improvement, divergence_rate * 100,
    )

    return results
