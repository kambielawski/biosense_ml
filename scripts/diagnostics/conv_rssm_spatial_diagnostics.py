#!/usr/bin/env python3
"""Spatial diagnostics for ConvRSSM predictions.

Analyzes whether the model captures organoid motion by examining:
1. Delta magnitude: actual vs predicted delta L2 norms per spatial cell
2. Spatial concentration: what fraction of energy is in top 5% of cells
3. KL spatial map: where is KL non-zero across the 16x16 grid
4. h_t spatial variance: does the hidden state develop spatial structure
5. Organoid cell ratio: fraction of "active" cells per timestep

Usage:
    python scripts/diagnostics/conv_rssm_spatial_diagnostics.py \
        --rssm_checkpoint outputs/conv_rssm/checkpoints/checkpoint_best.pt \
        --latent_h5 data/rssm/latents_16x32.h5 \
        --sequence_idx 139 \
        --output_dir outputs/diagnostics/conv_rssm_spatial
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from biosense_ml.models.conv_rssm import ConvRSSM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rssm_checkpoint", required=True, type=Path)
    parser.add_argument("--latent_h5", required=True, type=Path)
    parser.add_argument("--sequence_idx", default=139, type=int)
    parser.add_argument("--output_dir", default="outputs/diagnostics/conv_rssm_spatial", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def save_heatmap(data: np.ndarray, title: str, path: Path, cmap="viridis"):
    """Save a 16x16 heatmap upsampled for visibility."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(data, cmap=cmap, interpolation="nearest")
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=5,
                    color="white" if val < data.max() * 0.6 else "black")
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def load_conv_rssm(checkpoint_path: Path, device: torch.device) -> ConvRSSM:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = OmegaConf.create(ckpt["config"]["model"])
    model = ConvRSSM(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading ConvRSSM from {args.rssm_checkpoint}")
    model = load_conv_rssm(args.rssm_checkpoint, device)
    C_ae = model.ae_latent_channels
    S = model.spatial_size

    # Load data
    print(f"Loading latents from {args.latent_h5}")
    with h5py.File(args.latent_h5, "r") as f:
        all_latents = f["latents"][:]
        all_actions = f["actions"][:]
        seq_starts = f["sequence_starts"][:]
        seq_lengths = f["sequence_lengths"][:]

    si = args.sequence_idx
    start = int(seq_starts[si])
    slen = int(seq_lengths[si])
    print(f"Sequence {si}: length={slen}, start={start}")

    # Extract and reshape to spatial
    seq_latents_flat = torch.tensor(all_latents[start:start + slen], dtype=torch.float32).to(device)
    seq_actions = torch.tensor(all_actions[start:start + slen], dtype=torch.float32).to(device)
    seq_latents = seq_latents_flat.view(slen, C_ae, S, S)

    # Compute actual deltas
    actual_deltas = torch.zeros_like(seq_latents)
    actual_deltas[1:] = seq_latents[1:] - seq_latents[:-1]

    # ===== Run posterior through entire sequence =====
    print(f"Running posterior for {slen} frames...")
    h_t, z_t = model.initial_state(1, device)

    all_pred_deltas = []
    all_prior_mu = []
    all_prior_sigma = []
    all_post_mu = []
    all_post_sigma = []
    all_kl = []
    all_h_t = []

    for t in range(slen):
        a_prev = torch.zeros(1, model.action_dim, device=device) if t == 0 else seq_actions[t - 1].unsqueeze(0)
        delta_t = actual_deltas[t].unsqueeze(0)

        step = model.forward_single_step(h_t, z_t, a_prev, delta_t)
        h_t = step["h_t"]
        z_t = step["z_t"]

        all_pred_deltas.append(step["obs_pred"].squeeze(0))
        all_prior_mu.append(step["prior_mu"].squeeze(0))
        all_prior_sigma.append(step["prior_sigma"].squeeze(0))
        all_post_mu.append(step["post_mu"].squeeze(0))
        all_post_sigma.append(step["post_sigma"].squeeze(0))

        # Per-element KL
        kl = model.kl_divergence(
            step["post_mu"], step["post_sigma"],
            step["prior_mu"], step["prior_sigma"],
        ).squeeze(0)  # (C_z, H, W)
        all_kl.append(kl)
        all_h_t.append(h_t.squeeze(0))

    pred_deltas = torch.stack(all_pred_deltas)    # (T, C_ae, H, W)
    kl_all = torch.stack(all_kl)                  # (T, C_z, H, W)
    h_all = torch.stack(all_h_t)                  # (T, C_h, H, W)

    # Move to CPU for analysis
    actual_deltas_np = actual_deltas.cpu().numpy()
    pred_deltas_np = pred_deltas.cpu().numpy()
    kl_np = kl_all.cpu().numpy()
    h_np = h_all.cpu().numpy()

    print("\n" + "=" * 70)
    print("CONV RSSM SPATIAL DIAGNOSTICS")
    print("=" * 70)

    # ===== 1. Delta magnitude analysis =====
    print("\n--- 1. DELTA MAGNITUDE ANALYSIS ---")

    # L2 norm per spatial cell across channels: (T, H, W)
    actual_delta_norm = np.sqrt((actual_deltas_np ** 2).sum(axis=1))  # (T, H, W)
    pred_delta_norm = np.sqrt((pred_deltas_np ** 2).sum(axis=1))      # (T, H, W)

    # Mean/max across frames -> (H, W)
    actual_mean_norm = actual_delta_norm.mean(axis=0)
    actual_max_norm = actual_delta_norm.max(axis=0)
    pred_mean_norm = pred_delta_norm.mean(axis=0)
    pred_max_norm = pred_delta_norm.max(axis=0)

    print(f"  Actual deltas:")
    print(f"    Global mean L2 norm:    {actual_delta_norm.mean():.6f}")
    print(f"    Global max L2 norm:     {actual_delta_norm.max():.6f}")
    print(f"    Per-cell mean range:    [{actual_mean_norm.min():.6f}, {actual_mean_norm.max():.6f}]")
    print(f"    Per-cell max range:     [{actual_max_norm.min():.6f}, {actual_max_norm.max():.6f}]")

    print(f"  Predicted deltas (posterior):")
    print(f"    Global mean L2 norm:    {pred_delta_norm.mean():.6f}")
    print(f"    Global max L2 norm:     {pred_delta_norm.max():.6f}")
    print(f"    Per-cell mean range:    [{pred_mean_norm.min():.6f}, {pred_mean_norm.max():.6f}]")
    print(f"    Per-cell max range:     [{pred_max_norm.min():.6f}, {pred_max_norm.max():.6f}]")

    # Are predicted deltas literally zero?
    pred_abs = np.abs(pred_deltas_np)
    n_zero = (pred_abs < 1e-8).sum()
    n_total = pred_abs.size
    print(f"  Predicted delta elements < 1e-8: {n_zero}/{n_total} ({100*n_zero/n_total:.2f}%)")
    print(f"  Predicted delta elements < 1e-4: {(pred_abs < 1e-4).sum()}/{n_total} ({100*(pred_abs < 1e-4).sum()/n_total:.2f}%)")
    print(f"  Ratio pred/actual mean norm: {pred_delta_norm.mean() / (actual_delta_norm.mean() + 1e-10):.4f}")

    save_heatmap(actual_mean_norm, "Actual Delta Mean L2 Norm (per cell)", args.output_dir / "actual_delta_mean_norm.png")
    save_heatmap(pred_mean_norm, "Predicted Delta Mean L2 Norm (per cell)", args.output_dir / "pred_delta_mean_norm.png")
    save_heatmap(actual_max_norm, "Actual Delta Max L2 Norm (per cell)", args.output_dir / "actual_delta_max_norm.png")
    save_heatmap(pred_max_norm, "Predicted Delta Max L2 Norm (per cell)", args.output_dir / "pred_delta_max_norm.png")

    # ===== 2. Spatial concentration =====
    print("\n--- 2. SPATIAL CONCENTRATION ---")

    # Energy per cell = sum of squared deltas across time and channels
    actual_energy = (actual_deltas_np ** 2).sum(axis=(0, 1))  # (H, W)
    total_energy = actual_energy.sum()
    n_cells = S * S
    top_5pct_count = max(1, int(0.05 * n_cells))

    flat_energy = actual_energy.flatten()
    top_indices = np.argsort(flat_energy)[-top_5pct_count:]
    top_5pct_energy = flat_energy[top_indices].sum()
    top_5pct_frac = top_5pct_energy / (total_energy + 1e-10)

    print(f"  Total cells: {n_cells}")
    print(f"  Top 5% ({top_5pct_count} cells) contain {100 * top_5pct_frac:.1f}% of delta energy")
    print(f"  Top cell locations: {[np.unravel_index(i, (S, S)) for i in top_indices]}")
    print(f"  Energy range: [{actual_energy.min():.6f}, {actual_energy.max():.6f}]")
    print(f"  Energy ratio max/mean: {actual_energy.max() / (actual_energy.mean() + 1e-10):.1f}x")

    # Same for predicted
    pred_energy = (pred_deltas_np ** 2).sum(axis=(0, 1))
    pred_total = pred_energy.sum()
    pred_flat = pred_energy.flatten()
    pred_top_idx = np.argsort(pred_flat)[-top_5pct_count:]
    pred_top_frac = pred_flat[pred_top_idx].sum() / (pred_total + 1e-10)

    print(f"  Predicted: top 5% contain {100 * pred_top_frac:.1f}% of predicted delta energy")
    print(f"  Predicted energy ratio max/mean: {pred_energy.max() / (pred_energy.mean() + 1e-10):.1f}x")

    save_heatmap(actual_energy, "Actual Delta Energy (per cell)", args.output_dir / "actual_energy_map.png", cmap="hot")
    save_heatmap(pred_energy, "Predicted Delta Energy (per cell)", args.output_dir / "pred_energy_map.png", cmap="hot")

    # ===== 3. KL spatial map =====
    print("\n--- 3. KL SPATIAL MAP ---")

    # Average KL over time and channels -> (H, W)
    kl_spatial_mean = kl_np.mean(axis=(0, 1))  # avg over time and channels
    # Also per-channel average over time -> (C_z, H, W), then sum channels -> (H, W)
    kl_spatial_sum_channels = kl_np.mean(axis=0).sum(axis=0)

    print(f"  KL spatial range (mean over T, C_z): [{kl_spatial_mean.min():.6f}, {kl_spatial_mean.max():.6f}]")
    print(f"  KL spatial std: {kl_spatial_mean.std():.6f}")
    print(f"  KL spatial max/min ratio: {kl_spatial_mean.max() / (kl_spatial_mean.min() + 1e-10):.1f}x")

    # Is KL uniform or spatially structured?
    kl_flat = kl_spatial_mean.flatten()
    kl_cv = kl_flat.std() / (kl_flat.mean() + 1e-10)
    print(f"  KL coefficient of variation: {kl_cv:.4f}")
    print(f"  (CV < 0.1 = nearly uniform, CV > 0.5 = spatially structured)")

    # Top KL cells
    kl_top_idx = np.argsort(kl_flat)[-5:]
    print(f"  Top 5 KL cells: {[(np.unravel_index(i, (S,S)), f'{kl_flat[i]:.4f}') for i in kl_top_idx]}")

    save_heatmap(kl_spatial_mean, "Mean KL Divergence (per cell, avg over T & C_z)", args.output_dir / "kl_spatial_map.png", cmap="magma")
    save_heatmap(kl_spatial_sum_channels, "Sum KL Divergence (per cell, sum over C_z)", args.output_dir / "kl_spatial_sum_channels.png", cmap="magma")

    # ===== 4. h_t spatial variance =====
    print("\n--- 4. h_t SPATIAL VARIANCE ---")

    # Variance of h_t across spatial locations at each timestep, averaged over channels
    # h_np: (T, C_h, H, W)
    h_spatial_var_per_t = h_np.var(axis=(2, 3)).mean(axis=1)  # (T,) avg over channels
    h_spatial_var_mean = h_spatial_var_per_t.mean()
    h_spatial_var_final = h_spatial_var_per_t[-1]

    # Also: variance at each spatial location over time (temporal variation)
    h_spatial_std_map = h_np.std(axis=0).mean(axis=0)  # (H, W) avg over channels

    # Snapshot: final h_t channel-averaged spatial map
    h_final_map = h_np[-1].mean(axis=0)  # (H, W)

    print(f"  h_t spatial variance (across 16x16 locations, avg over channels):")
    print(f"    First frame:  {h_spatial_var_per_t[0]:.6f}")
    print(f"    Frame 50:     {h_spatial_var_per_t[min(49, slen-1)]:.6f}")
    print(f"    Last frame:   {h_spatial_var_final:.6f}")
    print(f"    Mean:         {h_spatial_var_mean:.6f}")
    print(f"  h_t channel-avg spatial map range (final frame): [{h_final_map.min():.4f}, {h_final_map.max():.4f}]")
    print(f"  h_t spatial uniformity check (std of final map): {h_final_map.std():.6f}")

    save_heatmap(h_final_map, "h_t Final Frame (channel avg)", args.output_dir / "h_final_spatial.png", cmap="coolwarm")
    save_heatmap(h_spatial_std_map, "h_t Temporal Std (per cell, channel avg)", args.output_dir / "h_temporal_std.png")

    # Plot spatial variance over time
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(h_spatial_var_per_t)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Spatial variance (avg over channels)")
    ax.set_title("h_t Spatial Variance Over Time")
    fig.tight_layout()
    fig.savefig(str(args.output_dir / "h_spatial_variance_time.png"), dpi=150)
    plt.close(fig)

    # ===== 5. Organoid cell ratio =====
    print("\n--- 5. ORGANOID CELL RATIO ---")

    # Per timestep: count cells with |delta| > 1% of max |delta| at that timestep
    active_fracs = []
    for t in range(slen):
        delta_norm_t = actual_delta_norm[t]  # (H, W)
        max_norm = delta_norm_t.max()
        if max_norm < 1e-10:
            active_fracs.append(0.0)
            continue
        threshold = 0.01 * max_norm
        n_active = (delta_norm_t > threshold).sum()
        active_fracs.append(n_active / n_cells)

    active_fracs = np.array(active_fracs)
    print(f"  Fraction of 'active' cells (|delta| > 1% of max |delta|):")
    print(f"    Mean: {active_fracs.mean():.4f} ({active_fracs.mean() * n_cells:.1f} / {n_cells} cells)")
    print(f"    Min:  {active_fracs.min():.4f}")
    print(f"    Max:  {active_fracs.max():.4f}")
    print(f"    Std:  {active_fracs.std():.4f}")

    # Same for predicted
    pred_active_fracs = []
    for t in range(slen):
        pnorm = pred_delta_norm[t]
        max_pnorm = pnorm.max()
        if max_pnorm < 1e-10:
            pred_active_fracs.append(0.0)
            continue
        thresh = 0.01 * max_pnorm
        pred_active_fracs.append((pnorm > thresh).sum() / n_cells)

    pred_active_fracs = np.array(pred_active_fracs)
    print(f"  Predicted 'active' cells:")
    print(f"    Mean: {pred_active_fracs.mean():.4f} ({pred_active_fracs.mean() * n_cells:.1f} / {n_cells} cells)")

    # Plot active fraction over time
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(active_fracs, label="Actual", alpha=0.8)
    ax.plot(pred_active_fracs, label="Predicted", alpha=0.8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Fraction of active cells")
    ax.set_title("Active Cell Fraction Over Time (|delta| > 1% of max)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(args.output_dir / "active_cell_fraction.png"), dpi=150)
    plt.close(fig)

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Actual delta mean norm:     {actual_delta_norm.mean():.6f}")
    print(f"  Predicted delta mean norm:  {pred_delta_norm.mean():.6f}")
    print(f"  Pred/actual ratio:          {pred_delta_norm.mean() / (actual_delta_norm.mean() + 1e-10):.4f}")
    print(f"  Spatial concentration (actual): top 5% cells = {100*top_5pct_frac:.1f}% energy")
    print(f"  Spatial concentration (pred):   top 5% cells = {100*pred_top_frac:.1f}% energy")
    print(f"  KL spatial CV:              {kl_cv:.4f} ({'uniform' if kl_cv < 0.1 else 'structured'})")
    print(f"  h_t spatial var (final):    {h_spatial_var_final:.6f}")
    print(f"  Active cells (actual):      {active_fracs.mean():.1%}")
    print(f"  Active cells (pred):        {pred_active_fracs.mean():.1%}")
    print(f"\n  Heatmaps saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
