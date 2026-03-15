"""PyTorch dataset for trajectory prediction from HDF5.

Loads the HDF5 file produced by build_trajectory_dataset.py and provides
windowed samples for training trajectory prediction models (MLP and GRU).

Each sample is a contiguous window of `seq_len` timesteps from a single
sequence, containing the 10-dim feature vector and the target (x, y).
"""

import json
import logging

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TrajectoryDataset(Dataset):
    """Dataset of fixed-length trajectory windows for next-step prediction.

    Loads all data into memory (small dataset). Each item is a window of
    seq_len consecutive timesteps from a single batch sequence.

    Feature vector (10-dim): centroid_xy(2) + velocity_xy(2) + dt(1) + actions(5)
    Target: centroid_xy at each timestep (shifted by 1 for next-step prediction)
    """

    def __init__(
        self,
        h5_path: str,
        split: str = "train",
        seq_len: int = 50,
        context_len: int | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            h5_path: Path to trajectory HDF5 file.
            split: One of "train", "val", "test".
            seq_len: Window length for GRU training. For MLP, context_len is used.
            context_len: Context window size for MLP (K). If None, not used.
        """
        super().__init__()
        self.seq_len = seq_len
        self.context_len = context_len
        self.split = split

        with h5py.File(h5_path, "r") as f:
            centroid_xy = f["centroid_xy"][:]  # (N, 2)
            velocity_xy = f["velocity_xy"][:]  # (N, 2)
            dt = f["dt"][:]  # (N,)
            actions = f["actions"][:]  # (N, 5)
            seq_starts = f["sequence_starts"][:]  # (S,)
            seq_lengths = f["sequence_lengths"][:]  # (S,)
            split_assignments = f["split_assignments"][:]  # (S,) bytes
            self.norm_stats = json.loads(f.attrs["norm_stats"])

        # Build feature matrix: (N, 10)
        self.features = np.concatenate([
            centroid_xy,           # (N, 2)
            velocity_xy,           # (N, 2)
            dt[:, None],           # (N, 1)
            actions,               # (N, 5)
        ], axis=1).astype(np.float32)

        self.targets = centroid_xy.astype(np.float32)  # (N, 2)
        self.timestamps = dt.astype(np.float32)  # for cumulative time in eval

        # Filter sequences by split
        split_bytes = split.encode("utf-8")
        self.windows = []
        n_seqs = 0

        for i in range(len(seq_starts)):
            if split_assignments[i].strip() != split_bytes:
                continue
            n_seqs += 1
            start = int(seq_starts[i])
            length = int(seq_lengths[i])

            # Generate sliding windows with stride 1
            # Each window needs seq_len+1 frames (seq_len inputs + 1 target)
            min_len = seq_len + 1
            if length < min_len:
                # Skip sequences too short to form a full window
                continue

            for w_start in range(start, start + length - seq_len):
                self.windows.append((w_start, seq_len + 1))

        logger.info(
            "%s split: %d sequences -> %d windows (seq_len=%d)",
            split, n_seqs, len(self.windows), seq_len,
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a training sample.

        Returns dict with:
            features: (T, 10) — input feature vectors
            targets: (T, 2) — target (x, y) positions (shifted by 1)
            dt: (T,) — inter-frame time deltas
            length: int — actual sequence length (for variable-length handling)
        """
        start, length = self.windows[idx]
        # Input: features[start : start+length-1]
        # Target: targets[start+1 : start+length]
        T = length - 1
        feat = torch.from_numpy(self.features[start : start + T])
        tgt = torch.from_numpy(self.targets[start + 1 : start + length])
        dt = torch.from_numpy(self.timestamps[start + 1 : start + length])

        return {
            "features": feat,    # (T, 10)
            "targets": tgt,      # (T, 2)
            "dt": dt,            # (T,)
            "length": T,
        }


class TrajectorySequenceDataset(Dataset):
    """Full-sequence dataset for evaluation.

    Returns entire sequences (not windows) for autoregressive rollout evaluation.
    """

    def __init__(self, h5_path: str, split: str = "test") -> None:
        super().__init__()
        self.split = split

        with h5py.File(h5_path, "r") as f:
            centroid_xy = f["centroid_xy"][:]
            velocity_xy = f["velocity_xy"][:]
            dt = f["dt"][:]
            actions = f["actions"][:]
            seq_starts = f["sequence_starts"][:]
            seq_lengths = f["sequence_lengths"][:]
            split_assignments = f["split_assignments"][:]
            self.norm_stats = json.loads(f.attrs["norm_stats"])

        features = np.concatenate([
            centroid_xy, velocity_xy, dt[:, None], actions,
        ], axis=1).astype(np.float32)

        targets = centroid_xy.astype(np.float32)

        split_bytes = split.encode("utf-8")
        self.sequences = []

        for i in range(len(seq_starts)):
            if split_assignments[i].strip() != split_bytes:
                continue
            start = int(seq_starts[i])
            length = int(seq_lengths[i])
            if length < 10:
                continue

            self.sequences.append({
                "features": torch.from_numpy(features[start : start + length]),
                "targets": torch.from_numpy(targets[start : start + length]),
                "dt": torch.from_numpy(dt[start : start + length]),
                "length": length,
            })

        logger.info("%s split: %d full sequences for evaluation", split, len(self.sequences))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.sequences[idx]
