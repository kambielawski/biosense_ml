"""PyTorch dataset for gaze-based dynamics model from HDF5.

Loads the HDF5 file produced by build_gaze_dataset.py and provides
sliding-window samples of 32×32 crops for training the GazeDynamics model.

Each sample is a contiguous window of K+1 frames: K context crops as input,
1 target crop + delta as supervision.
"""

import logging

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class GazeCropDataset(Dataset):
    """Dataset of sliding-window crop sequences for gaze dynamics training.

    Loads all data into memory (crops are 32×32 so this is manageable).
    Only frames with has_motion=True are used for training windows, but
    the window can include static frames as context.

    Each item returns K context crops and the next crop + delta as targets.
    """

    def __init__(
        self,
        h5_path: str,
        split: str = "train",
        context_len: int = 10,
    ) -> None:
        """Initialize dataset.

        Args:
            h5_path: Path to gaze HDF5 file.
            split: One of "train", "val", "test".
            context_len: Number of past frames in the context window (K).
        """
        super().__init__()
        self.context_len = context_len
        self.split = split

        self.windows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        with h5py.File(h5_path, "r") as f:
            n_batches = 0
            n_windows = 0

            for batch_key in sorted(f.keys()):
                grp = f[batch_key]
                batch_split = grp.attrs.get("split", "")
                if batch_split != split:
                    continue

                n_batches += 1
                crops = grp["crops"][:]  # (T, 3, 32, 32) float32
                deltas = grp["deltas"][:]  # (T, 2) float32
                has_motion = grp["has_motion"][:]  # (T,) bool

                T = len(crops)
                # Need at least K+1 frames to form one window
                if T < context_len + 1:
                    continue

                # Generate sliding windows where the TARGET frame has motion
                for t in range(context_len, T):
                    if not has_motion[t]:
                        continue
                    # Context: crops[t-K:t], Target: crops[t], delta[t]
                    ctx = crops[t - context_len : t]  # (K, 3, 32, 32)
                    tgt_crop = crops[t]  # (3, 32, 32)
                    tgt_delta = deltas[t]  # (2,)
                    self.windows.append((ctx, tgt_crop, tgt_delta))
                    n_windows += 1

        logger.info(
            "%s split: %d batches -> %d windows (context_len=%d)",
            split, n_batches, n_windows, context_len,
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a training sample.

        Returns dict with:
            context_crops: (K, 3, 32, 32) — input context window
            target_crop: (3, 32, 32) — target next crop
            target_delta: (2,) — target (dy, dx) position shift
        """
        ctx, tgt_crop, tgt_delta = self.windows[idx]
        return {
            "context_crops": torch.from_numpy(ctx),
            "target_crop": torch.from_numpy(tgt_crop),
            "target_delta": torch.from_numpy(tgt_delta),
        }


class GazeSequenceDataset(Dataset):
    """Full-sequence dataset for evaluation rollout.

    Returns entire sequences of crops, centers, and deltas for
    autoregressive rollout evaluation.
    """

    def __init__(self, h5_path: str, split: str = "test") -> None:
        super().__init__()
        self.split = split
        self.sequences: list[dict[str, torch.Tensor]] = []

        with h5py.File(h5_path, "r") as f:
            for batch_key in sorted(f.keys()):
                grp = f[batch_key]
                batch_split = grp.attrs.get("split", "")
                if batch_split != split:
                    continue

                crops = grp["crops"][:]  # (T, 3, 32, 32)
                centers = grp["centers"][:]  # (T, 2)
                deltas = grp["deltas"][:]  # (T, 2)
                has_motion = grp["has_motion"][:]  # (T,)
                timestamps = grp["timestamps"][:]  # (T,)

                if len(crops) < 10:
                    continue

                self.sequences.append({
                    "crops": torch.from_numpy(crops),
                    "centers": torch.from_numpy(centers),
                    "deltas": torch.from_numpy(deltas),
                    "has_motion": torch.from_numpy(has_motion),
                    "timestamps": torch.from_numpy(timestamps),
                    "batch_key": batch_key,
                    "length": len(crops),
                })

        logger.info("%s split: %d full sequences for evaluation", split, len(self.sequences))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.sequences[idx]
