"""Intervention encoding: convert stimulus metadata to action vectors for the RSSM.

Two encoding schemes:
  - 3D: [stim_active, elapsed_frac, total_duration_normalized]
  - 2D: [stim_active, time_since_onset / max_time]

The 3D scheme is the default. Both produce vectors suitable for conditioning
the RSSM sequence model at each timestep.
"""

import numpy as np


# Maximum stimulus duration for normalization (seconds).
MAX_STIMULUS_DURATION = 300.0


def encode_action_3d(
    stimulus: dict,
    total_duration_s: float = 0.0,
) -> np.ndarray:
    """Encode stimulus metadata into a 3D action vector.

    Args:
        stimulus: Stimulus dict from preprocessing metadata, with keys:
            - "electrical": {"active": bool, ...} or {}
            - "time_since_electrical_stimulus_onset": float (-1 if no stimulus)
        total_duration_s: Total prescribed duration of the electrical stimulus
            in seconds. If unknown, defaults to 0.

    Returns:
        np.ndarray of shape (3,) with:
            [0] stim_active: 1.0 if electrical stimulus is active, else 0.0
            [1] elapsed_frac: fraction of prescribed duration elapsed, clipped to [0, 1]
            [2] total_duration_normalized: total_duration_s / MAX_STIMULUS_DURATION
    """
    electrical = stimulus.get("electrical", {})
    active = float(electrical.get("active", False))

    onset = stimulus.get("time_since_electrical_stimulus_onset", -1.0)
    if onset < 0 or total_duration_s <= 0:
        elapsed_frac = 0.0
    else:
        elapsed_frac = min(onset / total_duration_s, 1.0)

    duration_norm = min(total_duration_s / MAX_STIMULUS_DURATION, 1.0)

    return np.array([active, elapsed_frac, duration_norm], dtype=np.float32)


def encode_action_2d(stimulus: dict, max_time: float = 300.0) -> np.ndarray:
    """Encode stimulus metadata into a simpler 2D action vector.

    Args:
        stimulus: Stimulus dict from preprocessing metadata.
        max_time: Normalization constant for time_since_onset.

    Returns:
        np.ndarray of shape (2,) with:
            [0] stim_active: 1.0 if electrical stimulus is active, else 0.0
            [1] time_since_onset / max_time, clipped to [0, 1], or 0 if no stimulus
    """
    electrical = stimulus.get("electrical", {})
    active = float(electrical.get("active", False))

    onset = stimulus.get("time_since_electrical_stimulus_onset", -1.0)
    if onset < 0:
        time_norm = 0.0
    else:
        time_norm = min(onset / max_time, 1.0)

    return np.array([active, time_norm], dtype=np.float32)


def encode_actions_for_sequence(
    metadata_list: list[dict],
    action_dim: int = 3,
    total_duration_s: float = 0.0,
) -> np.ndarray:
    """Encode a full temporal sequence of metadata into action vectors.

    Args:
        metadata_list: List of metadata dicts, one per frame, temporally ordered.
        action_dim: 2 or 3, selecting the encoding scheme.
        total_duration_s: Total prescribed stimulus duration (for 3D encoding).

    Returns:
        np.ndarray of shape (T, action_dim).
    """
    actions = []
    for meta in metadata_list:
        stimulus = meta.get("stimulus", {})
        if action_dim == 3:
            actions.append(encode_action_3d(stimulus, total_duration_s))
        elif action_dim == 2:
            actions.append(encode_action_2d(stimulus))
        else:
            raise ValueError(f"action_dim must be 2 or 3, got {action_dim}")
    return np.stack(actions, axis=0)


def estimate_stimulus_duration(metadata_list: list[dict]) -> float:
    """Estimate the total electrical stimulus duration from a sequence of metadata.

    Looks at all frames where electrical stimulus is active and computes the
    time span from first active frame to last active frame.

    Args:
        metadata_list: List of metadata dicts, temporally ordered.

    Returns:
        Estimated duration in seconds, or 0.0 if no stimulus found.
    """
    active_times = []
    for meta in metadata_list:
        stimulus = meta.get("stimulus", {})
        electrical = stimulus.get("electrical", {})
        if electrical.get("active", False):
            t = meta.get("time_since_batch_start", 0.0)
            active_times.append(t)
    if len(active_times) < 2:
        return 0.0
    return active_times[-1] - active_times[0]
