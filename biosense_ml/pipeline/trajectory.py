"""Trajectory preprocessing: extract organoid centroids and build feature vectors.

Ports centroid extraction from data_processing/extract_velocity.py and combines
it with stimulus annotations from preprocessing.py to produce trajectory features
for autoregressive forward-prediction of organoid centroid (x, y) trajectories.

Feature set (per timestep):
  - centroid_xy (2): normalized (x,y) position in [0,1] (÷ image_dim)
  - velocity_xy (2): signed (dx/dt, dy/dt) in normalized coords/sec
  - dt (1): seconds since previous frame
  - actions (5): [estim_active, estim_current_ma_norm, time_since_estim_onset_norm,
                   chemical_active, dt]
"""

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from biosense_ml.pipeline.preprocessing import (
    annotate_stimulus,
    discover_batch_dirs,
    discover_batch_files,
    load_commands,
    parse_image_timestamp,
)

logger = logging.getLogger(__name__)

# Image processing defaults (matching extract_velocity.py at 512px)
DEFAULT_CROP_LEFT = 1358
DEFAULT_CROP_RIGHT = 850
DEFAULT_DIMENSION = 512
DEFAULT_BLOCK_SIZE = 17
DEFAULT_C = 29
DEFAULT_MASK_RADIUS = 240

# Normalization constants
MAX_STIMULUS_DURATION = 300.0  # seconds, for time_since_onset normalization


# ---------------------------------------------------------------------------
# Image processing (ported from extract_velocity.py)
# ---------------------------------------------------------------------------


def crop_and_downsample(
    image_path: str | Path,
    crop_left: int = DEFAULT_CROP_LEFT,
    crop_right: int = DEFAULT_CROP_RIGHT,
    dimension: int = DEFAULT_DIMENSION,
) -> np.ndarray:
    """Crop raw image to square and downsample to target dimension.

    Raw images are 9152x6944. Crop to 6944x6944 square by removing
    crop_left from left and crop_right from right, then resize.
    """
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")
    h, w = frame.shape[:2]
    cropped = frame[0:h, crop_left : w - crop_right]
    return cv2.resize(cropped, (dimension, dimension))


def mask_frame_adaptive(
    frame: np.ndarray,
    block_size: int = DEFAULT_BLOCK_SIZE,
    C: int = DEFAULT_C,
) -> np.ndarray:
    """Adaptive thresholding to isolate organoid from background."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C
    )


def remove_isolated_pixels(binary_image: np.ndarray) -> np.ndarray:
    """Remove single-pixel connected components."""
    num_labels, labels = cv2.connectedComponents(binary_image, connectivity=8)
    output = binary_image.copy()
    for label in range(1, num_labels):
        if np.sum(labels == label) == 1:
            output[labels == label] = 0
    return output


def apply_circular_mask(
    binary_image: np.ndarray, center: tuple[int, int], radius: int
) -> np.ndarray:
    """Zero out everything outside a circular region."""
    h, w = binary_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return cv2.bitwise_and(binary_image, mask)


def get_largest_centroid(binary_image: np.ndarray) -> tuple[int, int] | None:
    """Find the centroid of the largest contour in a binary image.

    Returns (x, y) in pixel coords, or None if no contours found.
    """
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    best_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(best_contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def extract_centroid(
    image_path: str | Path,
    crop_left: int = DEFAULT_CROP_LEFT,
    crop_right: int = DEFAULT_CROP_RIGHT,
    block_size: int = DEFAULT_BLOCK_SIZE,
    C: int = DEFAULT_C,
    dimension: int = DEFAULT_DIMENSION,
    mask_radius: int = DEFAULT_MASK_RADIUS,
) -> tuple[int, int] | None:
    """Extract organoid centroid from a single raw image.

    Returns (x, y) in pixel coords within the downsampled image, or None.
    """
    frame = crop_and_downsample(str(image_path), crop_left, crop_right, dimension)
    binary = mask_frame_adaptive(frame, block_size, C)
    binary = remove_isolated_pixels(binary)
    binary = apply_circular_mask(
        binary, center=(dimension // 2, dimension // 2), radius=mask_radius
    )
    return get_largest_centroid(binary)


# ---------------------------------------------------------------------------
# Per-batch trajectory extraction
# ---------------------------------------------------------------------------


def process_single_batch(
    batch_id: int,
    batch_dir: Path,
    max_gap_frames: int = 2,
    dimension: int = DEFAULT_DIMENSION,
    block_size: int = DEFAULT_BLOCK_SIZE,
    C: int = DEFAULT_C,
    mask_radius: int = DEFAULT_MASK_RADIUS,
    crop_left: int = DEFAULT_CROP_LEFT,
    crop_right: int = DEFAULT_CROP_RIGHT,
) -> list[dict] | None:
    """Extract centroid trajectory and stimulus annotations for one batch.

    Returns a list of sequence dicts, each containing:
        - centroid_xy: (T, 2) normalized positions
        - timestamps: (T,) datetime objects
        - valid_mask: (T,) bool — True for real detections, False for interpolated
        - stimulus_annotations: list of dicts from annotate_stimulus()
        - batch_id: int
        - frame_indices: (T,) original frame indices

    Gaps of <= max_gap_frames are linearly interpolated; longer gaps split sequences.
    Leading/trailing missing frames are trimmed.
    """
    image_files = discover_batch_files(batch_dir)
    if not image_files:
        logger.warning("Batch %06d: no images found", batch_id)
        return None

    commands = load_commands(batch_dir)

    # Extract centroid for every frame
    raw_centroids: list[tuple[int, int] | None] = []
    timestamps: list[datetime] = []
    frame_indices: list[int] = []

    for idx, img_path in enumerate(image_files):
        try:
            ts = parse_image_timestamp(img_path)
        except ValueError:
            logger.warning("Batch %06d: skipping unparseable filename %s", batch_id, img_path.name)
            continue
        centroid = extract_centroid(
            img_path,
            crop_left=crop_left,
            crop_right=crop_right,
            block_size=block_size,
            C=C,
            dimension=dimension,
            mask_radius=mask_radius,
        )
        raw_centroids.append(centroid)
        timestamps.append(ts)
        frame_indices.append(idx)

    if not raw_centroids:
        logger.warning("Batch %06d: no frames processed", batch_id)
        return None

    # Build stimulus annotations for each frame
    stimulus_annotations = [annotate_stimulus(ts, commands) for ts in timestamps]

    # Split into sequences at gaps > max_gap_frames
    sequences = _split_and_interpolate(
        raw_centroids, timestamps, frame_indices, stimulus_annotations,
        batch_id, max_gap_frames, dimension,
    )

    n_frames = sum(s["centroid_xy"].shape[0] for s in sequences)
    n_detected = sum(s["valid_mask"].sum() for s in sequences)
    logger.info(
        "Batch %06d: %d frames -> %d sequences (%d detected, %d interpolated)",
        batch_id, len(raw_centroids), len(sequences), n_detected, n_frames - n_detected,
    )
    return sequences


def _split_and_interpolate(
    raw_centroids: list[tuple[int, int] | None],
    timestamps: list[datetime],
    frame_indices: list[int],
    stimulus_annotations: list[dict],
    batch_id: int,
    max_gap_frames: int,
    dimension: int,
) -> list[dict]:
    """Handle missing detections: interpolate short gaps, split at long gaps.

    Returns list of sequence dicts with normalized centroid_xy.
    """
    n = len(raw_centroids)
    if n == 0:
        return []

    # Find runs of consecutive None values
    sequences = []
    current_start = 0

    while current_start < n:
        # Trim leading Nones
        while current_start < n and raw_centroids[current_start] is None:
            current_start += 1
        if current_start >= n:
            break

        # Build a segment until we hit a gap that's too long
        segment_end = current_start
        consecutive_nones = 0

        for i in range(current_start, n):
            if raw_centroids[i] is None:
                consecutive_nones += 1
                if consecutive_nones > max_gap_frames:
                    # Split here: segment ends before the gap
                    segment_end = i - consecutive_nones
                    break
            else:
                consecutive_nones = 0
                segment_end = i
        else:
            # Reached end of array
            # Trim trailing Nones
            while segment_end >= current_start and raw_centroids[segment_end] is None:
                segment_end -= 1

        if segment_end >= current_start:
            seq = _build_sequence_segment(
                raw_centroids[current_start : segment_end + 1],
                timestamps[current_start : segment_end + 1],
                frame_indices[current_start : segment_end + 1],
                stimulus_annotations[current_start : segment_end + 1],
                batch_id,
                dimension,
            )
            if seq is not None:
                sequences.append(seq)

        current_start = segment_end + 1 + consecutive_nones if consecutive_nones > max_gap_frames else segment_end + 1

    return sequences


def _build_sequence_segment(
    centroids: list[tuple[int, int] | None],
    timestamps: list[datetime],
    frame_indices: list[int],
    stimulus_annotations: list[dict],
    batch_id: int,
    dimension: int,
) -> dict | None:
    """Build a single sequence segment, interpolating short internal gaps."""
    n = len(centroids)
    if n < 2:
        return None

    xy = np.zeros((n, 2), dtype=np.float32)
    valid = np.ones(n, dtype=bool)

    for i, c in enumerate(centroids):
        if c is not None:
            xy[i] = [c[0], c[1]]
        else:
            valid[i] = False

    # Linear interpolation for missing frames
    for dim in range(2):
        known_idx = np.where(valid)[0]
        if len(known_idx) < 2:
            return None
        missing_idx = np.where(~valid)[0]
        if len(missing_idx) > 0:
            xy[missing_idx, dim] = np.interp(missing_idx, known_idx, xy[known_idx, dim])

    # Normalize to [0, 1]
    xy /= dimension

    return {
        "centroid_xy": xy,
        "timestamps": timestamps,
        "frame_indices": np.array(frame_indices, dtype=np.int32),
        "valid_mask": valid.astype(np.uint8),
        "stimulus_annotations": stimulus_annotations,
        "batch_id": batch_id,
    }


# ---------------------------------------------------------------------------
# Feature computation (velocity, dt, actions)
# ---------------------------------------------------------------------------


def compute_velocity(
    centroid_xy: np.ndarray, timestamps: list[datetime]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute signed velocity and dt arrays from centroid positions.

    Args:
        centroid_xy: (T, 2) normalized positions.
        timestamps: List of T datetime objects.

    Returns:
        velocity_xy: (T, 2) — signed velocity in normalized coords/sec.
            First frame velocity is 0.
        dt: (T,) — seconds since previous frame. First frame dt is 0.
    """
    T = centroid_xy.shape[0]
    velocity_xy = np.zeros((T, 2), dtype=np.float32)
    dt = np.zeros(T, dtype=np.float32)

    for i in range(1, T):
        delta_t = (timestamps[i] - timestamps[i - 1]).total_seconds()
        dt[i] = delta_t
        if delta_t > 0:
            velocity_xy[i] = (centroid_xy[i] - centroid_xy[i - 1]) / delta_t

    return velocity_xy, dt


def encode_trajectory_actions(
    stimulus_annotations: list[dict],
    dt: np.ndarray,
    max_current_ma: float,
) -> np.ndarray:
    """Encode the 5D action vector for each timestep.

    Actions: [estim_active, estim_current_ma_norm, time_since_estim_onset_norm,
              chemical_active, dt]

    Args:
        stimulus_annotations: List of annotation dicts from annotate_stimulus().
        dt: (T,) inter-frame time deltas in seconds.
        max_current_ma: Maximum current_ma across entire dataset, for normalization.

    Returns:
        actions: (T, 5) float32 array.
    """
    T = len(stimulus_annotations)
    actions = np.zeros((T, 5), dtype=np.float32)

    for i, ann in enumerate(stimulus_annotations):
        elec = ann.get("electrical", {})
        estim_active = float(elec.get("active", False))
        current_ma = float(elec.get("current_ma", 0.0))
        current_ma_norm = current_ma / max_current_ma if max_current_ma > 0 else 0.0

        onset = ann.get("time_since_electrical_stimulus_onset", -1.0)
        onset_norm = max(onset, 0.0) / MAX_STIMULUS_DURATION if onset >= 0 else 0.0
        onset_norm = min(onset_norm, 1.0)

        chem = ann.get("chemical", {})
        chemical_active = float(chem.get("active", False))

        actions[i] = [estim_active, current_ma_norm, onset_norm, chemical_active, dt[i]]

    return actions


def compute_timestamps_since_start(
    timestamps: list[datetime],
) -> np.ndarray:
    """Compute seconds since batch start for each frame."""
    t0 = timestamps[0]
    return np.array(
        [(t - t0).total_seconds() for t in timestamps], dtype=np.float32
    )


# ---------------------------------------------------------------------------
# Dataset-wide normalization scan
# ---------------------------------------------------------------------------


def scan_max_current_ma(all_sequences: list[dict]) -> float:
    """Find the maximum current_ma across all sequences for normalization."""
    max_val = 0.0
    for seq in all_sequences:
        for ann in seq["stimulus_annotations"]:
            elec = ann.get("electrical", {})
            ma = float(elec.get("current_ma", 0.0))
            if ma > max_val:
                max_val = ma
    return max_val


# ---------------------------------------------------------------------------
# Train/val/test splitting
# ---------------------------------------------------------------------------


def assign_splits(
    batch_ids: list[int],
    all_sequences: list[dict],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> list[str]:
    """Assign train/val/test split per sequence, stratified by stimulus presence.

    Split is done at the batch level: all sequences from a batch go to the same split.

    Returns list of split strings ("train", "val", "test"), one per sequence.
    """
    rng = np.random.RandomState(seed)

    # Group unique batch IDs and check if they have stimulus
    unique_batches = sorted(set(batch_ids))
    batch_has_stim = {}
    for seq in all_sequences:
        bid = seq["batch_id"]
        has_stim = any(
            ann.get("electrical", {}).get("active", False)
            for ann in seq["stimulus_annotations"]
        )
        if bid not in batch_has_stim:
            batch_has_stim[bid] = has_stim
        else:
            batch_has_stim[bid] = batch_has_stim[bid] or has_stim

    # Stratified split: separate stim and no-stim batches
    stim_batches = [b for b in unique_batches if batch_has_stim.get(b, False)]
    no_stim_batches = [b for b in unique_batches if not batch_has_stim.get(b, False)]

    def split_list(items: list[int]) -> dict[int, str]:
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        assignments = {}
        for i, b in enumerate(items):
            if i < n_train:
                assignments[b] = "train"
            elif i < n_train + n_val:
                assignments[b] = "val"
            else:
                assignments[b] = "test"
        return assignments

    batch_splits = {}
    batch_splits.update(split_list(stim_batches))
    batch_splits.update(split_list(no_stim_batches))

    return [batch_splits[seq["batch_id"]] for seq in all_sequences]
