"""Gaze-based motion detection and crop extraction for organoid video prediction.

Detects organoid motion via frame differencing, extracts 32×32 crops centered
on the motion region, and produces metadata for training the GazeDynamics model.

No learned parameters — pure frame differencing with Gaussian blur.
"""

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, center_of_mass

from biosense_ml.pipeline.preprocessing import (
    annotate_stimulus,
    discover_batch_files,
    load_commands,
    parse_image_timestamp,
)
from biosense_ml.pipeline.trajectory import (
    crop_and_downsample,
    assign_splits,
)

logger = logging.getLogger(__name__)

# Gaze detection defaults
CROP_SIZE = 32
HALF_CROP = CROP_SIZE // 2
MOTION_SIGMA = 2.0
MOTION_THRESHOLD = 10.0  # pixel intensity threshold for motion detection
MAX_CENTER_JUMP = 50.0  # max pixels a center can move between frames (in 512x512 space)

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def detect_motion_center(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    sigma: float = MOTION_SIGMA,
    threshold: float = MOTION_THRESHOLD,
) -> tuple[float, float] | None:
    """Detect motion center via frame differencing.

    Args:
        frame_prev: Previous frame, uint8 (H, W, 3) BGR.
        frame_curr: Current frame, uint8 (H, W, 3) BGR.
        sigma: Gaussian blur sigma for smoothing the difference map.
        threshold: Minimum mean intensity for motion to be considered significant.

    Returns:
        (center_y, center_x) as float coords, or None if no significant motion.
    """
    # Grayscale absolute difference
    diff = np.abs(
        frame_curr.astype(np.float32) - frame_prev.astype(np.float32)
    ).mean(axis=2)

    # Smooth
    diff = gaussian_filter(diff, sigma=sigma)

    # Threshold
    motion_mask = diff > threshold
    if not motion_mask.any():
        return None

    # Center of mass of thresholded region
    cy, cx = center_of_mass(motion_mask)
    return (float(cy), float(cx))


def extract_crop(
    frame: np.ndarray,
    center_y: float,
    center_x: float,
    crop_size: int = CROP_SIZE,
) -> np.ndarray:
    """Extract a square crop from a frame, clamping to bounds.

    Args:
        frame: (H, W, 3) uint8 BGR image.
        center_y: Center row of the crop.
        center_x: Center column of the crop.
        crop_size: Side length of the square crop.

    Returns:
        (crop_size, crop_size, 3) uint8 crop.
    """
    h, w = frame.shape[:2]
    half = crop_size // 2

    # Clamp center so crop stays within frame
    cy = int(np.clip(round(center_y), half, h - half))
    cx = int(np.clip(round(center_x), half, w - half))

    return frame[cy - half : cy + half, cx - half : cx + half].copy()


def normalize_crop_imagenet(crop_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 crop to ImageNet-normalized float32 (C, H, W).

    Args:
        crop_bgr: (H, W, 3) uint8 BGR image.

    Returns:
        (3, H, W) float32 ImageNet-normalized tensor.
    """
    # BGR -> RGB, uint8 -> float [0, 1]
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # ImageNet normalize
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    # HWC -> CHW
    return normalized.transpose(2, 0, 1)


def process_single_batch_gaze(
    batch_id: int,
    batch_dir: Path,
    dimension: int = 512,
    crop_left: int = 1358,
    crop_right: int = 850,
    sigma: float = MOTION_SIGMA,
    threshold: float = MOTION_THRESHOLD,
    max_jump: float = MAX_CENTER_JUMP,
) -> dict | None:
    """Extract gaze crops and metadata for one batch.

    Returns a dict with:
        - crops: (T, 3, 32, 32) float32 ImageNet-normalized
        - centers: (T, 2) float32 (y, x) in 512×512 coords
        - deltas: (T, 2) float32 (dy, dx) shift from previous frame
        - timestamps: (T,) float64 unix timestamps
        - has_motion: (T,) bool
        - batch_id: int
        - stimulus_annotations: list of dicts

    Or None if batch has no usable frames.
    """
    image_files = discover_batch_files(batch_dir)
    if len(image_files) < 2:
        logger.warning("Batch %06d: fewer than 2 images, skipping", batch_id)
        return None

    commands = load_commands(batch_dir)

    # Load all frames into memory (512×512)
    frames = []
    timestamps = []
    stimulus_annotations = []

    for img_path in image_files:
        try:
            ts = parse_image_timestamp(img_path)
            frame = crop_and_downsample(
                img_path, crop_left=crop_left, crop_right=crop_right, dimension=dimension
            )
            frames.append(frame)
            timestamps.append(ts)
            stimulus_annotations.append(annotate_stimulus(ts, commands))
        except Exception:
            logger.warning("Batch %06d: failed to load %s, skipping frame", batch_id, img_path.name)
            continue

    T = len(frames)
    if T < 2:
        logger.warning("Batch %06d: fewer than 2 valid frames after loading", batch_id)
        return None

    # Detect motion and extract crops
    crops = np.zeros((T, 3, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    centers = np.zeros((T, 2), dtype=np.float32)
    deltas = np.zeros((T, 2), dtype=np.float32)
    has_motion = np.zeros(T, dtype=bool)
    ts_unix = np.zeros(T, dtype=np.float64)

    # First frame: no previous frame, so no motion detection
    # Use image center as default
    last_center = (dimension / 2.0, dimension / 2.0)

    for t in range(T):
        ts_unix[t] = timestamps[t].timestamp()

        if t == 0:
            # No previous frame — mark as static, use center of image
            has_motion[t] = False
            centers[t] = [dimension / 2.0, dimension / 2.0]
            deltas[t] = [0.0, 0.0]
        else:
            motion_center = detect_motion_center(
                frames[t - 1], frames[t], sigma=sigma, threshold=threshold
            )
            if motion_center is not None:
                # Spatial continuity: reject if center jumped too far
                # (likely noise — e.g. electrode bubbles, not organoid)
                dy = motion_center[0] - last_center[0]
                dx = motion_center[1] - last_center[1]
                jump_dist = (dy**2 + dx**2) ** 0.5

                if jump_dist <= max_jump:
                    has_motion[t] = True
                    centers[t] = [motion_center[0], motion_center[1]]
                    deltas[t] = [
                        centers[t, 0] - centers[t - 1, 0],
                        centers[t, 1] - centers[t - 1, 1],
                    ]
                    last_center = (centers[t, 0], centers[t, 1])
                else:
                    # Jump too large — treat as noise, carry forward
                    has_motion[t] = False
                    centers[t] = [last_center[0], last_center[1]]
                    deltas[t] = [0.0, 0.0]
            else:
                has_motion[t] = False
                # Carry forward last known center
                centers[t] = [last_center[0], last_center[1]]
                deltas[t] = [0.0, 0.0]

        # Extract and normalize crop at current center
        crop_bgr = extract_crop(frames[t], centers[t, 0], centers[t, 1])
        crops[t] = normalize_crop_imagenet(crop_bgr)

    n_motion = int(has_motion.sum())
    logger.info(
        "Batch %06d: %d frames, %d with motion (%.0f%%)",
        batch_id, T, n_motion, 100.0 * n_motion / T if T > 0 else 0,
    )

    return {
        "crops": crops,
        "centers": centers,
        "deltas": deltas,
        "timestamps": ts_unix,
        "has_motion": has_motion,
        "batch_id": batch_id,
        "stimulus_annotations": stimulus_annotations,
    }
