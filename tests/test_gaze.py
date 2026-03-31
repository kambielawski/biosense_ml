"""Tests for the gaze motion detection and crop extraction pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from biosense_ml.pipeline.gaze import (
    CROP_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    detect_motion_center,
    extract_crop,
    normalize_crop_imagenet,
)


def _make_frame(h: int = 512, w: int = 512, value: int = 128) -> np.ndarray:
    """Create a uniform BGR frame."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _add_blob(frame: np.ndarray, cy: int, cx: int, radius: int = 10, value: int = 255) -> np.ndarray:
    """Add a bright circular blob to a frame."""
    out = frame.copy()
    yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    out[mask] = value
    return out


class TestDetectMotionCenter:
    """Tests for detect_motion_center."""

    def test_no_motion_returns_none(self):
        frame = _make_frame()
        result = detect_motion_center(frame, frame)
        assert result is None

    def test_detects_moving_blob(self):
        bg = _make_frame(value=50)
        # Blob at (200, 300) in frame 2
        frame2 = _add_blob(bg, cy=200, cx=300, radius=15, value=200)
        center = detect_motion_center(bg, frame2)
        assert center is not None
        cy, cx = center
        # Center should be near (200, 300)
        assert abs(cy - 200) < 20
        assert abs(cx - 300) < 20

    def test_detects_shifted_blob(self):
        bg = _make_frame(value=50)
        frame1 = _add_blob(bg, cy=200, cx=200, radius=10, value=200)
        frame2 = _add_blob(bg, cy=220, cx=230, radius=10, value=200)
        center = detect_motion_center(frame1, frame2)
        assert center is not None
        # Motion should be detected somewhere between the two positions
        cy, cx = center
        assert 180 < cy < 260
        assert 180 < cx < 260

    def test_low_threshold_detects_subtle_motion(self):
        bg = _make_frame(value=100)
        frame2 = _add_blob(bg, cy=256, cx=256, radius=20, value=115)
        # Default threshold might miss this, but low threshold catches it
        center = detect_motion_center(bg, frame2, threshold=5.0)
        assert center is not None

    def test_high_threshold_misses_subtle_motion(self):
        bg = _make_frame(value=100)
        frame2 = _add_blob(bg, cy=256, cx=256, radius=5, value=110)
        center = detect_motion_center(bg, frame2, threshold=50.0)
        assert center is None


class TestExtractCrop:
    """Tests for extract_crop."""

    def test_basic_crop(self):
        frame = _make_frame()
        crop = extract_crop(frame, center_y=256.0, center_x=256.0)
        assert crop.shape == (CROP_SIZE, CROP_SIZE, 3)
        assert crop.dtype == np.uint8

    def test_crop_at_edge_clamped(self):
        frame = _make_frame()
        # Top-left corner — should clamp
        crop = extract_crop(frame, center_y=5.0, center_x=5.0)
        assert crop.shape == (CROP_SIZE, CROP_SIZE, 3)

    def test_crop_at_bottom_right_clamped(self):
        frame = _make_frame()
        crop = extract_crop(frame, center_y=510.0, center_x=510.0)
        assert crop.shape == (CROP_SIZE, CROP_SIZE, 3)

    def test_crop_content_correct(self):
        """Verify crop extracts the right region."""
        frame = np.zeros((512, 512, 3), dtype=np.uint8)
        # Paint a known region
        frame[240:272, 240:272] = 255
        crop = extract_crop(frame, center_y=256.0, center_x=256.0)
        # The crop should be all white
        assert crop.mean() == 255.0


class TestNormalizeCropImagenet:
    """Tests for normalize_crop_imagenet."""

    def test_output_shape(self):
        crop = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = normalize_crop_imagenet(crop)
        assert result.shape == (3, 32, 32)
        assert result.dtype == np.float32

    def test_output_range(self):
        """ImageNet-normalized values should be approximately in [-2.2, 2.7]."""
        # All-zero image
        crop_black = np.zeros((32, 32, 3), dtype=np.uint8)
        result = normalize_crop_imagenet(crop_black)
        # (0 - mean) / std -> should be negative
        assert result.min() < 0

        # All-255 image
        crop_white = np.full((32, 32, 3), 255, dtype=np.uint8)
        result = normalize_crop_imagenet(crop_white)
        # (1 - mean) / std -> should be positive
        assert result.max() > 0

        # Check range is approximately [-2.2, 2.7]
        assert result.max() < 3.0
        assert result.min() > -3.0

    def test_denormalization_roundtrip(self):
        """Verify we can reverse the normalization."""
        crop_bgr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        normalized = normalize_crop_imagenet(crop_bgr)

        # Reverse: CHW -> HWC, denorm, RGB -> BGR
        chw = normalized
        hwc = chw.transpose(1, 2, 0)  # (H, W, 3) RGB
        rgb_float = hwc * IMAGENET_STD + IMAGENET_MEAN
        rgb_uint8 = np.clip(rgb_float * 255, 0, 255).astype(np.uint8)
        # RGB -> BGR
        import cv2
        bgr_recovered = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

        # Should be close (rounding errors from uint8 conversion)
        assert np.allclose(bgr_recovered, crop_bgr, atol=2)


class TestDeltaConsistency:
    """Test that deltas are consistent with centers."""

    def test_deltas_sum_to_trajectory(self):
        """cumsum of deltas should approximately reconstruct centers."""
        # Simulate a simple trajectory
        centers = np.array([
            [100.0, 200.0],
            [102.0, 203.0],
            [105.0, 201.0],
            [107.0, 205.0],
        ])
        deltas = np.zeros_like(centers)
        deltas[0] = [0.0, 0.0]
        for t in range(1, len(centers)):
            deltas[t] = centers[t] - centers[t - 1]

        # Reconstruct via cumsum
        reconstructed = centers[0] + np.cumsum(deltas, axis=0)
        np.testing.assert_allclose(reconstructed, centers, atol=1e-5)
