"""Tests for gaze compositing pipeline."""

import numpy as np
import pytest

from biosense_ml.pipeline.compositing import (
    build_alpha_mask,
    composite_crop_on_frame,
    denormalize_crop,
    generate_rollout_video,
)
from biosense_ml.pipeline.gaze import CROP_SIZE, IMAGENET_MEAN, IMAGENET_STD


class TestBuildAlphaMask:
    """Tests for build_alpha_mask()."""

    def test_shape(self):
        mask = build_alpha_mask()
        assert mask.shape == (CROP_SIZE, CROP_SIZE)

    def test_dtype(self):
        mask = build_alpha_mask()
        assert mask.dtype == np.float32

    def test_center_is_one(self):
        mask = build_alpha_mask(crop_size=32, feather_width=4)
        # Center region should be 1.0
        assert mask[16, 16] == 1.0
        assert mask[10, 10] == 1.0

    def test_edge_is_zero(self):
        mask = build_alpha_mask(crop_size=32, feather_width=4)
        # Outermost pixel (d=0 from edge) should be 0
        assert mask[0, 16] == pytest.approx(0.0, abs=1e-6)
        assert mask[16, 0] == pytest.approx(0.0, abs=1e-6)
        assert mask[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_values_in_range(self):
        mask = build_alpha_mask()
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_symmetric(self):
        mask = build_alpha_mask()
        np.testing.assert_array_almost_equal(mask, mask[::-1, :])
        np.testing.assert_array_almost_equal(mask, mask[:, ::-1])

    def test_monotonic_from_edge(self):
        mask = build_alpha_mask(crop_size=32, feather_width=4)
        # Along the middle row, values should increase from left edge
        mid = 16
        for i in range(3):
            assert mask[mid, i] <= mask[mid, i + 1]

    def test_custom_size(self):
        mask = build_alpha_mask(crop_size=64, feather_width=8)
        assert mask.shape == (64, 64)
        assert mask[32, 32] == 1.0


class TestDenormalizeCrop:
    """Tests for denormalize_crop()."""

    def test_shape(self):
        crop = np.zeros((3, 32, 32), dtype=np.float32)
        result = denormalize_crop(crop)
        assert result.shape == (32, 32, 3)

    def test_dtype(self):
        crop = np.zeros((3, 32, 32), dtype=np.float32)
        result = denormalize_crop(crop)
        assert result.dtype == np.uint8

    def test_mean_maps_to_half(self):
        """A zero-normalized crop should map to ~ImageNet mean * 255."""
        crop = np.zeros((3, 32, 32), dtype=np.float32)
        result = denormalize_crop(crop)
        # Channel 0 (R): 0.485 * 255 ≈ 124
        expected_r = int(round(IMAGENET_MEAN[0] * 255))
        assert abs(int(result[0, 0, 0]) - expected_r) <= 1

    def test_clipping(self):
        """Extreme values should be clipped to [0, 255]."""
        crop = np.full((3, 32, 32), 100.0, dtype=np.float32)
        result = denormalize_crop(crop)
        assert result.max() == 255

        crop_neg = np.full((3, 32, 32), -100.0, dtype=np.float32)
        result_neg = denormalize_crop(crop_neg)
        assert result_neg.min() == 0

    def test_roundtrip_approximate(self):
        """Normalizing then denormalizing should approximately recover original."""
        original = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        # Normalize (RGB, HWC -> CHW)
        rgb_float = original.astype(np.float32) / 255.0
        normalized = ((rgb_float - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
        # Denormalize
        recovered = denormalize_crop(normalized)
        np.testing.assert_allclose(recovered, original, atol=2)


class TestCompositeCropOnFrame:
    """Tests for composite_crop_on_frame()."""

    def test_output_shape(self):
        bg = np.zeros((512, 512, 3), dtype=np.uint8)
        crop = np.full((32, 32, 3), 200, dtype=np.uint8)
        result = composite_crop_on_frame(bg, crop, 256.0, 256.0)
        assert result.shape == (512, 512, 3)

    def test_output_dtype(self):
        bg = np.zeros((512, 512, 3), dtype=np.uint8)
        crop = np.full((32, 32, 3), 200, dtype=np.uint8)
        result = composite_crop_on_frame(bg, crop, 256.0, 256.0)
        assert result.dtype == np.uint8

    def test_does_not_modify_background(self):
        bg = np.full((512, 512, 3), 100, dtype=np.uint8)
        bg_copy = bg.copy()
        crop = np.full((32, 32, 3), 200, dtype=np.uint8)
        composite_crop_on_frame(bg, crop, 256.0, 256.0)
        np.testing.assert_array_equal(bg, bg_copy)

    def test_center_pixel_matches_with_full_alpha(self):
        """With alpha=1 at center, the center pixels should be from the crop."""
        bg = np.zeros((512, 512, 3), dtype=np.uint8)
        crop = np.full((32, 32, 3), 200, dtype=np.uint8)
        alpha = build_alpha_mask()
        result = composite_crop_on_frame(bg, crop, 256.0, 256.0, alpha)
        # Center of crop (where alpha=1): should be 200
        assert result[256, 256, 0] == 200

    def test_far_pixels_unchanged(self):
        """Pixels far from the crop should remain as background."""
        bg = np.full((512, 512, 3), 50, dtype=np.uint8)
        crop = np.full((32, 32, 3), 200, dtype=np.uint8)
        result = composite_crop_on_frame(bg, crop, 256.0, 256.0)
        # Corner of the frame — far from crop center
        assert result[0, 0, 0] == 50

    def test_edge_clamping(self):
        """Crop near frame edge should be clamped and not error."""
        bg = np.zeros((512, 512, 3), dtype=np.uint8)
        crop = np.full((32, 32, 3), 200, dtype=np.uint8)
        # Center at (5, 5) would go out of bounds without clamping
        result = composite_crop_on_frame(bg, crop, 5.0, 5.0)
        assert result.shape == (512, 512, 3)

    def test_edge_clamping_bottom_right(self):
        """Crop near bottom-right edge should be clamped."""
        bg = np.zeros((512, 512, 3), dtype=np.uint8)
        crop = np.full((32, 32, 3), 200, dtype=np.uint8)
        result = composite_crop_on_frame(bg, crop, 510.0, 510.0)
        assert result.shape == (512, 512, 3)

    def test_feathered_blend(self):
        """Edge pixels of the crop should blend with background."""
        bg = np.full((512, 512, 3), 0, dtype=np.uint8)
        crop = np.full((32, 32, 3), 200, dtype=np.uint8)
        alpha = build_alpha_mask()
        result = composite_crop_on_frame(bg, crop, 256.0, 256.0, alpha)
        # At the crop edge (1px from border), alpha < 1, so value < 200
        # Top edge of crop region: row = 256 - 16 = 240
        edge_val = result[240, 256, 0]
        center_val = result[256, 256, 0]
        assert edge_val < center_val


class TestGenerateRolloutVideo:
    """Tests for generate_rollout_video()."""

    def test_creates_video_file(self, tmp_path):
        bg = np.zeros((64, 64, 3), dtype=np.uint8)
        T = 5
        # Create fake normalized crops
        pred_crops = np.random.randn(T, 3, 32, 32).astype(np.float32) * 0.1
        pred_centers = np.full((T, 2), 32.0, dtype=np.float32)

        output_path = tmp_path / "test_video.mp4"
        generate_rollout_video(
            background=bg,
            pred_crops=pred_crops,
            pred_centers=pred_centers,
            output_path=output_path,
            fps=5,
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_side_by_side_video(self, tmp_path):
        bg = np.zeros((64, 64, 3), dtype=np.uint8)
        T = 3
        pred_crops = np.random.randn(T, 3, 32, 32).astype(np.float32) * 0.1
        pred_centers = np.full((T, 2), 32.0, dtype=np.float32)
        gt_crops = np.random.randn(T, 3, 32, 32).astype(np.float32) * 0.1
        gt_centers = np.full((T, 2), 32.0, dtype=np.float32)

        output_path = tmp_path / "test_sbs.mp4"
        generate_rollout_video(
            background=bg,
            pred_crops=pred_crops,
            pred_centers=pred_centers,
            output_path=output_path,
            fps=5,
            gt_crops=gt_crops,
            gt_centers=gt_centers,
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_creates_parent_dirs(self, tmp_path):
        bg = np.zeros((64, 64, 3), dtype=np.uint8)
        T = 2
        pred_crops = np.zeros((T, 3, 32, 32), dtype=np.float32)
        pred_centers = np.full((T, 2), 32.0, dtype=np.float32)

        output_path = tmp_path / "nested" / "dir" / "video.mp4"
        generate_rollout_video(bg, pred_crops, pred_centers, output_path, fps=5)
        assert output_path.exists()
