"""Full-frame compositing for gaze-based rollout videos.

Composites predicted 32×32 crops back into 512×512 frames using feathered
alpha blending at the crop boundary (4-pixel cosine falloff).

The background is a static frame (first frame or last burn-in frame).
The foreground is the predicted crop placed at the predicted gaze center.
"""

import cv2
import numpy as np

from biosense_ml.pipeline.gaze import CROP_SIZE, HALF_CROP, IMAGENET_MEAN, IMAGENET_STD


def build_alpha_mask(crop_size: int = CROP_SIZE, feather_width: int = 4) -> np.ndarray:
    """Build a 2D alpha mask with cosine falloff at the edges.

    Args:
        crop_size: Side length of the square crop.
        feather_width: Width of the feathering zone in pixels.

    Returns:
        (crop_size, crop_size) float32 array with values in [0, 1].
        alpha=1 in the interior (>= feather_width from edge),
        alpha=0 at the outermost edge pixel,
        smooth cosine transition in between.
    """
    mask_1d = np.ones(crop_size, dtype=np.float32)
    for i in range(feather_width):
        # d = distance from edge (0 at edge, feather_width at interior)
        d = i
        alpha = 0.5 * (1.0 - np.cos(np.pi * d / feather_width))
        mask_1d[i] = alpha
        mask_1d[crop_size - 1 - i] = alpha
    # Outer product for 2D mask
    return mask_1d[:, None] * mask_1d[None, :]


def denormalize_crop(crop: np.ndarray) -> np.ndarray:
    """Convert ImageNet-normalized (3, H, W) crop to (H, W, 3) RGB uint8.

    Args:
        crop: (3, H, W) float32 ImageNet-normalized crop.

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    # CHW -> HWC
    hwc = crop.transpose(1, 2, 0)
    # Denormalize
    rgb = hwc * IMAGENET_STD + IMAGENET_MEAN
    # Clip and convert
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def composite_crop_on_frame(
    background: np.ndarray,
    crop_rgb: np.ndarray,
    center_y: float,
    center_x: float,
    alpha_mask: np.ndarray | None = None,
    crop_size: int = CROP_SIZE,
) -> np.ndarray:
    """Place a crop on a background frame with feathered alpha blending.

    Args:
        background: (H, W, 3) uint8 RGB background frame.
        crop_rgb: (crop_size, crop_size, 3) uint8 RGB crop.
        center_y: Y coordinate of crop center in the background.
        center_x: X coordinate of crop center in the background.
        alpha_mask: Precomputed alpha mask. Built if None.
        crop_size: Side length of the crop.

    Returns:
        (H, W, 3) uint8 RGB composited frame.
    """
    if alpha_mask is None:
        alpha_mask = build_alpha_mask(crop_size)

    h, w = background.shape[:2]
    half = crop_size // 2

    # Clamp center so crop stays within frame
    cy = int(np.clip(round(center_y), half, h - half))
    cx = int(np.clip(round(center_x), half, w - half))

    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half

    # Blend
    result = background.copy()
    alpha_3d = alpha_mask[:, :, None]  # (crop_size, crop_size, 1)
    bg_patch = result[y0:y1, x0:x1].astype(np.float32)
    fg_patch = crop_rgb.astype(np.float32)

    blended = alpha_3d * fg_patch + (1.0 - alpha_3d) * bg_patch
    result[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)

    return result


def load_background_frame(
    batch_dir,
    frame_index: int = 0,
    dimension: int = 512,
    crop_left: int = 1358,
    crop_right: int = 850,
) -> np.ndarray:
    """Load a single 512×512 frame from a batch directory as RGB.

    Args:
        batch_dir: Path to the batch directory.
        frame_index: Which frame to load (0 = first).
        dimension: Target resolution.
        crop_left: Left crop for raw image.
        crop_right: Right crop for raw image.

    Returns:
        (dimension, dimension, 3) uint8 RGB frame.
    """
    from biosense_ml.pipeline.preprocessing import discover_batch_files
    from biosense_ml.pipeline.trajectory import crop_and_downsample

    image_files = discover_batch_files(batch_dir)
    if frame_index >= len(image_files):
        frame_index = 0

    bgr = crop_and_downsample(
        image_files[frame_index],
        crop_left=crop_left,
        crop_right=crop_right,
        dimension=dimension,
    )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def generate_rollout_video(
    background: np.ndarray,
    pred_crops: np.ndarray,
    pred_centers: np.ndarray,
    output_path,
    fps: int = 5,
    gt_crops: np.ndarray | None = None,
    gt_centers: np.ndarray | None = None,
) -> None:
    """Generate a full-frame composited rollout video as MP4.

    Args:
        background: (H, W, 3) uint8 RGB static background frame.
        pred_crops: (T, 3, 32, 32) float32 ImageNet-normalized predicted crops.
        pred_centers: (T, 2) float32 (y, x) predicted gaze centers.
        output_path: Path for the output MP4 file.
        fps: Frames per second.
        gt_crops: Optional (T, 3, 32, 32) ground truth crops for side-by-side.
        gt_centers: Optional (T, 2) ground truth centers for GT compositing.
    """
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    T = pred_crops.shape[0]
    h, w = background.shape[:2]
    alpha_mask = build_alpha_mask()

    side_by_side = gt_crops is not None and gt_centers is not None
    video_w = w * 2 if side_by_side else w
    video_h = h

    # Convert to BGR for cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (video_w, video_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    for t in range(T):
        # Predicted frame
        crop_rgb = denormalize_crop(pred_crops[t])
        pred_frame = composite_crop_on_frame(
            background, crop_rgb, pred_centers[t, 0], pred_centers[t, 1], alpha_mask
        )

        if side_by_side:
            # Ground truth frame
            gt_crop_rgb = denormalize_crop(gt_crops[t])
            gt_frame = composite_crop_on_frame(
                background, gt_crop_rgb, gt_centers[t, 0], gt_centers[t, 1], alpha_mask
            )
            combined = np.concatenate([gt_frame, pred_frame], axis=1)
        else:
            combined = pred_frame

        # RGB -> BGR for cv2
        writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    writer.release()
