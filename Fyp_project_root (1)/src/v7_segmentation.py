"""
v7_segmentation.py
------------------
Background-removal utility using MediaPipe Pose segmentation mask.

Key improvement over V6:
  EfficientNet features are extracted from a SEGMENTED crop (person pixels only,
  background replaced with neutral grey). This removes distracting background
  textures/colours that previously polluted the deep feature space.

Returns:
  segmented_bgr  — BGR image with background replaced by neutral grey (128,128,128)
  coords         — (33,3) MediaPipe landmark array in pixel space, or None
  mask           — float32 (H,W) probability map (0..1) from MediaPipe
"""

import cv2
import numpy as np
import mediapipe as mp

try:
    _mp_pose = mp.solutions.pose
except AttributeError:
    import sys
    print("\n[ENVIRONMENT ERROR] Your 'mediapipe' installation is corrupted or missing the 'solutions' module.")
    print(f"Python is loading mediapipe from: {getattr(mp, '__file__', 'Unknown')}")
    print("To fix this, please run the following command in this terminal:")
    print("    pip uninstall -y mediapipe")
    print("    pip install mediapipe\n")
    sys.exit(1)

# Background fill colour — neutral grey avoids biasing mean/std normalisation
_BG_COLOUR = (128, 128, 128)


def get_segmented_crop(
    bgr: np.ndarray,
    mask_threshold: float = 0.5,
    model_complexity: int = 1,
) -> tuple:
    """
    Run MediaPipe Pose with segmentation and return the person-masked image.

    Args:
        bgr              : Input BGR image (any resolution).
        mask_threshold   : Probability cutoff for "person" pixels (default 0.5).
        model_complexity : MediaPipe model complexity 0/1/2 (default 1 = balanced).

    Returns:
        (segmented_bgr, coords_or_None, mask)
          segmented_bgr : np.ndarray uint8 BGR, same shape as bgr.
          coords        : np.ndarray float32 (33, 3) in pixel space, or None.
          mask          : np.ndarray float32 (H, W), raw segmentation probability.
    """
    h, w = bgr.shape[:2]
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    with _mp_pose.Pose(
        static_image_mode=True,
        model_complexity=model_complexity,
        enable_segmentation=True,
        min_detection_confidence=0.5,
    ) as pose:
        res = pose.process(rgb)

    # ---- Segmentation mask ----
    if res.segmentation_mask is not None:
        mask = res.segmentation_mask          # float32 (H, W), 0..1
        person_bin = (mask > mask_threshold).astype(np.uint8)

        # Morphological clean-up: close small holes, dilate slightly to include edges
        kern = np.ones((7, 7), np.uint8)
        person_bin = cv2.morphologyEx(person_bin, cv2.MORPH_CLOSE,  kern)
        person_bin = cv2.morphologyEx(person_bin, cv2.MORPH_DILATE, kern, iterations=1)

        # Replace background with neutral grey
        m3         = person_bin[:, :, None]          # (H,W,1) broadcast
        bg         = np.full_like(bgr, _BG_COLOUR)
        segmented  = (bgr * m3 + bg * (1 - m3)).astype(np.uint8)
    else:
        # Fallback: return original if segmentation unavailable
        mask      = np.ones((h, w), dtype=np.float32)
        segmented = bgr.copy()

    # ---- Landmark coordinates ----
    coords = None
    if res.pose_landmarks:
        pts    = res.pose_landmarks.landmark
        coords = np.array(
            [[p.x * w, p.y * h, p.z] for p in pts],
            dtype=np.float32,
        )

    return segmented, coords, mask


def apply_mask_to_crop(
    crop: np.ndarray,
    full_mask: np.ndarray,
    y1: int,
    x1: int,
) -> np.ndarray:
    """
    Apply a mask slice (from a full-frame mask) to a crop.
    Useful in the live demo where the mask is already computed.

    Args:
        crop      : BGR crop of shape (h_c, w_c, 3).
        full_mask : float32 (H_full, W_full) probability map.
        y1, x1    : top-left coordinates of crop in the full frame.

    Returns:
        Masked BGR crop, background replaced with neutral grey.
    """
    h_c, w_c = crop.shape[:2]
    crop_mask = full_mask[y1:y1 + h_c, x1:x1 + w_c]
    m3        = (crop_mask > 0.5).astype(np.uint8)[:, :, None]
    bg        = np.full_like(crop, _BG_COLOUR)
    return (crop * m3 + bg * (1 - m3)).astype(np.uint8)
