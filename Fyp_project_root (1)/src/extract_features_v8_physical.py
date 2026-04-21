"""
extract_features_v8_physical.py
-------------------------------
V8: The Physics Model.

Extracts Absolute PHYSICAL dimensions (Area, Widths, Height).
Uses MediaPipe Tasks API with robust image pre-processing.

Key fix: All images are resized to a uniform dimension before passing
to MediaPipe to avoid the ChannelSize crash on Windows.
"""

import os
import re
import cv2
import numpy as np
import urllib.request
import mediapipe as mp
from tqdm import tqdm

from project_paths import dataset_2dimage_dir, features_dir

# ---------------------------------------------------------------------------
# Download MediaPipe Task Model if missing
# ---------------------------------------------------------------------------
def get_task_model():
    model_path = "pose_landmarker_heavy.task"
    if not os.path.exists(model_path):
        print(f"\n[INIT] Downloading MediaPipe Task Model ({model_path})...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        try:
            urllib.request.urlretrieve(url, model_path)
            print("       Download complete.")
        except Exception as e:
            print(f"       Failed to download model! Error: {e}")
            import sys; sys.exit(1)
    return model_path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_truth(fname: str):
    name  = fname.split(".")[0]
    parts = name.split("_")
    m3 = re.search(r'\d+', parts[3])
    m4 = re.search(r'\d+', parts[4])
    if m3 is None or m4 is None:
        raise ValueError("Missing digit groups")
    h_m  = int(m3.group()) / 100_000.0   # mm -> m
    w_kg = int(m4.group()) / 100_000.0   # g*10 -> kg
    if not (1.0 < h_m < 2.5) or not (20 < w_kg < 300):
        raise ValueError(f"Implausible truth values: height {h_m}m, weight {w_kg}kg")
    return h_m, w_kg


def safe_read_rgb(img_path: str, target_size: int = 640) -> np.ndarray | None:
    """
    Read an image and return a 3-channel RGB uint8 contiguous array.
    Resizes to (target_size x target_size) keeping aspect ratio via padding
    to avoid MediaPipe Tasks API crashes on variable-sized images.
    
    Returns (rgb_padded, scale_info) or (None, None).
    scale_info = (orig_h, orig_w, pad_top, pad_left, scale)
    """
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None

    orig_h, orig_w = bgr.shape[:2]

    # Scale to fit within target_size
    scale = target_size / max(orig_h, orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    bgr_resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to square
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = bgr_resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

    return rgb, (orig_h, orig_w, pad_top, pad_left, scale)


def process_image(img_path: str, true_h_m: float, landmarker):
    rgb, scale_info = safe_read_rgb(img_path, target_size=640)
    if rgb is None:
        return None

    orig_h, orig_w, pad_top, pad_left, scale = scale_info
    img_size = rgb.shape[0]  # 640

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    try:
        res = landmarker.detect(mp_image)
    except Exception:
        return None

    if not res.pose_landmarks or not res.segmentation_masks:
        return None

    # Landmarks are in normalized [0, 1] coordinates of the 640x640 padded image
    landmarks = res.pose_landmarks[0]
    pts_padded = np.array(
        [[lm.x * img_size, lm.y * img_size] for lm in landmarks],
        dtype=np.float32
    )

    # Convert padded coordinates back to original image coordinates
    pts_orig = np.zeros_like(pts_padded)
    pts_orig[:, 0] = (pts_padded[:, 0] - pad_left) / scale
    pts_orig[:, 1] = (pts_padded[:, 1] - pad_top) / scale

    # Pixel height in original image coordinates
    min_y = pts_orig[:, 1].min()
    max_y = pts_orig[:, 1].max()
    pixel_height = max_y - min_y
    if pixel_height < 50:
        return None

    px_per_m = pixel_height / true_h_m
    px2_per_m2 = px_per_m ** 2

    # Segmentation mask area (in padded image coords, then convert)
    mask = res.segmentation_masks[0].numpy_view()
    # Count mask pixels in padded image, then convert to original scale
    mask_pixels_padded = np.sum(mask > 0.5)
    # Each padded pixel = (1/scale)^2 original pixels
    mask_pixels_orig = mask_pixels_padded / (scale ** 2)
    area_m2 = mask_pixels_orig / (px_per_m ** 2)

    L_SH, R_SH = 11, 12
    L_HIP, R_HIP = 23, 24

    # Use original-scale coordinates for physical dimensions
    sh_w_px = np.linalg.norm(pts_orig[L_SH] - pts_orig[R_SH])
    hip_w_px = np.linalg.norm(pts_orig[L_HIP] - pts_orig[R_HIP])
    mid_sh = (pts_orig[L_SH] + pts_orig[R_SH]) / 2.0
    mid_hip = (pts_orig[L_HIP] + pts_orig[R_HIP]) / 2.0
    torso_l_px = np.linalg.norm(mid_sh - mid_hip)

    sh_w_m    = sh_w_px / px_per_m
    hip_w_m   = hip_w_px / px_per_m
    torso_l_m = torso_l_px / px_per_m

    return np.array([area_m2, sh_w_m, hip_w_m, torso_l_m, true_h_m], dtype=np.float32)


def extract_split(split_name: str, img_dir: str, landmarker):
    fnames = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    X_phys, Y_weight, names = [], [], []
    skipped = 0

    print(f"\nExtracting Physical Dimensions for {split_name} ({len(fnames)} images) ...")
    for fname in tqdm(fnames, desc=split_name):
        try:
            h_m, w_kg = parse_truth(fname)
        except ValueError:
            skipped += 1
            continue

        path = os.path.join(img_dir, fname)
        try:
            feats = process_image(path, h_m, landmarker)
        except Exception as e:
            skipped += 1
            continue

        if feats is not None:
            X_phys.append(feats)
            Y_weight.append(w_kg)
            names.append(fname)
        else:
            skipped += 1

    print(f"  -> {len(names)} samples valid, {skipped} skipped.")
    return (
        np.array(X_phys, dtype=np.float32),
        np.array(Y_weight, dtype=np.float32),
        np.array(names)
    )


if __name__ == "__main__":
    base_data = str(dataset_2dimage_dir())
    out_dir   = str(features_dir("v8"))
    os.makedirs(out_dir, exist_ok=True)

    model_path = get_task_model()

    # Configure MediaPipe Tasks
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        output_segmentation_masks=True,
        min_pose_detection_confidence=0.5
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        def get_split_dir(base, name):
            p1 = os.path.join(base, f"{name} (1)")
            p2 = os.path.join(base, name)
            return p1 if os.path.exists(p1) else p2

        splits = {
            "train": get_split_dir(base_data, "Image_train"),
            "val":   get_split_dir(base_data, "Image_val"),
            "test":  get_split_dir(base_data, "Image_test"),
        }

        for split, img_dir in splits.items():
            X, Y, fnames = extract_split(split, img_dir, landmarker)
            if len(X) > 0:
                out_path = os.path.join(out_dir, f"{split}_v8_phys.npz")
                np.savez_compressed(out_path, features=X, weight=Y, names=fnames)
                print(f"  Saved {split}: X={X.shape}, weight={Y.shape}")
            else:
                print(f"  [Error] No valid features extracted for {split}.")

    print("\nV8 extraction complete.")
