"""
extract_features_v8_physical.py
-------------------------------
V8: The Physics Model.

Extracts Absolute PHYSICAL dimensions (Area, Widths, Height).
Updated to use MediaPipe Tasks API natively.
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


def process_image(img_path: str, true_h_m: float, landmarker):
    bgr = cv2.imread(img_path)
    if bgr is None: return None
    
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    res = landmarker.detect(mp_image)
    if not res.pose_landmarks or not res.segmentation_masks:
        return None
        
    h, w = bgr.shape[:2]
    landmarks = res.pose_landmarks[0]
    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
    
    min_y = pts[:, 1].min()
    max_y = pts[:, 1].max()
    pixel_height = max_y - min_y
    if pixel_height < 50: return None
    
    px_per_m = pixel_height / true_h_m
    px2_per_m2 = px_per_m ** 2
    
    mask = res.segmentation_masks[0].numpy_view()
    mask_pixels = np.sum(mask > 0.5)
    area_m2 = mask_pixels / px2_per_m2
    
    L_SH, R_SH = 11, 12
    L_HIP, R_HIP = 23, 24
    
    sh_w_px = np.linalg.norm(pts[L_SH] - pts[R_SH])
    hip_w_px = np.linalg.norm(pts[L_HIP] - pts[R_HIP])
    mid_sh = (pts[L_SH] + pts[R_SH]) / 2.0
    mid_hip = (pts[L_HIP] + pts[R_HIP]) / 2.0
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
        feats = process_image(path, h_m, landmarker)
        
        if feats is not None:
            X_phys.append(feats)
            Y_weight.append(w_kg)
            names.append(fname)
        else:
            skipped += 1

    print(f"  → {len(names)} samples valid, {skipped} skipped.")
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

    print("\n✅ V8 extraction complete.")
