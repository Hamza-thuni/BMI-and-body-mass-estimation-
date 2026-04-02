"""
extract_features_v7.py
----------------------
Full V7 feature extraction pipeline.

Improvements over V4/V6:
  1. Background removal via MediaPipe segmentation mask before EfficientNet.
  2. Rich 24-D pose features (v7_pose_features) instead of 16-D.
  3. Training-time augmentation (horizontal flip + brightness variants)
     — triples effective training set size at zero annotation cost.
  4. Graceful fallback: if segmentation mask unavailable, uses plain crop.
  5. YOLO crop always attempted first to focus on the person.

Outputs (per split):
  features_v7/{split}_v7.npz
    deep  : float32 (N, 1536)  EfficientNet-B3 features on segmented crop
    pose  : float32 (N, 24)    Rich scale-invariant pose ratios
    bmi   : float32 (N,)       Ground-truth BMI parsed from filename
    names : str     (N,)       Source filenames

Usage:
    cd src
    python extract_features_v7.py          # uses GPU if available
"""

import os
import re
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from deep_features_v3 import EfficientNetFeatureExtractor
from person_detector_yolo_v4 import YOLOPersonDetector
from v7_segmentation import get_segmented_crop
from v7_pose_features import extract_rich_pose_features
from project_paths import dataset_2dimage_dir, features_dir


# ---- EfficientNet preprocessing (identical to V4/V6 for comparability) ----
_PREPROCESS = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def _parse_bmi(fname: str) -> float:
    """
    Parse BMI from 2DImage2BMI filename convention.
    Format: {id}_{gender}_{?}_{height_cm*1000}_{weight_kg*1000}...
    """
    name  = fname.split(".")[0]
    parts = name.split("_")
    try:
        h_raw = int("".join(filter(str.isdigit, parts[3])))
        w_raw = int("".join(filter(str.isdigit, parts[4])))
    except (IndexError, ValueError):
        raise ValueError(f"Cannot parse BMI from filename: {fname!r}")
    h_m  = h_raw / 100_000.0
    w_kg = w_raw / 100_000.0
    return w_kg / (h_m ** 2 + 1e-8)


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _deep_features(bgr_crop: np.ndarray,
                   model: EfficientNetFeatureExtractor,
                   device: torch.device) -> np.ndarray:
    """EfficientNet-B3 feature vector (1536-D) from a BGR crop."""
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    t   = _PREPROCESS(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(t).cpu().numpy().flatten()
    return feat


def _augment(bgr: np.ndarray):
    """
    Yield training augmentations of a BGR image.
    Applied only to training split to avoid data leakage.
    """
    yield bgr                                                 # original
    yield cv2.flip(bgr, 1)                                    # horizontal flip
    yield cv2.convertScaleAbs(bgr, alpha=1.20, beta=15)      # brighter
    yield cv2.convertScaleAbs(bgr, alpha=0.82, beta=-10)     # darker


def _make_pose_box(coords: np.ndarray, img_w: int, img_h: int,
                   margin: float = 0.20):
    """Tight bounding box around all pose landmarks with margin."""
    xs, ys = coords[:, 0], coords[:, 1]
    bw = xs.max() - xs.min()
    bh = ys.max() - ys.min()
    x1 = max(0,     int(xs.min() - bw * margin / 2))
    x2 = min(img_w, int(xs.max() + bw * margin / 2))
    y1 = max(0,     int(ys.min() - bh * margin / 2))
    y2 = min(img_h, int(ys.max() + bh * margin / 2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image(img_path: str,
                  model: EfficientNetFeatureExtractor,
                  device: torch.device,
                  yolo: YOLOPersonDetector,
                  augment: bool = False) -> list:
    """
    Returns a list of (deep_feat, pose_feat) tuples.
    Each entry corresponds to one augmented variant of the image.
    Empty list ⟹ image should be skipped.
    """
    bgr = cv2.imread(img_path)
    if bgr is None:
        return []

    # YOLO crop — focuses model on person; falls back to full frame
    roi = yolo.detect_person_roi(bgr)
    person_bgr = bgr[roi[1]:roi[3], roi[0]:roi[2]].copy() if roi else bgr

    if person_bgr.size == 0:
        return []

    variants = list(_augment(person_bgr)) if augment else [person_bgr]
    results  = []

    for img_var in variants:
        h_v, w_v = img_var.shape[:2]

        # --- Segmentation + Pose (single MediaPipe call) ---
        seg_img, coords, _ = get_segmented_crop(img_var)

        if coords is None:
            continue            # skip variant if pose not detected

        # --- Pose-guided crop on segmented image ---
        box = _make_pose_box(coords, w_v, h_v, margin=0.20)
        crop = seg_img[box[1]:box[3], box[0]:box[2]] if box else seg_img

        if crop.size == 0:
            crop = seg_img

        # --- Feature extraction ---
        deep = _deep_features(crop, model, device)
        pose = extract_rich_pose_features(coords)

        results.append((deep, pose))

    return results


# ---------------------------------------------------------------------------
# Split-level extraction
# ---------------------------------------------------------------------------

def extract_split(split_name: str,
                  img_dir: str,
                  model: EfficientNetFeatureExtractor,
                  device: torch.device,
                  yolo: YOLOPersonDetector,
                  augment: bool = False):
    fnames = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    X_deep, X_pose, Y, names = [], [], [], []
    skipped = 0

    print(f"\nExtracting {split_name}  ({len(fnames)} images, augment={augment}) ...")
    for fname in tqdm(fnames, desc=split_name):
        try:
            bmi = _parse_bmi(fname)
        except ValueError:
            skipped += 1
            continue

        path  = os.path.join(img_dir, fname)
        pairs = process_image(path, model, device, yolo, augment=augment)

        for deep, pose in pairs:
            X_deep.append(deep)
            X_pose.append(pose)
            Y.append(bmi)
            names.append(fname)

        if not pairs:
            skipped += 1

    print(f"  → {len(Y)} samples collected, {skipped} images skipped.")
    return (
        np.array(X_deep, dtype=np.float32),
        np.array(X_pose, dtype=np.float32),
        np.array(Y,      dtype=np.float32),
        np.array(names),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base_data = str(dataset_2dimage_dir())
    out_dir   = str(features_dir("v7"))
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = EfficientNetFeatureExtractor(pretrained=True).to(device)
    model.eval()
    yolo  = YOLOPersonDetector(conf=0.4)

    def get_split_dir(base, name):
        p1 = os.path.join(base, f"{name} (1)")
        p2 = os.path.join(base, name)
        return p1 if os.path.exists(p1) else p2

    splits = {
        "train": (get_split_dir(base_data, "Image_train"), True),   # augmented
        "val":   (get_split_dir(base_data, "Image_val"),   False),
        "test":  (get_split_dir(base_data, "Image_test"),  False),
    }

    for split, (img_dir, aug) in splits.items():
        deep, pose, bmi, names = extract_split(
            split, img_dir, model, device, yolo, augment=aug
        )
        out_path = os.path.join(out_dir, f"{split}_v7.npz")
        np.savez_compressed(out_path, deep=deep, pose=pose, bmi=bmi, names=names)
        print(f"  Saved {split}: deep={deep.shape}, pose={pose.shape}, bmi={bmi.shape}")

    print("\n✅  V7 feature extraction complete.")
    print(f"    Output directory: {out_dir}")
