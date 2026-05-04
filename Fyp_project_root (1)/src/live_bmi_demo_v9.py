"""
live_bmi_demo_v9.py
----------------------
V9: Hybrid Live Demo — Physics + Deep Features + Pose.

This demo combines physical measurements (from MediaPipe segmentation + 
known height) with EfficientNet-B3 deep features and 24-D pose ratios
for accurate weight and BMI estimation.

If the model was trained in hybrid mode (physics+deep+pose), all three
feature types are extracted live. If physics-only, only physical features.

Controls:
  q - quit

Usage:
    cd src
    python live_bmi_demo_v9.py
"""

import cv2
import numpy as np
import sys
import os
import urllib.request
import mediapipe as mp
import joblib
import torch
from collections import deque
from PIL import Image
from torchvision import transforms

from project_paths import models_dir

# ---------------------------------------------------------------------------
# EfficientNet-B3 deep feature extraction (same as V7 pipeline)
# ---------------------------------------------------------------------------
_PREPROCESS = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_effnet_model = None
_effnet_device = None


def _init_effnet():
    """Lazy-load EfficientNet-B3 for deep features."""
    global _effnet_model, _effnet_device
    if _effnet_model is not None:
        return
    from deep_features_v3 import EfficientNetFeatureExtractor
    _effnet_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _effnet_model = EfficientNetFeatureExtractor(pretrained=True).to(_effnet_device)
    _effnet_model.eval()
    print(f"  EfficientNet-B3 loaded on {_effnet_device}")


def extract_deep_features(bgr_crop):
    """Extract 1536-D EfficientNet-B3 features from a BGR crop."""
    _init_effnet()
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    t = _PREPROCESS(pil).unsqueeze(0).to(_effnet_device)
    with torch.no_grad():
        feat = _effnet_model(t).cpu().numpy().flatten()
    return feat


# ---------------------------------------------------------------------------
# Pose feature extraction (same as V7)
# ---------------------------------------------------------------------------
def extract_pose_features(coords):
    """Extract 24-D rich pose features from landmark coordinates."""
    from v7_pose_features import extract_rich_pose_features
    return extract_rich_pose_features(coords)


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
            sys.exit(1)
    return model_path


# ---------------------------------------------------------------------------
# Setup MediaPipe
# ---------------------------------------------------------------------------
def create_pose_landmarker(model_path):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        output_segmentation_masks=True,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return PoseLandmarker.create_from_options(options)


# ---------------------------------------------------------------------------
# Load V9 Model
# ---------------------------------------------------------------------------
def load_v9_model():
    path = models_dir("v9") / "ensemble_v9_hybrid.joblib"
    if not path.exists():
        print(f"\n[ERROR] V9 Model not found at: {path}")
        print("Please run `python train_v9_hybrid.py` first.")
        return None
    bundle = joblib.load(str(path))
    ftype = bundle.get("feature_type", "physical_v8")
    print(f"  Model type: {ftype}")
    return bundle


# ---------------------------------------------------------------------------
# ArUco Marker Detection
# ---------------------------------------------------------------------------
def detect_aruco(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=cv2.aruco.DetectorParameters_create())
    return corners, ids


# ---------------------------------------------------------------------------
# Physics feature extraction (from image + landmarks + height)
# ---------------------------------------------------------------------------
def extract_physics(image, landmarker_result, px_per_m):
    """
    Computes absolute physical dimensions (Area, Widths) from pixel data.
    Returns: (phys_feats_5D, coords, seg_mask, is_cut_off) or (None, None, None, False)
    """
    if not landmarker_result.pose_landmarks or not landmarker_result.segmentation_masks:
        return None, None, None, False

    h, w = image.shape[:2]
    landmarks = landmarker_result.pose_landmarks[0]
    
    # Check edges
    min_y_norm = min(lm.y for lm in landmarks)
    max_y_norm = max(lm.y for lm in landmarks)
    is_cut_off = (min_y_norm < 0.01 or max_y_norm > 0.99)

    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

    L_SH, R_SH = 11, 12
    L_HIP, R_HIP = 23, 24

    sh_w_px = np.linalg.norm(pts[L_SH] - pts[R_SH])
    hip_w_px = np.linalg.norm(pts[L_HIP] - pts[R_HIP])
    mid_sh = (pts[L_SH] + pts[R_SH]) / 2.0
    mid_hip = (pts[L_HIP] + pts[R_HIP]) / 2.0
    torso_l_px = np.linalg.norm(mid_sh - mid_hip)

    # Restore original robust pixel_height computation (domain shift fix)
    min_y = pts[:, 1].min()
    max_y = pts[:, 1].max()
    pixel_height = max_y - min_y

    if pixel_height < 50:
        return None, None, None, is_cut_off

    true_h_m = pixel_height / px_per_m
    px2_per_m2 = px_per_m ** 2

    mask = landmarker_result.segmentation_masks[0].numpy_view()
    mask_pixels = np.sum(mask > 0.5)
    area_m2 = mask_pixels / px2_per_m2

    sh_w_m = sh_w_px / px_per_m
    hip_w_m = hip_w_px / px_per_m
    torso_l_m = torso_l_px / px_per_m

    phys = np.array([area_m2, sh_w_m, hip_w_m, torso_l_m, true_h_m], dtype=np.float32)

    # Also return coordinates (with z=0) for pose feature extraction
    coords = np.column_stack([pts, np.zeros(len(pts))])

    return phys, coords, mask, is_cut_off


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_weight(bundle, features):
    """Run prediction through the saved pipeline (scaler -> PCA -> ensemble)."""
    X = features.reshape(1, -1)

    # Apply Scaler
    if "scaler" in bundle and bundle["scaler"] is not None:
        Xp = bundle["scaler"].transform(X)
    else:
        Xp = X

    w_ridge = bundle["ridge"].predict(Xp)[0]
    if bundle.get("xgb") is not None:
        w_xgb = bundle["xgb"].predict(Xp)[0]
        # 50/50 Ensemble V9
        pred = (w_ridge * 0.5) + (w_xgb * 0.5)
    else:
        pred = w_ridge

    # Safety clamp
    pred = float(np.clip(pred, 30.0, 250.0))
    return pred


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 50)
    print(" V9 Hybrid Live Demo (Physics + Deep + Pose)")
    print("=" * 50)

    # 1. Automated Height Scale
    print("\n[CALIBRATION] Using ArUco markers (ID 0 & ID 1) for automated scale.")
    last_px_per_m = None
    last_h_meters = 0.0

    # 2. Load model & setup
    bundle = load_v9_model()
    if bundle is None:
        return

    feat_type = bundle.get("feature_type", "physical_v8")
    is_hybrid = feat_type == "hybrid_v9"

    if is_hybrid:
        print("  Mode: HYBRID (physics + deep + pose) = best accuracy")
        print("  Loading EfficientNet-B3 ...")
        _init_effnet()
    else:
        print("  Mode: PHYSICS-ONLY (5 features)")

    model_path = get_task_model()
    landmarker = create_pose_landmarker(model_path)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    weight_hist = deque(maxlen=30)

    print("\nStarting live feed...")
    print("Stand back so your FULL BODY is visible. Press 'q' to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        display = frame.copy()
        h_frame, w_frame = frame.shape[:2]

        # 1. Detect ArUco Scale
        corners, ids = detect_aruco(frame)
        if ids is not None and 0 in ids and 1 in ids:
            idx0 = np.where(ids == 0)[0][0]
            idx1 = np.where(ids == 1)[0][0]
            c0 = corners[idx0][0].mean(axis=0)
            c1 = corners[idx1][0].mean(axis=0)
            pixel_dist = np.linalg.norm(c0 - c1)
            last_px_per_m = pixel_dist / 1.0  # Markers are 1m apart
            
            # Draw scale line
            cv2.line(display, (int(c0[0]), int(c0[1])), (int(c1[0]), int(c1[1])), (255, 0, 255), 2)
            cv2.putText(display, "Scale Locked", (10, h_frame - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        else:
            if last_px_per_m is None:
                cv2.putText(display, "Waiting for ArUco Markers (0 & 1)...", (10, h_frame - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display, "Using Last Known Scale", (10, h_frame - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Ensure 3-channel contiguous
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            res = landmarker.detect(mp_image)
        except Exception:
            res = None

        if last_px_per_m is not None and res and res.pose_landmarks:
            # Draw landmarks
            landmarks = res.pose_landmarks[0]
            for lm in landmarks:
                cx, cy = int(lm.x * w_frame), int(lm.y * h_frame)
                cv2.circle(display, (cx, cy), 3, (0, 255, 0), -1)

            # Extract physics
            phys_feats, coords, seg_mask, is_cut_off = extract_physics(frame, res, last_px_per_m)

            if phys_feats is not None and bundle:
                try:
                    if is_hybrid:
                        # Get pose-guided crop for deep features
                        xs, ys = coords[:, 0], coords[:, 1]
                        mg = 0.20
                        bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
                        x1 = max(0, int(xs.min() - bw * mg / 2))
                        x2 = min(w_frame, int(xs.max() + bw * mg / 2))
                        y1 = max(0, int(ys.min() - bh * mg / 2))
                        y2 = min(h_frame, int(ys.max() + bh * mg / 2))
                        crop = frame[y1:y2, x1:x2]

                        if crop.size > 0:
                            # Apply segmentation mask
                            if seg_mask is not None:
                                mask_crop = seg_mask[y1:y2, x1:x2]
                                mask_3ch = (mask_crop > 0.5).astype(np.uint8)[:, :, None]
                                crop = crop * mask_3ch

                            deep_feat = extract_deep_features(crop)      # 1536-D
                            pose_feat = extract_pose_features(coords)    # 24-D

                            # Combine: [physics(5) | deep(1536) | pose(24)]
                            combined = np.concatenate([phys_feats, deep_feat, pose_feat])
                            pred_w_kg = predict_weight(bundle, combined)
                        else:
                            pred_w_kg = predict_weight(bundle, phys_feats)
                    else:
                        # Physics-only mode
                        pred_w_kg = predict_weight(bundle, phys_feats)

                    weight_hist.append(pred_w_kg)

                except Exception as exc:
                    print(f"[prediction error] {exc}")

            if phys_feats is not None:
                last_h_meters = phys_feats[4]

            if weight_hist:
                smooth_w = float(np.mean(weight_hist))
                inst_w = float(weight_hist[-1])
                bmi = smooth_w / (last_h_meters ** 2) if last_h_meters > 0 else 0

                # Draw HUD
                cv2.rectangle(display, (10, 10), (370, 160), (20, 20, 40), -1)
                label = "V9 Hybrid" if is_hybrid else "V9 Physics"
                cv2.putText(display, label, (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

                cv2.putText(display, f"Height: {last_h_meters:.2f} m", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

                cv2.putText(display, f"Weight: {smooth_w:.1f} kg", (20, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
                cv2.putText(display, f"(instant: {inst_w:.1f})", (230, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 100), 1)

                cv2.putText(display, f"BMI: {bmi:.1f}", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 200, 255), 2)

                # BMI category
                if bmi < 18.5:
                    cat, col = "Underweight", (0, 210, 255)
                elif bmi < 25:
                    cat, col = "Normal", (0, 200, 50)
                elif bmi < 30:
                    cat, col = "Overweight", (0, 150, 255)
                else:
                    cat, col = "Obese", (30, 30, 220)
                cv2.putText(display, cat, (20, 155),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                            
                # Warning if cut off
                if is_cut_off:
                    cv2.putText(display, "WARNING: Move back! Body cut off", (w_frame // 2 - 150, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display, "Analyzing body...", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        else:
            if last_px_per_m is not None:
                cv2.putText(display, "No person detected - stand back", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("V9 Weight & BMI Estimation", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Demo closed.")


if __name__ == "__main__":
    main()
