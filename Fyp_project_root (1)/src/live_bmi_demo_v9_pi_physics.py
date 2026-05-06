"""
live_bmi_demo_v9_pi_physics.py
------------------------------
Lightweight V9: Physics-Only Live Demo for Raspberry Pi.
NO Deep Learning (EfficientNet/Torch) - runs fast on CPU.

Features:
  1. Area (segmented)
  2. Shoulder Width
  3. Hip Width
  4. Torso Length
  5. Height
  + 8 Engineered ratios

Usage:
    python live_bmi_demo_v9_pi_physics.py
"""

import cv2
import numpy as np
import sys
import os
import urllib.request
import mediapipe as mp
import joblib
from collections import deque

# ---------------------------------------------------------------------------
# Project Paths Shim (so we don't depend on the complex project_paths.py)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "models_v9")

# ---------------------------------------------------------------------------
# Physical setup
# ---------------------------------------------------------------------------
MARKER_BOTTOM_HEIGHT_M   = 0.80   
MARKER_TOP_HEIGHT_M      = 1.80   
MARKER_SEPARATION_M      = MARKER_TOP_HEIGHT_M - MARKER_BOTTOM_HEIGHT_M  

PARALLAX_FACTOR          = 0.94  
STATURE_ADJUST           = 1.08  

# ---------------------------------------------------------------------------
# Feature Engineering (Must match train_v9_physics_only.py)
# ---------------------------------------------------------------------------
def engineer_features_single(phys_5d):
    """
    Input: [area, sh_w, hip_w, torso_l, height]
    Output: 13-D feature vector
    """
    area, sh_w, hip_w, torso_l, height = phys_5d
    eps = 1e-6
    
    bmi_proxy     = area / (height ** 2 + eps)
    sh_hip_ratio  = sh_w / (hip_w + eps)
    area_per_h    = area / (height + eps)
    volume_proxy  = area * (sh_w + hip_w) / 2.0
    trunk_avg     = (sh_w + hip_w) / 2.0
    torso_ratio   = torso_l / (height + eps)
    compactness   = area / (sh_w * torso_l + eps)
    width_area    = (sh_w * hip_w) / (area + eps)

    return np.array([
        area, sh_w, hip_w, torso_l, height,
        bmi_proxy, sh_hip_ratio, area_per_h, volume_proxy,
        trunk_avg, torso_ratio, compactness, width_area
    ], dtype=np.float32)

# ---------------------------------------------------------------------------
# MediaPipe Task Model (Using LITE for Pi)
# ---------------------------------------------------------------------------
def get_lite_task_model():
    model_path = os.path.join(os.path.dirname(SCRIPT_DIR), "pose_landmarker_lite.task")
    if not os.path.exists(model_path):
        print(f"\n[INIT] Downloading MediaPipe LITE Model...")
        url = (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        )
        try:
            urllib.request.urlretrieve(url, model_path)
            print("       Download complete.")
        except Exception as e:
            print(f"       Failed to download! Falling back to heavy if exists.")
            heavy = os.path.join(os.path.dirname(SCRIPT_DIR), "pose_landmarker_heavy.task")
            if os.path.exists(heavy): return heavy
            sys.exit(1)
    return model_path

def create_pose_landmarker(model_path):
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_segmentation_masks=True,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)

# ---------------------------------------------------------------------------
# ArUco and Physics
# ---------------------------------------------------------------------------
def detect_aruco(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
    return corners, ids

def compute_scale(corners, ids, scale_hist):
    if ids is None or 0 not in ids or 1 not in ids:
        return scale_hist, (float(np.mean(scale_hist)) if scale_hist else None)
    
    ids_flat = ids.flatten()
    idx0 = np.where(ids_flat == 0)[0][0]
    idx1 = np.where(ids_flat == 1)[0][0]
    
    c0 = corners[idx0][0].mean(axis=0)
    c1 = corners[idx1][0].mean(axis=0)
    px_sep = abs(c1[1] - c0[1])
    
    if px_sep < 20: return scale_hist, (float(np.mean(scale_hist)) if scale_hist else None)
    
    inst_px_m = px_sep / MARKER_SEPARATION_M
    scale_hist.append(inst_px_m)
    return scale_hist, float(np.mean(scale_hist))

def extract_physics(image, res, wall_px_m):
    if not res.pose_landmarks or not res.segmentation_masks:
        return None, False
    
    h, w = image.shape[:2]
    landmarks = res.pose_landmarks[0]
    is_cut_off = landmarks[0].y < 0.05 or landmarks[28].y > 0.95 or landmarks[27].y > 0.95
    
    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
    person_px_m = wall_px_m / PARALLAX_FACTOR
    
    # Keypoints
    L_SH, R_SH = 11, 12
    L_HIP, R_HIP = 23, 24
    
    sh_w_m = np.linalg.norm(pts[L_SH] - pts[R_SH]) / person_px_m
    hip_w_m = np.linalg.norm(pts[L_HIP] - pts[R_HIP]) / person_px_m
    mid_sh = (pts[L_SH] + pts[R_SH]) / 2.0
    mid_hip = (pts[L_HIP] + pts[R_HIP]) / 2.0
    torso_l_m = np.linalg.norm(mid_sh - mid_hip) / person_px_m
    
    pixel_height = pts[:, 1].max() - pts[:, 1].min()
    true_h_m = (pixel_height / person_px_m) * STATURE_ADJUST
    
    mask = res.segmentation_masks[0].numpy_view()
    area_m2 = np.sum(mask > 0.5) / (person_px_m ** 2)
    
    return np.array([area_m2, sh_w_m, hip_w_m, torso_l_m, true_h_m], dtype=np.float32), is_cut_off

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_weight(bundle, phys_5d):
    X_eng = engineer_features_single(phys_5d).reshape(1, -1)
    
    if bundle.get("scaler"):
        X_scaled = bundle["scaler"].transform(X_eng)
    else:
        X_scaled = X_eng
        
    w_ridge = bundle["ridge"].predict(X_scaled)[0]
    if bundle.get("xgb"):
        w_xgb = bundle["xgb"].predict(X_scaled)[0]
        raw_pred = (w_ridge * 0.4) + (w_xgb * 0.6)
    else:
        raw_pred = w_ridge
        
    # Calibration Patch (Linear stretch)
    if raw_pred > 70:
        pred = 70 + (raw_pred - 70) * 1.2
    else:
        pred = raw_pred * 0.98
        
    # Safety Bounds
    area_m2 = phys_5d[0]
    lower = area_m2 * 105
    upper = area_m2 * 190
    pred = np.clip(pred, lower, upper)
    
    return float(np.clip(pred, 30, 200))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    model_path = os.path.join(MODELS_DIR, "ensemble_v9_physics.joblib")
    if not os.path.exists(model_path):
        print(f"Error: Physics model not found at {model_path}")
        return
    
    bundle = joblib.load(model_path)
    print("Loaded Physics-Only V9 Model.")
    
    mp_model_path = get_lite_task_model()
    landmarker = create_pose_landmarker(mp_model_path)
    
    cap = cv2.VideoCapture(0)
    scale_hist = deque(maxlen=30)
    weight_hist = deque(maxlen=20)
    last_px_m = None
    
    print("Starting Raspberry Pi Physics-Only Demo...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display = frame.copy()
        corners, ids = detect_aruco(frame)
        scale_hist, last_px_m = compute_scale(corners, ids, scale_hist)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect(mp_image)
        
        if last_px_m and res.pose_landmarks:
            phys_5d, is_cut_off = extract_physics(frame, res, last_px_m)
            if phys_5d is not None:
                w_kg = predict_weight(bundle, phys_5d)
                weight_hist.append(w_kg)
                
                h_m = phys_5d[4]
                avg_w = np.mean(weight_hist)
                bmi = avg_w / (h_m**2) if h_m > 0 else 0
                
                # UI
                cv2.rectangle(display, (10, 10), (300, 150), (0,0,0), -1)
                cv2.putText(display, f"H: {h_m:.2f}m", (20, 40), 1, 1.5, (0,255,0), 2)
                cv2.putText(display, f"W: {avg_w:.1f}kg", (20, 80), 1, 2.0, (0,255,0), 2)
                cv2.putText(display, f"BMI: {bmi:.1f}", (20, 120), 1, 1.5, (0,255,255), 2)
                
                if is_cut_off:
                    cv2.putText(display, "STEP BACK", (100, 200), 1, 3, (0,0,255), 3)

        cv2.imshow("Pi Physics V9", display)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
