"""
live_bmi_demo_v8_pi.py
----------------------
V8: The Physics Model Live Demo (Raspberry Pi Optimized).

This demo uses the new MediaPipe Tasks API, which fixes the Python 3.12 
`AttributeError: module 'mediapipe' has no attribute 'solutions'` bugs.

Features:
- Takes a manual height input (Placeholder for physical scale calibration)
- Calculates Absolute Physical Area and Widths using the input height.
- Very fast inference using Ridge/XGBoost.
- Optimized for lightweight processors (Raspberry Pi).

Controls:
  q - quit
"""

import cv2
import numpy as np
import time
import sys
import os
import urllib.request
import mediapipe as mp
import joblib

from project_paths import models_dir

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
# Setup MediaPipe Options
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
# Load V8 Model
# ---------------------------------------------------------------------------
def load_v8_model():
    path = models_dir("v8") / "ensemble_v8_physical.joblib"
    if not path.exists():
        print(f"\n[ERROR] V8 Model not found at: {path}")
        print("Please run `python train_v8_physical.py` first.")
        return None
    return joblib.load(str(path))

# ---------------------------------------------------------------------------
# Extraction Logic
# ---------------------------------------------------------------------------
def extract_physics(image, landmarker_result, true_h_m):
    """
    Computes absolute physical dimensions (Area, Widths) from pixel data.
    """
    if not landmarker_result.pose_landmarks or not landmarker_result.segmentation_masks:
        return None
        
    h, w = image.shape[:2]
    # In Tasks API, landmark coordinates are normalized [0.0, 1.0]
    landmarks = landmarker_result.pose_landmarks[0]
    
    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
    
    min_y = pts[:, 1].min()
    max_y = pts[:, 1].max()
    pixel_height = max_y - min_y
    if pixel_height < 50: 
        return None
        
    px_per_m = pixel_height / true_h_m
    px2_per_m2 = px_per_m ** 2
    
    mask = landmarker_result.segmentation_masks[0].numpy_view()
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

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_weight(bundle, features):
    Xs = bundle["scaler"].transform(features.reshape(1, -1))
    
    w_ridge = bundle["ridge"].predict(Xs)[0]
    if bundle.get("xgb"):
        w_xgb = bundle["xgb"].predict(Xs)[0]
        return (w_ridge + w_xgb) / 2.0
    return w_ridge

# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------
def main():
    print("\n" + "="*50)
    print(" V8 Physical Live Demo (Raspberry Pi Optimized)")
    print("="*50)
    
    # 1. Manual Height Input Placeholder
    print("\n[CALIBRATION] Since you do not have a background scale yet,")
    print("we will use a manual height input to convert pixels to centimeters.")
    while True:
        try:
            val = input("Please enter your exact height in CENTIMETERS (e.g. 175): ")
            h_meters = float(val) / 100.0
            if 1.0 < h_meters < 2.5:
                break
            else:
                print("Height must be between 100 and 250 cm.")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    print(f"\n✅ Height locked to {h_meters:.2f} m.")
    
    # 2. Setup
    model_path = get_task_model()
    landmarker = create_pose_landmarker(model_path)
    bundle = load_v8_model()
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("\nStarting live feed...")
    print("Stand back so your FULL BODY is visible. Press 'q' to quit.")
    
    while True:
        ok, frame = cap.read()
        if not ok: continue
        
        display = frame.copy()
        
        # Process MediaPipe Pose & Segmentation (Tasks API)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect(mp_image)
        
        if res.pose_landmarks:
            # Draw simple stick figure
            landmarks = res.pose_landmarks[0]
            for lm in landmarks:
                h, w = frame.shape[:2]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(display, (cx, cy), 3, (0, 255, 0), -1)
                
            # Extract physical dimensions & predict
            feats = extract_physics(frame, res, h_meters)
            if feats is not None:
                # feats: [area_m2, sh_w_m, hip_w_m, torso_l_m, height_m]
                if bundle:
                    pred_w_kg = predict_weight(bundle, feats)
                    bmi = pred_w_kg / (h_meters ** 2)
                    
                    # Display HUD
                    cv2.rectangle(display, (10, 10), (350, 140), (20, 20, 40), -1)
                    cv2.putText(display, "V8 Physical Model", (20, 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
                    
                    cv2.putText(display, f"True Height: {h_meters:.2f} m", (20, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                    cv2.putText(display, f"Est. Weight: {pred_w_kg:.1f} kg", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(display, f"BMI Target:  {bmi:.1f}", (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                else:
                    cv2.putText(display, "[Waiting for V8 Model]", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                 cv2.putText(display, "Align entire body in frame", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
        else:
            cv2.putText(display, "No person detected.", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
        cv2.imshow("V8 Physics Pipeline", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
