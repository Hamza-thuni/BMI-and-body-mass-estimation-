"""
live_bmi_demo_pi.py
-------------------
V9 Physics-Only Demo — Raspberry Pi 4 + Camera Rev 4 (IMX708).

Uses picamera2 for camera capture instead of cv2.VideoCapture.
Runs in PHYSICS-ONLY mode (5 features) — no EfficientNet on the Pi.

Controls:
  q - quit

Usage:
    cd ~/fyp/src
    source ../bmi_env/bin/activate
    python live_bmi_demo_pi.py
"""

import cv2
import numpy as np
import sys
import os
import mediapipe as mp
import joblib
from collections import deque
from picamera2 import Picamera2

from project_paths import models_dir

# ---------------------------------------------------------------------------
# Load V9 Model (physics-only inference path)
# ---------------------------------------------------------------------------
def load_v9_model():
    path = models_dir("v9") / "ensemble_v9_hybrid.joblib"
    if not path.exists():
        print(f"\n[ERROR] V9 Model not found at: {path}")
        print("Transfer the model file first. See deployment guide.")
        return None
    bundle = joblib.load(str(path))
    ftype = bundle.get("feature_type", "physical_v8")
    print(f"  Model loaded. Feature type: {ftype}")
    return bundle


# ---------------------------------------------------------------------------
# Setup MediaPipe Pose Landmarker
# ---------------------------------------------------------------------------
def create_pose_landmarker():
    # Task model is in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "pose_landmarker_heavy.task")

    if not os.path.exists(model_path):
        print(f"[ERROR] MediaPipe task model not found at: {model_path}")
        print("Transfer pose_landmarker_heavy.task to ~/fyp/src/")
        sys.exit(1)

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
# Physics feature extraction (torso-ratio method — robust to partial body)
# ---------------------------------------------------------------------------
def extract_physics(image, landmarker_result, true_h_m):
    if not landmarker_result.pose_landmarks or not landmarker_result.segmentation_masks:
        return None

    h, w = image.shape[:2]
    landmarks = landmarker_result.pose_landmarks[0]
    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

    L_SH, R_SH = 11, 12
    L_HIP, R_HIP = 23, 24

    sh_w_px  = np.linalg.norm(pts[L_SH] - pts[R_SH])
    hip_w_px = np.linalg.norm(pts[L_HIP] - pts[R_HIP])
    mid_sh   = (pts[L_SH] + pts[R_SH]) / 2.0
    mid_hip  = (pts[L_HIP] + pts[R_HIP]) / 2.0
    torso_l_px = np.linalg.norm(mid_sh - mid_hip)

    if torso_l_px < 20:
        return None

    # Torso-ratio scaling (robust to feet/head being cut off)
    pixel_height = torso_l_px / 0.315
    px_per_m     = pixel_height / true_h_m
    px2_per_m2   = px_per_m ** 2

    mask = landmarker_result.segmentation_masks[0].numpy_view()
    mask_pixels = np.sum(mask > 0.5)
    area_m2  = mask_pixels / px2_per_m2
    sh_w_m   = sh_w_px   / px_per_m
    hip_w_m  = hip_w_px  / px_per_m
    torso_l_m = torso_l_px / px_per_m

    return np.array([area_m2, sh_w_m, hip_w_m, torso_l_m, true_h_m], dtype=np.float32)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_weight(bundle, features):
    X = features.reshape(1, -1)
    if "scaler" in bundle and bundle["scaler"] is not None:
        Xp = bundle["scaler"].transform(X)
    else:
        Xp = X

    w_ridge = bundle["ridge"].predict(Xp)[0]
    if bundle.get("xgb") is not None:
        w_xgb = bundle["xgb"].predict(Xp)[0]
        pred = (w_ridge * 0.5) + (w_xgb * 0.5)
    else:
        pred = w_ridge

    return float(np.clip(pred, 30.0, 250.0))


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 50)
    print(" V9 Physics Demo — Raspberry Pi 4")
    print("=" * 50)

    # Height input
    print("\n[CALIBRATION] Enter your height.")
    while True:
        try:
            val = input("Enter height in CENTIMETERS (e.g. 175): ")
            h_meters = float(val) / 100.0
            if 1.0 < h_meters < 2.5:
                break
            else:
                print("Must be between 100 and 250 cm.")
        except ValueError:
            print("Invalid input.")

    print(f"\n  Height locked to {h_meters:.2f} m.")

    # Load model
    bundle = load_v9_model()
    if bundle is None:
        return

    # Setup MediaPipe
    print("\nLoading MediaPipe Pose Landmarker...")
    landmarker = create_pose_landmarker()
    print("  MediaPipe ready.")

    # Setup picamera2
    print("\nInitializing Pi Camera Rev 4...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    print("  Camera ready. Press q to quit.\n")

    weight_hist = deque(maxlen=20)

    while True:
        # Capture frame from picamera2 (returns RGB numpy array)
        rgb_frame = picam2.capture_array()

        # Convert to BGR for OpenCV display
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        display = frame.copy()
        h_frame, w_frame = frame.shape[:2]

        # MediaPipe expects RGB
        rgb_cont = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_cont)

        try:
            res = landmarker.detect(mp_image)
        except Exception:
            res = None

        if res and res.pose_landmarks:
            landmarks = res.pose_landmarks[0]
            for lm in landmarks:
                cx, cy = int(lm.x * w_frame), int(lm.y * h_frame)
                cv2.circle(display, (cx, cy), 3, (0, 255, 0), -1)

            phys_feats = extract_physics(frame, res, h_meters)

            if phys_feats is not None:
                try:
                    pred_w = predict_weight(bundle, phys_feats)
                    weight_hist.append(pred_w)
                except Exception as exc:
                    print(f"[prediction error] {exc}")

            if weight_hist:
                smooth_w = float(np.mean(weight_hist))
                bmi = smooth_w / (h_meters ** 2)

                # HUD
                cv2.rectangle(display, (10, 10), (370, 165), (20, 20, 40), -1)
                cv2.putText(display, "V9 Physics — Pi4", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 255), 2)
                cv2.putText(display, f"Height : {h_meters:.2f} m", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
                cv2.putText(display, f"Weight : {smooth_w:.1f} kg", (20, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
                cv2.putText(display, f"BMI    : {bmi:.1f}", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 200, 255), 2)

                if bmi < 18.5:
                    cat, col = "Underweight", (0, 210, 255)
                elif bmi < 25:
                    cat, col = "Normal", (0, 200, 50)
                elif bmi < 30:
                    cat, col = "Overweight", (0, 150, 255)
                else:
                    cat, col = "Obese", (30, 30, 220)

                cv2.putText(display, cat, (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            else:
                cv2.putText(display, "Analyzing...", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        else:
            cv2.putText(display, "No person — stand back", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("V9 BMI — Pi4", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()
    print("Demo closed.")


if __name__ == "__main__":
    main()
