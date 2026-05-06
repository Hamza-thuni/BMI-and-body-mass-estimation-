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
    if not landmarker_result.pose_landmarks or not landmarker_result.segmentation_masks:
        return None, False

    h, w = image.shape[:2]
    landmarks = landmarker_result.pose_landmarks[0]

    # Check edges
    min_y_norm = min(lm.y for lm in landmarks)
    max_y_norm = max(lm.y for lm in landmarks)
    is_cut_off = (min_y_norm < 0.01 or max_y_norm > 0.99)

    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

    L_SH, R_SH = 11, 12
    L_HIP, R_HIP = 23, 24

    sh_w_px  = np.linalg.norm(pts[L_SH] - pts[R_SH])
    hip_w_px = np.linalg.norm(pts[L_HIP] - pts[R_HIP])
    mid_sh   = (pts[L_SH] + pts[R_SH]) / 2.0
    mid_hip  = (pts[L_HIP] + pts[R_HIP]) / 2.0
    torso_l_px = np.linalg.norm(mid_sh - mid_hip)

    # Restore original robust pixel_height computation (domain shift fix)
    min_y = pts[:, 1].min()
    max_y = pts[:, 1].max()
    pixel_height = max_y - min_y

    if pixel_height < 50:
        return None, is_cut_off

    true_h_m     = pixel_height / px_per_m
    px2_per_m2   = px_per_m ** 2

    mask = landmarker_result.segmentation_masks[0].numpy_view()
    mask_pixels = np.sum(mask > 0.5)
    area_m2  = mask_pixels / px2_per_m2
    sh_w_m   = sh_w_px   / px_per_m
    hip_w_m  = hip_w_px  / px_per_m
    torso_l_m = torso_l_px / px_per_m

    return np.array([area_m2, sh_w_m, hip_w_m, torso_l_m, true_h_m], dtype=np.float32), is_cut_off


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

    # Automated Height Scale
    print("\n[CALIBRATION] Using ArUco markers (ID 0 & ID 1) for automated scale.")
    last_px_per_m = None
    last_h_meters = 0.0

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

        # 1. Detect ArUco Scale
        corners, ids = detect_aruco(frame)
        if ids is not None and 0 in ids and 1 in ids:
            idx0 = np.where(ids == 0)[0][0]
            idx1 = np.where(ids == 1)[0][0]
            c0_corners = corners[idx0][0]
            c1_corners = corners[idx1][0]
            c0 = c0_corners.mean(axis=0)
            c1 = c1_corners.mean(axis=0)
            
            focal_length = w_frame
            cam_mat = np.array([
                [focal_length, 0, w_frame / 2],
                [0, focal_length, h_frame / 2],
                [0, 0, 1]
            ], dtype=np.float32)
            dist_coeffs = np.zeros((4, 1), dtype=np.float32)
            MARKER_SIZE_M = 0.05
            obj_points = np.array([
                [-MARKER_SIZE_M/2,  MARKER_SIZE_M/2, 0],
                [ MARKER_SIZE_M/2,  MARKER_SIZE_M/2, 0],
                [ MARKER_SIZE_M/2, -MARKER_SIZE_M/2, 0],
                [-MARKER_SIZE_M/2, -MARKER_SIZE_M/2, 0]
            ], dtype=np.float32)
            
            _, _, tvec0 = cv2.solvePnP(obj_points, c0_corners, cam_mat, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            _, _, tvec1 = cv2.solvePnP(obj_points, c1_corners, cam_mat, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            
            DEPTH_OFFSET_M = 0.50
            z_wall = (tvec0[2][0] + tvec1[2][0]) / 2.0
            
            if z_wall > 0:
                z_person = z_wall - DEPTH_OFFSET_M
                if z_person > 0:
                    last_px_per_m = focal_length / z_person
            
            # Draw scale line
            cv2.line(display, (int(c0[0]), int(c0[1])), (int(c1[0]), int(c1[1])), (255, 0, 255), 2)
            if last_px_per_m is not None:
                cv2.putText(display, f"Wall Depth: {z_wall:.2f}m | Scale: {last_px_per_m:.1f} px/m", (10, h_frame - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        else:
            if last_px_per_m is None:
                cv2.putText(display, "Waiting for ArUco Markers (0 & 1)...", (10, h_frame - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display, "Using Last Known Scale", (10, h_frame - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # MediaPipe expects RGB
        rgb_cont = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_cont)

        try:
            res = landmarker.detect(mp_image)
        except Exception:
            res = None

        if last_px_per_m is not None and res and res.pose_landmarks:
            landmarks = res.pose_landmarks[0]
            for lm in landmarks:
                cx, cy = int(lm.x * w_frame), int(lm.y * h_frame)
                cv2.circle(display, (cx, cy), 3, (0, 255, 0), -1)

            phys_feats, is_cut_off = extract_physics(frame, res, last_px_per_m)

            if phys_feats is not None:
                try:
                    pred_w = predict_weight(bundle, phys_feats)
                    weight_hist.append(pred_w)
                except Exception as exc:
                    print(f"[prediction error] {exc}")
                    
                last_h_meters = phys_feats[4]

            if weight_hist:
                smooth_w = float(np.mean(weight_hist))
                bmi = smooth_w / (last_h_meters ** 2) if last_h_meters > 0 else 0

                # HUD
                cv2.rectangle(display, (10, 10), (370, 165), (20, 20, 40), -1)
                cv2.putText(display, "V9 Physics — Pi4", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 255), 2)
                cv2.putText(display, f"Height : {last_h_meters:.2f} m", (20, 65),
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
                            
                # Warning if cut off
                if is_cut_off:
                    cv2.putText(display, "WARNING: Move back! Body cut off", (w_frame // 2 - 150, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display, "Analyzing...", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        else:
            if last_px_per_m is not None:
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
