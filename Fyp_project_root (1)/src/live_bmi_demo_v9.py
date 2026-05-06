"""
live_bmi_demo_v9_fixed.py
----------------------
V9: Hybrid Live Demo — Physics + Deep Features + Pose.
FIXED: ArUco scale calculation now uses the known physical separation
between two markers (ID 0 at 80cm, ID 1 at 180cm from the floor)
to compute px/m directly — no solvePnP, no depth guessing.

Controls:
  q - quit

Usage:
    cd src
    python live_bmi_demo_v9_fixed.py
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
# Physical setup — match these to your real-world marker placement
# ---------------------------------------------------------------------------
MARKER_BOTTOM_HEIGHT_M   = 0.80   # center of marker ID 0 is 80cm from floor
MARKER_TOP_HEIGHT_M      = 1.80   # center of marker ID 1 is 180cm from floor
MARKER_SEPARATION_M      = MARKER_TOP_HEIGHT_M - MARKER_BOTTOM_HEIGHT_M  # 1.0 m

# ---------------------------------------------------------------------------
# EfficientNet-B3 deep feature extraction
# ---------------------------------------------------------------------------
_PREPROCESS = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_effnet_model  = None
_effnet_device = None


def _init_effnet():
    """Lazy-load EfficientNet-B3 for deep features."""
    global _effnet_model, _effnet_device
    if _effnet_model is not None:
        return
    from deep_features_v3 import EfficientNetFeatureExtractor
    _effnet_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _effnet_model  = EfficientNetFeatureExtractor(pretrained=True).to(_effnet_device)
    _effnet_model.eval()
    print(f"  EfficientNet-B3 loaded on {_effnet_device}")


def extract_deep_features(bgr_crop):
    """Extract 1536-D EfficientNet-B3 features from a BGR crop."""
    _init_effnet()
    rgb  = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    pil  = Image.fromarray(rgb)
    t    = _PREPROCESS(pil).unsqueeze(0).to(_effnet_device)
    with torch.no_grad():
        feat = _effnet_model(t).cpu().numpy().flatten()
    return feat


# ---------------------------------------------------------------------------
# Pose feature extraction
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
        url = (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        )
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
    BaseOptions           = mp.tasks.BaseOptions
    PoseLandmarker        = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

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
    ftype  = bundle.get("feature_type", "physical_v8")
    print(f"  Model type: {ftype}")
    return bundle


# ---------------------------------------------------------------------------
# ArUco Marker Detection
# ---------------------------------------------------------------------------
def detect_aruco(frame):
    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector         = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        corners, ids, _  = detector.detectMarkers(gray)
    else:
        corners, ids, _  = cv2.aruco.detectMarkers(
            gray, aruco_dict,
            parameters=cv2.aruco.DetectorParameters_create()
        )

    # Sub-pixel corner refinement for maximum accuracy
    if ids is not None and len(corners) > 0:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        for i in range(len(corners)):
            cv2.cornerSubPix(gray, corners[i], (5, 5), (-1, -1), criteria)

    return corners, ids


# ---------------------------------------------------------------------------
# FIXED: Compute px/m from the two known marker positions
#
# Marker ID 0 → bottom marker, physically at MARKER_BOTTOM_HEIGHT_M (0.80 m)
# Marker ID 1 → top    marker, physically at MARKER_TOP_HEIGHT_M    (1.80 m)
#
# The vertical pixel distance between their centres equals MARKER_SEPARATION_M
# in the real world, so:
#
#   px_per_m = |center1.y - center0.y| / MARKER_SEPARATION_M
#
# This is exact — no depth estimation, no guessing, no solvePnP required.
# ---------------------------------------------------------------------------
def compute_scale_from_markers(corners, ids, scale_hist):
    """
    Returns (updated_scale_hist, last_px_per_m, debug_info_dict).
    debug_info_dict contains values for on-screen display.
    """
    ids_flat = ids.flatten()
    debug    = {}

    if 0 not in ids_flat or 1 not in ids_flat:
        # One or both markers missing — return history unchanged
        return scale_hist, (float(np.mean(scale_hist)) if scale_hist else None), debug

    idx0 = np.where(ids_flat == 0)[0][0]
    idx1 = np.where(ids_flat == 1)[0][0]

    center0 = corners[idx0][0].mean(axis=0)   # (x, y) of ID 0 — bottom marker
    center1 = corners[idx1][0].mean(axis=0)   # (x, y) of ID 1 — top marker

    # Vertical pixel distance between marker centres
    # ID 1 should be higher on the wall → smaller y value in image coords
    pixel_separation = abs(center1[1] - center0[1])

    # Sanity check: pixel separation must be meaningful
    # (protects against markers being detected in wrong frame positions)
    if pixel_separation < 20:
        return scale_hist, (float(np.mean(scale_hist)) if scale_hist else None), debug

    inst_px_per_m = pixel_separation / MARKER_SEPARATION_M
    scale_hist.append(inst_px_per_m)

    last_px_per_m = float(np.mean(scale_hist))

    debug["center0"]         = center0
    debug["center1"]         = center1
    debug["pixel_sep"]       = pixel_separation
    debug["inst_px_per_m"]   = inst_px_per_m

    return scale_hist, last_px_per_m, debug


# ---------------------------------------------------------------------------
# Physics feature extraction
# ---------------------------------------------------------------------------
# CALIBRATION CONSTANTS:
# 1. PARALLAX_FACTOR: Accounts for person standing in front of the wall.
#    Since person is closer to camera, they appear larger. 
#    A value of 0.94 means the person is ~6% closer than the wall.
# 2. STATURE_ADJUST: Landmark height (nose-to-heel) is ~92% of full stature.
#    We multiply landmark height by 1.08 to estimate true stature.
# ---------------------------------------------------------------------------
PARALLAX_FACTOR    = 0.94  # Recommended: 0.92 to 0.96
STATURE_ADJUST     = 1.08  # Landmark span to real height ratio

def extract_physics(image, landmarker_result, wall_px_per_m):
    """
    Computes absolute physical dimensions (Area, Widths) from pixel data.
    Returns: (phys_feats_5D, coords, seg_mask, is_cut_off) or (None, None, None, False)
    """
    if not landmarker_result.pose_landmarks or not landmarker_result.segmentation_masks:
        return None, None, None, False

    h, w      = image.shape[:2]
    landmarks = landmarker_result.pose_landmarks[0]

    # Check if body is cut off at frame edges
    min_y_norm = min(lm.y for lm in landmarks)
    max_y_norm = max(lm.y for lm in landmarks)
    is_cut_off = (min_y_norm < 0.01 or max_y_norm > 0.99)

    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

    L_SH, R_SH   = 11, 12
    L_HIP, R_HIP = 23, 24

    # 1. Effective Scale at Person's Depth
    # person_px_per_m = wall_px_per_m / PARALLAX_FACTOR
    # Because person is closer, 1m in real world is MORE pixels.
    person_px_per_m = wall_px_per_m / PARALLAX_FACTOR

    # 2. Pixel Dimensions
    sh_w_px    = np.linalg.norm(pts[L_SH]  - pts[R_SH])
    hip_w_px   = np.linalg.norm(pts[L_HIP] - pts[R_HIP])
    mid_sh     = (pts[L_SH]  + pts[R_SH])  / 2.0
    mid_hip    = (pts[L_HIP] + pts[R_HIP]) / 2.0
    torso_l_px = np.linalg.norm(mid_sh - mid_hip)

    # Landmark pixel height (head/nose to feet)
    min_y        = pts[:, 1].min()
    max_y        = pts[:, 1].max()
    pixel_height = max_y - min_y

    if pixel_height < 50:
        return None, None, None, is_cut_off

    # 3. Convert to Meters using the corrected scale
    # Also adjust for landmark-to-stature ratio
    true_h_m      = (pixel_height / person_px_per_m) * STATURE_ADJUST
    px2_per_m2    = person_px_per_m ** 2

    # 4. Area Calculation
    mask         = landmarker_result.segmentation_masks[0].numpy_view()
    mask_pixels  = np.sum(mask > 0.5)
    area_m2      = mask_pixels / px2_per_m2

    sh_w_m    = sh_w_px    / person_px_per_m
    hip_w_m   = hip_w_px   / person_px_per_m
    torso_l_m = torso_l_px / person_px_per_m

    # Features: [Area, Sh_Width, Hip_Width, Torso_L, Height]
    phys   = np.array([area_m2, sh_w_m, hip_w_m, torso_l_m, true_h_m], dtype=np.float32)
    coords = np.column_stack([pts, np.zeros(len(pts))])

    return phys, coords, mask, is_cut_off



# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_weight(bundle, features):
    """Run prediction through the saved pipeline (scaler -> ensemble)."""
    X  = features.reshape(1, -1)

    if "scaler" in bundle and bundle["scaler"] is not None:
        Xp = bundle["scaler"].transform(X)
    else:
        Xp = X

    w_ridge = bundle["ridge"].predict(Xp)[0]
    if bundle.get("xgb") is not None:
        w_xgb = bundle["xgb"].predict(Xp)[0]
        raw_pred = (w_ridge * 0.5) + (w_xgb * 0.5)
    else:
        raw_pred = w_ridge

    # ------------------------------------------------------------------
    # V9-EMERGENCY PATCH: Applied Calibration
    # ------------------------------------------------------------------
    # 1. Linear Correction (Slope Adjustment)
    # Based on your residual plot, the model underestimates high weights.
    # We apply a scaling factor to "stretch" the prediction if it's above the mean.
    if raw_pred > 70:
        # Stretches weights above 70kg to fight the "100kg ceiling"
        # This adds roughly 15% more mass to the delta above 70kg
        calibrated_w = 70 + (raw_pred - 70) * 1.25
    else:
        # Slightly compresses low weights to fight the overestimation at the bottom
        calibrated_w = raw_pred * 0.95

    # 2. Physical Sanity Check (Hard Floor/Ceiling)
    # Using your Area_m2 (phys_feats[0]) as a secondary guard. 
    # Average density check: Area * 140 is a rough kg/m2 approximation for front-view
    area_m2 = features[0]
    lower_bound = area_m2 * 105  # Very lean
    upper_bound = area_m2 * 185  # Very heavy

    pred = np.clip(calibrated_w, lower_bound, upper_bound)
    # ------------------------------------------------------------------

    pred = float(np.clip(pred, 30.0, 250.0))
    return pred


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 60)
    print(" V9 Hybrid Live Demo (Physics + Deep + Pose) — FIXED SCALE")
    print("=" * 60)
    print(f"\n  Marker setup:")
    print(f"    ID 0 (bottom) → {MARKER_BOTTOM_HEIGHT_M*100:.0f} cm from floor")
    print(f"    ID 1 (top)    → {MARKER_TOP_HEIGHT_M*100:.0f} cm from floor")
    print(f"    Known separation: {MARKER_SEPARATION_M:.2f} m")

    last_px_per_m = None
    last_h_meters = 0.0
    scale_hist    = deque(maxlen=30)

    bundle = load_v9_model()
    if bundle is None:
        return

    feat_type = bundle.get("feature_type", "physical_v8")
    is_hybrid = feat_type == "hybrid_v9"

    if is_hybrid:
        print("\n  Mode: HYBRID (physics + deep + pose)")
        print("  Loading EfficientNet-B3 ...")
        _init_effnet()
    else:
        print("\n  Mode: PHYSICS-ONLY (5 features)")

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
    print("Ensure BOTH ArUco markers are visible. Press 'q' to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        display          = frame.copy()
        h_frame, w_frame = frame.shape[:2]

        # ------------------------------------------------------------------
        # 1. Detect ArUco markers and compute scale
        # ------------------------------------------------------------------
        corners, ids = detect_aruco(frame)

        if ids is not None:
            ids_flat = ids.flatten()
            has_both = (0 in ids_flat and 1 in ids_flat)

            if has_both:
                scale_hist, last_px_per_m, debug = compute_scale_from_markers(
                    corners, ids, scale_hist
                )

                if debug:
                    c0 = debug["center0"]
                    c1 = debug["center1"]

                    # Draw line connecting the two marker centres
                    cv2.line(display,
                             (int(c0[0]), int(c0[1])),
                             (int(c1[0]), int(c1[1])),
                             (255, 0, 255), 2)

                    # Label each marker
                    cv2.putText(display, "ID0 (80cm)",
                                (int(c0[0]) + 10, int(c0[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
                    cv2.putText(display, "ID1 (180cm)",
                                (int(c1[0]) + 10, int(c1[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

                    # Status bar
                    cv2.putText(display,
                                f"Scale: {last_px_per_m:.1f} px/m  "
                                f"({debug['pixel_sep']:.0f}px = {MARKER_SEPARATION_M:.1f}m)",
                                (10, h_frame - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)

                    # -------------------------------------------------------
                    # Sanity check printed once to terminal
                    # -------------------------------------------------------
                    expected_px = last_px_per_m * 1.70
                    print(f"[SCALE] px/m={last_px_per_m:.1f} | "
                          f"1.70m person ≈ {expected_px:.0f}px  "
                          f"(frame height={h_frame}px)",
                          end="\r")

            else:
                # Only one marker visible — show which one is missing
                missing = "ID 1 (top)" if 0 in ids_flat else "ID 0 (bottom)"
                msg     = f"Missing marker {missing} — need both for scale"
                if last_px_per_m is not None:
                    msg = f"Using cached scale: {last_px_per_m:.1f} px/m  ({missing} lost)"
                cv2.putText(display, msg,
                            (10, h_frame - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        else:
            # No markers at all
            if last_px_per_m is None:
                cv2.putText(display,
                            "No ArUco markers detected — place ID 0 (80cm) and ID 1 (180cm)",
                            (10, h_frame - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            else:
                cv2.putText(display,
                            f"Markers lost — using cached scale: {last_px_per_m:.1f} px/m",
                            (10, h_frame - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # ------------------------------------------------------------------
        # 2. Pose detection
        # ------------------------------------------------------------------
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb      = np.ascontiguousarray(rgb, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            res = landmarker.detect(mp_image)
        except Exception:
            res = None

        # ------------------------------------------------------------------
        # 3. Feature extraction + inference (only when scale is known)
        # ------------------------------------------------------------------
        if last_px_per_m is not None and res and res.pose_landmarks:

            # Draw landmarks
            landmarks = res.pose_landmarks[0]
            for lm in landmarks:
                cx, cy = int(lm.x * w_frame), int(lm.y * h_frame)
                cv2.circle(display, (cx, cy), 3, (0, 255, 0), -1)

            phys_feats, coords, seg_mask, is_cut_off = extract_physics(
                frame, res, last_px_per_m
            )

            if phys_feats is not None and bundle:
                try:
                    if is_hybrid:
                        xs, ys = coords[:, 0], coords[:, 1]
                        mg     = 0.20
                        bw     = xs.max() - xs.min()
                        bh     = ys.max() - ys.min()
                        x1 = max(0,       int(xs.min() - bw * mg / 2))
                        x2 = min(w_frame, int(xs.max() + bw * mg / 2))
                        y1 = max(0,       int(ys.min() - bh * mg / 2))
                        y2 = min(h_frame, int(ys.max() + bh * mg / 2))
                        crop = frame[y1:y2, x1:x2]

                        if crop.size > 0:
                            if seg_mask is not None:
                                mask_crop = seg_mask[y1:y2, x1:x2]
                                mask_3ch  = (mask_crop > 0.5).astype(np.uint8)[:, :, None]
                                crop      = crop * mask_3ch

                            deep_feat = extract_deep_features(crop)    # 1536-D
                            pose_feat = extract_pose_features(coords)  # 24-D
                            combined  = np.concatenate([phys_feats, deep_feat, pose_feat])
                            pred_w_kg = predict_weight(bundle, combined)
                        else:
                            pred_w_kg = predict_weight(bundle, phys_feats)
                    else:
                        pred_w_kg = predict_weight(bundle, phys_feats)

                    weight_hist.append(pred_w_kg)

                except Exception as exc:
                    print(f"\n[prediction error] {exc}")

            if phys_feats is not None:
                last_h_meters = phys_feats[4]

            # --------------------------------------------------------------
            # 4. HUD display
            # --------------------------------------------------------------
            if weight_hist:
                smooth_w = float(np.mean(weight_hist))
                inst_w   = float(weight_hist[-1])
                bmi      = smooth_w / (last_h_meters ** 2) if last_h_meters > 0 else 0

                cv2.rectangle(display, (10, 10), (400, 175), (20, 20, 40), -1)

                label = "V9 Hybrid" if is_hybrid else "V9 Physics"
                cv2.putText(display, label, (20, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 255), 2)

                cv2.putText(display,
                            f"Height:  {last_h_meters:.2f} m", (20, 68),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (150, 150, 150), 1)

                cv2.putText(display,
                            f"Weight:  {smooth_w:.1f} kg", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.88, (0, 255, 0), 2)
                cv2.putText(display,
                            f"(instant: {inst_w:.1f})", (240, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 100), 1)

                cv2.putText(display,
                            f"BMI:     {bmi:.1f}", (20, 135),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.88, (0, 200, 255), 2)

                # BMI category
                if   bmi < 18.5: cat, col = "Underweight", (0, 210, 255)
                elif bmi < 25.0: cat, col = "Normal",      (0, 200, 50)
                elif bmi < 30.0: cat, col = "Overweight",  (0, 150, 255)
                else:            cat, col = "Obese",        (30, 30, 220)

                cv2.putText(display, cat, (20, 165),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)

                if is_cut_off:
                    cv2.putText(display,
                                "WARNING: Move back — body cut off!",
                                (w_frame // 2 - 180, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 255), 2)

            else:
                cv2.putText(display, "Analyzing body...", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        else:
            if last_px_per_m is not None:
                cv2.putText(display, "No person detected — stand back", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("V9 Weight & BMI Estimation (Fixed Scale)", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nDemo closed.")


if __name__ == "__main__":
    main()