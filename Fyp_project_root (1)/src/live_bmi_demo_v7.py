"""
live_bmi_demo_v7.py
-------------------
V7 Live BMI Demo — Segmentation + Pose + EfficientNet + Ensemble

New features vs V6 demo:
  • Background removed via MediaPipe segmentation mask before EfficientNet
  • 24-D rich pose features fed to ensemble regressor
  • Ensemble prediction (KRR + SVR + XGBoost) for more stable estimates
  • Interactive segmentation overlay toggle ('s' key)
  • Colour-coded BMI category bar in the info panel
  • Graceful model fallback: V7 hybrid → V7 deep-only → V6

Controls:
  q  — quit
  s  — toggle green segmentation mask overlay

Usage:
    cd src
    python live_bmi_demo_v7.py
"""

import cv2
import numpy as np
import joblib
import mediapipe as mp
from collections import deque

from person_detector_yolo_v4 import YOLOPersonDetector
from v7_segmentation import apply_mask_to_crop
from v7_pose_features import extract_rich_pose_features
from v6_utils_pose_effnet import extract_deep_features   # EfficientNet-B3 loader
from project_paths import models_dir

try:
    mp_draw = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
except AttributeError:
    import sys
    print("\n[ENVIRONMENT ERROR] Your 'mediapipe' installation is corrupted or missing the 'solutions' module.")
    print(f"Python is loading mediapipe from: {getattr(mp, '__file__', 'Unknown')}")
    print("To fix this, please run the following command in this terminal:")
    print("    pip uninstall -y mediapipe")
    print("    pip install mediapipe\n")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Model loading — try V7 hybrid → V7 deep-only → V6 fallback
# ---------------------------------------------------------------------------

def _load_bundle():
    """Return (bundle_dict, mode_label)."""
    m7 = models_dir("v7")

    hybrid_path   = m7 / "ensemble_v7_hybrid.joblib"
    deeponly_path = m7 / "ensemble_v7_deeponly.joblib"

    if hybrid_path.exists():
        b = joblib.load(str(hybrid_path))
        print("✅ Loaded V7 hybrid ensemble (seg + deep + pose).")
        return b, "V7-Hybrid"

    if deeponly_path.exists():
        b = joblib.load(str(deeponly_path))
        print("✅ Loaded V7 deep-only ensemble (no V7 extraction yet).")
        return b, "V7-DeepOnly"

    # V6 fallback
    m6 = models_dir("v6")
    b = {
        "scaler":       joblib.load(str(m6 / "scaler_v6.joblib")),
        "model":        joblib.load(str(m6 / "krr_hybrid_v6.joblib")),
        "feature_type": "v6_fallback",
        "has_xgb":      False,
    }
    print("⚠️  V7 model not found — using V6 fallback.")
    return b, "V6-Fallback"


BUNDLE, MODEL_LABEL = _load_bundle()
FEATURE_TYPE = BUNDLE.get("feature_type", "v6_fallback")

# ---------------------------------------------------------------------------
# BMI categories
# ---------------------------------------------------------------------------

_CATEGORIES = [
    (0,    18.5, "Underweight", (0,   210, 255)),
    (18.5, 25.0, "Normal",      (0,   200,  50)),
    (25.0, 30.0, "Overweight",  (0,   150, 255)),
    (30.0, 999,  "Obese",       (30,   30, 220)),
]

def bmi_label(bmi: float):
    for lo, hi, label, colour in _CATEGORIES:
        if lo <= bmi < hi:
            return label, colour
    return "Obese", (30, 30, 220)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _ensemble_predict(feat: np.ndarray) -> float:
    """Run ensemble or single-model prediction, clamped to plausible BMI range."""
    ftype = BUNDLE.get("feature_type", "v6_fallback")

    if ftype == "v6_fallback":
        Xs  = BUNDLE["scaler"].transform(feat.reshape(1, -1))
        raw = float(BUNDLE["model"].predict(Xs)[0])
    else:
        # V7 ensemble path
        Xs = BUNDLE["pca"].transform(
             BUNDLE["scaler"].transform(feat.reshape(1, -1)))
        preds = [BUNDLE["krr"].predict(Xs)[0],
                 BUNDLE["svr"].predict(Xs)[0]]
        if BUNDLE.get("has_xgb") and "xgb" in BUNDLE:
            preds.append(BUNDLE["xgb"].predict(Xs)[0])
        raw = float(np.mean(preds))

    # Safety clamp — real BMI is always in [10, 60]
    clamped = float(np.clip(raw, 10.0, 60.0))
    if abs(raw - clamped) > 0.5:
        print(f"[BMI clamp] raw={raw:.1f} → clamped={clamped:.1f}")
    return clamped


def _build_feature(deep_feat, coords):
    """Concatenate features according to model type."""
    ftype = BUNDLE.get("feature_type", "v6_fallback")
    if ftype == "hybrid_v7":
        pose_feat = extract_rich_pose_features(coords)          # 24-D
        return np.concatenate([deep_feat, pose_feat])
    if ftype == "v6_fallback":
        from v6_utils_pose_effnet import extract_pose_features  # 16-D
        pose_feat = extract_pose_features(coords)
        return np.concatenate([deep_feat, pose_feat])
    # deep-only (V7 deep-only or V4 style) — no pose
    return deep_feat


# ---------------------------------------------------------------------------
# Info panel drawing
# ---------------------------------------------------------------------------

_PANEL_W = 290
_BAR_BMI_MIN, _BAR_BMI_MAX = 15.0, 40.0


def _draw_panel(bmi_smooth: float, bmi_inst: float, model_lbl: str,
                panel_h: int) -> np.ndarray:
    panel = np.zeros((panel_h, _PANEL_W, 3), dtype=np.uint8)
    panel[:] = (18, 18, 28)              # dark navy background

    cat_label, cat_col = bmi_label(bmi_smooth)

    # Title
    cv2.putText(panel, f"BMI  [{model_lbl}]", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (160, 160, 180), 1)

    # Large BMI value
    cv2.putText(panel, f"{bmi_smooth:.2f}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, cat_col, 3)

    # Category
    cv2.putText(panel, cat_label, (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, cat_col, 2)

    # Instant reading
    cv2.putText(panel, "Instant:", (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 130), 1)
    cv2.putText(panel, f"{bmi_inst:.2f}", (80, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1)

    # BMI colour bar
    bar_x, bar_y, bar_h = 10, 210, 18
    bar_w = _PANEL_W - 20
    # Background track
    cv2.rectangle(panel, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (50, 50, 60), -1)
    # Gradient segments (Underweight / Normal / Overweight / Obese)
    seg_cols = [(0, 210, 255), (0, 200, 50), (0, 150, 255), (30, 30, 220)]
    seg_lo   = [15.0, 18.5, 25.0, 30.0]
    seg_hi   = [18.5, 25.0, 30.0, 40.0]
    for lo, hi, col in zip(seg_lo, seg_hi, seg_cols):
        sx = bar_x + int((lo - _BAR_BMI_MIN) / (_BAR_BMI_MAX - _BAR_BMI_MIN) * bar_w)
        ex = bar_x + int((hi - _BAR_BMI_MIN) / (_BAR_BMI_MAX - _BAR_BMI_MIN) * bar_w)
        cv2.rectangle(panel, (sx, bar_y), (ex, bar_y + bar_h), col, -1)
    # Needle
    needle_x = bar_x + int(
        (min(max(bmi_smooth, _BAR_BMI_MIN), _BAR_BMI_MAX) - _BAR_BMI_MIN)
        / (_BAR_BMI_MAX - _BAR_BMI_MIN) * bar_w
    )
    cv2.rectangle(panel, (needle_x - 2, bar_y - 4),
                  (needle_x + 2, bar_y + bar_h + 4), (255, 255, 255), -1)
    # Labels
    cv2.putText(panel, "15", (bar_x, bar_y + bar_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 110), 1)
    cv2.putText(panel, "40", (bar_x + bar_w - 18, bar_y + bar_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 110), 1)

    # Controls hint
    cv2.putText(panel, "'q' quit   's' seg mask",
                (10, panel_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 90), 1)

    return panel


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print(f"\nV7 Live BMI Demo  [{MODEL_LABEL}]")
    print("Stand 2–3 m from camera, full body visible.")
    print("Press 'q' to quit, 's' to toggle segmentation overlay.\n")

    yolo     = YOLOPersonDetector(conf=0.35)
    bmi_hist = deque(maxlen=30)
    show_seg = False

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  Could not open webcam.")
        return

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as pose_tracker:

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            h, w = frame.shape[:2]
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res  = pose_tracker.process(rgb)
            display = frame.copy()

            coords = None
            seg_mask = None

            if res.pose_landmarks:
                # Draw skeleton overlay
                mp_draw.draw_landmarks(
                    display, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                coords = np.array(
                    [[lm.x * w, lm.y * h, lm.z]
                     for lm in res.pose_landmarks.landmark],
                    dtype=np.float32,
                )

                if res.segmentation_mask is not None:
                    seg_mask = res.segmentation_mask  # float32 (H,W)

                    if show_seg:
                        green = np.zeros_like(display)
                        green[:, :, 1] = 200
                        alpha_map = (seg_mask > 0.5).astype(np.float32)[:, :, None] * 0.35
                        display = (display * (1 - alpha_map)
                                   + green * alpha_map).astype(np.uint8)
            else:
                # YOLO fallback for detection
                roi = yolo.detect_person_roi(frame)
                if roi is None:
                    cv2.putText(display, "No pose detected — step back / centre",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 220), 2)
                else:
                    x1r, y1r, x2r, y2r = roi
                    cv2.rectangle(display, (x1r, y1r), (x2r, y2r), (0, 120, 255), 2)
                    cv2.putText(display, "YOLO: pose not found",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 120, 255), 2)

            # ---- Feature extraction & prediction ----
            if coords is not None:
                xs, ys = coords[:, 0], coords[:, 1]
                mg     = 0.20
                bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
                x1 = max(0,  int(xs.min() - bw * mg / 2))
                x2 = min(w,  int(xs.max() + bw * mg / 2))
                y1 = max(0,  int(ys.min() - bh * mg / 2))
                y2 = min(h,  int(ys.max() + bh * mg / 2))

                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                crop = frame[y1:y2, x1:x2]

                if crop.size > 0:
                    # Apply segmentation mask to crop if available
                    if seg_mask is not None:
                        crop = apply_mask_to_crop(crop, seg_mask, y1, x1)

                    try:
                        deep_feat = extract_deep_features(crop)
                        feat      = _build_feature(deep_feat, coords)
                        bmi       = _ensemble_predict(feat)
                        bmi_hist.append(bmi)
                    except Exception as exc:
                        print(f"[prediction error] {exc}")

            # ---- Draw info panel ----
            if bmi_hist:
                bmi_smooth = float(np.mean(bmi_hist))
                bmi_inst   = float(bmi_hist[-1])
                panel      = _draw_panel(bmi_smooth, bmi_inst, MODEL_LABEL, h)
                disp_frame = cv2.resize(display, (w, h))
                combo      = np.hstack([panel, disp_frame])
            else:
                cv2.putText(display, "Waiting for stable pose ...",
                            (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (180, 180, 0), 2)
                combo = display

            cv2.imshow("BMI V7 — Seg + Pose + EffNet + Ensemble", combo)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                show_seg = not show_seg
                print(f"Segmentation overlay: {'ON' if show_seg else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("Demo closed.")


if __name__ == "__main__":
    main()
