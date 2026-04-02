import os
import sys
import cv2
import numpy as np
import torch
import joblib
from torchvision import transforms
from PIL import Image
from collections import deque

from ultralytics import YOLO
from deep_features_v3 import EfficientNetFeatureExtractor
from project_paths import models_dir, yolo_weights


# ================== CONFIG ==================
YOLO_WEIGHTS = yolo_weights("yolov8n.pt")
YOLO_CONF = 0.35              # detection confidence
YOLO_PERSON_CLASS = 0         # COCO class id for "person"

FRAME_RESIZE_WIDTH = 640      # for YOLO input
YOLO_FRAME_INTERVAL = 5       # run YOLO every N frames

BOX_SMOOTH_ALPHA = 0.25       # 0..1 (higher = more following, lower = more smoothing)

BMI_WINDOW_SIZE = 30          # sliding window for averaging BMI

_m4 = models_dir("v4")
SCALER_PATH = str(_m4 / "scaler_v4.joblib")
REG_PATH = str(_m4 / "deep_krr_v4.joblib")
# ============================================


def debug_print(*args):
    print("[DEBUG]", *args)


# ---------- BOX UTILITIES ----------

def clamp(val, low, high):
    return max(low, min(high, val))


def normalize_box(box, frame_w, frame_h, margin=0.15):
    """
    Make the box more consistent:
    - Expand by margin
    - Make it roughly square (so body scale is similar)
    - Clamp to image bounds
    """
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

    # Expand box by margin
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0

    w_exp = w * (1.0 + margin * 2.0)
    h_exp = h * (1.0 + margin * 2.0)

    # Make box square-ish: side = max(w_exp, h_exp)
    side = max(w_exp, h_exp)
    x1_new = cx - side / 2.0
    y1_new = cy - side / 2.0
    x2_new = cx + side / 2.0
    y2_new = cy + side / 2.0

    # Clamp to image bounds
    x1_new = clamp(int(round(x1_new)), 0, frame_w - 1)
    y1_new = clamp(int(round(y1_new)), 0, frame_h - 1)
    x2_new = clamp(int(round(x2_new)), x1_new + 1, frame_w)
    y2_new = clamp(int(round(y2_new)), y1_new + 1, frame_h)

    return np.array([x1_new, y1_new, x2_new, y2_new], dtype=np.int32)


def smooth_box(new_box, prev_box, alpha):
    if prev_box is None:
        return new_box
    return alpha * new_box + (1.0 - alpha) * prev_box


# ---------- DEEP MODEL & REGRESSOR ----------

def build_deep_model(device):
    model = EfficientNetFeatureExtractor(pretrained=True).to(device)
    model.eval()
    return model


def build_preprocess():
    return transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def extract_deep_features(bgr_crop, model, preprocess, device):
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    t = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(t).cpu().numpy()  # shape (1, F)

    return feat  # (1, F)


# ---------- YOLO PERSON DETECTION ----------

def detect_person_box_yolo(yolo_model, frame_bgr_small, conf_thres, person_class=0):
    """
    Run YOLO on a resized frame and return the largest person box in that resized frame coordinates.
    Return None if no person.
    """
    # YOLO expects RGB
    results = yolo_model.predict(frame_bgr_small[:, :, ::-1], conf=conf_thres, verbose=False)

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)

    # Filter person boxes
    person_boxes = [xyxy[i] for i in range(len(xyxy)) if cls_ids[i] == person_class]
    if not person_boxes:
        return None

    # pick largest area box
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in person_boxes]
    best_box = person_boxes[int(np.argmax(areas))]
    return best_box  # in resized frame coords


# ---------- MAIN LOOP ----------

def main():
    # Debug info
    print("Python exe:", sys.executable)
    print("Working directory:", os.getcwd())

    # Load scaler & regressor
    if not os.path.exists(SCALER_PATH):
        print("ERROR: scaler file not found:", SCALER_PATH)
        return
    if not os.path.exists(REG_PATH):
        print("ERROR: regressor file not found:", REG_PATH)
        return

    scaler = joblib.load(SCALER_PATH)
    reg = joblib.load(REG_PATH)
    print("✅ Loaded scaler and regressor.")

    # Device & deep model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    deep_model = build_deep_model(device)
    preprocess = build_preprocess()
    print("✅ Deep model ready.")

    # YOLO
    print("Loading YOLO model:", YOLO_WEIGHTS)
    yolo = YOLO(YOLO_WEIGHTS)
    print("✅ YOLO model loaded.")

    # Webcam
    print("Opening webcam 0...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Failed to open webcam 0. Trying webcam 1...")
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ ERROR: Could not open any webcam.")
        return

    print("✅ Webcam opened.")
    print("Instructions:")
    print("- Stand 2–3 meters away, full body visible.")
    print("- Stand straight, arms relaxed at sides.")
    print("- Avoid very loose jackets if possible.")
    print("- Press 'q' to quit.\n")

    frame_count = 0
    last_box_fullres = None   # smoothed box in original frame coordinates
    last_yolo_box_small = None

    bmi_buffer = deque(maxlen=BMI_WINDOW_SIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Failed to read frame from webcam.")
            continue

        frame_h, frame_w = frame.shape[:2]

        # Prepare resized frame for YOLO (keep aspect ratio)
        scale = FRAME_RESIZE_WIDTH / float(frame_w)
        new_w = FRAME_RESIZE_WIDTH
        new_h = int(round(frame_h * scale))
        frame_small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # YOLO only every N frames
        if frame_count % YOLO_FRAME_INTERVAL == 0:
            yolo_box_small = detect_person_box_yolo(
                yolo_model=yolo,
                frame_bgr_small=frame_small,
                conf_thres=YOLO_CONF,
                person_class=YOLO_PERSON_CLASS
            )
            if yolo_box_small is not None:
                last_yolo_box_small = yolo_box_small

                # map box from small coords to full-res frame
                x1s, y1s, x2s, y2s = yolo_box_small
                x1f = x1s / scale
                y1f = y1s / scale
                x2f = x2s / scale
                y2f = y2s / scale
                new_box_fullres = np.array([x1f, y1f, x2f, y2f], dtype=np.float32)

                # smooth box
                smoothed = smooth_box(new_box_fullres, last_box_fullres, BOX_SMOOTH_ALPHA)
                # normalize (expand + square + clamp)
                norm_box = normalize_box(smoothed, frame_w, frame_h, margin=0.15)
                last_box_fullres = norm_box
            # else keep previous last_box_fullres
        # else, we just re-use last_box_fullres

        # If we still have no box, show info & continue
        if last_box_fullres is None:
            msg = "No person detected. Stand fully in front of camera."
            info_frame = frame.copy()
            cv2.putText(info_frame, msg, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("BMI v5 (YOLO + EfficientNet)", info_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            continue

        x1, y1, x2, y2 = last_box_fullres.astype(int)
        # Guard against bad crops
        if x2 <= x1 or y2 <= y1:
            frame_count += 1
            continue

        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            frame_count += 1
            continue

        # Deep features + BMI prediction
        try:
            feat = extract_deep_features(crop, deep_model, preprocess, device)  # (1, F)
            feat_scaled = scaler.transform(feat)
            bmi_pred = float(reg.predict(feat_scaled)[0])
        except Exception as e:
            print("Prediction error:", repr(e))
            frame_count += 1
            continue

        # Add BMI to buffer and compute stable mean
        bmi_buffer.append(bmi_pred)
        stable_bmi = float(np.mean(bmi_buffer))

        # Prepare display crop
        disp = cv2.resize(crop, (320, 480))
        text1 = f"Stable BMI: {stable_bmi:.2f}"
        text2 = f"Instant BMI: {bmi_pred:.2f}"

        cv2.putText(disp, text1, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(disp, text2, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # Optional: draw box on original frame (for debugging)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"BMI~{stable_bmi:.1f}", (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show cropped view (main) and small original (debug)
        combined = disp
        cv2.imshow("BMI v5 (YOLO + EfficientNet)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Closed cleanly.")


if __name__ == "__main__":
    main()
