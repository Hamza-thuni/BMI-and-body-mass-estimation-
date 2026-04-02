import os
import sys
import cv2
import numpy as np
import torch
import joblib
from torchvision import transforms
from PIL import Image

from deep_features_v3 import EfficientNetFeatureExtractor
from person_detector_yolo_v4 import YOLOPersonDetector
from project_paths import models_dir

# ---- LOAD MODELS (with clear errors) ----
_v4 = models_dir("v4")
scaler_path = str(_v4 / "scaler_v4.joblib")
reg_path = str(_v4 / "deep_krr_v4.joblib")

if not os.path.exists(scaler_path):
    print("ERROR: scaler file not found:", scaler_path)
    sys.exit(1)
if not os.path.exists(reg_path):
    print("ERROR: regressor file not found:", reg_path)
    sys.exit(1)

print("Loading scaler and regressor...")
scaler = joblib.load(scaler_path)
reg    = joblib.load(reg_path)
print("Models loaded successfully.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

deep_model = EfficientNetFeatureExtractor(pretrained=True).to(device)
deep_model.eval()

yolo = YOLOPersonDetector(conf=0.4)

preprocess = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def deep_features_from_frame(bgr):
    # Crop person with YOLO
    cropped = yolo.detect_and_crop(bgr)

    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    img_t = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)

    with torch.no_grad():
        deep = deep_model(img_t).cpu().numpy()

    deep_s = scaler.transform(deep)
    return deep_s, cropped


# ---- OPEN WEBCAM (with debug + fallback) ----
print("Opening webcam index 0...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW for Windows

if not cap.isOpened():
    print("Failed to open webcam 0. Trying webcam 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ ERROR: Could not open any webcam (index 0 or 1).")
    sys.exit(1)

print("✅ Webcam opened successfully.")
print("Live BMI (YOLOv8s + EfficientNet v4). Press 'q' to quit.")

# ---- MAIN LOOP ----
while True:
    ret, frame = cap.read()
    if not ret:
        print("WARNING: Failed to read frame from webcam.")
        continue

    frame_small = cv2.resize(frame, (640, 480))

    try:
        X, cropped = deep_features_from_frame(frame_small)
        bmi = reg.predict(X)[0]
        text = f"BMI: {bmi:.2f}"
    except Exception as e:
        text = "BMI: ---"
        cropped = frame_small
        print("Error during prediction:", repr(e))

    disp = cv2.resize(cropped, (320, 480))
    cv2.putText(disp, text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Live BMI (YOLOv8s v4)", disp)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Demo closed cleanly.")
