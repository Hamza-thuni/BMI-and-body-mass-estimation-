import cv2
import mediapipe as mp
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import joblib

from deep_features_v3 import EfficientNetFeatureExtractor
from project_paths import models_dir


# =================== CONFIG ===================
_m4 = models_dir("v4")
SCALER_PATH = str(_m4 / "scaler_v4.joblib")
REG_PATH = str(_m4 / "deep_krr_v4.joblib")

IMG_SIZE = 320     # for EfficientNet
DISPLAY_SIZE = (360, 640)
# ==============================================


# Load BMI model
scaler = joblib.load(SCALER_PATH)
reg = joblib.load(REG_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

deep_model = EfficientNetFeatureExtractor(pretrained=True).to(device)
deep_model.eval()

preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------- MEDIAPIPE SETUP ----------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ---------------- FEATURE EXTRACTION ----------------
def extract_deep_features(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    t = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = deep_model(t).cpu().numpy()
    return feat


# ---------------- MAIN LOOP ----------------
cap = cv2.VideoCapture(0)

print("\n🔥 Live BMI Prediction (MediaPipe Pose Version)")
print("Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Run mediapipe pose
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        # Draw skeleton like screenshot
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 140, 0), thickness=3)
        )

        # Use bounding box from pose landmarks
        pts = np.array([
            (int(lm.x * w), int(lm.y * h))
            for lm in results.pose_landmarks.landmark
            if lm.visibility > 0.6
        ])

        if len(pts) > 0:
            x1 = max(0, np.min(pts[:, 0]) - 20)
            y1 = max(0, np.min(pts[:, 1]) - 20)
            x2 = min(w, np.max(pts[:, 0]) + 20)
            y2 = min(h, np.max(pts[:, 1]) + 20)

            crop = frame[y1:y2, x1:x2]

            if crop.size > 0:
                feat = extract_deep_features(crop)
                feat_scaled = scaler.transform(feat)
                pred_bmi = float(reg.predict(feat_scaled)[0])

                cv2.putText(frame, f"BMI: {pred_bmi:.2f}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "No pose detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Live BMI Pose Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
