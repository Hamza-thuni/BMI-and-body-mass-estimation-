import cv2
import numpy as np
import torch
import joblib
from torchvision import transforms
from PIL import Image

from deep_features import ResNetFeatureExtractor
from anthropometric_features import extract_mixed_features
from project_paths import models_dir


# --------------------------
# 1. Load Models
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Deep feature extractor (same architecture used during training)
deep_model = ResNetFeatureExtractor("resnet50", pretrained=True).to(device)
deep_model.eval()

# Regression model
reg = joblib.load(str(models_dir("v1") / "krr_mixed_features.joblib"))


# --------------------------
# 2. Preprocessing function
# --------------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def extract_features_from_frame(frame_bgr):
    # Deep Features
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    img_t = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        deep_feat = deep_model(img_t).cpu().numpy().flatten()

    # Anthropometric Features
    antro_feat = extract_mixed_features(frame_bgr)  # shape (13,)

    # Concatenate
    full_feat = np.concatenate([deep_feat, antro_feat], axis=0).reshape(1, -1)
    return full_feat


# --------------------------
# 3. Live Webcam Loop
# --------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not access webcam.")
    exit()

print("📷 Live BMI prediction started...")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for stability
    frame_small = cv2.resize(frame, (480, 640))

    # Extract full feature vector
    try:
        feats = extract_features_from_frame(frame_small)
        bmi_pred = reg.predict(feats)[0]
        bmi_text = f"BMI: {bmi_pred:.2f}"
    except Exception as e:
        bmi_text = "BMI: ---"
        print("Feature extraction error:", e)

    # Overlay BMI on frame
    cv2.putText(frame_small, bmi_text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Live BMI Prediction", frame_small)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
