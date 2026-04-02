import cv2
import numpy as np
import torch
import joblib
from torchvision import transforms
from PIL import Image

from deep_features import ResNetFeatureExtractor  # ResNet101 to match v2
from project_paths import models_dir


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

deep_model = ResNetFeatureExtractor("resnet101", pretrained=True).to(device)
deep_model.eval()

_m2 = models_dir("v2")
scaler = joblib.load(str(_m2 / "scaler_deep_only.joblib"))
reg = joblib.load(str(_m2 / "deep_only_krr.joblib"))

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def deep_features_from_frame(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img_t = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        deep = deep_model(img_t).cpu().numpy()
    deep_s = scaler.transform(deep)
    return deep_s


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not found")
    raise SystemExit

print("Live BMI (deep-only). Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_small = cv2.resize(frame, (480, 640))

    try:
        X = deep_features_from_frame(frame_small)
        bmi = reg.predict(X)[0]
        text = f"BMI: {bmi:.2f}"
    except Exception as e:
        text = "BMI: ---"
        print("Error:", e)

    cv2.putText(frame_small, text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Live BMI (Deep Only)", frame_small)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
