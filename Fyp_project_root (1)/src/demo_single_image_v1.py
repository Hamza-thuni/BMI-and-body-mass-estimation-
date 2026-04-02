import os
import random
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import joblib

from deep_features import ResNetFeatureExtractor
from anthropometric_features import extract_mixed_features
from project_paths import dataset_2dimage_dir, models_dir


if __name__ == "__main__":
    base_data = str(dataset_2dimage_dir())
    test_dir = os.path.join(base_data, "Image_test")

    # pick a random image
    img_name = random.choice([
        f for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    img_path = os.path.join(test_dir, img_name)
    print("Random test image:", img_name)

    # compute ground-truth BMI from filename (same logic as dataset)
    parts = img_name.split("_")
    # id, sex, age, height_raw, weight_raw
    height_raw = int(parts[3])
    weight_raw = int(parts[4].split(".")[0])
    height_m = height_raw / 100000.0
    weight_kg = weight_raw / 100000.0
    true_bmi = weight_kg / (height_m ** 2 + 1e-8)
    print(f"True BMI: {true_bmi:.2f}")

    # load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deep_model = ResNetFeatureExtractor("resnet50", pretrained=True).to(device)
    deep_model.eval()

    reg = joblib.load(str(models_dir("v1") / "krr_mixed_features.joblib"))

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # prepare image
    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    img_t = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        deep_feat = deep_model(img_t).cpu().numpy().flatten()

    antro_feat = extract_mixed_features(bgr)

    feat = np.concatenate([deep_feat, antro_feat], axis=0).reshape(1, -1)
    pred_bmi = reg.predict(feat)[0]

    print(f"Predicted BMI: {pred_bmi:.2f}")

    # show image with overlay
    cv2.putText(bgr, f"True: {true_bmi:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(bgr, f"Pred: {pred_bmi:.1f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    cv2.imshow("Random Test Image - BMI Prediction", bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
