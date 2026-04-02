import os
import random
import cv2
import numpy as np
import mediapipe as mp
import torch
from torchvision import transforms
from PIL import Image
import joblib

from deep_features_v3 import EfficientNetFeatureExtractor
from project_paths import dataset_2dimage_dir, models_dir


# ===================== PATHS ========================
_m4 = models_dir("v4")
DATASET_DIR = str(dataset_2dimage_dir() / "Image_test")
SCALER_PATH = str(_m4 / "scaler_v4.joblib")
REG_PATH = str(_m4 / "deep_krr_v4.joblib")
# ====================================================


# -----------------------------------------------
#  EfficientNet Preprocessing
# -----------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# -----------------------------------------------
#  Ground Truth Parser (correct for Image_test)
# -----------------------------------------------
def parse_ground_truth(fname):
    """
    Dataset format:
      height_raw = height(m) * 100000
      weight_raw = weight(kg) * 100000
    """
    name = os.path.splitext(fname)[0]
    parts = name.split("_")

    try:
        height_raw = int(parts[3])          # e.g. 182880
        weight_raw = int(parts[4])          # e.g. 12337713

        height_m = height_raw / 100000.0    # -> meters
        weight_kg = weight_raw / 100000.0   # -> kg

        bmi = weight_kg / (height_m**2 + 1e-8)

        return height_m, weight_kg, bmi

    except Exception as e:
        print("❌ GT Parse Error:", e)
        return None, None, None


# -----------------------------------------------
#  Deep Feature Extraction
# -----------------------------------------------
def extract_deep_features(img_bgr, model, device):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    t = preprocess(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(t).cpu().numpy()  # shape (1, F)
    return feat


# -----------------------------------------------
#  Crop Person Using MediaPipe Pose
# -----------------------------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=True,        # Important for still images
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def crop_person_pose(image_bgr):
    h, w, _ = image_bgr.shape

    results = pose.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        print("❌ No pose detected.")
        return None, None

    # Draw skeleton for visualization
    annotated = image_bgr.copy()
    mp_drawing.draw_landmarks(
        annotated,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=3),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 140, 0), thickness=3)
    )

    # Create tight crop around pose landmarks
    pts = np.array([
        (int(lm.x * w), int(lm.y * h))
        for lm in results.pose_landmarks.landmark
        if lm.visibility > 0.5
    ])

    if len(pts) == 0:
        print("❌ Not enough visible landmarks.")
        return annotated, None

    x1 = max(0, np.min(pts[:, 0]) - 20)
    y1 = max(0, np.min(pts[:, 1]) - 20)
    x2 = min(w, np.max(pts[:, 0]) + 20)
    y2 = min(h, np.max(pts[:, 1]) + 20)

    crop = image_bgr[y1:y2, x1:x2]

    return annotated, crop


# ================= MAIN SCRIPT ==================

if __name__ == "__main__":

    # Pick random image
    fname = random.choice([
        f for f in os.listdir(DATASET_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    IMG_PATH = os.path.join(DATASET_DIR, fname)
    print("\n🖼️ Selected Image:", fname)

    # Ground truth BMI
    h, w, true_bmi = parse_ground_truth(fname)
    if true_bmi is not None:
        print(f"📌 True BMI: {true_bmi:.2f} (H={h:.3f} m, W={w:.3f} kg)")
    else:
        print("⚠️ Ground truth unavailable.")

    # Load image
    image = cv2.imread(IMG_PATH)
    if image is None:
        raise FileNotFoundError("❌ Image does not exist:", IMG_PATH)

    # Pose crop + skeleton visualization
    annotated_img, crop = crop_person_pose(image)
    if crop is None:
        print("❌ Could not crop body using pose.")
        exit()

    # Load BMI model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deep_model = EfficientNetFeatureExtractor(pretrained=True).to(device)
    deep_model.eval()
    scaler = joblib.load(SCALER_PATH)
    reg = joblib.load(REG_PATH)

    # Extract features
    feat = extract_deep_features(crop, deep_model, device)

    # Scale + Predict
    feat_scaled = scaler.transform(feat)
    pred_bmi = float(reg.predict(feat_scaled)[0])

    print(f"🤖 Predicted BMI: {pred_bmi:.2f}")

    # Errors
    if true_bmi is not None:
        abs_err = abs(pred_bmi - true_bmi)
        pct_err = (abs_err / true_bmi) * 100

        print(f"\n📉 Absolute Error: {abs_err:.2f}")
        print(f"📊 Percentage Error: {pct_err:.2f}%")

    # Show annotated image
    cv2.putText(annotated_img, f"Pred BMI: {pred_bmi:.2f}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    if true_bmi is not None:
        cv2.putText(annotated_img, f"GT: {true_bmi:.2f}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

    cv2.imshow("Pose-Based BMI Prediction", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
