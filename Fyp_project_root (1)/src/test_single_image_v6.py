import os
import cv2
import numpy as np
import joblib

from person_detector_yolo_v4 import YOLOPersonDetector
from v6_utils_pose_effnet import (
    pose_coords_full_image,
    make_pose_box,
    extract_pose_features,
    extract_deep_features,
)
from project_paths import dataset_2dimage_dir, models_dir

# ---------------------- PATHS -----------------------
_m6 = models_dir("v6")
DATASET_DIR = str(dataset_2dimage_dir() / "Image_test")

SCALER_PATH = str(_m6 / "scaler_v6.joblib")
MODEL_PATH = str(_m6 / "krr_hybrid_v6.joblib")

# ----------------------------------------------------


def parse_bmi_from_name(fname):
    """
    Extract BMI ground truth from the filename.
    Handles dataset filenames with extra trailing fields.
    """
    name = fname.split(".")[0]
    parts = name.split("_")

    # Height = parts[3], Weight = parts[4]
    height_raw = int(''.join(filter(str.isdigit, parts[3])))
    weight_raw = int(''.join(filter(str.isdigit, parts[4])))

    h_m = height_raw / 100000.0
    w_kg = weight_raw / 100000.0
    return w_kg / (h_m * h_m + 1e-8)


def main():
    # Load model + scaler
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    yolo = YOLOPersonDetector(conf=0.4)

    # Pick a random image from test folder
    files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith((".jpg",".png",".jpeg"))]
    fname = np.random.choice(files)
    img_path = os.path.join(DATASET_DIR, fname)

    print("\nTesting V6 on single image:")
    print("Selected image:", fname)

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Could not load image.")
        return

    h, w = img.shape[:2]

    coords = pose_coords_full_image(img, yolo)
    if coords is None:
        print("❌ No pose detected (full frame + YOLO crop).")
        return

    # Make consistent bounding box
    box = make_pose_box(coords, w, h, margin=0.25)
    if box is None:
        print("❌ Could not compute pose-based box.")
        return

    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        print("❌ Crop invalid.")
        return

    # Feature extraction
    deep_feat = extract_deep_features(crop)
    pose_feat = extract_pose_features(coords)
    feat = np.concatenate([deep_feat, pose_feat]).reshape(1, -1)

    # Predict
    Xs = scaler.transform(feat)
    bmi_pred = float(model.predict(Xs)[0])

    # Ground truth BMI
    bmi_gt = parse_bmi_from_name(fname)

    print(f"Predicted BMI: {bmi_pred:.2f}")
    print(f"Ground Truth : {bmi_gt:.2f}")
    print(f"Absolute Error: {abs(bmi_gt - bmi_pred):.2f}")

    # Draw visualization
    vis = img.copy()
    cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(vis, f"Pred: {bmi_pred:.2f}", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.putText(vis, f"GT: {bmi_gt:.2f}", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

    crop_disp = cv2.resize(crop, (300, 450))
    cv2.imshow("Full Image (V6)", vis)
    cv2.imshow("Crop Used for V6 Prediction", crop_disp)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
