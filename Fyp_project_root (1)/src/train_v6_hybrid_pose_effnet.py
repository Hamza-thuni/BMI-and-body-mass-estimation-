import os
import numpy as np
import cv2
from tqdm import tqdm
import joblib

from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from person_detector_yolo_v4 import YOLOPersonDetector
from v6_utils_pose_effnet import (
    pose_coords_full_image,
    make_pose_box,
    extract_pose_features,
    extract_deep_features,
)
from project_paths import dataset_2dimage_dir, models_dir


BASE_DATA = dataset_2dimage_dir()
OUT_DIR = models_dir("v6")
os.makedirs(OUT_DIR, exist_ok=True)


def parse_bmi(fname):
    name = fname.split(".")[0]
    parts = name.split("_")

    # Remove any non-digit junk (like trailing bracket)
    height_raw = int(''.join(filter(str.isdigit, parts[3])))
    weight_raw = int(''.join(filter(str.isdigit, parts[4])))

    h_m = height_raw / 100000.0
    w_kg = weight_raw / 100000.0

    bmi = w_kg / (h_m * h_m + 1e-8)
    return bmi



def build_split(split, yolo: YOLOPersonDetector):
    IMG_DIR = os.path.join(str(BASE_DATA), f"Image_{split}")
    files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(("jpg", "png"))]

    X, Y = [], []
    print(f"\nExtracting {split} features ({len(files)} images)...")

    for f in tqdm(files):
        img_path = os.path.join(IMG_DIR, f)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        coords = pose_coords_full_image(img, yolo)
        if coords is None:
            continue

        box = make_pose_box(coords, w, h, margin=0.25)
        if box is None:
            continue

        x1, y1, x2, y2 = box
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        deep = extract_deep_features(crop)
        pose = extract_pose_features(coords)
        feat = np.concatenate([deep, pose])

        X.append(feat)
        Y.append(parse_bmi(f))

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


if __name__ == "__main__":
    yolo = YOLOPersonDetector(conf=0.4)
    X_train, y_train = build_split("train", yolo)
    X_val, y_val = build_split("val", yolo)
    X_test, y_test = build_split("test", yolo)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    alphas = (0.05, 0.1, 0.3, 1.0, 3.0, 10.0)
    gammas = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3)
    best_mae = float("inf")
    best_params = (1.0, 1e-4)
    print("\nTuning KRR on validation (MAE)...")
    for alpha in alphas:
        for gamma in gammas:
            krr = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)
            krr.fit(X_train_s, y_train)
            mae = mean_absolute_error(y_val, krr.predict(X_val_s))
            if mae < best_mae:
                best_mae = mae
                best_params = (alpha, gamma)
    print(f"Best val MAE {best_mae:.4f} → alpha={best_params[0]}, gamma={best_params[1]}")

    X_tv = np.vstack([X_train_s, X_val_s])
    y_tv = np.concatenate([y_train, y_val])
    model = KernelRidge(
        kernel="rbf", alpha=best_params[0], gamma=best_params[1]
    )
    model.fit(X_tv, y_tv)

    def evaluate(name, Xs, ys):
        pred = model.predict(Xs)
        mae = mean_absolute_error(ys, pred)
        rmse = np.sqrt(mean_squared_error(ys, pred))
        print(f"{name} → MAE {mae:.3f}, RMSE {rmse:.3f}")

    print("\n=== V6 HYBRID RESULTS (train+val fit, test held out) ===")
    evaluate("Train+Val", X_tv, y_tv)
    evaluate("Test ", X_test_s, y_test)

    joblib.dump(scaler, os.path.join(str(OUT_DIR), "scaler_v6.joblib"))
    joblib.dump(model, os.path.join(str(OUT_DIR), "krr_hybrid_v6.joblib"))

    print("\nSaved V6 model and scaler.")
