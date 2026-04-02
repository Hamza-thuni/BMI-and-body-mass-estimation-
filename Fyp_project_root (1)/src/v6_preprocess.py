import json
import os

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

from project_paths import project_root, yolo_weights


_data = project_root() / "data"
SRC_DIR = str(_data / "images")
OUT_CROP = str(_data / "yolo_crops")
OUT_SIL = str(_data / "silhouettes")
OUT_POSE = str(_data / "pose_landmarks")

os.makedirs(OUT_CROP, exist_ok=True)
os.makedirs(OUT_SIL, exist_ok=True)
os.makedirs(OUT_POSE, exist_ok=True)

mp_pose = mp.solutions.pose.Pose(static_image_mode=True)
yolo = YOLO(yolo_weights("yolov8n.pt"))


def extract_pose(img, save_path):
    h, w = img.shape[:2]
    results = mp_pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return False

    pts = []
    for lm in results.pose_landmarks.landmark:
        pts.append([lm.x * w, lm.y * h, lm.z])

    with open(save_path, "w") as f:
        json.dump(pts, f)
    return True


def extract_silhouette(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = cv2.medianBlur(mask, 5)
    return mask


def preprocess_all():
    files = [f for f in os.listdir(SRC_DIR) if f.lower().endswith((".jpg",".png"))]

    for f in files:
        path = os.path.join(SRC_DIR, f)
        img = cv2.imread(path)
        if img is None:
            continue

        # YOLO crop
        results = yolo.predict(img, conf=0.3, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes)==0:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        person_boxes = [xyxy[i] for i in range(len(xyxy)) if cls[i]==0]
        if not person_boxes:
            continue

        # largest human
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in person_boxes]
        box = person_boxes[int(np.argmax(areas))]
        x1,y1,x2,y2 = map(int, box)
        crop = img[y1:y2, x1:x2]

        # Save crop
        cv2.imwrite(os.path.join(OUT_CROP, f), crop)

        # Silhouette
        sil = extract_silhouette(crop)
        cv2.imwrite(os.path.join(OUT_SIL, f), sil)

        # Pose
        extract_pose(crop, os.path.join(OUT_POSE, f.replace(".jpg", ".json")))

        print("Processed:", f)


if __name__ == "__main__":
    preprocess_all()
