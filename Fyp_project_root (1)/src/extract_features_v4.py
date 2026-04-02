import os
import re

from project_paths import dataset_2dimage_dir, features_dir
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import cv2

from deep_features_v3 import EfficientNetFeatureExtractor
from person_detector_yolo_v4 import YOLOPersonDetector


def parse_bmi_from_filename(fname: str):
    pattern = re.compile(r"\d+?_([FfMm])_(\d+?)_(\d+?)_(\d+).+")
    m = pattern.match(fname)
    if m is None:
        raise ValueError(f"Bad filename format: {fname}")
    _, _, h_raw, w_raw = m.groups()
    h_m = int(h_raw) / 100000.0
    w_kg = int(w_raw) / 100000.0
    return w_kg / (h_m ** 2 + 1e-8)


def get_transform():
    return transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def extract_split(split_name: str, img_root: str,
                  model: EfficientNetFeatureExtractor,
                  detector: YOLOPersonDetector,
                  device: torch.device):

    t = get_transform()

    fnames = sorted([
        f for f in os.listdir(img_root)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    feats = []
    bmis = []
    names = []

    model.eval()

    for fname in tqdm(fnames, desc=f"{split_name}"):
        path = os.path.join(img_root, fname)
        bmi = parse_bmi_from_filename(fname)

        bgr = cv2.imread(path)
        if bgr is None:
            continue

        cropped = detector.detect_and_crop(bgr)

        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        img_t = t(img).unsqueeze(0).to(device)

        with torch.no_grad():
            deep = model(img_t).cpu().numpy().reshape(-1)

        feats.append(deep)
        bmis.append(bmi)
        names.append(fname)

    deep_arr = np.stack(feats, axis=0)
    bmi_arr = np.array(bmis, dtype=np.float32)
    names_arr = np.array(names)

    return deep_arr, bmi_arr, names_arr


if __name__ == "__main__":
    base_data = str(dataset_2dimage_dir())
    out_dir = str(features_dir("v4"))
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    feat_model = EfficientNetFeatureExtractor(pretrained=True).to(device)
    yolo = YOLOPersonDetector(conf=0.4)

    splits = {
        "train": os.path.join(base_data, "Image_train"),
        "val":   os.path.join(base_data, "Image_val"),
        "test":  os.path.join(base_data, "Image_test"),
    }

    for split, img_root in splits.items():
        deep, bmi, names = extract_split(split, img_root, feat_model, yolo, device)
        np.savez_compressed(
            os.path.join(out_dir, f"{split}_deep_v4.npz"),
            deep=deep,
            bmi=bmi,
            names=names,
        )
        print(f"{split}: deep {deep.shape}, bmi {bmi.shape}")
