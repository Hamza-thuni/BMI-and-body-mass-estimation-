import os
import numpy as np

from project_paths import dataset_2dimage_dir, features_dir
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

from dataset_2dimage import Image2BMIDataset
from deep_features import ResNetFeatureExtractor
from anthropometric_features_plus import extract_mixed_features_plus


def extract_for_split(split_name: str, img_root: str, batch_size: int,
                      model: ResNetFeatureExtractor, device: torch.device):

    ds = Image2BMIDataset(img_root)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    all_deep, all_antro, all_bmi, all_names = [], [], [], []

    model.eval()

    for imgs, meta, bmi, fname, path in tqdm(dl, desc=split_name):
        imgs = imgs.to(device)
        with torch.no_grad():
            deep = model(imgs)  # (B, C)

        for p in path:
            bgr = cv2.imread(p)
            feats = extract_mixed_features_plus(bgr)
            all_antro.append(feats)

        all_deep.append(deep.cpu().numpy())
        all_bmi.extend(bmi.numpy().tolist())
        all_names.extend(list(fname))

    deep_arr = np.concatenate(all_deep, axis=0)
    antro_arr = np.stack(all_antro, axis=0)
    bmi_arr = np.array(all_bmi, dtype=np.float32)

    return deep_arr, antro_arr, bmi_arr, np.array(all_names)


if __name__ == "__main__":
    base_data = str(dataset_2dimage_dir())
    out_dir = str(features_dir("v2"))
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    feat_model = ResNetFeatureExtractor("resnet101", pretrained=True).to(device)

    splits = {
        "train": os.path.join(base_data, "Image_train"),
        "val":   os.path.join(base_data, "Image_val"),
        "test":  os.path.join(base_data, "Image_test"),
    }

    for split, img_root in splits.items():
        deep, antro, bmi, names = extract_for_split(
            split, img_root, batch_size=16, model=feat_model, device=device
        )
        np.savez_compressed(
            os.path.join(out_dir, f"{split}_features_v2.npz"),
            deep=deep, antro=antro, bmi=bmi, names=names
        )
        print(f"{split}: deep {deep.shape}, antro {antro.shape}, bmi {bmi.shape}")
