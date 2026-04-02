import os
import re
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class Image2BMIDataset(Dataset):
    """
    Dataset for 2DImage2BMI-style filenames:
    000197_M_18_180940_5443319.jpg
           ^ ^   ^      ^
           | |   |      |
           | |   |      weight * 1e5
           | |   height * 1e3 (cm * 1e3)
           | age
           sex (M/F)
    """
    def __init__(self, root_dir: str, img_size: int = 224):
        self.root_dir = root_dir
        self.img_names = sorted([
            f for f in os.listdir(root_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        self.transform = transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Precompile regex
        # Format: id_sex_age_height_weight.suffix
        self._pattern = re.compile(r"\d+?_([FfMm])_(\d+?)_(\d+?)_(\d+).+")

    def __len__(self) -> int:
        return len(self.img_names)

    def _parse_filename(self, fname: str) -> Tuple[int, int, float, float, float]:
        """Extract metadata & compute BMI."""
        m = self._pattern.match(fname)
        if m is None:
            raise ValueError(f"Filename does not match pattern: {fname}")

        sex_char, age_str, height_raw, weight_raw = m.groups()
        sex = 0 if sex_char.upper() == "F" else 1
        age = int(age_str)

        # Convert back to real values
        height_m = int(height_raw) / 100000.0
        weight_kg = int(weight_raw) / 100000.0
        bmi = weight_kg / (height_m ** 2 + 1e-8)

        return sex, age, height_m, weight_kg, bmi

    def __getitem__(self, idx: int):
        fname = self.img_names[idx]
        path = os.path.join(self.root_dir, fname)
        img = Image.open(path).convert("RGB")
        img_t = self.transform(img)

        sex, age, height_m, weight_kg, bmi = self._parse_filename(fname)

        meta = torch.tensor([sex, age, height_m, weight_kg], dtype=torch.float32)
        bmi_tensor = torch.tensor(bmi, dtype=torch.float32)

        return img_t, meta, bmi_tensor, fname, path
