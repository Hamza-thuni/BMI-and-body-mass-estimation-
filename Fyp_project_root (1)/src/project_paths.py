"""
Portable paths for the BMI / 2D-image pipeline.

- FYP_PROJECT_ROOT: directory that contains src/, features/, models/, etc.
- FYP_DATASET_DIR: directory with Image_train, Image_val, Image_test
- FYP_YOLO_WEIGHTS: full path to a YOLO weights file (optional)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


def project_root() -> Path:
    env = os.environ.get("FYP_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return _THIS_DIR.parent


def dataset_2dimage_dir() -> Path:
    env = os.environ.get("FYP_DATASET_DIR")
    if env:
        return Path(env).expanduser().resolve()
    root = project_root()
    candidates = [
        root / "data (1)" / "2DImage2BMI-main (1)" / "2DImage2BMI-main (1)" / "datasets (1)",
        root / "data (1)" / "2DImage2BMI-main (1)" / "2DImage2BMI-main (1)" / "datasets",
        root / "data" / "2DImage2BMI-main" / "2DImage2BMI-main" / "datasets",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]


def features_dir(which: str = "v1") -> Path:
    root = project_root()
    names = {"v1": "features", "v2": "features_v2", "v4": "features_v4", "v7": "features_v7"}
    key = which.lower()
    if key not in names:
        raise ValueError(f"features_dir: expected one of {list(names)}, got {which!r}")
    return root / names[key]


def models_dir(which: str = "v1") -> Path:
    root = project_root()
    names = {"v1": "models", "v2": "models_v2", "v4": "models_v4", "v6": "models_v6", "v7": "models_v7"}
    key = which.lower()
    if key not in names:
        raise ValueError(f"models_dir: expected one of {list(names)}, got {which!r}")
    return root / names[key]


def yolo_weights(filename: str = "yolov8s.pt") -> str:
    """Local weights path if found; otherwise basename for Ultralytics download."""
    env = os.environ.get("FYP_YOLO_WEIGHTS")
    if env:
        return env
    for base in (project_root(), project_root() / "Notebooks (1)", _THIS_DIR):
        p = base / filename
        if p.is_file():
            return str(p)
    return filename
