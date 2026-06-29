# BMI & Body Mass Estimation from 2D Images

> **Final Year Project** — Non-contact estimation of Body Weight, BMI, and Body Composition from a single RGB camera using Computer Vision and Machine Learning.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-00A67E?logo=google&logoColor=white)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-Academic-blue)](#license)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Pipeline Versions](#pipeline-versions)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
  - [1. Feature Extraction](#1-feature-extraction)
  - [2. Model Training](#2-model-training)
  - [3. Live Demo (CLI)](#3-live-demo-cli)
  - [4. Web Application](#4-web-application)
- [Hardware Setup — ArUco Calibration](#hardware-setup--aruco-calibration)
- [Model Architecture (V9 — Final)](#model-architecture-v9--final)
- [Evaluation Results](#evaluation-results)
- [Technologies Used](#technologies-used)
- [Future Work](#future-work)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project implements a **non-invasive, camera-based system** that estimates a person's:

| Metric | Description |
|--------|-------------|
| **Height** (m) | Computed from pose landmarks calibrated against ArUco reference markers |
| **Weight** (kg) | Predicted by a hybrid ML ensemble (Ridge + XGBoost) |
| **BMI** | Derived as `weight / height²` |
| **Body Fat %** | Estimated via the Deurenberg formula using age and sex |
| **Fat Mass / Lean Mass** | Decomposed from predicted weight and body fat percentage |
| **BMI Category** | Classified as Underweight, Healthy, Overweight, or Obese |

The system works in **real-time** using a standard webcam and two printed ArUco markers mounted on a wall for absolute scale calibration.

---

## Key Features

- **Real-time inference** at ~5 FPS from a live webcam feed
- **ArUco-based absolute scale** — no manual height input required
- **Hybrid 1565-D feature vector** combining physics, deep learning, and pose features
- **Web-based dashboard** (Flask + SocketIO) with live video overlay and real-time results
- **Multi-dataset training** using 2DImage2BMI and Celeb-FBI datasets for bias reduction
- **Temporal smoothing** via sliding-window averaging for stable predictions
- **Background segmentation** removes scene clutter before deep feature extraction
- **Distribution-balanced training** using histogram flattening to reduce mean-regression bias

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        INPUT: RGB Camera Frame                       │
└───────────────┬──────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────┐    ┌──────────────────────────────────┐
│   ArUco Marker Detection  │    │   MediaPipe Pose Landmarker      │
│   (cv2.aruco, 4×4 dict)   │    │   (Heavy model, 33 landmarks)   │
│                           │    │   + Segmentation Mask            │
│   ID 0 @ 80cm from floor  │    └──────────┬───────────────────────┘
│   ID 1 @ 180cm from floor │               │
│           ↓               │               ▼
│   px_per_m scale factor   │    ┌──────────────────────────────────┐
└───────────┬───────────────┘    │        Feature Extraction         │
            │                    │                                    │
            │                    │  ┌────────────┐  ┌─────────────┐  │
            └────────────────────┤  │ Physics 5-D│  │ Deep 1536-D │  │
                                 │  │ (Area, Sh,  │  │ EfficientNet│  │
                                 │  │  Hip, Torso,│  │ B3 on seg.  │  │
                                 │  │  Height)    │  │ crop        │  │
                                 │  └────────────┘  └─────────────┘  │
                                 │  ┌─────────────┐                  │
                                 │  │ Pose 24-D   │                  │
                                 │  │ Scale-inv.  │                  │
                                 │  │ skeletal    │                  │
                                 │  │ proportions │                  │
                                 │  └─────────────┘                  │
                                 └──────────┬───────────────────────┘
                                            │
                                            ▼
                                 ┌──────────────────────────────────┐
                                 │   Concatenated 1565-D Vector     │
                                 │   [Physics | Deep | Pose]        │
                                 └──────────┬───────────────────────┘
                                            │
                                            ▼
                                 ┌──────────────────────────────────┐
                                 │   StandardScaler → Ensemble      │
                                 │   Ridge (α=100) ⊕ XGBoost       │
                                 │   50/50 weighted average         │
                                 └──────────┬───────────────────────┘
                                            │
                                            ▼
                                 ┌──────────────────────────────────┐
                                 │   Post-Processing & Calibration  │
                                 │   • Linear correction (>70kg)    │
                                 │   • Area-based sanity bounds     │
                                 │   • Temporal smoothing (30-frame)│
                                 └──────────────────────────────────┘
                                            │
                                            ▼
                                 ┌──────────────────────────────────┐
                                 │   OUTPUT: Weight, BMI, BF%,      │
                                 │           Category, Fat/Lean Mass │
                                 └──────────────────────────────────┘
```

---

## Project Structure

```
Fyp_project_root/
│
├── src/                              # Core source code
│   ├── project_paths.py              # Portable path resolution for all modules
│   ├── deep_features_v3.py           # EfficientNet-B3 feature extractor (1536-D)
│   ├── v7_pose_features.py           # 24-D scale-invariant pose feature extractor
│   ├── v7_segmentation.py            # MediaPipe background removal utility
│   ├── person_detector_yolo_v4.py    # YOLOv8s person detection & cropping
│   ├── anthropometric_features.py    # Contour + pose anthropometric features (13-D)
│   │
│   ├── extract_features_v7.py        # V7: Deep + Pose feature extraction pipeline
│   ├── extract_features_v8_physical.py # V8: Absolute physical dimension extraction
│   │
│   ├── train_v9_hybrid.py            # V9: Final hybrid model training script
│   ├── train_v9_physics_only.py      # V9: Physics-only model variant
│   ├── train_v9_demo.py              # V9: Lightweight demo model
│   │
│   ├── live_bmi_demo_v9.py           # V9: Real-time CLI demo with webcam
│   ├── live_bmi_demo_v8_pi.py        # V8: Raspberry Pi optimized demo
│   ├── test_random_image_v9.py       # Single image test script
│   └── aruco.py                      # ArUco marker generation utilities
│
├── demo_app/                         # Flask web application
│   ├── demo_app.py                   # Flask + SocketIO server
│   ├── v9_pipeline_wrapper.py        # Wraps V9 pipeline for web use
│   ├── config.py                     # Environment configuration (DEV/PROD)
│   ├── mock_models.py                # Mock pipeline for UI testing
│   ├── requirements.txt              # Web app dependencies
│   ├── run_on_laptop.bat             # Windows launch script
│   ├── templates/
│   │   └── index.html                # Dashboard UI
│   └── static/
│       ├── css/                      # Stylesheets
│       └── js/                       # Client-side JavaScript
│
├── Notebooks (1)/                    # Jupyter notebooks
│   ├── EDA (1).ipynb                 # Exploratory data analysis
│   ├── cnn (1).ipynb                 # CNN experiments
│   ├── preprocessing_pipeline_demo.ipynb  # Pipeline demonstration
│   └── presentation_demo.ipynb       # Project presentation demo
│
├── data (1)/                         # Datasets (git-ignored)
│   ├── 2DImage2BMI-main/             # Primary dataset (train/val/test splits)
│   ├── Celeb-FBI Dataset/            # Celebrity dataset for bias correction
│   └── rgb_cleaned.csv               # Cleaned metadata
│
├── content/
│   └── bodym-dataset/                # Additional body measurement dataset
│
├── features_v7/                      # Extracted V7 features (deep + pose)
├── features_v8/                      # Extracted V8 features (physical dims)
├── features_v9/                      # Extracted V9 features (Celeb-FBI)
│
├── models_v9/                        # Trained model bundles
│   ├── ensemble_v9_hybrid.joblib     # Final hybrid model (1565-D)
│   ├── ensemble_v9_physics.joblib    # Physics-only variant (5-D)
│   ├── ensemble_v9_demo.joblib       # Lightweight demo model
│   └── ensemble_v8_physical.joblib   # V8 baseline model
│
├── pose_landmarker_heavy.task        # MediaPipe pose model (auto-downloaded)
├── yolov8s.pt                        # YOLOv8s weights for person detection
├── .gitignore                        # Excludes large binaries & datasets
└── README.md                         # This file
```

---

## Datasets

### 1. 2DImage2BMI Dataset (Primary)

The primary dataset used for training and evaluation. Full-body RGB images with ground-truth height and weight encoded in the filename:

```
{id}_{gender}_{age}_{height_mm}_{weight_g}.jpg

Example: 000006_F_19_177800_8300740.jpg
  → Height: 177800 / 100000 = 1.778 m
  → Weight: 8300740 / 100000 = 83.01 kg
  → BMI: 83.01 / 1.778² = 26.3
```

| Split | Purpose |
|-------|---------|
| `Image_train` | Model training (with augmentation) |
| `Image_val`   | Hyperparameter tuning |
| `Image_test`  | Final evaluation (strict — never seen during training) |

### 2. Celeb-FBI Dataset (Bias Correction)

A supplementary dataset of celebrity images with known body measurements. Used in V9 to **flatten the weight distribution** and combat mean-regression bias that causes the model to underestimate heavy subjects and overestimate light subjects.

---

## Pipeline Versions

The project evolved through multiple iterations, each addressing limitations of the previous:

| Version | Features | Dimensionality | Key Innovation |
|---------|----------|----------------|----------------|
| **V1** | Contour + Pose | 13-D | Baseline anthropometric features |
| **V2** | Deep (EfficientNet) | 1536-D | CNN-based appearance features |
| **V4** | Deep + YOLO crop | 1536-D | Person detection before feature extraction |
| **V6** | Deep + Pose (16-D) | 1552-D | Combined deep and skeletal features |
| **V7** | Deep + Pose (24-D) + Segmentation | 1560-D | Background removal, richer pose features, augmentation |
| **V8** | Physical dimensions | 5-D | Absolute measurements (m, m²) via ArUco calibration |
| **V9 (Final)** | **Physics + Deep + Pose** | **1565-D** | Hybrid ensemble, Celeb-FBI data, distribution balancing |

---

## Installation & Setup

### Prerequisites

- **Python 3.10+**
- **CUDA-compatible GPU** (recommended for EfficientNet inference, not required)
- **Webcam** (for live demo)
- **Two printed ArUco markers** (for scale calibration — see [Hardware Setup](#hardware-setup--aruco-calibration))

### 1. Clone the Repository

```bash
git clone https://github.com/Hamza-thuni/BMI-and-body-mass-estimation-.git
cd BMI-and-body-mass-estimation-
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Core dependencies
pip install numpy opencv-python mediapipe torch torchvision Pillow joblib scikit-learn tqdm

# For XGBoost ensemble (recommended)
pip install xgboost

# For YOLO person detection
pip install ultralytics

# For the web application
pip install flask flask-socketio python-dotenv eventlet
```

Or install from the demo app's requirements file:

```bash
pip install -r demo_app/requirements.txt
pip install scikit-learn xgboost ultralytics tqdm
```

### 4. Download Model Files

The following files are auto-downloaded on first run but can be obtained manually:

| File | Size | Source |
|------|------|--------|
| `pose_landmarker_heavy.task` | ~30 MB | [MediaPipe Models](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task) |
| `yolov8s.pt` | ~23 MB | Auto-downloaded by Ultralytics |

---

## Usage

### 1. Feature Extraction

Extract features from the dataset images before training. Run from the `src/` directory:

```bash
cd src

# Step 1: Extract V8 physical features (Area, Widths, Height)
python extract_features_v8_physical.py

# Step 2: Extract V7 deep + pose features (EfficientNet + 24-D pose)
python extract_features_v7.py
```

**Output:**
- `features_v8/{split}_v8_phys.npz` — Physical features (5-D) + ground-truth weight
- `features_v7/{split}_v7.npz` — Deep features (1536-D) + Pose features (24-D) + ground-truth BMI

### 2. Model Training

```bash
cd src

# Train the final V9 hybrid model (recommended)
python train_v9_hybrid.py

# Alternative: Physics-only model (faster, no GPU needed)
python train_v9_physics_only.py
```

**Output:** Saved model bundles in `models_v9/`:
- `ensemble_v9_hybrid.joblib` — Full hybrid ensemble
- `ensemble_v9_physics.joblib` — Physics-only variant

### 3. Live Demo (CLI)

Real-time weight and BMI estimation via webcam with on-screen HUD:

```bash
cd src
python live_bmi_demo_v9.py
```

**Controls:**
| Key | Action |
|-----|--------|
| `q` | Quit the demo |

**On-screen display:**
- Pose landmarks (green dots)
- ArUco marker labels and scale line (pink)
- Height, Weight, BMI, and BMI category in a dark HUD panel
- Warning if the body is cut off at frame edges

### 4. Web Application

A full-featured web dashboard with live video feed and real-time body composition results:

```bash
cd demo_app

# Option 1: Run directly
python demo_app.py

# Option 2: Use the batch file (Windows)
run_on_laptop.bat
```

The dashboard automatically opens at **http://127.0.0.1:5000** and provides:

- Live camera feed with pose overlay and ArUco detection
- Continuous scanning mode via WebSocket
- Real-time display of Height, Weight, BMI, Body Fat %, Fat Mass, and Lean Mass
- Adjustable parameters (age, sex, parallax factor)

**Environment variables** (optional, set via `demo_app/.env`):

```env
APP_MODE=DEV        # DEV (laptop webcam) or PROD (Raspberry Pi)
CAMERA_INDEX=0      # Camera device index
```

---

## Hardware Setup — ArUco Calibration

The system uses **two ArUco markers** (Dictionary: `DICT_4X4_50`) mounted on a wall to establish absolute scale without any manual input.

### Marker Placement

```
Wall
─────────────────────────────────
         ┌─────┐
         │ ID 1│  ← 180 cm from floor (top marker)
         └─────┘
            │
            │  ← 1.0 m separation (known distance)
            │
         ┌─────┐
         │ ID 0│  ← 80 cm from floor (bottom marker)
         └─────┘
─────────────────────────────────
         Floor
```

### How Scale Calibration Works

1. Both markers are detected in the camera frame using OpenCV's ArUco module
2. The vertical pixel distance between their centres is measured
3. Since the physical separation is known (1.0 m), the scale factor is computed:

$$\text{px\_per\_m} = \frac{|\text{center}_1.y - \text{center}_0.y|}{\text{MARKER\_SEPARATION\_M}}$$

4. A **parallax correction factor** (default 0.94) accounts for the person standing closer to the camera than the wall
5. A **stature adjustment factor** (default 1.08) converts landmark span to true height

### Generating ArUco Markers

You can generate the required markers using the `src/aruco.py` utility or any ArUco generator. Print them at a reasonable size (8–12 cm) and mount them vertically aligned on a flat wall.

---

## Model Architecture (V9 — Final)

### Feature Vector (1565-D)

| Component | Dimensions | Source | Description |
|-----------|-----------|--------|-------------|
| **Physical** | 5 | V8 Pipeline | Body silhouette area (m²), shoulder width (m), hip width (m), torso length (m), height (m) |
| **Deep** | 1536 | EfficientNet-B3 | Appearance features from background-removed, pose-cropped body image |
| **Pose** | 24 | MediaPipe Landmarks | Scale-invariant skeletal proportions (widths, limb ratios, cross-body diagonals) |

### Ensemble Model

```
Input (1565-D) → StandardScaler → ┬─ Ridge Regression (α=100)  ─┬→ 50/50 Average → Weight (kg)
                                   └─ XGBoost (1000 trees, lr=0.03) ─┘
```

- **Ridge** provides smooth, stable predictions across the feature space
- **XGBoost** captures non-linear relationships and handles domain-shifted features via tree logic
- The 50/50 ensemble balances smoothness (Ridge) with sharpness (XGBoost)

### Training Strategy

1. **Multi-dataset merging** — Combines 2DImage2BMI (controlled photos) with Celeb-FBI (in-the-wild photos)
2. **Histogram flattening** — Oversamples underrepresented weight ranges so every 10 kg bucket has ~2000 samples
3. **No PCA** — Raw 1565-D features are used directly; PCA was found to smear domain-shifted features
4. **Strict validation** — Test set is exclusively from 2DImage2BMI to maintain benchmark comparability

### Post-Processing Calibration

```python
# Linear correction for high-weight underestimation
if raw_pred > 70:
    calibrated = 70 + (raw_pred - 70) * 1.25
else:
    calibrated = raw_pred * 0.95

# Area-based sanity bounds (kg/m² density check)
lower_bound = area_m2 * 105   # very lean
upper_bound = area_m2 * 185   # very heavy
pred = clip(calibrated, lower_bound, upper_bound)
```

---

## Evaluation Results

### Per-Bucket Weight Prediction Analysis

| Weight Range | Description | Metric |
|-------------|-------------|--------|
| 20–50 kg | Underweight | Per-bucket MAE reported |
| 50–70 kg | Normal weight | Per-bucket MAE reported |
| 70–90 kg | Overweight | Per-bucket MAE reported |
| 90–120 kg | Obese | Per-bucket MAE reported |
| 120+ kg | Severely obese | Per-bucket MAE reported |

### Model Variants Comparison

| Model | Features | Test MAE (kg) |
|-------|----------|---------------|
| V1 (KRR + Anthropometric) | 13-D | Baseline |
| V7 (Ridge + Deep + Pose) | 1560-D | Improved |
| V8 (Ridge + Physics) | 5-D | Physical baseline |
| **V9 Hybrid (Ridge + XGB)** | **1565-D** | **Best** |

> **Note:** Exact metric values depend on the specific training run and calibration constants. Run `train_v9_hybrid.py` to reproduce and view detailed per-bucket results.

---

## Technologies Used

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.10+ | Core development |
| **Deep Learning** | PyTorch, EfficientNet-B3 | Deep feature extraction (1536-D) |
| **Pose Estimation** | MediaPipe Pose Landmarker (Heavy) | 33 landmarks + segmentation mask |
| **Object Detection** | YOLOv8s (Ultralytics) | Person detection and cropping |
| **Computer Vision** | OpenCV 4.8+ | ArUco detection, image processing |
| **ML Regression** | scikit-learn (Ridge), XGBoost | Weight prediction ensemble |
| **Web Framework** | Flask + Flask-SocketIO | Real-time web dashboard |
| **Real-time Comms** | WebSocket (SocketIO + Eventlet) | Streaming inference results |
| **Data Science** | NumPy, Pandas, Matplotlib | Data processing, EDA, visualization |
| **Notebooks** | Jupyter | Prototyping and analysis |

---

## Future Work

- **Mobile deployment** — Convert to TensorFlow Lite or ONNX for on-device inference
- **Raspberry Pi optimisation** — Profile and optimise the V8 physics-only pipeline for edge deployment
- **Multi-view estimation** — Use front + side images for improved volume estimation
- **Longitudinal tracking** — Track body composition changes over time per individual
- **Clinical validation** — Evaluate against DEXA scan ground truth for body fat accuracy
- **End-to-end deep learning** — Replace the handcrafted physics features with a learned representation

---

## License

This project was developed as an **academic Final Year Project**. All rights reserved. The code is provided for educational and research purposes. Please contact the author for any commercial use inquiries.

---

## Acknowledgements

- [**2DImage2BMI**](https://github.com/) — Primary dataset for full-body BMI estimation from 2D images
- [**MediaPipe**](https://mediapipe.dev/) — Google's on-device ML framework for pose estimation and segmentation
- [**EfficientNet**](https://arxiv.org/abs/1905.11946) — Tan & Le, 2019 — Efficient CNN architecture for feature extraction
- [**YOLOv8**](https://github.com/ultralytics/ultralytics) — Ultralytics — State-of-the-art real-time object detection
- [**XGBoost**](https://xgboost.readthedocs.io/) — Chen & Guestrin, 2016 — Scalable gradient boosting

---

<p align="center">
  <b>Built as a Final Year Project</b>
</p>
