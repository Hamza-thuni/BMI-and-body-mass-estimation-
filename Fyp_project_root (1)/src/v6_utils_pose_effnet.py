import cv2
import numpy as np
import mediapipe as mp
import torch
from torchvision import transforms
from PIL import Image
from deep_features_v3 import EfficientNetFeatureExtractor

try:
    mp_pose = mp.solutions.pose
except AttributeError:
    import sys
    print("\n[ENVIRONMENT ERROR] Your 'mediapipe' installation is corrupted or missing the 'solutions' module.")
    print(f"Python is loading mediapipe from: {getattr(mp, '__file__', 'Unknown')}")
    print("To fix this, please run the following command in this terminal:")
    print("    pip uninstall -y mediapipe")
    print("    pip install mediapipe\n")
    sys.exit(1)

# ---------- Device & Deep model ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

deep_model = EfficientNetFeatureExtractor(pretrained=True).to(device)
deep_model.eval()

preprocess = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def run_pose(image_bgr):
    """Run MediaPipe Pose and return landmarks (or None)."""
    h, w = image_bgr.shape[:2]
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as pose:

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if not res.pose_landmarks:
            return None, None

        pts = res.pose_landmarks.landmark
        coords = np.array([[p.x * w, p.y * h, p.z] for p in pts], dtype=np.float32)
        return coords, res


def make_pose_box(coords, img_w, img_h, margin=0.20):
    xs = coords[:, 0]
    ys = coords[:, 1]
    x1, x2 = max(0, xs.min()), min(img_w - 1, xs.max())
    y1, y2 = max(0, ys.min()), min(img_h - 1, ys.max())

    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    side = max(w, h) * (1.0 + margin)

    x1b = int(max(0, cx - side / 2))
    x2b = int(min(img_w - 1, cx + side / 2))
    y1b = int(max(0, cy - side / 2))
    y2b = int(min(img_h - 1, cy + side / 2))

    if x2b <= x1b or y2b <= y1b:
        return None

    return x1b, y1b, x2b, y2b


def extract_pose_features(coords):
    """Scale-invariant ratios vs body height (13-D after extension; retrain v6 if changed)."""
    NOSE = 0
    L_SH = 11
    R_SH = 12
    L_ELB = 13
    R_ELB = 14
    L_WR = 15
    R_WR = 16
    L_HIP = 23
    R_HIP = 24
    L_KNEE = 25
    R_KNEE = 26
    L_ANK = 27
    R_ANK = 28

    mid_sh = (coords[L_SH, :2] + coords[R_SH, :2]) / 2
    mid_ank = (coords[L_ANK, :2] + coords[R_ANK, :2]) / 2
    body_h = np.linalg.norm(mid_sh - mid_ank) + 1e-6

    def rd(a, b):
        return np.linalg.norm(coords[a, :2] - coords[b, :2]) / body_h

    feats = [
        rd(L_SH, R_SH),
        rd(L_HIP, R_HIP),
        rd(L_SH, L_HIP),
        rd(R_SH, R_HIP),
        rd(L_HIP, L_ANK),
        rd(R_HIP, R_ANK),
        rd(L_SH, L_ELB) + rd(L_ELB, L_WR),
        rd(R_SH, R_ELB) + rd(R_ELB, R_WR),
        rd(L_HIP, L_KNEE),
        rd(R_HIP, R_KNEE),
        rd(L_KNEE, L_ANK),
        rd(R_KNEE, R_ANK),
        rd(L_SH, R_HIP),
        rd(R_SH, L_HIP),
        rd(NOSE, L_SH),
        rd(NOSE, R_SH),
    ]

    return np.array(feats, dtype=np.float32)


def pose_coords_full_image(bgr: np.ndarray, yolo=None) -> np.ndarray | None:
    """
    Landmarks in full-image pixel space. Tries full-frame pose, then YOLO crop + pose.
    Pass a shared YOLOPersonDetector to avoid reloading weights on every call.
    """
    coords, _ = run_pose(bgr)
    if coords is not None:
        return coords

    if yolo is None:
        from person_detector_yolo_v4 import YOLOPersonDetector

        yolo = YOLOPersonDetector(conf=0.4)

    roi = yolo.detect_person_roi(bgr)
    if roi is None:
        return None
    x1, y1, x2, y2 = roi
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    coords_c, _ = run_pose(crop)
    if coords_c is None:
        return None
    out = coords_c.copy()
    out[:, 0] += float(x1)
    out[:, 1] += float(y1)
    return out.astype(np.float32)


def extract_deep_features(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    t = preprocess(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = deep_model(t).cpu().numpy().flatten()

    return feat
