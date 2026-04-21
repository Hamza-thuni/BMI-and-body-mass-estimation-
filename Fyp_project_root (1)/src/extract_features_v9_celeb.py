import os
import re
import cv2
import numpy as np
import urllib.request
import torch
import mediapipe as mp
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from project_paths import project_root, features_dir
from v7_pose_features import extract_rich_pose_features
from deep_features_v3 import EfficientNetFeatureExtractor

# --- MediaPipe Model Download ---
def get_task_model():
    model_path = "pose_landmarker_heavy.task"
    if not os.path.exists(model_path):
        print(f"\n[INIT] Downloading MediaPipe Task Model ({model_path})...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        urllib.request.urlretrieve(url, model_path)
    return model_path

# --- Parsing ---
def parse_celeb_filename(fname: str):
    # e.g., 1021_5.5h_51w_female_26a.png
    name = os.path.splitext(fname)[0]
    parts = name.split("_")
    
    if len(parts) < 3:
        raise ValueError(f"Invalid format: {fname}")
        
    h_str = parts[1] # '5.5h'
    w_str = parts[2] # '51w'
    
    if not h_str.endswith('h') or not w_str.endswith('w'):
        raise ValueError("Missing 'h' or 'w' suffix")
        
    h_val = h_str[:-1]
    w_val = w_str[:-1]
    
    # height parsing: feet.inches => meters
    ft_in = h_val.split('.')
    if len(ft_in) == 1:
        ft = float(ft_in[0])
        inches = 0
    else:
        ft = float(ft_in[0])
        inches = float(ft_in[1])
        
    h_m = (ft * 12 + inches) * 0.0254
    w_kg = float(w_val)
    
    if not (1.0 < h_m < 2.5) or not (20 < w_kg < 300):
        raise ValueError(f"Implausible: height {h_m}m, weight {w_kg}kg")
        
    return h_m, w_kg

# --- Transforms ---
_PREPROCESS = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_deep_features(bgr_crop, effnet_model, device):
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    t = _PREPROCESS(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = effnet_model(t).cpu().numpy().flatten()
    return feat

def safe_read_rgb(img_path: str, target_size: int = 640):
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None
    orig_h, orig_w = bgr.shape[:2]
    scale = target_size / max(orig_h, orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    bgr_resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = bgr_resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    return rgb, (orig_h, orig_w, pad_top, pad_left, scale)

def process_celeb_image(img_path, true_h_m, landmarker, effnet_model, device):
    rgb, scale_info = safe_read_rgb(img_path, target_size=640)
    if rgb is None:
        return None
    orig_h, orig_w, pad_top, pad_left, scale = scale_info
    img_size = rgb.shape[0]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try:
        res = landmarker.detect(mp_image)
    except Exception:
        return None

    if not res.pose_landmarks or not res.segmentation_masks:
        return None

    landmarks = res.pose_landmarks[0]
    pts_padded = np.array(
        [[lm.x * img_size, lm.y * img_size] for lm in landmarks],
        dtype=np.float32
    )

    pts_orig = np.zeros_like(pts_padded)
    pts_orig[:, 0] = (pts_padded[:, 0] - pad_left) / scale
    pts_orig[:, 1] = (pts_padded[:, 1] - pad_top) / scale

    min_y, max_y = pts_orig[:, 1].min(), pts_orig[:, 1].max()
    pixel_height = max_y - min_y
    if pixel_height < 50:
        return None

    px_per_m = pixel_height / true_h_m
    mask = res.segmentation_masks[0].numpy_view()
    mask_pixels_padded = np.sum(mask > 0.5)
    mask_pixels_orig = mask_pixels_padded / (scale ** 2)
    area_m2 = mask_pixels_orig / (px_per_m ** 2)

    L_SH, R_SH = 11, 12
    L_HIP, R_HIP = 23, 24
    sh_w_px = np.linalg.norm(pts_orig[L_SH] - pts_orig[R_SH])
    hip_w_px = np.linalg.norm(pts_orig[L_HIP] - pts_orig[R_HIP])
    mid_sh = (pts_orig[L_SH] + pts_orig[R_SH]) / 2.0
    mid_hip = (pts_orig[L_HIP] + pts_orig[R_HIP]) / 2.0
    torso_l_px = np.linalg.norm(mid_sh - mid_hip)

    sh_w_m = sh_w_px / px_per_m
    hip_w_m = hip_w_px / px_per_m
    torso_l_m = torso_l_px / px_per_m

    phys_5D = np.array([area_m2, sh_w_m, hip_w_m, torso_l_m, true_h_m], dtype=np.float32)

    # Pose 24D
    coords = np.column_stack([pts_orig, np.zeros(len(pts_orig))])
    pose_24D = extract_rich_pose_features(coords)

    # Deep 1536D
    frame = cv2.imread(img_path)
    h_f, w_f = frame.shape[:2]
    
    xs, ys = coords[:, 0], coords[:, 1]
    mg = 0.20
    bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
    x1 = max(0, int(xs.min() - bw * mg / 2))
    x2 = min(w_f, int(xs.max() + bw * mg / 2))
    y1 = max(0, int(ys.min() - bh * mg / 2))
    y2 = min(h_f, int(ys.max() + bh * mg / 2))
    
    crop = frame[y1:y2, x1:x2]
    
    if crop.size > 0:
        mask_cropped = mask[pad_top:pad_top+int(orig_h*scale), pad_left:pad_left+int(orig_w*scale)]
        mask_orig = cv2.resize((mask_cropped > 0.5).astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        mask_crop = mask_orig[y1:y2, x1:x2]
        crop = crop * mask_crop[:, :, None]
        deep_1536D = extract_deep_features(crop, effnet_model, device)
    else:
        deep_1536D = np.zeros(1536, dtype=np.float32)
        
    return np.concatenate([phys_5D, deep_1536D, pose_24D])

def extract_celeb_split(img_dir: str, landmarker, effnet_model, device):
    fnames = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    X, Y, names = [], [], []
    skipped = 0

    print(f"\nExtracting V9 Hybrid Features for Celeb-FBI ({len(fnames)} images) ...")
    for fname in tqdm(fnames, desc="Celeb-FBI"):
        try:
            h_m, w_kg = parse_celeb_filename(fname)
        except ValueError:
            skipped += 1
            continue

        path = os.path.join(img_dir, fname)
        try:
            feats = process_celeb_image(path, h_m, landmarker, effnet_model, device)
        except Exception as e:
            skipped += 1
            continue

        if feats is not None:
            X.append(feats)
            Y.append(w_kg)
            names.append(fname)
        else:
            skipped += 1

    print(f"  -> {len(names)} samples valid, {skipped} skipped.")
    return (
        np.array(X, dtype=np.float32),
        np.array(Y, dtype=np.float32),
        np.array(names)
    )

if __name__ == "__main__":
    celeb_data_dir = str(project_root() / "data (1)" / "Celeb-FBI Dataset")
    out_dir = project_root() / "features_v9"
    os.makedirs(out_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    effnet_model = EfficientNetFeatureExtractor(pretrained=True).to(device)
    effnet_model.eval()

    model_path = get_task_model()
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        output_segmentation_masks=True,
        min_pose_detection_confidence=0.5
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        X, Y, fnames = extract_celeb_split(celeb_data_dir, landmarker, effnet_model, device)
        if len(X) > 0:
            out_path = os.path.join(out_dir, "celeb_v9.npz")
            np.savez_compressed(out_path, features=X, weight=Y, names=fnames)
            print(f"  Saved Celeb-FBI: features={X.shape}, weight={Y.shape}")
        else:
            print("  [Error] No valid features extracted for Celeb-FBI.")

    print("\nV9 extraction complete.")
