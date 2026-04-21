import os
import random
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp

from project_paths import project_root
from live_bmi_demo_v9 import (
    load_v9_model, get_task_model, create_pose_landmarker,
    extract_physics, extract_deep_features, extract_pose_features,
    predict_weight, _init_effnet
)

def parse_celeb_filename(fname: str):
    # Format: 1021_5.5h_51w_female_26a.png
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
    bmi = w_kg / (h_m ** 2)
    
    # Ensure 0-weight/implausible images skip
    if not (1.0 < h_m < 2.5) or not (20 < w_kg < 300):
        raise ValueError(f"Implausible height {h_m} / weight {w_kg}")
        
    return h_m, w_kg, bmi

def main():
    print("Loading V9 Model...")
    bundle = load_v9_model()
    if bundle is None:
        return
        
    is_hybrid = bundle.get("feature_type", "physical_v8") == "hybrid_v9"
    if is_hybrid:
        print("Model is HYBRID. Initializing EfficientNet...")
        _init_effnet()
        
    print("Initializing MediaPipe...")
    landmarker = create_pose_landmarker(get_task_model())
    
    # Point directly to the Celeb-FBI dataset
    img_dir = project_root() / "data (1)" / "Celeb-FBI Dataset"
    if not img_dir.exists():
        print(f"Celeb directory not found: {img_dir}")
        return
        
    fnames = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not fnames:
        print(f"No images found in {img_dir}")
        return
        
    while True:
        fname = random.choice(fnames)
        path = os.path.join(img_dir, fname)
        
        try:
            true_h_m, true_w_kg, true_bmi = parse_celeb_filename(fname)
            break
        except Exception as e:
            # Skip unparseable files
            continue
        
    print(f"\n[SELECTED CELEB IMAGE]")
    print(f"File: {fname}")
    print(f"True Height: {true_h_m:.2f} m")
    print(f"True Weight: {true_w_kg:.1f} kg")
    print(f"True BMI:    {true_bmi:.1f}")
    print("-" * 30)

    frame = cv2.imread(path)
    if frame is None:
        print("Could not read image.")
        return

    # MediaPipe detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect(mp_image)
    
    if not res or not res.pose_landmarks:
        print("No person detected by MediaPipe. Run script again for new image.")
        return

    phys_feats, coords, seg_mask = extract_physics(frame, res, true_h_m)
    if phys_feats is None:
        print("Failed to extract physical features. Run script again for new image.")
        return

    if is_hybrid:
        h_frame, w_frame = frame.shape[:2]
        xs, ys = coords[:, 0], coords[:, 1]
        mg = 0.20 # 20% margin
        bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
        x1 = max(0, int(xs.min() - bw * mg / 2))
        x2 = min(w_frame, int(xs.max() + bw * mg / 2))
        y1 = max(0, int(ys.min() - bh * mg / 2))
        y2 = min(h_frame, int(ys.max() + bh * mg / 2))
        crop = frame[y1:y2, x1:x2]

        if crop.size > 0:
            if seg_mask is not None:
                mask_crop = seg_mask[y1:y2, x1:x2]
                mask_3ch = (mask_crop > 0.5).astype(np.uint8)[:, :, None]
                crop = crop * mask_3ch

            deep_feat = extract_deep_features(crop)
            pose_feat = extract_pose_features(coords)
            
            combined = np.concatenate([phys_feats, deep_feat, pose_feat])
            pred_w_kg = predict_weight(bundle, combined)
        else:
            pred_w_kg = predict_weight(bundle, phys_feats)
    else:
        pred_w_kg = predict_weight(bundle, phys_feats)

    pred_bmi = pred_w_kg / (true_h_m ** 2)

    print(f"\n[ESTIMATION RESULTS]")
    print(f"Estimated Weight: {pred_w_kg:.1f} kg")
    print(f"Estimated BMI:    {pred_bmi:.1f}")
    print(f"Weight Error:     {(pred_w_kg - true_w_kg):+.1f} kg")
    print(f"BMI Error:        {(pred_bmi - true_bmi):+.1f}")
    print("=" * 30)

    # Draw on image
    disp = frame.copy()
    
    # Draw landmarks
    landmarks = res.pose_landmarks[0]
    for lm in landmarks:
        cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
        cv2.circle(disp, (cx, cy), 4, (0, 255, 0), -1)

    # Info HUD
    cv2.rectangle(disp, (10, 10), (450, 150), (20, 20, 40), -1)
    
    cv2.putText(disp, f"True W/BMI: {true_w_kg:.1f}kg / {true_bmi:.1f}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
    cv2.putText(disp, f"Pred W/BMI: {pred_w_kg:.1f}kg / {pred_bmi:.1f}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
    
    color = (0, 255, 0) if abs(pred_w_kg - true_w_kg) <= 5 else (0, 150, 255)
    if abs(pred_w_kg - true_w_kg) > 10: color = (0, 0, 255)
    
    cv2.putText(disp, f"Err : {pred_w_kg - true_w_kg:+.1f}kg", (20, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Scale down for viewing if too large
    max_h = 800
    if disp.shape[0] > max_h:
        scale = max_h / disp.shape[0]
        disp = cv2.resize(disp, (int(disp.shape[1] * scale), max_h))

    cv2.imshow("V9 Celeb Random Test", disp)
    print("\nPress any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
