import sys
import os
import cv2
import numpy as np

# Add the src dir to sys.path so we can import from live_bmi_demo_v9
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Fyp_project_root (1)", "src"))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import live_bmi_demo_v9 as v9
import mediapipe as mp

# Physical setup — match these to your real-world marker placement
MARKER_BOTTOM_HEIGHT_M   = 0.80   # center of marker ID 0 is 80cm from floor
MARKER_TOP_HEIGHT_M      = 1.80   # center of marker ID 1 is 180cm from floor
MARKER_SEPARATION_M      = MARKER_TOP_HEIGHT_M - MARKER_BOTTOM_HEIGHT_M  # 1.0 m

class RealV9Pipeline:
    def __init__(self):
        print("[RealV9Pipeline] Initializing pipeline...")
        # 1. Load the model bundle
        self.bundle = v9.load_v9_model()
        if not self.bundle:
            raise RuntimeError("Could not load V9 Model. Ensure it is trained.")
        
        # 2. Check hybrid vs physical
        feat_type = self.bundle.get("feature_type", "physical_v8")
        self.is_hybrid = feat_type == "hybrid_v9"
        if self.is_hybrid:
            print("[RealV9Pipeline] Initializing EfficientNet for Hybrid mode...")
            v9._init_effnet()
            
        # 3. Load MediaPipe Task Model
        # Change cwd to SRC_DIR temporarily so it finds/downloads the model there
        old_cwd = os.getcwd()
        os.chdir(SRC_DIR)
        model_path = v9.get_task_model()
        if not os.path.isabs(model_path):
            model_path = os.path.join(SRC_DIR, model_path)
        os.chdir(old_cwd)
            
        self.landmarker = v9.create_pose_landmarker(model_path)
        print("[RealV9Pipeline] Initialization complete.")

    def calculate_scale(self, frame):
        """Finds ArUco markers and returns px_per_m if both found, else None"""
        corners, ids = v9.detect_aruco(frame)
        if ids is not None:
            ids_flat = ids.flatten()
            if 0 in ids_flat and 1 in ids_flat:
                idx0 = np.where(ids_flat == 0)[0][0]
                idx1 = np.where(ids_flat == 1)[0][0]
                
                center0 = corners[idx0][0].mean(axis=0)   # (x, y) of ID 0 — bottom
                center1 = corners[idx1][0].mean(axis=0)   # (x, y) of ID 1 — top
                
                pixel_separation = abs(center1[1] - center0[1])
                if pixel_separation > 20:
                    px_per_m = pixel_separation / MARKER_SEPARATION_M
                    return px_per_m
        return None

    def predict(self, frame, age, sex_is_male, px_per_m, parallax_factor):
        """Runs the V9 prediction given a frame and parallax compensation."""
        if frame is None:
            raise ValueError("No frame provided")
        if px_per_m is None or px_per_m <= 0:
            raise ValueError("Scale is invalid. Ensure both ArUco markers are visible.")

        # Adjust scale for parallax (person is closer to camera than the wall)
        person_px_per_m = px_per_m / parallax_factor

        h_frame, w_frame = frame.shape[:2]
        
        # 1. MediaPipe extraction
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        try:
            res = self.landmarker.detect(mp_image)
        except Exception as e:
            raise RuntimeError(f"MediaPipe error: {e}")
            
        if not res or not res.pose_landmarks:
            raise RuntimeError("No person detected in frame.")
            
        # 2. Extract Physical Features
        phys_feats, coords, seg_mask, is_cut_off = v9.extract_physics(frame, res, px_per_m)
        if phys_feats is None:
            raise RuntimeError(f"Could not extract physical features (cut_off={is_cut_off}). Try stepping further back.")
            
        # 3. Deep / Pose features (if hybrid)
        if self.is_hybrid:
            xs, ys = coords[:, 0], coords[:, 1]
            mg = 0.20
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

                deep_feat = v9.extract_deep_features(crop)
                pose_feat = v9.extract_pose_features(coords)

                combined = np.concatenate([phys_feats, deep_feat, pose_feat])
                pred_w_kg = v9.predict_weight(self.bundle, combined)
            else:
                pred_w_kg = v9.predict_weight(self.bundle, phys_feats)
        else:
            pred_w_kg = v9.predict_weight(self.bundle, phys_feats)
            
        # 4. Results
        height_m = float(phys_feats[4])
        weight_kg = float(pred_w_kg)
        if height_m <= 0:
            raise RuntimeError("Calculated height is zero or negative.")
            
        bmi = weight_kg / (height_m ** 2)
        
        sex_val = 1 if sex_is_male else 0
        bfp = (1.20 * bmi) + (0.23 * age) - (10.8 * sex_val) - 5.4
        bfp = max(2.0, min(60.0, float(bfp)))
        
        if bmi < 18.5: cat = 'Underweight'
        elif bmi < 25: cat = 'Healthy'
        elif bmi < 30: cat = 'Overweight'
        else: cat = 'Obese'
        
        fat_mass = weight_kg * (bfp / 100.0)
        lean_mass = weight_kg - fat_mass
        
        return {
            'height_m': round(height_m, 2),
            'weight_kg': round(weight_kg, 1),
            'bmi': round(bmi, 1),
            'body_fat_pct': round(bfp, 1),
            'fat_mass': round(fat_mass, 1),
            'lean_mass': round(lean_mass, 1),
            'category': cat,
            'is_cut_off': is_cut_off
        }
