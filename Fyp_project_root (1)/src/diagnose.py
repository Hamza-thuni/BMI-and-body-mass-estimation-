import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
import sys
import os

# Add src to path to import project_paths
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    from project_paths import models_dir, features_dir
except ImportError:
    # Fallback if project_paths is not found
    def models_dir(v): return Path(r"D:\Fyp_project_root\Fyp_project_root (1)\models_v9")
    def features_dir(v): return Path(r"D:\Fyp_project_root\Fyp_project_root (1)\features_v9")

# --- UPDATED PATHS ---
MODEL_PATH = models_dir("v9") / "ensemble_v9_hybrid.joblib"
NPZ_PATH = features_dir("v9") / "celeb_v9.npz"

def predict_ensemble(bundle, X_scaled, X_orig=None):
    """Predict using the ensemble logic (Ridge + XGB) from the bundle."""
    w_ridge = bundle["ridge"].predict(X_scaled)
    
    if "xgb" in bundle and bundle["xgb"] is not None:
        w_xgb = bundle["xgb"].predict(X_scaled)
        # XGB might return a different shape or need flattening
        if hasattr(w_xgb, 'flatten'):
            w_xgb = w_xgb.flatten()
        preds = (w_ridge * 0.5) + (w_xgb * 0.5)
    else:
        preds = w_ridge

    if X_orig is not None:
        # ------------------------------------------------------------------
        # V9-EMERGENCY PATCH: Applied Calibration
        # ------------------------------------------------------------------
        # 1. Linear Correction (Slope Adjustment)
        calibrated = np.where(preds > 70, 70 + (preds - 70) * 1.25, preds * 0.95)
        
        # 2. Physical Sanity Check (Hard Floor/Ceiling)
        area_m2 = X_orig[:, 0]
        lower_bound = area_m2 * 105
        upper_bound = area_m2 * 185
        
        preds = np.clip(calibrated, lower_bound, upper_bound)
    
    return preds

def run_diagnosis():
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model file not found at: {MODEL_PATH}")
        return

    # 1. Load the model bundle
    print(f"[INIT] Loading model: {MODEL_PATH.name}...")
    bundle = joblib.load(str(MODEL_PATH))
    
    # 2. Attempt to load the dataset
    if NPZ_PATH.exists():
        print(f"[INIT] Loading features from NPZ: {NPZ_PATH.name}...")
        data = np.load(str(NPZ_PATH))
        X = data['features']
        y = data['weight']
    else:
        print(f"[ERROR] Features NPZ not found at: {NPZ_PATH}")
        return

    # 3. Split data to check generalization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply scaling if present in bundle
    if "scaler" in bundle and bundle["scaler"] is not None:
        X_train_scaled = bundle["scaler"].transform(X_train)
        X_test_scaled = bundle["scaler"].transform(X_test)
    else:
        X_train_scaled, X_test_scaled = X_train, X_test

    # 4. Predict using the full Ensemble
    print("[RUN] Generating predictions...")
    train_preds = predict_ensemble(bundle, X_train_scaled, X_train)
    test_preds = predict_ensemble(bundle, X_test_scaled, X_test)

    # --- VISUALIZATION ---
    print("[PLOT] Generating diagnostic charts...")
    fig = plt.figure(figsize=(15, 10))

    # PLOT 1: Error Distribution
    plt.subplot(2, 2, 1)
    errors = y_test - test_preds
    sns.histplot(errors, kde=True, color="skyblue")
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title("Error Distribution (Actual - Predicted)")
    plt.xlabel("Error (kg)")
    plt.ylabel("Frequency")

    # PLOT 2: Residuals
    plt.subplot(2, 2, 2)
    plt.scatter(test_preds, errors, alpha=0.5, color='blue')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title("Residual Plot")
    plt.xlabel("Predicted Weight (kg)")
    plt.ylabel("Error (kg)")

    # PLOT 3: Actual vs Predicted
    plt.subplot(2, 2, 3)
    plt.scatter(y_test, test_preds, alpha=0.5, color='purple')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.title("Actual vs Predicted Weight")
    plt.xlabel("True Weight (kg)")
    plt.ylabel("Predicted Weight (kg)")

    # PLOT 4: Feature Importance (From Ridge component)
    plt.subplot(2, 2, 4)
    model_for_imp = bundle["ridge"]
    if hasattr(model_for_imp, 'coef_'):
        importance = np.abs(model_for_imp.coef_)
        phys_names = ["Area", "Sh-Width", "Hip-Width", "Torso-L", "Height"]
        top_n = 15
        top_idx = np.argsort(importance)[::-1][:top_n]
        
        display_names = []
        for idx in top_idx:
            if idx < 5: display_names.append(phys_names[idx])
            elif idx < 5 + 1536: display_names.append(f"Deep_{idx-5}")
            else: display_names.append(f"Pose_{idx-5-1536}")
            
        sns.barplot(x=importance[top_idx], y=display_names)
        plt.title(f"Top {top_n} Feature Importance (Weights)")

    plt.tight_layout()
    
    # Save output
    output_png = _THIS_DIR.parent / "diagnose_results.png"
    plt.savefig(str(output_png))
    print(f"[SAVE] Diagnostic plots saved to: {output_png}")
    
    plt.show()

    print("\n" + "="*30)
    print(f"TRAIN MAE: {mean_absolute_error(y_train, train_preds):.2f} kg")
    print(f"TEST MAE:  {mean_absolute_error(y_test, test_preds):.2f} kg")
    print(f"R2 SCORE:  {r2_score(y_test, test_preds):.4f}")
    print("="*30)

if __name__ == "__main__":
    run_diagnosis()
