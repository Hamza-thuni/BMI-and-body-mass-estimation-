"""
train_v8_physical.py
--------------------
V8: The Physics Model Trainer.

Trains an ultra-lightweight XGBoost and Ridge model on physically grounded
dimensions (Area, Shoulder Width, Hip Width, Torso, True Height) to predict
exact Weight in kg.

Because we are dealing with pure physics/geometry correlation with body mass,
the model does not need 1500+ deep neural network features. It only needs 5.

This model is < 1MB and executes in microseconds, perfect for Raspberry Pi.

Usage:
  python train_v8_physical.py
"""

import os
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed. Falling back to Ridge only.")

from project_paths import features_dir, models_dir


def load_data(split, base_dir):
    path = os.path.join(base_dir, f"{split}_v8_phys.npz")
    data = np.load(path)
    X = data["features"]  # [area_m2, sh_w_m, hip_w_m, torso_l_m, height_m]
    y = data["weight"]    # target: Weight (kg)
    return X, y

def report(name, preds, y_true):
    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mape = np.mean(np.abs((y_true - preds) / (y_true + 1e-6))) * 100
    print(f"  {name:<15s} | MAE={mae:.2f} kg | RMSE={rmse:.2f} kg | MAPE={mape:.1f}%")

if __name__ == "__main__":
    feat_dir = str(features_dir("v8"))
    out_dir  = models_dir("v8")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir  = str(out_dir)

    print("=== V8 Physical Model Training ===")
    
    X_train, y_train = load_data("train", feat_dir)
    X_val,   y_val   = load_data("val",   feat_dir)
    X_test,  y_test  = load_data("test",  feat_dir)
    
    print(f"Loaded train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    
    # Preprocessing
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    
    # 1. Train Ridge Baseline
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_s, y_train)
    
    print("\n--- Validation ---")
    report("Ridge", ridge.predict(X_val_s), y_val)
    
    xgb_model = None
    if HAS_XGB:
        xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
        xgb_model.fit(X_train_s, y_train)
        report("XGBoost", xgb_model.predict(X_val_s), y_val)
        
    print("\n--- Test Set ---")
    r_te = ridge.predict(X_test_s)
    report("Ridge", r_te, y_test)
    
    if xgb_model:
        x_te = xgb_model.predict(X_test_s)
        report("XGBoost", x_te, y_test)
        preds = (r_te + x_te) / 2.0
        report("Ensemble", preds, y_test)
    
    # --- Final Training on Train+Val ---
    print("\nTraining final models on Train+Val...")
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])
    
    scaler_final = StandardScaler()
    X_tv_s = scaler_final.fit_transform(X_tv)
    
    ridge_final = Ridge(alpha=1.0)
    ridge_final.fit(X_tv_s, y_tv)
    
    xgb_final = None
    if HAS_XGB:
        xgb_final = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
        xgb_final.fit(X_tv_s, y_tv)
        
    bundle = {
        "scaler": scaler_final,
        "ridge": ridge_final,
        "xgb": xgb_final,
        "feature_type": "physical_v8"
    }
    
    save_path = os.path.join(out_dir, "ensemble_v8_physical.joblib")
    joblib.dump(bundle, save_path)
    print(f"\n✅ Saved to {save_path}")
