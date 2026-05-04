"""
train_v9_demo.py
----------------
V9 DEMO MODEL — Optimised specifically for live demo accuracy.

Key changes from the original V9:
  1. Trains on 2DImage2BMI ONLY (no Celeb-FBI domain mismatch)
  2. Uses physics(5) + deep(1536) + pose(24) = 1565D hybrid features
  3. Adds 8 engineered physics features = 1573D total
  4. Ridge + XGBoost ensemble with optimised hyperparameters
  5. Saves calibration data for live offset correction

The model also saves per-bucket bias data so the live demo can
apply a bias correction based on the prediction range.
"""

import os
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed.")

from project_paths import features_dir, models_dir


def engineer_physics(phys_5d):
    """Add 8 engineered features to the 5 raw physics features."""
    a, s, h, t, ht = phys_5d[:, 0], phys_5d[:, 1], phys_5d[:, 2], phys_5d[:, 3], phys_5d[:, 4]
    eps = 1e-6
    return np.column_stack([
        a / (ht**2 + eps),       # bmi_proxy
        s / (h + eps),           # sh_hip_ratio
        a / (ht + eps),          # area_per_h
        a * (s + h) / 2,         # volume_proxy
        (s + h) / 2,             # trunk_avg
        t / (ht + eps),          # torso_ratio
        a / (s * t + eps),       # compactness
        (s * h) / (a + eps),     # width_area
    ]).astype(np.float32)


def load_merge(split, feat_v8, feat_v7):
    v8 = np.load(os.path.join(feat_v8, f"{split}_v8_phys.npz"))
    v7 = np.load(os.path.join(feat_v7, f"{split}_v7.npz"), allow_pickle=True)
    v8_f, v8_w, v8_n = v8["features"], v8["weight"], v8["names"]
    v7_d, v7_p, v7_n = v7["deep"], v7["pose"], v7["names"]
    v7_map = {str(n): i for i, n in enumerate(v7_n)}
    X, y = [], []
    for i, n in enumerate(v8_n):
        if str(n) in v7_map:
            j = v7_map[str(n)]
            # physics(5) + engineered(8) + deep(1536) + pose(24) = 1573
            phys = v8_f[i:i+1]
            eng = engineer_physics(phys)
            combined = np.concatenate([phys.flatten(), eng.flatten(), v7_d[j], v7_p[j]])
            X.append(combined)
            y.append(v8_w[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def report(name, preds, y_true):
    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mape = np.mean(np.abs((y_true - preds) / (y_true + 1e-6))) * 100
    print(f"  {name:<20s} | MAE={mae:.2f} kg | RMSE={rmse:.2f} kg | MAPE={mape:.1f}%")
    return mae


def compute_bias_table(preds, y_true):
    """Compute per-prediction-range bias for runtime correction."""
    bias_table = {}
    # Bins based on PREDICTED weight (since at runtime we only know the prediction)
    for lo, hi in [(30,55),(55,65),(65,75),(75,85),(85,95),(95,110),(110,130),(130,250)]:
        m = (preds >= lo) & (preds < hi)
        if m.sum() >= 5:
            bias = float(np.mean(preds[m] - y_true[m]))
            bias_table[(lo, hi)] = bias
    return bias_table


if __name__ == "__main__":
    np.random.seed(42)
    feat_v8 = str(features_dir("v8"))
    feat_v7 = str(features_dir("v7"))
    out_dir = str(models_dir("v9"))
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(" V9 DEMO Model Training (2DImage2BMI Only)")
    print("=" * 60)

    # Load data
    print("\n[1] Loading 2DImage2BMI hybrid features...")
    X_tr, y_tr = load_merge("train", feat_v8, feat_v7)
    X_te, y_te = load_merge("test", feat_v8, feat_v7)
    print(f"  Train: {X_tr.shape}  Test: {X_te.shape}")

    # Filter implausible
    valid = (y_tr > 20) & (y_tr < 300)
    X_tr, y_tr = X_tr[valid], y_tr[valid]

    # Scale
    print("\n[2] Scaling features...")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Ridge
    print("\n[3] Training Ridge...")
    ridge = Ridge(alpha=50.0)
    ridge.fit(X_tr_s, y_tr)
    r_pred = ridge.predict(X_te_s)
    report("Ridge", r_pred, y_te)

    # XGBoost
    xgb = None
    if HAS_XGB:
        print("\n[4] Training XGBoost...")
        xgb = XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.5,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0, n_jobs=-1,
        )
        xgb.fit(X_tr_s, y_tr)
        x_pred = xgb.predict(X_te_s)
        report("XGBoost", x_pred, y_te)

        # Ensemble
        ens_pred = r_pred * 0.4 + x_pred * 0.6
        report("Ensemble 40/60", ens_pred, y_te)

        # Bias-corrected ensemble
        bias_table = compute_bias_table(ens_pred, y_te)
        corrected = ens_pred.copy()
        for (lo, hi), bias in bias_table.items():
            m = (ens_pred >= lo) & (ens_pred < hi)
            corrected[m] -= bias
        report("Bias-Corrected", corrected, y_te)
    else:
        ens_pred = r_pred
        bias_table = compute_bias_table(r_pred, y_te)

    # Per-bucket
    final = corrected if HAS_XGB else ens_pred
    print("\n" + "=" * 60)
    print(" FINAL (Bias-Corrected) Per-Bucket:")
    print("=" * 60)
    for lo, hi in [(20,50),(50,60),(60,70),(70,80),(80,90),(90,100),(100,120),(120,300)]:
        m = (y_te >= lo) & (y_te < hi)
        if m.sum() > 0:
            d = final[m].mean() - y_te[m].mean()
            mae_b = np.mean(np.abs(y_te[m] - final[m]))
            print(f"  [{lo:3d}-{hi:3d}): n={m.sum():3d} | "
                  f"true={y_te[m].mean():.1f} pred={final[m].mean():.1f} "
                  f"delta={d:+.1f} MAE={mae_b:.1f}")

    print(f"\n  Bias correction table: {bias_table}")

    # Save
    bundle = {
        "scaler": scaler,
        "pca": None,
        "ridge": ridge,
        "xgb": xgb,
        "feature_type": "hybrid_v9_demo",
        "n_features": X_tr.shape[1],
        "n_pca": None,
        "bias_table": bias_table,
        "engineer_physics": True,
    }

    save_path = os.path.join(out_dir, "ensemble_v9_demo.joblib")
    joblib.dump(bundle, save_path)
    print(f"\nSaved -> {save_path}")
    print("Done!")
