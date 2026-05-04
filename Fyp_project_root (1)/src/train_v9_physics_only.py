"""
train_v9_physics_only.py
------------------------
V9 Physics-Only Model — optimised for LIVE DEMO accuracy.

Uses ONLY the 5 physical features + engineered body-ratio features.
No deep features = no domain gap between training photos and live webcam.

Features (5 raw + 8 engineered = 13 total):
  Raw:  area_m2, sh_w_m, hip_w_m, torso_l_m, true_h_m
  Eng:  bmi_proxy, sh_hip_ratio, area_per_h, volume_proxy,
        trunk_avg, torso_ratio, compactness, width_area_ratio

Trains: Ridge + XGBoost ensemble on 2DImage2BMI + Celeb-FBI physics.
Tests:  2DImage2BMI test split only.
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
    print("XGBoost not installed. Falling back to Ridge only.")

from project_paths import features_dir, models_dir, project_root


# ---------------------------------------------------------------------------
# Engineered physics features
# ---------------------------------------------------------------------------
def engineer_features(X_raw):
    """
    From 5 raw physics features, create 13 total features that encode
    body proportions the linear model can't discover on its own.
    
    Input:  X_raw (N, 5) = [area_m2, sh_w_m, hip_w_m, torso_l_m, true_h_m]
    Output: X_eng (N, 13)
    """
    area    = X_raw[:, 0]
    sh_w    = X_raw[:, 1]
    hip_w   = X_raw[:, 2]
    torso_l = X_raw[:, 3]
    height  = X_raw[:, 4]

    eps = 1e-6  # avoid division by zero

    # Engineered features
    bmi_proxy     = area / (height ** 2 + eps)          # 2D BMI analogue
    sh_hip_ratio  = sh_w / (hip_w + eps)                # V-shape vs pear
    area_per_h    = area / (height + eps)               # area normalised by height
    volume_proxy  = area * (sh_w + hip_w) / 2.0         # crude volume estimate
    trunk_avg     = (sh_w + hip_w) / 2.0                # average trunk width
    torso_ratio   = torso_l / (height + eps)            # torso proportion
    compactness   = area / (sh_w * torso_l + eps)       # how filled-in the silhouette is
    width_area    = (sh_w * hip_w) / (area + eps)       # width product vs area

    X_eng = np.column_stack([
        X_raw,          # 5 original
        bmi_proxy,      # 5
        sh_hip_ratio,   # 6
        area_per_h,     # 7
        volume_proxy,   # 8
        trunk_avg,      # 9
        torso_ratio,    # 10
        compactness,    # 11
        width_area,     # 12
    ])
    return X_eng.astype(np.float32)


FEATURE_NAMES = [
    "area_m2", "sh_w_m", "hip_w_m", "torso_l_m", "height_m",
    "bmi_proxy", "sh_hip_ratio", "area_per_h", "volume_proxy",
    "trunk_avg", "torso_ratio", "compactness", "width_area"
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_v8_physics(split):
    path = features_dir("v8") / f"{split}_v8_phys.npz"
    data = np.load(str(path))
    return data["features"], data["weight"]


def load_celeb_physics():
    path = project_root() / "features_v9" / "celeb_v9.npz"
    if not path.exists():
        print(f"[WARNING] {path} not found.")
        return np.empty((0, 5), dtype=np.float32), np.empty((0,), dtype=np.float32)
    data = np.load(str(path))
    # First 5 columns are the physics features
    return data["features"][:, :5], data["weight"]


# ---------------------------------------------------------------------------
# Distribution balancing
# ---------------------------------------------------------------------------
def balance_distribution(X, y, target_per_bin=800):
    """Oversample underrepresented weight bins to flatten the histogram."""
    bins = np.arange(20, 160, 10)
    indices = np.digitize(y, bins)

    X_bal, y_bal = [], []
    for i in range(1, len(bins)):
        mask = (indices == i)
        X_b, y_b = X[mask], y[mask]
        n = len(X_b)
        if n == 0:
            continue
        if n < target_per_bin:
            reps = target_per_bin // n
            rem = target_per_bin % n
            X_bal.append(np.tile(X_b, (reps, 1)))
            y_bal.append(np.tile(y_b, reps))
            if rem > 0:
                idx = np.random.choice(n, rem, replace=False)
                X_bal.append(X_b[idx])
                y_bal.append(y_b[idx])
        else:
            X_bal.append(X_b)
            y_bal.append(y_b)

    # Keep >150kg as-is
    mask_high = (y >= 150)
    if mask_high.sum() > 0:
        X_bal.append(X[mask_high])
        y_bal.append(y[mask_high])

    return np.vstack(X_bal), np.concatenate(y_bal)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report(name, preds, y_true):
    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mape = np.mean(np.abs((y_true - preds) / (y_true + 1e-6))) * 100
    print(f"  {name:<20s} | MAE={mae:.2f} kg | RMSE={rmse:.2f} kg | MAPE={mape:.1f}%")
    return mae


def report_buckets(preds, y_true):
    print("  Per-bucket analysis:")
    for lo, hi in [(20,50),(50,60),(60,70),(70,80),(80,90),(90,100),(100,120),(120,300)]:
        m = (y_true >= lo) & (y_true < hi)
        if m.sum() > 0:
            delta = preds[m].mean() - y_true[m].mean()
            mae_b = np.mean(np.abs(y_true[m] - preds[m]))
            print(f"    [{lo:3d}-{hi:3d}) kg: n={m.sum():3d} | "
                  f"true={y_true[m].mean():.1f} pred={preds[m].mean():.1f} "
                  f"delta={delta:+.1f} MAE={mae_b:.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print(" V9 PHYSICS-ONLY Model Training")
    print(" (Optimised for Live Demo — No Domain Gap)")
    print("=" * 60)

    # --- 1. Load data ---
    print("\n[1] Loading physics features...")
    X_train_raw, y_train = load_v8_physics("train")
    X_test_raw, y_test = load_v8_physics("test")
    X_celeb_raw, y_celeb = load_celeb_physics()

    print(f"  2DImage2BMI train: {X_train_raw.shape[0]}")
    print(f"  2DImage2BMI test:  {X_test_raw.shape[0]}")
    print(f"  Celeb-FBI:         {X_celeb_raw.shape[0]}")

    # Filter implausible
    valid = (y_train > 20) & (y_train < 300)
    X_train_raw, y_train = X_train_raw[valid], y_train[valid]
    valid = (y_test > 20) & (y_test < 300)
    X_test_raw, y_test = X_test_raw[valid], y_test[valid]
    valid = (y_celeb > 20) & (y_celeb < 300)
    X_celeb_raw, y_celeb = X_celeb_raw[valid], y_celeb[valid]

    # Combine training data
    if len(X_celeb_raw) > 0:
        X_all_raw = np.vstack([X_train_raw, X_celeb_raw])
        y_all = np.concatenate([y_train, y_celeb])
    else:
        X_all_raw = X_train_raw
        y_all = y_train

    print(f"  Combined train:    {X_all_raw.shape[0]}")

    # --- 2. Engineer features ---
    print("\n[2] Engineering 13 physics features...")
    X_train_eng = engineer_features(X_all_raw)
    X_test_eng = engineer_features(X_test_raw)
    print(f"  Feature vector: {X_train_eng.shape[1]}D = {FEATURE_NAMES}")

    # --- 3. Balance distribution ---
    print("\n[3] Balancing weight distribution...")
    X_bal, y_bal = balance_distribution(X_train_eng, y_all, target_per_bin=1500)
    print(f"  Balanced train: {X_bal.shape[0]} samples")

    # --- 4. Scale ---
    print("\n[4] Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_bal)
    X_test_s = scaler.transform(X_test_eng)

    # --- 5. Train Ridge ---
    print("\n[5] Training Ridge (alpha=10)...")
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train_s, y_bal)
    r_pred = ridge.predict(X_test_s)
    report("Ridge", r_pred, y_test)

    # --- 6. Train XGBoost ---
    xgb_model = None
    if HAS_XGB:
        print("\n[6] Training XGBoost...")
        xgb_model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )
        xgb_model.fit(X_train_s, y_bal)
        x_pred = xgb_model.predict(X_test_s)
        report("XGBoost", x_pred, y_test)

        # --- 7. Ensemble ---
        print("\n[7] Ensemble (40% Ridge + 60% XGBoost)...")
        ens_pred = r_pred * 0.4 + x_pred * 0.6
        report("Ensemble", ens_pred, y_test)

        # Try different blends
        for rw in [0.3, 0.5]:
            blend = r_pred * rw + x_pred * (1 - rw)
            mae = mean_absolute_error(y_test, blend)
            print(f"    Blend {rw:.0%}R/{1-rw:.0%}X: MAE={mae:.2f}")
    else:
        ens_pred = r_pred

    # --- 8. Per-bucket analysis ---
    print("\n" + "=" * 60)
    print(" FINAL TEST EVALUATION (Physics-Only V9)")
    print("=" * 60)
    report_buckets(ens_pred, y_test)

    # --- 9. Feature importance ---
    if HAS_XGB and xgb_model is not None:
        print("\n  Feature importance (XGBoost):")
        imp = xgb_model.feature_importances_
        for name, val in sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1]):
            bar = "#" * int(val * 100)
            print(f"    {name:<16s}: {val:.3f} {bar}")

    # --- 10. Save ---
    out_dir = str(models_dir("v9"))
    os.makedirs(out_dir, exist_ok=True)

    bundle = {
        "scaler": scaler,
        "pca": None,
        "ridge": ridge,
        "xgb": xgb_model,
        "feature_type": "physics_only_v9",
        "n_features": X_train_eng.shape[1],
        "feature_names": FEATURE_NAMES,
        "n_pca": None,
    }

    save_path = os.path.join(out_dir, "ensemble_v9_physics.joblib")
    joblib.dump(bundle, save_path)
    print(f"\nSaved -> {save_path}")
    print("Done!")
