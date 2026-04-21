"""
train_v9_hybrid.py
------------------
V9: Solves extreme bias by combining massive diverse Celeb-FBI data
with the established 2DImage2BMI architecture.

This merges:
  1) V8 Physical + V7 Deep/Pose features from 2DImage2BMI (train splits)
  2) V9 Hybrid features extracted from Celeb-FBI dataset.
  
Validation is reserved ONLY for 2DImage2BMI test/val splits to ensure our
model retains accuracy on the original scenario while fixing the mean-regression.
"""

import os
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed. Falling back to Ridge only.")

from project_paths import features_dir, models_dir, project_root

def load_v8_data(split, feat_dir_v8):
    path = os.path.join(feat_dir_v8, f"{split}_v8_phys.npz")
    data = np.load(path)
    return data["features"], data["weight"], data["names"]

def load_v7_data(split, feat_dir_v7):
    path = os.path.join(feat_dir_v7, f"{split}_v7.npz")
    data = np.load(path, allow_pickle=True)
    return data["deep"], data["pose"], data["names"]

def merge_features(v8_feats, v8_weight, v8_names, v7_deep, v7_pose, v7_names):
    v7_map = {str(n): i for i, n in enumerate(v7_names)}
    X_list, y_list = [], []
    matched = 0
    for i, n8 in enumerate(v8_names):
        n8_str = str(n8)
        if n8_str in v7_map:
            j = v7_map[n8_str]
            combined = np.concatenate([v8_feats[i], v7_deep[j], v7_pose[j]])
            X_list.append(combined)
            y_list.append(v8_weight[i])
            matched += 1
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

def compute_sample_weights(y, n_bins=15):
    """Inverse-frequency weights capped at 10x multiplier"""
    bin_edges = np.linspace(y.min() - 1, y.max() + 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y, bin_edges) - 1, 0, n_bins - 1)
    bin_counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
    bin_counts[bin_counts == 0] = 1.0
    
    raw_weights = 1.0 / bin_counts[bin_idx]
    weights = raw_weights / raw_weights.mean()
    weights = np.clip(weights, 0.1, 10.0)
    weights = weights / weights.mean()
    return weights

def load_v9_celeb(feat_dir_v9):
    path = os.path.join(feat_dir_v9, "celeb_v9.npz")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        return np.empty((0, 1565), dtype=np.float32), np.empty((0,), dtype=np.float32)
    data = np.load(path)
    return data["features"], data["weight"]

def flatten_distribution(X, y, target_per_bin=1000):
    # Flatten the histogram: ensure every 10kg bucket has roughly 'target_per_bin' samples
    # This prevents the model from favoring ANY single weight class (eliminates all bias).
    bins = np.arange(20, 160, 10)  # 20, 30, 40... 150
    indices = np.digitize(y, bins)
    
    X_bal, y_bal = [], []
    for i in range(1, len(bins)):
        mask = (indices == i)
        X_b = X[mask]
        y_b = y[mask]
        n_cur = len(X_b)
        if n_cur == 0:
            continue
        
        if n_cur < target_per_bin:
            reps = target_per_bin // n_cur
            rem = target_per_bin % n_cur
            X_bal.append(np.tile(X_b, (reps, 1)))
            y_bal.append(np.tile(y_b, reps))
            if rem > 0:
                idx = np.random.choice(n_cur, rem, replace=False)
                X_bal.append(X_b[idx])
                y_bal.append(y_b[idx])
        else:
            X_bal.append(X_b)
            y_bal.append(y_b)
            
    # Keep >150kg as is
    mask_high = (y >= 150)
    if mask_high.sum() > 0:
        X_bal.append(X[mask_high])
        y_bal.append(y[mask_high])
        
    return np.vstack(X_bal), np.concatenate(y_bal)

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

if __name__ == "__main__":
    feat_dir_v8 = str(features_dir("v8"))
    feat_dir_v7 = str(features_dir("v7"))
    feat_dir_v9 = str(project_root() / "features_v9")
    out_dir = str(project_root() / "models_v9")
    os.makedirs(out_dir, exist_ok=True)

    print("=== V9 Hybrid Model Training (Physics + Deep + Pose + Celeb-FBI) ===\n")

    # ---- 1. Load 2DImage2BMI Train data ----
    print("Loading 2DImage2BMI Base Data...")
    v8_tr_X, v8_tr_y, v8_tr_n = load_v8_data("train", feat_dir_v8)
    v7_tr_d, v7_tr_p, v7_tr_n = load_v7_data("train", feat_dir_v7)
    X_train_base, y_train_base = merge_features(v8_tr_X, v8_tr_y, v8_tr_n, v7_tr_d, v7_tr_p, v7_tr_n)
    
    # Validation / Test data is STRICTLY from 2DImage2BMI
    v8_te_X, v8_te_y, v8_te_n = load_v8_data("test", feat_dir_v8)
    v7_te_d, v7_te_p, v7_te_n = load_v7_data("test", feat_dir_v7)
    X_test_base, y_test_base = merge_features(v8_te_X, v8_te_y, v8_te_n, v7_te_d, v7_te_p, v7_te_n)

    # ---- 2. Load Celeb-FBI V9 Data ----
    print("Loading Celeb-FBI Dataset...")
    X_celeb, y_celeb = load_v9_celeb(feat_dir_v9)
    if len(X_celeb) == 0:
        print("[WARNING] Celeb-FBI Data not found or empty.")

    # Drop implausible
    valid_celeb = (y_celeb > 20) & (y_celeb < 300)
    X_celeb, y_celeb = X_celeb[valid_celeb], y_celeb[valid_celeb]
    
    valid_base = (y_train_base > 20) & (y_train_base < 300)
    X_train_base, y_train_base = X_train_base[valid_base], y_train_base[valid_base]
    
    valid_test = (y_test_base > 20) & (y_test_base < 300)
    X_test_base, y_test_base = X_test_base[valid_test], y_test_base[valid_test]

    # Combine into unified dataset (1565-D)
    # [V9 Architecture Breakthrough]: We reintegrate Celeb-FBI, but we MUST
    # drop PCA. PCA smears domain-shifted deep features across all components.
    # XGBoost on raw 1565-D can natively isolate domains using tree logic!
    X_train = np.vstack([X_train_base, X_celeb]) if len(X_celeb) > 0 else X_train_base
    y_train = np.concatenate([y_train_base, y_celeb]) if len(y_celeb) > 0 else y_train_base

    print(f"\n[DATA] Train Base: {X_train_base.shape[0]} | Train Celeb: {X_celeb.shape[0]} | Total Train: {X_train.shape[0]}")
    print(f"[DATA] Test Base: {X_test_base.shape[0]} (Strict Validation)")

    # ---- 3. Distribution Balancing ----
    print("\nFlattening Dataset Histogram to minimize error across ALL ranges...")
    X_train_os, y_train_os = flatten_distribution(X_train, y_train, target_per_bin=2000)
    # We do NOT apply inverse-frequency sample weights anymore, because
    # mathematically it perfectly cancels out our manual duplication above!
    print(f"[DATA] Final Oversampled Train: {X_train_os.shape[0]}")

    # ---- 4. StandardScaler (Raw 1565-D Domain, NO PCA) ----
    print("\nFitting Scaler on Massive 1565-D Matrix (Bypassing PCA Domain Smear)...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_os)
    X_test_s = scaler.transform(X_test_base)

    # ---- 5. Train Ridge ----
    print("\nTraining Ridge...")
    ridge = Ridge(alpha=100.0)
    ridge.fit(X_train_s, y_train_os)
    
    r_pred = ridge.predict(X_test_s)
    report("Ridge", r_pred, y_test_base)

    # ---- 6. Train XGBoost ----
    xgb = None
    if HAS_XGB:
        print("\nTraining XGBoost...")
        xgb = XGBRegressor(
            n_estimators=1000, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
            reg_lambda=1.0, random_state=42, verbosity=0,
            n_jobs=-1
        )
        xgb.fit(X_train_s, y_train_os)
        x_pred = xgb.predict(X_test_s)
        report("XGBoost", x_pred, y_test_base)
        
        # 50/50 Ensemble restores balance across all brackets using Smooth Ridge + Sharp XGBoost
        ens_pred = (r_pred * 0.5) + (x_pred * 0.5)
        report("Ensemble V9", ens_pred, y_test_base)
    else:
        ens_pred = r_pred
        
    print("\n=== FINAL TEST EVALUATION (V9) ===")
    report_buckets(ens_pred, y_test_base)

    bundle = {
        "scaler": scaler,
        "pca": None,
        "ridge": ridge,
        "xgb": xgb,
        "feature_type": "hybrid_v9",
        "n_pca": None,
    }

    save_path = os.path.join(out_dir, "ensemble_v9_hybrid.joblib")
    joblib.dump(bundle, save_path)
    print(f"\nSaved -> {save_path}")
    print("\nDone!")
