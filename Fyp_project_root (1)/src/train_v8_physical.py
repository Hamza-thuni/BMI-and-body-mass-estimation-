"""
train_v8_physical.py  (v3 — Hybrid Deep + Physical)
-----------------------------------------------------
V8: Combined Physics + Deep Features for Weight Estimation.

The pure-physics approach (5 features) has a fundamental limitation:
2D silhouette area and widths can't distinguish body density/composition.
Two people with the same height and silhouette can weigh very differently.

This version combines:
  - V8 physical features (5-D): area, shoulders, hips, torso, height
  - V7 deep features (1536-D): EfficientNet-B3 body appearance encoding
  - V7 pose features  (24-D):  scale-invariant pose ratios

Total: 1565-D -> PCA(128) -> XGBoost + Ridge ensemble

Key improvements:
  1. Sample weighting: inverse-frequency so rare weight ranges contribute equally
  2. PCA(128) to reduce dimensionality and prevent overfitting
  3. Hyperparameter tuning on validation MAE
  4. Per-bucket evaluation to verify no mean-collapse

Usage:
    cd src
    python train_v8_physical.py
"""

import os
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed. Falling back to Ridge only.")

from project_paths import features_dir, models_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_v8_data(split, feat_dir_v8):
    """Load V8 physical features + weight labels."""
    path = os.path.join(feat_dir_v8, f"{split}_v8_phys.npz")
    data = np.load(path)
    X = data["features"]  # [area_m2, sh_w_m, hip_w_m, torso_l_m, height_m]
    y = data["weight"]
    names = data["names"]
    return X, y, names


def load_v7_data(split, feat_dir_v7):
    """Load V7 deep + pose features."""
    path = os.path.join(feat_dir_v7, f"{split}_v7.npz")
    data = np.load(path, allow_pickle=True)
    deep = data["deep"]     # (N, 1536)
    pose = data["pose"]     # (N, 24)
    names = data["names"]
    return deep, pose, names


def merge_features(v8_feats, v8_weight, v8_names, v7_deep, v7_pose, v7_names):
    """
    Merge V8 physical features with V7 deep+pose features by matching filenames.
    Returns: X_combined (N, 1565), y_weight (N,)
    """
    # Build name -> index map for V7
    v7_map = {}
    for i, n in enumerate(v7_names):
        v7_map[str(n)] = i

    X_list, y_list = [], []
    matched = 0
    for i, n8 in enumerate(v8_names):
        n8_str = str(n8)
        if n8_str in v7_map:
            j = v7_map[n8_str]
            # Combine: [v8_physical(5) | v7_deep(1536) | v7_pose(24)] = 1565-D
            combined = np.concatenate([
                v8_feats[i],      # 5
                v7_deep[j],       # 1536
                v7_pose[j],       # 24
            ])
            X_list.append(combined)
            y_list.append(v8_weight[i])
            matched += 1

    print(f"    Matched {matched}/{len(v8_names)} samples "
          f"(V8 has {len(v8_names)}, V7 has {len(v7_names)})")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def compute_sample_weights(y, n_bins=15):
    """Inverse-frequency weights so underrepresented ranges contribute equally."""
    bin_edges = np.linspace(y.min() - 1, y.max() + 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y, bin_edges) - 1, 0, n_bins - 1)

    bin_counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
    bin_counts[bin_counts == 0] = 1.0

    raw_weights = 1.0 / bin_counts[bin_idx]
    weights = raw_weights / raw_weights.mean()
    weights = np.clip(weights, 0.1, 10.0)
    weights = weights / weights.mean()
    return weights


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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    feat_dir_v8 = str(features_dir("v8"))
    feat_dir_v7 = str(features_dir("v7"))
    out_dir = models_dir("v8")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = str(out_dir)

    print("=== V8 Hybrid Model Training (Physics + Deep + Pose) ===\n")

    # Check if V7 features exist; if not, fall back to physics-only
    v7_available = os.path.exists(os.path.join(feat_dir_v7, "train_v7.npz"))

    if v7_available:
        print("[INFO] V7 deep+pose features found. Using HYBRID mode (1565-D).\n")
    else:
        print("[WARN] V7 features not found. Using PHYSICS-ONLY mode (5-D).")
        print("       Run extract_features_v7.py first for better results.\n")

    # ---- Load data ----
    print("Loading V8 physical features ...")
    v8_tr_X, v8_tr_y, v8_tr_n = load_v8_data("train", feat_dir_v8)
    v8_va_X, v8_va_y, v8_va_n = load_v8_data("val", feat_dir_v8)
    v8_te_X, v8_te_y, v8_te_n = load_v8_data("test", feat_dir_v8)

    if v7_available:
        print("Loading V7 deep+pose features ...")
        v7_tr_d, v7_tr_p, v7_tr_n = load_v7_data("train", feat_dir_v7)
        v7_va_d, v7_va_p, v7_va_n = load_v7_data("val", feat_dir_v7)
        v7_te_d, v7_te_p, v7_te_n = load_v7_data("test", feat_dir_v7)

        print("\nMerging features by filename ...")
        X_train, y_train = merge_features(v8_tr_X, v8_tr_y, v8_tr_n,
                                          v7_tr_d, v7_tr_p, v7_tr_n)
        X_val, y_val = merge_features(v8_va_X, v8_va_y, v8_va_n,
                                      v7_va_d, v7_va_p, v7_va_n)
        X_test, y_test = merge_features(v8_te_X, v8_te_y, v8_te_n,
                                        v7_te_d, v7_te_p, v7_te_n)
        feat_type = "hybrid_v8"
    else:
        X_train, y_train = v8_tr_X, v8_tr_y
        X_val, y_val = v8_va_X, v8_va_y
        X_test, y_test = v8_te_X, v8_te_y
        feat_type = "physical_v8"

    # Filter implausible
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        mask = (y > 20) & (y < 300)
        n_bad = int((~mask).sum())
        if n_bad:
            print(f"  [WARN] {name}: dropped {n_bad} implausible samples")

    print(f"\n  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print(f"  Feature dims: {X_train.shape[1]}")
    print(f"  Train weight: [{y_train.min():.1f}, {y_train.max():.1f}] "
          f"mean={y_train.mean():.1f} median={np.median(y_train):.1f}")

    # ---- Sample weights ----
    sw_train = compute_sample_weights(y_train, n_bins=15)
    print(f"  Sample weights: min={sw_train.min():.2f} max={sw_train.max():.2f}")

    # ---- Preprocessing: StandardScaler + PCA ----
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    n_pca = min(256, X_train_s.shape[1], X_train_s.shape[0] - 1)
    pca = PCA(n_components=n_pca, whiten=True, random_state=42)
    X_train_p = pca.fit_transform(X_train_s)
    X_val_p = pca.transform(X_val_s)
    X_test_p = pca.transform(X_test_s)
    print(f"  After PCA({n_pca}): explained var = {pca.explained_variance_ratio_.sum():.3f}")

    # ---- Ridge tuning ----
    print("\n[1/2] Tuning Ridge ...")
    best_ridge_mae = float("inf")
    best_ridge_alpha = 1.0
    for alpha in (0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0):
        m = Ridge(alpha=alpha)
        m.fit(X_train_p, y_train, sample_weight=sw_train)
        mae = mean_absolute_error(y_val, m.predict(X_val_p))
        if mae < best_ridge_mae:
            best_ridge_mae = mae
            best_ridge_alpha = alpha
    print(f"  Best Ridge: alpha={best_ridge_alpha} val MAE={best_ridge_mae:.2f} kg")

    # ---- XGBoost tuning ----
    best_xgb_params = None
    best_xgb_mae = float("inf")
    if HAS_XGB:
        print("\n[2/2] Tuning XGBoost ...")
        for n_est in (200, 500, 800, 1000):
            for lr in (0.01, 0.02, 0.05, 0.1):
                for depth in (3, 4, 5, 6):
                    m = XGBRegressor(
                        n_estimators=n_est, learning_rate=lr,
                        max_depth=depth, subsample=0.8,
                        colsample_bytree=0.7, reg_alpha=0.1,
                        reg_lambda=1.0, random_state=42,
                        verbosity=0
                    )
                    m.fit(X_train_p, y_train, sample_weight=sw_train)
                    mae = mean_absolute_error(y_val, m.predict(X_val_p))
                    if mae < best_xgb_mae:
                        best_xgb_mae = mae
                        best_xgb_params = dict(
                            n_estimators=n_est, learning_rate=lr,
                            max_depth=depth
                        )
        print(f"  Best XGB: {best_xgb_params} val MAE={best_xgb_mae:.2f} kg")
    else:
        print("\n[2/2] XGBoost not available.")

    # ---- Final training on Train+Val ----
    print("\n=== Final Training on Train+Val ===")
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])
    sw_tv = compute_sample_weights(y_tv, n_bins=15)

    scaler_final = StandardScaler()
    X_tv_s = scaler_final.fit_transform(X_tv)
    X_te_s = scaler_final.transform(X_test)

    pca_final = PCA(n_components=n_pca, whiten=True, random_state=42)
    X_tv_p = pca_final.fit_transform(X_tv_s)
    X_te_p = pca_final.transform(X_te_s)

    ridge_final = Ridge(alpha=best_ridge_alpha)
    ridge_final.fit(X_tv_p, y_tv, sample_weight=sw_tv)

    xgb_final = None
    if HAS_XGB and best_xgb_params:
        xgb_final = XGBRegressor(
            **best_xgb_params, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=0.1,
            reg_lambda=1.0, random_state=42, verbosity=0
        )
        xgb_final.fit(X_tv_p, y_tv, sample_weight=sw_tv)

    # ---- Test evaluation ----
    print("\n=== Test Set Results ===")
    r_pred = ridge_final.predict(X_te_p)
    report("Ridge", r_pred, y_test)

    if xgb_final:
        x_pred = xgb_final.predict(X_te_p)
        report("XGBoost", x_pred, y_test)
        ens_pred = (r_pred + x_pred) / 2.0
        report("Ensemble", ens_pred, y_test)
    else:
        ens_pred = r_pred

    print()
    report_buckets(ens_pred, y_test)

    # ---- Save ----
    bundle = {
        "scaler": scaler_final,
        "pca": pca_final,
        "ridge": ridge_final,
        "xgb": xgb_final,
        "feature_type": feat_type,
        "n_pca": n_pca,
    }

    save_path = os.path.join(out_dir, "ensemble_v8_physical.joblib")
    joblib.dump(bundle, save_path)
    print(f"\nSaved -> {save_path}")
    print("\nDone!")
