"""
train_v7_hybrid.py
------------------
Trains the FULL V7 hybrid model on V7 features (segmented deep + rich 24-D pose).

Requires:  features_v7/{train,val,test}_v7.npz  (run extract_features_v7.py first)

Pipeline:
  1. Load deep (1536-D) + pose (24-D)  → concat → 1560-D
  2. StandardScaler + PCA(256, whiten)
  3. Hyperparameter grid search on validation MAE for KRR, SVR, XGBoost
  4. Final models retrained on train+val combined
  5. 5-fold cross-validation for honest error estimate
  6. Simple average ensemble at inference

Outputs:
  models_v7/ensemble_v7_hybrid.joblib  — bundle dict:
    scaler, pca, krr, svr, [xgb], has_xgb, feature_type="hybrid_v7"

Usage:
    cd src
    python train_v7_hybrid.py
"""

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed. Skipping XGBRegressor.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_split(split: str, feat_dir: str):
    path = os.path.join(feat_dir, f"{split}_v7.npz")
    data = np.load(path, allow_pickle=True)
    X    = np.concatenate([data["deep"], data["pose"]], axis=1)
    y    = data["bmi"].astype(np.float32)

    # ── Sanity guard: reject samples with implausible BMI ────────────────────
    # Human BMI can't be below 10 or above 60 in any real dataset.
    # Corrupted filenames sometimes produce 200-450 values; drop them here.
    mask = (y > 10.0) & (y < 60.0)
    n_bad = int((~mask).sum())
    if n_bad:
        print(f"  [WARN] {split}: dropped {n_bad} samples with implausible BMI "
              f"(min raw={y.min():.1f}, max raw={y.max():.1f})")
    X, y = X[mask], y[mask]
    print(f"  {split}: {len(y)} samples  BMI [{y.min():.1f}, {y.max():.1f}]  "
          f"mean={y.mean():.2f}")
    return X, y


def report(name: str, preds: np.ndarray, y: np.ndarray):
    mae  = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mape = np.mean(np.abs((y - preds) / (y + 1e-6))) * 100.0
    print(f"  {name:<25s}  MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.2f}%")
    return mae, rmse, mape


def tune_krr(Xtr, ytr, Xva, yva):
    best_mae, best_p = float("inf"), (0.05, 3e-5)
    for alpha in (0.01, 0.05, 0.1, 0.3, 1.0):
        for gamma in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3):
            m = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)
            m.fit(Xtr, ytr)
            mae = mean_absolute_error(yva, m.predict(Xva))
            if mae < best_mae:
                best_mae, best_p = mae, (alpha, gamma)
    return best_p, best_mae


def tune_svr(Xtr, ytr, Xva, yva):
    best_mae, best_p = float("inf"), (10.0, "scale")
    for C in (1.0, 5.0, 10.0, 30.0):
        for gamma in ("scale", 1e-4, 3e-4, 1e-3):
            m = SVR(C=C, kernel="rbf", gamma=gamma, epsilon=0.1)
            m.fit(Xtr, ytr)
            mae = mean_absolute_error(yva, m.predict(Xva))
            if mae < best_mae:
                best_mae, best_p = mae, (C, gamma)
    return best_p, best_mae


def tune_xgb(Xtr, ytr, Xva, yva):
    best_mae, best_p = float("inf"), None
    for lr in (0.02, 0.05, 0.1):
        for depth in (3, 4, 6):
            for n in (300, 500):
                m = XGBRegressor(
                    n_estimators=n, learning_rate=lr, max_depth=depth,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbosity=0,
                )
                m.fit(Xtr, ytr)
                mae = mean_absolute_error(yva, m.predict(Xva))
                if mae < best_mae:
                    best_mae  = mae
                    best_p    = dict(n_estimators=n, learning_rate=lr, max_depth=depth)
    return best_p, best_mae


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from project_paths import features_dir, models_dir

    feat_dir = str(features_dir("v7"))
    out_dir  = models_dir("v7")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir  = str(out_dir)

    # ---- Load ----
    print("=== V7 Hybrid Model Training (Seg + Deep + Rich Pose + Ensemble) ===\n")
    print("Loading V7 features ...")
    X_train, y_train = load_split("train", feat_dir)
    X_val,   y_val   = load_split("val",   feat_dir)
    X_test,  y_test  = load_split("test",  feat_dir)
    print(f"  Train {X_train.shape}  Val {X_val.shape}  Test {X_test.shape}")

    # ---- Preprocessing (on train only for search) ----
    scaler_sel = StandardScaler()
    Xtr_s = scaler_sel.fit_transform(X_train)
    Xva_s = scaler_sel.transform(X_val)

    n_components = min(256, Xtr_s.shape[1])
    pca_sel = PCA(n_components=n_components, whiten=True, random_state=42)
    Xtr_p = pca_sel.fit_transform(Xtr_s)
    Xva_p = pca_sel.transform(Xva_s)
    print(f"  After PCA: {Xtr_p.shape}")

    # ---- Hyperparameter search ----
    print("\n[1/3] Tuning KRR ...")
    krr_p, krr_val_mae = tune_krr(Xtr_p, y_train, Xva_p, y_val)
    print(f"  Best  MAE={krr_val_mae:.4f}  alpha={krr_p[0]}  gamma={krr_p[1]}")

    print("\n[2/3] Tuning SVR ...")
    svr_p, svr_val_mae = tune_svr(Xtr_p, y_train, Xva_p, y_val)
    print(f"  Best  MAE={svr_val_mae:.4f}  C={svr_p[0]}  gamma={svr_p[1]}")

    xgb_p = None
    if HAS_XGB:
        print("\n[3/3] Tuning XGBoost ...")
        xgb_p, xgb_val_mae = tune_xgb(Xtr_p, y_train, Xva_p, y_val)
        print(f"  Best  MAE={xgb_val_mae:.4f}  {xgb_p}")
    else:
        print("\n[3/3] XGBoost not available — skipping.")

    # ---- Final preprocessing on train+val ----
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])

    scaler_final = StandardScaler()
    X_tv_s = scaler_final.fit_transform(X_tv)
    X_te_s = scaler_final.transform(X_test)

    pca_final = PCA(n_components=n_components, whiten=True, random_state=42)
    X_tv_p = pca_final.fit_transform(X_tv_s)
    X_te_p = pca_final.transform(X_te_s)

    # ---- Final models ----
    print("\n=== Fitting Final Models on Train+Val ===")
    krr_final = KernelRidge(kernel="rbf", alpha=krr_p[0], gamma=krr_p[1])
    krr_final.fit(X_tv_p, y_tv)

    svr_final = SVR(C=svr_p[0], kernel="rbf", gamma=svr_p[1], epsilon=0.1)
    svr_final.fit(X_tv_p, y_tv)

    xgb_final = None
    if HAS_XGB and xgb_p:
        xgb_final = XGBRegressor(
            **xgb_p, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )
        xgb_final.fit(X_tv_p, y_tv)

    # ---- 5-fold CV ----
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae = -cross_val_score(
        KernelRidge(kernel="rbf", alpha=krr_p[0], gamma=krr_p[1]),
        X_tv_p, y_tv, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    print(f"\n  5-Fold CV MAE (KRR, train+val): {cv_mae.mean():.3f} ± {cv_mae.std():.3f}")

    # ---- Test set ----
    print("\n=== Test Set Results ===")
    krr_pred = krr_final.predict(X_te_p)
    svr_pred = svr_final.predict(X_te_p)
    report("KRR", krr_pred, y_test)
    report("SVR", svr_pred, y_test)

    preds_list = [krr_pred, svr_pred]
    if xgb_final is not None:
        xgb_pred = xgb_final.predict(X_te_p)
        report("XGBoost", xgb_pred, y_test)
        preds_list.append(xgb_pred)

    ens_pred = np.mean(preds_list, axis=0)
    report("Ensemble (avg)", ens_pred, y_test)

    # ---- Save ----
    bundle = {
        "scaler":       scaler_final,
        "pca":          pca_final,
        "krr":          krr_final,
        "svr":          svr_final,
        "has_xgb":      xgb_final is not None,
        "feature_type": "hybrid_v7",   # deep(1536) + pose(24) input
    }
    if xgb_final is not None:
        bundle["xgb"] = xgb_final

    save_path = os.path.join(out_dir, "ensemble_v7_hybrid.joblib")
    joblib.dump(bundle, save_path)
    print(f"\n✅  Saved → {save_path}")
