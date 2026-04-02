"""
train_v7_ensemble.py  —  QUICK-WIN SCRIPT
-----------------------------------------
Trains an improved ensemble regressor on the ALREADY-EXTRACTED V4 deep features.
No re-extraction needed — just run this script directly.

Pipeline improvements over V6 single-KRR:
  1. PCA(256, whiten=True) — reduces 1536-D deep features to 256-D, improves
     KRR kernel quality and speeds up training.
  2. Hyperparameter search on validation MAE for all three regressors.
  3. Three regressors: KRR + SVR + XGBoost (if installed).
  4. Simple average ensemble of all available regressors.
  5. 5-fold cross-validation on train+val for a reliable error estimate.

Outputs:
  models_v7/ensemble_v7_deeponly.joblib  — bundle dict with keys:
    scaler, pca, krr, svr, [xgb], has_xgb

Usage:
    cd src
    python train_v7_ensemble.py
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
    path = os.path.join(feat_dir, f"{split}_deep_v4.npz")
    data = np.load(path)
    return data["deep"], data["bmi"]


def report(name: str, preds: np.ndarray, y: np.ndarray):
    mae  = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mape = np.mean(np.abs((y - preds) / (y + 1e-6))) * 100.0
    print(f"  {name:<22s}  MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.2f}%")
    return mae, rmse, mape


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from project_paths import features_dir, models_dir

    feat_dir = str(features_dir("v4"))
    out_dir  = models_dir("v7")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir  = str(out_dir)

    # ---- Load data ----
    print("=== V7 Ensemble (Deep-Only Quick Win) ===\n")
    print("Loading V4 deep features ...")
    X_train, y_train = load_split("train", feat_dir)
    X_val,   y_val   = load_split("val",   feat_dir)
    X_test,  y_test  = load_split("test",  feat_dir)
    print(f"  Train {X_train.shape}  Val {X_val.shape}  Test {X_test.shape}")

    # ---- Preprocessing: scale → PCA(256) ----
    # Fit on train only for hyperparameter search
    scaler_sel = StandardScaler()
    Xtr_s = scaler_sel.fit_transform(X_train)
    Xva_s = scaler_sel.transform(X_val)

    pca_sel = PCA(n_components=256, whiten=True, random_state=42)
    Xtr_p = pca_sel.fit_transform(Xtr_s)
    Xva_p = pca_sel.transform(Xva_s)

    # ---- KRR grid search ----
    print("\n[1/3] Tuning KRR on validation MAE ...")
    best_krr_mae, best_krr_p = float("inf"), (0.05, 3e-5)
    for alpha in (0.01, 0.05, 0.1, 0.3, 1.0):
        for gamma in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3):
            m = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)
            m.fit(Xtr_p, y_train)
            mae = mean_absolute_error(y_val, m.predict(Xva_p))
            if mae < best_krr_mae:
                best_krr_mae, best_krr_p = mae, (alpha, gamma)
    print(f"  Best KRR  val MAE={best_krr_mae:.4f}  "
          f"alpha={best_krr_p[0]}  gamma={best_krr_p[1]}")

    # ---- SVR grid search ----
    print("\n[2/3] Tuning SVR on validation MAE ...")
    best_svr_mae, best_svr_p = float("inf"), (10.0, "scale")
    for C in (1.0, 5.0, 10.0, 30.0):
        for gamma in ("scale", 1e-4, 3e-4, 1e-3):
            m = SVR(C=C, kernel="rbf", gamma=gamma, epsilon=0.1)
            m.fit(Xtr_p, y_train)
            mae = mean_absolute_error(y_val, m.predict(Xva_p))
            if mae < best_svr_mae:
                best_svr_mae, best_svr_p = mae, (C, gamma)
    print(f"  Best SVR  val MAE={best_svr_mae:.4f}  "
          f"C={best_svr_p[0]}  gamma={best_svr_p[1]}")

    # ---- XGBoost grid search ----
    best_xgb_p = None
    if HAS_XGB:
        print("\n[3/3] Tuning XGBoost on validation MAE ...")
        best_xgb_mae = float("inf")
        for lr in (0.02, 0.05, 0.1):
            for depth in (3, 4, 6):
                for n in (300, 500):
                    m = XGBRegressor(
                        n_estimators=n, learning_rate=lr,
                        max_depth=depth, subsample=0.8,
                        colsample_bytree=0.8, random_state=42,
                        verbosity=0,
                    )
                    m.fit(Xtr_p, y_train)
                    mae = mean_absolute_error(y_val, m.predict(Xva_p))
                    if mae < best_xgb_mae:
                        best_xgb_mae = mae
                        best_xgb_p = dict(n_estimators=n, learning_rate=lr, max_depth=depth)
        print(f"  Best XGB  val MAE={best_xgb_mae:.4f}  {best_xgb_p}")
    else:
        print("\n[3/3] XGBoost not available — skipping.")

    # ---- Re-fit preprocessing on train+val ----
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])

    scaler_final = StandardScaler()
    X_tv_s = scaler_final.fit_transform(X_tv)
    X_te_s = scaler_final.transform(X_test)

    pca_final = PCA(n_components=256, whiten=True, random_state=42)
    X_tv_p = pca_final.fit_transform(X_tv_s)
    X_te_p = pca_final.transform(X_te_s)

    # ---- Final models on train+val ----
    print("\n=== Fitting Final Models on Train+Val ===")
    krr_final = KernelRidge(kernel="rbf", alpha=best_krr_p[0], gamma=best_krr_p[1])
    krr_final.fit(X_tv_p, y_tv)

    svr_final = SVR(C=best_svr_p[0], kernel="rbf", gamma=best_svr_p[1], epsilon=0.1)
    svr_final.fit(X_tv_p, y_tv)

    xgb_final = None
    if HAS_XGB and best_xgb_p:
        xgb_final = XGBRegressor(
            **best_xgb_p, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )
        xgb_final.fit(X_tv_p, y_tv)

    # ---- 5-Fold CV (train+val) for honest error estimate ----
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae = -cross_val_score(
        KernelRidge(kernel="rbf", alpha=best_krr_p[0], gamma=best_krr_p[1]),
        X_tv_p, y_tv, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    print(f"\n  5-Fold CV MAE (KRR, train+val): {cv_mae.mean():.3f} ± {cv_mae.std():.3f}")

    # ---- Test-set results ----
    print("\n=== Test Set Results ===")
    krr_pred = krr_final.predict(X_te_p)
    svr_pred = svr_final.predict(X_te_p)
    report("KRR (PCA256)", krr_pred, y_test)
    report("SVR (PCA256)", svr_pred, y_test)

    preds_list = [krr_pred, svr_pred]
    if xgb_final is not None:
        xgb_pred = xgb_final.predict(X_te_p)
        report("XGBoost", xgb_pred, y_test)
        preds_list.append(xgb_pred)

    ens_pred = np.mean(preds_list, axis=0)
    report("Ensemble (avg)", ens_pred, y_test)

    # ---- Save bundle ----
    bundle = {
        "scaler":   scaler_final,
        "pca":      pca_final,
        "krr":      krr_final,
        "svr":      svr_final,
        "has_xgb":  xgb_final is not None,
        "feature_type": "deep_only_v4",   # signals to live demo: no pose features
    }
    if xgb_final is not None:
        bundle["xgb"] = xgb_final

    save_path = os.path.join(out_dir, "ensemble_v7_deeponly.joblib")
    joblib.dump(bundle, save_path)
    print(f"\n✅  Saved → {save_path}")
