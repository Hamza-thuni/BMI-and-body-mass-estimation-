import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


def load_features(split: str, feat_dir: str):
    data = np.load(os.path.join(feat_dir, f"{split}_features_v2.npz"))
    deep = data["deep"]
    antro = data["antro"]
    bmi = data["bmi"]
    X = np.concatenate([deep, antro], axis=1)
    y = bmi
    return X, y


def evaluate(name, model, X, y):
    pred = model.predict(X)
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    mape = np.mean(np.abs((y - pred) / (y + 1e-6))) * 100.0
    print(f"{name} -> MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.2f}%")
    return mae, rmse, mape


if __name__ == "__main__":
    from project_paths import features_dir, models_dir

    feat_dir = str(features_dir("v2"))
    model_dir = str(models_dir("v2"))
    os.makedirs(model_dir, exist_ok=True)

    X_train, y_train = load_features("train", feat_dir)
    X_val,   y_val   = load_features("val",   feat_dir)
    X_test,  y_test  = load_features("test",  feat_dir)

    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    # Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # Base models
    krr = KernelRidge(alpha=1.0, kernel="rbf", gamma=1e-3)
    svr = SVR(C=5.0, kernel="rbf", gamma="scale")
    rf  = RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_split=4, random_state=42
    )

    print("\nFitting base models...")
    krr.fit(X_train_s, y_train)
    svr.fit(X_train_s, y_train)
    rf.fit(X_train_s, y_train)

    print("\nBase model performance (Val):")
    evaluate("KRR", krr, X_val_s, y_val)
    evaluate("SVR", svr, X_val_s, y_val)
    evaluate("RF ", rf,  X_val_s, y_val)

    # Ensemble (stacking)
    estimators = [("krr", krr), ("svr", svr), ("rf", rf)]
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=KernelRidge(alpha=1.0, kernel="rbf", gamma=1e-3),
        passthrough=True,
        n_jobs=-1,
    )
    print("\nFitting Stacking Regressor...")
    stack.fit(X_train_s, y_train)

    print("\nFinal Ensemble performance:")
    evaluate("Train", stack, X_train_s, y_train)
    evaluate("Val",   stack, X_val_s,   y_val)
    evaluate("Test",  stack, X_test_s,  y_test)

    # Save everything
    joblib.dump(scaler, os.path.join(model_dir, "scaler_v2.joblib"))
    joblib.dump(stack,  os.path.join(model_dir, "stack_ensemble.joblib"))
    print("\nSaved scaler and ensemble regressor to models_v2/")
