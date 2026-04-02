import os
import numpy as np
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


def load_features(split: str, feat_dir: str):
    data = np.load(os.path.join(feat_dir, f"{split}_features.npz"))
    deep = data["deep"]
    antro = data["antro"]
    bmi = data["bmi"]
    # concatenate
    X = np.concatenate([deep, antro], axis=1)
    y = bmi
    return X, y


if __name__ == "__main__":
    from project_paths import features_dir, models_dir

    feat_dir = str(features_dir("v1"))
    model_out = str(models_dir("v1"))
    os.makedirs(model_out, exist_ok=True)

    X_train, y_train = load_features("train", feat_dir)
    X_val,   y_val   = load_features("val", feat_dir)
    X_test,  y_test  = load_features("test", feat_dir)

    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    # Example: Kernel Ridge Regression (similar spirit to original paper)
    reg = KernelRidge(alpha=1.0, kernel="rbf", gamma=1e-3)

    reg.fit(X_train, y_train)

    def eval_split(name, X, y):
        pred = reg.predict(X)
        mae = mean_absolute_error(y, pred)
        rmse = np.sqrt(mean_squared_error(y, pred))
        mape = np.mean(np.abs((y - pred) / (y + 1e-6))) * 100.0
        print(f"{name} -> MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.2f}%")

    eval_split("Train", X_train, y_train)
    eval_split("Val",   X_val,   y_val)
    eval_split("Test",  X_test,  y_test)

    joblib.dump(reg, os.path.join(model_out, "krr_mixed_features.joblib"))
    print("Saved regressor.")
