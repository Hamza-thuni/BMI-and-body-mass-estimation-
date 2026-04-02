import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


def load_deep(split, feat_dir):
    data = np.load(os.path.join(feat_dir, f"{split}_features_v2.npz"))
    return data["deep"], data["bmi"]


def eval_model(name, model, X, y):
    pred = model.predict(X)
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    print(f"{name} -> MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    return mae, rmse


if __name__ == "__main__":
    from project_paths import features_dir, models_dir

    feat_dir = str(features_dir("v2"))
    model_dir = str(models_dir("v2"))
    os.makedirs(model_dir, exist_ok=True)

    X_train, y_train = load_deep("train", feat_dir)
    X_val,   y_val   = load_deep("val",   feat_dir)
    X_test,  y_test  = load_deep("test",  feat_dir)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    reg = KernelRidge(alpha=1.0, kernel="rbf", gamma=1e-3)
    reg.fit(X_train_s, y_train)

    print("Deep-only performance:")
    eval_model("Train", reg, X_train_s, y_train)
    eval_model("Val",   reg, X_val_s,   y_val)
    eval_model("Test",  reg, X_test_s,  y_test)

    joblib.dump(scaler, os.path.join(model_dir, "scaler_deep_only.joblib"))
    joblib.dump(reg,    os.path.join(model_dir, "deep_only_krr.joblib"))
    print("Saved deep-only models.")
