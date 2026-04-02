import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


def load_split(split: str, feat_dir: str):
    data = np.load(os.path.join(feat_dir, f"{split}_deep_v4.npz"))
    return data["deep"], data["bmi"]


def evaluate(name, model, X, y):
    pred = model.predict(X)
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    mape = np.mean(np.abs((y - pred) / (y + 1e-6))) * 100.0
    print(f"{name} -> MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.2f}%")
    return mae, rmse, mape


if __name__ == "__main__":
    from project_paths import features_dir, models_dir

    feat_dir = str(features_dir("v4"))
    model_dir = str(models_dir("v4"))
    os.makedirs(model_dir, exist_ok=True)

    X_train, y_train = load_split("train", feat_dir)
    X_val,   y_val   = load_split("val",   feat_dir)
    X_test,  y_test  = load_split("test",  feat_dir)

    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    reg = KernelRidge(alpha=0.5, kernel="rbf", gamma=1e-3)

    reg.fit(X_train_s, y_train)

    print("\nYOLO + EfficientNet v4 performance:")
    evaluate("Train", reg, X_train_s, y_train)
    evaluate("Val",   reg, X_val_s,   y_val)
    evaluate("Test",  reg, X_test_s,  y_test)

    joblib.dump(scaler, os.path.join(model_dir, "scaler_v4.joblib"))
    joblib.dump(reg,    os.path.join(model_dir, "deep_krr_v4.joblib"))
    print("\nSaved v4 scaler and regressor to models_v4/")
