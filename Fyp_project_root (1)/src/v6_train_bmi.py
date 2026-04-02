import numpy as np
import joblib
from xgboost import XGBRegressor


X = np.load("models_v6/features_v6.npy")
names = np.load("models_v6/names_v6.npy")

# Load labels (parse BMI from filename)
BMI = []
for f in names:
    parts = f.split("_")
    h = int(parts[3]) / 100000.0
    w = int(parts[4].split(".")[0]) / 100000.0
    BMI.append(w / (h*h+1e-8))

BMI = np.array(BMI)

# Train model
model = XGBRegressor(
    n_estimators=600,
    max_depth=8,
    learning_rate=0.015,
    subsample=0.9,
    colsample_bytree=0.8
)

model.fit(X, BMI)

joblib.dump(model, "models_v6/xgb_bmi_v6.json")
print("Saved BMI model.")
