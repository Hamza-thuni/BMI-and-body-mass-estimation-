import numpy as np
import joblib
from xgboost import XGBRegressor

X = np.load("models_v6/features_v6.npy")
names = np.load("models_v6/names_v6.npy")

def navy_body_fat(h, w, gender):
    # A proxy estimate
    if gender == "M":
        return 1.20*(w/(h*h)) + 0.23*25 - 16.2
    else:
        return 1.20*(w/(h*h)) + 0.23*25 - 5.4

fat = []
for f in names:
    p = f.split("_")
    gender = p[1]
    h = int(p[3]) / 100000.0
    w = int(p[4].split(".")[0]) / 100000.0
    fat.append(navy_body_fat(h,w,gender))

fat = np.array(fat)

# Train
model = XGBRegressor(
    n_estimators=600,
    max_depth=8,
    learning_rate=0.015
)

model.fit(X, fat)
joblib.dump(model, "models_v6/xgb_fat_v6.json")
print("Saved FAT% model.")
