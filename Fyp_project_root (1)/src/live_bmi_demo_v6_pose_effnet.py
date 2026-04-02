import cv2
import numpy as np
import joblib
import mediapipe as mp

from person_detector_yolo_v4 import YOLOPersonDetector
from v6_utils_pose_effnet import (
    pose_coords_full_image,
    make_pose_box,
    extract_pose_features,
    extract_deep_features,
)
from project_paths import models_dir

mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles
mp_pose  = mp.solutions.pose

_yolo_fallback = YOLOPersonDetector(conf=0.35)

# ---------------- PATHS ----------------
_m6 = models_dir("v6")
SCALER_PATH = str(_m6 / "scaler_v6.joblib")
MODEL_PATH = str(_m6 / "krr_hybrid_v6.joblib")

SCALER = joblib.load(SCALER_PATH)
MODEL  = joblib.load(MODEL_PATH)

# ------------ USER CONFIG --------------
# Change these for yourself during demo
USER_GENDER = "M"   # "M" or "F"
USER_AGE    = 25    # approximate age in years
# ---------------------------------------





def main():
    print("Loaded V6 scaler and model.")
    print(f"Assuming Gender={USER_GENDER}, Age={USER_AGE} for fat% estimation.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(" Could not open webcam.")
        return

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as pose_tracker:

        bmi_hist = []

        print("Live BMI+Fat% V6 started. Press 'q' to quit.")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed.")
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose_tracker.process(rgb)

            coords = None
            used_yolo_pose = False
            if res.pose_landmarks:
                if hasattr(mp_style, "get_default_pose_connections_style"):
                    mp_draw.draw_landmarks(
                        frame,
                        res.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_style.get_default_pose_landmarks_style(),
                        connection_drawing_spec=mp_style.get_default_pose_connections_style(),
                    )
                else:
                    mp_draw.draw_landmarks(
                        frame,
                        res.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )
                coords = np.array(
                    [[lm.x * w, lm.y * h, lm.z] for lm in res.pose_landmarks.landmark],
                    dtype=np.float32,
                )
            else:
                coords = pose_coords_full_image(frame, _yolo_fallback)
                used_yolo_pose = coords is not None
                if coords is None:
                    cv2.putText(
                        frame,
                        "No person/pose. Step back / center.",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow("BMI V6 - Pose+EffNet", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue
                cv2.putText(
                    frame,
                    "Pose: YOLO crop fallback",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )

            
            box = make_pose_box(coords, w, h, margin=0.25)
            if box is None:
                cv2.imshow("BMI V6 - Pose+EffNet", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                cv2.imshow("BMI V6 - Pose+EffNet", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            
            deep_feat = extract_deep_features(crop)
            pose_feat = extract_pose_features(coords)
            feat = np.concatenate([deep_feat, pose_feat], axis=0).reshape(1, -1)

            feat_scaled = SCALER.transform(feat)
            bmi_pred = float(MODEL.predict(feat_scaled)[0])

            
            bmi_hist.append(bmi_pred)
            if len(bmi_hist) > 30:
                bmi_hist.pop(0)
            bmi_smooth = float(np.mean(bmi_hist))

            

            
            panel = np.zeros((380, 260, 3), dtype=np.uint8)

            cv2.putText(panel, "V6 Hybrid Pose+EffNet", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            if used_yolo_pose:
                cv2.putText(panel, "(YOLO pose path)", (10, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

            cv2.putText(panel, f"BMI (avg): {bmi_smooth:.2f}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(panel, f"BMI (inst): {bmi_pred:.2f}", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            

            cv2.putText(panel, f"Gender: {USER_GENDER}", (20, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(panel, f"Age: {USER_AGE}", (20, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.putText(panel, "Stand straight, full body", (20, 310),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
            cv2.putText(panel, "visible for best results.", (20, 330),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

           
            disp_crop = cv2.resize(crop, (260, 380))
            frame_resized = cv2.resize(frame, (520, 380))
            combo = np.hstack([panel, frame_resized, disp_crop])

            cv2.imshow("BMI V6 - Pose+EffNet (Hybrid)", combo)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
