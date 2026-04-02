import cv2
import numpy as np
import mediapipe as mp


mp_pose = mp.solutions.pose.Pose(static_image_mode=True)


# -------------------------
# 1. CONTOUR FEATURES
# -------------------------
def _largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def contour_features(bgr: np.ndarray) -> np.ndarray:
    h, w, _ = bgr.shape

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41, 5
    )
    mask = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    cnt = _largest_contour(mask)
    if cnt is None:
        return np.zeros(7, dtype=np.float32)

    x, y, w_box, h_box = cv2.boundingRect(cnt)
    aspect = h_box / (w_box + 1e-6)
    area = cv2.contourArea(cnt) / float(h * w + 1e-6)

    y_mid = y + h_box // 2
    upper_mask = np.zeros_like(mask)
    lower_mask = np.zeros_like(mask)
    cv2.drawContours(upper_mask, [cnt], -1, 255, cv2.FILLED)
    lower_mask = upper_mask.copy()
    upper_mask[y_mid:, :] = 0
    lower_mask[:y_mid, :] = 0

    upper_area = (upper_mask > 0).sum() / (h * w + 1e-6)
    lower_area = (lower_mask > 0).sum() / (h * w + 1e-6)

    ys = [int(y + h_box * r) for r in (0.25, 0.5, 0.75)]
    widths = []
    for yy in ys:
        row = mask[yy, :]
        xs = np.where(row > 0)[0]
        if len(xs) < 2:
            widths.append(0.0)
        else:
            widths.append((xs[-1] - xs[0]) / (w_box + 1e-6))

    return np.array([
        aspect, area,
        upper_area, lower_area,
        widths[0], widths[1], widths[2]
    ], dtype=np.float32)


# -------------------------
# 2. POSE FEATURES
# -------------------------
def pose_features(bgr: np.ndarray) -> np.ndarray:
    h, w, _ = bgr.shape
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = mp_pose.process(rgb)

    if not res.pose_landmarks:
        return np.zeros(6, dtype=np.float32)

    lm = res.pose_landmarks.landmark

    def xy(idx):
        return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)

    L_SH, R_SH = 11, 12
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    NOSE = 0

    shoulder_w = np.linalg.norm(xy(L_SH) - xy(R_SH))
    hip_w = np.linalg.norm(xy(L_HIP) - xy(R_HIP))
    torso_len = np.linalg.norm(xy(NOSE) - 0.5 * (xy(L_HIP) + xy(R_HIP)))
    leg_len = np.linalg.norm(0.5 * (xy(L_HIP) + xy(R_HIP)) -
                             0.5 * (xy(L_KNEE) + xy(R_KNEE)))

    height_pix = lm[NOSE].y * h - min(xy(L_HIP)[1], xy(R_HIP)[1])
    denom = abs(height_pix) + 1e-6

    return np.array([
        shoulder_w / denom,
        hip_w / denom,
        torso_len / denom,
        leg_len / denom,
        shoulder_w / (hip_w + 1e-6),
        torso_len / (leg_len + 1e-6),
    ], dtype=np.float32)


# -------------------------
# 3. MIXED FEATURE (FINAL)
# -------------------------
def extract_mixed_features(bgr: np.ndarray) -> np.ndarray:
    """
    Final anthropometric feature vector:
    contour features (7 dims) + pose features (6 dims) = 13 dims
    """
    cf = contour_features(bgr)
    pf = pose_features(bgr)
    return np.concatenate([cf, pf], axis=0)
