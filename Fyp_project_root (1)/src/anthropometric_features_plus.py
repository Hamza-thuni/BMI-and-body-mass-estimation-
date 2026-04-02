import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose.Pose(static_image_mode=True)


def _largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def contour_features_plus(bgr: np.ndarray) -> np.ndarray:
    h, w, _ = bgr.shape

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41, 5
    )
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnt = _largest_contour(mask)
    if cnt is None:
        return np.zeros(10, dtype=np.float32)

    x, y, w_box, h_box = cv2.boundingRect(cnt)
    aspect = h_box / (w_box + 1e-6)
    area = cv2.contourArea(cnt) / float(h * w + 1e-6)

    y_mid = y + h_box // 2
    upper = (mask[:y_mid] > 0).sum() / (h * w + 1e-6)
    lower = (mask[y_mid:] > 0).sum() / (h * w + 1e-6)

    ys = [int(y + h_box * r) for r in (0.2, 0.4, 0.6, 0.8)]
    widths = []
    for yy in ys:
        if yy <= 0 or yy >= h:
            widths.append(0.0)
            continue
        row = mask[yy]
        xs = np.where(row > 0)[0]
        widths.append(0.0 if len(xs) < 2 else (xs[-1] - xs[0]) / (w_box + 1e-6))

    peri = cv2.arcLength(cnt, True)
    compact = (peri ** 2) / (4.0 * np.pi * (cv2.contourArea(cnt) + 1e-6))

    feats = np.array(
        [aspect, area, upper, lower, widths[0], widths[1], widths[2], widths[3], compact, h_box/(h+1e-6)],
        dtype=np.float32,
    )
    return feats


def pose_features_plus(bgr: np.ndarray) -> np.ndarray:
    h, w, _ = bgr.shape
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = mp_pose.process(rgb)

    if not res.pose_landmarks:
        return np.zeros(8, dtype=np.float32)

    lm = res.pose_landmarks.landmark

    def xy(i):
        return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)

    L_S, R_S = 11, 12
    L_H, R_H = 23, 24
    L_A, R_A = 27, 28
    NOSE = 0

    shoulder_w = np.linalg.norm(xy(L_S) - xy(R_S))
    hip_w = np.linalg.norm(xy(L_H) - xy(R_H))
    torso_len = np.linalg.norm(xy(NOSE) - (xy(L_H) + xy(R_H)) / 2)
    leg_len = np.linalg.norm((xy(L_H) + xy(R_H)) / 2 - (xy(L_A) + xy(R_A)) / 2)

    height_pix = abs(lm[NOSE].y * h - min(xy(L_A)[1], xy(R_A)[1])) + 1e-6

    feats = np.array([
        shoulder_w / height_pix,
        hip_w / height_pix,
        torso_len / height_pix,
        leg_len / height_pix,
        shoulder_w / (hip_w + 1e-6),
        torso_len / (leg_len + 1e-6),
        torso_len / height_pix,
        leg_len / height_pix,
    ], dtype=np.float32)

    return feats


def extract_mixed_features_plus(bgr: np.ndarray) -> np.ndarray:
    cf = contour_features_plus(bgr)
    pf = pose_features_plus(bgr)
    return np.concatenate([cf, pf], axis=0)
