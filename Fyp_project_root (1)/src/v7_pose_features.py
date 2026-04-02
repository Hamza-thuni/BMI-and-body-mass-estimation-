"""
v7_pose_features.py
-------------------
Extended 24-D scale-invariant pose feature extractor for the V7 pipeline.

Improvements over V6 (16-D):
  - Added per-limb segment ratios (upper arm, forearm, thigh, shin separately)
  - Added head/ear width proxy (correlates with face/head shape)
  - Added cross-body diagonal ratios (trunk shape indicator)
  - Added arm-to-leg and torso-to-leg ratio features

All features are normalised by estimated body height (mid-shoulder to mid-ankle)
so they are invariant to camera distance.
"""

import numpy as np


# MediaPipe Pose landmark indices
_NOSE   = 0
_L_EAR  = 7;  _R_EAR  = 8
_L_SH   = 11; _R_SH   = 12
_L_ELB  = 13; _R_ELB  = 14
_L_WR   = 15; _R_WR   = 16
_L_HIP  = 23; _R_HIP  = 24
_L_KNEE = 25; _R_KNEE = 26
_L_ANK  = 27; _R_ANK  = 28


def extract_rich_pose_features(coords: np.ndarray) -> np.ndarray:
    """
    Compute 24 scale-invariant skeletal proportion features.

    Args:
        coords: np.ndarray of shape (33, 3) — MediaPipe landmark pixel
                coordinates [x, y, z] in image space.

    Returns:
        np.ndarray of shape (24,), dtype float32.
        Returns zeros if body_height is degenerate (< 10 px).
    """
    def pt(idx):
        return coords[idx, :2]

    def dist(a, b):
        return float(np.linalg.norm(pt(a) - pt(b)))

    # --- Body height normalisation ---
    mid_sh  = (pt(_L_SH)  + pt(_R_SH))  / 2.0
    mid_ank = (pt(_L_ANK) + pt(_R_ANK)) / 2.0
    body_h  = float(np.linalg.norm(mid_sh - mid_ank))

    if body_h < 10.0:
        return np.zeros(24, dtype=np.float32)

    def rd(a, b):
        """Normalised distance between landmarks a and b."""
        return dist(a, b) / body_h

    # ---- Width ratios ----
    shoulder_w   = rd(_L_SH,  _R_SH)           # f0  shoulder breadth
    hip_w        = rd(_L_HIP, _R_HIP)           # f1  hip breadth
    ear_w        = rd(_L_EAR, _R_EAR)           # f2  head width proxy
    sh_hip_r     = shoulder_w / (hip_w + 1e-6)  # f3  shoulder-to-hip ratio

    # ---- Torso ----
    l_torso      = rd(_L_SH, _L_HIP)            # f4  left torso length
    r_torso      = rd(_R_SH, _R_HIP)            # f5  right torso length
    torso_avg    = (l_torso + r_torso) / 2.0    # f6  avg torso length
    neck_h       = float(np.linalg.norm(pt(_NOSE) - mid_sh)) / body_h  # f7  nose-to-shoulder

    # ---- Arms ----
    l_upper_arm  = rd(_L_SH,  _L_ELB)           # f8
    r_upper_arm  = rd(_R_SH,  _R_ELB)           # f9
    l_forearm    = rd(_L_ELB, _L_WR)            # f10
    r_forearm    = rd(_R_ELB, _R_WR)            # f11
    l_arm        = l_upper_arm + l_forearm       # f12  total left arm
    r_arm        = r_upper_arm + r_forearm       # f13  total right arm

    # ---- Legs ----
    l_thigh      = rd(_L_HIP,  _L_KNEE)         # f14
    r_thigh      = rd(_R_HIP,  _R_KNEE)         # f15
    l_shin       = rd(_L_KNEE, _L_ANK)          # f16
    r_shin       = rd(_R_KNEE, _R_ANK)          # f17
    l_leg        = l_thigh + l_shin             # f18  total left leg
    r_leg        = r_thigh + r_shin             # f19  total right leg

    # ---- Cross-body diagonals (trunk shape) ----
    cross_l      = rd(_L_SH, _R_HIP)            # f20  left-shoulder to right-hip
    cross_r      = rd(_R_SH, _L_HIP)            # f21  right-shoulder to left-hip

    # ---- Proportion ratios ----
    leg_avg      = (l_leg + r_leg) / 2.0 + 1e-6
    arm_avg      = (l_arm + r_arm) / 2.0 + 1e-6
    torso_leg_r  = torso_avg / leg_avg           # f22  torso vs. leg
    arm_leg_r    = arm_avg   / leg_avg           # f23  arm vs. leg

    return np.array([
        shoulder_w, hip_w, ear_w, sh_hip_r,      # 0-3
        l_torso, r_torso, torso_avg, neck_h,      # 4-7
        l_upper_arm, r_upper_arm,                  # 8-9
        l_forearm,   r_forearm,                    # 10-11
        l_arm,       r_arm,                        # 12-13
        l_thigh,     r_thigh,                      # 14-15
        l_shin,      r_shin,                       # 16-17
        l_leg,       r_leg,                        # 18-19
        cross_l,     cross_r,                      # 20-21
        torso_leg_r, arm_leg_r,                   # 22-23
    ], dtype=np.float32)
