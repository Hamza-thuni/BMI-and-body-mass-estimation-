"""
preprocessing_timeline_viz.py
------------------------------
Visualises the full preprocessing timeline (V1→V9) for:
  - 2 images from 2DImage2BMI dataset
  - 2 images from Celeb-FBI dataset

Produces a publication-quality figure saved as preprocessing_timeline.png
"""

import os, re, sys, warnings
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = r"d:\Fyp_project_root\Fyp_project_root (1)"
DATASET_2D  = os.path.join(ROOT, "data (1)", "2DImage2BMI-main (1)",
                           "2DImage2BMI-main (1)", "datasets (1)", "Image_train (1)")
DATASET_CELEB = os.path.join(ROOT, "data (1)", "Celeb-FBI Dataset")
TASK_MODEL  = os.path.join(ROOT, "src", "pose_landmarker_heavy.task")
OUTPUT_PATH = os.path.join(ROOT, "src", "preprocessing_timeline.png")

# ── Colour palette ────────────────────────────────────────────────────────────
BG_DARK   = "#0d1117"
PANEL_BG  = "#161b22"
ACCENT1   = "#58a6ff"   # blue  – 2DImage2BMI
ACCENT2   = "#3fb950"   # green – Celeb-FBI
STAGE_COLS = {
    "Original"      : "#8b949e",
    "YOLO Crop"     : "#d2a679",
    "Segmentation"  : "#a371f7",
    "Pose Overlay"  : "#58a6ff",
    "Contour Mask"  : "#f78166",
    "Phys Dims"     : "#3fb950",
    "Final Feature" : "#e3b341",
}


# ═══════════════════════════════════════════════════════════════════════════════
# LABEL PARSERS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_2d_filename(fname):
    """→ (height_m, weight_kg, bmi, sex, age)"""
    name  = os.path.splitext(fname)[0]
    parts = name.split("_")
    try:
        sex   = parts[1].upper()
        age   = int(parts[2])
        m3    = re.search(r'\d+', parts[3]); h_m  = int(m3.group()) / 100_000.0
        m4    = re.search(r'\d+', parts[4]); w_kg = int(m4.group()) / 100_000.0
        bmi   = w_kg / (h_m**2 + 1e-8)
        return h_m, w_kg, bmi, sex, age
    except Exception:
        return None, None, None, "?", 0


def parse_celeb_filename(fname):
    """→ (height_m, weight_kg, bmi, sex, age)"""
    name  = os.path.splitext(fname)[0]
    parts = name.split("_")
    try:
        h_str = parts[1]; w_str = parts[2]
        ft_in = h_str.rstrip("h").split(".")
        ft = float(ft_in[0]); inches = float(ft_in[1]) if len(ft_in) > 1 else 0
        h_m  = (ft * 12 + inches) * 0.0254
        w_kg = float(w_str.rstrip("w"))
        bmi  = w_kg / (h_m**2 + 1e-8)
        sex  = parts[3].upper() if len(parts) > 3 else "?"
        age  = int(parts[4].rstrip("a")) if len(parts) > 4 else 0
        return h_m, w_kg, bmi, sex, age
    except Exception:
        return None, None, None, "?", 0


# ═══════════════════════════════════════════════════════════════════════════════
# OPENCV PREPROCESSING STAGES
# ═══════════════════════════════════════════════════════════════════════════════

def stage_original(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def stage_yolo_crop(bgr):
    """YOLOv8s person detection and crop."""
    try:
        from ultralytics import YOLO
        yolo_path = os.path.join(ROOT, "yolov8s.pt")
        if not os.path.exists(yolo_path):
            yolo_path = os.path.join(ROOT, "Fyp_project_root (1)", "yolov8s.pt")
        yolo = YOLO(yolo_path, verbose=False)
        results = yolo.predict(bgr[..., ::-1], conf=0.35, verbose=False, classes=[0])
        boxes = results[0].boxes
        if boxes and len(boxes.xyxy) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls  = boxes.cls.cpu().numpy()
            pbs  = [xyxy[i] for i in range(len(xyxy)) if int(cls[i]) == 0]
            if pbs:
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b in pbs]
                x1,y1,x2,y2 = map(int, pbs[int(np.argmax(areas))])
                h,w = bgr.shape[:2]
                x1=max(0,x1); y1=max(0,y1); x2=min(w-1,x2); y2=min(h-1,y2)
                crop = bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    # Draw box on original for visualization
                    vis = bgr.copy()
                    cv2.rectangle(vis, (x1,y1), (x2,y2), (0,200,100), 3)
                    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), crop
    except Exception as e:
        print(f"  [YOLO] {e}")
    h,w = bgr.shape[:2]
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), bgr


def _run_tasks_landmarker(bgr, segmentation=False):
    """Shared Tasks API runner. Returns (pts_orig, bin_mask_or_None)."""
    import mediapipe as mp
    if not os.path.exists(TASK_MODEL):
        return None, None
    BaseOptions        = mp.tasks.BaseOptions
    PoseLandmarker     = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOpts = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode  = mp.tasks.vision.RunningMode
    opts = PoseLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=TASK_MODEL),
        running_mode=VisionRunningMode.IMAGE,
        output_segmentation_masks=segmentation,
        min_pose_detection_confidence=0.45)
    target = 640
    orig_h, orig_w = bgr.shape[:2]
    scale  = target / max(orig_h, orig_w)
    new_w, new_h = int(orig_w*scale), int(orig_h*scale)
    bgr_r  = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_t  = (target-new_h)//2; pad_l = (target-new_w)//2
    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    canvas[pad_t:pad_t+new_h, pad_l:pad_l+new_w] = bgr_r
    rgb = np.ascontiguousarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    with PoseLandmarker.create_from_options(opts) as lm:
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.pose_landmarks:
        return None, None
    pts_pad  = np.array([[l.x*target, l.y*target] for l in res.pose_landmarks[0]])
    pts_orig = np.zeros_like(pts_pad)
    pts_orig[:,0] = (pts_pad[:,0]-pad_l)/scale
    pts_orig[:,1] = (pts_pad[:,1]-pad_t)/scale
    mask = None
    if segmentation and res.segmentation_masks:
        m_pad = res.segmentation_masks[0].numpy_view()
        # Shrink to new_h x new_w region, then resize to orig
        m_crop = m_pad[pad_t:pad_t+new_h, pad_l:pad_l+new_w]
        mask   = cv2.resize((m_crop > 0.5).astype(np.uint8), (orig_w, orig_h),
                            interpolation=cv2.INTER_NEAREST)
    return pts_orig, mask


def stage_mediapipe_pose(bgr):
    """Pose landmarks drawn via Tasks API."""
    try:
        pts, _ = _run_tasks_landmarker(bgr, segmentation=False)
        vis = bgr.copy()
        if pts is not None:
            CONNECTIONS = [
                (11,12),(11,13),(13,15),(12,14),(14,16),
                (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28)
            ]
            for a,b in CONNECTIONS:
                p1 = tuple(pts[a].astype(int)); p2 = tuple(pts[b].astype(int))
                cv2.line(vis, p1, p2, (0,200,100), 2)
            for p in pts:
                cv2.circle(vis, tuple(p.astype(int)), 4, (86,166,255), -1)
        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), pts
    except Exception as e:
        print(f"  [Pose] {e}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None


def stage_contour_mask(bgr):
    """V1/V2: Adaptive threshold contour mask."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thr  = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 41, 5)
    mask = cv2.morphologyEx(thr, cv2.MORPH_CLOSE,
                             np.ones((5,5), np.uint8), iterations=2)
    colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    # Green overlay for mask
    overlay = np.zeros_like(bgr)
    overlay[mask > 0] = [0, 200, 80]
    vis = cv2.addWeighted(bgr, 0.5, overlay, 0.5, 0)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def stage_segmentation(bgr):
    """V7: Tasks API segmentation mask → background removal."""
    try:
        _, bin_mask = _run_tasks_landmarker(bgr, segmentation=True)
        if bin_mask is not None:
            kern = np.ones((7,7), np.uint8)
            bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kern)
            bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_DILATE, kern, iterations=1)
            bg = np.full_like(bgr, 128)
            m3 = bin_mask[:,:,None]
            segmented = (bgr * m3 + bg * (1 - m3)).astype(np.uint8)
            return cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"  [Seg] {e}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def stage_physical_dims(bgr, true_h_m, true_w_kg):
    """V8/V9: Physical dimension overlay using known height."""
    try:
        import mediapipe as mp
        if not os.path.exists(TASK_MODEL):
            raise FileNotFoundError(f"Task model not found at {TASK_MODEL}")

        BaseOptions        = mp.tasks.BaseOptions
        PoseLandmarker     = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOpts = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode  = mp.tasks.vision.RunningMode

        opts = PoseLandmarkerOpts(
            base_options=BaseOptions(model_asset_path=TASK_MODEL),
            running_mode=VisionRunningMode.IMAGE,
            output_segmentation_masks=True,
            min_pose_detection_confidence=0.5)

        target = 640
        orig_h, orig_w = bgr.shape[:2]
        scale  = target / max(orig_h, orig_w)
        new_w, new_h = int(orig_w*scale), int(orig_h*scale)
        bgr_r = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pad_t = (target-new_h)//2; pad_l = (target-new_w)//2
        canvas = np.zeros((target, target, 3), dtype=np.uint8)
        canvas[pad_t:pad_t+new_h, pad_l:pad_l+new_w] = bgr_r
        rgb = np.ascontiguousarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

        with PoseLandmarker.create_from_options(opts) as lm:
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        if not res.pose_landmarks:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), {}

        pts_pad  = np.array([[l.x*target, l.y*target] for l in res.pose_landmarks[0]])
        pts_orig = np.zeros_like(pts_pad)
        pts_orig[:,0] = (pts_pad[:,0]-pad_l)/scale
        pts_orig[:,1] = (pts_pad[:,1]-pad_t)/scale

        min_y, max_y = pts_orig[:,1].min(), pts_orig[:,1].max()
        pix_h = max_y - min_y
        if pix_h < 10:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), {}

        px_pm  = pix_h / true_h_m
        sh_w   = np.linalg.norm(pts_orig[11]-pts_orig[12]) / px_pm
        hip_w  = np.linalg.norm(pts_orig[23]-pts_orig[24]) / px_pm
        mid_sh = (pts_orig[11]+pts_orig[12])/2
        mid_hp = (pts_orig[23]+pts_orig[24])/2
        torso  = np.linalg.norm(mid_sh-mid_hp) / px_pm

        # Segmentation area
        phys = {"height_m": true_h_m, "weight_kg": true_w_kg,
                "shoulder_m": sh_w, "hip_m": hip_w, "torso_m": torso}

        # Draw overlay on original image
        vis = bgr.copy()
        col = (0, 200, 100)
        def ip(i): return (int(pts_orig[i,0]), int(pts_orig[i,1]))
        cv2.line(vis, ip(11), ip(12), (255,200,0), 3)   # shoulder bar
        cv2.line(vis, ip(23), ip(24), (0,180,255), 3)   # hip bar
        cv2.line(vis, tuple(mid_sh.astype(int)), tuple(mid_hp.astype(int)), (200,100,255), 2)

        # Labels
        cv2.putText(vis, f"H:{true_h_m:.2f}m  W:{true_w_kg:.0f}kg",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(vis, f"Sh:{sh_w:.2f}m  Hip:{hip_w:.2f}m  Torso:{torso:.2f}m",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,255,100), 2)

        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), phys

    except Exception as e:
        print(f"  [PhysDims] {e}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), {}


def stage_feature_bar(phys, deep_dim=1536, pose_dim=24):
    """Render a bar chart showing the feature vector composition."""
    fig2, ax = plt.subplots(figsize=(4, 2), facecolor=PANEL_BG)
    dims   = []
    labels = []
    colors = []
    if phys:
        dims.append(5);        labels.append("Physical\n(5D)");      colors.append("#3fb950")
    dims.append(pose_dim);     labels.append(f"Pose\n({pose_dim}D)"); colors.append("#58a6ff")
    dims.append(deep_dim);     labels.append(f"Deep CNN\n({deep_dim}D)"); colors.append("#e3b341")

    total = sum(dims)
    left  = 0
    for d, l, c in zip(dims, labels, colors):
        ax.barh(0, d, left=left, color=c, height=0.5, edgecolor="none")
        if d > 20:
            ax.text(left + d/2, 0, l, ha="center", va="center",
                    fontsize=7, color="black", fontweight="bold")
        left += d

    ax.set_xlim(0, total)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
    ax.set_facecolor(PANEL_BG)
    fig2.patch.set_facecolor(PANEL_BG)
    ax.set_title(f"Feature Vector: {total}D", color="white", fontsize=8, pad=4)

    fig2.canvas.draw()
    w2, h2 = fig2.canvas.get_width_height()
    raw = fig2.canvas.buffer_rgba()
    buf = np.frombuffer(raw, dtype=np.uint8).reshape(h2, w2, 4)[:, :, :3]
    plt.close(fig2)
    return buf


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FIGURE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def get_sample_images(directory, n=2):
    files = sorted([f for f in os.listdir(directory)
                    if f.lower().endswith(('.jpg','.jpeg','.png'))])
    return files[:n]


def process_single_image(img_path, dataset_type, fname):
    """Run all stages and return list of (title, rgb_image) tuples."""
    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"  [WARN] Could not read {img_path}")
        return []

    stages = []

    # ── Stage 0: Original ──────────────────────────────────────────────────
    print("    Stage 0: Original...")
    if dataset_type == "2d":
        h_m, w_kg, bmi, sex, age = parse_2d_filename(fname)
    else:
        h_m, w_kg, bmi, sex, age = parse_celeb_filename(fname)

    orig_rgb = stage_original(bgr)
    label = (f"Original\n{sex}, {age}y  H:{h_m:.2f}m  W:{w_kg:.0f}kg\nBMI:{bmi:.1f}"
             if h_m else "Original")
    stages.append(("Original", label, orig_rgb))

    # ── Stage 1: V1/V2 Contour Mask ───────────────────────────────────────
    print("    Stage 1: Contour mask...")
    stages.append(("Contour Mask", "V1/V2: Adaptive\nThreshold\nContour Mask",
                   stage_contour_mask(bgr)))

    # ── Stage 2: V1/V2 Pose Overlay ───────────────────────────────────────
    print("    Stage 2: Pose overlay...")
    pose_rgb, lmarks = stage_mediapipe_pose(bgr)
    stages.append(("Pose Overlay", "V1/V2 Pose:\nMediaPipe\nLandmarks", pose_rgb))

    # ── Stage 3: V4 YOLO Crop ─────────────────────────────────────────────
    print("    Stage 3: YOLO crop...")
    yolo_vis, person_crop = stage_yolo_crop(bgr)
    stages.append(("YOLO Crop", "V4: YOLOv8s\nPerson\nDetect & Crop", yolo_vis))

    # ── Stage 4: V7 Segmentation ──────────────────────────────────────────
    print("    Stage 4: Segmentation...")
    crop_to_seg = person_crop if person_crop.shape[0] > 10 else bgr
    stages.append(("Segmentation",
                   "V7: MediaPipe\nSegmentation\n(BG → Grey)",
                   stage_segmentation(crop_to_seg)))

    # ── Stage 5: V8/V9 Physical Dims ──────────────────────────────────────
    if h_m and w_kg:
        print("    Stage 5: Physical dims...")
        phys_rgb, phys_vals = stage_physical_dims(bgr, h_m, w_kg)
        stages.append(("Phys Dims",
                       "V8/V9: Physical\nDimensions\n(m, m²)",
                       phys_rgb))
    else:
        phys_vals = {}
        stages.append(("Phys Dims", "Physical Dims\n(N/A)", orig_rgb.copy()))

    # ── Stage 6: Feature composition bar ──────────────────────────────────
    print("    Stage 6: Feature bar...")
    use_phys = bool(phys_vals)
    if dataset_type == "celeb":
        bar = stage_feature_bar(phys_vals, deep_dim=1536, pose_dim=24)
    else:
        bar = stage_feature_bar(None,      deep_dim=2048, pose_dim=13)
    stages.append(("Final Feature", "Final Feature\nVector\nComposition", bar))

    return stages


def build_figure(images_data):
    """
    images_data: list of (fname, dataset_label, accent_color, stages_list)
    """
    n_images = len(images_data)
    n_stages = max(len(d[3]) for d in images_data)

    fig_w = 3.2 * n_stages + 0.5
    fig_h = 3.8 * n_images + 1.8
    fig   = plt.figure(figsize=(fig_w, fig_h), facecolor=BG_DARK)

    # Title
    fig.text(0.5, 0.985, "Preprocessing Timeline  —  BMI & Body Mass Estimation",
             ha="center", va="top", fontsize=17, fontweight="bold", color="white")
    fig.text(0.5, 0.968, "V1/V2 (ResNet+Anthro)  →  V4 (EfficientNet+YOLO)  →  "
             "V7 (Segmentation+Pose)  →  V8/V9 (Physics+Hybrid Fusion)",
             ha="center", va="top", fontsize=9, color="#8b949e")

    gs = GridSpec(n_images, n_stages,
                  figure=fig,
                  left=0.04, right=0.98,
                  top=0.93, bottom=0.04,
                  hspace=0.55, wspace=0.12)

    STAGE_COLOR_MAP = {
        "Original"      : "#8b949e",
        "Contour Mask"  : "#f78166",
        "Pose Overlay"  : "#58a6ff",
        "YOLO Crop"     : "#d2a679",
        "Segmentation"  : "#a371f7",
        "Phys Dims"     : "#3fb950",
        "Final Feature" : "#e3b341",
    }

    for row, (fname, ds_label, accent, stages) in enumerate(images_data):
        for col, (stage_key, title, img) in enumerate(stages):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor(PANEL_BG)

            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(STAGE_COLOR_MAP.get(stage_key, "#444"))
                spine.set_linewidth(2.0)

            # Stage title
            ax.set_title(title, fontsize=7.5, color="white", pad=3,
                         multialignment="center",
                         bbox=dict(boxstyle="round,pad=0.2",
                                   fc=STAGE_COLOR_MAP.get(stage_key, "#222"),
                                   alpha=0.85, ec="none"))

            # Arrow between stages
            if col < len(stages) - 1:
                ax.annotate("", xy=(1.07, 0.5), xytext=(1.0, 0.5),
                            xycoords="axes fraction",
                            arrowprops=dict(arrowstyle="->",
                                            color="#8b949e", lw=1.5))

        # Row label (dataset)
        fig.text(0.01, 1.0 - (row + 0.5) / n_images * 0.90,
                 ds_label, ha="left", va="center",
                 rotation=90, fontsize=9, fontweight="bold", color=accent)

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=k)
                      for k, c in STAGE_COLOR_MAP.items()]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=len(legend_patches), fontsize=8,
               facecolor=PANEL_BG, edgecolor="#30363d",
               labelcolor="white", framealpha=0.9,
               bbox_to_anchor=(0.5, 0.005))

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Preprocessing Timeline Visualiser")
    print("=" * 65)

    # ── Pick 2 images from each dataset ───────────────────────────────────
    twod_files  = get_sample_images(DATASET_2D,    n=2)
    celeb_files = get_sample_images(DATASET_CELEB, n=2)

    print(f"\n2DImage2BMI  images : {twod_files}")
    print(f"Celeb-FBI    images : {celeb_files}")

    all_data = []

    for i, fname in enumerate(twod_files):
        print(f"\n[2DImage2BMI {i+1}] Processing: {fname}")
        path   = os.path.join(DATASET_2D, fname)
        stages = process_single_image(path, "2d", fname)
        label  = f"2DImage2BMI\n#{i+1}"
        all_data.append((fname, label, ACCENT1, stages))

    for i, fname in enumerate(celeb_files):
        print(f"\n[Celeb-FBI {i+1}] Processing: {fname}")
        path   = os.path.join(DATASET_CELEB, fname)
        stages = process_single_image(path, "celeb", fname)
        label  = f"Celeb-FBI\n#{i+1}"
        all_data.append((fname, label, ACCENT2, stages))

    print("\n[BUILD] Composing figure...")
    fig = build_figure(all_data)

    print(f"[SAVE] Saving → {OUTPUT_PATH}")
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight",
                facecolor=BG_DARK)
    plt.close(fig)
    print(f"[DONE] Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
