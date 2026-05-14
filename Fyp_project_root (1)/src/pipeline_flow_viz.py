"""
pipeline_flow_viz.py
--------------------
1 image from 2DImage2BMI  +  1 image from Celeb-FBI
Shows the full preprocessing flow with prominent arrows.
Output: pipeline_flow.png
"""

import os, re, sys, warnings
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = r"d:\Fyp_project_root\Fyp_project_root (1)"
DIR_2D      = os.path.join(ROOT, "data (1)", "2DImage2BMI-main (1)",
                            "2DImage2BMI-main (1)", "datasets (1)", "Image_train (1)")
DIR_CELEB   = os.path.join(ROOT, "data (1)", "Celeb-FBI Dataset")
TASK_MODEL  = os.path.join(ROOT, "src", "pose_landmarker_heavy.task")
OUT_PATH    = os.path.join(ROOT, "src", "pipeline_flow.png")

# ── Theme ─────────────────────────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
WHITE   = "#f0f6fc"
GREY    = "#8b949e"

STAGE_META = [
    ("Original",     "#8b949e", "Original\nImage"),
    ("Contour",      "#f78166", "V1/V2\nContour Mask"),
    ("Pose",         "#58a6ff", "V1–V6\nPose Overlay"),
    ("YOLO",         "#d2a679", "V4+\nYOLO Crop"),
    ("Segment",      "#a371f7", "V7\nSegmentation"),
    ("PhysDims",     "#3fb950", "V8/V9\nPhysical Dims"),
    ("Features",     "#e3b341", "Final\nFeature Vec"),
]

# ── Parsers ───────────────────────────────────────────────────────────────────
def parse_2d(fname):
    parts = os.path.splitext(fname)[0].split("_")
    try:
        sex = parts[1].upper(); age = int(parts[2])
        h = int(re.search(r'\d+', parts[3]).group()) / 100_000
        w = int(re.search(r'\d+', parts[4]).group()) / 100_000
        return h, w, w/(h**2+1e-8), sex, age
    except: return None,None,None,"?",0

def parse_celeb(fname):
    parts = os.path.splitext(fname)[0].split("_")
    try:
        ft_in = parts[1].rstrip("h").split(".")
        h = (float(ft_in[0])*12 + float(ft_in[1] if len(ft_in)>1 else 0))*0.0254
        w = float(parts[2].rstrip("w"))
        sex = parts[3].upper() if len(parts)>3 else "?"
        age = int(parts[4].rstrip("a")) if len(parts)>4 else 0
        return h, w, w/(h**2+1e-8), sex, age
    except: return None,None,None,"?",0

# ── MediaPipe Tasks helper ────────────────────────────────────────────────────
def run_landmarker(bgr, seg=False):
    import mediapipe as mp
    T = 640
    oh,ow = bgr.shape[:2]
    sc = T/max(oh,ow)
    nw,nh = int(ow*sc),int(oh*sc)
    res_bgr = cv2.resize(bgr,(nw,nh),interpolation=cv2.INTER_AREA)
    pt,pl = (T-nh)//2,(T-nw)//2
    can = np.zeros((T,T,3),dtype=np.uint8)
    can[pt:pt+nh,pl:pl+nw] = res_bgr
    rgb = np.ascontiguousarray(cv2.cvtColor(can,cv2.COLOR_BGR2RGB))

    opts = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=TASK_MODEL),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_segmentation_masks=seg,
        min_pose_detection_confidence=0.45)

    with mp.tasks.vision.PoseLandmarker.create_from_options(opts) as lm:
        r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    if not r.pose_landmarks:
        return None, None

    raw = np.array([[l.x*T,l.y*T] for l in r.pose_landmarks[0]])
    pts = np.zeros_like(raw)
    pts[:,0]=(raw[:,0]-pl)/sc; pts[:,1]=(raw[:,1]-pt)/sc

    mask=None
    if seg and r.segmentation_masks:
        m = r.segmentation_masks[0].numpy_view()
        mc = m[pt:pt+nh,pl:pl+nw]
        mask = cv2.resize((mc>0.5).astype(np.uint8),(ow,oh),interpolation=cv2.INTER_NEAREST)
    return pts, mask

# ── Stage functions ───────────────────────────────────────────────────────────
CONNS = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28)]

def s_original(bgr, info):
    h_m,w_kg,bmi,sex,age = info
    vis = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    txt = f"{sex}, {age}y\nH:{h_m:.2f}m  W:{w_kg:.0f}kg\nBMI:{bmi:.1f}" if h_m else ""
    return vis, txt

def s_contour(bgr, _):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thr  = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,41,5)
    mask = cv2.morphologyEx(thr,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8),iterations=2)
    ov   = np.zeros_like(bgr); ov[mask>0]=[0,200,80]
    vis  = cv2.addWeighted(bgr,0.45,ov,0.55,0)
    return cv2.cvtColor(vis,cv2.COLOR_BGR2RGB), "7-D contour features\nextracted"

def s_pose(bgr, _):
    vis = bgr.copy()
    try:
        pts,_ = run_landmarker(bgr, seg=False)
        if pts is not None:
            for a,b in CONNS:
                cv2.line(vis,tuple(pts[a].astype(int)),tuple(pts[b].astype(int)),(0,220,100),3)
            for p in pts:
                cv2.circle(vis,tuple(p.astype(int)),5,(86,166,255),-1)
    except Exception as e: print(f"  [Pose] {e}")
    return cv2.cvtColor(vis,cv2.COLOR_BGR2RGB), "24-D rich\npose features"

def s_yolo(bgr, _):
    vis = bgr.copy()
    crop = bgr.copy()
    try:
        from ultralytics import YOLO
        yp = os.path.join(ROOT,"yolov8s.pt")
        if not os.path.exists(yp): yp=os.path.join(ROOT,"Fyp_project_root (1)","yolov8s.pt")
        yolo = YOLO(yp,verbose=False)
        res = yolo.predict(bgr[...,::-1],conf=0.35,verbose=False,classes=[0])
        boxes=res[0].boxes
        if boxes and len(boxes.xyxy)>0:
            xyxy=boxes.xyxy.cpu().numpy(); cls=boxes.cls.cpu().numpy()
            pbs=[xyxy[i] for i in range(len(xyxy)) if int(cls[i])==0]
            if pbs:
                areas=[(b[2]-b[0])*(b[3]-b[1]) for b in pbs]
                x1,y1,x2,y2=map(int,pbs[int(np.argmax(areas))])
                h,w=bgr.shape[:2]
                x1=max(0,x1);y1=max(0,y1);x2=min(w-1,x2);y2=min(h-1,y2)
                crop=bgr[y1:y2,x1:x2]
                cv2.rectangle(vis,(x1,y1),(x2,y2),(0,220,100),4)
                for corner,dx,dy in [((x1,y1),1,1),((x2,y1),-1,1),((x1,y2),1,-1),((x2,y2),-1,-1)]:
                    cv2.line(vis,corner,(corner[0]+dx*20,corner[1]),(0,220,100),4)
                    cv2.line(vis,corner,(corner[0],corner[1]+dy*20),(0,220,100),4)
    except Exception as e: print(f"  [YOLO] {e}")
    return cv2.cvtColor(vis,cv2.COLOR_BGR2RGB), "Person isolated\nfor feature extract"

def s_segment(bgr, _):
    try:
        pts,mask = run_landmarker(bgr, seg=True)
        if mask is not None:
            k=np.ones((7,7),np.uint8)
            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k)
            mask=cv2.morphologyEx(mask,cv2.MORPH_DILATE,k,iterations=1)
            bg=np.full_like(bgr,128); m3=mask[:,:,None]
            seg=(bgr*m3+bg*(1-m3)).astype(np.uint8)
            return cv2.cvtColor(seg,cv2.COLOR_BGR2RGB),"BG → neutral grey\n(128,128,128)"
    except Exception as e: print(f"  [Seg] {e}")
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),"(segmentation\nnot available)"

def s_phys(bgr, info):
    h_m,w_kg,_,_,_ = info
    if not h_m:
        return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),"N/A"
    vis = bgr.copy()
    txt = "N/A"
    try:
        pts,_ = run_landmarker(bgr, seg=False)
        if pts is not None:
            min_y,max_y = pts[:,1].min(),pts[:,1].max()
            pix_h = max_y-min_y
            if pix_h > 10:
                ppm = pix_h/h_m
                sh_w  = np.linalg.norm(pts[11]-pts[12])/ppm
                hip_w = np.linalg.norm(pts[23]-pts[24])/ppm
                mid_sh=(pts[11]+pts[12])/2; mid_hp=(pts[23]+pts[24])/2
                torso = np.linalg.norm(mid_sh-mid_hp)/ppm
                cv2.line(vis,tuple(pts[11].astype(int)),tuple(pts[12].astype(int)),(255,200,0),4)
                cv2.line(vis,tuple(pts[23].astype(int)),tuple(pts[24].astype(int)),(0,200,255),4)
                cv2.line(vis,tuple(mid_sh.astype(int)),tuple(mid_hp.astype(int)),(220,100,255),3)
                cv2.putText(vis,f"Sh:{sh_w:.2f}m",(10,35),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,200,0),2)
                cv2.putText(vis,f"Hip:{hip_w:.2f}m",(10,65),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,200,255),2)
                cv2.putText(vis,f"Tor:{torso:.2f}m",(10,95),cv2.FONT_HERSHEY_SIMPLEX,0.9,(220,100,255),2)
                txt = f"area + sh({sh_w:.2f}m)\nhip({hip_w:.2f}m) + H({h_m:.2f}m)"
    except Exception as e: print(f"  [Phys] {e}")
    return cv2.cvtColor(vis,cv2.COLOR_BGR2RGB), txt

def s_features(phys_feats, dataset_type):
    """Stacked bar showing feature vector."""
    fig,ax = plt.subplots(figsize=(3.8,3.2),facecolor=PANEL)
    ax.set_facecolor(PANEL)

    if dataset_type=="celeb":
        segs = [("Physical\n5D",5,"#3fb950"),("Pose\n24D",24,"#58a6ff"),("Deep CNN\n1536D",1536,"#e3b341")]
    else:
        segs = [("Pose\n13D",13,"#58a6ff"),("Deep CNN\n2048D",2048,"#e3b341")]

    total=sum(d for _,d,_ in segs)
    y=0
    for lbl,d,col in segs:
        frac=d/total
        ax.barh(0,frac,left=y,color=col,height=0.55,edgecolor=BG,linewidth=1.5)
        if frac>0.03:
            ax.text(y+frac/2,0,lbl,ha="center",va="center",fontsize=9,
                    color="black",fontweight="bold")
        y+=frac

    ax.set_xlim(0,1); ax.set_ylim(-0.5,0.5)
    ax.axis("off")
    ax.set_title(f"Feature Vector  [{total}D]",color=WHITE,fontsize=11,pad=6,fontweight="bold")

    sub="V9 Hybrid: Phys+Pose+Deep" if dataset_type=="celeb" else "V1 Hybrid: Pose+Deep"
    ax.text(0.5,-0.42,sub,ha="center",transform=ax.transAxes,fontsize=8,color=GREY)

    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    w2,h2=fig.canvas.get_width_height()
    buf=np.frombuffer(fig.canvas.buffer_rgba(),dtype=np.uint8).reshape(h2,w2,4)[:,:,:3]
    plt.close(fig)
    return buf, f"{'1565' if dataset_type=='celeb' else '2061'}D total"

# ── Resize to uniform height ──────────────────────────────────────────────────
def uniform_h(img, h=380):
    oh,ow = img.shape[:2]
    nw = int(ow*h/oh)
    return cv2.resize(img,(nw,h),interpolation=cv2.INTER_AREA)

# ── Main ──────────────────────────────────────────────────────────────────────
def run_pipeline(bgr, info, dataset_type):
    stages_rgb = []
    stages_sub = []

    print("  Stage 0: Original"); r,s=s_original(bgr,info); stages_rgb.append(r); stages_sub.append(s)
    print("  Stage 1: Contour");  r,s=s_contour(bgr,info);  stages_rgb.append(r); stages_sub.append(s)
    print("  Stage 2: Pose");     r,s=s_pose(bgr,info);     stages_rgb.append(r); stages_sub.append(s)
    print("  Stage 3: YOLO");     r,s=s_yolo(bgr,info);     stages_rgb.append(r); stages_sub.append(s)
    print("  Stage 4: Segment");  r,s=s_segment(bgr,info);  stages_rgb.append(r); stages_sub.append(s)
    print("  Stage 5: PhysDims"); r,s=s_phys(bgr,info);     stages_rgb.append(r); stages_sub.append(s)
    print("  Stage 6: Features"); r,s=s_features(None,dataset_type); stages_rgb.append(r); stages_sub.append(s)

    return stages_rgb, stages_sub


def build_figure(row_data):
    """
    row_data: list of (dataset_label, accent_col, stages_rgb, stages_sub, stage_meta)
    """
    N_ROWS  = len(row_data)
    N_COLS  = len(STAGE_META)
    IMG_H   = 340   # px height for images

    # Figure size
    COL_W = 2.6; ROW_H = 4.0; ARROW_W = 0.45
    fw = N_COLS*COL_W + (N_COLS-1)*ARROW_W + 1.4
    fh = N_ROWS*ROW_H + 2.0
    fig = plt.figure(figsize=(fw,fh), facecolor=BG)

    # Title
    fig.text(0.5, 0.985,
             "End-to-End Preprocessing Pipeline — BMI & Body Mass Estimation",
             ha="center",va="top",fontsize=15,fontweight="bold",color=WHITE)
    fig.text(0.5, 0.962,
             "V1/V2  →  V4  →  V6  →  V7  →  V8/V9   |   Two Datasets — Same Pipeline",
             ha="center",va="top",fontsize=9,color=GREY)

    # Build a flat grid: N_ROWS rows, each row = N_COLS image panels + (N_COLS-1) arrow gaps
    total_cols = N_COLS + (N_COLS-1)  # alternating: img | arrow | img | arrow ...
    col_ratios = []
    for i in range(N_COLS):
        col_ratios.append(COL_W)
        if i < N_COLS-1:
            col_ratios.append(ARROW_W)

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(N_ROWS, len(col_ratios),
                  figure=fig,
                  width_ratios=col_ratios,
                  left=0.06, right=0.99,
                  top=0.93, bottom=0.06,
                  hspace=0.22, wspace=0.0)

    STAGE_COLOR_MAP = {s[0]: s[1] for s in STAGE_META}
    STAGE_TITLE_MAP = {s[0]: s[2] for s in STAGE_META}

    for row_idx, (ds_label, accent, stages_rgb, stages_sub, _) in enumerate(row_data):

        # Row label on left margin
        fig.text(0.005,
                 1.0 - (row_idx+0.5)/N_ROWS*0.88,
                 ds_label,
                 ha="left",va="center",rotation=90,
                 fontsize=11,fontweight="bold",color=accent)

        for col_idx, (key, col, title) in enumerate(STAGE_META):
            gs_col = col_idx * 2          # every other GridSpec column is an image

            ax = fig.add_subplot(gs[row_idx, gs_col])
            ax.set_facecolor(PANEL)

            img = stages_rgb[col_idx]
            img = uniform_h(img, IMG_H)
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])

            # Coloured border
            for sp in ax.spines.values():
                sp.set_edgecolor(col); sp.set_linewidth(2.5)

            # Stage title box
            ax.set_title(title, fontsize=8, color=WHITE, pad=3,
                         multialignment="center",
                         bbox=dict(boxstyle="round,pad=0.25",fc=col,alpha=0.88,ec="none"))

            # Sub-caption below
            sub = stages_sub[col_idx]
            ax.text(0.5,-0.04,sub,transform=ax.transAxes,
                    ha="center",va="top",fontsize=6.5,color=GREY,
                    multialignment="center")

            # ── Arrow between panels ──────────────────────────────────────
            if col_idx < N_COLS-1:
                ax_arr = fig.add_subplot(gs[row_idx, gs_col+1])
                ax_arr.set_facecolor(BG)
                ax_arr.axis("off")
                ax_arr.annotate(
                    "", xy=(0.85,0.5), xytext=(0.15,0.5),
                    xycoords="axes fraction",
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#c9d1d9",
                        lw=2.0,
                        mutation_scale=18,
                    )
                )
                # Version tag on arrow
                ver_tags = ["V1/V2","V4+","V6+","V7","V8/V9","—"]
                tag = ver_tags[col_idx] if col_idx < len(ver_tags) else ""
                ax_arr.text(0.5,0.68,tag,ha="center",va="center",
                            transform=ax_arr.transAxes,
                            fontsize=6,color=accent,fontweight="bold")

    # Legend at bottom
    patches = [mpatches.Patch(color=s[1],label=s[2].replace("\n"," ")) for s in STAGE_META]
    fig.legend(handles=patches,loc="lower center",ncol=N_COLS,fontsize=7.5,
               facecolor=PANEL,edgecolor=BORDER,labelcolor=WHITE,framealpha=0.9,
               bbox_to_anchor=(0.5,0.002))

    return fig


def main():
    print("="*60)
    print("  Pipeline Flow Visualiser — 1 image × 2 datasets")
    print("="*60)

    # Pick one image from each dataset
    f2d    = sorted(f for f in os.listdir(DIR_2D) if f.lower().endswith(('.jpg','.png')))[0]
    fceleb = sorted(f for f in os.listdir(DIR_CELEB) if f.lower().endswith(('.jpg','.png')))[2]

    print(f"\n2DImage2BMI : {f2d}")
    print(f"Celeb-FBI   : {fceleb}")

    bgr2d    = cv2.imread(os.path.join(DIR_2D,    f2d))
    bgr_cel  = cv2.imread(os.path.join(DIR_CELEB, fceleb))

    info2d   = parse_2d(f2d)
    info_cel = parse_celeb(fceleb)

    print("\n[2DImage2BMI] Running pipeline...")
    rgb2d, sub2d = run_pipeline(bgr2d, info2d, "2d")

    print("\n[Celeb-FBI]   Running pipeline...")
    rgb_cel, sub_cel = run_pipeline(bgr_cel, info_cel, "celeb")

    row_data = [
        ("2DImage2BMI", "#58a6ff", rgb2d,  sub2d,  STAGE_META),
        ("Celeb-FBI",   "#3fb950", rgb_cel, sub_cel, STAGE_META),
    ]

    print("\n[BUILD] Composing figure...")
    fig = build_figure(row_data)

    print(f"[SAVE] → {OUT_PATH}")
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[DONE]")


if __name__ == "__main__":
    main()
