"""
v9_pipeline_flow.py
--------------------
Shows the V9 hybrid pipeline for 1 image from each dataset:
  2DImage2BMI  and  Celeb-FBI

V9 Stages:
  Original → YOLO Crop → Segmentation → Physical Dims → Pose Features → Deep Crop → 1565D Feature Vec

Output: v9_pipeline_flow.png
"""

import os, re, warnings
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = r"d:\Fyp_project_root\Fyp_project_root (1)"
DIR_2D    = os.path.join(ROOT, "data (1)", "2DImage2BMI-main (1)",
                          "2DImage2BMI-main (1)", "datasets (1)", "Image_train (1)")
DIR_CELEB = os.path.join(ROOT, "data (1)", "Celeb-FBI Dataset")
TASK_MODEL= os.path.join(ROOT, "src", "pose_landmarker_heavy.task")
OUT_PATH  = os.path.join(ROOT, "src", "v9_pipeline_flow.png")

# ── Theme ─────────────────────────────────────────────────────────────────────
BG, PANEL, WHITE, GREY = "#0d1117", "#161b22", "#f0f6fc", "#8b949e"
ACC_2D, ACC_CEL = "#58a6ff", "#3fb950"

# V9-only stages
V9_STAGES = [
    ("original",   "#8b949e", "① Original\nInput Image"),
    ("yolo",       "#d2a679", "② YOLO\nPerson Crop"),
    ("segment",    "#a371f7", "③ MediaPipe\nSegmentation"),
    ("phys",       "#3fb950", "④ Physical\nDimensions"),
    ("pose",       "#58a6ff", "⑤ Rich Pose\nFeatures (24D)"),
    ("deep",       "#f78166", "⑥ EfficientNet\nDeep Crop (1536D)"),
    ("features",   "#e3b341", "⑦ Final\n1565D Vector"),
]

CONNS = [(11,12),(11,13),(13,15),(12,14),(14,16),
         (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28)]

# ── Parsers ───────────────────────────────────────────────────────────────────
def parse_2d(fname):
    parts = os.path.splitext(fname)[0].split("_")
    try:
        sex=parts[1].upper(); age=int(parts[2])
        h=int(re.search(r'\d+',parts[3]).group())/100_000
        w=int(re.search(r'\d+',parts[4]).group())/100_000
        return h,w,w/(h**2+1e-8),sex,age
    except: return None,None,None,"?",0

def parse_celeb(fname):
    parts=os.path.splitext(fname)[0].split("_")
    try:
        ft_in=parts[1].rstrip("h").split(".")
        h=(float(ft_in[0])*12+(float(ft_in[1]) if len(ft_in)>1 else 0))*0.0254
        w=float(parts[2].rstrip("w"))
        sex=parts[3].upper() if len(parts)>3 else "?"
        age=int(parts[4].rstrip("a")) if len(parts)>4 else 0
        return h,w,w/(h**2+1e-8),sex,age
    except: return None,None,None,"?",0

# ── MediaPipe Tasks helper ────────────────────────────────────────────────────
def run_landmarker(bgr, seg=True):
    import mediapipe as mp
    T=640; oh,ow=bgr.shape[:2]
    sc=T/max(oh,ow); nw,nh=int(ow*sc),int(oh*sc)
    res_bgr=cv2.resize(bgr,(nw,nh),interpolation=cv2.INTER_AREA)
    pt,pl=(T-nh)//2,(T-nw)//2
    can=np.zeros((T,T,3),dtype=np.uint8)
    can[pt:pt+nh,pl:pl+nw]=res_bgr
    rgb=np.ascontiguousarray(cv2.cvtColor(can,cv2.COLOR_BGR2RGB))
    opts=mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=TASK_MODEL),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_segmentation_masks=seg,
        min_pose_detection_confidence=0.45)
    with mp.tasks.vision.PoseLandmarker.create_from_options(opts) as lm:
        r=lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
    if not r.pose_landmarks:
        return None,None,None,(oh,ow,pt,pl,sc)
    raw=np.array([[l.x*T,l.y*T] for l in r.pose_landmarks[0]])
    pts=np.zeros_like(raw)
    pts[:,0]=(raw[:,0]-pl)/sc; pts[:,1]=(raw[:,1]-pt)/sc
    mask=None
    if seg and r.segmentation_masks:
        m=r.segmentation_masks[0].numpy_view()
        mc=m[pt:pt+nh,pl:pl+nw]
        mask=cv2.resize((mc>0.5).astype(np.uint8),(ow,oh),interpolation=cv2.INTER_NEAREST)
    return pts, mask, r, (oh,ow,pt,pl,sc)

# ── V9 stage processors ───────────────────────────────────────────────────────
def v9_original(bgr, h_m, w_kg, bmi, sex, age):
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    caption=f"{sex}, {age}y  |  H: {h_m:.2f}m  W: {w_kg:.0f}kg  BMI: {bmi:.1f}" if h_m else ""
    return rgb, caption

def v9_yolo(bgr):
    vis=bgr.copy(); crop=bgr; box=None
    try:
        from ultralytics import YOLO
        yp=os.path.join(ROOT,"yolov8s.pt")
        if not os.path.exists(yp):
            yp=os.path.join(ROOT,"Fyp_project_root (1)","yolov8s.pt")
        yolo=YOLO(yp,verbose=False)
        res=yolo.predict(bgr[...,::-1],conf=0.35,verbose=False,classes=[0])
        bxs=res[0].boxes
        if bxs and len(bxs.xyxy)>0:
            xyxy=bxs.xyxy.cpu().numpy(); cls=bxs.cls.cpu().numpy()
            pbs=[xyxy[i] for i in range(len(xyxy)) if int(cls[i])==0]
            if pbs:
                h,w=bgr.shape[:2]
                areas=[(b[2]-b[0])*(b[3]-b[1]) for b in pbs]
                x1,y1,x2,y2=map(int,pbs[int(np.argmax(areas))])
                x1,y1=max(0,x1),max(0,y1); x2,y2=min(w-1,x2),min(h-1,y2)
                crop=bgr[y1:y2,x1:x2]; box=(x1,y1,x2,y2)
                cv2.rectangle(vis,(x1,y1),(x2,y2),(100,220,100),4)
                for cx,cy,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                    cv2.line(vis,(cx,cy),(cx+dx*24,cy),(100,220,100),5)
                    cv2.line(vis,(cx,cy),(cx,cy+dy*24),(100,220,100),5)
    except Exception as e:
        print(f"    [YOLO] {e}")
    return cv2.cvtColor(vis,cv2.COLOR_BGR2RGB), crop, box, "YOLOv8s — conf=0.35\nlargest person box"

def v9_segment(bgr):
    """Background → neutral grey (128,128,128)."""
    try:
        pts,mask,_,_ = run_landmarker(bgr, seg=True)
        if mask is not None:
            k=np.ones((7,7),np.uint8)
            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k)
            mask=cv2.morphologyEx(mask,cv2.MORPH_DILATE,k,iterations=1)
            bg=np.full_like(bgr,128); m3=mask[:,:,None]
            seg=(bgr*m3+bg*(1-m3)).astype(np.uint8)
            return cv2.cvtColor(seg,cv2.COLOR_BGR2RGB), pts, mask, "BG → (128,128,128)\nthreshold=0.5 + morph"
    except Exception as e:
        print(f"    [Seg] {e}")
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB), None, None, "(fallback: no mask)"

def v9_phys(bgr, h_m):
    """Physical measurements in metres."""
    vis=bgr.copy(); caption="h_m required"
    if not h_m:
        return cv2.cvtColor(vis,cv2.COLOR_BGR2RGB), {}, caption
    try:
        pts,_,_,_ = run_landmarker(bgr, seg=False)
        if pts is not None:
            min_y,max_y=pts[:,1].min(),pts[:,1].max()
            pix_h=max_y-min_y
            if pix_h>10:
                ppm=pix_h/h_m
                sh_w =np.linalg.norm(pts[11]-pts[12])/ppm
                hip_w=np.linalg.norm(pts[23]-pts[24])/ppm
                msh=(pts[11]+pts[12])/2; mhp=(pts[23]+pts[24])/2
                torso=np.linalg.norm(msh-mhp)/ppm

                # Draw measurements
                cv2.line(vis,tuple(pts[11].astype(int)),tuple(pts[12].astype(int)),(255,200,0),4)
                cv2.line(vis,tuple(pts[23].astype(int)),tuple(pts[24].astype(int)),(0,220,255),4)
                cv2.line(vis,tuple(msh.astype(int)),tuple(mhp.astype(int)),(220,80,255),3)
                # Height bar on side
                h_img=vis.shape[0]
                bar_x=vis.shape[1]-15
                y_top=int(pts[:,1].min()); y_bot=int(pts[:,1].max())
                cv2.line(vis,(bar_x,y_top),(bar_x,y_bot),(255,255,100),4)
                cv2.line(vis,(bar_x-8,y_top),(bar_x+8,y_top),(255,255,100),3)
                cv2.line(vis,(bar_x-8,y_bot),(bar_x+8,y_bot),(255,255,100),3)

                fs=max(0.55, min(0.9, vis.shape[0]/500))
                cv2.putText(vis,f"Sh: {sh_w:.2f}m",(8,32),cv2.FONT_HERSHEY_SIMPLEX,fs,(255,200,0),2)
                cv2.putText(vis,f"Hip:{hip_w:.2f}m",(8,60),cv2.FONT_HERSHEY_SIMPLEX,fs,(0,220,255),2)
                cv2.putText(vis,f"Tor:{torso:.2f}m",(8,88),cv2.FONT_HERSHEY_SIMPLEX,fs,(220,80,255),2)
                cv2.putText(vis,f"H:  {h_m:.2f}m",(8,116),cv2.FONT_HERSHEY_SIMPLEX,fs,(255,255,100),2)

                caption=(f"area_m² + sh({sh_w:.2f}) +\n"
                         f"hip({hip_w:.2f}) + tor({torso:.2f}) + H({h_m:.2f})")
                return cv2.cvtColor(vis,cv2.COLOR_BGR2RGB),\
                       dict(sh=sh_w,hip=hip_w,torso=torso,h=h_m), caption
    except Exception as e:
        print(f"    [Phys] {e}")
    return cv2.cvtColor(vis,cv2.COLOR_BGR2RGB), {}, "detection failed"

def v9_pose(bgr):
    """Rich 24-D pose landmarks."""
    vis=bgr.copy()
    try:
        pts,_,_,_ = run_landmarker(bgr, seg=False)
        if pts is not None:
            for a,b in CONNS:
                cv2.line(vis,tuple(pts[a].astype(int)),tuple(pts[b].astype(int)),(0,220,100),3)
            for i,p in enumerate(pts):
                col=(86,166,255) if i in (11,12,23,24) else (200,200,200)
                cv2.circle(vis,tuple(p.astype(int)),5,col,-1)
    except Exception as e:
        print(f"    [Pose] {e}")
    return cv2.cvtColor(vis,cv2.COLOR_BGR2RGB), "24-D scale-invariant\nskeletal ratios"

def v9_deep_crop(seg_bgr, pts):
    """Pose-guided crop on segmented image → EfficientNet input."""
    vis=seg_bgr.copy()
    crop=seg_bgr
    if pts is not None:
        h,w=seg_bgr.shape[:2]
        xs,ys=pts[:,0],pts[:,1]
        bw,bh=xs.max()-xs.min(),ys.max()-ys.min()
        mg=0.20
        x1=max(0,int(xs.min()-bw*mg/2)); x2=min(w,int(xs.max()+bw*mg/2))
        y1=max(0,int(ys.min()-bh*mg/2)); y2=min(h,int(ys.max()+bh*mg/2))
        if x2>x1 and y2>y1:
            crop=seg_bgr[y1:y2,x1:x2]
            cv2.rectangle(vis,(x1,y1),(x2,y2),(245,100,80),4)
    return cv2.cvtColor(vis,cv2.COLOR_BGR2RGB), cv2.cvtColor(crop,cv2.COLOR_BGR2RGB),\
           "Pose-guided crop\nEfficientNet-B3 → 1536D"

def v9_feature_bar(dataset_type, phys_vals):
    """Horizontal stacked bar of final feature composition."""
    fig,ax=plt.subplots(figsize=(4.0,3.5),facecolor=PANEL)
    ax.set_facecolor(PANEL)

    segs=[
        ("Physical\n(5D)",  5,   "#3fb950"),
        ("Pose\n(24D)",     24,  "#58a6ff"),
        ("Deep CNN\n(1536D)",1536,"#e3b341"),
    ]
    total=sum(d for _,d,_ in segs)
    y=0
    for lbl,d,col in segs:
        frac=d/total
        bar=ax.barh(0,frac,left=y,color=col,height=0.55,edgecolor=BG,linewidth=2)
        if frac>0.02:
            ax.text(y+frac/2,0,lbl,ha="center",va="center",fontsize=9.5,
                    color="black",fontweight="bold",linespacing=1.3)
        y+=frac

    ax.set_xlim(0,1); ax.set_ylim(-0.6,0.6); ax.axis("off")
    ax.set_title(f"V9 Feature Vector\n[{total}D Total]",color=WHITE,
                 fontsize=11,pad=6,fontweight="bold",linespacing=1.4)

    if phys_vals:
        details=(f"Physical: area, shoulder({phys_vals.get('sh',0):.2f}m), "
                 f"hip({phys_vals.get('hip',0):.2f}m),\n"
                 f"torso({phys_vals.get('torso',0):.2f}m), H({phys_vals.get('h',0):.2f}m)\n"
                 f"Pose: 24 scale-invariant skeletal ratios\n"
                 f"Deep: EfficientNet-B3 (pretrained ImageNet)")
    else:
        details="5D Physical + 24D Pose + 1536D Deep CNN\n= 1565D V9 Hybrid Feature"

    ax.text(0.5,-0.55,details,ha="center",transform=ax.transAxes,
            fontsize=6.8,color=GREY,multialignment="center",linespacing=1.5)

    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    w2,h2=fig.canvas.get_width_height()
    buf=np.frombuffer(fig.canvas.buffer_rgba(),dtype=np.uint8).reshape(h2,w2,4)[:,:,:3]
    plt.close(fig)
    return buf, f"5D + 24D + 1536D = {total}D"

# ── Resize to fixed height ────────────────────────────────────────────────────
def to_h(img, h=380):
    oh,ow=img.shape[:2]; nw=int(ow*h/oh)
    return cv2.resize(img,(nw,h),interpolation=cv2.INTER_AREA)

# ── Run full V9 pipeline for one image ───────────────────────────────────────
def run_v9(bgr, h_m, w_kg, bmi, sex, age, dataset_type):
    panels=[]; captions=[]

    print("  [1] Original");    r,c=v9_original(bgr,h_m,w_kg,bmi,sex,age); panels.append(r); captions.append(c)
    print("  [2] YOLO Crop");   r,crop,box,c=v9_yolo(bgr);                  panels.append(r); captions.append(c)
    print("  [3] Segmentation");r,pts,mask,c=v9_segment(bgr);               panels.append(r); captions.append(c)

    # For phys and pose use original bgr (full image, known scale)
    print("  [4] Physical Dims");r,phys,c=v9_phys(bgr,h_m);                 panels.append(r); captions.append(c)
    print("  [5] Pose Features");r,c=v9_pose(bgr);                          panels.append(r); captions.append(c)

    # Deep crop uses segmented image
    seg_bgr=cv2.cvtColor(panels[2],cv2.COLOR_RGB2BGR)
    print("  [6] Deep Crop");   r,crop_r,c=v9_deep_crop(seg_bgr,pts);      panels.append(r); captions.append(c)
    print("  [7] Feature Vec"); r,c=v9_feature_bar(dataset_type,phys);      panels.append(r); captions.append(c)

    return panels, captions, phys

# ── Figure builder ────────────────────────────────────────────────────────────
def build(row_data):
    N_ROWS=len(row_data); N_COLS=len(V9_STAGES)
    IMG_H=350
    CW=2.7; AW=0.38; RH=4.2
    fw=N_COLS*CW+(N_COLS-1)*AW+1.2; fh=N_ROWS*RH+2.0
    fig=plt.figure(figsize=(fw,fh),facecolor=BG)

    fig.text(0.5,0.990,"V9 Hybrid Preprocessing Pipeline",
             ha="center",va="top",fontsize=16,fontweight="bold",color=WHITE)
    fig.text(0.5,0.968,
             "Physical Dimensions (5D)  +  Rich Pose Features (24D)  +  EfficientNet-B3 Deep Features (1536D)  =  1565D",
             ha="center",va="top",fontsize=9,color=GREY)

    from matplotlib.gridspec import GridSpec
    col_ratios=[]
    for i in range(N_COLS):
        col_ratios.append(CW)
        if i<N_COLS-1: col_ratios.append(AW)

    gs=GridSpec(N_ROWS,len(col_ratios),figure=fig,
                width_ratios=col_ratios,
                left=0.055,right=0.995,top=0.935,bottom=0.055,
                hspace=0.15,wspace=0.0)

    for ri,(ds_label,accent,panels,captions) in enumerate(row_data):
        # Row label
        fig.text(0.003,1.0-(ri+0.5)/N_ROWS*0.89,
                 ds_label,ha="left",va="center",rotation=90,
                 fontsize=12,fontweight="bold",color=accent)

        for ci,(key,col,title) in enumerate(V9_STAGES):
            gs_c=ci*2
            ax=fig.add_subplot(gs[ri,gs_c])
            ax.set_facecolor(PANEL)

            img=to_h(panels[ci],IMG_H)
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(col); sp.set_linewidth(2.8)

            # Title badge
            ax.set_title(title,fontsize=8.5,color=WHITE,pad=4,
                         multialignment="center",linespacing=1.35,
                         bbox=dict(boxstyle="round,pad=0.28",fc=col,alpha=0.92,ec="none"))

            # Caption below
            cap=captions[ci]
            ax.text(0.5,-0.03,cap,transform=ax.transAxes,
                    ha="center",va="top",fontsize=6.2,color=GREY,
                    multialignment="center",linespacing=1.4)

            # Arrow
            if ci<N_COLS-1:
                ax_a=fig.add_subplot(gs[ri,gs_c+1])
                ax_a.set_facecolor(BG); ax_a.axis("off")
                ax_a.annotate("",xy=(0.88,0.5),xytext=(0.12,0.5),
                              xycoords="axes fraction",
                              arrowprops=dict(arrowstyle="-|>",color="#c9d1d9",
                                              lw=2.2,mutation_scale=20))

    # Legend
    patches=[mpatches.Patch(color=s[1],label=s[2].replace("\n"," ")) for s in V9_STAGES]
    fig.legend(handles=patches,loc="lower center",ncol=N_COLS,fontsize=7.5,
               facecolor=PANEL,edgecolor="#30363d",labelcolor=WHITE,
               framealpha=0.9,bbox_to_anchor=(0.5,0.0))
    return fig

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print("="*58)
    print("  V9 Pipeline Flow  —  1 image per dataset")
    print("="*58)

    f2d    = "000304_M_21_167640_7711071 (1).jpg"
    fceleb = "1000_5.7h_70w_male_46a.jpg"

    print(f"\n2DImage2BMI : {f2d}")
    print(f"Celeb-FBI   : {fceleb}")

    bgr2d  =cv2.imread(os.path.join(DIR_2D,    f2d))
    bgr_cel=cv2.imread(os.path.join(DIR_CELEB, fceleb))

    h2d,w2d,bmi2d,sex2d,age2d=parse_2d(f2d)
    hcel,wcel,bmicel,sexcel,agecel=parse_celeb(fceleb)

    print("\n[2DImage2BMI] Running V9 pipeline...")
    p2d,c2d,phys2d=run_v9(bgr2d,h2d,w2d,bmi2d,sex2d,age2d,"2d")

    print("\n[Celeb-FBI]   Running V9 pipeline...")
    pcel,ccel,physcel=run_v9(bgr_cel,hcel,wcel,bmicel,sexcel,agecel,"celeb")

    row_data=[
        ("2DImage2BMI", ACC_2D,  p2d,  c2d),
        ("Celeb-FBI",   ACC_CEL, pcel, ccel),
    ]

    print("\n[BUILD] Composing figure...")
    fig=build(row_data)
    print(f"[SAVE] → {OUT_PATH}")
    fig.savefig(OUT_PATH,dpi=160,bbox_inches="tight",facecolor=BG)
    plt.close(fig)
    print("[DONE]")

if __name__=="__main__":
    main()
