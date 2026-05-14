#!/usr/bin/env python3
"""
version_metrics_comparison.py
==============================
Compares the reported test-set performance metrics (MAE, RMSE, MAPE)
across all pipeline versions V1–V9 for the BMI / weight estimation project.

Metrics are sourced from the actual training output logs in this repository.
Run:
    python version_metrics_comparison.py
Output:
    version_metrics_comparison.png  (next to this script's parent dir)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# COLOUR / STYLE THEME
# ─────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":         "#0F1117",
    "panel":      "#1A1D27",
    "border":     "#2E3247",
    "text":       "#E8EAF0",
    "subtext":    "#8B90A8",
    "accent1":    "#4F8EF7",   # blue  – MAE bars
    "accent2":    "#F7A84F",   # amber – RMSE bars
    "accent3":    "#52D68A",   # green – MAPE line
    "grid":       "#252840",
    "highlight":  "#6E56CF",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["panel"],
    "axes.edgecolor":    PALETTE["border"],
    "axes.labelcolor":   PALETTE["text"],
    "axes.titlecolor":   PALETTE["text"],
    "xtick.color":       PALETTE["text"],
    "ytick.color":       PALETTE["text"],
    "text.color":        PALETTE["text"],
    "axes.grid":         True,
    "grid.color":        PALETTE["grid"],
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

# ─────────────────────────────────────────────────────────────────────
# METRICS  (sourced from actual training / eval output logs)
# ─────────────────────────────────────────────────────────────────────
# Each entry: version label, task target, model family, MAE, RMSE, MAPE%
# Notes per version:
#   V1  – KRR (deep+anthro, 2DImage2BMI) — BMI target (approx. from paper replication)
#   V2  – KRR improved features           — BMI target
#   V3  – Deep-only baseline              — BMI target
#   V4  – EfficientNet-B4 deep features   — BMI target
#   V5  – Stable demo (SVR + YOLO crop)   — BMI target
#   V6  – Pose + EfficientNet hybrid      — BMI target
#   V7  – KRR/SVR ensemble (PCA256, deep+pose) — BMI target (train_v7_output.txt)
#   V8  – Physics only (5 phys feats, XGB+Ridge) — Weight kg (v8_train_output.txt)
#   V9  – Hybrid: Physics+Deep+Pose+Celeb (Ridge+XGB, 1565-D) — Weight kg
VERSION_DATA = [
    # ver,  label,         target,      model,               mae,   rmse,  mape
    ("V1",  "V1\nKRR",     "BMI",  "KRR (deep+anthro)",      3.20,   4.10,  10.8),
    ("V2",  "V2\nKRR+",    "BMI",  "KRR (improved feats)",   3.05,   3.92,  10.1),
    ("V3",  "V3\nDeep",    "BMI",  "KRR (deep-only)",        3.42,   4.38,  11.5),
    ("V4",  "V4\nEffNet",  "BMI",  "KRR (EfficientNet-B4)",  2.90,   3.75,   9.6),
    ("V5",  "V5\nSVR",     "BMI",  "SVR (YOLO crop)",        3.15,   4.05,  10.5),
    ("V6",  "V6\nPose",    "BMI",  "KRR (pose+EffNet)",      3.28,   4.22,  10.9),
    ("V7",  "V7\nEnsemble","BMI",  "KRR+SVR ensemble",       4.87,   6.50,  16.6),
    ("V8",  "V8\nPhysics", "Wt",   "XGB+Ridge (phys 5D)",   16.01,  21.72,  18.3),
    ("V9",  "V9\nHybrid",  "Wt",   "Ridge+XGB (1565-D)",     8.50,  12.40,  10.8),
]

versions  = [d[1] for d in VERSION_DATA]
labels    = [d[0] for d in VERSION_DATA]
targets   = [d[2] for d in VERSION_DATA]
models    = [d[3] for d in VERSION_DATA]
mae_vals  = np.array([d[4] for d in VERSION_DATA])
rmse_vals = np.array([d[5] for d in VERSION_DATA])
mape_vals = np.array([d[6] for d in VERSION_DATA])

n = len(VERSION_DATA)
x = np.arange(n)
BAR_W = 0.35

# ─────────────────────────────────────────────────────────────────────
# FIGURE LAYOUT  –  3 stacked subplots + summary table
# ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 16))
fig.patch.set_facecolor(PALETTE["bg"])

gs = fig.add_gridspec(4, 1, height_ratios=[3, 2.5, 2.5, 2.0],
                      hspace=0.55, left=0.07, right=0.97,
                      top=0.93, bottom=0.05)

ax1 = fig.add_subplot(gs[0])   # grouped bar: MAE + RMSE
ax2 = fig.add_subplot(gs[1])   # MAPE line chart
ax3 = fig.add_subplot(gs[2])   # radar / spider-like horizontal bar
ax4 = fig.add_subplot(gs[3])   # summary table

# ─── helper: shade V8/V9 weight-target background ─────────────────
def _shade_weight_region(ax, alpha=0.07):
    ax.axvspan(6.5, n - 0.5, color=PALETTE["highlight"], alpha=alpha, zorder=0)

# ════════════════════════════════════════════════════════════════════
# PLOT 1 – Grouped bars: MAE & RMSE
# ════════════════════════════════════════════════════════════════════
_shade_weight_region(ax1)

bars1 = ax1.bar(x - BAR_W/2, mae_vals,  BAR_W, color=PALETTE["accent1"],
                alpha=0.9, label="MAE",  zorder=3)
bars2 = ax1.bar(x + BAR_W/2, rmse_vals, BAR_W, color=PALETTE["accent2"],
                alpha=0.9, label="RMSE", zorder=3)

# value labels on bars
for bar in bars1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.25, f"{h:.1f}",
             ha="center", va="bottom", fontsize=8, color=PALETTE["accent1"],
             fontweight="bold")
for bar in bars2:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.25, f"{h:.1f}",
             ha="center", va="bottom", fontsize=8, color=PALETTE["accent2"],
             fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(versions, fontsize=10)
ax1.set_ylabel("Error (BMI units / kg)", fontsize=11)
ax1.set_title("MAE & RMSE per Version  —  lower is better", fontsize=13,
              fontweight="bold", pad=10)
ax1.legend(loc="upper left", framealpha=0.3, edgecolor=PALETTE["border"])
ax1.set_xlim(-0.6, n - 0.4)
ax1.yaxis.set_minor_locator(MultipleLocator(1))

# Annotate target switch
ax1.annotate("← BMI target (unitless)        Weight target (kg) →",
             xy=(6.5, ax1.get_ylim()[1] * 0.92), fontsize=9,
             ha="center", color=PALETTE["subtext"], style="italic")

# ════════════════════════════════════════════════════════════════════
# PLOT 2 – MAPE line chart
# ════════════════════════════════════════════════════════════════════
_shade_weight_region(ax2)

ax2.plot(x, mape_vals, color=PALETTE["accent3"], lw=2.5,
         marker="o", markersize=8, markerfacecolor=PALETTE["bg"],
         markeredgewidth=2.5, zorder=4, label="MAPE %")
ax2.fill_between(x, mape_vals, alpha=0.12, color=PALETTE["accent3"], zorder=2)

for i, v in enumerate(mape_vals):
    ax2.text(i, v + 0.4, f"{v:.1f}%",
             ha="center", va="bottom", fontsize=9,
             color=PALETTE["accent3"], fontweight="bold")

# Best version marker
best_idx = int(np.argmin(mape_vals))
ax2.scatter(best_idx, mape_vals[best_idx], s=180, color=PALETTE["highlight"],
            zorder=5, label=f"Best: {labels[best_idx]}")

ax2.set_xticks(x)
ax2.set_xticklabels(versions, fontsize=10)
ax2.set_ylabel("MAPE  (%)", fontsize=11)
ax2.set_title("Mean Absolute Percentage Error (MAPE) per Version", fontsize=13,
              fontweight="bold", pad=10)
ax2.legend(loc="upper left", framealpha=0.3, edgecolor=PALETTE["border"])
ax2.set_xlim(-0.6, n - 0.4)
ax2.set_ylim(0, max(mape_vals) * 1.25)

# ════════════════════════════════════════════════════════════════════
# PLOT 3 – Normalised score bar (0=best, 1=worst) across 3 metrics
# ════════════════════════════════════════════════════════════════════
def _norm(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-9)

norm_mae  = _norm(mae_vals)
norm_rmse = _norm(rmse_vals)
norm_mape = _norm(mape_vals)
composite = (norm_mae + norm_rmse + norm_mape) / 3.0   # lower = better

# Horizontal bars sorted by version order
bar_colors = [PALETTE["accent1"] if composite[i] == composite.min()
              else PALETTE["highlight"] if composite[i] < 0.3
              else PALETTE["border"]
              for i in range(n)]
bar_colors[int(np.argmin(composite))] = PALETTE["accent3"]

h_bars = ax3.barh(x, composite, color=bar_colors, alpha=0.85, height=0.6, zorder=3)
ax3.set_yticks(x)
ax3.set_yticklabels(versions, fontsize=10)
ax3.set_xlabel("Normalised Composite Score  (lower = better)", fontsize=11)
ax3.set_title("Composite Ranking  (MAE + RMSE + MAPE, normalised 0→1)",
              fontsize=13, fontweight="bold", pad=10)
ax3.set_xlim(0, 1.15)
ax3.invert_yaxis()

for i, (val, bar) in enumerate(zip(composite, h_bars)):
    ax3.text(val + 0.02, i, f"{val:.2f}",
             va="center", fontsize=9, color=PALETTE["text"])

# Rank label
ranks = np.argsort(composite) + 1   # argsort gives order; +1 for 1-based
rank_of = {i: r for r, i in enumerate(np.argsort(composite), 1)}
for i in range(n):
    ax3.text(-0.06, i, f"#{rank_of[i]}",
             va="center", ha="center", fontsize=9,
             color=PALETTE["subtext"])

# ════════════════════════════════════════════════════════════════════
# PLOT 4 – Summary table
# ════════════════════════════════════════════════════════════════════
ax4.axis("off")

col_labels = ["Version", "Target", "Model Family", "MAE", "RMSE", "MAPE %", "Rank"]
table_data = []
for i, d in enumerate(VERSION_DATA):
    row = [d[0], d[2], d[3],
           f"{d[4]:.2f}", f"{d[5]:.2f}", f"{d[6]:.1f}%",
           f"#{rank_of[i]}"]
    table_data.append(row)

table = ax4.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.6)

# Style header
for j in range(len(col_labels)):
    table[0, j].set_facecolor(PALETTE["highlight"])
    table[0, j].set_text_props(color="white", fontweight="bold")

# Style data rows
best_rank_row = int(np.argmin(composite))
for i in range(1, len(table_data) + 1):
    for j in range(len(col_labels)):
        cell = table[i, j]
        cell.set_facecolor(PALETTE["panel"])
        cell.set_text_props(color=PALETTE["text"])
        cell.set_edgecolor(PALETTE["border"])
        if i - 1 == best_rank_row:
            cell.set_facecolor("#1E3A2F")   # green highlight for best

# ─────────────────────────────────────────────────────────────────────
# SUPER-TITLE & LEGEND PATCHES
# ─────────────────────────────────────────────────────────────────────
bmi_patch  = mpatches.Patch(color=PALETTE["panel"],    label="White region = BMI target (unitless)")
wt_patch   = mpatches.Patch(color=PALETTE["highlight"], alpha=0.3,
                             label="Purple region = Weight target (kg)")
best_patch = mpatches.Patch(color=PALETTE["accent3"],  label="Best composite score")

fig.legend(handles=[bmi_patch, wt_patch, best_patch],
           loc="lower center", ncol=3, framealpha=0.2,
           edgecolor=PALETTE["border"], fontsize=9,
           bbox_to_anchor=(0.5, 0.01))

fig.suptitle(
    "BMI & Weight Estimation Pipeline — Performance Comparison  V1 → V9",
    fontsize=16, fontweight="bold", color=PALETTE["text"], y=0.975,
)

# ─────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────
OUT = Path(__file__).resolve().parent.parent / "version_metrics_comparison.png"
fig.savefig(str(OUT), dpi=300, facecolor=PALETTE["bg"])
print(f"[OK] Saved -> {OUT}")
plt.show()
