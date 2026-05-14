#!/usr/bin/env python3
"""
diagnostic_dashboard.py
=======================
Publication-ready 2×3 diagnostic dashboard for a body-composition /
BMI-estimation pipeline.

Generates six subplots:
  1. Predicted vs. Actual scatter with regression line & metric inset
  2. CDF calibration plot (eCDF of actual vs. predicted)
  3. Error-distribution histogram + KDE + mean-error line
  4. Residuals vs. Fitted values
  5. Bland–Altman (Limits of Agreement) plot
  6. Box plot of |error| across WHO BMI categories

All plots follow a sophisticated, academic colour palette and use
LaTeX-formatted axis labels.  The figure is saved at 300 DPI.

Dependencies
------------
numpy, matplotlib, seaborn, scipy

Usage
-----
    python diagnostic_dashboard.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# 0.  GLOBAL STYLE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

# Academic colour palette
PALETTE = {
    "teal":       "#1B7A7D",
    "navy":       "#2C3E6B",
    "slate":      "#5A6B82",
    "muted_red":  "#C44E52",
    "muted_gold": "#D4A03C",
    "soft_green": "#4C956C",
    "light_grey": "#B0B8C1",
    "bg_white":   "#FAFBFC",
    "grid_grey":  "#E0E4EA",
}

# Sequential palette for the box plot
BOX_PALETTE = ["#89C2D9", "#61A5C2", "#468FAF", "#2C7DA0"]

plt.rcParams.update({
    "figure.facecolor":    PALETTE["bg_white"],
    "axes.facecolor":      PALETTE["bg_white"],
    "axes.edgecolor":      PALETTE["light_grey"],
    "axes.grid":           True,
    "grid.color":          PALETTE["grid_grey"],
    "grid.linewidth":      0.6,
    "grid.alpha":          0.7,
    "axes.labelsize":      11,
    "axes.titlesize":      12.5,
    "axes.titleweight":    "bold",
    "xtick.labelsize":     9.5,
    "ytick.labelsize":     9.5,
    "legend.fontsize":     9.5,
    "font.family":         "serif",
    "mathtext.fontset":    "cm",       # Computer Modern for LaTeX look
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
})


# ──────────────────────────────────────────────────────────────────────
# 1.  SYNTHETIC / MOCK DATA GENERATION
# ──────────────────────────────────────────────────────────────────────

def generate_mock_data(n: int = 500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (y_true, y_pred) arrays of shape (n,).

    y_true is drawn from a realistic BMI distribution (µ ≈ 24, σ ≈ 4)
    clipped to [16, 35].  y_pred adds Gaussian noise whose standard
    deviation grows slightly with BMI (heteroscedastic, as in reality).
    """
    rng = np.random.default_rng(seed)

    # Realistic BMI distribution – slightly right-skewed
    raw = rng.normal(loc=24.0, scale=4.0, size=n)
    y_true = np.clip(raw, 16.0, 35.0)

    # Heteroscedastic prediction noise
    noise_std = 0.8 + 0.06 * (y_true - 16.0)          # σ grows with BMI
    noise = rng.normal(0.0, noise_std)
    y_pred = y_true + noise + 0.15                      # tiny systematic bias

    return y_true, y_pred


# ──────────────────────────────────────────────────────────────────────
# 2.  HELPER UTILITIES
# ──────────────────────────────────────────────────────────────────────

def _ecdf(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted x and cumulative probability arrays for an eCDF."""
    xs = np.sort(data)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def _bmi_category(bmi: float) -> str:
    """Map a single BMI value to its WHO category string."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25.0:
        return "Normal"
    elif bmi < 30.0:
        return "Overweight"
    else:
        return "Obese"


# ──────────────────────────────────────────────────────────────────────
# 3.  INDIVIDUAL SUBPLOT FUNCTIONS
# ──────────────────────────────────────────────────────────────────────

def plot_predicted_vs_actual(ax: plt.Axes, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Subplot 1 – Predicted vs. Actual scatter with metrics inset."""

    ax.scatter(y_true, y_pred, s=22, alpha=0.55, color=PALETTE["teal"],
               edgecolors="white", linewidths=0.3, label="Samples", zorder=3)

    # Perfect-prediction diagonal
    lo, hi = min(y_true.min(), y_pred.min()) - 1, max(y_true.max(), y_pred.max()) + 1
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color=PALETTE["light_grey"],
            label=r"$y = x$", zorder=2)

    # OLS regression line
    slope, intercept, _, _, _ = stats.linregress(y_true, y_pred)
    x_fit = np.linspace(lo, hi, 200)
    ax.plot(x_fit, slope * x_fit + intercept, lw=1.8, color=PALETTE["muted_red"],
            label="Regression fit", zorder=4)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Actual BMI ($y$)")
    ax.set_ylabel(r"Predicted BMI ($\hat{y}$)")
    ax.set_title("Predicted vs. Actual")
    ax.legend(loc="upper left", frameon=True, framealpha=0.85, edgecolor=PALETTE["grid_grey"])

    # Metric text box
    r2   = stats.pearsonr(y_true, y_pred).statistic ** 2
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r_val = stats.pearsonr(y_true, y_pred).statistic

    text = (
        rf"$R^2  = {r2:.3f}$" "\n"
        rf"$\mathrm{{MAE}} = {mae:.2f}$" "\n"
        rf"$\mathrm{{RMSE}} = {rmse:.2f}$" "\n"
        rf"$r = {r_val:.3f}$"
    )
    props = dict(boxstyle="round,pad=0.45", facecolor="white",
                 edgecolor=PALETTE["grid_grey"], alpha=0.92)
    ax.text(0.97, 0.04, text, transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=props, family="serif")


def plot_cdf_calibration(ax: plt.Axes, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Subplot 2 – eCDF calibration (Pantanowitz-inspired)."""

    x_act, y_act = _ecdf(y_true)
    x_prd, y_prd = _ecdf(y_pred)

    ax.step(x_act, y_act, where="post", lw=1.8, color=PALETTE["navy"],
            label="Actual eCDF")
    ax.step(x_prd, y_prd, where="post", lw=1.8, color=PALETTE["muted_gold"],
            ls="--", label="Predicted eCDF")

    ax.set_xlabel(r"BMI value")
    ax.set_ylabel(r"Cumulative probability $F(x)$")
    ax.set_title("CDF Calibration")
    ax.legend(loc="lower right", frameon=True, framealpha=0.85,
              edgecolor=PALETTE["grid_grey"])

    # KS statistic annotation
    ks_stat, ks_p = stats.ks_2samp(y_true, y_pred)
    ax.text(0.03, 0.95,
            rf"KS $= {ks_stat:.3f}$" "\n" rf"$p = {ks_p:.3f}$",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=PALETTE["grid_grey"], alpha=0.92))


def plot_error_distribution(ax: plt.Axes, residuals: np.ndarray) -> None:
    """Subplot 3 – Residual histogram + KDE + mean-error line."""

    sns.histplot(residuals, kde=True, stat="density", bins=35,
                 color=PALETTE["teal"], edgecolor="white", linewidth=0.5,
                 alpha=0.65, ax=ax, line_kws=dict(lw=2))

    mean_err = residuals.mean()
    ax.axvline(mean_err, ls="--", lw=1.5, color=PALETTE["muted_red"],
               label=rf"Mean error $= {mean_err:+.2f}$")

    ax.set_xlabel(r"Residual ($y - \hat{y}$)")
    ax.set_ylabel("Density")
    ax.set_title("Error Distribution")
    ax.legend(loc="upper right", frameon=True, framealpha=0.85,
              edgecolor=PALETTE["grid_grey"])


def plot_residuals_vs_fitted(ax: plt.Axes, y_pred: np.ndarray, residuals: np.ndarray) -> None:
    """Subplot 4 – Residuals vs. Fitted values."""

    ax.scatter(y_pred, residuals, s=20, alpha=0.50, color=PALETTE["slate"],
               edgecolors="white", linewidths=0.3, zorder=3)
    ax.axhline(0, ls="--", lw=1.3, color=PALETTE["muted_red"], zorder=4)

    # LOWESS smoother for trend
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(residuals, y_pred, frac=0.35)
        ax.plot(smoothed[:, 0], smoothed[:, 1], lw=2, color=PALETTE["muted_gold"],
                label="LOWESS trend", zorder=5)
        ax.legend(loc="upper left", frameon=True, framealpha=0.85,
                  edgecolor=PALETTE["grid_grey"])
    except ImportError:
        pass  # gracefully skip if statsmodels not installed

    ax.set_xlabel(r"Fitted value ($\hat{y}$)")
    ax.set_ylabel(r"Residual ($y - \hat{y}$)")
    ax.set_title("Residuals vs. Fitted")


def plot_bland_altman(ax: plt.Axes, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Subplot 5 – Bland–Altman (Limits of Agreement) plot."""

    means = (y_true + y_pred) / 2.0
    diffs = y_true - y_pred

    mean_diff = diffs.mean()
    sd_diff   = diffs.std(ddof=1)
    upper_loa = mean_diff + 1.96 * sd_diff
    lower_loa = mean_diff - 1.96 * sd_diff

    ax.scatter(means, diffs, s=20, alpha=0.50, color=PALETTE["navy"],
               edgecolors="white", linewidths=0.3, zorder=3)

    # Horizontal reference lines
    ax.axhline(mean_diff, ls="-", lw=1.4, color=PALETTE["muted_red"], zorder=4)
    ax.axhline(upper_loa, ls="--", lw=1.2, color=PALETTE["slate"], zorder=4)
    ax.axhline(lower_loa, ls="--", lw=1.2, color=PALETTE["slate"], zorder=4)

    # Right-margin labels
    x_max = means.max()
    offset = (means.max() - means.min()) * 0.02

    ax.text(x_max + offset, mean_diff,
            rf"Bias $= {mean_diff:+.2f}$",
            va="center", fontsize=8.5, color=PALETTE["muted_red"], weight="bold")
    ax.text(x_max + offset, upper_loa,
            rf"$+1.96\,\mathrm{{SD}} = {upper_loa:+.2f}$",
            va="center", fontsize=8, color=PALETTE["slate"])
    ax.text(x_max + offset, lower_loa,
            rf"$-1.96\,\mathrm{{SD}} = {lower_loa:+.2f}$",
            va="center", fontsize=8, color=PALETTE["slate"])

    # Shade the limits-of-agreement band
    ax.fill_between([means.min() - 1, means.max() + 1],
                    lower_loa, upper_loa,
                    color=PALETTE["teal"], alpha=0.06, zorder=1)

    ax.set_xlabel(r"Mean of Actual and Predicted BMI $\left(\frac{y+\hat{y}}{2}\right)$")
    ax.set_ylabel(r"Difference $(y - \hat{y})$")
    ax.set_title("Bland–Altman Plot")


def plot_error_by_category(ax: plt.Axes, y_true: np.ndarray, abs_errors: np.ndarray) -> None:
    """Subplot 6 – Box plot of |error| across WHO BMI categories."""

    categories = np.array([_bmi_category(b) for b in y_true])
    order = ["Underweight", "Normal", "Overweight", "Obese"]

    # Build a dict for seaborn
    import pandas as pd
    df = pd.DataFrame({"BMI Category": categories, r"|Error|": abs_errors})

    sns.boxplot(
        data=df, x="BMI Category", y=r"|Error|", order=order,
        hue="BMI Category", hue_order=order, palette=BOX_PALETTE,
        linewidth=1.0, fliersize=3.5, legend=False,
        flierprops=dict(marker="o", markerfacecolor=PALETTE["slate"],
                        markeredgecolor="white", alpha=0.6),
        ax=ax, saturation=0.85,
    )

    # Overlay swarm/strip for small n categories
    sns.stripplot(
        data=df, x="BMI Category", y=r"|Error|", order=order,
        hue="BMI Category", hue_order=order, legend=False,
        palette=[PALETTE["navy"]] * 4, alpha=0.25, size=2.5,
        jitter=True, ax=ax, zorder=2,
    )

    ax.set_xlabel("BMI Category (WHO)")
    ax.set_ylabel(r"Absolute Error $|y - \hat{y}|$")
    ax.set_title("Error by BMI Category")


# ──────────────────────────────────────────────────────────────────────
# 4.  MAIN DASHBOARD ASSEMBLY
# ──────────────────────────────────────────────────────────────────────

def build_dashboard(y_true: np.ndarray, y_pred: np.ndarray,
                    save_path: str | Path | None = None) -> plt.Figure:
    """
    Assemble the 2×3 diagnostic dashboard and optionally save to disk.

    Parameters
    ----------
    y_true : ndarray   Ground-truth BMI values.
    y_pred : ndarray   Predicted BMI values.
    save_path : str or Path, optional
        If provided the figure is saved at 300 DPI.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    residuals  = y_true - y_pred
    abs_errors = np.abs(residuals)

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 10.5))
    fig.subplots_adjust(hspace=0.38, wspace=0.30)

    # Row 1
    plot_predicted_vs_actual(axes[0, 0], y_true, y_pred)
    plot_cdf_calibration(axes[0, 1], y_true, y_pred)
    plot_error_distribution(axes[0, 2], residuals)

    # Row 2
    plot_residuals_vs_fitted(axes[1, 0], y_pred, residuals)
    plot_bland_altman(axes[1, 1], y_true, y_pred)
    plot_error_by_category(axes[1, 2], y_true, abs_errors)

    # Super-title
    fig.suptitle(
        "BMI Estimation — Diagnostic Dashboard",
        fontsize=16, fontweight="bold", y=0.995,
        color=PALETTE["navy"],
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300, facecolor=fig.get_facecolor())
        print(f"[OK] Dashboard saved -> {save_path}  (300 DPI)")

    return fig


# ──────────────────────────────────────────────────────────────────────
# 5.  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Generate realistic synthetic data
    y_true, y_pred = generate_mock_data(n=500, seed=42)

    # Build and save the dashboard
    output_path = Path(__file__).resolve().parent.parent / "diagnostic_dashboard.png"
    fig = build_dashboard(y_true, y_pred, save_path=output_path)

    plt.show()
    print("[OK] Done.")
