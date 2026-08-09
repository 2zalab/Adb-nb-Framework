"""Shared utilities for the ABD-NB experiments: figure style, palette,
calibration metrics and result I/O."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIGURES = os.path.join(HERE, "..", "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# ---------------------------------------------------------------------------
# Palette (validated categorical order; light-surface variant)
# ---------------------------------------------------------------------------
PALETTE = {
    "blue": "#2a78d6",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
    "magenta": "#e87ba4",
    "orange": "#eb6834",
}
SERIES = list(PALETTE.values())

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SURFACE = "#ffffff"

# fixed colour per model family (colour follows the entity)
MODEL_COLORS = {
    "GNB": PALETTE["yellow"],
    "MI-WNB": PALETTE["magenta"],
    "ABD-NB-L": PALETTE["aqua"],
    "ABD-NB-G": PALETTE["violet"],
    "ABD-NB-C": PALETTE["blue"],
    "LR": PALETTE["orange"],
    "kNN": PALETTE["green"],
    "CART": "#8a7350",
    "RF": PALETTE["red"],
    "SVM": "#5b8ba8",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "axes.edgecolor": BASE,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK2,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "figure.dpi": 200,
            "pdf.fonttype": 42,
        }
    )


def despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def new_fig(width=3.45, height=2.3, ncols=1, nrows=1, **kw):
    set_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), **kw)
    for ax in np.atleast_1d(axes).ravel():
        despine(ax)
    return fig, axes


def save_fig(fig, name: str) -> str:
    path = os.path.join(FIGURES, name if name.endswith(".pdf") else name + ".pdf")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  [fig] {os.path.basename(path)}")
    return path


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(y_true, proba, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def reliability_curve(y_true, proba, n_bins: int = 10):
    """Bin-wise (confidence, accuracy, count) for reliability diagrams."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() > 0:
            rows.append((conf[mask].mean(), correct[mask].mean(), int(mask.sum())))
    return np.array(rows)


def save_csv(df: pd.DataFrame, name: str) -> str:
    path = os.path.join(RESULTS, name if name.endswith(".csv") else name + ".csv")
    df.to_csv(path, index=False)
    print(f"  [csv] {os.path.basename(path)}")
    return path


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS, name if name.endswith(".csv") else name + ".csv"))
