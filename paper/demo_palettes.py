"""Generate example figures to preview project palettes."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from palettes import get_palette

OUTDIR = Path(__file__).resolve().parent / "figures" / "demo"
os.makedirs(OUTDIR, exist_ok=True)

import numpy as np
np.random.seed(42)


def demo_bar() -> None:
    """Bar chart with Nature Cancer palette."""
    categories = ["A", "B", "C", "D", "E"]
    values = np.array([23, 45, 56, 78, 32])
    colors = get_palette("nature_cancer", n=len(categories))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(categories, values, color=colors, edgecolor="white", linewidth=1)
    ax.set_title("Nature Cancer Palette - Bar Chart", fontsize=12, weight="bold")
    ax.set_ylabel("Value")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(OUTDIR / "demo_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def demo_line() -> None:
    """Multi-series line chart with Nature NPG palette."""
    x = np.linspace(0, 10, 50)
    colors = get_palette("nature_npg", n=5)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i in range(5):
        y = np.sin(x + i * 0.5) + np.random.normal(0, 0.05, size=x.shape)
        ax.plot(x, y, color=colors[i], linewidth=2, label=f"Series {i + 1}")
    ax.set_title("Nature NPG Palette - Line Chart", fontsize=12, weight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(OUTDIR / "demo_line.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def demo_scatter() -> None:
    """Scatter plot with Okabe-Ito palette."""
    colors = get_palette("okabe_ito", n=4)
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, col in enumerate(colors):
        n = 60
        x = np.random.normal(loc=i * 2, scale=0.6, size=n)
        y = np.random.normal(loc=i * 2, scale=0.6, size=n)
        ax.scatter(x, y, c=col, s=40, alpha=0.7, edgecolors="white", linewidth=0.5, label=f"Group {i + 1}")
    ax.set_title("Okabe-Ito Palette - Scatter Plot", fontsize=12, weight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(OUTDIR / "demo_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def demo_box() -> None:
    """Box plot with Cell Press palette."""
    colors = get_palette("cell_press", n=5)
    data = [np.random.normal(loc=i, scale=0.6, size=100) for i in range(5)]
    labels = [f"Cond {i + 1}" for i in range(5)]

    fig, ax = plt.subplots(figsize=(7, 4))
    bplot = ax.boxplot(data, patch_artist=True, tick_labels=labels, showfliers=False)
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("white")
        patch.set_linewidth(1.5)
    for whisker in bplot["whiskers"]:
        whisker.set(color="gray", linewidth=1)
    ax.set_title("Cell Press Palette - Box Plot", fontsize=12, weight="bold")
    ax.set_ylabel("Measurement")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(OUTDIR / "demo_box.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def demo_pie_donut() -> None:
    """Donut chart with Lancet palette."""
    colors = get_palette("lancet", n=5)
    values = np.array([25, 20, 30, 15, 10])
    labels = [f"Class {i + 1}" for i in range(5)]

    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops=dict(width=0.4, edgecolor="white"),
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_weight("bold")
    ax.set_title("Lancet Palette - Donut Chart", fontsize=12, weight="bold")
    fig.savefig(OUTDIR / "demo_donut.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    demo_bar()
    demo_line()
    demo_scatter()
    demo_box()
    demo_pie_donut()
    print(f"[DONE] Demo figures saved to {OUTDIR}")
