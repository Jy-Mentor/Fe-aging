"""Verify project palettes load correctly and render a preview figure."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from palettes import get_palette, load_palettes

OUTDIR = Path(__file__).resolve().parent / "figures"
os.makedirs(OUTDIR, exist_ok=True)


def plot_palette_preview() -> None:
    """Render all palettes as horizontal color bars."""
    pals = load_palettes()
    names = [k for k in pals.keys() if not k.startswith("_")]
    n_pals = len(names)

    fig, ax = plt.subplots(figsize=(10, 0.6 * n_pals + 1))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n_pals)
    ax.axis("off")

    for i, name in enumerate(names):
        colors = get_palette(name)
        n_cols = len(colors)
        y = n_pals - i - 1
        for j, col in enumerate(colors):
            x_start = j * (10 / n_cols)
            width = 10 / n_cols
            ax.add_patch(plt.Rectangle((x_start, y), width, 0.8, color=col, ec="white"))
        ax.text(-0.2, y + 0.4, name, ha="right", va="center", fontsize=9)

    ax.set_title("Project Publication Palette Preview", fontsize=12, weight="bold", pad=10)
    png_path = OUTDIR / "palette_preview.png"
    pdf_path = OUTDIR / "palette_preview.pdf"
    plt.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] PNG: {png_path}")
    print(f"[OK] PDF: {pdf_path}")


if __name__ == "__main__":
    print("Loading palettes...")
    names = sorted(k for k in load_palettes().keys() if not k.startswith("_"))
    print("Available:", ", ".join(names))
    for name in names:
        cols = get_palette(name)
        print(f"  {name}: {len(cols)} colors")
    plot_palette_preview()
    print("[DONE] All palettes verified.")
