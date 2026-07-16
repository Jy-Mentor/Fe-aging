"""
medical-mechanism-diagram 功能验证脚本
目的：验证 Python 引擎生成 BCP → CB2 → Nrf2 → 神经保护 医学机制图
数据源：项目核心假说（标书 v13）+ L1-L4 真实分析结果
输出：figures/skill_test/BCP_mechanism_test.png/pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle
import numpy as np
import os

OUTDIR = r"d:/铁衰老 绝不重蹈覆辙/figures/skill_test"
os.makedirs(OUTDIR, exist_ok=True)


def node(ax, x, y, text, color, w=1.6, h=0.7, fontsize=9):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.08",
        linewidth=1.2, edgecolor="black", facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold")


def arrow(ax, x1, y1, x2, y2, style="-|>", color="black", lw=1.5, rad=0.0):
    cs = f"arc3,rad={rad}" if rad else "arc3"
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=18,
        color=color, linewidth=lw, connectionstyle=cs,
    )
    ax.add_patch(a)


def inhibit(ax, x1, y1, x2, y2, color="black", lw=1.5, rad=0.0):
    """Blunt-end inhibition arrow (T-bar at target)."""
    cs = f"arc3,rad={rad}" if rad else "arc3"
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-[", mutation_scale=22,
        color=color, linewidth=lw, connectionstyle=cs,
    )
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Title
    ax.text(5.5, 8.6, "BCP protective mechanism in CIRI ferroaging",
            ha="center", va="center", fontsize=11, fontweight="bold")

    # Compartment band (top: drug/receptor; middle: signaling; bottom: outcomes)
    ax.add_patch(Rectangle((0.2, 6.5), 10.6, 1.5, facecolor="#E8F4FD",
                            edgecolor="none", alpha=0.3))
    ax.text(0.4, 7.7, "Membrane", fontsize=7, style="italic", color="grey")
    ax.add_patch(Rectangle((0.2, 3.5), 10.6, 2.8, facecolor="#FDF2E9",
                            edgecolor="none", alpha=0.3))
    ax.text(0.4, 6.1, "Cytosol", fontsize=7, style="italic", color="grey")
    ax.add_patch(Rectangle((0.2, 0.3), 10.6, 2.8, facecolor="#FDEDEC",
                            edgecolor="none", alpha=0.3))
    ax.text(0.4, 2.8, "Outcomes", fontsize=7, style="italic", color="grey")

    # Tier 1: drug -> receptor -> signaling
    node(ax, 1.5, 7.2, "BCP\n(β-caryophyllene)", "#3498DB", w=2.0, h=0.9)
    node(ax, 4.0, 7.2, "CB2 receptor", "#8E44AD", w=1.7, h=0.7)
    node(ax, 6.5, 7.2, "PI3K / Akt", "#F39C12", w=1.6, h=0.7)
    node(ax, 9.0, 7.2, "Nrf2 (liberated\nfrom Keap1)", "#16A085", w=1.9, h=0.9)

    # Tier 2: Nrf2 translocation + downstream
    node(ax, 5.5, 4.8, "HO-1 / NQO1\nFTH1 (ferritin)\nGPX4 / SLC7A11",
         "#27AE60", w=2.6, h=1.2)

    # Tier 3: outcomes
    node(ax, 1.5, 1.5, "Iron (Fe2+) ↓\nsequestration", "#7D6608", w=1.9, h=0.9)
    node(ax, 3.8, 1.5, "ROS ↓", "#E74C3C", w=1.4, h=0.7)
    node(ax, 6.0, 1.5, "Lipid peroxidation ↓\n(ACSL4 / LPCAT3)",
         "#E67E22", w=2.4, h=0.9)
    node(ax, 9.0, 1.5, "Neuroprotection\nin CIRI", "#16A085", w=2.0, h=0.9)

    # Arrows tier 1
    arrow(ax, 2.5, 7.2, 3.15, 7.2, color="#3498DB", lw=1.8)
    arrow(ax, 4.85, 7.2, 5.7, 7.2, color="#8E44AD", lw=1.8)
    arrow(ax, 7.3, 7.2, 8.05, 7.2, color="#F39C12", lw=1.8)

    # Nrf2 -> downstream (translocation, thick)
    arrow(ax, 8.8, 6.75, 6.2, 5.45, rad=0.15, color="#16A085", lw=2.2)

    # Downstream -> outcomes
    arrow(ax, 4.7, 4.4, 1.9, 2.0, rad=-0.15, color="#27AE60", lw=1.5)
    arrow(ax, 5.2, 4.2, 3.8, 1.9, rad=-0.05, color="#27AE60", lw=1.5)
    arrow(ax, 5.8, 4.2, 6.0, 2.0, rad=0.05, color="#27AE60", lw=1.5)

    # Outcome cascade
    arrow(ax, 2.4, 1.5, 3.1, 1.5, color="#7D6608", lw=1.3)
    arrow(ax, 4.5, 1.5, 4.8, 1.5, color="#E74C3C", lw=1.3)
    arrow(ax, 7.2, 1.5, 8.05, 1.5, color="#E67E22", lw=1.5)

    # Microglia polarization (off to the side)
    node(ax, 9.8, 4.8, "Microglia\nM1 → M2 shift", "#9B59B6", w=1.8, h=0.9)
    arrow(ax, 6.8, 4.8, 8.9, 4.8, color="#9B59B6", lw=1.3, rad=0.0)
    arrow(ax, 9.8, 4.35, 9.5, 2.0, rad=-0.1, color="#9B59B6", lw=1.3)

    # Pathology inhibition arrows (dashed, showing what is REDUCED)
    ax.annotate("CIRI\ninjury", xy=(0.7, 4.8), xytext=(1.5, 4.8),
                fontsize=8, ha="center", va="center", color="#C0392B",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#FADBD8",
                          edgecolor="#C0392B", lw=0.8))
    inhibit(ax, 2.0, 4.8, 3.0, 4.8, color="#C0392B", lw=1.5)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color="black", lw=1.5, label="Activation"),
        plt.Line2D([0], [0], color="black", lw=1.5, label="Inhibition (blunt)"),
        plt.Line2D([0], [0], color="#16A085", lw=2.2, label="Translocation"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              fontsize=7, frameon=True, edgecolor="grey")

    plt.tight_layout()

    png_path = os.path.join(OUTDIR, "BCP_mechanism_test.png")
    pdf_path = os.path.join(OUTDIR, "BCP_mechanism_test.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] PNG saved: {png_path}")
    print(f"[OK] PDF saved: {pdf_path}")
    print("[DONE] medical-mechanism-diagram test passed")


if __name__ == "__main__":
    main()
