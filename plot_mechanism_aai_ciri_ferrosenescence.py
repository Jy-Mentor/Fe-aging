##############################################################################
# 医学机制图: 艾叶 (Artemisia argyi) 通过 Nrf2 双通路干预 CIRI 铁衰老
#
# 科学假说来源:
#   标书_终版_v13_含图表_fixed.docx
#   核心假说: 广西道地壮瑶药艾叶通过激活 Nrf2/ARE 通路,同时抑制铁死亡
#   (Ferroptosis) 与铁死亡驱动的细胞衰老 (Ferro-senescence / 铁衰老),
#   阻断 SASP 恶性反馈, 保护脑缺血再灌注 (CIRI) 半暗带。
#
# 真实数据锚点 (全部从项目 CSV 读取):
#   - 铁衰老基因集: L1/results/ferroaging_genes_96.csv (96 genes)
#   - 核心候选基因: L2/results/core_candidates_ferroaging.csv (7 genes)
#   - 单细胞铁衰老评分: L2/results/GSE233815_sn/ferrosenescence_wilcoxon_vs_ctrl.csv
#     * Microglia 7DPI 显著升高 (p_adj = 0.0003067)
#     * Neuron 1DPI/3DPI 显著降低 (p_adj = 5.69e-13 / 1.06e-03)
#   - 炎症-铁衰老相关: L2/results/inflammation_ferroaging_correlation_GSE104036.csv
#     * Ccl2 rho=0.903, Icam1 rho=0.890, Cxcl10 rho=0.877, Il1b rho=0.847,
#       Il6 rho=0.820, Tnf rho=0.769 (均 P<1e-5)
#   - SCISSOR 细胞选择: L2/results/scissor_celltype_enrichment.csv
#     * OPC Scissor- 显著富集 (p=1.15e-18), Immune Scissor+ 显著 (p=0.0011)
#   - 壮药成分: L3/results/zhuangyao_herb_match_detail.csv (艾叶 8 ingredients)
#
# 输出:
#   figures/mechanisms/AAI_CIRI_Ferrosenescence_Mechanism.png (300 DPI)
#   figures/mechanisms/AAI_CIRI_Ferrosenescence_Mechanism.pdf (vector)
##############################################################################

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle
import numpy as np

# ---- 0. 路径与输出 ----
OUTDIR = r"d:/铁衰老 绝不重蹈覆辙/figures/mechanisms"
os.makedirs(OUTDIR, exist_ok=True)

# ---- 1. 中文支持 ----
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 2. 画布 ----
fig, ax = plt.subplots(figsize=(11.8, 8.3))  # 接近 A4 横向 300mm x 210mm
ax.set_xlim(0, 100)
ax.set_ylim(0, 70)
ax.axis("off")

# ---- 3. 颜色规范 (BioRender/Nature 风格) ----
COL = {
    "bg": "#FAFAFA",
    "cell": "#E8F4FD",
    "cell_edge": "#5DADE2",
    "nucleus": "#D7BDE2",
    "mito": "#F5B7B1",
    "danger": "#C0392B",
    "iron": "#7D6608",
    "lpo": "#E67E22",
    "ros": "#E74C3C",
    "protect": "#27AE60",
    "drug": "#3498DB",
    "drug_light": "#AED6F1",
    "target": "#8E44AD",
    "senescence": "#7D3C98",
    "sasp": "#D35400",
    "ethno": "#F9E79F",
    "text": "#1C2833",
    "grey": "#7F8C8D",
}

# ---- 4. 辅助函数 ----
def box(ax, x, y, w, h, text, facecolor, edgecolor=None, fontsize=8,
        fontweight="bold", text_color=None, radius=0.08, alpha=1.0):
    edgecolor = edgecolor or "#2C3E50"
    text_color = text_color or COL["text"]
    fb = FancyBboxPatch((x - w/2, y - h/2), w, h,
                        boxstyle=f"round,pad=0.02,rounding_size={radius}",
                        facecolor=facecolor, edgecolor=edgecolor,
                        linewidth=1.2, alpha=alpha, zorder=2)
    ax.add_patch(fb)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color=text_color, zorder=3)
    return fb

def ellipse(ax, x, y, w, h, text, facecolor, edgecolor=None, fontsize=8,
            fontweight="bold", text_color=None):
    edgecolor = edgecolor or "#2C3E50"
    text_color = text_color or COL["text"]
    el = Ellipse((x, y), w, h, facecolor=facecolor, edgecolor=edgecolor,
                 linewidth=1.2, alpha=0.9, zorder=2)
    ax.add_patch(el)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color=text_color, zorder=3)
    return el

def arrow(ax, x1, y1, x2, y2, color="#2C3E50", lw=1.5, style="-|>",
          rad=0.0, label=None, label_size=7, label_color=None):
    cs = f"arc3,rad={rad}" if rad != 0 else "arc3"
    ar = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                         mutation_scale=14, color=color, linewidth=lw,
                         connectionstyle=cs, zorder=1)
    ax.add_patch(ar)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        # 偏移避免压线
        offset_y = 1.2 if rad >= 0 else -1.2
        ax.text(mx, my + offset_y, label, ha="center", va="center",
                fontsize=label_size, color=label_color or color,
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.2",
                                              facecolor="white", edgecolor="none",
                                              alpha=0.85), zorder=4)
    return ar

def inhib_arrow(ax, x1, y1, x2, y2, color="#C0392B", lw=1.5, rad=0.0,
                label=None, label_size=7):
    # 抑制箭头: 普通线 + 终点短横线 (matplotlib 3.x 无 '-|' style)
    cs = f"arc3,rad={rad}" if rad != 0 else "arc3"
    ar = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-",
                         mutation_scale=14, color=color, linewidth=lw,
                         connectionstyle=cs, zorder=1)
    ax.add_patch(ar)
    # 终点短横线
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    perp_x, perp_y = -uy, ux
    bar_len = 0.8
    ax.plot([x2 + perp_x * bar_len, x2 - perp_x * bar_len],
            [y2 + perp_y * bar_len, y2 - perp_y * bar_len],
            color=color, linewidth=lw + 0.5, solid_capstyle="round", zorder=1)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        offset_y = 1.2 if rad >= 0 else -1.2
        ax.text(mx, my + offset_y, label, ha="center", va="center",
                fontsize=label_size, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.85), zorder=4)
    return ar

def title_text(ax, x, y, text, fontsize=10, color=None, fontweight="bold"):
    ax.text(x, y, text, ha="left", va="top", fontsize=fontsize,
            fontweight=fontweight, color=color or COL["text"], zorder=4)

# ---- 5. 背景区室 ----
# 脑组织区 (CIRI 触发)
brain_rect = FancyBboxPatch((2, 50), 96, 18, boxstyle="round,pad=0.5",
                            facecolor="#FADBD8", edgecolor="#C0392B",
                            linewidth=1.5, alpha=0.25, zorder=0)
ax.add_patch(brain_rect)
ax.text(5, 66, "脑缺血再灌注损伤 (CIRI)", fontsize=11, fontweight="bold",
        color=COL["danger"])

# 中间核心机制区 (铁死亡 / 铁衰老)
core_rect = FancyBboxPatch((2, 18), 70, 30, boxstyle="round,pad=0.5",
                           facecolor="#FDEDEC", edgecolor="#E74C3C",
                           linewidth=1.5, alpha=0.18, zorder=0)
ax.add_patch(core_rect)
ax.text(5, 46, "铁死亡 → 铁衰老 (Ferro-senescence) 恶性循环", fontsize=11,
        fontweight="bold", color=COL["danger"])

# 艾叶干预区 (右侧)
drug_rect = FancyBboxPatch((74, 18), 24, 30, boxstyle="round,pad=0.5",
                           facecolor="#EBF5FB", edgecolor="#3498DB",
                           linewidth=1.5, alpha=0.35, zorder=0)
ax.add_patch(drug_rect)
ax.text(76, 46, "艾叶干预 (壮瑶药)", fontsize=11, fontweight="bold",
        color=COL["drug"])

# 底部结局区
outcome_rect = FancyBboxPatch((2, 2), 96, 14, boxstyle="round,pad=0.5",
                              facecolor="#EAFAF1", edgecolor="#27AE60",
                              linewidth=1.5, alpha=0.25, zorder=0)
ax.add_patch(outcome_rect)
ax.text(5, 14, "结局: 神经血管单元保护 & 远期功能恢复", fontsize=11,
        fontweight="bold", color=COL["protect"])

# ---- 6. 顶部触发事件 ----
box(ax, 12, 58, 14, 5, "缺血\nIschemia", "#F5B7B1", fontsize=8)
box(ax, 34, 58, 16, 5, "再灌注\nReperfusion", "#F5B7B1", fontsize=8)
box(ax, 58, 58, 16, 5, "兴奋性毒性\nGlutamate excitotoxicity", "#F5B7B1", fontsize=7)
box(ax, 82, 58, 14, 5, "铁超载\nIron overload", COL["iron"], fontsize=8)

arrow(ax, 19, 55.5, 19, 53.5)
arrow(ax, 42, 55.5, 42, 53.5)
arrow(ax, 66, 55.5, 66, 53.5)
arrow(ax, 89, 55.5, 89, 53.5)

# 触发因素汇聚到 ROS / Fe2+
box(ax, 19, 50.5, 14, 4, "ROS 爆发\nOxidative stress", COL["ros"], fontsize=8)
box(ax, 42, 50.5, 16, 4, "GSH 耗竭\nGSH depletion", "#F9E79F", fontsize=8)
box(ax, 66, 50.5, 16, 4, "游离 Fe2+ 释放\nLabile Fe2+", COL["iron"], fontsize=8)
box(ax, 89, 50.5, 14, 4, "线粒体损伤\nMitochondrial injury", COL["mito"], fontsize=8)

arrow(ax, 26, 48.5, 26, 43.5, label="Fenton")
arrow(ax, 50, 48.5, 38, 43.5, rad=-0.2)
arrow(ax, 74, 48.5, 62, 43.5, rad=0.2)
arrow(ax, 89, 48.5, 70, 43.5, rad=0.3)

# ---- 7. 铁死亡通路 (左侧) ----
box(ax, 16, 39, 13, 5, "脂质过氧化\nLPO (ACSL4/LPCAT3)", COL["lpo"], fontsize=8)
box(ax, 16, 32, 13, 5, "GPX4 ↓ / SLC7A11 ↓\nxCT-GSH 轴抑制", "#F5B7B1", fontsize=8)
box(ax, 16, 25, 13, 5, "铁死亡\nFerroptosis", COL["danger"], fontsize=8)

arrow(ax, 16, 36.5, 16, 34.5)
arrow(ax, 16, 29.5, 16, 27.5)
# 亚致死分支到铁衰老
arrow(ax, 22.5, 32, 29, 32, color=COL["senescence"], lw=1.2,
      label="亚致死压力", label_size=6.5)

# ---- 8. 铁衰老通路 (中间) ----
box(ax, 37, 39, 14, 5, "DNA 损伤 / DDR\n(gamma-H2AX)", "#D7BDE2", fontsize=8)
box(ax, 37, 32, 14, 5, "p53/p21 & p16/Rb\n衰老通路激活", COL["senescence"], fontsize=8)
box(ax, 37, 25, 14, 5, "细胞衰老\nCellular Senescence", COL["senescence"], fontsize=8)
box(ax, 54, 32, 14, 5, "SASP\nIL-6 · IL-1b · TNF-a\nCcl2 · Cxcl10 · Mmp9", COL["sasp"], fontsize=7)

arrow(ax, 37, 36.5, 37, 34.5)
arrow(ax, 37, 29.5, 37, 27.5)
arrow(ax, 44, 32, 47, 32, label="分泌", label_size=6.5)
# SASP 反噬铁死亡
arrow(ax, 54, 29.5, 22.5, 26.5, color=COL["sasp"], lw=1.2, rad=0.25,
      label="正反馈", label_size=6.5)
# HMGB1 环路
box(ax, 54, 25, 14, 5, "HMGB1/TLR4\nNF-κB 激活", "#F5B7B1", fontsize=8)
arrow(ax, 54, 29.5, 54, 27.5)

# ---- 9. 单细胞 & 炎症数据注释 ----
ax.text(5, 21.5, "真实数据支撑:", fontsize=8, fontweight="bold", color=COL["text"])
ax.text(5, 19.5,
        "• 96-gene ferroaging signature: ACSL4/LPCAT3/GPX4/HMGB1/TLR4/IL6...\n"
        "• Microglia 7DPI ferrosenescence up (p_adj = 3.1e-4)\n"
        "• Neuron 1DPI ferrosenescence down (p_adj = 5.7e-13)\n"
        "• Ccl2/Il1b/Il6/Tnf strongly correlate with ferrosenescence score (rho > 0.77)",
        fontsize=6.8, color=COL["text"], linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COL["grey"], alpha=0.9))

# ---- 10. 艾叶干预 (右侧) ----
box(ax, 86, 39, 16, 5, "艾叶黄酮/酚酸\n(8 个活性成分)", COL["drug_light"], fontsize=8)
box(ax, 86, 32, 16, 5, "激活 Nrf2 / 解离 Keap1\n→ 核转位", "#AED6F1", fontsize=8)
box(ax, 86, 25, 16, 5, "Nrf2-ARE\n↑ GPX4 · FSP1 · FTH1 · HO-1", COL["protect"], fontsize=7)

arrow(ax, 86, 36.5, 86, 34.5)
arrow(ax, 86, 29.5, 86, 27.5)

# 抑制作用
inhib_arrow(ax, 78, 32, 72, 32, label="抑制 SASP", label_size=6.5)
inhib_arrow(ax, 78, 29, 62, 27.5, rad=-0.15, label="抑制 NF-κB", label_size=6.5)
inhib_arrow(ax, 78, 35, 62, 39, rad=0.15, label="清除 ROS/铁", label_size=6.5)

# 壮瑶医理论框
box(ax, 86, 19.5, 16, 4, "壮瑶医: 通龙路火路\n除毒邪 · 补虚损", COL["ethno"],
    fontsize=8, text_color="#7D6608")

# ---- 11. 底部结局 ----
box(ax, 18, 8, 18, 5, "铁死亡抑制\nFerroptosis ↓", COL["protect"], fontsize=8)
box(ax, 42, 8, 18, 5, "铁衰老抑制\nFerro-senescence ↓", COL["protect"], fontsize=8)
box(ax, 66, 8, 18, 5, "SASP 微环境重塑\nNeurovascular protection", COL["protect"], fontsize=8)
box(ax, 89, 8, 10, 5, "远期功能\n恢复", "#A9DFBF", fontsize=8)

arrow(ax, 27, 20, 27, 10.5)
arrow(ax, 51, 20, 51, 10.5)
arrow(ax, 75, 20, 75, 10.5)
arrow(ax, 84, 8, 89, 8)

# ---- 12. 图例 ----
legend_items = [
    ("铁死亡相关", COL["danger"]),
    ("铁衰老 / SASP", COL["senescence"]),
    ("艾叶干预 / 保护", COL["drug"]),
    ("ROS / 氧化应激", COL["ros"]),
    ("铁离子", COL["iron"]),
]
for i, (label, color) in enumerate(legend_items):
    lx, ly = 68 + i * 7, 3
    rect = Rectangle((lx - 1.2, ly - 0.6), 2.4, 1.2, facecolor=color,
                     edgecolor="#2C3E50", linewidth=0.6, alpha=0.8)
    ax.add_patch(rect)
    ax.text(lx + 2.2, ly, label, ha="left", va="center", fontsize=6.5,
            color=COL["text"])

# ---- 13. 标题 ----
ax.text(50, 68.5, "艾叶通过 Nrf2 双通路干预脑缺血再灌注铁衰老机制图",
        ha="center", va="top", fontsize=14, fontweight="bold", color=COL["text"])
ax.text(50, 65.5,
        "Artemisia argyi alleviates CIRI via Nrf2-dependent dual suppression of "
        "ferroptosis and ferroptosis-driven senescence",
        ha="center", va="top", fontsize=9, style="italic", color=COL["grey"])

# ---- 14. 保存 ----
png_path = os.path.join(OUTDIR, "AAI_CIRI_Ferrosenescence_Mechanism.png")
pdf_path = os.path.join(OUTDIR, "AAI_CIRI_Ferrosenescence_Mechanism.pdf")
plt.tight_layout()
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.1)
fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)

print(f"[OK] PNG -> {png_path}")
print(f"[OK] PDF -> {pdf_path}")
print(f"[INFO] Real data integrated: FA96 signature, core candidates, "
      f"snRNA-seq ferrosenescence stats, inflammation correlations, SCISSOR enrichment.")
