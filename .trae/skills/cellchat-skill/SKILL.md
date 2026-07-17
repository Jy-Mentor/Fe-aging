---
name: "cellchat-skill"
description: "CellChat 细胞通讯网络可视化技能。生成弦图、通讯热图、LR 气泡图、通路贡献图。Invoke when user asks for cell-cell communication analysis, ligand-receptor network visualization, or CellChat result plotting."
---

# CellChat 细胞通讯网络可视化技能

## When to Invoke

当用户需要:
- 绘制 CellChat / cell-cell communication 网络图
- 可视化 ligand-receptor (LR) 互作
- 通讯强度热图、气泡图、通路贡献条形图
- 弦图(chord diagram)展示细胞群间通讯

## Environment Setup

```r
# 必需 R 包(项目已安装)
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr)
  library(stringr); library(patchwork); library(viridis); library(ggsci)
  library(circlize); library(cowplot); library(Cairo)
})
# Cairo::CairoPDF 替代 cairo_pdf(Windows winCairo.dll 加载失败 workaround)
stopifnot(requireNamespace("Cairo", quietly = TRUE))
```

## 真实数据源(项目)

| 文件 | 路径 | 内容 |
|------|------|------|
| 通路通讯概率 | `d:/铁衰老 绝不重蹈覆辙/L2/results/cellchat_signaling_pathways.csv` | source/target/pathway_name/prob |
| LR 互作对 | `d:/铁衰老 绝不重蹈覆辙/L2/results/cellchat_lr_pairs.csv` | interaction_name/prob/source/target |

**禁止**:不得模拟 CellChat 输出。所有数据必须从上述真实 CSV 读取。

## Visualization Specifications

### Panel A: Chord Diagram(弦图)
- 用 `circlize::chordDiagram()` 绘制细胞群间总通讯概率邻接矩阵
- NPG 配色(`pal_npg("nrc")`),≤10 色用 NPG,>10 用 viridis("D")
- `directional = 0`, `transparency = 0.2`, `annotationTrack = c("grid","name")`
- 弦图是 base R 图形设备 → 导出 PNG → `cowplot::draw_image()` 读回嵌入 patchwork

### Panel B: Communication Heatmap(通讯热图)
- `geom_tile()` source×total_prob×target 矩阵
- `scale_fill_gradient(low="white", high="#08519c")`
- 数值标签 `geom_text(size=1.8, color="grey20")`

### Panel C: LR Bubble Plot(LR 气泡图)
- Top 50 LR 互作(按 total_prob 排序)
- 横轴 = interaction_name,纵轴 = total_prob
- 气泡大小 = sender-receiver pair 数,颜色 = log10(total_prob+1)

### Panel D: Pathway Contribution(通路贡献)
- Top 25 通路(按 total_prob)
- 按 pathway_name 模式分 4 类:Secreted/ECM-Receptor/Cell-Cell Contact/Non-protein
- 配色: `#E41A1C` / `#00CED1` / `#4DAF4A` / `#1F78B4`

## Code Template

完整模板见 [templates/test_cellchat.R](file:///d:/铁衰老 绝不重蹈覆辙/.trae/skills/cellchat-skill/templates/test_cellchat.R)

关键代码骨架:
```r
pathways <- read_csv("d:/铁衰老 绝不重蹈覆辙/L2/results/cellchat_signaling_pathways.csv")
lr_pairs <- read_csv("d:/铁衰老 绝不重蹈覆辙/L2/results/cellchat_lr_pairs.csv")
stopifnot(nrow(pathways) > 0, nrow(lr_pairs) > 0)

# Panel A: chord
adj_mat <- pathways %>% filter(prob > 0) %>%
  group_by(source, target) %>% summarise(total_prob = sum(prob), .groups="drop") %>%
  { mat <- matrix(0, nrow=length(unique(c(.$source,.$target))), ncol=length(unique(c(.$source,.$target))));
    rownames(mat) <- colnames(mat) <- sort(unique(c(.$source,.$target)));
    for (i in seq_len(nrow(.))) mat[.[i,"source"][[1]], .[i,"target"][[1]]] <- .[i,"total_prob"][[1]];
    mat }

png(path, width=9, height=9, units="in", res=300, bg="white")
circos.clear(); circos.par(gap.after=2, cell.padding=c(0.02,0,0.02,0))
chordDiagram(adj_mat, grid.col=cell_colors, transparency=0.2, directional=0,
             annotationTrack=c("grid","name"),
             preAllocateTracks=list(track.height=0.08))
circos.clear(); dev.off()
```

## Quality Checklist

- [ ] 300 DPI PNG + 矢量 PDF 双输出
- [ ] NPG/viridis CVD-safe 配色
- [ ] 真实数据(`stopifnot(nrow > 0)` 断言)
- [ ] `circos.clear()` 在每个 chord 前后调用(避免状态泄漏)
- [ ] Cairo::CairoPDF 输出 PDF(避免 winCairo.dll 错误)
- [ ] 4 面板 composite: A=chord / B=heatmap | C=bubble / D=pathway
- [ ] panel tag (A/B/C/D) 加粗 14pt

## Common Failure Modes

1. **circlize 状态泄漏**: 忘记 `circos.clear()` 导致下一图错乱 → 每次绘图前后必须 clear
2. **chord 无法嵌入 patchwork**: circlize 是 base R → 必须导出 PNG 后 `cowplot::draw_image()` 读回
3. **cellchat_lr_pairs.csv 巨大(56MB)**: 读取后立即 `filter(prob > 0)` 减少内存
4. **winCairo.dll 缺失**: 用 `Cairo::CairoPDF` 替代 `cairo_pdf`
5. **通路分类规则冲突**: pathway_name 匹配按优先级(Secreted > ECM > Contact > Non-protein)
