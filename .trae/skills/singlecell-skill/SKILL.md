---
name: "singlecell-skill"
description: "单细胞可视化技能。生成 UMAP/tSNE、marker 热图、拟时序、SCISSOR 细胞选择图。Invoke when user asks for single-cell RNA-seq visualization, UMAP plot, marker genes, or SCISSOR cell selection plot."
---

# Single-Cell 单细胞可视化技能

## When to Invoke

当用户需要:
- UMAP/tSNE 散点图(按 cell type / condition / score 着色)
- Marker 基因气泡图/热图
- 模块评分 violin plot
- SCISSOR selected cells 可视化
- 拟时序轨迹(slingshot/monocle)
- 细胞比例堆叠柱

## Environment Setup

```r
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(readr); library(tidyr)
  library(ggpubr); library(ggrepel); library(patchwork); library(viridis)
  library(ggsci); library(Cairo)
})
# Seurat 可选(若已安装)
has_seurat <- requireNamespace("Seurat", quietly=TRUE)
if (has_seurat) suppressPackageStartupMessages(library(Seurat))
```

## 真实数据源(项目)

| 文件 | 路径 | 用途 |
|------|------|------|
| UMAP 元数据 | `d:/铁衰老 绝不重蹈覆辙/figures/meta_with_umap.csv` | 7414 细胞 UMAP 坐标 + cell_type + Condition + FA score |
| SCISSOR UMAP | `d:/铁衰老 绝不重蹈覆辙/figures/scissor_umap_metadata.csv` | SCISSOR 选择细胞 |
| SCISSOR 选择 | `d:/铁衰老 绝不重蹈覆辙/L2/results/scissor_selected_cells.csv` | 被选细胞列表 |
| SCISSOR 富集 | `d:/铁衰老 绝不重蹈覆辙/L2/results/scissor_celltype_enrichment.csv` | 细胞型富集 |
| 细胞元数据 | `d:/铁衰老 绝不重蹈覆辙/L2/results/GSE233815_sn/cell_metadata_with_ferroaging_score.csv` | 完整元数据 |
| 微胶 marker | `d:/铁衰老 绝不重蹈覆辙/L2/results/GSE233815_sn/microglia_subcluster/microglia_high_fa_markers_annotated.csv` | 微胶 marker |
| FA by cellclass | `d:/铁衰老 绝不重蹈覆辙/L2/results/GSE233815_sn/ferroaging_score_by_condition_cellclass.csv` | 聚合评分 |

**禁止**:不得模拟单细胞数据或生成随机 UMAP 坐标。

## Visualization Specifications

### Type 1: UMAP 散点图
- 横轴 UMAP_1,纵轴 UMAP_2
- 着色模式: cell_type_1(discrete, NPG/viridis) / FA score(continuous, viridis)
- `geom_point(size=0.25, alpha=0.7)`
- 图例 `override.aes=list(size=2.5, alpha=1)`

### Type 2: Marker Violin / Box
- 按细胞型 × Condition 分组
- `geom_violin` + `geom_boxplot(width=0.12)` 双层
- Wilcoxon 检验 `***`/`**`/`*`/NS 标记

### Type 3: SCISSOR Selected Cells
- UMAP 上高亮 SCISSOR selected(红)vs 其他(灰)
- 右侧细胞型富集条形图

### Type 4: Cell Proportion Stacked Bar
- 按 Condition 分组,cell_type 占比

### Type 5: Pseudotime(仅当 slingshot 已装)
- 项目当前无拟时序结果 → 仅写规范

## Code Template

完整模板见 [templates/test_singlecell.R](file:///d:/铁衰老 绝不重蹈覆辙/.trae/skills/singlecell-skill/templates/test_singlecell.R)

关键代码骨架:
```r
meta <- read_csv("d:/铁衰老 绝不重蹈覆辙/figures/meta_with_umap.csv", show_col_types=FALSE)
stopifnot(nrow(meta) > 0, all(c("UMAP_1","UMAP_2","cell_type_1") %in% names(meta)))
score_col <- intersect(c("AddModuleScore_FA96","FA_96_UCell","AddModuleScore_FA95"), names(meta))[1]
stopifnot(!is.na(score_col))

# Panel A: UMAP by cell type
n_types <- length(unique(meta$cell_type_1))
type_colors <- setNames(viridis(n_types, option="D"), unique(meta$cell_type_1))
pA <- ggplot(meta, aes(x=UMAP_1, y=UMAP_2, color=cell_type_1)) +
  geom_point(size=0.25, alpha=0.7) +
  scale_color_manual(values=type_colors, name="Cell Type") +
  labs(x="UMAP 1", y="UMAP 2", tag="A") +
  theme_bw(base_size=9) +
  guides(color=guide_legend(override.aes=list(size=2.5, alpha=1), ncol=1))

# Panel B: FA score UMAP
pB <- ggplot(meta %>% filter(!is.na(.data[[score_col]])),
             aes(x=UMAP_1, y=UMAP_2, color=.data[[score_col]])) +
  geom_point(size=0.25, alpha=0.7) +
  scale_color_viridis_c(option="C", name="FA Score") +
  labs(x="UMAP 1", y="UMAP 2", tag="B") + theme_bw(base_size=9)

# Panel C: Violin by cell_type × Condition
meta_clean <- meta %>% filter(!is.na(.data[[score_col]]), !is.na(cell_type_1), !is.na(Condition)) %>%
  mutate(Condition = factor(Condition, levels=c("Ctrl","MCAO")))
stat_tests <- meta_clean %>% group_by(cell_type_1) %>%
  summarise(p_value = tryCatch(
    wilcox.test(.data[[score_col]][Condition=="MCAO"], .data[[score_col]][Condition=="Ctrl"])$p.value,
    error=function(e) NA), .groups="drop") %>%
  mutate(p_label = case_when(is.na(p_value)~"NS", p_value<0.001~"***", p_value<0.01~"**", p_value<0.05~"*", TRUE~"NS"))
pC <- ggplot(meta_clean, aes(x=cell_type_1, y=.data[[score_col]], fill=Condition)) +
  geom_violin(alpha=0.5, linewidth=0.3, position=position_dodge(0.8)) +
  geom_boxplot(width=0.12, alpha=0.6, linewidth=0.25, position=position_dodge(0.8), outlier.size=0.2) +
  scale_fill_manual(values=c("Ctrl"="#6FB2C1","MCAO"="#E07524")) +
  labs(x=NULL, y="Ferroaging Score", tag="C") + theme_bw(base_size=9)
```

## Quality Checklist

- [ ] 300 DPI PNG + Cairo::CairoPDF
- [ ] 真实 7414 细胞 UMAP 坐标(非模拟)
- [ ] CVD-safe 配色(viridis/NPG)
- [ ] Wilcoxon 检验 p 值标注
- [ ] Condition 因子 levels=c("Ctrl","MCAO")
- [ ] UMAP 点 size ≤ 0.3(避免重叠)
- [ ] 多面板 patchwork 组合 + panel tag

## Common Failure Modes

1. **score_col 名称不匹配**: 项目数据可能用 `AddModuleScore_FA96`/`FA_96_UCell`/`AddModuleScore_FA95` → `intersect()` 取第一个
2. **NA 行导致 wilcox 报错**: `filter(!is.na(score))` 前置
3. **cell_type 因子顺序乱**: 按生物学逻辑排(levels 显式指定)
4. **图例点过大**: `override.aes=list(size=2.5)` 控制
5. **SCISSOR CSV 列名变化**: 读取后 `names()` 检查,不假设列名
6. **Seurat 对象过大**: 项目仅用元数据 CSV,不要求 Seurat 对象
