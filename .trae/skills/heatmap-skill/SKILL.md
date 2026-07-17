---
name: "heatmap-skill"
description: "聚类热图可视化技能。生成 pheatmap 聚类热图、ComplexHeatmap 临床注释热图、DE 基因热图。Invoke when user asks for heatmap with clustering, expression heatmap, or complex annotation heatmap."
---

# Heatmap 聚类热图可视化技能

## When to Invoke

当用户需要:
- 差异基因表达热图(DE heatmap)
- 聚类热图(hierarchical clustering)
- 带临床/分组注释的 ComplexHeatmap
- 微胶亚群 marker 热图

## Environment Setup

```r
suppressPackageStartupMessages({
  library(pheatmap); library(ggplot2); library(dplyr); library(readr)
  library(viridis); library(RColorBrewer); library(Cairo)
})
# ComplexHeatmap 可选(若已安装)
has_ch <- requireNamespace("ComplexHeatmap", quietly=TRUE)
if (has_ch) suppressPackageStartupMessages(library(ComplexHeatmap))
```

## 真实数据源(项目)

| 文件 | 路径 | 用途 |
|------|------|------|
| GSE61616 DE | `d:/铁衰老 绝不重蹈覆辙/L1/results/GSE61616_DE_results.csv` | 主 DE 结果(logFC/adj.P.Val) |
| GSE61616 表达矩阵 | `d:/铁衰老 绝不重蹈覆辙/L1/results/GSE61616_expression_matrix.csv` | 原始表达值(行=探针,列=样本) |
| GSE61616 样本元数据 | `d:/铁衰老 绝不重蹈覆辙/L1/results/GSE61616_sample_meta.csv` | 分组信息 |
| GPL1355 探针映射 | `d:/铁衰老 绝不重蹈覆辙/L1/results/GPL1355_probe_to_gene.csv` | 探针→基因符号 |
| 微胶 DEG | `d:/铁衰老 绝不重蹈覆辙/L2/results/GSE233815_sn/microglia_subcluster/microglia_high_fa_vs_low_fa_deg.csv` | 微胶高/低铁衰老 DEG |

**禁止**:不得生成模拟表达矩阵。

## Visualization Specifications

### Type 1: pheatmap DE 聚类热图
- Top 50 DE 基因(|logFC|>1, padj<0.01)
- 行/列双层 hierarchical clustering(ward.D2 + euclidean)
- Z-score 行标准化(`scale="row"`)
- 配色: `colorRampPalette(c("blue","white","red"))(100)`
- 列注释: Condition (Sham/MCAO)

### Type 2: ComplexHeatmap 临床注释
- 仅当 ComplexHeatmap 已安装时使用
- Top annotation: Condition, Time(如有)
- 行 annotation: gene function category(如有)

### Type 3: 微胶亚群 marker 热图
- 用 microglia_high_fa_vs_low_fa_deg.csv
- Top 30 高铁衰老 vs 低铁衰老 marker

## Code Template

完整模板见 [templates/test_heatmap.R](file:///d:/铁衰老 绝不重蹈覆辙/.trae/skills/heatmap-skill/templates/test_heatmap.R)

关键代码骨架:
```r
de <- read_csv("d:/铁衰老 绝不重蹈覆辙/L1/results/GSE61616_DE_results.csv")
expr <- read_csv("d:/铁衰老 绝不重蹈覆辙/L1/results/GSE61616_expression_matrix.csv")
meta <- read_csv("d:/铁衰老 绝不重蹈覆辙/L1/results/GSE61616_sample_meta.csv")
gpl  <- read_csv("d:/铁衰老 绝不重蹈覆辙/L1/results/GPL1355_probe_to_gene.csv")
stopifnot(nrow(de) > 0, nrow(expr) > 0, nrow(meta) > 0)

# 选 Top 50 DE 基因
top_genes <- de %>% filter(!is.na(adj.P.Val)) %>%
  mutate(sig = adj.P.Val < 0.01 & abs(logFC) > 1) %>%
  filter(sig) %>% arrange(adj.P.Val) %>% head(50) %>% pull(Probe)

# 探针→基因映射
probe2gene <- setNames(gpl$GeneSymbol, gpl$Probe)
expr_sub <- expr %>% filter(Probe %in% top_genes) %>%
  tibble::column_to_rownames("Probe")
rownames(expr_sub) <- coalesce(probe2gene[rownames(expr_sub)], rownames(expr_sub))

# 样本注释
annot_col <- data.frame(Condition = meta$condition[match(colnames(expr_sub), meta$sample_id)],
                        row.names = colnames(expr_sub))
annot_colors <- list(Condition = c(Sham="#6FB2C1", MCAO="#E07524"))

pheatmap(expr_sub, scale="row", cluster_rows=TRUE, cluster_cols=TRUE,
         clustering_distance_rows="euclidean", clustering_distance_cols="euclidean",
         clustering_method="ward.D2",
         color=colorRampPalette(c("blue","white","red"))(100),
         annotation_col=annot_col, annotation_colors=annot_colors,
         show_colnames=FALSE, fontsize_row=7,
         filename=png_path, width=10, height=8, dpi=300)
```

## Quality Checklist

- [ ] 300 DPI PNG + 矢量 PDF
- [ ] 行 Z-score 标准化(`scale="row"`)
- [ ] ward.D2 + euclidean 聚类(不用默认 complete)
- [ ] 真实表达矩阵(非模拟)
- [ ] 列注释显示 Condition
- [ ] CVD-safe diverging 配色(blue-white-red)
- [ ] 基因符号可读(fontsize_row ≥ 7)

## Common Failure Modes

1. **表达矩阵含 NA/Inf**: pheatmap 报错 → 读取后 `na.omit()` + `filter_all(all_vars(is.finite))`
2. **探针未映射**: rownames 显示探针 ID → 用 GPL1355 映射为基因符号
3. **样本元数据 sample_id 不匹配**: `match()` 前确认列名一致
4. **ward.D vs ward.D2 陷阱**: pheatmap 默认 `ward.D`,必须显式 `ward.D2`
5. **行标准化失败**: 全零行导致 sd=0 → 先过滤低表达基因
6. **ComplexHeatmap 未安装**: 回退到 pheatmap,不强制 ComplexHeatmap
