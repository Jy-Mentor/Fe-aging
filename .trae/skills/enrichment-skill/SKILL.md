---
name: "enrichment-skill"
description: "富集分析可视化技能。生成 GO/KEGG/Reactome/WikiPathway 富集图、GSEA 图、cnetplot、emapplot。Invoke when user asks for enrichment visualization, GO/KEGG bar plot, GSEA running score, or pathway enrichment network."
---

# Enrichment 富集分析可视化技能

## When to Invoke

当用户需要:
- GO/KEGG/Reactome/WikiPathway 富集条形图/气泡图
- GSEA running score plot
- cnetplot(基因-通路网络)
- emapplot(通路相似性网络)
- ridgeplot(GSEA 分布)
- 多通路热图(NES matrix)

## Environment Setup

```r
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(readr); library(tidyr)
  library(stringr); library(viridis); library(ggsci); library(patchwork)
  library(ggrepel); library(Cairo)
})
# clusterProfiler/enrichplot 可选(若已安装)
has_cp <- requireNamespace("clusterProfiler", quietly=TRUE)
has_ep <- requireNamespace("enrichplot", quietly=TRUE)
if (has_cp) suppressPackageStartupMessages(library(clusterProfiler))
if (has_ep) suppressPackageStartupMessages(library(enrichplot))
```

## 真实数据源(项目)

| 文件 | 路径 | 用途 |
|------|------|------|
| GO BP 富集 | `d:/铁衰老 绝不重蹈覆辙/L2/results/core_go_bp_enrichment.csv` | 主 GO 条形图 |
| GO CC 富集 | `d:/铁衰老 绝不重蹈覆辙/L2/results/core_go_cc_enrichment.csv` | 细胞组分 |
| GO MF 富集 | `d:/铁衰老 绝不重蹈覆辙/L2/results/core_go_mf_enrichment.csv` | 分子功能 |
| KEGG 富集 | `d:/铁衰老 绝不重蹈覆辙/L2/results/core_kegg_enrichment.csv` | 通路富集 |
| Reactome | `d:/铁衰老 绝不重蹈覆辙/L2/results/core_reactome_enrichment.csv` | Reactome |
| WikiPathway | `d:/铁衰老 绝不重蹈覆辙/L2/results/core_wikipathway_enrichment.csv` | WikiPathway |
| GSEA 结果 | `d:/铁衰老 绝不重蹈覆辙/L2/results/gsea_ferroaging_vs_ferroptosis.csv` | GSEA NES/padj |
| GSEA NES matrix | `d:/铁衰老 绝不重蹈覆辙/L2/results/gsea_nes_matrix.csv` | 多通路 NES |
| GSEA padj matrix | `d:/铁衰老 绝不重蹈覆辙/L2/results/gsea_padj_matrix.csv` | 多通路 padj |
| STRING 富集 | `d:/铁衰老 绝不重蹈覆辙/L1/results/string_enrichment.csv` | STRING 富集 |

**禁止**:不得模拟富集结果。

## Visualization Specifications

### Type 1: GO/KEGG Bar Plot
- Top 20 by padj
- 横轴 = -log10(padj),纵轴 = Description
- 颜色 = padj 或 count
- 按 ontology 分面(BP/CC/MF)

### Type 2: GO/KEGG Bubble Plot
- 横轴 = GeneRatio,纵轴 = Description
- 气泡大小 = Count,颜色 = padj
- Top 25

### Type 3: GSEA Running Score(需 clusterProfiler 对象)
- 项目仅 CSV → 用 ggplot2 手动绘制 NES 棒棒糖
- 横轴 = NES,纵轴 = pathway
- 颜色 = padj

### Type 4: NES Heatmap
- `gsea_nes_matrix.csv` 行=通路 列=数据集
- `geom_tile` + `scale_fill_gradient2`

### Type 5: cnetplot/emapplot(需 enrichplot)
- 仅当有 clusterProfiler 对象时使用
- 项目仅 CSV → 此类型仅写规范

## Code Template

完整模板见 [templates/test_enrichment.R](file:///d:/铁衰老 绝不重蹈覆辙/.trae/skills/enrichment-skill/templates/test_enrichment.R)

关键代码骨架:
```r
go_bp <- read_csv("d:/铁衰老 绝不重蹈覆辙/L2/results/core_go_bp_enrichment.csv")
kegg  <- read_csv("d:/铁衰老 绝不重蹈覆辙/L2/results/core_kegg_enrichment.csv")
stopifnot(nrow(go_bp) > 0, nrow(kegg) > 0)

# 检查列名(不同来源可能不同)
go_names <- names(go_bp)
padj_col <- intersect(c("p.adjust","padj","FDR","fdr"), go_names)[1]
desc_col <- intersect(c("Description","Term","term","ID"), go_names)[1]
count_col <- intersect(c("Count","count","gene_count"), go_names)[1]
stopifnot(!is.na(padj_col), !is.na(desc_col))

# Top 20 bar
go_top <- go_bp %>% filter(!is.na(.data[[padj_col]])) %>%
  mutate(neg_log10_padj = -log10(.data[[padj_col]])) %>%
  arrange(desc(neg_log10_padj)) %>% head(20) %>%
  mutate(Description = str_trunc(.data[[desc_col]], 60),
         Description = factor(Description, levels=rev(Description)))

p_bar <- ggplot(go_top, aes(x=neg_log10_padj, y=Description, fill=neg_log10_padj)) +
  geom_col(width=0.7, alpha=0.9) +
  scale_fill_viridis_c(option="C", direction=-1, name="-log10(padj)") +
  labs(x="-log10(adjusted P-value)", y=NULL, tag="A",
       title="GO BP Enrichment (Top 20)") +
  theme_bw(base_size=9) +
  theme(axis.text.y=element_text(size=7),
        plot.title=element_text(face="bold", size=10))

# KEGG bubble
kegg_top <- kegg %>% filter(!is.na(.data[[padj_col]])) %>%
  arrange(.data[[padj_col]]) %>% head(25) %>%
  mutate(Description = str_trunc(.data[[desc_col]], 55),
         Description = factor(Description, levels=rev(Description)))
p_bubble <- ggplot(kegg_top, aes(x=GeneRatio, y=Description,
                                  size=.data[[count_col]], color=.data[[padj_col]])) +
  geom_point(alpha=0.85) +
  scale_color_viridis_c(option="D", name="padj") +
  scale_size_continuous(range=c(2,7), name="Count") +
  labs(x="GeneRatio", y=NULL, tag="B", title="KEGG Enrichment (Top 25)") +
  theme_bw(base_size=9) +
  theme(axis.text.y=element_text(size=7))
```

## Quality Checklist

- [ ] 300 DPI PNG + Cairo::CairoPDF
- [ ] 真实富集 CSV(非模拟)
- [ ] 列名自适应(`intersect()` 取第一个匹配)
- [ ] Top N 过滤(默认 20-25,避免过密)
- [ ] -log10(padj) 颜色映射
- [ ] pathway 标签截断(`str_trunc(_, 60)`)
- [ ] CVD-safe 配色

## Common Failure Modes

1. **列名不统一**: core_go_bp_enrichment.csv vs clusterProfiler 输出列名不同 → `intersect()` 自适应
2. **padj 含 0**: -log10(0) = Inf → `mutate(padj = pmax(padj, 1e-300))` 钳制
3. **Description 过长**: 标签重叠 → `str_trunc(_, 60)`
4. **GeneRatio 是字符串**: "5/100" → 需解析或用 Count
5. **GSEA running score 无对象**: 项目仅 CSV → 改用 NES 棒棒糖
6. **cnetplot/emapplot 需对象**: 仅 clusterProfiler 对象可用,CSV 输入不直接支持
