---
name: "circos-skill"
description: "Circos 圈图可视化技能。生成 PPI 互作圈图、GO 富集圈图、基因组圈图。Invoke when user asks for circos plot, chord diagram of gene interactions, or circular enrichment visualization."
---

# Circos 圈图可视化技能

## When to Invoke

当用户需要:
- PPI(蛋白互作)网络圈图/弦图
- GO/KEGG 富集圈图(gene-pathway 二分图)
- 基因组位置圈图(需染色体坐标)
- 多层注释圈图

## Environment Setup

```r
suppressPackageStartupMessages({
  library(circlize); library(ggplot2); library(dplyr); library(readr)
  library(viridis); library(ggsci); library(Cairo); library(cowplot)
})
stopifnot(requireNamespace("circlize", quietly = TRUE))
```

## 真实数据源(项目)

| 文件 | 路径 | 用途 |
|------|------|------|
| PPI 边 | `d:/铁衰老 绝不重蹈覆辙/L2/results/core_ppi_edges.csv` | 蛋白互作弦图 |
| PPI 拓扑 | `d:/铁衰老 绝不重蹈覆辙/L2/results/core_ppi_topology.csv` | 度中心性着色 |
| GO BP 富集 | `d:/铁衰老 绝不重蹈覆辙/L2/results/core_go_bp_enrichment.csv` | gene-pathway 圈图 |
| KEGG 富集 | `d:/铁衰老 绝不重蹈覆辙/L2/results/core_kegg_enrichment.csv` | 通路圈图 |

**禁止**:不得模拟 PPI 边或富集结果。

## Visualization Specifications

### Type 1: PPI Chord Diagram
- 节点 = 基因,边 = PPI 互作(weight = combined_score)
- 按 Degree 排序,Top 25 基因入图
- 节点颜色 = Degree(viridis "D")
- 边透明度 = combined_score 归一化

### Type 2: GO Enrichment Circos
- 左半圈 = GO term(Top 20 by padj),右半圈 = gene
- 连线 = gene-pathway membership
- GO 颜色 = -log10(padj)
- gene 颜色 = logFC(如有)或灰度

### Type 3: Multi-track Genomic Circos
- 仅当有染色体坐标数据时使用
- 项目当前无基因组坐标数据 → 此类型仅写规范,不运行

## Code Template

完整模板见 [templates/test_circos.R](file:///d:/铁衰老 绝不重蹈覆辙/.trae/skills/circos-skill/templates/test_circos.R)

关键代码骨架:
```r
ppi_edges <- read_csv("d:/铁衰老 绝不重蹈覆辙/L2/results/core_ppi_edges.csv")
ppi_topo  <- read_csv("d:/铁衰老 绝不重蹈覆辙/L2/results/core_ppi_topology.csv")
stopifnot(nrow(ppi_edges) > 0, nrow(ppi_topo) > 0)

# Top 25 hub genes by degree
hub_genes <- ppi_topo %>% arrange(desc(Degree)) %>% head(25) %>% pull(Gene)
ppi_sub <- ppi_edges %>%
  filter(source %in% hub_genes & target %in% hub_genes) %>%
  mutate(weight = combined_score / 1000)

# Build matrix
mat <- matrix(0, nrow=length(hub_genes), ncol=length(hub_genes),
              dimnames=list(hub_genes, hub_genes))
for (i in seq_len(nrow(ppi_sub))) {
  mat[ppi_sub$source[i], ppi_sub$target[i]] <- ppi_sub$weight[i]
}

col_vec <- viridis(length(hub_genes), option="D")[order(order(ppi_topo$Degree[ppi_topo$Gene %in% hub_genes]))]
names(col_vec) <- hub_genes

png(path, width=9, height=9, units="in", res=300, bg="white")
circos.clear(); circos.par(gap.after=2, cell.padding=c(0.02,0,0.02,0))
chordDiagram(mat, grid.col=col_vec, transparency=0.3,
             annotationTrack=c("grid","name"),
             preAllocateTracks=list(track.height=0.08))
circos.clear(); dev.off()
```

## Quality Checklist

- [ ] 300 DPI PNG + Cairo::CairoPDF
- [ ] `circos.clear()` 前后调用
- [ ] 真实数据断言(`stopifnot`)
- [ ] CVD-safe 配色(viridis/NPG)
- [ ] Top N 过滤(避免过密,默认 25)
- [ ] 节点标签可读(size ≥ 6pt in PNG at 300 DPI)

## Common Failure Modes

1. **圈图过密**: 节点 > 50 时标签重叠 → Top 25 过滤 + 透明度 0.3
2. **mat 维度不匹配**: 确保 rownames/colnames 严格一致
3. **weight 未归一化**: combined_score 0-1000 → 除以 1000 转换为 0-1
4. **circos.par 残留**: 上一次运行的参数影响下次 → `circos.clear()` 必调
5. **多圈图冲突**: 同一脚本绘多个圈图,每个之间必须 `circos.clear()` + `dev.off()`
