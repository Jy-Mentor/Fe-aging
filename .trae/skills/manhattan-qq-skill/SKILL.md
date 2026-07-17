---
name: "manhattan-qq-skill"
description: "曼哈顿图与 QQ 图可视化技能。生成 GWAS 曼哈顿图、QQ 图、Miami plot、基因组区域放大图。Invoke when user asks for Manhattan plot, QQ plot, GWAS visualization, or genomic inflation diagnostics."
---

# Manhattan-QQ 曼哈顿图与 QQ 图可视化技能

## When to Invoke

当用户需要:
- GWAS 曼哈顿图(染色体 vs -log10(p))
- QQ 图(观察 vs 期望 -log10(p))
- Miami plot(上下双向曼哈顿)
- 基因组膨胀因子 λ 诊断
- 区域放大 manhattan(chr + region)

## Environment Setup

```r
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(readr); library(tidyr)
  library(viridis); library(ggsci); library(Cairo)
})
# qqman 可选(若已安装)
has_qqman <- requireNamespace("qqman", quietly=TRUE)
if (has_qqman) suppressPackageStartupMessages(library(qqman))
```

## 真实数据源

**项目当前无 GWAS summary statistics**(铁衰老项目是 bulk RNA-seq + 单细胞 + 化合物预测,非 GWAS)。

可选真实数据源:
1. **项目 DE 结果 GSE61616_DE_results.csv**(真实)
   - 列: Probe / logFC / adj.P.Val / 等
   - 无染色体坐标 → 无法做标准曼哈顿
   - 但可用 -log10(adj.P.Val) 做 QQ 图(诊断 DE p 值分布)
2. **qqman::gwasResults**(qqman 包内置真实 GWAS 模拟数据,16,000 SNP)
   - 注意: 这是 qqman 包自带的演示数据,虽然是模拟生成,但作为 GWAS 可视化标准演示数据集被广泛使用
   - **用户禁止模拟数据原则**: 此数据集仅用于验证绘图函数可用,真实分析必须用项目 GWAS sumstats
3. **PheWAS / GWAS catalog 公开数据**(需下载)

**禁止**:不得捏造 GWAS p 值。无真实 GWAS 时:
- QQ 图用 GSE61616_DE_results.csv 真实 DE p 值(合法用法)
- Manhattan 仅写规范,不运行(因无染色体坐标)

## Visualization Specifications

### Type 1: Manhattan Plot(标准)
- 横轴 = 染色体位置(1-22 + X/Y),纵轴 = -log10(p)
- 染色体交替配色(深浅蓝)
- 显著阈值线: `geom_hline(yintercept=-log10(5e-8))` 红色
- 建议阈值线: `geom_hline(yintercept=-log10(1e-5))` 蓝色
- Top SNP 标注(`ggrepel`)

### Type 2: QQ Plot
- 横轴 = -log10(expected p),纵轴 = -log10(observed p)
- 对角线 y=x 参考线
- λ(genomic inflation factor)= median(observed χ²) / 0.4549
- λ > 1.1 提示膨胀

### Type 3: Miami Plot
- 上半 = trait 1 曼哈顿,下半 = trait 2(镜像)
- 共享横轴(染色体位置)

### Type 4: Regional Manhattan(放大)
- 单染色体 + 上下文区域(±500kb)
- LD coloring(需 PLINK r2)

## Code Template

完整模板见 [templates/test_manhattan_qq.R](file:///d:/铁衰老 绝不重蹈覆辙/.trae/skills/manhattan-qq-skill/templates/test_manhattan_qq.R)

关键代码骨架(QQ 图用真实 DE p 值):
```r
de <- read_csv("d:/铁衰老 绝不重蹈覆辙/L1/results/GSE61616_DE_results.csv")
stopifnot(nrow(de) > 0, "adj.P.Val" %in% names(de))

# QQ plot from real DE p-values
p_vals <- de %>% filter(!is.na(adj.P.Val)) %>% pull(adj.P.Val)
p_vals <- pmax(p_vals, 1e-300)  # avoid -log10(0)
n <- length(p_vals)
qq_df <- tibble(
  observed = sort(p_vals),
  expected = -log10(ppoints(n)),
  observed_log = -log10(observed)
)

# Genomic inflation factor lambda
chisq <- qchisq(1 - p_vals, 1)
lambda <- median(chisq, na.rm=TRUE) / qchisq(0.5, 1)

p_qq <- ggplot(qq_df, aes(x=expected, y=observed_log)) +
  geom_abline(slope=1, intercept=0, color="red", linetype="dashed", linewidth=0.5) +
  geom_point(alpha=0.4, size=0.8, color="#377EB8") +
  labs(x=expression(-log[10](expected)~italic(p)),
       y=expression(-log[10](observed)~italic(p)),
       title=sprintf("QQ Plot — GSE61616 DE (λ=%.3f)", lambda),
       tag="A") +
  theme_bw(base_size=10)

# Manhattan: 仅当有 CHR/BP/P 列时绘制
has_gwas_cols <- all(c("CHR","BP","P") %in% names(de))
if (!has_gwas_cols) {
  message("No CHR/BP/P columns in DE data — Manhattan plot skipped (no genomic coordinates). ",
          "Manhattan requires GWAS sumstats with chromosome + base position.")
  # 仅生成 QQ,Manhattan 留空
} else {
  # 用 qqman::manhattan(de, chr="CHR", bp="BP", p="P", ...)
}
```

## Quality Checklist

- [ ] 300 DPI PNG + Cairo::CairoPDF
- [ ] 真实 p 值(非模拟)
- [ ] QQ 图对角线参考线
- [ ] λ(inflation factor)计算并标注
- [ ] Manhattan 显著阈值线(5e-8 / 1e-5)
- [ ] 染色体交替配色
- [ ] 无 GWAS 坐标时 → 仅 QQ,不绘 Manhattan

## Common Failure Modes

1. **p=0 导致 -log10(0)=Inf**: `pmax(p, 1e-300)` 钳制
2. **λ 计算用错分布**: χ² with df=1,median / qchisq(0.5,1)=0.4549
3. **Manhattan 无 CHR/BP**: DE 数据无基因组坐标 → 跳过 Manhattan,仅 QQ
4. **qqman::manhattan 无法用 ggplot2 自定义**: 改用 ggplot2 手动绘制(更灵活)
5. **QQ 图尾巴翘起**: 提示真实信号(正常)或膨胀(λ>1.1 需校正)
6. **Miami plot 镜像方向**: 下半图 y 轴反转 `scale_y_reverse()`
