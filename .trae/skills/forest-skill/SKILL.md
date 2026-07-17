---
name: "forest-skill"
description: "森林图可视化技能。生成 meta 分析森林图、效应量森林图、亚组分析森林图。Invoke when user asks for forest plot, meta-analysis visualization, or effect size plot with confidence intervals."
---

# Forest 森林图可视化技能

## When to Invoke

当用户需要:
- Meta 分析森林图(多研究合并)
- 效应量 + 95% CI 森林图
- 亚组分析森林图
- 外部验证相关系数森林图(Spearman rho + Fisher CI)
- 模型性能比较森林图(AUC/AUPR)

## Environment Setup

```r
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(readr); library(stringr)
  library(patchwork); library(ggsci); library(viridis); library(Cairo)
})
# forestplot 包可选(若已安装)
has_fp <- requireNamespace("forestplot", quietly=TRUE)
if (has_fp) suppressPackageStartupMessages(library(forestplot))
```

## 真实数据源(项目)

| 文件 | 路径 | 用途 |
|------|------|------|
| 外部验证 | `d:/铁衰老 绝不重蹈覆辙/L2/results/external_validation_results.csv` | Spearman rho + Fisher CI |
| 模型性能 | `d:/铁衰老 绝不重蹈覆辙/L4/results_v10_minibatch/model_performance_v67.csv` | AUC/AUPR |
| LASSO 候选 | `d:/铁衰老 绝不重蹈覆辙/L2/results/ciri_ferroaging_lasso_candidates.csv` | Cohen's d 效应量 |
| ssGSEA 效应量 | `d:/铁衰老 绝不重蹈覆辙/L2/results/ssgsea_effect_size.csv` | Cohen's d |

**禁止**:不得模拟效应量或 CI。

## Visualization Specifications

### Type 1: 外部验证森林图(主用)
- 横轴 = Spearman rho,纵轴 = Dataset
- 95% CI via Fisher z-transform(`atanh` / `tanh`)
- 点大小 = FA AUC,颜色 = rho 值
- 显著性标记: `***` (p<0.001) / `**` (p<0.01) / `*` (p<0.05) / NS
- `geom_vline(xintercept=0, linetype="dashed")` 参考线

### Type 2: 模型性能森林图
- 横轴 = AUC/AUPR,纵轴 = Model
- CI 用 bootstrap 或直接报告点估计
- 按 metric 分面或 dodged

### Type 3: 亚组分析森林图
- 按 cell_type / condition 分层
- 每亚组一行 + 总合并行

### Type 4: LASSO 基因效应量
- 横轴 = Cohen's d,纵轴 = Gene
- 95% CI via t 分布

## Code Template

完整模板见 [templates/test_forest.R](file:///d:/铁衰老 绝不重蹈覆辙/.trae/skills/forest-skill/templates/test_forest.R)

关键代码骨架(Fisher z-transform CI):
```r
ext <- read_csv("d:/铁衰老 绝不重蹈覆辙/L2/results/external_validation_results.csv")
stopifnot(nrow(ext) > 0, all(c("Spearman_rho","N_Valid","Spearman_p") %in% names(ext)))

fisher_ci <- function(rho, n, alpha=0.05) {
  z <- atanh(rho); se <- 1/sqrt(n-3); zc <- qnorm(1-alpha/2)
  list(lower=tanh(z-zc*se), upper=tanh(z+zc*se))
}

ext_plot <- ext %>% rowwise() %>%
  mutate(ci = list(fisher_ci(Spearman_rho, N_Valid)),
         lower = ci$lower, upper = ci$upper,
         sig = case_when(Spearman_p < 0.001 ~ "***",
                         Spearman_p < 0.01 ~ "**",
                         Spearman_p < 0.05 ~ "*", TRUE ~ "NS")) %>%
  ungroup() %>%
  mutate(Dataset = factor(Dataset, levels=rev(Dataset)))

p <- ggplot(ext_plot, aes(x=Spearman_rho, y=Dataset)) +
  geom_vline(xintercept=0, linetype="dashed", color="grey50", linewidth=0.4) +
  geom_segment(aes(x=lower, xend=upper, y=Dataset, yend=Dataset),
               color="grey50", linewidth=1.2, alpha=0.6) +
  geom_point(aes(color=Spearman_rho, size=FA_AUC), alpha=0.9) +
  geom_text(aes(label=sprintf("%.3f [%.2f,%.2f]%s", Spearman_rho, lower, upper, sig)),
            hjust=-0.15, size=3.2, fontface="bold") +
  scale_color_viridis_c(option="A", direction=-1, name="Spearman rho") +
  scale_size_continuous(range=c(3,8), name="FA AUC") +
  labs(x="Spearman Correlation [95% CI via Fisher z]", y=NULL) +
  theme_bw(base_size=10)
```

## Quality Checklist

- [ ] 95% CI 用 Fisher z-transform(Spearman)或 t 分布(Cohen's d)
- [ ] 显著性星号正确(`***`/`**`/`*`/NS)
- [ ] 参考线 `geom_vline(xintercept=0)` 虚线
- [ ] 点大小映射效应量或样本量
- [ ] 数值标签 `[lower, upper]` 格式
- [ ] 300 DPI PNG + Cairo::CairoPDF
- [ ] 真实数据断言

## Common Failure Modes

1. **CI 计算用错分布**: Spearman 用 Fisher z, Cohen's d 用 t, OR 用正态近似 → 不可混用
2. **Dataset 因子顺序反向**: 森林图从上到下应为逻辑顺序 → `levels=rev(Dataset)`
3. **NA 行未过滤**: Fisher z 对 NA/Inf 报错 → 先 `filter(!is.na(Spearman_rho))`
4. **标签越界**: `scale_x_continuous(limits=...)` 留 25% 余量给标签
5. **forestplot 包表格对齐**: 表格列宽与图不对齐 → 优先用 ggplot2 实现
