---
name: "survival-skill"
description: "生存分析可视化技能。生成 KM 曲线、风险表、log-rank 检验、cutoff 分析。Invoke when user asks for Kaplan-Meier plot, survival analysis, log-rank test, or optimal cutoff survival visualization."
---

# Survival 生存分析可视化技能

## When to Invoke

当用户需要:
- Kaplan-Meier 生存曲线
- log-rank 检验
- 风险表(risk table)
- 最佳 cutoff 分析(surv_cutpoint)
- Cox 风险比例森林图

## Environment Setup

```r
suppressPackageStartupMessages({
  library(survival); library(survminer); library(ggplot2)
  library(dplyr); library(readr); library(Cairo)
})
stopifnot(requireNamespace("survival", quietly=TRUE),
          requireNamespace("survminer", quietly=TRUE))
```

## 真实数据源

**项目当前无患者生存数据**(铁衰老项目是细胞/动物实验 + 单细胞测序)。

可选真实数据源:
1. **survival::lung**(R 内置真实 NCCTG 肺癌临床数据, North Central Cancer Treatment Group)
   - 228 例患者,time(天) + status(1=alive,2=dead) + sex + age + ECOG
   - 这是真实临床数据,非模拟
2. **TCGA 公开数据**(需 TCGAbiolinks 下载,本技能仅写规范)

**禁止**:不得模拟生存时间或事件。无真实生存数据时,测试脚本明确标注"用 survival::lung 真实 NCCTG 数据演示"。

## Visualization Specifications

### Type 1: Kaplan-Meier 曲线(主用)
- `survfit(Surv(time, status) ~ group)` 拟合
- `ggsurvplot()` 绘图
- 必含元素:
  - 两条(或多条)生存曲线 + 95% CI 阴影
  - at-risk table(底部)
  - log-rank p 值
  - 中位生存时间虚线
  - 图例(group + n)
- 配色: `c("#00599B","#D55E00")`(CVD-safe)

### Type 2: Cox 森林图
- `coxph(Surv(time, status) ~ var1 + var2 + ...)` 拟合
- `ggforest()` 绘制 HR + 95% CI

### Type 3: 最佳 Cutoff
- `surv_cutpoint()` 找最佳切割点
- `survminer::ggsurvplot()` 比较 high vs low
- 警告: cutoff 分析有偏倚,需 Bootstrap 校正

### Type 4: 累积风险
- `ggsurvplot(fun="cumhaz")` 累积风险曲线

## Code Template

完整模板见 [templates/test_survival.R](file:///d:/铁衰老 绝不重蹈覆辙/.trae/skills/survival-skill/templates/test_survival.R)

关键代码骨架(用 survival::lung 真实数据):
```r
# 真实 NCCTG 肺癌数据(R survival 包内置)
data(lung, package="survival")
lung <- lung %>% mutate(sex = factor(sex, levels=c(1,2), labels=c("Male","Female")))
stopifnot(nrow(lung) > 0, all(c("time","status","sex") %in% names(lung)))

fit <- survfit(Surv(time, status) ~ sex, data=lung)
logrank <- survdiff(Surv(time, status) ~ sex, data=lung)
p_val <- 1 - pchisq(logrank$chisq, length(logrank$n)-1)

p_surv <- ggsurvplot(fit, data=lung,
                     pval=TRUE, pval.method=TRUE,
                     conf.int=TRUE,
                     risk.table=TRUE,
                     risk.table.col="strata",
                     risk.table.height=0.25,
                     legend.labs=c("Male","Female"),
                     legend.title="Sex",
                     xlab="Time (days)", ylab="Survival Probability",
                     title="NCCTG Lung Cancer Survival by Sex",
                     palette=c("#00599B","#D55E00"),
                     ggtheme=theme_bw(base_size=10),
                     surv.median.line="hv")

# 保存( survminer 返回列表,需用 print + file)
png(png_path, width=8, height=8, units="in", res=300, bg="white")
print(p_surv, newpage=TRUE)
dev.off()

# Cairo PDF
Cairo::CairoPDF(pdf_path, width=8, height=8)
print(p_surv, newpage=TRUE)
dev.off()
```

## Quality Checklist

- [ ] 300 DPI PNG + Cairo::CairoPDF
- [ ] 真实生存数据(项目无 → 用 survival::lung)
- [ ] at-risk table 必含
- [ ] log-rank p 值标注
- [ ] 95% CI 阴影
- [ ] 中位生存时间虚线
- [ ] CVD-safe 配色
- [ ] `ggsurvplot` 返回列表 → `print()` 到设备

## Common Failure Modes

1. **status 编码错误**: survival::lung 用 1=alive,2=dead;部分数据用 0/1 → 确认编码
2. **ggsurvplot 保存失败**: 返回列表不能用 `ggsave()` → 必须用 `print(p, newpage=TRUE)` + `png/pdf`+`dev.off()`
3. **risk.table.height 过大**: 默认 0.5 挤压主图 → 设 0.25
4. **pval=TRUE 不显示**: 需同时 `pval.method=TRUE`
5. **cutoff 分析过拟合**: 单样本 cutoff 有偏倚 → 报告 Bootstrap 校正结果
6. **NA 行未删**: `na.omit()` 前置
