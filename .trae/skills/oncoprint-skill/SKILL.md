---
name: "oncoprint-skill"
description: "OncoPrint 瀑布图可视化技能。生成突变景观图、TMB 图、互斥共发生分析。Invoke when user asks for OncoPrint, mutation landscape, waterfall plot, or TMB visualization."
---

# OncoPrint 瀑布图可视化技能

## When to Invoke

当用户需要:
- OncoPrint /瀑布图(基因×样本突变矩阵)
- TMB(tumor mutational burden)图
- 互斥/共发生分析(Fisher 精确检验)
- 突变签名贡献图
- maftools 标准图集

## Environment Setup

```r
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(readr); library(tidyr)
  library(viridis); library(Cairo)
})
# ComplexHeatmap / maftools 可选(若已安装)
has_ch  <- requireNamespace("ComplexHeatmap", quietly=TRUE)
has_maf <- requireNamespace("maftools", quietly=TRUE)
if (has_ch)  suppressPackageStartupMessages(library(ComplexHeatmap))
if (has_maf) suppressPackageStartupMessages(library(maftools))
```

## 真实数据源

**项目当前无 MAF / 突变数据**(铁衰老项目是转录组 + 单细胞 + 化合物,无肿瘤体细胞突变调用)。

可选真实数据源:
1. **TCGA 公开 MAF**(通过 TCGAbiolinks / maftools::tcgaOmicsData 下载)
2. **maftools::tcgaOmicsData**(maftools 包内置 TCGA LAML 真实数据)
3. **项目免疫浸润 immune_infiltration.csv**(无突变 → 不适用)

**禁止**:不得模拟突变调用。无真实 MAF 时:
- SKILL.md 写完整规范
- 测试脚本仅加载包验证可用性 + 打印说明,不生成假瀑布图
- 等待用户提供真实 MAF 文件

## Visualization Specifications

### Type 1: OncoPrint(标准瀑布图)
- 行 = 基因(Top 20-30 by mutation frequency),列 = 样本
- 颜色编码: Missense_Mutation / Frame_Shift_Del / Nonsense_Mutation / Splice_Site / Multi_Hit
- 顶部 annotation: TMB / gender / stage(如有)
- 右侧 annotation: 每基因突变频率条形图
- 排序: 按 mutation frequency 降序,样本按 TMB 降序
- `ComplexHeatmap::oncoPrint()` 是标准实现

### Type 2: TMB 图
- 横轴 = 样本(按 TMB 排序),纵轴 = TMB count
- 高 TMB(>10 mut/Mb)/ 低 TMB 阈值线

### Type 3: 互斥/共发生
- Fisher 精确检验 / DISCOVER 算法
- 上三角 = p 值,下三角 = odds ratio
- 热图可视化

### Type 4: 突变签名
- 横轴 = 96 三核苷酸上下文,纵轴 = 签名
- 堆叠柱状图

### Type 5: maftools 标准图集
- `plotmafSummary()`: variants per sample / variant classification / variant type / SNV class
- `oncoplot()`: Top genes 瀑布图
- `titv()`: Transition/Transversion 分类
- `lollipopPlot()`: 蛋白结构域 + 突变位置棒棒糖

## Code Template

完整模板见 [templates/test_oncoprint.R](file:///d:/铁衰老 绝不重蹈覆辙/.trae/skills/oncoprint-skill/templates/test_oncoprint.R)

关键代码骨架(需真实 MAF):
```r
# 标准流程(需真实 MAF 文件)
# maf <- read.maf(maf="path/to/TCGA_LAML.maf.gz", clinicalData="clinical.csv")
# plotmafSummary(maf)
# oncoplot(maf, top=20)
# titv(maf)
# lollipopPlot(maf, gene="TP53")

# OncoPrint via ComplexHeatmap(需 mutation matrix)
# mat <- read.csv("mutation_matrix.csv")  # 行=基因 列=样本 值=mutation type
# oncoPrint(mat, alter_fun=list(
#   Missense_Mutation = function(x,y,w,h) grid.rect(x,y,w,h*0.9, gp=gpar(fill="#00BFFF")),
#   Frame_Shift_Del   = function(x,y,w,h) grid.rect(x,y,w,h*0.9, gp=gpar(fill="#FF6347")),
#   Nonsense_Mutation = function(x,y,w,h) grid.rect(x,y,w,h*0.9, gp=gpar(fill="#FF0000"))
# ), col=list(
#   Missense_Mutation="#00BFFF", Frame_Shift_Del="#FF6347", Nonsense_Mutation="#FF0000"
# ))
```

## Quality Checklist

- [ ] 300 DPI PNG + Cairo::CairoPDF
- [ ] 真实 MAF 文件(非模拟)
- [ ] 突变类型颜色编码统一(Missense=蓝 / Frameshift=红 / Nonsense=深红 / Splice=绿 / Multi=黑)
- [ ] 样本按 TMB 或 mutation count 排序
- [ ] 基因按 mutation frequency 降序
- [ ] 顶部/右侧 annotation 条
- [ ] TMB 阈值线(10 mut/Mb)

## Common Failure Modes

1. **无 MAF 数据**: 项目无体细胞突变 → SKILL.md 写规范,测试脚本不生成假图
2. **MAF 列名错误**: 必需列 Hugo_Symbol / Variant_Classification / Tumor_Sample_Barcode
3. **oncoPrint 矩阵稀疏**: 大量 0 → 用 sparse matrix
4. **样本顺序乱**: 按 mutation count 排序,非字母序
5. **maftools 版本兼容**: v2.10+ API 变化 → 检查 `packageVersion("maftools")`
6. **基因标签重叠**: Top 20-30,fontsize_row ≤ 8

## 项目适用性声明

铁衰老项目当前无 MAF / 体细胞突变调用数据。本技能:
- SKILL.md 提供完整规范,供未来接入 TCGA / ICGC / 项目自有 WES 数据时使用
- 测试脚本仅验证 maftools/ComplexHeatmap 包加载可用性,不生成假瀑布图
- 用户提供真实 MAF 后,按本规范生成标准 OncoPrint
