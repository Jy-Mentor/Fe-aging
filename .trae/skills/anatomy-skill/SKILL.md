---
name: "anatomy-skill"
description: "器官解剖图可视化技能。用 gganatogram 生成人体/小鼠器官表达热图、免疫浸润器官分布图。Invoke when user asks for anatomical figure, organ-level expression visualization, or gganatogram plot."
---

# Anatomy 器官解剖图可视化技能

## When to Invoke

当用户需要:
- 器官级表达热图(gganatogram)
- 免疫细胞器官分布
- 组织特异性基因表达
- 解剖学示意图 + 数据映射

## Environment Setup

```r
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(readr); library(tidyr)
  library(viridis); library(Cairo)
})
# gganatogram 可选(若已安装)
has_gganato <- requireNamespace("gganatogram", quietly=TRUE)
if (has_gganato) suppressPackageStartupMessages(library(gganatogram))
```

## 真实数据源(项目)

| 文件 | 路径 | 用途 |
|------|------|------|
| 免疫浸润 | `d:/铁衰老 绝不重蹈覆辙/L2/results/immune_infiltration.csv` | 免疫细胞评分 |
| 免疫细胞评分 | `d:/铁衰老 绝不重蹈覆辙/L2/results/immune_cell_scores_GSE104036.csv` | 每样本免疫评分 |
| 免疫 FA 相关 | `d:/铁衰老 绝不重蹈覆辙/L2/results/immune_ferroaging_correlation_GSE104036.csv` | 免疫-铁衰老相关 |
| 炎症表达 | `d:/铁衰老 绝不重蹈覆辙/L2/results/inflammation_expression_GSE104036.csv` | 炎症基因表达 |

**禁止**:不得模拟器官表达值。

## Visualization Specifications

### Type 1: gganatogram 人体器官图(主用)
- `gganatogram::anatogram_img("male")` 或 `"female"`
- 器官 = brain / heart / liver / lung / kidney / etc.
- 器官颜色 = 表达值(连续, viridis)
- 器官大小 = 效应量(如有)

### Type 2: 器官-基因热图
- 横轴 = 基因,纵轴 = 器官
- `geom_tile` + viridis 配色

### Type 3: 免疫细胞器官分布
- 项目数据是 GSE104036(脑缺血再灌注),非多器官数据
- 可视化:大脑不同区域(hippocampus / cortex / striatum)免疫评分
- 用 gganatogram 的 brain 子图

### Type 4: 物种映射
- 大鼠/小鼠数据 → 用 `anatogram_img("mice")`
- 人类直系同源 → 用 `anatogram_img("male")`

## Code Template

完整模板见 [templates/test_anatomy.R](file:///d:/铁衰老 绝不重蹈覆辙/.trae/skills/anatomy-skill/templates/test_anatomy.R)

关键代码骨架(需 gganatogram 已安装):
```r
immune_scores <- read_csv("d:/铁衰老 绝不重蹈覆辙/L2/results/immune_cell_scores_GSE104036.csv")
stopifnot(nrow(immune_scores) > 0)

if (!requireNamespace("gganatogram", quietly=TRUE)) {
  message("gganatogram not installed. Install with: ",
          "devtools::install_github('jespermaag/gganatogram')")
  message("Generating organ-gene heatmap as fallback.")
  # Fallback: organ × cell type heatmap
  # ...
} else {
  # 真实 gganatogram 流程
  # 将免疫评分映射到器官(本项目主要在 brain,需自定义组织-器官映射)
  organ_df <- immune_scores %>%
    pivot_longer(-sample_id, names_to="cell_type", values_to="score") %>%
    group_by(cell_type) %>%
    summarise(mean_score = mean(score, na.rm=TRUE), .groups="drop") %>%
    mutate(organ = "brain",  # 项目数据主要来自脑组织
           value = mean_score)

  fig <- gganatogram::gganatogram(data=organ_df, fill="value",
                                  organism="human", sex="male",
                                  fill_palette=viridis(100)) +
    labs(title="Immune Cell Score — Brain (GSE104036)") +
    theme_bw(base_size=10)
}
```

## Quality Checklist

- [ ] 300 DPI PNG + Cairo::CairoPDF
- [ ] 真实免疫/表达数据(非模拟)
- [ ] gganatogram 未装时回退到器官-基因热图
- [ ] CVD-safe 配色(viridis)
- [ ] 器官标签清晰
- [ ] 物种匹配(大鼠数据用 mice anatogram 或映射到 human)

## Common Failure Modes

1. **gganatogram 未安装**: GitHub 包,CRAN 无 → 回退到 geom_tile 热图
2. **器官名不匹配**: gganatogram 用特定器官名(brain / heart / liver_Kupffer_cell 等)→ 查 `data(organGroups)`
3. **项目单器官(脑)**: 数据全在 brain → gganatogram 仅 brain 着色,其余灰
4. **大鼠→人类映射**: 用 rat_to_human_ortholog_mygene.csv 先映射基因
5. **数值范围差异大**: organ 值归一化到 0-1
6. **anatogram_img 性别**: male/female 器官略不同 → 明确指定

## 项目适用性声明

铁衰老项目数据主要来自:
- GSE61616 / GSE104036 / GSE16561 等:大鼠 MCAO 模型(脑组织)
- GSE233815:小鼠 snRNA-seq(脑)
- 无多器官数据

因此 gganatogram 主要用于:
- 大鼠/小鼠脑解剖图 + 区域表达(hippocampus / cortex / striatum)
- 免疫细胞器官分布(若有多器官数据)
- 未来人类临床数据扩展时用 human anatogram
