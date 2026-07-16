---
name: "nature-skill"
description: "Generate Nature/Science/Cell journal-style publication-ready scientific figures via R (ggsci + ggplot2 + patchwork) and Python (SciencePlots). Invoke when user asks for publication-quality plots, journal-style figures, multi-panel composites, NPG/AAAS/Lancet palettes, or figures meeting journal submission standards (300+ DPI, vector output, CVD-safe colors)."
---

# Nature Skill — Publication-Ready Journal-Style Plotting

## 1. When to Invoke

Invoke this skill IMMEDIATELY when the user asks for any of the following:
- "Nature 风格图" / "Science 风格图" / "Cell 风格图" / "publication-ready figure"
- Multi-panel composite figures (Fig 1A/B/C, Fig 2A-D, etc.)
- Journal submission figures (300+ DPI, TIFF/PDF/EPS, single/double column)
- NPG / AAAS / Lancet / NEJM / JAMA color palettes
- CVD-safe (colorblind-friendly) scientific palettes
- Volcano / MA / heatmap / UMAP / violin / boxplot / survival / forest / lollipop / bubble / Sankey / radar / UpSet / Manhattan / QQ / PCA / circos / oncoprint in journal style
- Reuse of existing project figures (figures/ directory) with style unification

Do NOT invoke for: pure data analysis without plotting, generic matplotlib defaults, or quick exploratory plots.

## 2. Environment Setup (MANDATORY preflight)

### R engine (primary — journal standard)
```r
# R 4.3.3 + packages already installed on this machine
# Verify before each run:
stopifnot(requireNamespace("ggplot2", quietly=TRUE))
stopifnot(requireNamespace("ggsci",  quietly=TRUE))   # NPG/AAAS/Lancet palettes
stopifnot(requireNamespace("patchwork", quietly=TRUE)) # multi-panel
stopifnot(requireNamespace("ggrepel",  quietly=TRUE))  # label repulsion
stopifnot(requireNamespace("ggpubr",   quietly=TRUE))  # stat comparisons
stopifnot(requireNamespace("Cairo",    quietly=TRUE))  # TrueType embedding
```

Available R packages (verified): ggplot2, ggsci, patchwork, ggrepel, ggpubr, ggsignif, scales, RColorBrewer, viridis, pheatmap, circlize, survival, survminer, Cairo, sysfonts, showtext, ggtext, ggupset, concaveman, isoband.

If a required R package is missing, STOP and report — do NOT silently fall back to base R.

### Python engine (secondary — for SciencePlots style sheets)
```powershell
# Dedicated venv (do NOT use conda base — has pyparsing conflict):
&D:\铁衰老 绝不重蹈覆辙\.plotenv\Scripts\python.exe -c "import scienceplots, matplotlib, pandas, numpy, seaborn; print('OK')"
```

The venv path is `D:\铁衰老 绝不重蹈覆辙\.plotenv\`. Use this exact interpreter for all Python-based plotting in this skill. Never use `python` from PATH (broken pyparsing in TraeAI-7 env).

## 3. Journal Style Specifications (HARD constraints)

### 3.1 Sizing (Nature / Cell / Science)
| Layout | Width | Use case |
|---|---|---|
| Single column | 89 mm (~3.5 in) | Single panel, narrow |
| 1.5 column | 120 mm (~4.7 in) | Medium panel |
| Double column | 183 mm (~7.2 in) | Multi-panel composite |

Heights: keep aspect ratio 4:3 or 3:2 by default; cap at 230 mm for full page.

### 3.2 Resolution & Format
- **Raster**: TIFF (LZW compression) or PNG at **300 DPI minimum** (line art: 600-1200 DPI)
- **Vector**: PDF (preferred) or EPS — use `cairo_pdf` for TrueType embedding
- **Never** output JPEG for figures (lossy compression rejected by journals)

### 3.3 Typography
- **Font family**: Arial or Helvetica (sans-serif); Times New Roman only for traditional journals
- **Font sizes**: axis labels 8-9 pt; tick labels 7-8 pt; titles 10-11 pt bold; panel tags (A/B/C) 14 pt bold
- **Embed fonts**: always use `cairo_pdf` or `Cairo::CairoPDF()` to embed TrueType

### 3.4 Color palettes (CVD-safe, journal-matched)
| Journal | ggsci scale | Hex (first 5) |
|---|---|---|
| Nature (NPG) | `scale_color_npg()` / `scale_fill_npg()` | #E64B35 #4DBBD5 #00A087 #3C5488 #F39B7F |
| Science (AAAS) | `scale_color_aaas()` | #3B4992 #EE0000 #008B45 #631879 #008280 |
| Lancet | `scale_color_lancet()` | #00468B #ED0000 #42B540 #0099B4 #925E9F |
| NEJM | `scale_color_nejm()` | #BC3C29 #0072B5 #E18727 #20854E #787DCC |
| JAMA | `scale_color_jama()` | #374E55 #DF8F44 #00A1D5 #B24745 #79AF97 |
| JCO | `scale_color_jco()` | #0073C2 #EFC000 #868686 #CD534C #7AA6DC |

**Continuous**: prefer `viridis` family (magma, plasma, inferno, cividis) — perceptually uniform + CVD-safe. Avoid jet/rainbow.

### 3.5 Theme baseline (R)
```r
theme_nature <- theme_bw(base_size = 9) +
  theme(
    panel.grid.major   = element_line(color = "grey92", linewidth = 0.25),
    panel.grid.minor   = element_blank(),
    panel.border       = element_rect(color = "black", linewidth = 0.6),
    axis.title         = element_text(face = "bold", size = 10),
    axis.text          = element_text(size = 8, color = "black"),
    axis.ticks         = element_line(color = "black", linewidth = 0.4),
    plot.title         = element_text(face = "bold", size = 11, hjust = 0),
    plot.tag           = element_text(face = "bold", size = 14),
    plot.tag.position  = "topleft",
    legend.position    = "right",
    legend.title       = element_text(face = "bold", size = 8),
    legend.text        = element_text(size = 7),
    legend.key.size    = unit(0.5, "cm"),
    strip.background   = element_rect(fill = "grey95", color = "black", linewidth = 0.4),
    strip.text         = element_text(face = "bold", size = 9)
  )
```

This matches the established `theme_composite` in `plot_composite.R` — reuse, do not redefine.

### 3.6 Theme baseline (Python)
```python
import matplotlib.pyplot as plt
import scienceplots  # MUST import before setting style

plt.style.use(['science', 'nature'])  # Nature sans-serif + column width
# For Chinese text: plt.style.use(['science', 'nature', 'no-latex', 'CJK+'])
```

If LaTeX is not installed, append `'no-latex'` to the style list to avoid `LaTeX not found` errors.

## 4. Multi-Panel Composition (patchwork)

```r
library(patchwork)
combined <- (pA | pB) / (pC | pD) +
  plot_layout(widths = c(1, 1.2), heights = c(1, 1)) +
  plot_annotation(
    tag_levels = 'A',                     # auto A, B, C, D...
    tag_suffix = ')',
    title = 'Figure N. Main caption title.',
    caption = 'n=...; statistical test: ...; data are mean ± SEM.'
  ) &
  theme(plot.tag = element_text(face = 'bold', size = 14))
```

Save:
```r
ggsave("FigN_Composite_xxx.png", combined,
       width = 183, height = 120, units = "mm", dpi = 300)
ggsave("FigN_Composite_xxx.pdf", combined,
       width = 183, height = 120, units = "mm",
       device = Cairo::CairoPDF)
```

**Note**: Use `Cairo::CairoPDF` (from the `Cairo` package) for TrueType embedding — the built-in `cairo_pdf` device fails on this machine (`winCairo.dll` dependency issue with anaconda R). Always load `library(Cairo)` before plotting.

## 5. Real-Data Workflow (MANDATORY — no synthetic data)

All figures must read from real project data. The following datasets are verified:

| Figure | Data source | Read function |
|---|---|---|
| Fig1 UMAP + violin + bar | `figures/meta_with_umap.csv` | `readr::read_csv()` |
| Fig2 model perf + LASSO + TCM | `L4/results_v10_minibatch/model_performance_v*.csv`, `L2/results/ciri_ferroaging_lasso_candidates.csv`, `L4/results_v10_minibatch/tcm_top_candidates_v*.csv` | `read_csv()` |
| Fig3 DE + external + PPI | `L1/...`, `L2/results/...` | `read_csv()` / `read_tsv()` |
| Fig4 chemistry + microglia | `L3/...`, `L2/...` | `read_csv()` |
| Fig5 SCISSOR | `figures/scissor_umap_metadata.csv` | `read_csv()` |
| Fig6 CellChat | `L2/results/...` | `readRDS()` / `read_csv()` |

**NEVER fabricate data**. If a data file is missing, STOP and report the missing path. Do NOT substitute `rnorm()` or random values.

## 6. Statistical Annotation Rules

- Two-group comparison: Wilcoxon rank-sum (unpaired) or paired t-test (paired)
- Multi-group: Kruskal-Wallis with Dunn's post-hoc, or ANOVA with Tukey
- Use `ggpubr::stat_compare_means()` with `method=` explicitly set
- Significance symbols: `****` P<0.0001, `***` P<0.001, `**` P<0.01, `*` P<0.05, `NS` P≥0.05
- Always show exact p-values when N<30; otherwise symbol notation is acceptable
- Survival: log-rank test via `survminer::surv_pvalue()`

## 7. Export Checklist (run before declaring done)

- [ ] Vector PDF saved with `cairo_pdf` (TrueType embedded)
- [ ] PNG/TIFF at ≥300 DPI saved
- [ ] Width matches journal column (89 / 120 / 183 mm)
- [ ] Font: Arial/Helvetica, sizes ≥7 pt
- [ ] Color palette is CVD-safe (NPG/viridis — not jet)
- [ ] Panel tags A/B/C bold 14 pt
- [ ] No truncated labels (use `ggrepel` for crowded text)
- [ ] Legend does not overlap plot area
- [ ] Axis ticks point inward or outward consistently
- [ ] Figure legend written to `Figure_Legends.txt` with full statistical details

## 8. Common Failure Modes & Fixes

| Symptom | Cause | Fix |
|---|---|---|
| `LaTeX not found` (Python) | SciencePlots defaults to LaTeX rendering | `plt.style.use(['science','nature','no-latex'])` |
| Chinese text as boxes (R) | No CJK font registered | `showtext::font_add_google('Noto Sans SC')` + `showtext_auto()` |
| PDF fonts not embedded | Default pdf() device | Use `cairo_pdf()` or `Cairo::CairoPDF()` |
| patchwork legend clipping | `plot_layout` guide_area missing | Add `guides='collect'` in `plot_layout()` |
| ggrepel labels overlap | Too many points | Increase `max.overlaps=20`, reduce `size=2`, or filter top N |
| ggsci palette too few colors | Categorical N>10 | Switch to `scale_color_npg()` + manual extension, or use `viridis_d` |

## 9. Integration with Existing Project

This project's `plot_composite.R` / `plot_fig1_singlecell.R` ... `plot_fig6_cellchat.R` are the canonical R plotting scripts. New figures should:
1. Reuse `theme_composite` from `plot_composite.R` (do NOT redefine).
2. Write to `figures/` with naming convention `Fig{N}{Letter}_{description}.png/pdf`.
3. Append legend to `figures/Figure_Legends.txt` in the established format.
4. Use `ggsci` palettes consistent with existing Fig1-Fig6.

## 10. Reference Templates Location

- R templates: `D:/R语言绘图模板/` (1335+ templates, 50+ chart types — verify by `LS` before citing)
- Project templates: `plot_composite.R`, `plot_fig1_singlecell.R`, ..., `plot_fig6_cellchat.R`

When reusing a template, ALWAYS read it first with `Read` and adapt the data path — never copy-paste blindly.
