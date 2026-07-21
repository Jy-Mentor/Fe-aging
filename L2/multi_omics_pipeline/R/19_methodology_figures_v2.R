# ===========================================================================
# 19_methodology_figures_v2.R
# 铁衰老项目 - 方法学配图 v2 (基于真实 CSV 数据)
# 三图组合:
#   Fig1A: 跨组学通路轴匹配率条形图 (按证据等级着色)
#   Fig1B: KEGG 共享通路 cross_omics_score Top 15 (横向条形图)
#   Fig1C: SAT1-多胺轴 7 代谢物 3w vs 59w log2FC (带显著性标注)
# 数据来源: L2 pipeline 输出 (Table1_cross_omics_evidence.csv,
#          cross_omics_shared_pathways.csv, metabolite_pairwise_comparison.csv,
#          cross_omics_axis_table.csv)
# ===========================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(patchwork)
  library(stringr)
})

# --------------------------------------------------------------------------
# 路径配置
# --------------------------------------------------------------------------
BASE_DIR <- "D:/铁衰老 绝不重蹈覆辙/L2/multi_omics_pipeline"
OUT_DIR  <- file.path(BASE_DIR, "output", "methodology_figures")
FIG_DIR  <- OUT_DIR  # 直接输出到 methodology_figures 目录
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

# --------------------------------------------------------------------------
# 通用主题: 学术出版物风格 (Nature/Cell 标准)
# --------------------------------------------------------------------------
theme_pub <- function(base_size = 11) {
  theme_classic(base_size = base_size) +
    theme(
      plot.title       = element_text(hjust = 0.5, face = "bold", size = base_size + 1),
      plot.subtitle    = element_text(hjust = 0.5, size = base_size - 1, color = "gray30"),
      axis.title       = element_text(face = "bold"),
      axis.text.x      = element_text(color = "black"),
      axis.text.y      = element_text(color = "black"),
      axis.line        = element_line(linewidth = 0.6, color = "black"),
      axis.ticks       = element_line(linewidth = 0.6, color = "black"),
      legend.position  = "right",
      legend.title     = element_text(face = "bold"),
      plot.margin      = margin(8, 12, 8, 8)
    )
}

# ===========================================================================
# Fig1A: 跨组学通路轴匹配率条形图
# ===========================================================================
cat("[Fig1A] Reading cross-omics evidence table...\n")
tab1_path <- file.path(OUT_DIR, "tables", "Table1_cross_omics_evidence.csv")
stopifnot(file.exists(tab1_path))
tab1 <- read.csv(tab1_path, stringsAsFactors = FALSE)

# 提取匹配率分子/分母 (清理尾部空格, 防止 NA)
tab1$Match.Rate  <- trimws(tab1$Match.Rate)
tab1$match_frac  <- trimws(sub("\\s*\\(.*\\)$", "", tab1$Match.Rate))
tab1$match_num   <- as.numeric(sub("/.*", "", tab1$match_frac))
tab1$match_den   <- as.numeric(sub(".*/", "", tab1$match_frac))
tab1$match_pct   <- 100 * tab1$match_num / tab1$match_den
if (any(is.na(tab1$match_pct))) {
  stop("Failed to parse match rates:",
       paste(tab1$Match.Rate[is.na(tab1$match_pct)], collapse = "; "))
}

# 排序: 按匹配率降序
tab1 <- tab1[order(-tab1$match_pct), ]
tab1$axis_label <- paste0(tab1$Pathway.Axis, " (", tab1$Driver.Gene, ")")
tab1$axis_label <- factor(tab1$axis_label, levels = rev(tab1$axis_label))

# 证据等级颜色 (CVD-safe Okabe-Ito 调色板)
evidence_colors <- c(
  "Moderate" = "#D55E00",  # orange
  "Weak"     = "#56B4E9",  # sky blue
  "Strong"   = "#009E73"   # green (none in data, reserved)
)

p1A <- ggplot(tab1, aes(x = axis_label, y = match_pct, fill = Evidence)) +
  geom_col(width = 0.7, color = "black", linewidth = 0.4) +
  geom_text(aes(label = sprintf("%d/%d (%.0f%%)",
                                 match_num, match_den, match_pct)),
            hjust = -0.15, size = 3.2, fontface = "bold") +
  geom_hline(yintercept = 70, linetype = "dashed", color = "gray50", linewidth = 0.5) +
  geom_hline(yintercept = 50, linetype = "dotted",  color = "gray50", linewidth = 0.5) +
  annotate("text", x = 0.4, y = 72, label = "Strong >=70%", size = 2.8, hjust = 0,
           color = "gray30") +
  annotate("text", x = 0.4, y = 52, label = "Moderate >=50%", size = 2.8, hjust = 0,
           color = "gray30") +
  scale_fill_manual(values = evidence_colors,
                    name   = "Evidence",
                    breaks = c("Strong", "Moderate", "Weak")) +
  scale_y_continuous(limits = c(0, 100), expand = c(0, 0)) +
  coord_flip() +
  labs(title    = "Cross-omics pathway axis validation",
       subtitle = "ST001637 aging (3w vs 59w) metabolite direction match rate",
       x        = "Pathway axis (driver gene)",
       y        = "Direction match rate (%)") +
  theme_pub(base_size = 10) +
  theme(legend.position = "right",
        plot.title      = element_text(size = 12),
        axis.text.y     = element_text(size = 9))

# ===========================================================================
# Fig1B: KEGG 共享通路 cross_omics_score Top 15
# ===========================================================================
cat("[Fig1B] Reading KEGG shared pathways...\n")
kegg_path <- file.path(BASE_DIR, "output", "kegg_pathway_integration",
                        "tables", "cross_omics_shared_pathways.csv")
stopifnot(file.exists(kegg_path))
kegg <- read.csv(kegg_path, stringsAsFactors = FALSE)

# 清理 pathway_name (去除物种后缀)
kegg$pathway_short <- sub(" - Mus musculus \\(house mouse\\)", "",
                          kegg$pathway_name)

# Top 15 by cross_omics_score
kegg_top <- kegg %>%
  arrange(desc(cross_omics_score)) %>%
  head(15) %>%
  mutate(label = sprintf("mmu%s | %s", pathway_code, pathway_short))

kegg_top$label <- factor(kegg_top$label, levels = rev(kegg_top$label))

p1B <- ggplot(kegg_top, aes(x = label, y = cross_omics_score)) +
  geom_col(width = 0.7, fill = "#0072B2", color = "black", linewidth = 0.4) +
  geom_text(aes(label = sprintf("%d (%dG, %dM)",
                                 cross_omics_score, n_genes, n_metabolites)),
            hjust = -0.1, size = 3.0) +
  scale_y_continuous(expand = c(0, 0),
                     limits = c(0, max(kegg_top$cross_omics_score) * 1.35)) +
  coord_flip() +
  labs(title    = "Top 15 shared KEGG pathways (FA-96 genes x metabolites)",
       subtitle = "70 pathways with both gene and metabolite evidence",
       x        = "KEGG pathway",
       y        = "Cross-omics score (gene + metabolite count)") +
  theme_pub(base_size = 10) +
  theme(axis.text.y = element_text(size = 8),
        plot.title  = element_text(size = 12))

# ===========================================================================
# Fig1C: SAT1-多胺轴 7 代谢物 3w vs 59w log2FC
# ===========================================================================
cat("[Fig1C] Reading SAT1-polyamine axis metabolites...\n")
axis_path <- file.path(BASE_DIR, "output", "cross_omics_integration",
                        "tables", "cross_omics_axis_table.csv")
stopifnot(file.exists(axis_path))
axis_df <- read.csv(axis_path, stringsAsFactors = FALSE)

# 筛选 SAT1-polyamine 轴
sat1_axis <- axis_df %>%
  filter(axis_name == "SAT1-polyamine") %>%
  select(metabolite, display_name, log2FC_aging, p_adj_aging,
         direction_aging, expected_aging, match_aging)

# 按显示名分组 (重复代谢物保留最显著的)
sat1_plot <- sat1_axis %>%
  group_by(display_name) %>%
  slice_min(p_adj_aging, n = 1) %>%
  ungroup() %>%
  arrange(log2FC_aging)

sat1_plot$sig <- ifelse(sat1_plot$p_adj_aging < 0.001, "***",
                  ifelse(sat1_plot$p_adj_aging < 0.01, "**",
                    ifelse(sat1_plot$p_adj_aging < 0.05, "*", "ns")))

# 标签: 代谢物名 + 显著性
sat1_plot$label <- paste0(sat1_plot$display_name,
                          ifelse(sat1_plot$sig == "ns", "",
                                 sprintf(" (%s)", sat1_plot$sig)))
sat1_plot$label <- factor(sat1_plot$label, levels = rev(sat1_plot$label))

# 颜色: match=TRUE 绿色, FALSE 灰色
sat1_plot$match_label <- ifelse(sat1_plot$match_aging,
                                 "Expected direction",
                                 "Opposite to expected")

p1C <- ggplot(sat1_plot, aes(x = label, y = log2FC_aging, fill = match_label)) +
  geom_col(width = 0.7, color = "black", linewidth = 0.4) +
  geom_hline(yintercept = 0, color = "black", linewidth = 0.6) +
  geom_text(aes(label = sprintf("log2FC=%.2f, p.adj=%.2g",
                                 log2FC_aging, p_adj_aging)),
            nudge_y = ifelse(sat1_plot$log2FC_aging >= 0, 0.06, -0.06),
            size = 2.8, vjust = 0.4, hjust = 0) +
  scale_fill_manual(values = c("Expected direction"     = "#009E73",
                                "Opposite to expected" = "#CC79A7"),
                    name = "Direction match") +
  coord_flip(clip = "off") +
  labs(title    = "SAT1 / polyamine axis: aging (3w -> 59w) metabolite changes",
       subtitle = "SAT1 activation -> spermine/spermidine acetylation -> depletion (expected down)",
       x        = "Metabolite",
       y        = expression(log[2]~"fold change (59w / 3w)")) +
  theme_pub(base_size = 10) +
  theme(legend.position = "right",
        axis.text.y     = element_text(size = 9),
        plot.title      = element_text(size = 12),
        plot.margin     = margin(8, 60, 8, 8))

# ===========================================================================
# 组合三图 (patchwork): Fig1A | Fig1B | Fig1C  (竖排)
# ===========================================================================
cat("[Combine] Assembling Figure 1 (A/B/C)...\n")
fig1 <- (p1A + p1B + p1C) +
  plot_layout(nrow = 3, heights = c(1.0, 1.0, 0.9)) +
  plot_annotation(
    title    = "Figure 1. Cross-omics integration of iron-aging signature",
    caption  = "Data: ST001637 (metabolomics) x GSE233815 (snRNA-seq) x KEGG REST API",
    tag_levels = "A",
    theme = theme_pub(base_size = 11)
  ) &
  theme(plot.tag = element_text(face = "bold", size = 14))

# 输出
out_pdf <- file.path(FIG_DIR, "Figure1_cross_omics_integration.pdf")
out_png <- file.path(FIG_DIR, "Figure1_cross_omics_integration.png")
ggsave(out_pdf, fig1, width = 10, height = 14, units = "in", device = cairo_pdf)
ggsave(out_png, fig1, width = 10, height = 14, units = "in", dpi = 300,
       bg = "white")

cat(sprintf("[Done] Figure 1 saved:\n  %s\n  %s\n", out_pdf, out_png))

# ===========================================================================
# 单独保存各子图 (供期刊投稿)
# ===========================================================================
ggsave(file.path(FIG_DIR, "Figure1A_pathway_axis_match_rate.pdf"),
       p1A, width = 8, height = 5, units = "in", device = cairo_pdf)
ggsave(file.path(FIG_DIR, "Figure1B_kegg_top15.pdf"),
       p1B, width = 8, height = 5, units = "in", device = cairo_pdf)
ggsave(file.path(FIG_DIR, "Figure1C_sat1_polyamine.pdf"),
       p1C, width = 8, height = 4.5, units = "in", device = cairo_pdf)

cat("[Done] Sub-panels saved (Figure1A/B/C).\n")
