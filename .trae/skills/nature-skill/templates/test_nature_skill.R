##############################################################################
# nature-skill 功能验证脚本
# 目的：验证 R 引擎使用真实数据生成 Nature 风格 publication-ready 图
# 数据源：figures/meta_with_umap.csv (真实 snRNA-seq UMAP 元数据)
# 输出：figures/skill_test/nature_test_umap_violin.png/pdf
##############################################################################

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggsci)
  library(patchwork)
  library(ggrepel)
  library(readr)
  library(dplyr)
  library(Cairo)
})

# ---- 0. 环境自检 ----
stopifnot(requireNamespace("ggplot2", quietly=TRUE))
stopifnot(requireNamespace("ggsci",  quietly=TRUE))
stopifnot(requireNamespace("patchwork", quietly=TRUE))
stopifnot(requireNamespace("Cairo",   quietly=TRUE))
cat("[OK] R environment verified\n")

# ---- 1. 加载真实数据 ----
meta_path <- "d:/铁衰老 绝不重蹈覆辙/figures/meta_with_umap.csv"
stopifnot(file.exists(meta_path))
meta <- read_csv(meta_path, show_col_types = FALSE)
cat(sprintf("[OK] Loaded %d cells from real data\n", nrow(meta)))

score_col <- intersect(c("AddModuleScore_FA96","FA_96_UCell","AddModuleScore_FA95"), colnames(meta))[1]
cat(sprintf("[OK] Using score column: %s\n", score_col))

# ---- 2. Nature 风格主题 ----
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

# ---- 3. Panel A: UMAP by cell type (NPG palette) ----
pA <- ggplot(meta, aes(x = UMAP_1, y = UMAP_2, color = cell_class)) +
  geom_point(size = 0.3, alpha = 0.7) +
  scale_color_npg(name = "Cell class") +
  guides(color = guide_legend(override.aes = list(size = 2.5))) +
  labs(x = "UMAP 1", y = "UMAP 2", title = "Cell type atlas") +
  theme_nature

# ---- 4. Panel B: Violin of ferroaging score by condition ----
pB <- ggplot(meta, aes(x = Condition, y = .data[[score_col]], fill = Condition)) +
  geom_violin(alpha = 0.6, linewidth = 0.4) +
  geom_boxplot(width = 0.12, outlier.size = 0.3, fill = "white", linewidth = 0.3) +
  scale_fill_npg(name = "Condition") +
  labs(x = NULL, y = "Ferroaging score (FA96)",
       title = "FA score by condition") +
  theme_nature +
  theme(legend.position = "none")

# ---- 5. Panel C: Bar of mean FA ± SEM by cell class ----
agg <- meta %>%
  group_by(cell_class, Condition) %>%
  summarise(mean_fa = mean(.data[[score_col]], na.rm = TRUE),
            sem_fa  = sd(.data[[score_col]], na.rm = TRUE) / sqrt(n()),
            .groups = "drop")

pC <- ggplot(agg, aes(x = cell_class, y = mean_fa, fill = Condition)) +
  geom_col(position = position_dodge(0.8), width = 0.7) +
  geom_errorbar(aes(ymin = mean_fa - sem_fa, ymax = mean_fa + sem_fa),
                position = position_dodge(0.8), width = 0.3, linewidth = 0.3) +
  scale_fill_npg(name = "Condition") +
  labs(x = NULL, y = "Mean FA96 score ± SEM",
       title = "FA score by cell class") +
  theme_nature +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ---- 6. 组合 ----
combined <- (pA | pB) / pC +
  plot_layout(widths = c(1, 1), heights = c(1, 1)) +
  plot_annotation(
    tag_levels = 'A',
    tag_suffix = ')',
    title = 'nature-skill test: real snRNA-seq ferroaging data',
    caption = sprintf('Source: figures/meta_with_umap.csv | N=%d cells | ggsci::npg palette', nrow(meta))
  ) &
  theme(plot.tag = element_text(face = 'bold', size = 14))

# ---- 7. 导出（300 DPI + cairo_pdf TrueType 嵌入） ----
outdir <- "d:/铁衰老 绝不重蹈覆辙/figures/skill_test"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

png_path <- file.path(outdir, "nature_test_umap_violin.png")
pdf_path <- file.path(outdir, "nature_test_umap_violin.pdf")

ggsave(png_path, combined, width = 183, height = 120, units = "mm", dpi = 300)
ggsave(pdf_path, combined, width = 183, height = 120, units = "mm",
       device = Cairo::CairoPDF)

cat(sprintf("[OK] PNG saved: %s\n", png_path))
cat(sprintf("[OK] PDF saved: %s (TrueType embedded)\n", pdf_path))
cat("[DONE] nature-skill functional test passed\n")
