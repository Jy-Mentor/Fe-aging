##############################################################################
# Fig1: 单细胞铁衰老分析 (参照 Nature/MedComm 风格)
# - (A) UMAP 按细胞类型着色
# - (B) 小提琴图：细胞类型 x Condition 铁衰老评分分布
# - (C) 分组条形图：各细胞类型铁衰老 mean±SE
##############################################################################

library(ggplot2)
library(ggpubr)
library(ggrepel)
library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(ggsci)
library(viridis)

OUTDIR <- "d:/铁衰老 绝不重蹈覆辙/figures"
OUTDIR_PDF <- file.path(OUTDIR, "pdf")

# ============================================================================
# 0. 读取真实数据
# ============================================================================
cat("[Fig1] Reading data...\n")

meta_path <- "d:/铁衰老 绝不重蹈覆辙/figures/meta_with_umap.csv"
agg_path  <- "d:/铁衰老 绝不重蹈覆辙/L2/results/GSE233815_sn/ferroaging_score_by_condition_cellclass.csv"

stopifnot(file.exists(meta_path))
stopifnot(file.exists(agg_path))

meta <- read_csv(meta_path, show_col_types = FALSE)
agg  <- read_csv(agg_path, show_col_types = FALSE)

cat(sprintf("  Metadata: %d cells, columns: %s\n", nrow(meta), paste(colnames(meta), collapse=", ")))
cat(sprintf("  Aggregated: %d rows, columns: %s\n", nrow(agg), paste(colnames(agg), collapse=", ")))

# Verify expected columns exist
stopifnot("cell_type_1" %in% colnames(meta))
stopifnot("Condition" %in% colnames(meta))

# Determine which FA score column exists
score_col <- NULL
for (col in c("AddModuleScore_FA96", "FA_96_UCell", "AddModuleScore_FA95", "FA_95_UCell")) {
  if (col %in% colnames(meta)) { score_col <- col; break }
}
stopifnot(!is.null(score_col))
cat(sprintf("  Using score column: %s\n", score_col))

# ============================================================================
# 主题设置 (参照 Nature Communications 风格)
# ============================================================================
theme_nature <- theme_bw(base_size = 11) +
  theme(
    panel.grid.major = element_line(color = "grey90", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 0.8),
    axis.title = element_text(face = "bold", size = 12),
    axis.text = element_text(size = 10, color = "black"),
    axis.ticks = element_line(color = "black", linewidth = 0.5),
    legend.position = "right",
    legend.title = element_text(face = "bold", size = 9),
    legend.text = element_text(size = 8),
    plot.title = element_text(face = "bold", size = 14, hjust = 0),
    plot.tag = element_text(face = "bold", size = 16),
    strip.background = element_rect(fill = "grey95", color = "black"),
    strip.text = element_text(face = "bold", size = 10)
  )

# ============================================================================
# (A) UMAP 散点图 -- 按细胞类型着色
# ============================================================================
cat("[Fig1-A] Plotting UMAP by cell type...\n")

# Check for UMAP coordinates
has_umap <- all(c("UMAP_1", "UMAP_2") %in% colnames(meta))
if (has_umap) {
  cell_types <- unique(meta$cell_type_1)
  n_types <- length(cell_types)
  cat(sprintf("  %d cell types found\n", n_types))

  # Use Nature-inspired palette
  if (n_types <= 10) {
    type_colors <- setNames(
      pal_npg("nrc")(n_types)[1:n_types],
      cell_types
    )
  } else {
    type_colors <- setNames(
      viridis(n_types, option = "D"),
      cell_types
    )
  }

  p_umap <- ggplot(meta, aes(x = UMAP_1, y = UMAP_2, color = cell_type_1)) +
    geom_point(size = 0.3, alpha = 0.7) +
    scale_color_manual(values = type_colors, name = "Cell Type") +
    labs(x = "UMAP 1", y = "UMAP 2", tag = "A") +
    theme_nature +
    theme(legend.position = "right") +
    guides(color = guide_legend(override.aes = list(size = 3, alpha = 1), ncol = 1))

  ggsave(file.path(OUTDIR, "Fig1A_UMAP_celltype.png"), p_umap, width = 9, height = 6, dpi = 300)
  ggsave(file.path(OUTDIR_PDF, "Fig1A_UMAP_celltype.pdf"), p_umap, width = 9, height = 6)
  cat("  -> Fig1A_UMAP_celltype saved\n")
} else {
  cat("  WARNING: No UMAP coordinates found, skipping UMAP plot\n")
}

# ============================================================================
# (B) 小提琴图 -- 细胞类型 x Condition 铁衰老评分
# ============================================================================
cat("[Fig1-B] Plotting violin plot...\n")

# Clean data: remove NA
meta_clean <- meta %>%
  filter(!is.na(.data[[score_col]]),
         !is.na(cell_type_1),
         !is.na(Condition)) %>%
  mutate(
    cell_type_1 = as.character(cell_type_1),
    Condition = as.character(Condition)
  )

# Ensure factor order
meta_clean$Condition <- factor(meta_clean$Condition, levels = c("Ctrl", "MCAO"))
meta_clean$cell_type_1 <- factor(meta_clean$cell_type_1)

# Calculate medians for annotation
medians <- meta_clean %>%
  group_by(cell_type_1, Condition) %>%
  summarise(median_score = median(.data[[score_col]], na.rm = TRUE), .groups = "drop")

# Statistical test per cell type
stat_tests <- meta_clean %>%
  group_by(cell_type_1) %>%
  summarise(
    p_value = tryCatch(
      wilcox.test(
        .data[[score_col]][Condition == "MCAO"],
        .data[[score_col]][Condition == "Ctrl"]
      )$p.value,
      error = function(e) NA
    ),
    .groups = "drop"
  ) %>%
  mutate(p_label = case_when(
    is.na(p_value) ~ "NS",
    p_value < 0.001 ~ "***",
    p_value < 0.01 ~ "**",
    p_value < 0.05 ~ "*",
    TRUE ~ "NS"
  ))

cat(sprintf("  Statistical tests done for %d cell types\n", nrow(stat_tests)))

# Prepare significance annotation positions
meta_clean_annot <- meta_clean %>%
  group_by(cell_type_1) %>%
  summarise(
    max_score = max(.data[[score_col]], na.rm = TRUE),
    .groups = "drop"
  ) %>%
  left_join(stat_tests, by = "cell_type_1") %>%
  mutate(y_pos = max_score * 1.08)

p_violin <- ggplot(meta_clean, aes(x = cell_type_1, y = .data[[score_col]], fill = Condition)) +
  geom_violin(alpha = 0.5, linewidth = 0.4, position = position_dodge(width = 0.8), draw_quantiles = 0.5) +
  geom_boxplot(width = 0.15, alpha = 0.6, linewidth = 0.3, position = position_dodge(width = 0.8), outlier.size = 0.3) +
  geom_text(data = meta_clean_annot, aes(x = cell_type_1, y = y_pos, label = p_label),
            inherit.aes = FALSE, size = 3.5, fontface = "bold", color = "black") +
  scale_fill_manual(values = c("Ctrl" = "#6FB2C1", "MCAO" = "#E07524")) +
  labs(x = NULL, y = "Ferroaging Score", tag = "B") +
  theme_nature +
  theme(
    axis.text.x = element_text(angle = 35, hjust = 1, size = 9),
    legend.position.inside = c(0.88, 0.85)
  )

ggsave(file.path(OUTDIR, "Fig1B_violin_ferroaging_by_celltype.png"), p_violin, width = 10, height = 6, dpi = 300)
ggsave(file.path(OUTDIR_PDF, "Fig1B_violin_ferroaging_by_celltype.pdf"), p_violin, width = 10, height = 6)
cat("  -> Fig1B_violin_ferroaging_by_celltype saved\n")

# ============================================================================
# (C) 分组条形图 -- mean±SE (参照 Nat Commun 蝴蝶图简化版)
# ============================================================================
cat("[Fig1-C] Plotting grouped bar chart...\n")

agg_clean <- agg %>%
  filter(!is.na(cell_class), !is.na(Condition)) %>%
  mutate(
    Condition = factor(Condition, levels = c("Ctrl", "MCAO")),
    cell_class = factor(cell_class)
  )

p_bar <- ggplot(agg_clean, aes(x = cell_class, y = mean_score, fill = Condition)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7, alpha = 0.85) +
  geom_errorbar(
    aes(ymin = mean_score - se_score, ymax = mean_score + se_score),
    position = position_dodge(width = 0.8),
    width = 0.2, linewidth = 0.6
  ) +
  scale_fill_manual(values = c("Ctrl" = "#6FB2C1", "MCAO" = "#E07524")) +
  labs(x = NULL, y = "Mean Ferroaging Score", tag = "C") +
  theme_nature +
  theme(
    axis.text.x = element_text(angle = 35, hjust = 1, size = 9),
    legend.position.inside = c(0.88, 0.85)
  )

ggsave(file.path(OUTDIR, "Fig1C_bar_ferroaging_mean_se.png"), p_bar, width = 9, height = 6, dpi = 300)
ggsave(file.path(OUTDIR_PDF, "Fig1C_bar_ferroaging_mean_se.pdf"), p_bar, width = 9, height = 6)
cat("  -> Fig1C_bar_ferroaging_mean_se saved\n")

cat("[Fig1] All done!\n")
