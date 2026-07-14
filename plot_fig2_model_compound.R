##############################################################################
# Fig2: GNN模型性能 + LASSO基因 + 化合物排名 (参照 TCMNP/SCIPainter 风格)
# - (A) GNN模型性能并列柱状图
# - (B) LASSO五基因棒棒糖图 (水平)
# - (C) Top-K 候选化合物 lollipop 图
# - (D) 化合物 composite_score vs n_hits 散点图
##############################################################################

library(ggplot2)
library(ggrepel)
library(dplyr)
library(tidyr)
library(readr)
library(ggsci)
library(viridis)

OUTDIR <- "d:/铁衰老 绝不重蹈覆辙/figures"
OUTDIR_PDF <- file.path(OUTDIR, "pdf")

# ============================================================================
# 主题设置 (参照公众号文章风格)
# ============================================================================
theme_pub <- theme_bw(base_size = 11) +
  theme(
    panel.grid.major = element_line(color = "grey90", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 0.8),
    axis.title = element_text(face = "bold", size = 12),
    axis.text = element_text(size = 10, color = "black"),
    plot.title = element_text(face = "bold", size = 13, hjust = 0),
    plot.tag = element_text(face = "bold", size = 16),
    legend.position = "right"
  )

# ============================================================================
# (A) GNN 模型性能柱状图
# ============================================================================
cat("[Fig2-A] GNN model performance...\n")

model_path <- "d:/铁衰老 绝不重蹈覆辙/L4/results_v10_minibatch/model_performance_v67.csv"
if (!file.exists(model_path)) {
  model_path <- "d:/铁衰老 绝不重蹈覆辙/L4/results/model_performance.csv"
}
stopifnot(file.exists(model_path))

perf <- read_csv(model_path, show_col_types = FALSE)
cat(sprintf("  Models: %s\n", paste(perf$model, collapse=", ")))

# Reshape for grouped bar
perf_long <- perf %>%
  select(model, best_auc, best_aupr) %>%
  pivot_longer(cols = c(best_auc, best_aupr), names_to = "metric", values_to = "value") %>%
  mutate(
    metric = recode(metric, best_auc = "AUC", best_aupr = "AUPR"),
    model = factor(model)
  )

p_model <- ggplot(perf_long, aes(x = model, y = value, fill = metric)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6, alpha = 0.9) +
  geom_text(
    aes(label = sprintf("%.3f", value)),
    position = position_dodge(width = 0.7),
    vjust = -0.5, size = 3.2, fontface = "bold"
  ) +
  scale_fill_manual(values = c("AUC" = "#1f87be", "AUPR" = "#e19433")) +
  scale_y_continuous(limits = c(0, max(perf_long$value) * 1.15)) +
  labs(x = NULL, y = "Score", tag = "A", title = "GNN Model Performance (v67)") +
  theme_pub

ggsave(file.path(OUTDIR, "Fig2A_GNN_model_performance.png"), p_model, width = 7, height = 5, dpi = 300)
ggsave(file.path(OUTDIR_PDF, "Fig2A_GNN_model_performance.pdf"), p_model, width = 7, height = 5)
cat("  -> Fig2A saved\n")

# ============================================================================
# (B) LASSO 五基因棒棒糖图 (参照 Nat Commun 蝴蝶图风格)
# ============================================================================
cat("[Fig2-B] LASSO gene lollipop...\n")

lasso_path <- "d:/铁衰老 绝不重蹈覆辙/L2/results/ciri_ferroaging_lasso_candidates.csv"
stopifnot(file.exists(lasso_path))

lasso <- read_csv(lasso_path, show_col_types = FALSE)
# Extract the relevant columns
stopifnot(all(c("Gene_Human", "Selection_Rate", "Log2FC", "Cohens_d") %in% colnames(lasso)))

lasso_plot <- lasso %>%
  mutate(
    Gene_Human = factor(Gene_Human, levels = rev(Gene_Human)),
    direction = ifelse(Log2FC > 0, "Up in High FA", "Down in High FA")
  )

p_lasso <- ggplot(lasso_plot, aes(x = Selection_Rate, y = Gene_Human)) +
  geom_segment(aes(xend = 0, yend = Gene_Human), linewidth = 0.8, color = "grey60") +
  geom_point(aes(size = abs(Cohens_d), fill = Log2FC), shape = 21, stroke = 0.3) +
  scale_fill_gradient2(low = "#2171b5", mid = "white", high = "#d72422", name = "log2FC") +
  scale_size_continuous(range = c(3, 8), name = "|Cohen's d|") +
  scale_x_continuous(limits = c(0, max(lasso_plot$Selection_Rate) * 1.1), labels = scales::percent) +
  labs(
    x = "LASSO Selection Rate", y = NULL,
    tag = "B", title = "CIRI-Ferroaging LASSO Signature Genes"
  ) +
  theme_pub +
  theme(axis.text.y = element_text(face = "bold", size = 11))

ggsave(file.path(OUTDIR, "Fig2B_LASSO_lollipop.png"), p_lasso, width = 8, height = 5, dpi = 300)
ggsave(file.path(OUTDIR_PDF, "Fig2B_LASSO_lollipop.pdf"), p_lasso, width = 8, height = 5)
cat("  -> Fig2B saved\n")

# ============================================================================
# (C) Top-K 候选化合物 lollipop 图
# ============================================================================
cat("[Fig2-C] Top compounds lollipop...\n")

tcm_path <- "d:/铁衰老 绝不重蹈覆辙/L4/results_v10_minibatch/tcm_top_candidates_v67.csv"
if (!file.exists(tcm_path)) {
  tcm_path <- "d:/铁衰老 绝不重蹈覆辙/L4/results/tcm_top_candidates.csv"
}
stopifnot(file.exists(tcm_path))

tcm <- read_csv(tcm_path, show_col_types = FALSE)
cat(sprintf("  TCM candidates: %d rows\n", nrow(tcm)))

# Use top 25 compounds
top_n <- min(25, nrow(tcm))
tcm_top <- tcm %>%
  arrange(desc(composite_score)) %>%
  head(top_n) %>%
  mutate(
    molecule_name = ifelse(is.na(molecule_name) | molecule_name == "", MOL_ID, molecule_name),
    molecule_name = stringr::str_trunc(molecule_name, 30),
    molecule_name = factor(molecule_name, levels = rev(molecule_name))
  )

p_tcm <- ggplot(tcm_top, aes(x = composite_score, y = molecule_name)) +
  geom_segment(aes(xend = 0, yend = molecule_name), linewidth = 0.6, color = "grey70") +
  geom_point(aes(size = n_hits_all, color = max_score_all), alpha = 0.9) +
  scale_color_viridis_c(option = "C", name = "Max Score") +
  scale_size_continuous(range = c(2, 7), name = "N Hits") +
  labs(
    x = "Composite Score", y = NULL,
    tag = "C", title = sprintf("Top %d TCM Compound Candidates", top_n)
  ) +
  theme_pub +
  theme(axis.text.y = element_text(size = 8))

ggsave(file.path(OUTDIR, "Fig2C_top_compounds_lollipop.png"), p_tcm, width = 8, height = 7, dpi = 300)
ggsave(file.path(OUTDIR_PDF, "Fig2C_top_compounds_lollipop.pdf"), p_tcm, width = 8, height = 7)
cat("  -> Fig2C saved\n")

# ============================================================================
# (D) 散点图：composite_score vs n_hits
# ============================================================================
cat("[Fig2-D] Score vs n_hits scatter...\n")

tcm_full_path <- "d:/铁衰老 绝不重蹈覆辙/L4/results_v10_minibatch/tcm_predictions_full_v67.csv"
if (!file.exists(tcm_full_path)) {
  tcm_full_path <- "d:/铁衰老 绝不重蹈覆辙/L4/results/tcm_predictions_full.csv"
}
stopifnot(file.exists(tcm_full_path))

tcm_full <- read_csv(tcm_full_path, show_col_types = FALSE)

# Choose top 50 to label
tcm_full_labeled <- tcm_full %>%
  arrange(desc(composite_score)) %>%
  mutate(
    label = ifelse(row_number() <= 15, molecule_name, ""),
    label = ifelse(is.na(label) | label == "", MOL_ID, label),
    label = stringr::str_trunc(label, 25)
  )

p_scatter <- ggplot(tcm_full_labeled, aes(x = n_hits_all, y = composite_score)) +
  geom_point(aes(color = max_score_all), alpha = 0.5, size = 1.5) +
  geom_text_repel(
    aes(label = label),
    size = 2.8, max.overlaps = 15, box.padding = 0.5,
    segment.size = 0.2, color = "grey30"
  ) +
  scale_color_viridis_c(option = "D", name = "Max Score") +
  labs(
    x = "Number of Hits (All Targets)", y = "Composite Score",
    tag = "D", title = "TCM Compound Screening Landscape"
  ) +
  theme_pub

ggsave(file.path(OUTDIR, "Fig2D_compound_scatter.png"), p_scatter, width = 8, height = 6, dpi = 300)
ggsave(file.path(OUTDIR_PDF, "Fig2D_compound_scatter.pdf"), p_scatter, width = 8, height = 6)
cat("  -> Fig2D saved\n")

cat("[Fig2] All done!\n")
