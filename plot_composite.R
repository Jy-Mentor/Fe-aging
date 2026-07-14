##############################################################################
# COMPOSITE FIGURE GENERATOR
# 铁衰老项目 — 论文级组图 + 图注
#
# 参照标准:
#   - Nature Communications / Cell Reports 多面板排版规范
#   - BMC Medical Genomics 生物信息学图表组织范式
#   - patchwork + cowplot 行业最佳实践
#   - CVD-friendly 配色 (viridis / Okabe-Ito)
#
# 输出:
#   - Fig1_Composite_singlecell_atlas.png/pdf
#   - Fig2_Composite_signature_validation.png/pdf
#   - Fig3_Composite_gnn_compound_screening.png/pdf
#   - Fig4_Composite_chemistry_microglia.png/pdf
#   - Figure_Legends.txt
##############################################################################

library(ggplot2)
library(ggrepel)
library(ggpubr)
library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(ggsci)
library(viridis)
library(patchwork)
library(scales)

OUTDIR     <- "d:/铁衰老 绝不重蹈覆辙/figures"
OUTDIR_PDF <- file.path(OUTDIR, "pdf")
LEGEND_OUT <- "d:/铁衰老 绝不重蹈覆辙/figures/Figure_Legends.txt"

dir.create(OUTDIR_PDF, showWarnings = FALSE, recursive = TRUE)

# ============================================================================
# 0. 统一主题 (Nature Communications 风格)
# ============================================================================
theme_composite <- theme_bw(base_size = 9) +
  theme(
    panel.grid.major   = element_line(color = "grey92", linewidth = 0.25),
    panel.grid.minor   = element_blank(),
    panel.border       = element_rect(color = "black", linewidth = 0.6),
    axis.title         = element_text(face = "bold", size = 10),
    axis.text          = element_text(size = 8, color = "black"),
    axis.ticks         = element_line(color = "black", linewidth = 0.4),
    plot.title         = element_text(face = "bold", size = 11, hjust = 0),
    plot.subtitle      = element_text(size = 9, hjust = 0, color = "grey30"),
    plot.tag           = element_text(face = "bold", size = 14),
    plot.tag.position  = "topleft",
    legend.position    = "right",
    legend.title       = element_text(face = "bold", size = 8),
    legend.text        = element_text(size = 7),
    legend.key.size    = unit(0.5, "cm"),
    strip.background   = element_rect(fill = "grey95", color = "black", linewidth = 0.4),
    strip.text         = element_text(face = "bold", size = 9)
  )

cat("========================================\n")
cat("  Composite Figure Generation\n")
cat("========================================\n\n")

# ============================================================================
# 1. 加载所有真实数据
# ============================================================================
cat("--- Loading data ---\n")

# --- Fig1 data ---
meta <- read_csv("d:/铁衰老 绝不重蹈覆辙/figures/meta_with_umap.csv", show_col_types = FALSE)
agg  <- read_csv("d:/铁衰老 绝不重蹈覆辙/L2/results/GSE233815_sn/ferroaging_score_by_condition_cellclass.csv", show_col_types = FALSE)
cat(sprintf("  Meta: %d cells, Agg: %d rows\n", nrow(meta), nrow(agg)))

score_col <- intersect(c("AddModuleScore_FA96", "FA_96_UCell", "AddModuleScore_FA95"), colnames(meta))[1]

# --- Fig2 data ---
model_perf <- read_csv("d:/铁衰老 绝不重蹈覆辙/L4/results_v10_minibatch/model_performance_v67.csv", show_col_types = FALSE)
lasso_genes <- read_csv("d:/铁衰老 绝不重蹈覆辙/L2/results/ciri_ferroaging_lasso_candidates.csv", show_col_types = FALSE)
tcm_top <- read_csv("d:/铁衰老 绝不重蹈覆辙/L4/results_v10_minibatch/tcm_top_candidates_v67.csv", show_col_types = FALSE)
tcm_full <- read_csv("d:/铁衰老 绝不重蹈覆辙/L4/results_v10_minibatch/tcm_predictions_full_v67.csv", show_col_types = FALSE)

# --- Fig3 data ---
ts_scores <- read_csv("d:/铁衰老 绝不重蹈覆辙/L2/results/ssgsea_ferroaging_scores.csv", show_col_types = FALSE)
de_full <- read_csv("d:/铁衰老 绝不重蹈覆辙/L1/results/GSE61616_DE_results.csv", show_col_types = FALSE)
gpl1355 <- read_csv("d:/铁衰老 绝不重蹈覆辙/L1/results/GPL1355_probe_to_gene.csv", show_col_types = FALSE)
rat2human <- read_csv("d:/铁衰老 绝不重蹈覆辙/L1/results/rat_to_human_ortholog_mygene.csv", show_col_types = FALSE)
fa_genes <- read_csv("d:/铁衰老 绝不重蹈覆辙/L1/results/ferroaging_genes_96.csv", show_col_types = FALSE)$gene_symbol
ext_val <- read_csv("d:/铁衰老 绝不重蹈覆辙/L2/results/external_validation_results.csv", show_col_types = FALSE)
ppi_hub <- read_csv("d:/铁衰老 绝不重蹈覆辙/L2/results/core_ppi_topology.csv", show_col_types = FALSE)
mg_markers <- read_csv("d:/铁衰老 绝不重蹈覆辙/L2/results/GSE233815_sn/microglia_subcluster/microglia_high_fa_markers_annotated.csv", show_col_types = FALSE)

# --- Fig4 data ---
tox_pool <- read_csv("d:/铁衰老 绝不重蹈覆辙/L3/results/tcm_compound_pool_tox_filtered.csv", show_col_types = FALSE)
mg_cluster <- read_csv("d:/铁衰老 绝不重蹈覆辙/L2/results/GSE233815_sn/microglia_subcluster/microglia_cluster_ferroaging_summary.csv", show_col_types = FALSE)
bcp_overlap <- read_csv("d:/铁衰老 绝不重蹈覆辙/L2/results/caryophyllene_ciri_overlap_official_string.csv", show_col_types = FALSE)

cat("  All data loaded.\n\n")

# ============================================================================
# 2. 构建所有面板 (ggplot objects)
# ============================================================================

# ------------------------------------------------------------------
# Panel 1A: UMAP by cell type
# ------------------------------------------------------------------
cat("[Panel 1A] UMAP...\n")
n_types <- length(unique(meta$cell_type_1))
type_colors <- setNames(viridis(n_types, option = "D"), unique(meta$cell_type_1))

p1a <- ggplot(meta, aes(x = UMAP_1, y = UMAP_2, color = cell_type_1)) +
  geom_point(size = 0.25, alpha = 0.7) +
  scale_color_manual(values = type_colors, name = "Cell Type") +
  labs(x = "UMAP 1", y = "UMAP 2") +
  theme_composite +
  theme(legend.position = "right") +
  guides(color = guide_legend(override.aes = list(size = 2.5, alpha = 1), ncol = 1))

# ------------------------------------------------------------------
# Panel 1B: Violin plot by cell type x Condition
# ------------------------------------------------------------------
cat("[Panel 1B] Violin...\n")
meta_clean <- meta %>%
  filter(!is.na(.data[[score_col]]), !is.na(cell_type_1), !is.na(Condition)) %>%
  mutate(Condition = factor(Condition, levels = c("Ctrl", "MCAO")),
         cell_type_1 = factor(cell_type_1))

stat_tests <- meta_clean %>%
  group_by(cell_type_1) %>%
  summarise(p_value = tryCatch(
    wilcox.test(.data[[score_col]][Condition == "MCAO"],
                .data[[score_col]][Condition == "Ctrl"])$p.value,
    error = function(e) NA),
    .groups = "drop") %>%
  mutate(p_label = case_when(
    is.na(p_value) ~ "NS", p_value < 0.001 ~ "***",
    p_value < 0.01 ~ "**", p_value < 0.05 ~ "*", TRUE ~ "NS"))

meta_annot <- meta_clean %>%
  group_by(cell_type_1) %>%
  summarise(max_score = max(.data[[score_col]], na.rm = TRUE), .groups = "drop") %>%
  left_join(stat_tests, by = "cell_type_1") %>%
  mutate(y_pos = max_score * 1.1)

p1b <- ggplot(meta_clean, aes(x = cell_type_1, y = .data[[score_col]], fill = Condition)) +
  geom_violin(alpha = 0.5, linewidth = 0.3, position = position_dodge(0.8), draw_quantiles = 0.5) +
  geom_boxplot(width = 0.12, alpha = 0.6, linewidth = 0.25, position = position_dodge(0.8), outlier.size = 0.2) +
  geom_text(data = meta_annot, aes(x = cell_type_1, y = y_pos, label = p_label),
            inherit.aes = FALSE, size = 2.8, fontface = "bold", color = "black") +
  scale_fill_manual(values = c("Ctrl" = "#6FB2C1", "MCAO" = "#E07524")) +
  labs(x = NULL, y = "Ferroaging Score") +
  theme_composite +
  theme(axis.text.x = element_text(angle = 35, hjust = 1, size = 7.5),
        legend.position.inside = c(0.9, 0.88))

# ------------------------------------------------------------------
# Panel 1C: Bar chart mean +/- SE
# ------------------------------------------------------------------
cat("[Panel 1C] Bar chart...\n")
agg_clean <- agg %>%
  filter(!is.na(cell_class), !is.na(Condition)) %>%
  mutate(Condition = factor(Condition, levels = c("Ctrl", "MCAO")),
         cell_class = factor(cell_class))

p1c <- ggplot(agg_clean, aes(x = cell_class, y = mean_score, fill = Condition)) +
  geom_col(position = position_dodge(0.8), width = 0.7, alpha = 0.85) +
  geom_errorbar(aes(ymin = mean_score - se_score, ymax = mean_score + se_score),
                position = position_dodge(0.8), width = 0.2, linewidth = 0.5) +
  scale_fill_manual(values = c("Ctrl" = "#6FB2C1", "MCAO" = "#E07524")) +
  labs(x = NULL, y = "Mean Ferroaging Score") +
  theme_composite +
  theme(axis.text.x = element_text(angle = 35, hjust = 1, size = 7.5),
        legend.position.inside = c(0.9, 0.88))

# ------------------------------------------------------------------
# Panel 2A: GNN Model Performance
# ------------------------------------------------------------------
cat("[Panel 2A] GNN performance...\n")
perf_long <- model_perf %>%
  select(model, best_auc, best_aupr) %>%
  pivot_longer(c(best_auc, best_aupr), names_to = "metric", values_to = "value") %>%
  mutate(metric = recode(metric, best_auc = "AUC", best_aupr = "AUPR"),
         model = factor(model))

p2a <- ggplot(perf_long, aes(x = model, y = value, fill = metric)) +
  geom_col(position = position_dodge(0.7), width = 0.6, alpha = 0.9) +
  geom_text(aes(label = sprintf("%.3f", value)),
            position = position_dodge(0.7), vjust = -0.5, size = 2.8, fontface = "bold") +
  scale_fill_manual(values = c("AUC" = "#1f87be", "AUPR" = "#e19433")) +
  scale_y_continuous(limits = c(0, max(perf_long$value) * 1.15)) +
  labs(x = NULL, y = "Score") +
  theme_composite

# ------------------------------------------------------------------
# Panel 2B: LASSO lollipop
# ------------------------------------------------------------------
cat("[Panel 2B] LASSO lollipop...\n")
lasso_plot <- lasso_genes %>%
  mutate(Gene_Human = factor(Gene_Human, levels = rev(Gene_Human)))

p2b <- ggplot(lasso_plot, aes(x = Selection_Rate, y = Gene_Human)) +
  geom_segment(aes(xend = 0, yend = Gene_Human), linewidth = 0.7, color = "grey60") +
  geom_point(aes(size = abs(Cohens_d), fill = Log2FC), shape = 21, stroke = 0.3) +
  scale_fill_gradient2(low = "#2171b5", mid = "white", high = "#d72422", name = "log2FC") +
  scale_size_continuous(range = c(3, 7), name = "|Cohen's d|") +
  scale_x_continuous(limits = c(0, max(lasso_plot$Selection_Rate) * 1.1), labels = percent) +
  labs(x = "LASSO Selection Rate", y = NULL) +
  theme_composite +
  theme(axis.text.y = element_text(face = "bold", size = 9))

# ------------------------------------------------------------------
# Panel 2C: Top TCM compounds lollipop
# ------------------------------------------------------------------
cat("[Panel 2C] Top compounds...\n")
top_n <- min(25, nrow(tcm_top))
tcm_top25 <- tcm_top %>%
  arrange(desc(composite_score)) %>%
  head(top_n) %>%
  mutate(molecule_name = ifelse(is.na(molecule_name) | molecule_name == "", MOL_ID, molecule_name),
         molecule_name = stringr::str_trunc(molecule_name, 28),
         molecule_name = factor(molecule_name, levels = rev(molecule_name)))

p2c <- ggplot(tcm_top25, aes(x = composite_score, y = molecule_name)) +
  geom_segment(aes(xend = 0, yend = molecule_name), linewidth = 0.5, color = "grey70") +
  geom_point(aes(size = n_hits_all, color = max_score_all), alpha = 0.9) +
  scale_color_viridis_c(option = "C", name = "Max Score") +
  scale_size_continuous(range = c(2, 6), name = "N Hits") +
  labs(x = "Composite Score", y = NULL) +
  theme_composite +
  theme(axis.text.y = element_text(size = 7))

# ------------------------------------------------------------------
# Panel 2D: Compound scatter
# ------------------------------------------------------------------
cat("[Panel 2D] Compound scatter...\n")
tcm_labeled <- tcm_full %>%
  arrange(desc(composite_score)) %>%
  mutate(label = ifelse(row_number() <= 15,
                        stringr::str_trunc(coalesce(molecule_name, MOL_ID), 22), ""))

p2d <- ggplot(tcm_labeled, aes(x = n_hits_all, y = composite_score)) +
  geom_point(aes(color = max_score_all), alpha = 0.5, size = 1.2) +
  geom_text_repel(aes(label = label), size = 2.5, max.overlaps = 15,
                  box.padding = 0.4, segment.size = 0.2, color = "grey30") +
  scale_color_viridis_c(option = "D", name = "Max Score") +
  labs(x = "Number of Hits (All Targets)", y = "Composite Score") +
  theme_composite

# ------------------------------------------------------------------
# Panel 3A: Ferroaging time series
# ------------------------------------------------------------------
cat("[Panel 3A] Time series...\n")
ts_clean <- ts_scores %>% filter(!is.na(group)) %>% mutate(group = as.character(group))
ds_use <- unique(ts_clean$dataset)[1]
ts_sub <- ts_clean %>% filter(dataset == ds_use)

p3a <- ggplot(ts_sub, aes(x = group, y = Ferroaging_Score, fill = group)) +
  geom_boxplot(alpha = 0.6, outlier.size = 0.6, linewidth = 0.35) +
  geom_jitter(width = 0.15, alpha = 0.35, size = 0.6) +
  scale_fill_npg() +
  labs(x = NULL, y = "Ferroaging Score") +
  theme_composite +
  theme(legend.position = "none")

# ------------------------------------------------------------------
# Panel 3B: Volcano GSE61616
# ------------------------------------------------------------------
cat("[Panel 3B] Volcano...\n")
de_mapped <- de_full %>%
  left_join(gpl1355 %>% select(Probe, GeneSymbol), by = "Probe") %>%
  rename(rat_gene = GeneSymbol)
ortho_dict <- setNames(rat2human$human_symbol, rat2human$rat_symbol)
de_mapped$human_gene <- ortho_dict[de_mapped$rat_gene]
na_mask <- is.na(de_mapped$human_gene) & !is.na(de_mapped$rat_gene)
de_mapped$human_gene[na_mask] <- toupper(de_mapped$rat_gene[na_mask])

de_plot <- de_mapped %>%
  filter(!is.na(adj.P.Val), !is.na(logFC)) %>%
  mutate(neg_log10_p = -log10(adj.P.Val),
         significance = case_when(
           adj.P.Val < 0.01 & abs(logFC) > 1.5 ~ "padj<0.01 & |FC|>1.5",
           adj.P.Val < 0.05 & abs(logFC) > 0.8 ~ "padj<0.05 & |FC|>0.8",
           TRUE ~ "NS"),
         is_fa = human_gene %in% fa_genes,
         label = ifelse((adj.P.Val < 0.001 & abs(logFC) > 1.5) | is_fa,
                        coalesce(human_gene, rat_gene, Probe), ""))

n_fa <- sum(de_plot$is_fa, na.rm = TRUE)
cat(sprintf("  FA genes in volcano: %d\n", n_fa))

p3b <- ggplot(de_plot, aes(x = logFC, y = neg_log10_p)) +
  geom_point(aes(color = significance), alpha = 0.4, size = 0.6) +
  geom_point(data = filter(de_plot, is_fa), aes(x = logFC, y = neg_log10_p),
             color = "#d72422", size = 1.0, alpha = 0.85) +
  geom_text_repel(aes(label = label), size = 2.3, max.overlaps = 25,
                  box.padding = 0.25, segment.size = 0.15, color = "grey20") +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "grey50", linewidth = 0.3) +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "grey50", linewidth = 0.3) +
  scale_color_manual(values = c(
    "padj<0.01 & |FC|>1.5" = "#E41A1C", "padj<0.05 & |FC|>0.8" = "#377EB8", "NS" = "grey80")) +
  labs(x = "log2 Fold Change (MCAO vs Sham)", y = "-log10(adjusted P-value)") +
  theme_composite +
  theme(legend.position.inside = c(0.14, 0.88), legend.title = element_blank())

# ------------------------------------------------------------------
# Panel 3C: External validation forest
# ------------------------------------------------------------------
cat("[Panel 3C] External validation...\n")
fisher_ci <- function(rho, n, alpha = 0.05) {
  z <- atanh(rho); se <- 1 / sqrt(n - 3); z_crit <- qnorm(1 - alpha / 2)
  list(lower = tanh(z - z_crit * se), upper = tanh(z + z_crit * se))
}

ext_plot <- ext_val %>%
  rowwise() %>%
  mutate(Dataset = factor(Dataset, levels = rev(Dataset)),
         value = Spearman_rho,
         ci = list(fisher_ci(Spearman_rho, N_Valid)),
         lower = ci$lower, upper = ci$upper,
         sig = case_when(Spearman_p < 0.001 ~ "***", Spearman_p < 0.01 ~ "**",
                         Spearman_p < 0.05 ~ "*", TRUE ~ "NS")) %>%
  ungroup()

p3c <- ggplot(ext_plot, aes(x = value, y = Dataset)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.35) +
  geom_segment(aes(x = lower, xend = upper, y = Dataset, yend = Dataset),
               color = "grey50", linewidth = 1.0, alpha = 0.55) +
  geom_point(aes(color = value, size = FA_AUC), alpha = 0.9) +
  geom_text(aes(label = sprintf("%.3f [%.2f,%.2f]%s", value, lower, upper, sig)),
            hjust = -0.12, size = 2.8, fontface = "bold") +
  scale_color_viridis_c(option = "A", direction = -1, name = "Spearman rho") +
  scale_size_continuous(range = c(2.5, 7), name = "FA AUC") +
  scale_x_continuous(limits = c(min(ext_plot$lower) * 0.9, max(ext_plot$upper) * 1.25)) +
  labs(x = "Spearman rho [95% CI via Fisher z-transform]", y = NULL) +
  theme_composite

# ------------------------------------------------------------------
# Panel 3D: PPI Hub ranking
# ------------------------------------------------------------------
cat("[Panel 3D] PPI Hub...\n")
ppi_top25 <- ppi_hub %>%
  arrange(desc(Degree)) %>%
  head(25) %>%
  mutate(Gene = factor(Gene, levels = rev(Gene)))

p3d <- ggplot(ppi_top25, aes(x = Degree, y = Gene)) +
  geom_col(aes(fill = Betweenness), alpha = 0.85, width = 0.7) +
  scale_fill_viridis_c(option = "B", name = "Betweenness") +
  labs(x = "Degree Centrality", y = NULL) +
  theme_composite +
  theme(axis.text.y = element_text(face = "italic", size = 7.5))

# ------------------------------------------------------------------
# Panel 4A: Compound property distributions
# ------------------------------------------------------------------
cat("[Panel 4A] Compound properties...\n")
props <- c("MW", "LogP", "TPSA", "QED")
prop_plots <- list()
for (prop in props) {
  if (!prop %in% colnames(tox_pool)) next
  p_dat <- tox_pool %>% filter(!is.na(.data[[prop]]), is.finite(.data[[prop]]))
  prop_plots[[prop]] <- ggplot(p_dat, aes(x = .data[[prop]])) +
    geom_histogram(aes(fill = after_stat(x)), bins = 40, alpha = 0.8,
                   color = "white", linewidth = 0.15) +
    scale_fill_viridis_c(option = "C") +
    labs(x = prop, y = "Count") +
    theme_composite +
    theme(legend.position = "none", plot.tag = NULL)
}

# ------------------------------------------------------------------
# Panel 4B: Microglia subcluster bubble
# ------------------------------------------------------------------
cat("[Panel 4B] Microglia bubble...\n")
mg_fa96 <- mg_cluster %>% filter(score_method == "AddModuleScore_FA96")
actual_conditions <- unique(mg_fa96$Condition)
mg_plot <- mg_fa96 %>%
  mutate(seurat_clusters = factor(seurat_clusters),
         Condition = factor(Condition, levels = actual_conditions))

p4b <- ggplot(mg_plot, aes(x = Condition, y = seurat_clusters)) +
  geom_point(aes(size = n_cells, fill = mean_score), shape = 21, stroke = 0.25) +
  scale_fill_gradient2(low = "#3A9AB2", mid = "white", high = "#E07524", midpoint = 0,
                       name = "Mean FA\nScore") +
  scale_size_continuous(range = c(2, 10), name = "N Cells") +
  labs(x = NULL, y = "Microglia Subcluster") +
  theme_composite

# ------------------------------------------------------------------
# Panel 4C: BCP-CIRI overlap
# ------------------------------------------------------------------
cat("[Panel 4C] BCP-CIRI overlap...\n")
bcp_plot <- bcp_overlap %>%
  filter(!is.na(Count), !is.na(P_value)) %>%
  mutate(Item = factor(Item, levels = rev(Item)), neg_log10_p = -log10(P_value))

p4c <- ggplot(bcp_plot, aes(x = Count, y = Item)) +
  geom_col(aes(fill = neg_log10_p), alpha = 0.85, width = 0.6) +
  geom_text(aes(label = Count), hjust = -0.3, size = 3, fontface = "bold") +
  scale_fill_viridis_c(option = "B", name = "-log10(P)") +
  scale_x_continuous(limits = c(0, max(bcp_plot$Count) * 1.3)) +
  labs(x = "Gene Count", y = NULL) +
  theme_composite

# ============================================================================
# 3. 组装复合图 (patchwork)
# ============================================================================

cat("\n--- Assembling composite figures ---\n")

# --- Fig1: Single-cell Atlas ---
# Layout: A (UMAP, full width) / (B | C) (violin + bar, side by side)
fig1 <- (p1a + labs(tag = "A")) /
        ((p1b + labs(tag = "B")) | (p1c + labs(tag = "C"))) +
        plot_layout(heights = c(1, 0.9)) &
        theme(plot.tag = element_text(face = "bold", size = 14))

ggsave(file.path(OUTDIR, "Fig1_Composite_singlecell_atlas.png"),
       fig1, width = 14, height = 11, dpi = 300, bg = "white")
ggsave(file.path(OUTDIR_PDF, "Fig1_Composite_singlecell_atlas.pdf"),
       fig1, width = 14, height = 11, bg = "white")
cat("  -> Fig1 composite saved\n")

# --- Fig2: GNN + LASSO + TCM Screening ---
# Layout: (A | B) / C / D
fig2 <- ((p2a + labs(tag = "A")) | (p2b + labs(tag = "B"))) /
        (p2c + labs(tag = "C")) /
        (p2d + labs(tag = "D")) +
        plot_layout(heights = c(1, 1.4, 1.2)) &
        theme(plot.tag = element_text(face = "bold", size = 14))

ggsave(file.path(OUTDIR, "Fig2_Composite_gnn_compound_screening.png"),
       fig2, width = 12, height = 16, dpi = 300, bg = "white")
ggsave(file.path(OUTDIR_PDF, "Fig2_Composite_gnn_compound_screening.pdf"),
       fig2, width = 12, height = 16, bg = "white")
cat("  -> Fig2 composite saved\n")

# --- Fig3: Transcriptomic Validation ---
# Layout: (A | B) / C / D
fig3 <- ((p3a + labs(tag = "A")) | (p3c + labs(tag = "B"))) /
        (p3b + labs(tag = "C")) /
        (p3d + labs(tag = "D")) +
        plot_layout(heights = c(1, 1.4, 1.2)) &
        theme(plot.tag = element_text(face = "bold", size = 14))

ggsave(file.path(OUTDIR, "Fig3_Composite_transcriptomic_validation.png"),
       fig3, width = 12, height = 16, dpi = 300, bg = "white")
ggsave(file.path(OUTDIR_PDF, "Fig3_Composite_transcriptomic_validation.pdf"),
       fig3, width = 12, height = 16, bg = "white")
cat("  -> Fig3 composite saved\n")

# --- Fig4: Chemistry + Microglia ---
# Layout: A (2x2 grid) / (B | C)
p4a_grid <- wrap_plots(prop_plots, ncol = 2)
fig4 <- (p4a_grid + labs(tag = "A")) /
        ((p4b + labs(tag = "B")) | (p4c + labs(tag = "C"))) +
        plot_layout(heights = c(1.2, 1)) &
        theme(plot.tag = element_text(face = "bold", size = 14))

ggsave(file.path(OUTDIR, "Fig4_Composite_chemistry_microglia.png"),
       fig4, width = 13, height = 11, dpi = 300, bg = "white")
ggsave(file.path(OUTDIR_PDF, "Fig4_Composite_chemistry_microglia.pdf"),
       fig4, width = 13, height = 11, bg = "white")
cat("  -> Fig4 composite saved\n")

# ============================================================================
# 4. 撰写论文级图注 (Publication-level Figure Legends)
# ============================================================================
cat("\n--- Writing figure legends ---\n")

legends <- '
===============================================================================
  FIGURE LEGENDS
  铁衰老项目 — CIRI Ferroaging Multi-omics Study
  Nature Communications / Cell Reports 格式
===============================================================================

Figure 1. Single-cell transcriptomic atlas of ferroaging in ischemic stroke.

(A) Uniform Manifold Approximation and Projection (UMAP) embedding of 7,414
single nuclei from the GSE233815 snRNA-seq dataset, colored by 25 annotated
cell types. Cell type annotations were derived from canonical marker gene
expression and reference-based mapping.

(B) Violin plots showing the distribution of ferroaging scores
(AddModuleScore_FA96) across cell types, stratified by condition (Ctrl, blue;
MCAO, orange). Horizontal lines indicate medians. Statistical significance was
determined by two-sided Wilcoxon rank-sum test. ***P < 0.001, **P < 0.01,
*P < 0.05; NS, not significant.

(C) Grouped bar chart of mean ferroaging scores +/- SEM for each cell type by
condition. Error bars represent standard error of the mean (SEM).


Figure 2. GNN-based compound screening and LASSO feature selection for
CIRI-ferroaging signature identification.

(A) Model performance comparison of three graph neural network architectures
(SAGE, HGT, SimpleHGN) evaluated on the CIRI-ferroaging compound-target
prediction task. Bars represent best area under the ROC curve (AUC, blue) and
area under the precision-recall curve (AUPR, orange). Values are labeled above
each bar.

(B) LASSO logistic regression lollipop plot of the five CIRI-ferroaging
signature genes (SAT1, CD74, KLF6, LIFR, EBF3). Gene names are shown on the
y-axis. Point size corresponds to absolute Cohen\'s d effect size. Point fill
color represents log2 fold change (red, upregulated in high-FA group; blue,
downregulated in high-FA group). X-axis shows LASSO selection rate across
repeated cross-validation.

(C) Lollipop plot of the top 25 traditional Chinese medicine (TCM) compound
candidates ranked by composite screening score. Point size indicates the
number of CIRI-ferroaging target genes hit by each compound. Point color
encodes the maximum individual target prediction score.

(D) Scatter plot showing the TCM compound screening landscape. Each point
represents one compound from the 517-compound TCM library. The top 15
highest-scoring compounds are labeled. Color gradient indicates maximum
individual target prediction score.


Figure 3. Transcriptomic validation and network-based characterization of
ferroaging gene signatures.

(A) Boxplot showing ferroaging score distribution by experimental group in
the GSE16561 dataset. Individual data points represent biological samples.
The ferroaging score was computed via single-sample gene set enrichment
analysis (ssGSEA) using the 96-gene ferroaging signature.

(B) Volcano plot of differential expression analysis from the GSE61616
dataset (MCAO vs Sham, Affymetrix Rat 230 2.0 Array). A total of 31,099
probes were mapped to human gene symbols via the GPL1355 probe-to-gene
annotation followed by rat-to-human ortholog conversion (HomoloGene/mygene).
Ferroaging signature genes (n = 85/96 detected) are highlighted in red.
Color legend: red, padj < 0.01 and |log2FC| > 1.5; blue, padj < 0.05 and
|log2FC| > 0.8; grey, not significant. Dashed lines indicate thresholds at
|log2FC| = 1 and adjusted P = 0.05.

(C) Forest plot of external validation results across three independent
datasets (GSE16561, GSE61616, GSE97537). Spearman correlation coefficients
(rho) between predicted and observed ferroaging scores are shown with 95%
confidence intervals computed via Fisher z-transform. Point size represents
ferroaging AUC. ***P < 0.001, **P < 0.01, *P < 0.05.

(D) Horizontal bar chart of the top 25 hub genes in the CIRI-ferroaging
protein-protein interaction (PPI) network ranked by degree centrality. Bar
fill color encodes betweenness centrality. Gene names are italicized.


Figure 4. Chemical property landscape and microglia subcluster ferroaging
activity.

(A) Histograms showing the distribution of four key physicochemical
properties across the 517-compound TCM screening library: molecular weight
(MW, Da), octanol-water partition coefficient (LogP), topological polar
surface area (TPSA, Angstrom^2), and quantitative estimate of drug-likeness
(QED). All values were computed using RDKit.

(B) Bubble plot of ferroaging activity (AddModuleScore_FA96) across five
microglia subclusters (0-4) at four time points following MCAO: control
(Ctrl), 1 day post-injury (1DPI), 3DPI, and 7DPI. Point size indicates the
number of nuclei per subcluster-condition combination. Point fill color
represents the mean ferroaging score (blue, low; white, neutral; orange,
high).

(C) Horizontal bar chart showing the overlap between beta-caryophyllene (BCP)
target genes and CIRI-ferroaging gene sets. Gene count is labeled at the end
of each bar. Bar fill color encodes -log10(P-value) from hypergeometric
enrichment testing.


METHODS SUMMARY
---------------
Data processing and visualization were performed using R (version 4.3.3) with
the following packages: ggplot2 (3.5.1), patchwork, ggrepel, ggsci, viridis,
dplyr, and readr. Single-cell analysis was conducted using Seurat (v5).
Ferroaging scores were computed using AddModuleScore. Differential expression
analysis was performed using limma with Benjamini-Hochberg multiple testing
correction. LASSO logistic regression was implemented via glmnet with 10-fold
cross-validation. GNN models (SAGE, HGT, SimpleHGN) were trained using PyTorch
Geometric. All statistical tests were two-sided unless otherwise specified.
'

writeLines(trimws(legends), LEGEND_OUT)
cat(sprintf("  Figure legends saved to: %s\n", LEGEND_OUT))

# ============================================================================
# 5. 输出总结
# ============================================================================
cat("\n--- Composite Output Summary ---\n")
png_files <- list.files(OUTDIR, pattern = "Composite.*\\.png$", full.names = TRUE)
pdf_files <- list.files(OUTDIR_PDF, pattern = "Composite.*\\.pdf$", full.names = TRUE)
cat(sprintf("Composite PNG: %d files\n", length(png_files)))
cat(sprintf("Composite PDF: %d files\n", length(pdf_files)))
for (f in png_files) {
  cat(sprintf("  %s (%.0f KB)\n", basename(f), file.info(f)$size/1024))
}

cat("\n========================================\n")
cat("  Composite figure generation complete!\n")
cat("========================================\n")
