##############################################################################
# SCISSOR ANALYSIS FIGURES
# 铁衰老项目 — Cell Scissor 表型关联细胞亚群识别
#
# 数据来源:
#   - L2/results/scissor_selected_cells.csv (15000 cells, 150 Scissor+, 150 Scissor-)
#   - L2/results/scissor_celltype_enrichment.csv (8 cell types)
#   - L2/results/scissor_network_overlap.csv (5000 DE genes)
#   - figures/meta_with_umap.csv (UMAP coordinates)
#
# 参考标准:
#   - Sun et al., Nature Biotechnology (2021) — SCISSOR 原始论文
#   - 公众号 "小张聊科研" SCISSOR 可视化最佳实践
#   - Nature Communications 多面板排版规范
##############################################################################

library(ggplot2)
library(ggrepel)
library(dplyr)
library(readr)
library(tidyr)
library(patchwork)
library(ggsci)
library(scales)

OUTDIR     <- "d:/铁衰老 绝不重蹈覆辙/figures"
OUTDIR_PDF <- file.path(OUTDIR, "pdf")
dir.create(OUTDIR_PDF, showWarnings = FALSE, recursive = TRUE)

theme_pub <- theme_bw(base_size = 9) +
  theme(
    panel.grid.major = element_line(color = "grey92", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    panel.border     = element_rect(color = "black", linewidth = 0.6),
    axis.title       = element_text(face = "bold", size = 10),
    axis.text        = element_text(size = 8, color = "black"),
    plot.tag         = element_text(face = "bold", size = 14),
    plot.tag.position = "topleft",
    legend.title     = element_text(face = "bold", size = 8),
    legend.text      = element_text(size = 7),
    legend.key.size  = unit(0.5, "cm")
  )

cat("========================================\n")
cat("  SCISSOR Analysis Figures\n")
cat("========================================\n\n")

# ============================================================================
# 1. 加载数据
# ============================================================================
cat("--- Loading data ---\n")

scissor_cells <- read_csv("d:/铁衰老 绝不重蹈覆辙/L2/results/scissor_selected_cells.csv", show_col_types = FALSE)
ct_enrich     <- read_csv("d:/铁衰老 绝不重蹈覆辙/L2/results/scissor_celltype_enrichment.csv", show_col_types = FALSE)
net_overlap   <- read_csv("d:/铁衰老 绝不重蹈覆辙/L2/results/scissor_network_overlap.csv", show_col_types = FALSE)

scissor_umap <- read_csv("d:/铁衰老 绝不重蹈覆辙/figures/scissor_umap_metadata.csv", show_col_types = FALSE)
if ("X1" %in% colnames(scissor_umap)) scissor_umap <- scissor_umap %>% rename(cell_barcode = X1)

cat(sprintf("  Scissor cells (Python/GSE174574): %d (Scissor+: %d, Scissor-: %d)\n",
            nrow(scissor_cells),
            sum(scissor_cells$scissor_label == "Scissor+", na.rm = TRUE),
            sum(scissor_cells$scissor_label == "Scissor-", na.rm = TRUE)))
cat(sprintf("  Cell type enrichment: %d types\n", nrow(ct_enrich)))
cat(sprintf("  Network overlap: %d genes\n", nrow(net_overlap)))
cat(sprintf("  UMAP meta (R/GSE233815): %d cells\n", nrow(scissor_umap)))
cat(sprintf("  UMAP Scissor labels: %s\n", paste(names(table(scissor_umap$scissor)), collapse=", ")))

scissor_umap <- scissor_umap %>%
  filter(!is.na(UMAP_1), !is.na(UMAP_2)) %>%
  mutate(scissor = ifelse(is.na(scissor) | scissor == "", "Background", scissor),
         scissor = factor(scissor, levels = c("Background", "Scissor-", "Scissor+")))

cat(sprintf("  Plotting cells: %d\n", nrow(scissor_umap)))

# ============================================================================
# Panel A: SCISSOR UMAP — Scissor+ (red) / Scissor- (blue) / Background (grey)
# ============================================================================
cat("\n[Panel A] SCISSOR UMAP...\n")

p_scissor_umap <- ggplot(scissor_umap, aes(x = UMAP_1, y = UMAP_2, color = scissor)) +
  geom_point(data = filter(scissor_umap, scissor == "Background"),
             color = "grey85", size = 0.2, alpha = 0.4) +
  geom_point(data = filter(scissor_umap, scissor == "Scissor-"),
             color = "#377EB8", size = 0.8, alpha = 0.85) +
  geom_point(data = filter(scissor_umap, scissor == "Scissor+"),
             color = "#E41A1C", size = 0.8, alpha = 0.85) +
  scale_color_manual(values = c("Background" = "grey85", "Scissor-" = "#377EB8", "Scissor+" = "#E41A1C"),
                     name = "SCISSOR", drop = FALSE) +
  labs(x = "UMAP 1", y = "UMAP 2",
       title = "SCISSOR-selected cells on UMAP") +
  theme_pub +
  theme(legend.position.inside = c(0.02, 0.98),
        legend.justification = c(0, 1),
        plot.title = element_text(size = 10, face = "bold"))

ggsave(file.path(OUTDIR, "Fig5A_SCISSOR_UMAP.png"), p_scissor_umap, width = 7, height = 6, dpi = 300, bg = "white")
ggsave(file.path(OUTDIR_PDF, "Fig5A_SCISSOR_UMAP.pdf"), p_scissor_umap, width = 7, height = 6, bg = "white")
cat("  -> Fig5A saved\n")

# ============================================================================
# Panel B: Cell type enrichment — fold enrichment with significance
# ============================================================================
cat("[Panel B] Cell type enrichment...\n")

ct_long <- ct_enrich %>%
  select(cell_type, Scissor_plus_n = `Scissor+_n`, Scissor_plus_fold = `Scissor+_fold`,
         Scissor_plus_p = `Scissor+_p`, Scissor_minus_n = `Scissor-_n`,
         Scissor_minus_fold = `Scissor-_fold`, Scissor_minus_p = `Scissor-_p`) %>%
  mutate(Scissor_plus_sig = case_when(
           Scissor_plus_p < 0.001 ~ "***", Scissor_plus_p < 0.01 ~ "**",
           Scissor_plus_p < 0.05 ~ "*", TRUE ~ "NS"),
         Scissor_minus_sig = case_when(
           Scissor_minus_p < 0.001 ~ "***", Scissor_minus_p < 0.01 ~ "**",
           Scissor_minus_p < 0.05 ~ "*", TRUE ~ "NS")) %>%
  pivot_longer(cols = c(Scissor_plus_fold, Scissor_minus_fold),
               names_to = "group", values_to = "fold") %>%
  mutate(group = recode(group, Scissor_plus_fold = "Scissor+", Scissor_minus_fold = "Scissor-"),
         n = ifelse(group == "Scissor+", Scissor_plus_n, Scissor_minus_n),
         sig = ifelse(group == "Scissor+", Scissor_plus_sig, Scissor_minus_sig),
         cell_type = factor(cell_type, levels = ct_enrich$cell_type[order(ct_enrich$`Scissor+_fold`)]),
         label = sprintf("%.1f%s", fold, ifelse(sig == "NS", "", sig)))

p_ct_enrich <- ggplot(ct_long, aes(x = cell_type, y = fold, fill = group)) +
  geom_col(position = position_dodge(0.7), width = 0.6, alpha = 0.85) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey50", linewidth = 0.3) +
  geom_text(aes(label = label), position = position_dodge(0.7), vjust = -0.3, size = 2.5, fontface = "bold") +
  scale_fill_manual(values = c("Scissor+" = "#E41A1C", "Scissor-" = "#377EB8"), name = "SCISSOR") +
  labs(x = NULL, y = "Fold Enrichment",
       title = "Cell type enrichment of SCISSOR-selected cells") +
  theme_pub +
  theme(axis.text.x = element_text(angle = 35, hjust = 1, size = 8),
        legend.position.inside = c(0.9, 0.95),
        plot.title = element_text(size = 10, face = "bold"))

ggsave(file.path(OUTDIR, "Fig5B_SCISSOR_celltype_enrichment.png"), p_ct_enrich, width = 8, height = 5, dpi = 300, bg = "white")
ggsave(file.path(OUTDIR_PDF, "Fig5B_SCISSOR_celltype_enrichment.pdf"), p_ct_enrich, width = 8, height = 5, bg = "white")
cat("  -> Fig5B saved\n")

# ============================================================================
# Panel C: SCISSOR cell proportion by cell type (stacked bar)
# ============================================================================
cat("[Panel C] Cell proportion...\n")

ct_prop <- scissor_cells %>%
  filter(scissor_label %in% c("Scissor+", "Scissor-")) %>%
  count(cell_type, scissor_label) %>%
  group_by(cell_type) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ungroup() %>%
  mutate(cell_type = factor(cell_type),
         scissor_label = factor(scissor_label, levels = c("Scissor+", "Scissor-")))

p_ct_prop <- ggplot(ct_prop, aes(x = cell_type, y = pct, fill = scissor_label)) +
  geom_col(position = position_stack(reverse = TRUE), width = 0.7, alpha = 0.85) +
  geom_text(aes(label = ifelse(pct > 5, sprintf("%.0f%%", pct), "")),
            position = position_stack(vjust = 0.5, reverse = TRUE), size = 2.5, color = "white", fontface = "bold") +
  scale_fill_manual(values = c("Scissor+" = "#E41A1C", "Scissor-" = "#377EB8"), name = "SCISSOR") +
  labs(x = NULL, y = "Proportion (%)",
       title = "SCISSOR+/- proportion by cell type") +
  theme_pub +
  theme(axis.text.x = element_text(angle = 35, hjust = 1, size = 8),
        legend.position.inside = c(0.9, 0.95),
        plot.title = element_text(size = 10, face = "bold"))

ggsave(file.path(OUTDIR, "Fig5C_SCISSOR_celltype_proportion.png"), p_ct_prop, width = 7, height = 5, dpi = 300, bg = "white")
ggsave(file.path(OUTDIR_PDF, "Fig5C_SCISSOR_celltype_proportion.pdf"), p_ct_prop, width = 7, height = 5, bg = "white")
cat("  -> Fig5C saved\n")

# ============================================================================
# Panel D: Network overlap — top DE genes between Scissor+ vs Scissor-
# ============================================================================
cat("[Panel D] Network overlap DE genes...\n")

net_top <- net_overlap %>%
  filter(!is.na(pvalue), pvalue > 0, pvalue < 0.01, is.finite(log2FC), abs(log2FC) > 0.3) %>%
  arrange(pvalue) %>%
  head(25) %>%
  mutate(gene = factor(gene, levels = rev(gene)),
         direction = ifelse(log2FC > 0, "Up in Scissor+", "Down in Scissor+"),
         neg_log10_p = -log10(pvalue))

if (nrow(net_top) < 10) {
  net_top <- net_overlap %>%
    filter(!is.na(pvalue), pvalue > 0, is.finite(log2FC)) %>%
    arrange(pvalue) %>%
    head(25) %>%
    mutate(gene = factor(gene, levels = rev(gene)),
           direction = ifelse(log2FC > 0, "Up in Scissor+", "Down in Scissor+"),
           neg_log10_p = -log10(pvalue))
  cat(sprintf("  Used relaxed filter: %d genes\n", nrow(net_top)))
}

p_net_overlap <- ggplot(net_top, aes(x = log2FC, y = gene, fill = direction)) +
  geom_col(width = 0.7, alpha = 0.85) +
  geom_text(aes(label = sprintf("%.2f", log2FC)), hjust = ifelse(net_top$log2FC > 0, -0.2, 1.2),
            size = 2.3, fontface = "bold") +
  scale_fill_manual(values = c("Up in Scissor+" = "#E41A1C", "Down in Scissor+" = "#377EB8"), name = "Direction") +
  labs(x = "log2 Fold Change (Scissor+ vs Scissor-)", y = NULL,
       title = "Top DE genes: Scissor+ vs Scissor-") +
  theme_pub +
  theme(axis.text.y = element_text(face = "italic", size = 7.5),
        legend.position.inside = c(0.85, 0.15),
        plot.title = element_text(size = 10, face = "bold"))

ggsave(file.path(OUTDIR, "Fig5D_SCISSOR_network_DE.png"), p_net_overlap, width = 8, height = 6, dpi = 300, bg = "white")
ggsave(file.path(OUTDIR_PDF, "Fig5D_SCISSOR_network_DE.pdf"), p_net_overlap, width = 8, height = 6, bg = "white")
cat("  -> Fig5D saved\n")

# ============================================================================
# Panel E: Selection frequency distribution
# ============================================================================
cat("[Panel E] Selection frequency...\n")

sel_freq_plot <- scissor_cells %>%
  filter(selection_frequency > 0) %>%
  mutate(scissor_label = ifelse(is.na(scissor_label) | scissor_label == "", "Unselected", scissor_label))

p_sel_freq <- ggplot(sel_freq_plot, aes(x = selection_frequency, fill = scissor_label)) +
  geom_histogram(bins = 50, alpha = 0.7, color = "white", linewidth = 0.2) +
  scale_fill_manual(values = c("Scissor+" = "#E41A1C", "Scissor-" = "#377EB8", "Unselected" = "grey60"),
                    name = "SCISSOR") +
  labs(x = "Selection Frequency (bootstrap stability)", y = "Cell Count",
       title = "SCISSOR selection frequency distribution") +
  theme_pub +
  theme(legend.position.inside = c(0.9, 0.95),
        plot.title = element_text(size = 10, face = "bold"))

ggsave(file.path(OUTDIR, "Fig5E_SCISSOR_selection_freq.png"), p_sel_freq, width = 6, height = 4, dpi = 300, bg = "white")
ggsave(file.path(OUTDIR_PDF, "Fig5E_SCISSOR_selection_freq.pdf"), p_sel_freq, width = 6, height = 4, bg = "white")
cat("  -> Fig5E saved\n")

# ============================================================================
# 组图: Fig5 Composite
# ============================================================================
cat("\n--- Assembling SCISSOR composite ---\n")

fig5 <- (p_scissor_umap + labs(tag = "A")) /
        ((p_ct_enrich + labs(tag = "B")) | (p_ct_prop + labs(tag = "C"))) /
        ((p_net_overlap + labs(tag = "D")) | (p_sel_freq + labs(tag = "E"))) +
        plot_layout(heights = c(1, 0.9, 1)) &
        theme(plot.tag = element_text(face = "bold", size = 14))

ggsave(file.path(OUTDIR, "Fig5_Composite_SCISSOR.png"), fig5, width = 14, height = 16, dpi = 300, bg = "white")
ggsave(file.path(OUTDIR_PDF, "Fig5_Composite_SCISSOR.pdf"), fig5, width = 14, height = 16, bg = "white")
cat("  -> Fig5 composite saved\n")

cat("\n========================================\n")
cat("  SCISSOR figures complete!\n")
cat("========================================\n")
