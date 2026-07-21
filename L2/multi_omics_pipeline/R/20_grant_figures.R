# =============================================================================
# 铁衰老国自然申请书配图统一生成脚本
# 基于真实多组学数据，采用NPG期刊配色统一风格
# =============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggsci)
  library(patchwork)
  library(ComplexHeatmap)
  library(circlize)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggsignif)
  library(ggpubr)
  library(scales)
  library(grid)
  library(gridExtra)
  library(gghalves)
})

# -----------------------------------------------------------------------------
# 0. 全局主题与配色（NPG调色板，参考nanxstats/ggsci v5.0.0）
# -----------------------------------------------------------------------------

npg_colors <- c(
  "Ferroptosis"    = "#E64B35",
  "Senescence"     = "#3C5488",
  "Ferroaging"     = "#F39B7F",
  "Ferrosenescence"= "#7E6148",
  "BCP_Up"         = "#00A087",
  "BCP_Down"       = "#8491B4",
  "Nrf2"           = "#00A087",
  "SAT1"           = "#91D1C2",
  "X4HNE"          = "#DC0000",
  "Healthy"        = "#4DBBD5",
  "Penumbra"       = "#F39B7F",
  "InfarctCore"    = "#E64B35",
  "Other"          = "#B09C85"
)

region_colors <- c(
  "Healthy"    = "#4DBBD5",
  "Penumbra"   = "#F39B7F",
  "InfarctCore"= "#E64B35",
  "Other"      = "#B09C85"
)

evidence_colors <- c(
  "Moderate" = "#E64B35",
  "Weak"     = "#8491B4"
)

theme_nsfc <- function(base_size = 12) {
  theme_classic(base_size = base_size) +
    theme(
      plot.title = element_text(size = base_size + 2, face = "bold",
                                hjust = 0.5, family = "Arial"),
      axis.title = element_text(size = base_size, face = "bold", family = "Arial"),
      axis.text  = element_text(size = base_size - 1, color = "black", family = "Arial"),
      axis.line  = element_line(color = "black", linewidth = 0.6),
      legend.title = element_text(size = base_size, face = "bold", family = "Arial"),
      legend.text  = element_text(size = base_size - 1, family = "Arial"),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = margin(8, 8, 8, 8)
    )
}

panel_label <- function(label, x = -Inf, y = Inf, size = 5) {
  annotate("text", x = x, y = y, label = label, hjust = -0.3, vjust = 1.5,
           fontface = "bold", size = size, family = "Arial")
}

output_dir <- "D:/铁衰老 绝不重蹈覆辙/L2/multi_omics_pipeline/output/grant_figures"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

data_dir <- "D:/铁衰老 绝不重蹈覆辙/L2/multi_omics_pipeline/outputs/tables"
method_dir <- "D:/铁衰老 绝不重蹈覆辙/L2/multi_omics_pipeline/output/methodology_figures/tables"
cross_dir <- "D:/铁衰老 绝不重蹈覆辙/L2/multi_omics_pipeline/output/cross_omics_integration/tables"
kegg_dir <- "D:/铁衰老 绝不重蹈覆辙/L2/multi_omics_pipeline/output/kegg_pathway_integration/tables"
met_dir <- "D:/铁衰老 绝不重蹈覆辙/L2/multi_omics_pipeline/output/metabolomics/tables"

save_figure <- function(plot, filename, width = 8, height = 6) {
  pdf_path <- file.path(output_dir, paste0(filename, ".pdf"))
  png_path <- file.path(output_dir, paste0(filename, ".png"))
  ggsave(pdf_path, plot, width = width, height = height, device = cairo_pdf)
  ggsave(png_path, plot, width = width, height = height, dpi = 300, units = "in")
  message("Saved: ", pdf_path)
  message("Saved: ", png_path)
}

# =============================================================================
# 附图1：Bulk RNA-seq时序GSEA结果（NES热图 + FA-96 LFC热图）
# =============================================================================

generate_figS1 <- function() {
  message("\n=== Generating Figure S1: Bulk RNA-seq GSEA ===")

  gsea_sum <- read_csv(file.path(data_dir, "03_bulk_gsea_nes_summary.csv"),
                       show_col_types = FALSE)
  lfc_mat  <- read_csv(file.path(data_dir, "02_bulk_ferroaging_lfc_matrix.csv"),
                       show_col_types = FALSE)

  time_order <- c("3h", "12h", "24h", "3D", "7D")
  term_order <- c("Ferroptosis", "Ferroaging", "Ferrosenescence", "Senescence",
                  "BCP_Up", "BCP_Down")
  term_labels <- c(
    "Ferroptosis"    = "Ferroptosis",
    "Ferroaging"     = "Ferroaging (FA-96)",
    "Ferrosenescence"= "Ferrosenescence",
    "Senescence"     = "Senescence",
    "BCP_Up"         = "BCP Up-targets",
    "BCP_Down"       = "BCP Down-targets"
  )

  gsea_full <- gsea_sum %>%
    mutate(comparison = factor(comparison, levels = time_order),
           term = factor(term, levels = term_order)) %>%
    complete(comparison, term, fill = list(NES = NA, p.adjust = NA)) %>%
    mutate(star = case_when(
      is.na(p.adjust)  ~ "",
      p.adjust < 0.001 ~ "***",
      p.adjust < 0.01  ~ "**",
      p.adjust < 0.05  ~ "*",
      TRUE             ~ ""
    ))

  nes_matrix <- gsea_full %>%
    select(comparison, term, NES) %>%
    pivot_wider(names_from = comparison, values_from = NES) %>%
    arrange(factor(term, levels = term_order))
  nes_mat <- as.matrix(nes_matrix[, -1])
  rownames(nes_mat) <- term_labels[as.character(nes_matrix$term)]
  colnames(nes_mat) <- time_order

  star_matrix <- gsea_full %>%
    select(comparison, term, star) %>%
    pivot_wider(names_from = comparison, values_from = star) %>%
    arrange(factor(term, levels = term_order))
  star_mat <- as.matrix(star_matrix[, -1])
  rownames(star_mat) <- term_labels[as.character(star_matrix$term)]

  col_fun <- colorRamp2(c(-1, 0, 1, 2), c("#3C5488", "white", "#F39B7F", "#E64B35"))

  set.seed(42)
  top_genes <- lfc_mat %>%
    mutate(max_abs = pmax(abs(X3h), abs(X12h), abs(X24h), abs(X3D), abs(X7D))) %>%
    arrange(desc(max_abs)) %>%
    head(30)

  lfc_plot <- as.matrix(top_genes[, c("X3h", "X12h", "X24h", "X3D", "X7D")])
  rownames(lfc_plot) <- top_genes$gene
  colnames(lfc_plot) <- time_order

  lfc_fun <- colorRamp2(c(-1, 0, 1), c("#3C5488", "white", "#E64B35"))

  ht1 <- Heatmap(nes_mat, name = "NES",
                 col = col_fun,
                 cluster_rows = FALSE, cluster_columns = FALSE,
                 cell_fun = function(j, i, x, y, width, height, fill) {
                   if (!is.na(nes_mat[i, j])) {
                     grid.text(sprintf("%.2f", nes_mat[i, j]), x, y,
                               gp = gpar(fontsize = 9, col = "black"))
                     if (star_mat[i, j] != "") {
                       grid.text(star_mat[i, j], x + unit(0.35, "cm"), y,
                                 gp = gpar(fontsize = 8, col = "#E64B35"))
                     }
                   } else {
                     grid.text("ns", x, y, gp = gpar(fontsize = 8, col = "grey70"))
                   }
                 },
                 row_names_gp = gpar(fontsize = 10, fontface = "bold"),
                 column_names_gp = gpar(fontsize = 10, fontface = "bold"),
                 column_title = "GSEA NES (vs Ctrl)",
                 column_title_gp = gpar(fontsize = 12, fontface = "bold"),
                 border = TRUE)

  ht2 <- Heatmap(lfc_plot, name = "log2FC",
                 col = lfc_fun,
                 cluster_rows = TRUE, cluster_columns = FALSE,
                 show_row_names = TRUE,
                 row_names_gp = gpar(fontsize = 7),
                 column_names_gp = gpar(fontsize = 10, fontface = "bold"),
                 column_title = "FA-96 Top 30 Genes (log2FC vs Ctrl)",
                 column_title_gp = gpar(fontsize = 12, fontface = "bold"),
                 border = TRUE)

  ht_list <- ht1 %v% ht2
  png(file.path(output_dir, "FigS1_bulk_gsea.png"), width = 8, height = 9,
      units = "in", res = 300)
  draw(ht_list, column_title = "Appendix Figure 1: Temporal GSEA of Ferroptosis/Senescence Signatures in MCAO Bulk RNA-seq",
       column_title_gp = gpar(fontsize = 14, fontface = "bold"))
  dev.off()
  pdf(file.path(output_dir, "FigS1_bulk_gsea.pdf"), width = 8, height = 9)
  draw(ht_list, column_title = "Appendix Figure 1: Temporal GSEA of Ferroptosis/Senescence Signatures in MCAO Bulk RNA-seq",
       column_title_gp = gpar(fontsize = 14, fontface = "bold"))
  dev.off()
  message("Saved FigS1_bulk_gsea.pdf/png")
}

# =============================================================================
# 附图2：空间转录组铁衰老定位（区域得分小提琴图 + 神经元比例相关图）
# =============================================================================

generate_figS2 <- function() {
  message("\n=== Generating Figure S2: Spatial Transcriptomics ===")

  region_scores <- read_csv(file.path(data_dir, "06_spatial_region_scores.csv"),
                            show_col_types = FALSE)
  neuron_fp <- read_csv(file.path(data_dir, "10_neuron_prop_vs_ferroptosis.csv"),
                        show_col_types = FALSE)

  region_order <- c("Healthy", "Penumbra", "InfarctCore", "Other")
  region_scores <- region_scores %>%
    filter(region %in% region_order) %>%
    mutate(region = factor(region, levels = region_order))

  plot_data <- region_scores %>%
    select(region, Ferroptosis, Ferrosenescence) %>%
    pivot_longer(cols = c(Ferroptosis, Ferrosenescence),
                 names_to = "Signature", values_to = "Score") %>%
    mutate(Signature = factor(Signature,
                              levels = c("Ferroptosis", "Ferrosenescence")))

  comparisons <- list(c("Healthy", "Penumbra"), c("Penumbra", "InfarctCore"),
                      c("Healthy", "InfarctCore"))

  p_violin <- ggplot(plot_data, aes(x = region, y = Score, fill = region)) +
    geom_half_violin(side = "r", position = position_nudge(x = 0.2),
                     alpha = 0.7, color = NA) +
    geom_boxplot(width = 0.12, fill = "white", outlier.shape = NA,
                 position = position_nudge(x = 0.2)) +
    geom_jitter(aes(color = region), size = 0.3, alpha = 0.15,
                position = position_jitter(width = 0.05)) +
    facet_wrap(~Signature, scales = "free_y", nrow = 1) +
    geom_signif(comparisons = comparisons,
                map_signif_level = function(p) sprintf("p = %.2g", p),
                step_increase = 0.08, textsize = 3, tip_length = 0.005,
                vjust = -0.2) +
    scale_fill_manual(values = region_colors) +
    scale_color_manual(values = region_colors, guide = "none") +
    labs(x = "Brain Region", y = "UCell Score") +
    theme_nsfc(base_size = 11) +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 30, hjust = 1),
          strip.text = element_text(face = "bold", size = 12))

  neuron_plot <- neuron_fp %>%
    filter(region %in% c("Healthy", "Penumbra", "InfarctCore"))

  cor_test <- cor.test(neuron_plot$neuron_prop, neuron_plot$fp_score,
                        method = "spearman")
  rho_val <- round(cor_test$estimate, 3)
  p_val <- cor_test$p.value
  p_label <- ifelse(p_val < 2.22e-16, "p < 2.2e-16", sprintf("p = %.2g", p_val))

  p_cor <- ggplot(neuron_plot, aes(x = neuron_prop, y = fp_score, color = region)) +
    geom_point(alpha = 0.25, size = 0.8) +
    geom_smooth(method = "lm", se = TRUE, alpha = 0.15, linewidth = 0.8) +
    scale_color_manual(values = region_colors) +
    annotate("text", x = Inf, y = Inf,
             label = sprintf("Spearman rho = %s\n%s", rho_val, p_label),
             hjust = 1.1, vjust = 1.2, size = 3.5, fontface = "bold",
             color = "#3C5488") +
    labs(x = "Neuron Proportion", y = "Ferroptosis Score",
         color = "Region") +
    theme_nsfc(base_size = 11) +
    theme(legend.position = c(0.85, 0.85),
          legend.background = element_rect(fill = "white", color = "grey80"))

  p_combined <- (p_violin | p_cor) +
    plot_annotation(tag_levels = "A",
                    title = "Appendix Figure 2: Spatial Localization of Iron-Aging Signals in CIRI Penumbra",
                    theme = theme(plot.title = element_text(size = 14, face = "bold",
                                                            hjust = 0.5, family = "Arial")))

  save_figure(p_combined, "FigS2_spatial_iron_aging", width = 12, height = 5)
}

# =============================================================================
# 附图3：单细胞SAT1验证与铁衰老细胞分布
# =============================================================================

generate_figS3 <- function() {
  message("\n=== Generating Figure S3: Single-cell SAT1 ===")

  sat1_fp <- read_csv(file.path(data_dir, "08_sat1_vs_ferroptosis_score.csv"),
                      show_col_types = FALSE)
  cor_by_ct <- read_csv(file.path(data_dir, "08_sat1_ferroptosis_correlation_by_celltype.csv"),
                        show_col_types = FALSE)
  augur <- read_csv(file.path(data_dir, "09_augur_auc_ranking.csv"),
                    show_col_types = FALSE)
  props <- read_csv(file.path(data_dir, "08_ferrosenescence_proportions.csv"),
                    show_col_types = FALSE)

  celltype_colors <- c(
    "NeuronsGABA" = "#E64B35", "NeuronsGLUT" = "#F39B7F",
    "OLs" = "#00A087", "OPCs" = "#3C5488",
    "Astrocytes" = "#8491B4", "Microglia" = "#B09C85",
    "EndothelialCells" = "#4DBBD5", "EpendymalCells" = "#91D1C2",
    "VLMCs" = "#DC0000", "Neuroblasts" = "#7E6148"
  )

  p_scatter <- ggplot(sat1_fp, aes(x = Sat1, y = FP_Score, color = Celltypes)) +
    geom_point(alpha = 0.3, size = 0.8) +
    geom_smooth(method = "lm", se = FALSE, linewidth = 0.6) +
    scale_color_manual(values = celltype_colors) +
    labs(x = "SAT1 Expression", y = "Ferroptosis UCell Score",
         color = "Cell Type") +
    theme_nsfc(base_size = 10) +
    theme(legend.position = "right",
          legend.text = element_text(size = 7),
          legend.key.size = unit(0.3, "cm"))

  cor_plot <- cor_by_ct %>%
    mutate(cell_type = factor(cell_type, levels = cell_type),
           sig = ifelse(padj < 0.05, "Significant", "NS"))

  p_cor_bar <- ggplot(cor_plot, aes(x = rho, y = reorder(cell_type, rho),
                                     fill = sig)) +
    geom_col(width = 0.6) +
    geom_errorbarh(aes(xmin = rho - 0.02, xmax = rho + 0.02), height = 0.2) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
    scale_fill_manual(values = c("Significant" = "#E64B35", "NS" = "#B09C85")) +
    labs(x = "Spearman rho (SAT1 vs Ferroptosis)", y = "",
         fill = "Significance") +
    theme_nsfc(base_size = 10) +
    theme(legend.position = "top")

  augur_plot <- augur %>%
    mutate(cell_type = factor(cell_type, levels = rev(cell_type)))

  p_augur <- ggplot(augur_plot, aes(x = AUC, y = cell_type)) +
    geom_segment(aes(x = 0.5, xend = AUC, y = cell_type, yend = cell_type),
                 color = "grey70", linewidth = 0.8) +
    geom_point(aes(color = AUC), size = 5) +
    geom_vline(xintercept = 0.5, linetype = "dashed", color = "grey50") +
    scale_color_gradient(low = "#4DBBD5", high = "#E64B35",
                         limits = c(0.5, 0.56)) +
    labs(x = "Augur AUC (Perturbation Priority)", y = "",
         color = "AUC") +
    theme_nsfc(base_size = 10) +
    theme(legend.position = "right",
          legend.key.size = unit(0.4, "cm"))

  prop_plot <- props %>%
    filter(status == "Ferrosenescence_High") %>%
    mutate(condition = factor(condition, levels = c("Ctrl", "1DPI", "3DPI", "7DPI")),
           celltype = factor(celltype, levels = unique(celltype)))

  p_prop <- ggplot(prop_plot, aes(x = condition, y = proportion, fill = celltype)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    scale_fill_manual(values = celltype_colors) +
    labs(x = "Condition", y = "Ferrosenescence_High Proportion",
         fill = "Cell Type") +
    theme_nsfc(base_size = 10) +
    theme(legend.position = "right",
          legend.text = element_text(size = 7),
          legend.key.size = unit(0.3, "cm"),
          axis.text.x = element_text(angle = 30, hjust = 1))

  p_combined <- (p_scatter | p_cor_bar) / (p_augur | p_prop) +
    plot_annotation(tag_levels = "A",
                    title = "Appendix Figure 3: Single-cell SAT1 Validation and Ferrosenescence Cell Distribution",
                    theme = theme(plot.title = element_text(size = 14, face = "bold",
                                                            hjust = 0.5, family = "Arial")))

  save_figure(p_combined, "FigS3_singlecell_sat1", width = 13, height = 9)
}

# =============================================================================
# 附图4：代谢组跨组学整合（SAT1-多胺瀑布 + 通路轴匹配率 + KEGG通路）
# =============================================================================

generate_figS4 <- function() {
  message("\n=== Generating Figure S4: Metabolomics Cross-omics ===")

  axis_table <- read_csv(file.path(cross_dir, "cross_omics_axis_table.csv"),
                         show_col_types = FALSE,
                         locale = locale(encoding = "UTF-8"))
  table1 <- read_csv(file.path(method_dir, "Table1_cross_omics_evidence.csv"),
                     show_col_types = FALSE)
  kegg_paths <- read_csv(file.path(kegg_dir, "cross_omics_shared_pathways.csv"),
                         show_col_types = FALSE)

  sat1_meta <- axis_table %>%
    filter(axis_name == "SAT1-polyamine") %>%
    mutate(display_name = factor(display_name,
                                 levels = unique(display_name)))

  p_waterfall <- ggplot(sat1_meta, aes(x = display_name, y = log2FC_aging,
                                       fill = direction_aging)) +
    geom_col(width = 0.7) +
    geom_hline(yintercept = 0, color = "black", linewidth = 0.5) +
    geom_text(aes(label = ifelse(p_adj_aging < 0.001, "***",
                                 ifelse(p_adj_aging < 0.01, "**",
                                        ifelse(p_adj_aging < 0.05, "*", "")))),
              vjust = ifelse(sat1_meta$log2FC_aging > 0, -0.5, 1.5),
              size = 3.5, color = "#3C5488") +
    scale_fill_manual(values = c("DOWN" = "#3C5488", "UP" = "#E64B35"),
                      name = "Direction") +
    labs(x = "Metabolite", y = "log2FC (59w vs 3w)",
         title = "SAT1-Polyamine Axis Metabolites") +
    theme_nsfc(base_size = 10) +
    theme(axis.text.x = element_text(angle = 35, hjust = 1, size = 8),
          plot.title = element_text(size = 11),
          legend.position = "top")

  table1_plot <- table1 %>%
    mutate(match_num = as.numeric(sub("/.*", "", `Match Rate`)),
           match_den = as.numeric(sub(".*/", "", `Match Rate`)),
           match_pct = match_num / match_den,
           `Pathway Axis` = factor(`Pathway Axis`,
                                   levels = `Pathway Axis`[order(match_pct)]),
           evidence_color = ifelse(Evidence == "Moderate", "#E64B35", "#8491B4"))

  p_axis_bar <- ggplot(table1_plot, aes(x = match_pct, y = `Pathway Axis`,
                                        fill = Evidence)) +
    geom_col(width = 0.6) +
    geom_text(aes(label = `Match Rate`), hjust = -0.1, size = 3) +
    geom_vline(xintercept = 0.7, linetype = "dashed", color = "#E64B35",
               linewidth = 0.6) +
    annotate("text", x = 0.72, y = 1, label = "Strong (70%)",
             color = "#E64B35", size = 3, hjust = 0) +
    scale_fill_manual(values = evidence_colors) +
    labs(x = "Match Rate", y = "Pathway Axis", fill = "Evidence") +
    xlim(0, 1.0) +
    theme_nsfc(base_size = 10) +
    theme(legend.position = "top")

  kegg_top <- kegg_paths %>%
    mutate(pathway_short = gsub(" - Mus musculus.*", "", pathway_name),
           pathway_short = substr(pathway_short, 1, 40)) %>%
    arrange(desc(total_omics_elements)) %>%
    head(12) %>%
    mutate(pathway_short = factor(pathway_short,
                                  levels = pathway_short[order(total_omics_elements)]))

  p_kegg <- ggplot(kegg_top, aes(x = total_omics_elements, y = pathway_short,
                                 fill = n_genes)) +
    geom_col(width = 0.6) +
    geom_text(aes(label = total_omics_elements), hjust = -0.2, size = 3) +
    scale_fill_gradient(low = "#4DBBD5", high = "#E64B35", name = "n_genes") +
    labs(x = "Total Omics Elements (Genes + Metabolites)", y = "",
         title = "Top 12 Shared KEGG Pathways") +
    xlim(0, max(kegg_top$total_omics_elements) * 1.15) +
    theme_nsfc(base_size = 9) +
    theme(axis.text.y = element_text(size = 8),
          plot.title = element_text(size = 11),
          legend.position = "right",
          legend.key.size = unit(0.4, "cm"))

  p_combined <- (p_waterfall | p_axis_bar) / p_kegg +
    plot_layout(heights = c(1, 1.2)) +
    plot_annotation(tag_levels = "A",
                    title = "Appendix Figure 4: Metabolomics Cross-omics Integration of SAT1-Polyamine Axis",
                    theme = theme(plot.title = element_text(size = 14, face = "bold",
                                                            hjust = 0.5, family = "Arial")))

  save_figure(p_combined, "FigS4_metabolomics_crossomics", width = 14, height = 9)
}

# =============================================================================
# 图1：核心假说与多组学预验证整合示意图
# =============================================================================

generate_fig1 <- function() {
  message("\n=== Generating Figure 1: Core Hypothesis Schematic ===")

  cmap_all <- read_csv(file.path(data_dir, "12_fgsea_bcp_all_timepoints.csv"),
                       show_col_types = FALSE)

  cmap_summary <- cmap_all %>%
    group_by(comparison) %>%
    summarise(BCP_Up_NES = NES[pathway == "BCP_Up"],
              BCP_Down_NES = ifelse(any(pathway == "BCP_Down"),
                                    NES[pathway == "BCP_Down"], NA),
              .groups = "drop")

  cmap_plot <- cmap_summary %>%
    pivot_longer(cols = ends_with("_NES"), names_to = "Pathway",
                 values_to = "NES") %>%
    mutate(comparison = factor(comparison, levels = c("3h", "12h", "24h", "3D", "7D")),
           Pathway = recode(Pathway, "BCP_Up_NES" = "BCP Up-targets",
                                      "BCP_Down_NES" = "BCP Down-targets"))

  p_cmap <- ggplot(cmap_plot, aes(x = comparison, y = NES, fill = Pathway)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6) +
    geom_hline(yintercept = 0, color = "black", linewidth = 0.5) +
    scale_fill_manual(values = c("BCP Up-targets" = "#00A087",
                                  "BCP Down-targets" = "#8491B4")) +
    labs(x = "Time after Reperfusion", y = "NES (BCP signature vs Ischemia)",
         title = "CMap Reversal Analysis") +
    theme_nsfc(base_size = 10) +
    theme(legend.position = "top",
          plot.title = element_text(size = 11))

  node_df <- data.frame(
    x = c(2, 3.5, 5, 5, 6.5, 6.5, 5, 8, 8, 5, 2, 3.5),
    y = c(8, 8, 9, 7, 9, 7, 5, 9, 7, 3, 5, 5),
    label = c("Iron\nOverload", "Lipid\nPeroxidation", "4-HNE\n(Low)",
              "4-HNE\n(High)", "Keap1\nModified", "DNA\nDamage",
              "Nrf2\nActivated", "Antioxidant\nDefense", "p53\nPhosphorylated",
              "Ferroptosis\nCycle", "SAT1\nAmplifier", "Polyamine\nDepletion"),
    type = c("driver", "process", "molecule_low", "molecule_high",
             "target_low", "target_high", "defense", "defense",
             "aging", "outcome", "amplifier", "metabolite")
  )

  node_fill <- c("#E64B35", "#F39B7F", "#F39B7F", "#DC0000", "#00A087",
                 "#3C5488", "#00A087", "#00A087", "#3C5488", "#E64B35",
                 "#91D1C2", "#8491B4")

  arrow_df <- data.frame(
    x = c(2.4, 3.9, 4.6, 5.4, 5.4, 6.9, 6.9, 5, 5.3, 8.4, 5, 2.4, 3.9),
    y = c(8, 8, 8.7, 9, 7, 9, 7, 5, 3.5, 9, 4.2, 7, 5),
    xend = c(3.1, 4.6, 5, 6.1, 6.1, 7.6, 7.6, 6.5, 5, 8.5, 5, 3, 4.4),
    yend = c(8, 8, 9, 9, 7, 9, 7, 5, 3, 9, 5, 5.5, 5),
    type = c("positive", "positive", "positive_low", "positive",
             "positive_high", "positive", "positive_high",
             "inhibit", "positive", "inhibit", "feedback",
             "positive", "positive")
  )

  p_scheme <- ggplot() +
    geom_segment(data = arrow_df,
                 aes(x = x, y = y, xend = xend, yend = yend,
                     linetype = type, color = type),
                 arrow = arrow(length = unit(0.15, "cm"), type = "closed"),
                 linewidth = 0.7, show.legend = FALSE) +
    scale_linetype_manual(values = c("positive" = "solid",
                                      "positive_low" = "solid",
                                      "positive_high" = "solid",
                                      "inhibit" = "dashed",
                                      "feedback" = "dotted")) +
    scale_color_manual(values = c("positive" = "#3C5488",
                                   "positive_low" = "#00A087",
                                   "positive_high" = "#DC0000",
                                   "inhibit" = "#E64B35",
                                   "feedback" = "#7E6148")) +
    geom_point(data = node_df, aes(x = x, y = y), shape = 21,
               size = 11, fill = "white", color = "black", stroke = 0.8) +
    geom_text(data = node_df, aes(x = x, y = y, label = label),
              size = 2.8, fontface = "bold", family = "Arial",
              lineheight = 0.85) +
    annotate("text", x = 7.5, y = 5, label = "BCP Intervention",
             size = 4, fontface = "bold", color = "#00A087") +
    annotate("segment", x = 6.5, y = 5.3, xend = 7.2, yend = 5,
             arrow = arrow(length = unit(0.15, "cm")),
             color = "#00A087", linewidth = 1) +
    annotate("text", x = 7.5, y = 4.5, label = "SAT1 down | Nrf2 up | 4-HNE down",
             size = 2.5, color = "#00A087", fontface = "italic") +
    annotate("rect", xmin = 4.5, ymin = 4, xmax = 5.5, ymax = 6,
             alpha = 0.1, fill = "#E64B35") +
    annotate("text", x = 5, y = 2.5,
             label = "Threshold: 4-HNE switches from Nrf2 defense to p53 aging",
             size = 2.8, fontface = "italic", color = "#DC0000") +
    annotate("segment", x = 4.5, y = 2.7, xend = 5.5, yend = 2.7,
             linetype = "dashed", color = "#DC0000") +
    xlim(1, 9) + ylim(2, 9.5) +
    labs(title = "Core Hypothesis: 4-HNE-p53-SLC7A11 Positive Feedback Loop") +
    theme_void(base_size = 10) +
    theme(plot.title = element_text(size = 12, face = "bold", hjust = 0.5))

  p_combined <- (p_scheme | p_cmap) +
    plot_layout(widths = c(2, 1)) +
    plot_annotation(tag_levels = "A",
                    title = "Figure 1: Core Hypothesis and Multi-omics Pre-validation Schematic",
                    theme = theme(plot.title = element_text(size = 14, face = "bold",
                                                            hjust = 0.5, family = "Arial")))

  save_figure(p_combined, "Fig1_hypothesis_schematic", width = 14, height = 7)
}

# =============================================================================
# 图2：技术路线图
# =============================================================================

generate_fig2 <- function() {
  message("\n=== Generating Figure 2: Technical Roadmap ===")

  block_df <- data.frame(
    x = c(5, 5, 5, 5),
    y = c(9, 7, 5, 3),
    label = c(
      "Content 1: Spatiotemporal Localization\nMCAO/R 6h/24h/3d/7d\nMulti-IF | TEM | SA-ss-gal | LC-MS/MS | IP/IB\n-> 4-HNE threshold definition",
      "Content 2: Cellular Mechanism\nPrimary neurons + astrocytes\nErastin/OGD/R -> BCP + ML385 + siNrf2\nCETSA | SPR | 4-HNE-IP/IB",
      "Content 3: In vivo Efficacy\nMCAO/R Immediate / 3h-delay / 6h-delay\n7d gavage -> 28d endpoint\nmNSS | Rota-rod | Maze | TTC",
      "Content 4: Herb-Efficacy-Compound-Target\nGC-MS | GNN | Docking | Network\n-> Association spectrum"
    ),
    color = c("#E64B35", "#00A087", "#3C5488", "#F39B7F")
  )

  arrow_df <- data.frame(x = 5, y = c(8.2, 6.2, 4.2), xend = 5,
                         yend = c(7.8, 5.8, 3.8))

  p <- ggplot() +
    geom_rect(data = block_df,
              aes(xmin = 1, xmax = 9, ymin = y - 0.7, ymax = y + 0.7),
              fill = block_df$color, alpha = 0.15, color = block_df$color,
              linewidth = 1) +
    geom_text(data = block_df, aes(x = x, y = y, label = label),
              size = 3.2, fontface = "bold", family = "Arial",
              lineheight = 0.9) +
    geom_segment(data = arrow_df, aes(x = x, y = y, xend = xend, yend = yend),
                 arrow = arrow(length = unit(0.2, "cm"), type = "closed"),
                 color = "#3C5488", linewidth = 1) +
    annotate("text", x = 9.3, y = 9, label = "Iron-Aging Hypothesis\nPre-validation",
             size = 3, fontface = "italic", color = "#7E6148") +
    annotate("segment", x = 8.8, y = 9, xend = 9.2, yend = 9,
             arrow = arrow(length = unit(0.15, "cm")),
             color = "#7E6148", linewidth = 0.8) +
    annotate("text", x = 0.7, y = 1.5,
             label = "Zhuang Medicine Theory: 'Clear Poison-Evil, Unblock Dragon-Fire Pathways'",
             size = 3.5, fontface = "italic", color = "#7E6148") +
    annotate("text", x = 5, y = 0.8,
             label = "Beta-Caryophyllene (BCP) from Artemisia argyi volatile oil",
             size = 3.5, fontface = "bold", color = "#00A087") +
    xlim(0, 11) + ylim(0.5, 9.8) +
    labs(title = "Figure 2: Technical Roadmap of Four Research Contents") +
    theme_void(base_size = 11) +
    theme(plot.title = element_text(size = 13, face = "bold", hjust = 0.5))

  save_figure(p, "Fig2_technical_roadmap", width = 10, height = 8)
}

# =============================================================================
# 表1：跨组学通路轴验证汇总表
# =============================================================================

generate_table1 <- function() {
  message("\n=== Generating Table 1: Cross-omics Axis Summary ===")

  table1 <- read_csv(file.path(method_dir, "Table1_cross_omics_evidence.csv"),
                     show_col_types = FALSE)

  table1_sorted <- table1 %>%
    mutate(match_num = as.numeric(sub("/.*", "", `Match Rate`)),
           match_den = as.numeric(sub(".*/", "", `Match Rate`)),
           match_pct = match_num / match_den) %>%
    arrange(desc(match_pct)) %>%
    select(`Pathway Axis`, `Driver Gene`, Evidence, `N Metabolites`,
           `Match Rate`, `Mean log2FC`, `Key Metabolites`)

  tt <- ttheme_minimal(
    core = list(bg_params = list(fill = ifelse(table1_sorted$Evidence == "Moderate",
                                                "#FEEBE2", "#F0F0F0")),
                text_params = list(cex = 0.9, fontface = "bold")),
    colhead = list(bg_params = list(fill = "#3C5488"),
                   text_params = list(col = "white", cex = 0.9, fontface = "bold")),
    rowhead = list(bg_params = list(fill = "#E64B35"),
                   text_params = list(col = "white", cex = 0.9))
  )

  table_mat <- as.matrix(table1_sorted)

  png(file.path(output_dir, "Table1_cross_omics.png"), width = 12, height = 4,
      units = "in", res = 300)
  grid.newpage()
  grid.text("Table 1: Cross-omics Pathway Axis Validation Summary",
            x = 0.5, y = 0.95, gp = gpar(fontsize = 13, fontface = "bold"))
  grid.table(table_mat, theme = tt)
  dev.off()

  pdf(file.path(output_dir, "Table1_cross_omics.pdf"), width = 12, height = 4)
  grid.newpage()
  grid.text("Table 1: Cross-omics Pathway Axis Validation Summary",
            x = 0.5, y = 0.95, gp = gpar(fontsize = 13, fontface = "bold"))
  grid.table(table_mat, theme = tt)
  dev.off()

  message("Saved Table1_cross_omics.pdf/png")
}

# =============================================================================
# 主程序入口
# =============================================================================

main <- function() {
  message("=============================================")
  message("Iron-Aging NSFC Grant Figures Generator")
  message("Based on real multi-omics data (GSE233815, ST001637)")
  message("Color palette: NPG (nanxstats/ggsci v5.0.0)")
  message("=============================================")

  generate_figS1()
  generate_figS2()
  generate_figS3()
  generate_figS4()
  generate_fig1()
  generate_fig2()
  generate_table1()

  message("\n=============================================")
  message("All figures generated successfully!")
  message("Output directory: ", output_dir)
  message("=============================================")
}

main()
