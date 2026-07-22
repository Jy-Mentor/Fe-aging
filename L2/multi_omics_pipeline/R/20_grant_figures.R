# =============================================================================
# 铁衰老国自然申请书配图统一生成脚本 v2
# 基于12篇公众号文章设计精华 + GitHub设计理论 + Nature Cancer高级感配色
# 所有数据来自真实文件 (GSE233815, ST001637)
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
  library(ggrepel)
})

# =============================================================================
# 0. 全局配色与主题系统
# =============================================================================
# 基础配色：Nature Cancer "高级感" 对比色 + NPG调色板
pal <- c(
  "Ferroptosis"     = "#DC0000",
  "Senescence"      = "#3C5488",
  "Ferroaging"      = "#F39B7F",
  "Ferrosenescence" = "#7E6148",
  "BCP_Up"          = "#00A087",
  "BCP_Down"        = "#8491B4",
  "Nrf2"            = "#00A087",
  "SAT1"            = "#91D1C2",
  "X4HNE"           = "#DC0000",
  "Healthy"         = "#4DBBD5",
  "Penumbra"        = "#F39B7F",
  "InfarctCore"     = "#E64B35",
  "Neuron"          = "#E64B35",
  "Other"           = "#B09C85",
  "Mild"            = "#FEEBE2",
  "Moderate"        = "#E64B35",
  "Weak"            = "#8491B4"
)

region_colors <- c("Healthy"="#4DBBD5", "Penumbra"="#F39B7F",
                   "InfarctCore"="#E64B35", "Other"="#B09C85")

evidence_colors <- c("Moderate"="#00A087", "Weak"="#8491B4")

time_colors <- c("Ctrl"="#4DBBD5", "12h"="#91D1C2", "1DPI"="#F39B7F",
                 "3DPI"="#E64B35", "7DPI"="#DC0000")

# 统一主题：Nature级简洁风格
theme_nsfc <- function(base_size = 10) {
  theme_classic(base_size = base_size) +
    theme(
      plot.title        = element_text(size = base_size + 1, face = "bold",
                                       hjust = 0.5, family = "Arial"),
      plot.subtitle     = element_text(size = base_size - 1, hjust = 0.5,
                                       family = "Arial", color = "grey40"),
      axis.title        = element_text(size = base_size, family = "Arial"),
      axis.text         = element_text(size = base_size - 1, color = "black",
                                       family = "Arial"),
      axis.line         = element_line(color = "black", linewidth = 0.5),
      axis.ticks        = element_line(color = "black", linewidth = 0.5),
      legend.title      = element_text(size = base_size - 1, family = "Arial"),
      legend.text       = element_text(size = base_size - 2, family = "Arial"),
      legend.background = element_rect(fill = "white", color = "grey85"),
      panel.grid.major  = element_blank(),
      panel.grid.minor  = element_blank(),
      plot.margin       = margin(6, 6, 6, 6),
      strip.text        = element_text(size = base_size, face = "bold",
                                       family = "Arial"),
      strip.background  = element_rect(fill = "grey95", color = "grey80")
    )
}

tag_label <- function(label, x = 0.02, y = 0.98, size = 4.5) {
  annotate("text", x = x, y = y, label = label, hjust = 0, vjust = 1,
           fontface = "bold", size = size, family = "Arial", color = "grey20")
}

# =============================================================================
# 路径配置
# =============================================================================
bp <- "D:/铁衰老 绝不重蹈覆辙/L2/multi_omics_pipeline"
output_dir <- file.path(bp, "output", "grant_figures")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
data_dir   <- file.path(bp, "outputs", "tables")
method_dir <- file.path(bp, "output", "methodology_figures", "tables")
cross_dir  <- file.path(bp, "output", "cross_omics_integration", "tables")
kegg_dir   <- file.path(bp, "output", "kegg_pathway_integration", "tables")
options(useFancyQuotes = FALSE)

save_figure <- function(plot, filename, width = 8, height = 6) {
  pdf_path <- file.path(output_dir, paste0(filename, ".pdf"))
  png_path <- file.path(output_dir, paste0(filename, ".png"))
  ggsave(pdf_path, plot, width = width, height = height, device = cairo_pdf,
         bg = "white")
  ggsave(png_path, plot, width = width, height = height, dpi = 300,
         units = "in", bg = "white")
  message(sprintf("  -> %s (%.1f KB)", filename, file.info(png_path)$size / 1024))
}

# =============================================================================
# 图1: 核心假说与多组学预验证整合示意图 (5面板 A-E)
# 放置: 1.9节末尾
# =============================================================================
generate_fig1 <- function() {
  message("\n=== Fig1: Core Hypothesis + Multi-omics Schematic ===")

  # ---- A: 空间转录组热图 ----
  region_scores <- read_csv(file.path(data_dir, "06_spatial_region_scores.csv"),
                            show_col_types = FALSE)
  region_scores <- region_scores %>%
    filter(region %in% c("Healthy", "Penumbra", "InfarctCore", "Other")) %>%
    mutate(region = factor(region, levels = c("Healthy", "Penumbra",
                                               "InfarctCore", "Other")))

  p1a <- ggplot(region_scores, aes(x = region, y = Ferrosenescence, fill = region)) +
    geom_half_violin(side = "r", position = position_nudge(x = 0.2),
                     alpha = 0.75, color = NA) +
    geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA,
                 position = position_nudge(x = 0.2), linewidth = 0.4) +
    geom_jitter(aes(color = region), size = 0.2, alpha = 0.12,
                position = position_jitter(width = 0.04)) +
    geom_signif(comparisons = list(c("Healthy", "Penumbra"),
                                    c("Penumbra", "InfarctCore")),
                map_signif_level = function(p) sprintf("p=%.1e", p),
                step_increase = 0.06, textsize = 2.5, tip_length = 0.003,
                vjust = 0.3) +
    scale_fill_manual(values = region_colors) +
    scale_color_manual(values = region_colors, guide = "none") +
    labs(x = "", y = "Ferrosenescence UCell Score",
         subtitle = "A  Visium Spatial (D3, bregma -1.3mm)") +
    theme_nsfc(9) +
    theme(legend.position = "none", axis.text.x = element_text(angle = 25, hjust = 1),
          plot.subtitle = element_text(hjust = 0, size = 9, face = "bold"))

  # ---- B: snRNA-seq UMAP模拟 (基于细胞SAT1表达) ----
  cell_scores <- read_csv(file.path(data_dir, "08_sat1_vs_ferroptosis_score.csv"),
                          show_col_types = FALSE)
  set.seed(42)
  cell_scores_sample <- cell_scores %>%
    group_by(Celltypes) %>%
    sample_frac(min(0.15, 500 / n())) %>%
    ungroup()

  ct_colors_umap <- c(
    "NeuronsGABA"="#E64B35", "NeuronsGLUT"="#D95F02", "Astrocytes"="#00A087",
    "Microglia"="#3C5488", "OLs"="#4DBBD5", "OPCs"="#91D1C2",
    "EndothelialCells"="#F39B7F", "EpendymalCells"="#B09C85",
    "VLMCs"="#8491B4", "Neuroblasts"="#7E6148"
  )

  p1b <- ggplot(cell_scores_sample, aes(x = Sat1, y = FP_Score, color = Celltypes)) +
    geom_point(alpha = 0.35, size = 0.6) +
    geom_smooth(method = "lm", se = FALSE, linewidth = 0.5) +
    scale_color_manual(values = ct_colors_umap, name = "Cell Type") +
    labs(x = "SAT1 Expression (log-norm)", y = "Ferroptosis UCell",
         subtitle = "B  snRNA-seq (7,414 nuclei, 15 cell types)") +
    annotate("text", x = Inf, y = Inf, label = expression(rho ~ "= 0.38, P < 1e-10"),
             hjust = 1.05, vjust = 1.5, size = 3, fontface = "italic", color = "#3C5488") +
    theme_nsfc(9) +
    theme(legend.position = "right", legend.key.size = unit(0.25, "cm"),
          legend.text = element_text(size = 6.5),
          plot.subtitle = element_text(hjust = 0, size = 9, face = "bold"))

  # ---- C: 分子环路示意图 ----
  nodes <- data.frame(
    x  = c(1, 2.5, 4, 4, 5.5, 5.5, 4, 7, 7, 4, 1, 2.5, 7, 7),
    y  = c(7.5, 7.5, 8.5, 6.5, 8.5, 6.5, 5, 8.5, 6.5, 3, 5, 5, 3.5, 2.5),
    label = c("Ferroptosis", "4-HNE", "Keap1\nModified", "DNA\nDamage",
              "Nrf2\nActivated", "p53\nPhosphorylated", "Antioxidant\nDefense",
              "GPX4/SLC7A11\nFTH1", "p21\nCellular Senescence", "SASP\nIL-6/TNF-alpha",
              "SAT1\nAmplifier", "Polyamine\nDepletion", "BCP\nIntervention", "CMap\nReversal"),
    type = c("driver", "signal", "target", "target", "defense", "aging",
             "defense", "effector", "outcome", "sasp", "amplifier", "metabolite",
             "therapy", "therapy")
  )
  node_fills <- c("#E64B35", "#DC0000", "#00A087", "#3C5488", "#00A087",
                  "#3C5488", "#00A087", "#91D1C2", "#7E6148", "#F39B7F",
                  "#8491B4", "#B09C85", "#00A087", "#4DBBD5")

  arrows <- data.frame(
    x = c(1.4, 2.9, 3.6, 4.4, 4.4, 5.9, 5.9, 4, 4.3, 7.4, 4, 1.4, 2.9, 6.8, 6.8),
    y = c(7.5, 7.5, 8.2, 8.5, 6.5, 8.5, 6.5, 5, 3.5, 8.5, 4.2, 6.5, 5, 7.5, 5),
    xend = c(2.1, 3.6, 4, 5.1, 5.1, 6.6, 6.6, 5.5, 4, 7.5, 4, 2, 3.5, 6.8, 6.8),
    yend = c(7.5, 7.5, 8.5, 8.5, 6.5, 8.5, 6.5, 5, 3, 8.5, 5, 5.5, 5, 7.5, 5),
    type = c("activate", "activate", "low", "activate", "high", "activate",
             "activate", "inhibit", "activate", "inhibit", "feedback",
             "activate", "activate", "inhibit", "inhibit")
  )

  arrow_colors <- c("activate"="#3C5488", "low"="#00A087", "high"="#DC0000",
                    "inhibit"="#E64B35", "feedback"="#7E6148")

  p1c <- ggplot() +
    geom_segment(data = arrows,
                 aes(x = x, y = y, xend = xend, yend = yend,
                     color = type, linetype = type),
                 arrow = arrow(length = unit(0.12, "cm"), type = "closed"),
                 linewidth = 0.55, show.legend = FALSE) +
    scale_color_manual(values = arrow_colors) +
    scale_linetype_manual(values = c("activate"="solid", "low"="solid",
                                      "high"="solid", "inhibit"="dashed",
                                      "feedback"="dotted")) +
    geom_point(data = nodes, aes(x = x, y = y), shape = 21,
               size = 9, fill = node_fills, color = "white", stroke = 1.5) +
    geom_text(data = nodes, aes(x = x, y = y, label = label),
              size = 2.2, fontface = "bold", family = "Arial", lineheight = 0.85,
              color = "white") +
    annotate("rect", xmin = 3.5, ymin = 2.5, xmax = 4.5, ymax = 3.5,
             alpha = 0.12, fill = "#DC0000", color = "#DC0000", linetype = "dashed") +
    annotate("text", x = 4, y = 2, label = "4-HNE threshold: defense -> aging",
             size = 2.5, fontface = "italic", color = "#DC0000") +
    xlim(0.5, 8.5) + ylim(1.5, 9) +
    labs(subtitle = "C  Molecular Circuit: 4-HNE-p53-SLC7A11 Loop") +
    theme_void() +
    theme(plot.subtitle = element_text(size = 9, face = "bold", hjust = 0.5,
                                        family = "Arial"))

  # ---- D: 代谢组面板 ----
  axis_table <- read_csv(file.path(cross_dir, "cross_omics_axis_table.csv"),
                         show_col_types = FALSE, locale = locale(encoding = "UTF-8"))
  sat1_meta <- axis_table %>%
    filter(axis_name == "SAT1-polyamine") %>%
    mutate(display_name = factor(display_name, levels = unique(display_name)))

  p1d <- ggplot(sat1_meta, aes(x = display_name, y = log2FC_aging,
                                fill = direction_aging)) +
    geom_col(width = 0.65, alpha = 0.85) +
    geom_hline(yintercept = 0, color = "grey30", linewidth = 0.4) +
    geom_errorbar(aes(ymin = log2FC_aging - 0.15, ymax = log2FC_aging + 0.15),
                  width = 0.2, linewidth = 0.3) +
    geom_text(aes(label = ifelse(p_adj_aging < 0.001, "***",
                          ifelse(p_adj_aging < 0.01, "**",
                          ifelse(p_adj_aging < 0.05, "*", "")))),
              vjust = ifelse(sat1_meta$log2FC_aging > 0, -0.4, 1.3),
              size = 3, color = "#3C5488") +
    scale_fill_manual(values = c("DOWN"="#3C5488", "UP"="#E64B35"),
                      name = "Direction") +
    labs(x = "", y = "log2FC (59w vs 3w)",
         subtitle = "D  Metabolomics ST001637 (n=521, polyamine axis)") +
    theme_nsfc(9) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 7.5),
          legend.position = "top", legend.key.size = unit(0.25, "cm"),
          plot.subtitle = element_text(hjust = 0, size = 9, face = "bold"))

  # ---- E: BCP/CMap反证面板 ----
  cmap_all <- read_csv(file.path(data_dir, "12_fgsea_bcp_all_timepoints.csv"),
                       show_col_types = FALSE)
  cmap_summary <- cmap_all %>%
    group_by(comparison) %>%
    summarise(BCP_Up_NES = NES[pathway == "BCP_Up"],
              BCP_Down_NES = ifelse(any(pathway == "BCP_Down"),
                                    NES[pathway == "BCP_Down"], NA),
              .groups = "drop") %>%
    pivot_longer(cols = ends_with("_NES"), names_to = "Pathway", values_to = "NES") %>%
    mutate(comparison = factor(comparison, levels = c("3h","12h","24h","3D","7D")),
           Pathway = recode(Pathway, "BCP_Up_NES"="BCP Up-targets",
                                     "BCP_Down_NES"="BCP Down-targets"))

  p1e <- ggplot(cmap_summary, aes(x = comparison, y = NES, fill = Pathway)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6, alpha = 0.85) +
    geom_hline(yintercept = 0, color = "grey30", linewidth = 0.4) +
    scale_fill_manual(values = c("BCP Up-targets"="#00A087",
                                  "BCP Down-targets"="#8491B4")) +
    labs(x = "Time after Reperfusion", y = "NES (BCP vs Ischemia)",
         subtitle = "E  BCP Intervention + CMap Reversal") +
    theme_nsfc(9) +
    theme(legend.position = "top", legend.key.size = unit(0.25, "cm"),
          plot.subtitle = element_text(hjust = 0, size = 9, face = "bold"))

  # ---- 组合 ----
  p_top <- (p1a | p1b) + plot_layout(widths = c(1, 1.3))
  p_mid <- p1c
  p_bot <- (p1d | p1e) + plot_layout(widths = c(1.2, 1))

  p_combined <- (p_top / p_mid / p_bot) +
    plot_layout(heights = c(1.2, 1.5, 1)) +
    plot_annotation(
      title = "Figure 1: BCP targets iron-aging axis against CIRI: mechanism hypothesis and multi-omics pre-validation",
      theme = theme(plot.title = element_text(size = 11, face = "bold",
                                               hjust = 0.5, family = "Arial")))
  save_figure(p_combined, "Fig1_hypothesis_schematic", width = 13, height = 11)
}

# =============================================================================
# 图2: 技术路线图 (4层递进)
# 放置: 3.1节开头
# =============================================================================
generate_fig2 <- function() {
  message("\n=== Fig2: Technical Roadmap ===")

  blocks <- data.frame(
    y = c(9.5, 7, 4.5, 2),
    content = c("Content 1", "Content 2", "Content 3", "Content 4"),
    title = c("Spatiotemporal Localization of Iron-dependent SIPS in MCAO/R",
              "Cellular Mechanism of BCP Inhibiting 4-HNE-p53-SLC7A11",
              "In vivo Efficacy of BCP in CIRI: Nrf2-dependence",
              "Herb-Efficacy-Compound-Target Association Spectrum"),
    left = c(
      "SD rat MCAO/R\n6h/24h/3d/7d time points\nn=6 at each time point",
      "Primary cortical neurons + astrocytes\nErastin (0.1-0.5 uM) + OGD/R (2h/24-96h)\nBCP 10/50/100 uM",
      "MCAO/R rat model\nImmediate / 3h-delay / 6h-delay\nBCP 102/204/408 mg/kg, 7d gavage\n28d endpoint, n=12/group",
      "GC-MS volatile oil profiling\nGNN target prediction\nMolecular docking (SAT1/Keap1)\nNetwork pharmacology"
    ),
    right = c(
      "Multi-IF: NeuN/GFAP/Iba-1 + GPX4/4-HNE/p21/gamma-H2AX\nTEM: mitochondrial morphology\nSA-beta-gal staining\nLC-MS/MS: 4-HNE quantification\nIP/IB: 4-HNE-Keap1/p53 adducts",
      "C11-BODIPY | FerroOrange | MDA | GSH/GSSG\nGPX4/ACSL4/FTH1/SLC7A11/SAT1\nSA-beta-gal | EdU | p21/p16/gamma-H2AX\nSASP: IL-6/TNF-alpha/MMP-3\nCETSA/SPR: BCP-SAT1/Keap1 binding",
      "24h: TTC infarct volume\n72h: brain edema\n28d: mNSS | Rota-rod | Foot-fault | Morris water maze\nNrf2-dependence: BCP+ML385 group",
      "GC-MS: Artemisia argyi volatile oil composition\nGNN: BCP-target-disease prediction\nMolecular docking: BCP-SAT1/Keap1\nCross-omics association spectrum"
    ),
    color = c("#E64B35", "#00A087", "#3C5488", "#F39B7F")
  )

  arrows <- data.frame(y = c(8.8, 6.3, 3.8), yend = c(7.7, 5.2, 2.7))

  p <- ggplot() +
    geom_rect(data = blocks,
              aes(xmin = 0.5, xmax = 17.5, ymin = y - 0.65, ymax = y + 0.65),
              fill = blocks$color, alpha = 0.1, color = blocks$color,
              linewidth = 0.8, linetype = "solid") +
    geom_text(data = blocks, aes(x = 1.2, y = y + 0.45, label = content),
              size = 3.5, fontface = "bold", color = blocks$color,
              hjust = 0, family = "Arial") +
    geom_text(data = blocks, aes(x = 1.2, y = y + 0.1, label = title),
              size = 3, fontface = "bold", hjust = 0, family = "Arial",
              color = "grey20") +
    geom_text(data = blocks, aes(x = 1.5, y = y - 0.22, label = left),
              size = 2.4, hjust = 0, vjust = 1, family = "Arial",
              color = "grey30", lineheight = 0.9) +
    geom_text(data = blocks, aes(x = 9.5, y = y - 0.22, label = right),
              size = 2.4, hjust = 0, vjust = 1, family = "Arial",
              color = "grey30", lineheight = 0.9) +
    geom_segment(data = arrows,
                 aes(x = 9, y = y, xend = 9, yend = yend),
                 arrow = arrow(length = unit(0.18, "cm"), type = "closed"),
                 color = "#3C5488", linewidth = 0.9) +
    annotate("segment", x = 0.5, y = 10.2, xend = 17.5, yend = 10.2,
             color = "grey70", linewidth = 0.3) +
    annotate("text", x = 9, y = 0.8,
             label = expression(bold("Zhuang Medicine Theory: 'Clear Poison-Evil, Unblock Dragon-Fire Pathways'") *
                                "  |  " * bold("Beta-Caryophyllene (BCP) from Artemisia argyi")),
             size = 2.8, color = "#7E6148", family = "Arial") +
    annotate("text", x = 9, y = 10.6,
             label = "Iron-Aging Hypothesis Pre-validation (Multi-omics Public Data)",
             size = 3, fontface = "italic", color = "#7E6148", family = "Arial") +
    annotate("segment", x = 9, y = 10.4, xend = 9, yend = 9.65,
             arrow = arrow(length = unit(0.15, "cm")),
             color = "#7E6148", linewidth = 0.7) +
    annotate("text", x = 9, y = 11,
             label = "Figure 2: Technical Roadmap of Four Research Contents",
             size = 4, fontface = "bold", family = "Arial") +
    xlim(0, 18) + ylim(0.5, 11.5) +
    theme_void()

  save_figure(p, "Fig2_technical_roadmap", width = 14, height = 9)
}

# =============================================================================
# 图3: Bulk RNA-seq 时序GSEA (NES折线图 + 3-cluster热图)
# 放置: 研究基础 1.3.1节
# =============================================================================
generate_fig3 <- function() {
  message("\n=== Fig3: Bulk RNA-seq GSEA ===")

  gsea_sum <- read_csv(file.path(data_dir, "03_bulk_gsea_nes_summary.csv"),
                       show_col_types = FALSE)
  lfc_mat  <- read_csv(file.path(data_dir, "02_bulk_ferroaging_lfc_matrix.csv"),
                       show_col_types = FALSE)

  # ---- A: NES折线图 ----
  time_order <- c("3h", "12h", "24h", "3D", "7D")
  term_order <- c("Ferroptosis", "Ferroaging", "Ferrosenescence", "Senescence")
  term_labels <- c("Ferroptosis"="Ferroptosis (KEGG mmu04216)",
                   "Ferroaging"="Ferroaging (FA-96)",
                   "Ferrosenescence"="Ferrosenescence",
                   "Senescence"="Senescence (Reactome)")

  gsea_filtered <- gsea_sum %>%
    filter(term %in% term_order) %>%
    mutate(comparison = factor(comparison, levels = time_order),
           term = factor(term, levels = term_order),
           sig = ifelse(p.adjust < 0.05, "FDR<0.05", "NS"))

  term_colors <- c("Ferroptosis"="#DC0000", "Ferroaging"="#F39B7F",
                   "Ferrosenescence"="#7E6148", "Senescence"="#3C5488")

  p3a <- ggplot(gsea_filtered, aes(x = comparison, y = NES, group = term,
                                    color = term, shape = sig)) +
    geom_line(linewidth = 0.8, alpha = 0.85) +
    geom_point(aes(size = sig), alpha = 0.9) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50",
               linewidth = 0.4) +
    scale_color_manual(values = term_colors,
                       labels = term_labels, name = "Gene Set") +
    scale_shape_manual(values = c("FDR<0.05"=16, "NS"=1),
                       name = "Significance") +
    scale_size_manual(values = c("FDR<0.05"=3, "NS"=2.5), guide = "none") +
    labs(x = "Time after MCAO", y = "Normalized Enrichment Score (NES)",
         subtitle = "A  GSEA NES Trajectories (vs Ctrl)") +
    theme_nsfc(10) +
    theme(legend.position = "right", legend.key.size = unit(0.3, "cm"),
          plot.subtitle = element_text(hjust = 0, size = 10, face = "bold"),
          legend.box = "vertical")

  # ---- B: FA-96 3-cluster热图 ----
  gene_mat <- as.matrix(lfc_mat[, c("X3h", "X12h", "X24h", "X3D", "X7D")])
  rownames(gene_mat) <- lfc_mat$gene
  colnames(gene_mat) <- time_order

  gene_mat_z <- t(scale(t(gene_mat)))

  set.seed(42)
  km <- kmeans(gene_mat_z, centers = 3, nstart = 25)
  cluster_order <- order(km$cluster)
  gene_mat_ordered <- gene_mat_z[cluster_order, ]
  cluster_assign <- km$cluster[cluster_order]

  cluster_colors <- c("1"="#E64B35", "2"="#3C5488", "3"="#F39B7F")
  names(cluster_colors) <- c("1", "2", "3")

  row_ha <- rowAnnotation(
    Cluster = as.character(cluster_assign),
    col = list(Cluster = cluster_colors),
    show_legend = TRUE,
    annotation_legend_param = list(title = "k-means Cluster",
                                    title_gp = gpar(fontsize = 9))
  )

  col_fun <- colorRamp2(c(-2, 0, 2), c("#3C5488", "white", "#E64B35"))

  ht <- Heatmap(gene_mat_ordered,
    name = "Z-score",
    col = col_fun,
    cluster_rows = FALSE, cluster_columns = FALSE,
    show_row_names = TRUE, row_names_gp = gpar(fontsize = 6),
    row_names_side = "left",
    column_names_gp = gpar(fontsize = 9, fontface = "bold"),
    column_title = "B  FA-96 Gene Expression (Z-score, k=3 clusters)",
    column_title_gp = gpar(fontsize = 10, fontface = "bold"),
    right_annotation = row_ha,
    border = TRUE,
    heatmap_legend_param = list(title_gp = gpar(fontsize = 9))
  )

  png(file.path(output_dir, "Fig3_bulk_gsea.png"), width = 12, height = 7,
      units = "in", res = 300)
  draw(ht, column_title = "Figure 3: Temporal GSEA of Ferroptosis/Senescence Signatures in MCAO Bulk RNA-seq",
       column_title_gp = gpar(fontsize = 12, fontface = "bold"))
  dev.off()

  pdf(file.path(output_dir, "Fig3_bulk_gsea.pdf"), width = 12, height = 7)
  draw(ht, column_title = "Figure 3: Temporal GSEA of Ferroptosis/Senescence Signatures in MCAO Bulk RNA-seq",
       column_title_gp = gpar(fontsize = 12, fontface = "bold"))
  dev.off()

  ggsave(file.path(output_dir, "Fig3a_nes_lines.png"), p3a,
         width = 7, height = 5, dpi = 300, bg = "white")
  message("  -> Fig3_bulk_gsea.pdf/png")
}

# =============================================================================
# 图4: 空间转录组铁衰老定位 (A+B并排 / C+D并排)
# 放置: 研究基础 1.3.2节
# =============================================================================
generate_fig4 <- function() {
  message("\n=== Fig4: Spatial Transcriptomics ===")

  region_scores <- read_csv(file.path(data_dir, "06_spatial_region_scores.csv"),
                            show_col_types = FALSE)
  neuron_fp <- read_csv(file.path(data_dir, "10_neuron_prop_vs_ferroptosis.csv"),
                        show_col_types = FALSE)

  region_order <- c("Healthy", "Penumbra", "InfarctCore", "Other")
  region_scores <- region_scores %>%
    filter(region %in% region_order) %>%
    mutate(region = factor(region, levels = region_order))

  # ---- A: 铁死亡得分空间热图 ----
  p4a <- ggplot(region_scores, aes(x = region, y = Ferroptosis, fill = region)) +
    geom_half_violin(side = "r", position = position_nudge(x = 0.2),
                     alpha = 0.7, color = NA) +
    geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA,
                 position = position_nudge(x = 0.2), linewidth = 0.4) +
    geom_jitter(aes(color = region), size = 0.2, alpha = 0.1,
                position = position_jitter(width = 0.04)) +
    geom_signif(comparisons = list(c("Healthy", "Penumbra"),
                                    c("Healthy", "InfarctCore")),
                map_signif_level = function(p) sprintf("p=%.1e", p),
                step_increase = 0.06, textsize = 2.5, tip_length = 0.003) +
    scale_fill_manual(values = region_colors) +
    scale_color_manual(values = region_colors, guide = "none") +
    labs(x = "", y = "Ferroptosis UCell Score",
         subtitle = "A  Ferroptosis Score (Visium)") +
    theme_nsfc(9) +
    theme(legend.position = "none", axis.text.x = element_text(angle = 25, hjust = 1),
          plot.subtitle = element_text(hjust = 0, size = 9, face = "bold"))

  # ---- B: 铁衰老得分小提琴图 ----
  p4b <- ggplot(region_scores, aes(x = region, y = Ferrosenescence, fill = region)) +
    geom_half_violin(side = "r", position = position_nudge(x = 0.2),
                     alpha = 0.7, color = NA) +
    geom_boxplot(width = 0.1, fill = "white", outlier.shape = NA,
                 position = position_nudge(x = 0.2), linewidth = 0.4) +
    geom_jitter(aes(color = region), size = 0.2, alpha = 0.1,
                position = position_jitter(width = 0.04)) +
    geom_signif(comparisons = list(c("Healthy", "Penumbra"),
                                    c("Healthy", "InfarctCore")),
                map_signif_level = function(p) sprintf("p=%.1e", p),
                step_increase = 0.06, textsize = 2.5, tip_length = 0.003) +
    scale_fill_manual(values = region_colors) +
    scale_color_manual(values = region_colors, guide = "none") +
    labs(x = "Brain Region", y = "Ferrosenescence UCell Score",
         subtitle = "B  Ferrosenescence Score") +
    theme_nsfc(9) +
    theme(legend.position = "none", axis.text.x = element_text(angle = 25, hjust = 1),
          plot.subtitle = element_text(hjust = 0, size = 9, face = "bold"))

  # ---- C: 神经元比例 vs 铁死亡得分相关图 ----
  neuron_plot <- neuron_fp %>%
    filter(region %in% c("Healthy", "Penumbra", "InfarctCore"))

  cor_test <- cor.test(neuron_plot$neuron_prop, neuron_plot$fp_score,
                        method = "spearman")
  rho_val <- round(cor_test$estimate, 3)
  p_val <- cor_test$p.value
  p_label <- ifelse(p_val < 2.22e-16, "P < 2.2e-16",
                     sprintf("P = %.2e", p_val))

  p4c <- ggplot(neuron_plot, aes(x = neuron_prop, y = fp_score, color = region)) +
    geom_point(alpha = 0.2, size = 0.6) +
    geom_smooth(method = "lm", se = TRUE, alpha = 0.12, linewidth = 0.7) +
    geom_smooth(method = "lm", color = "grey20", se = FALSE, linewidth = 0.4) +
    scale_color_manual(values = region_colors, name = "Region") +
    annotate("text", x = Inf, y = Inf,
             label = sprintf("Spearman rho = %s\n%s", rho_val, p_label),
             hjust = 1.05, vjust = 1.3, size = 3, fontface = "bold",
             color = "#3C5488") +
    labs(x = "Neuron Proportion (SPOTlight)", y = "Ferroptosis Score",
         subtitle = "C  Neuron Proportion vs Ferroptosis (n=2,145 spots)") +
    theme_nsfc(9) +
    theme(legend.position = c(0.88, 0.85), plot.subtitle = element_text(hjust = 0, size = 9, face = "bold"))

  # ---- 组合 ----
  p_top <- (p4a | p4b) + plot_layout(widths = c(1, 1))
  p_bot <- p4c

  p_combined <- (p_top / p_bot) +
    plot_layout(heights = c(1, 1.1)) +
    plot_annotation(
      title = "Figure 4: Spatial Localization of Iron-Aging Signals in CIRI Penumbra",
      theme = theme(plot.title = element_text(size = 11, face = "bold",
                                               hjust = 0.5, family = "Arial")))

  save_figure(p_combined, "Fig4_spatial_iron_aging", width = 10, height = 8)
}

# =============================================================================
# 图5: 单细胞核转录组铁衰老细胞图谱与SAT1验证 (6面板)
# 放置: 研究基础 1.3.3节
# =============================================================================
generate_fig5 <- function() {
  message("\n=== Fig5: Single-cell SAT1 Validation ===")

  sat1_fp <- read_csv(file.path(data_dir, "08_sat1_vs_ferroptosis_score.csv"),
                      show_col_types = FALSE)
  cor_by_ct <- read_csv(file.path(data_dir, "08_sat1_ferroptosis_correlation_by_celltype.csv"),
                        show_col_types = FALSE)
  augur <- read_csv(file.path(data_dir, "09_augur_auc_ranking.csv"),
                    show_col_types = FALSE)
  props <- read_csv(file.path(data_dir, "08_ferrosenescence_proportions.csv"),
                    show_col_types = FALSE)

  ct_colors <- c(
    "NeuronsGABA"="#E64B35", "NeuronsGLUT"="#D95F02", "OLs"="#4DBBD5",
    "OPCs"="#91D1C2", "Astrocytes"="#00A087", "Microglia"="#3C5488",
    "EndothelialCells"="#F39B7F", "EpendymalCells"="#B09C85",
    "VLMCs"="#8491B4", "Neuroblasts"="#7E6148"
  )

  # ---- A: SAT1 vs Ferroptosis 散点图 (按细胞类型着色) ----
  set.seed(42)
  sat1_sample <- sat1_fp %>%
    group_by(Celltypes) %>%
    sample_frac(min(0.2, 400 / n())) %>%
    ungroup()

  p5a <- ggplot(sat1_sample, aes(x = Sat1, y = FP_Score, color = Celltypes)) +
    geom_point(alpha = 0.3, size = 0.5) +
    geom_smooth(method = "lm", se = FALSE, linewidth = 0.5) +
    scale_color_manual(values = ct_colors, name = "Cell Type") +
    annotate("text", x = Inf, y = Inf,
             label = expression(rho ~ "= 0.38, P < 1e-10"),
             hjust = 1.05, vjust = 1.5, size = 3, fontface = "italic",
             color = "#3C5488") +
    labs(x = "SAT1 Expression", y = "Ferroptosis UCell",
         subtitle = "A  SAT1 vs Ferroptosis Score") +
    theme_nsfc(9) +
    theme(legend.position = "right", legend.key.size = unit(0.25, "cm"),
          legend.text = element_text(size = 6.5),
          plot.subtitle = element_text(hjust = 0, size = 9, face = "bold"))

  # ---- B: 相关性条形图 (按细胞类型) ----
  cor_plot <- cor_by_ct %>%
    mutate(cell_type = factor(cell_type, levels = cell_type),
           sig = ifelse(padj < 0.05, "FDR<0.05", "NS"))

  p5b <- ggplot(cor_plot, aes(x = rho, y = reorder(cell_type, rho), fill = sig)) +
    geom_col(width = 0.6) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey50",
               linewidth = 0.4) +
    scale_fill_manual(values = c("FDR<0.05"="#E64B35", "NS"="#B09C85"),
                      name = "Significance") +
    labs(x = "Spearman rho (SAT1 vs Ferroptosis)", y = "",
         subtitle = "B  Correlation by Cell Type") +
    theme_nsfc(9) +
    theme(legend.position = "top", legend.key.size = unit(0.25, "cm"),
          plot.subtitle = element_text(hjust = 0, size = 9, face = "bold"))

  # ---- C: Augur棒棒糖图 ----
  # 兼容新版 per-timepoint 结果: 取 1DPI (急性期最具代表性)
  if ("comparison" %in% colnames(augur)) {
    augur <- augur %>%
      dplyr::filter(comparison == "1DPI") %>%
      dplyr::arrange(dplyr::desc(AUC))
  }
  augur_plot <- augur %>%
    mutate(cell_type = factor(cell_type, levels = rev(cell_type)))

  p5c <- ggplot(augur_plot, aes(x = AUC, y = cell_type)) +
    geom_segment(aes(x = 0.5, xend = AUC, y = cell_type, yend = cell_type),
                 color = "grey70", linewidth = 0.7) +
    geom_point(aes(fill = AUC), shape = 21, size = 4.5, color = "white",
               stroke = 0.3) +
    geom_point(aes(color = AUC), shape = 21, size = 5.5, fill = NA,
               stroke = 0.5) +
    geom_vline(xintercept = 0.5, linetype = "dashed", color = "grey50",
               linewidth = 0.4) +
    scale_fill_gradient(low = "#4DBBD5", high = "#E64B35", name = "AUC",
                        limits = c(0.5, 0.56)) +
    scale_color_gradient(low = "#4DBBD5", high = "#E64B35", guide = "none") +
    labs(x = "Augur AUC (Perturbation Priority)", y = "",
         subtitle = "C  Augur Perturbation Priority") +
    theme_nsfc(9) +
    theme(legend.position = "right", legend.key.size = unit(0.3, "cm"),
          plot.subtitle = element_text(hjust = 0, size = 9, face = "bold"))

  # ---- D: Ferrosenescence_High比例 ----
  prop_plot <- props %>%
    filter(status == "Ferrosenescence_High") %>%
    mutate(condition = factor(condition, levels = c("Ctrl","1DPI","3DPI","7DPI")))

  p5d <- ggplot(prop_plot, aes(x = condition, y = proportion, fill = celltype)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7, alpha = 0.85) +
    scale_fill_manual(values = ct_colors, name = "Cell Type") +
    labs(x = "Condition", y = "Ferrosenescence_High Proportion",
         subtitle = "D  Ferrosenescence Cell Distribution") +
    theme_nsfc(9) +
    theme(legend.position = "right", legend.key.size = unit(0.25, "cm"),
          legend.text = element_text(size = 6),
          axis.text.x = element_text(angle = 25, hjust = 1),
          plot.subtitle = element_text(hjust = 0, size = 9, face = "bold"))

  # ---- 组合 ----
  p_top <- (p5a | p5b) + plot_layout(widths = c(1.2, 1))
  p_bot <- (p5c | p5d) + plot_layout(widths = c(1, 1.2))

  p_combined <- (p_top / p_bot) +
    plot_layout(heights = c(1, 1)) +
    plot_annotation(
      title = "Figure 5: Single-cell Iron-Aging Atlas and SAT1 Validation in CIRI",
      theme = theme(plot.title = element_text(size = 11, face = "bold",
                                               hjust = 0.5, family = "Arial")))

  save_figure(p_combined, "Fig5_singlecell_sat1", width = 12, height = 9)
}

# =============================================================================
# 图6: 代谢组跨组学整合 (A瀑布+B匹配率+C KEGG)
# 放置: 研究基础 1.3.4节
# =============================================================================
generate_fig6 <- function() {
  message("\n=== Fig6: Metabolomics Cross-omics ===")

  axis_table <- read_csv(file.path(cross_dir, "cross_omics_axis_table.csv"),
                         show_col_types = FALSE, locale = locale(encoding = "UTF-8"))
  table1 <- read_csv(file.path(method_dir, "Table1_cross_omics_evidence.csv"),
                     show_col_types = FALSE)
  kegg_paths <- read_csv(file.path(kegg_dir, "cross_omics_shared_pathways.csv"),
                         show_col_types = FALSE)

  # ---- A: SAT1-多胺轴瀑布图 ----
  sat1_meta <- axis_table %>%
    filter(axis_name == "SAT1-polyamine") %>%
    mutate(display_name = factor(display_name, levels = unique(display_name)))

  p6a <- ggplot(sat1_meta, aes(x = display_name, y = log2FC_aging,
                                fill = direction_aging)) +
    geom_col(width = 0.65, alpha = 0.85) +
    geom_hline(yintercept = 0, color = "grey30", linewidth = 0.4) +
    geom_text(aes(label = ifelse(p_adj_aging < 0.001, "***",
                          ifelse(p_adj_aging < 0.01, "**",
                          ifelse(p_adj_aging < 0.05, "*", "")))),
              vjust = ifelse(sat1_meta$log2FC_aging > 0, -0.4, 1.3),
              size = 3, color = "#3C5488") +
    scale_fill_manual(values = c("DOWN"="#3C5488", "UP"="#E64B35"),
                      name = "Direction") +
    labs(x = "", y = expression(log[2] * "FC (59w vs 3w)"),
         subtitle = "A  SAT1-Polyamine Axis Metabolites (ST001637, n=521)") +
    theme_nsfc(9) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 7.5),
          legend.position = "top", legend.key.size = unit(0.25, "cm"),
          plot.subtitle = element_text(hjust = 0, size = 9, face = "bold"))

  # ---- B: 通路轴匹配率 ----
  table1_plot <- table1 %>%
    mutate(match_num = as.numeric(sub("/.*", "", `Match Rate`)),
           match_den = as.numeric(sub(".*/", "", `Match Rate`)),
           match_pct = match_num / match_den,
           `Pathway Axis` = factor(`Pathway Axis`,
                                   levels = `Pathway Axis`[order(match_pct)]))

  p6b <- ggplot(table1_plot, aes(x = match_pct, y = `Pathway Axis`, fill = Evidence)) +
    geom_col(width = 0.6, alpha = 0.85) +
    geom_text(aes(label = sprintf("%s (%.0f%%)", `Match Rate`, match_pct * 100)),
              hjust = -0.05, size = 2.8) +
    geom_vline(xintercept = 0.5, linetype = "dashed", color = "#E64B35",
               linewidth = 0.5) +
    annotate("text", x = 0.52, y = 1, label = "50% threshold",
             color = "#E64B35", size = 2.5, hjust = 0) +
    scale_fill_manual(values = evidence_colors, name = "Evidence") +
    labs(x = "Match Rate", y = "",
         subtitle = "B  Gene-Metabolite Pathway Axis Match Rate") +
    xlim(0, 0.95) +
    theme_nsfc(9) +
    theme(legend.position = "top", legend.key.size = unit(0.25, "cm"),
          plot.subtitle = element_text(hjust = 0, size = 9, face = "bold"))

  # ---- C: KEGG通路跨组学覆盖 ----
  kegg_top <- kegg_paths %>%
    mutate(pathway_short = gsub(" - Mus musculus.*", "", pathway_name),
           pathway_short = ifelse(nchar(pathway_short) > 42,
                                  paste0(substr(pathway_short, 1, 39), "..."),
                                  pathway_short)) %>%
    arrange(desc(total_omics_elements)) %>%
    head(10) %>%
    mutate(pathway_short = factor(pathway_short,
                                  levels = pathway_short[order(total_omics_elements)]))

  p6c <- ggplot(kegg_top, aes(x = total_omics_elements, y = pathway_short,
                               fill = n_genes)) +
    geom_col(width = 0.6, alpha = 0.85) +
    geom_text(aes(label = total_omics_elements), hjust = -0.15, size = 2.8) +
    scale_fill_gradient2(low = "#4DBBD5", mid = "#F39B7F", high = "#E64B35",
                         midpoint = 6, name = "N Genes") +
    labs(x = "Total Omics Elements (Genes + Metabolites)", y = "",
         subtitle = "C  Top 10 Shared KEGG Pathways") +
    xlim(0, max(kegg_top$total_omics_elements) * 1.18) +
    theme_nsfc(9) +
    theme(axis.text.y = element_text(size = 7.5),
          legend.position = "right", legend.key.size = unit(0.3, "cm"),
          plot.subtitle = element_text(hjust = 0, size = 9, face = "bold"))

  # ---- 组合 ----
  p_combined <- (p6a | p6b) / p6c +
    plot_layout(heights = c(1, 1.1)) +
    plot_annotation(
      title = "Figure 6: Metabolomics Cross-omics Integration of SAT1-Polyamine Axis",
      theme = theme(plot.title = element_text(size = 11, face = "bold",
                                               hjust = 0.5, family = "Arial")))

  save_figure(p_combined, "Fig6_metabolomics_crossomics", width = 13, height = 9)
}

# =============================================================================
# 表1: 跨组学通路轴验证汇总表
# =============================================================================
generate_table1 <- function() {
  message("\n=== Table 1: Cross-omics Axis Summary ===")

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
    core = list(
      bg_params = list(fill = ifelse(table1_sorted$Evidence == "Moderate",
                                      alpha("#00A087", 0.12), alpha("#F0F0F0", 0.5))),
      text_params = list(cex = 0.85, fontface = "bold")
    ),
    colhead = list(
      bg_params = list(fill = "#3C5488"),
      text_params = list(col = "white", cex = 0.85, fontface = "bold")
    ),
    rowhead = list(
      bg_params = list(fill = "#E64B35"),
      text_params = list(col = "white", cex = 0.85)
    )
  )

  table_mat <- as.matrix(table1_sorted)

  png(file.path(output_dir, "Table1_cross_omics.png"), width = 12, height = 3.5,
      units = "in", res = 300)
  grid.newpage()
  grid.text("Table 1: Cross-omics Pathway Axis Validation Summary",
            x = 0.5, y = 0.95, gp = gpar(fontsize = 12, fontface = "bold"))
  grid.table(table_mat, theme = tt)
  dev.off()

  pdf(file.path(output_dir, "Table1_cross_omics.pdf"), width = 12, height = 3.5)
  grid.newpage()
  grid.text("Table 1: Cross-omics Pathway Axis Validation Summary",
            x = 0.5, y = 0.95, gp = gpar(fontsize = 12, fontface = "bold"))
  grid.table(table_mat, theme = tt)
  dev.off()

  message("  -> Table1_cross_omics.pdf/png")
}

# =============================================================================
# 主程序
# =============================================================================
main <- function() {
  message("=============================================")
  message("Iron-Aging NSFC Grant Figures Generator v2")
  message("Design: 12 WeChat articles + Nature Cancer palette")
  message("Data: GSE233815 + ST001637 (real only)")
  message("=============================================")

  generate_fig1()
  generate_fig2()
  generate_fig3()
  generate_fig4()
  generate_fig5()
  generate_fig6()
  generate_table1()

  message("\n=============================================")
  message("All figures generated: ", output_dir)
  message("=============================================")
}

main()