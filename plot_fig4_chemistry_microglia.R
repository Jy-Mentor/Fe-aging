##############################################################################
# Fig4: 化合物化学性质 + 微胶铁衰老亚群 (参照气泡热图/棒棒糖风格)
# - (A) 化合物池化学性质分布 (MW, LogP, QED, TPSA)
# - (B) 微胶亚群铁衰老气泡图
# - (C) 石竹烯-CIRI 靶标重叠韦恩图数据可视化
##############################################################################

library(ggplot2)
library(ggrepel)
library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(ggsci)
library(viridis)
library(patchwork)

OUTDIR <- "d:/铁衰老 绝不重蹈覆辙/figures"
OUTDIR_PDF <- file.path(OUTDIR, "pdf")

theme_pub <- theme_bw(base_size = 10) +
  theme(
    panel.grid.major = element_line(color = "grey90", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 0.8),
    axis.title = element_text(face = "bold", size = 11),
    axis.text = element_text(size = 9, color = "black"),
    plot.title = element_text(face = "bold", size = 12, hjust = 0),
    plot.tag = element_text(face = "bold", size = 16)
  )

# ============================================================================
# (A) 化合物化学性质分布
# ============================================================================
cat("[Fig4-A] Compound property distributions...\n")

tox_path <- "d:/铁衰老 绝不重蹈覆辙/L3/results/tcm_compound_pool_tox_filtered.csv"
stopifnot(file.exists(tox_path))

tox <- read_csv(tox_path, show_col_types = FALSE)
cat(sprintf("  Compounds: %d\n", nrow(tox)))

# Select key properties
prop_cols <- c("MW", "LogP", "TPSA", "QED")
available <- intersect(prop_cols, colnames(tox))
if (length(available) == 0) {
  # Try alternate names
  alt_names <- list(
    MW = c("MW", "mw", "MolecularWeight"),
    LogP = c("LogP", "alogp", "MolLogP"),
    TPSA = c("TPSA", "tpsa"),
    QED = c("QED", "qed")
  )
  for (nm in names(alt_names)) {
    found <- intersect(alt_names[[nm]], colnames(tox))
    if (length(found) > 0) {
      colnames(tox)[colnames(tox) == found[1]] <- nm
      available <- c(available, nm)
    }
  }
}

cat(sprintf("  Available properties: %s\n", paste(available, collapse=", ")))

if (length(available) >= 2) {
  plots <- list()
  colors <- c("MW" = "#3A9AB2", "LogP" = "#E19433", "TPSA" = "#6FB2C1", "QED" = "#CC5650")

  for (prop in available) {
    p_dat <- tox %>% filter(!is.na(.data[[prop]]), is.finite(.data[[prop]]))
    plots[[prop]] <- ggplot(p_dat, aes(x = .data[[prop]])) +
      geom_histogram(aes(fill = after_stat(x)), bins = 40, alpha = 0.8, color = "white", linewidth = 0.2) +
      scale_fill_viridis_c(option = "C") +
      labs(x = prop, y = "Count") +
      theme_pub +
      theme(legend.position = "none", plot.tag = NULL)
  }

  p_props <- wrap_plots(plots, ncol = 2) +
    plot_annotation(title = "TCM Compound Pool: Physicochemical Properties",
                    tag_levels = list(c("A1", "A2", "A3", "A4")))

  ggsave(file.path(OUTDIR, "Fig4A_compound_properties.png"), p_props, width = 10, height = 8, dpi = 300)
  ggsave(file.path(OUTDIR_PDF, "Fig4A_compound_properties.pdf"), p_props, width = 10, height = 8)
  cat("  -> Fig4A saved\n")
} else {
  cat("  WARNING: No property columns found, skipping\n")
}

# ============================================================================
# (B) 微胶亚群铁衰老气泡图 (参照 Nature 气泡热图)
# ============================================================================
cat("[Fig4-B] Microglia subcluster bubble...\n")

mg_path <- "d:/铁衰老 绝不重蹈覆辙/L2/results/GSE233815_sn/microglia_subcluster/microglia_cluster_ferroaging_summary.csv"
if (!file.exists(mg_path)) {
  mg_path <- "d:/铁衰老 绝不重蹈覆辙/L2/results/GSE233815_sn/microglia_subcluster/microglia_subcluster_summary.csv"
}
stopifnot(file.exists(mg_path))

mg <- read_csv(mg_path, show_col_types = FALSE)
cat(sprintf("  Microglia data: %d rows, cols: %s\n", nrow(mg), paste(colnames(mg), collapse=", ")))

# Filter to single score method: prefer FA96 over FA95
if ("score_method" %in% colnames(mg)) {
  methods_avail <- unique(mg$score_method)
  cat(sprintf("  Available score methods: %s\n", paste(methods_avail, collapse=", ")))
  if ("AddModuleScore_FA96" %in% methods_avail) {
    mg <- mg %>% filter(score_method == "AddModuleScore_FA96")
  } else if ("AddModuleScore_FA95" %in% methods_avail) {
    mg <- mg %>% filter(score_method == "AddModuleScore_FA95")
  } else {
    mg <- mg %>% filter(score_method == methods_avail[1])
  }
  cat(sprintf("  Filtered to %d rows using method: %s\n", nrow(mg), unique(mg$score_method)))
}

# Check what we have
if (all(c("seurat_clusters", "Condition", "mean_score") %in% colnames(mg))) {
  actual_conditions <- unique(mg$Condition)
  cat(sprintf("  Conditions found: %s\n", paste(actual_conditions, collapse=", ")))
  
  mg_plot <- mg %>%
    mutate(
      seurat_clusters = factor(seurat_clusters),
      Condition = factor(Condition, levels = actual_conditions)
    )

  p_mg <- ggplot(mg_plot, aes(x = Condition, y = seurat_clusters)) +
    geom_point(aes(size = n_cells, fill = mean_score), shape = 21, stroke = 0.3) +
    scale_fill_gradient2(low = "#3A9AB2", mid = "white", high = "#E07524", midpoint = 0,
                         name = "Mean FA\nScore") +
    scale_size_continuous(range = c(2, 12), name = "N Cells") +
    labs(x = NULL, y = "Microglia Subcluster", tag = "B",
         title = "Microglia Subcluster Ferroaging Activity") +
    theme_pub

  ggsave(file.path(OUTDIR, "Fig4B_microglia_bubble.png"), p_mg, width = 7, height = 5, dpi = 300)
  ggsave(file.path(OUTDIR_PDF, "Fig4B_microglia_bubble.pdf"), p_mg, width = 7, height = 5)
  cat("  -> Fig4B saved\n")
} else {
  cat("  WARNING: Expected columns not found in microglia data\n")
}

# ============================================================================
# (C) 石竹烯-CIRI 靶标重叠
# ============================================================================
cat("[Fig4-C] BCP-CIRI overlap...\n")

bcp_path <- "d:/铁衰老 绝不重蹈覆辙/L2/results/caryophyllene_ciri_overlap_official_string.csv"
stopifnot(file.exists(bcp_path))

bcp <- read_csv(bcp_path, show_col_types = FALSE)
cat(sprintf("  BCP overlap: %d rows\n", nrow(bcp)))

bcp_plot <- bcp %>%
  filter(!is.na(Count), !is.na(P_value)) %>%
  mutate(
    Item = factor(Item, levels = rev(Item)),
    neg_log10_p = -log10(P_value)
  )

p_bcp <- ggplot(bcp_plot, aes(x = Count, y = Item)) +
  geom_col(aes(fill = neg_log10_p), alpha = 0.85, width = 0.6) +
  geom_text(aes(label = Count), hjust = -0.3, size = 3.5, fontface = "bold") +
  scale_fill_viridis_c(option = "B", name = "-log10(P)") +
  scale_x_continuous(limits = c(0, max(bcp_plot$Count) * 1.3)) +
  labs(
    x = "Gene Count", y = NULL,
    tag = "C", title = "\u03b2-Caryophyllene Target Overlap with CIRI-Ferroaging"
  ) +
  theme_pub

ggsave(file.path(OUTDIR, "Fig4C_BCP_overlap.png"), p_bcp, width = 8, height = 5, dpi = 300)
ggsave(file.path(OUTDIR_PDF, "Fig4C_BCP_overlap.pdf"), p_bcp, width = 8, height = 5)
cat("  -> Fig4C saved\n")

cat("[Fig4] All done!\n")
