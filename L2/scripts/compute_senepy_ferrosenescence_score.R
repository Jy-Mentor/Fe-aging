# Compute SenePy senescence scores + ferroaging scores on GSE233815 snRNA-seq data.
# Integrates SenePy (Sanborn 2025, Nature Communications) universal senescence signatures
# with existing ferroaging gene set to create a "ferrosenescence" dual-axis scoring system.
#
# SenePy reference: Sanborn MA et al., Nature Communications (2025), DOI: 10.1038/s41467-025-57047-7

suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(patchwork))

set.seed(42)

# ---- Paths ----
rds_path <- "data/external/GSE233815/mendeley/Seurat_sn_MCAO_Zucha2023_QCFiltered.Rds"
gene_csv <- "L1/results/ferroaging_genes_96.csv"
senepy_csv <- "L2/results/senepy/senepy_mouse_universal_genes.csv"
out_dir <- "L2/results/GSE233815_sn"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---- Load data ----
message("Loading Seurat object...")
seu <- readRDS(rds_path)
message("Loaded: ", ncol(seu), " cells, ", nrow(seu), " features")

# ---- Load ferroaging gene set ----
message("\nLoading ferroaging gene set...")
fa_genes_df <- read.csv(gene_csv, stringsAsFactors = FALSE)
sym_col <- intersect(colnames(fa_genes_df), c("gene_symbol", "Gene", "gene", "Symbol", "symbol"))[1]
human_fa_genes <- unique(trimws(fa_genes_df[[sym_col]]))
human_fa_genes <- human_fa_genes[human_fa_genes != "" & !is.na(human_fa_genes)]
message("Human ferroaging genes: ", length(human_fa_genes))

ortho_df <- read.csv(file.path(out_dir, "human_to_mouse_orthologs.csv"), stringsAsFactors = FALSE)
human_to_mouse <- setNames(ortho_df$mouse_symbol, ortho_df$human_symbol)

fa_genes_96 <- unique(human_to_mouse[human_fa_genes])
fa_genes_96 <- fa_genes_96[fa_genes_96 != "" & !is.na(fa_genes_96)]
fa_genes_95 <- setdiff(fa_genes_96, human_to_mouse["ACSL4"])
message("Mouse orthologs FA-96: ", length(fa_genes_96))
message("Mouse orthologs FA-95 (excluding ACSL4): ", length(fa_genes_95))

# ---- Load SenePy universal senescence signatures ----
message("\nLoading SenePy universal senescence signatures...")
senepy_df <- read.csv(senepy_csv, stringsAsFactors = FALSE)
senepy_genes <- unique(trimws(senepy_df$gene_symbol))
senepy_genes <- senepy_genes[senepy_genes != "" & !is.na(senepy_genes)]
message("SenePy universal mouse genes: ", length(senepy_genes))

# ---- Map cell types ----
message("\nMapping fine-grained cell types to coarse classes...")
coarse_map <- list(
  Neuron = c("Glutamatergic 1", "Glutamatergic 2", "Glutamatergic 3", "Glutamatergic 4",
             "Glutamatergic 5", "Glutamatergic 6", "Glutamatergic 7", "Glutamatergic 8",
             "Glutamatergic 9", "Glutamatergic 10", "GABAergic 1", "GABAergic 2",
             "GABAergic 3", "GABAergic 4", "GABAergic 5", "GABAergic 6", "Neuro 1", "NB"),
  Astrocyte = c("ASTRO", "ASTRO/OLIGO", "EPEN"),
  Microglia = c("MG"),
  Oligodendrocyte = c("OLIGO", "OPC"),
  Endothelial_Pericyte = c("PER/Endo", "VLMC")
)

seu$cell_class <- NA_character_
for (cls in names(coarse_map)) {
  idx <- seu$cell_type_1 %in% coarse_map[[cls]]
  seu$cell_class[idx] <- cls
}
message("Coarse cell class table:")
print(table(seu$cell_class, useNA = "ifany"))

# ---- Check gene overlap ----
message("\nChecking gene overlap with Seurat object...")
overlap_fa95 <- intersect(fa_genes_95, rownames(seu))
overlap_fa96 <- intersect(fa_genes_96, rownames(seu))
overlap_senepy <- intersect(senepy_genes, rownames(seu))
message("FA-95 genes present: ", length(overlap_fa95), " / ", length(fa_genes_95))
message("FA-96 genes present: ", length(overlap_fa96), " / ", length(fa_genes_96))
message("SenePy genes present: ", length(overlap_senepy), " / ", length(senepy_genes))

# Compute gene set overlap between ferroaging and SenePy
overlap_fa_senepy <- intersect(fa_genes_96, senepy_genes)
message("Genes shared between ferroaging and SenePy: ", length(overlap_fa_senepy))
if (length(overlap_fa_senepy) > 0) {
  message("  Shared: ", paste(overlap_fa_senepy, collapse = ", "))
}

# ---- Compute SenePy scores ----
message("\nComputing SenePy senescence scores...")
seu <- AddModuleScore(seu, features = list(SenePy = senepy_genes), name = "SenePy_",
                      assay = "RNA", search = TRUE)
seu$SenePy_Score <- seu$SenePy_1
seu$SenePy_1 <- NULL

# ---- Compute Ferroaging scores ----
message("Computing Ferroaging scores...")
seu <- AddModuleScore(seu, features = list(FA_95 = fa_genes_95), name = "AddModuleScore_FA95_",
                      assay = "RNA", search = TRUE)
seu <- AddModuleScore(seu, features = list(FA_96 = fa_genes_96), name = "AddModuleScore_FA96_",
                      assay = "RNA", search = TRUE)
seu$AddModuleScore_FA95 <- seu$AddModuleScore_FA95_1
seu$AddModuleScore_FA96 <- seu$AddModuleScore_FA96_1
seu$AddModuleScore_FA95_1 <- NULL
seu$AddModuleScore_FA96_1 <- NULL

# ---- Compute Ferrosenescence combined score ----
message("Computing Ferrosenescence combined score...")
seu$FA96_z <- scale(seu$AddModuleScore_FA96)[,1]
seu$SenePy_z <- scale(seu$SenePy_Score)[,1]
seu$Ferrosenescence <- (seu$FA96_z + seu$SenePy_z) / 2

# ---- Correlation analysis ----
message("\nCorrelation analysis: SenePy vs Ferroaging...")
score_cols <- c("SenePy_Score",
                "AddModuleScore_FA95", "AddModuleScore_FA96",
                "Ferrosenescence")

cor_matrix <- cor(seu@meta.data[, score_cols], method = "spearman", use = "pairwise.complete.obs")
message("Spearman correlation matrix:")
print(round(cor_matrix, 3))

# ---- Aggregate statistics ----
message("\nAggregating scores...")
agg_cells <- seu@meta.data %>%
  filter(!is.na(cell_class)) %>%
  pivot_longer(cols = all_of(score_cols), names_to = "score_method", values_to = "score") %>%
  group_by(Condition, cell_class, score_method) %>%
  summarise(
    n_cells = n(),
    mean_score = mean(score, na.rm = TRUE),
    median_score = median(score, na.rm = TRUE),
    sd_score = sd(score, na.rm = TRUE),
    se_score = sd_score / sqrt(n_cells),
    .groups = "drop"
  )
write.csv(agg_cells, file.path(out_dir, "ferrosenescence_score_by_condition_cellclass_method.csv"),
          row.names = FALSE)

agg_condition <- seu@meta.data %>%
  pivot_longer(cols = all_of(score_cols), names_to = "score_method", values_to = "score") %>%
  group_by(Condition, score_method) %>%
  summarise(
    n_cells = n(),
    mean_score = mean(score, na.rm = TRUE),
    median_score = median(score, na.rm = TRUE),
    sd_score = sd(score, na.rm = TRUE),
    .groups = "drop"
  )
write.csv(agg_condition, file.path(out_dir, "ferrosenescence_score_by_condition_method.csv"),
          row.names = FALSE)

# ---- Wilcoxon tests ----
message("\nPerforming Wilcoxon tests per cell class (vs Ctrl)...")
meta <- seu@meta.data %>% filter(!is.na(cell_class))
meta$Condition <- factor(meta$Condition, levels = c("Ctrl", "1DPI", "3DPI", "7DPI"))

test_results <- data.frame()
for (method in score_cols) {
  for (cls in unique(meta$cell_class)) {
    sub <- meta %>% filter(cell_class == cls)
    ctrl_vals <- sub %>% filter(Condition == "Ctrl") %>% pull(!!sym(method))
    for (cond in c("1DPI", "3DPI", "7DPI")) {
      cond_vals <- sub %>% filter(Condition == cond) %>% pull(!!sym(method))
      if (length(ctrl_vals) < 3 || length(cond_vals) < 3) next
      test <- wilcox.test(cond_vals, ctrl_vals, alternative = "two.sided")
      test_results <- rbind(test_results, data.frame(
        score_method = method,
        cell_class = cls,
        condition = cond,
        n_ctrl = length(ctrl_vals),
        n_cond = length(cond_vals),
        median_ctrl = median(ctrl_vals, na.rm = TRUE),
        median_cond = median(cond_vals, na.rm = TRUE),
        mean_ctrl = mean(ctrl_vals, na.rm = TRUE),
        mean_cond = mean(cond_vals, na.rm = TRUE),
        statistic = test$statistic,
        p_value = test$p.value,
        stringsAsFactors = FALSE
      ))
    }
  }
}

test_results <- test_results %>%
  group_by(score_method, cell_class) %>%
  mutate(p_adj = p.adjust(p_value, method = "BH")) %>%
  ungroup() %>%
  arrange(score_method, cell_class, condition)
write.csv(test_results, file.path(out_dir, "ferrosenescence_wilcoxon_vs_ctrl.csv"),
          row.names = FALSE)

# ---- Save outputs ----
message("\nSaving outputs...")
write.csv(seu@meta.data, file.path(out_dir, "cell_metadata_with_ferrosenescence_score.csv"),
          row.names = FALSE)
saveRDS(seu, file.path(out_dir, "Seurat_sn_MCAO_with_ferrosenescence_score.rds"))

# ---- Visualizations ----
message("Generating visualizations...")
seu$Condition <- factor(seu$Condition, levels = c("Ctrl", "1DPI", "3DPI", "7DPI"))

# Nature-style publication theme
theme_publication <- function(base_size = 8) {
  theme_classic(base_size = base_size) %+replace%
    theme(
      panel.background = element_rect(fill = NA, color = "black", linewidth = 0.5),
      panel.grid.major = element_line(color = "grey92", linewidth = 0.2),
      panel.grid.minor = element_blank(),
      axis.line = element_blank(),
      axis.ticks = element_line(color = "black", linewidth = 0.3),
      axis.ticks.length = unit(2, "pt"),
      axis.text = element_text(color = "black", size = rel(0.9)),
      axis.title = element_text(color = "black", size = rel(1.0)),
      plot.title = element_text(size = rel(1.1), face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = rel(0.9), hjust = 0.5, color = "grey40"),
      legend.position = "bottom",
      legend.justification = "left",
      legend.box.spacing = unit(0, "pt"),
      legend.key.size = unit(3, "mm"),
      strip.text = element_text(size = rel(0.9), face = "bold"),
      strip.background = element_rect(fill = "grey95", color = "black", linewidth = 0.3),
      plot.margin = unit(c(5, 10, 5, 5), "pt")
    )
}

# CB-friendly palette for conditions
cond_colors <- c("Ctrl" = "#4E79A7", "1DPI" = "#F28E2B",
                 "3DPI" = "#E15759", "7DPI" = "#76B7B2")

# Panel labels
panel_labels <- c("A", "B", "C", "D", "E", "F", "G")

# 1. SenePy vs Ferroaging correlation scatter (Figure 7A)
p1 <- ggplot(seu@meta.data, aes(x = AddModuleScore_FA96, y = SenePy_Score)) +
  geom_hex(bins = 80) +
  scale_fill_viridis_c(option = "D", name = "Count") +
  geom_smooth(method = "lm", color = "#E15759", se = TRUE, alpha = 0.15,
              linewidth = 0.8) +
  annotate("text", x = -Inf, y = -Inf, hjust = -0.2, vjust = -1.5,
           label = paste0("rho = ", round(cor_matrix["AddModuleScore_FA96", "SenePy_Score"], 3)),
           size = 3.5, fontface = "italic") +
  labs(x = expression("Ferroaging AddModuleScore (FA-96)"),
       y = "SenePy Senescence Score",
       tag = panel_labels[1]) +
  theme_publication(8)

# 2. SenePy score by condition and cell class (Figure 7B)
p2 <- ggplot(seu@meta.data %>% filter(!is.na(cell_class)),
             aes(x = Condition, y = SenePy_Score, fill = Condition)) +
  geom_violin(scale = "width", trim = TRUE, alpha = 0.7, color = NA) +
  geom_boxplot(width = 0.12, outlier.shape = NA, alpha = 0.9,
               color = "grey30", linewidth = 0.25) +
  scale_fill_manual(values = cond_colors, guide = "none") +
  facet_wrap(~ cell_class, scales = "free_y", nrow = 2) +
  labs(y = "SenePy Senescence Score", x = NULL, tag = panel_labels[2]) +
  theme_publication(8) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# 3. Ferrosenescence combined score (Figure 7C)
p3 <- ggplot(seu@meta.data %>% filter(!is.na(cell_class)),
             aes(x = Condition, y = Ferrosenescence, fill = Condition)) +
  geom_violin(scale = "width", trim = TRUE, alpha = 0.7, color = NA) +
  geom_boxplot(width = 0.12, outlier.shape = NA, alpha = 0.9,
               color = "grey30", linewidth = 0.25) +
  scale_fill_manual(values = cond_colors, guide = "none") +
  facet_wrap(~ cell_class, scales = "free_y", nrow = 2) +
  labs(y = "Ferrosenescence Score", x = NULL, tag = panel_labels[3],
       subtitle = expression(italic(F) == (italic(z)[FA-96] + italic(z)[SenePy]) / 2)) +
  theme_publication(8) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# 4. Multi-method comparison bar plot (Figure 7F)
agg_plot <- agg_cells %>%
  filter(score_method %in% c("SenePy_Score", "AddModuleScore_FA96", "Ferrosenescence")) %>%
  mutate(score_method = recode(score_method,
    SenePy_Score = "SenePy",
    AddModuleScore_FA96 = "FA-96",
    Ferrosenescence = "Ferrosenescence"))
p4 <- ggplot(agg_plot, aes(x = cell_class, y = mean_score, fill = Condition)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8),
           width = 0.7, color = "black", linewidth = 0.2, alpha = 0.85) +
  geom_errorbar(aes(ymin = mean_score - se_score, ymax = mean_score + se_score),
                width = 0.2, position = position_dodge(width = 0.8),
                linewidth = 0.4) +
  scale_fill_manual(values = cond_colors, name = "Condition") +
  facet_wrap(~ score_method, scales = "free_y", ncol = 1) +
  labs(y = "Mean Score", x = NULL, tag = panel_labels[6]) +
  theme_publication(8) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 5. Significance dot plot for SenePy (Figure 7D)
sig_senepy <- test_results %>%
  filter(score_method == "SenePy_Score") %>%
  mutate(signif = case_when(
    p_adj < 0.001 ~ "***",
    p_adj < 0.01  ~ "**",
    p_adj < 0.05  ~ "*",
    TRUE          ~ "ns"
  ))
p5 <- ggplot(sig_senepy, aes(x = condition, y = cell_class)) +
  geom_point(aes(size = -log10(p_value + 1e-300), fill = mean_cond - mean_ctrl),
             shape = 21, color = "grey40", stroke = 0.3) +
  geom_text(aes(label = signif), vjust = 0.5, hjust = 0.5, size = 3.5,
            color = "grey20") +
  scale_fill_gradient2(low = "#377EB8", mid = "white", high = "#E41A1C",
                       midpoint = 0, name = expression(Delta * "Mean")) +
  scale_size_continuous(range = c(2, 8), name = expression(-log[10](italic(p)))) +
  labs(x = NULL, y = NULL, tag = panel_labels[4],
       subtitle = "SenePy Score vs Ctrl") +
  theme_publication(8)

# 6. Ferrosenescence significance dot plot (Figure 7E)
sig_fs <- test_results %>%
  filter(score_method == "Ferrosenescence") %>%
  mutate(signif = case_when(
    p_adj < 0.001 ~ "***",
    p_adj < 0.01  ~ "**",
    p_adj < 0.05  ~ "*",
    TRUE          ~ "ns"
  ))
p6 <- ggplot(sig_fs, aes(x = condition, y = cell_class)) +
  geom_point(aes(size = -log10(p_value + 1e-300), fill = mean_cond - mean_ctrl),
             shape = 21, color = "grey40", stroke = 0.3) +
  geom_text(aes(label = signif), vjust = 0.5, hjust = 0.5, size = 3.5,
            color = "grey20") +
  scale_fill_gradient2(low = "#377EB8", mid = "white", high = "#E41A1C",
                       midpoint = 0, name = expression(Delta * "Mean")) +
  scale_size_continuous(range = c(2, 8), name = expression(-log[10](italic(p)))) +
  labs(x = NULL, y = NULL, tag = panel_labels[5],
       subtitle = "Ferrosenescence Score vs Ctrl") +
  theme_publication(8)

# 7. Correlation heatmap (Figure 7G)
cor_df <- as.data.frame(as.table(cor_matrix))
colnames(cor_df) <- c("Method1", "Method2", "SpearmanR")
cor_df$Method1 <- recode(cor_df$Method1,
  SenePy_Score = "SenePy", AddModuleScore_FA95 = "FA-95",
  AddModuleScore_FA96 = "FA-96", Ferrosenescence = "Ferrosen.")
cor_df$Method2 <- recode(cor_df$Method2,
  SenePy_Score = "SenePy", AddModuleScore_FA95 = "FA-95",
  AddModuleScore_FA96 = "FA-96", Ferrosenescence = "Ferrosen.")
p7 <- ggplot(cor_df, aes(x = Method1, y = Method2, fill = SpearmanR)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.2f", SpearmanR)), size = 3.5, color = "grey20") +
  scale_fill_gradient2(low = "#377EB8", mid = "white", high = "#E41A1C",
                       midpoint = 0, limits = c(-1, 1), name = "Spearman rho") +
  labs(x = NULL, y = NULL, tag = panel_labels[7]) +
  theme_publication(8) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid.major = element_blank())

# ---- Save figures ----
message("Saving figures...")
out_width_short <- 5.2
out_width_long  <- 9.0
out_height_short <- 4.0
out_height_mid   <- 5.0
out_height_long  <- 8.0

ggsave(file.path(out_dir, "senepy_vs_ferroaging_scatter.pdf"), p1,
       width = out_width_short, height = out_width_short)
ggsave(file.path(out_dir, "senepy_vs_ferroaging_scatter.png"), p1,
       width = out_width_short, height = out_width_short, dpi = 300)

ggsave(file.path(out_dir, "senepy_score_by_condition_cellclass_violin.pdf"), p2,
       width = out_width_long, height = out_height_mid)
ggsave(file.path(out_dir, "senepy_score_by_condition_cellclass_violin.png"), p2,
       width = out_width_long, height = out_height_mid, dpi = 300)

ggsave(file.path(out_dir, "ferrosenescence_by_condition_cellclass_violin.pdf"), p3,
       width = out_width_long, height = out_height_mid)
ggsave(file.path(out_dir, "ferrosenescence_by_condition_cellclass_violin.png"), p3,
       width = out_width_long, height = out_height_mid, dpi = 300)

ggsave(file.path(out_dir, "multi_method_comparison_barplot.pdf"), p4,
       width = 7.5, height = out_height_long)
ggsave(file.path(out_dir, "multi_method_comparison_barplot.png"), p4,
       width = 7.5, height = out_height_long, dpi = 300)

ggsave(file.path(out_dir, "senepy_significance_dotplot.pdf"), p5,
       width = out_width_short, height = out_height_short)
ggsave(file.path(out_dir, "senepy_significance_dotplot.png"), p5,
       width = out_width_short, height = out_height_short, dpi = 300)

ggsave(file.path(out_dir, "ferrosenescence_significance_dotplot.pdf"), p6,
       width = out_width_short, height = out_height_short)
ggsave(file.path(out_dir, "ferrosenescence_significance_dotplot.png"), p6,
       width = out_width_short, height = out_height_short, dpi = 300)

ggsave(file.path(out_dir, "method_correlation_heatmap.pdf"), p7,
       width = out_width_short, height = out_width_short)
ggsave(file.path(out_dir, "method_correlation_heatmap.png"), p7,
       width = out_width_short, height = out_width_short, dpi = 300)

# ---- Save integration summary ----
integration_summary <- data.frame(
  item = c(
    "Dataset", "Total cells", "Conditions", "Cell classes",
    "SenePy universal genes (mouse)", "SenePy genes in object",
    "Ferroaging genes FA-96", "Ferroaging genes in object",
    "Genes shared (SenePy ∩ Ferroaging)",
    "SenePy citation",
    "SenePy vs FA-96 Spearman rho",
    "Integration date"
  ),
  value = c(
    "GSE233815 snRNA-seq (Zucha 2023)",
    ncol(seu),
    paste(levels(seu$Condition), collapse = ", "),
    paste(unique(meta$cell_class), collapse = ", "),
    length(senepy_genes),
    length(overlap_senepy),
    length(fa_genes_96),
    length(overlap_fa96),
    length(overlap_fa_senepy),
    "Sanborn MA et al., Nature Communications (2025), DOI: 10.1038/s41467-025-57047-7",
    round(cor_matrix["AddModuleScore_FA96", "SenePy_Score"], 4),
    as.character(Sys.Date())
  )
)
write.csv(integration_summary, file.path(out_dir, "senepy_integration_summary.csv"),
          row.names = FALSE)

message("\n=== SenePy Integration Complete ===")
message("Outputs in: ", normalizePath(out_dir))
message("Key finding: SenePy vs FA-96 Spearman rho = ",
        round(cor_matrix["AddModuleScore_FA96", "SenePy_Score"], 4))