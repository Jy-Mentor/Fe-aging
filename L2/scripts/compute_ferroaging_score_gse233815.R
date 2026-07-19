# Compute ferro-aging score on GSE233815 snRNA-seq data.
# Implements both AddModuleScore (Seurat) and UCell for robustness.
# Computes FA-95 (Liu 2026 marker genes) and FA-96 (+ ACSL4 target gene).
# Performs Wilcoxon differential testing per cell class vs Ctrl.

suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(UCell))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(tidyr))

set.seed(42)

# ---- Paths ----
rds_path <- "data/external/GSE233815/mendeley/Seurat_sn_MCAO_Zucha2023_QCFiltered.Rds"
gene_csv <- "L1/results/ferroaging_genes_96.csv"
out_dir <- "L2/results/GSE233815_sn"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---- Load data ----
message("Loading Seurat object...")
seu <- readRDS(rds_path)
message("Loaded: ", ncol(seu), " cells, ", nrow(seu), " features")

message("\nLoading ferro-aging gene set and human-mouse orthologs...")
fa_genes_df <- read.csv(gene_csv, stringsAsFactors = FALSE)
sym_col <- intersect(colnames(fa_genes_df), c("gene_symbol", "Gene", "gene", "Symbol", "symbol"))[1]
human_fa_genes <- unique(trimws(fa_genes_df[[sym_col]]))
human_fa_genes <- human_fa_genes[human_fa_genes != "" & !is.na(human_fa_genes)]
message("Human ferro-aging genes: ", length(human_fa_genes))

ortho_df <- read.csv(file.path(out_dir, "human_to_mouse_orthologs.csv"), stringsAsFactors = FALSE)
human_to_mouse <- setNames(ortho_df$mouse_symbol, ortho_df$human_symbol)

fa_genes_96 <- unique(human_to_mouse[human_fa_genes])
fa_genes_96 <- fa_genes_96[fa_genes_96 != "" & !is.na(fa_genes_96)]
fa_genes_95 <- setdiff(fa_genes_96, human_to_mouse["ACSL4"])
message("Mouse orthologs FA-96: ", length(fa_genes_96))
message("Mouse orthologs FA-95 (excluding ACSL4): ", length(fa_genes_95))

# ---- Map fine-grained cell_type_1 to coarse cell classes ----
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

# ---- Check overlap with object genes ----
message("\nChecking gene overlap with Seurat object...")
overlap_95 <- intersect(fa_genes_95, rownames(seu))
overlap_96 <- intersect(fa_genes_96, rownames(seu))
message("FA-95 genes present: ", length(overlap_95), " / ", length(fa_genes_95))
message("FA-96 genes present: ", length(overlap_96), " / ", length(fa_genes_96))
missing_95 <- setdiff(fa_genes_95, rownames(seu))
missing_96 <- setdiff(fa_genes_96, rownames(seu))
if (length(missing_96) > 0) {
  message("Missing FA-96 genes: ", paste(missing_96, collapse = ", "))
}

# ---- Compute AddModuleScore ----
message("\nComputing AddModuleScore scores...")
seu <- AddModuleScore(seu, features = list(FA_95 = fa_genes_95), name = "AddModuleScore_FA95_",
                      assay = "RNA", search = TRUE)
seu <- AddModuleScore(seu, features = list(FA_96 = fa_genes_96), name = "AddModuleScore_FA96_",
                      assay = "RNA", search = TRUE)
# AddModuleScore appends "1" to name
seu$AddModuleScore_FA95 <- seu$AddModuleScore_FA95_1
seu$AddModuleScore_FA96 <- seu$AddModuleScore_FA96_1
seu$AddModuleScore_FA95_1 <- NULL
seu$AddModuleScore_FA96_1 <- NULL

# ---- Compute UCell scores ----
message("Computing UCell scores...")
seu <- AddModuleScore_UCell(seu, features = list(FA_95 = fa_genes_95, FA_96 = fa_genes_96),
                            assay = "RNA", maxRank = 1500)

# ---- Correlation sensitivity analysis ----
message("\nSensitivity analysis: AddModuleScore vs UCell...")
cor_fa95 <- cor(seu$AddModuleScore_FA95, seu$FA_95_UCell, method = "spearman", use = "complete.obs")
cor_fa96 <- cor(seu$AddModuleScore_FA96, seu$FA_96_UCell, method = "spearman", use = "complete.obs")
message("Spearman correlation FA-95: ", round(cor_fa95, 4))
message("Spearman correlation FA-96: ", round(cor_fa96, 4))

# ---- Aggregate statistics ----
message("\nAggregating ferro-aging scores...")
score_cols <- c("AddModuleScore_FA95", "AddModuleScore_FA96", "FA_95_UCell", "FA_96_UCell")

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
write.csv(agg_cells, file.path(out_dir, "ferroaging_score_by_condition_cellclass_method.csv"), row.names = FALSE)

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
write.csv(agg_condition, file.path(out_dir, "ferroaging_score_by_condition_method.csv"), row.names = FALSE)

# ---- Wilcoxon differential testing per cell class ----
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
write.csv(test_results, file.path(out_dir, "ferroaging_score_wilcoxon_vs_ctrl.csv"), row.names = FALSE)
message("Wilcoxon tests saved.")

# ---- Save outputs ----
message("\nSaving outputs...")
write.csv(seu@meta.data, file.path(out_dir, "cell_metadata_with_ferroaging_score.csv"), row.names = FALSE)
saveRDS(seu, file.path(out_dir, "Seurat_sn_MCAO_with_ferroaging_score.rds"))

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

cond_colors <- c("Ctrl" = "#4E79A7", "1DPI" = "#F28E2B",
                 "3DPI" = "#E15759", "7DPI" = "#76B7B2")

# 1. Correlation scatter: AddModuleScore vs UCell for FA-96
p_corr <- ggplot(seu@meta.data, aes(x = AddModuleScore_FA96, y = FA_96_UCell)) +
  geom_hex(bins = 80) +
  scale_fill_viridis_c(option = "D", name = "Count") +
  geom_smooth(method = "lm", color = "#E15759", se = TRUE, alpha = 0.15,
              linewidth = 0.8) +
  annotate("text", x = Inf, y = -Inf, hjust = 1.1, vjust = -1.5,
           label = paste0("rho = ", round(cor_fa96, 3)),
           size = 3.5, fontface = "italic") +
  labs(x = "AddModuleScore FA-96", y = "UCell FA-96") +
  theme_publication(8)
ggsave(file.path(out_dir, "sensitivity_addmodulescore_vs_ucell_fa96.pdf"), p_corr, width = 5, height = 4.5)
ggsave(file.path(out_dir, "sensitivity_addmodulescore_vs_ucell_fa96.png"), p_corr, width = 5, height = 4.5, dpi = 300)

# 2. Violin by condition and method (FA-96)
plot_df <- seu@meta.data %>%
  select(Condition, AddModuleScore_FA96, FA_96_UCell) %>%
  pivot_longer(cols = -Condition, names_to = "method", values_to = "score") %>%
  mutate(method = recode(method, "AddModuleScore_FA96" = "AddModuleScore", "FA_96_UCell" = "UCell"))

p_method <- ggplot(plot_df, aes(x = Condition, y = score, fill = Condition)) +
  geom_violin(scale = "width", trim = TRUE, alpha = 0.7, color = NA) +
  geom_boxplot(width = 0.12, outlier.shape = NA, alpha = 0.9,
               color = "grey30", linewidth = 0.25) +
  scale_fill_manual(values = cond_colors, guide = "none") +
  facet_wrap(~ method, scales = "free_y") +
  labs(y = "Ferroaging score", x = NULL) +
  theme_publication(8)
ggsave(file.path(out_dir, "ferroaging_fa96_by_condition_method_violin.pdf"), p_method, width = 7, height = 3.5)
ggsave(file.path(out_dir, "ferroaging_fa96_by_condition_method_violin.png"), p_method, width = 7, height = 3.5, dpi = 300)

# 3. Bar plot of mean FA-96 UCell score by cell class and condition
agg_plot <- agg_cells %>% filter(score_method == "FA_96_UCell")
p_bar <- ggplot(agg_plot, aes(x = cell_class, y = mean_score, fill = Condition)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8),
           width = 0.7, color = "black", linewidth = 0.2, alpha = 0.85) +
  geom_errorbar(aes(ymin = mean_score - se_score, ymax = mean_score + se_score),
                width = 0.2, position = position_dodge(width = 0.8),
                linewidth = 0.4) +
  scale_fill_manual(values = cond_colors, name = "Condition") +
  labs(y = "Mean UCell score", x = NULL) +
  theme_publication(8) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file.path(out_dir, "ferroaging_fa96_ucell_mean_barplot.pdf"), p_bar, width = 7.5, height = 5)
ggsave(file.path(out_dir, "ferroaging_fa96_ucell_mean_barplot.png"), p_bar, width = 7.5, height = 5, dpi = 300)

# 4. Heatmap of FA-96 UCell mean scores
agg_wide <- agg_plot %>%
  select(Condition, cell_class, mean_score) %>%
  pivot_wider(names_from = Condition, values_from = mean_score)
write.csv(agg_wide, file.path(out_dir, "ferroaging_fa96_ucell_mean_heatmap.csv"), row.names = FALSE)

# 5. Significance dot plot for FA-96 UCell
sig_df <- test_results %>%
  filter(score_method == "FA_96_UCell") %>%
  mutate(signif = case_when(
    p_adj < 0.001 ~ "***",
    p_adj < 0.01  ~ "**",
    p_adj < 0.05  ~ "*",
    TRUE          ~ "ns"
  ))
p_sig <- ggplot(sig_df, aes(x = condition, y = cell_class)) +
  geom_point(aes(size = -log10(p_value + 1e-300), fill = mean_cond - mean_ctrl),
             shape = 21, color = "grey40", stroke = 0.3) +
  geom_text(aes(label = signif), vjust = 0.5, hjust = 0.5, size = 3.5,
            color = "grey20") +
  scale_fill_gradient2(low = "#377EB8", mid = "white", high = "#E41A1C",
                       midpoint = 0, name = expression(Delta * "Mean")) +
  scale_size_continuous(range = c(2, 8), name = expression(-log[10](italic(p)))) +
  labs(x = NULL, y = NULL,
       subtitle = "FA-96 UCell score vs Ctrl") +
  theme_publication(8)
ggsave(file.path(out_dir, "ferroaging_fa96_ucell_significance_dotplot.pdf"), p_sig, width = 5.5, height = 4)
ggsave(file.path(out_dir, "ferroaging_fa96_ucell_significance_dotplot.png"), p_sig, width = 5.5, height = 4, dpi = 300)

# ---- Save method summary ----
method_summary <- data.frame(
  item = c("Dataset", "Total cells", "Conditions", "Cell classes", "Human input genes",
           "Mouse orthologs FA-96", "Mouse orthologs FA-95", "FA-95 genes in object",
           "FA-96 genes in object", "Missing gene", "AddModuleScore vs UCell FA-95 rho",
           "AddModuleScore vs UCell FA-96 rho"),
  value = c("GSE233815 Seurat_sn_MCAO_Zucha2023_QCFiltered.Rds", ncol(seu),
            paste(levels(seu$Condition), collapse = ", "),
            paste(unique(meta$cell_class), collapse = ", "),
            length(human_fa_genes), length(fa_genes_96), length(fa_genes_95),
            length(overlap_95), length(overlap_96), paste(missing_96, collapse = ", "),
            round(cor_fa95, 4), round(cor_fa96, 4))
)
write.csv(method_summary, file.path(out_dir, "method_summary.csv"), row.names = FALSE)

message("\n=== Done ===")
message("Outputs in: ", normalizePath(out_dir))
