# Cross-check microglia high ferro-aging markers with ferro-aging gene set,
# human orthologs, known microglial states, and L4 CPI prediction results.

suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(babelgene))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(stringr))

set.seed(42)

base_dir <- "L2/results/GSE233815_sn"
micro_dir <- file.path(base_dir, "microglia_subcluster")
l4_pred_path <- "L4/results_v10_minibatch/tcm_predictions_full_v41.csv"

# ---- Load data ----
deg <- read.csv(file.path(micro_dir, "microglia_high_fa_vs_low_fa_deg.csv"), stringsAsFactors = FALSE)
fa_genes <- read.csv("L1/results/ferroaging_genes_96.csv", stringsAsFactors = FALSE)
ortho <- read.csv(file.path(base_dir, "human_to_mouse_orthologs.csv"), stringsAsFactors = FALSE)

# Identify human ferro-aging genes
sym_col <- intersect(colnames(fa_genes), c("gene_symbol", "Gene", "gene", "Symbol", "symbol"))[1]
human_fa <- unique(trimws(fa_genes[[sym_col]]))
human_fa <- human_fa[human_fa != "" & !is.na(human_fa)]

# Create mouse -> human mapping from ferro-aging orthologs
mouse_to_human_fa <- setNames(ortho$human_symbol, ortho$mouse_symbol)

# ---- Get top High_FA markers ----
high_fa <- deg %>%
  filter(cluster == "High_FA", p_val_adj < 0.1) %>%
  arrange(p_val_adj, desc(avg_log2FC))

if (nrow(high_fa) == 0) {
  stop("No High_FA markers found with p_val_adj < 0.1. Check differential expression results.")
}

# ---- Map all High_FA markers to human orthologs via babelgene ----
message("Mapping High_FA markers to human orthologs via babelgene...")
all_mouse_markers <- unique(high_fa$gene)
map_res <- orthologs(genes = all_mouse_markers, species = "mouse", human = FALSE)
# babelgene returns columns: symbol (mouse), entrez, ensembl, taxon_id, human_symbol, human_entrez, human_ensembl, ...
mouse_to_human_all <- setNames(map_res$human_symbol, map_res$symbol)

# ---- Annotate each marker ----
high_fa_annotated <- high_fa %>%
  mutate(
    human_ortholog = coalesce(mouse_to_human_fa[gene], mouse_to_human_all[gene]),
    in_ferroaging_96 = human_ortholog %in% human_fa,
    mouse_gene = gene
  ) %>%
  select(mouse_gene, human_ortholog, avg_log2FC, p_val_adj, pct.1, pct.2, in_ferroaging_96)

write.csv(high_fa_annotated, file.path(micro_dir, "microglia_high_fa_markers_annotated.csv"), row.names = FALSE)

message("High_FA markers annotated.")
message("Total High_FA markers with p_adj < 0.1: ", nrow(high_fa_annotated))
message("Markers with human ortholog: ", sum(!is.na(high_fa_annotated$human_ortholog)))
message("Markers whose human ortholog is in ferro-aging-96: ", sum(high_fa_annotated$in_ferroaging_96, na.rm = TRUE))
message("\nTop annotated markers:")
print(high_fa_annotated)

# ---- Known microglial state marker overlap ----
# References: Keren-Shaul et al. 2017 (DAM), Krasemann et al. 2017 (MGnD), Deczkowska et al. 2018 (IRM)
dam_markers <- c("Apoe", "Trem2", "Axl", "Cst7", "Ctsb", "Ctsd", "Cd9", "Spp1", "Lpl", "Csf1", "Itgax", "Clec7a", "Gpnmb")
mgn_markers <- c("Spp1", "Apoe", "Lgals3", "Clec7a", "Axl", "Csf1", "Ctsb", "Ctsd")
irm_markers <- c("Ifit3", "Ifit2", "Ifit1", "Isg15", "Oas3", "Oas2", "Mx2", "Stat1", "Irf7")
proliferation_markers <- c("Top2a", "Mki67", "Cdk1", "Prc1", "Cenpf")

all_mouse_genes <- high_fa_annotated$mouse_gene

state_overlap <- data.frame(
  state = c("DAM", "MGnD", "IRM", "Proliferation"),
  overlap_genes = c(
    paste(intersect(all_mouse_genes, dam_markers), collapse = ", "),
    paste(intersect(all_mouse_genes, mgn_markers), collapse = ", "),
    paste(intersect(all_mouse_genes, irm_markers), collapse = ", "),
    paste(intersect(all_mouse_genes, proliferation_markers), collapse = ", ")
  )
)
state_overlap$n_overlap <- sapply(strsplit(state_overlap$overlap_genes, ", "), function(x) sum(x != ""))

write.csv(state_overlap, file.path(micro_dir, "microglia_high_fa_state_overlap.csv"), row.names = FALSE)
message("\nOverlap with known microglial states:")
print(state_overlap)

# ---- Genes for L4 CPI cross-check ----
# Human orthologs of High_FA markers that are NOT already in ferro-aging-96
candidates_for_cpi <- high_fa_annotated %>%
  filter(!is.na(human_ortholog), !in_ferroaging_96) %>%
  pull(human_ortholog) %>%
  unique() %>%
  sort()

write.table(candidates_for_cpi, file.path(micro_dir, "microglia_high_fa_cpi_candidates_human.txt"),
            row.names = FALSE, col.names = FALSE, quote = FALSE)
message("\nHuman orthologs for L4 CPI cross-check (not in ferroaging-96): ", length(candidates_for_cpi))
if (length(candidates_for_cpi) > 0) {
  message(paste(candidates_for_cpi, collapse = ", "))
}

# ---- L4 CPI prediction cross-check ----
if (!file.exists(l4_pred_path)) {
  warning("L4 CPI prediction file not found: ", l4_pred_path)
} else {
  message("\nLoading L4 CPI predictions from: ", l4_pred_path)
  l4_pred <- read.csv(l4_pred_path, stringsAsFactors = FALSE)

  # Identify target protein columns (exclude _uncertainty and metadata columns)
  target_cols <- setdiff(
    colnames(l4_pred)[!grepl("_uncertainty$", colnames(l4_pred))],
    c("MOL_ID", "molecule_name", "SMILES", "mean_uncertainty", "max_uncertainty",
      "ferroptosis_prob", "herb_origins", "n_herbs", "tcm_pool_score", "tcm_pool_tier",
      "is_whitelist", "in_train", "uncertainty_penalty", "zs_avg_score", "zs_max_score",
      "zs_n_hits", "zs_n_targets", "zs_bonus", "ferroptosis_factor", "composite_score",
      "avg_score_all", "max_score_all", "n_hits_all", "n_targets_all", "avg_score_warm",
      "max_score_warm", "n_hits_warm", "n_targets_warm", "weighted_avg", "weighted_max",
      "weighted_hits", "top_targets", "rank")
  )

  message("L4 target proteins: ", length(target_cols))

  # All human orthologs from High_FA markers (including ferro-aging-96 members)
  all_human_orthologs <- high_fa_annotated %>%
    filter(!is.na(human_ortholog)) %>%
    pull(human_ortholog) %>%
    unique() %>%
    sort()

  # Targets that are both High_FA microglia markers and L4 CPI targets
  overlap_targets <- intersect(all_human_orthologs, target_cols)
  message("High_FA microglia markers overlapping with L4 CPI targets: ", length(overlap_targets))
  if (length(overlap_targets) > 0) {
    message(paste(overlap_targets, collapse = ", "))
  }

  # Targets in L4 that are NOT High_FA markers (for context)
  non_overlap_targets <- setdiff(target_cols, all_human_orthologs)

  overlap_summary <- data.frame(
    category = c("High_FA_marker_and_L4_target", "High_FA_marker_not_in_L4", "L4_target_not_High_FA_marker"),
    n = c(length(overlap_targets), length(setdiff(all_human_orthologs, target_cols)), length(non_overlap_targets)),
    genes = c(
      paste(overlap_targets, collapse = ", "),
      paste(setdiff(all_human_orthologs, target_cols), collapse = ", "),
      paste(head(non_overlap_targets, 20), collapse = ", ") # truncate for readability
    )
  )
  write.csv(overlap_summary, file.path(micro_dir, "microglia_high_fa_l4_target_overlap_summary.csv"), row.names = FALSE)

  # For each overlapping target, extract top predicted TCM compounds
  top_n <- 10
  cpi_cross_results <- list()

  for (tgt in overlap_targets) {
    col_name <- tgt
    # Select relevant columns for ranking
    cols_needed <- c("MOL_ID", "molecule_name", "SMILES", col_name, "composite_score", "rank", "herb_origins", "n_herbs", "tcm_pool_tier")
    missing_cols <- setdiff(cols_needed, colnames(l4_pred))
    if (length(missing_cols) > 0) {
      warning("Missing columns for target ", tgt, ": ", paste(missing_cols, collapse = ", "))
      next
    }
    tgt_df <- l4_pred[, cols_needed]
    colnames(tgt_df)[colnames(tgt_df) == col_name] <- "target_score"
    tgt_df$target <- tgt
    tgt_df <- tgt_df %>%
      arrange(desc(target_score)) %>%
      head(top_n) %>%
      mutate(rank_in_target = row_number())
    cpi_cross_results[[tgt]] <- tgt_df
  }

  if (length(cpi_cross_results) > 0) {
    cpi_cross_df <- bind_rows(cpi_cross_results)
    write.csv(cpi_cross_df, file.path(micro_dir, "microglia_high_fa_l4_cpi_top_compounds.csv"), row.names = FALSE)
    message("\nTop L4 CPI compounds for overlapping High_FA microglia targets:")
    print(cpi_cross_df %>% select(target, rank_in_target, MOL_ID, molecule_name, target_score, composite_score, rank, herb_origins))
  } else {
    message("No overlapping targets between High_FA microglia markers and L4 CPI predictions.")
  }
}

# ---- Visualizations ----
message("\nGenerating visualizations...")

# 1. Barplot of High_FA marker log2FC colored by ferro-aging membership
p1_df <- high_fa_annotated %>%
  filter(!is.na(human_ortholog)) %>%
  arrange(desc(avg_log2FC)) %>%
  head(20)
p1_df$mouse_gene <- factor(p1_df$mouse_gene, levels = p1_df$mouse_gene)
p1 <- ggplot(p1_df, aes(x = mouse_gene, y = avg_log2FC, fill = in_ferroaging_96)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = human_ortholog), vjust = -0.5, size = 3) +
  scale_fill_manual(values = c("TRUE" = "#E41A1C", "FALSE" = "#377EB8"),
                    labels = c("TRUE" = "In ferro-aging-96", "FALSE" = "Not in ferro-aging-96")) +
  labs(title = "Top 20 High_FA microglia markers",
       subtitle = "Labeled with human orthologs; red = ferro-aging-96 member",
       x = "Mouse gene", y = "avg_log2FC (High_FA vs Low_FA)", fill = NULL) +
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file.path(micro_dir, "microglia_high_fa_marker_log2fc.pdf"), p1, width = 10, height = 6)
ggsave(file.path(micro_dir, "microglia_high_fa_marker_log2fc.png"), p1, width = 10, height = 6, dpi = 300)

# 2. Barplot of known microglial state overlap
p2 <- ggplot(state_overlap, aes(x = state, y = n_overlap, fill = state)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = n_overlap), vjust = -0.5, size = 4) +
  labs(title = "Overlap with known microglial activation states",
       x = "Microglial state", y = "Number of overlapping markers") +
  theme_bw(base_size = 12) +
  theme(legend.position = "none")
ggsave(file.path(micro_dir, "microglia_high_fa_state_overlap.pdf"), p2, width = 6, height = 4)
ggsave(file.path(micro_dir, "microglia_high_fa_state_overlap.png"), p2, width = 6, height = 4, dpi = 300)

# 3. L4 CPI overlap summary barplot (if available)
if (exists("overlap_summary") && nrow(overlap_summary) > 0) {
  p3_df <- overlap_summary %>% filter(category != "L4_target_not_High_FA_marker")
  p3 <- ggplot(p3_df, aes(x = category, y = n, fill = category)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = n), vjust = -0.5, size = 4) +
    scale_x_discrete(labels = c("High_FA_marker_and_L4_target" = "High_FA marker\n& L4 target",
                                "High_FA_marker_not_in_L4" = "High_FA marker\nnot in L4")) +
    labs(title = "High_FA microglia markers vs L4 CPI targets",
         x = NULL, y = "Number of genes") +
    theme_bw(base_size = 12) +
    theme(legend.position = "none")
  ggsave(file.path(micro_dir, "microglia_high_fa_l4_target_overlap.pdf"), p3, width = 6, height = 4)
  ggsave(file.path(micro_dir, "microglia_high_fa_l4_target_overlap.png"), p3, width = 6, height = 4, dpi = 300)
}

# 4. Top L4 compound scores per overlapping target (if available)
if (exists("cpi_cross_df") && nrow(cpi_cross_df) > 0) {
  # Order molecule names within each target facet by target_score
  cpi_cross_df <- cpi_cross_df %>%
    group_by(target) %>%
    mutate(molecule_name = factor(molecule_name, levels = molecule_name[order(target_score)])) %>%
    ungroup()

  p4 <- ggplot(cpi_cross_df, aes(x = molecule_name, y = target_score, fill = target)) +
    geom_bar(stat = "identity") +
    facet_wrap(~ target, scales = "free_y", ncol = 2) +
    coord_flip() +
    labs(title = "Top 10 TCM compounds per overlapping High_FA microglia target",
         x = "Compound", y = "CPI prediction score") +
    theme_bw(base_size = 10) +
    theme(legend.position = "none")
  n_targets <- length(unique(cpi_cross_df$target))
  plot_height <- max(6, 2 * n_targets)
  ggsave(file.path(micro_dir, "microglia_high_fa_l4_cpi_top_compounds.pdf"), p4, width = 12, height = plot_height)
  ggsave(file.path(micro_dir, "microglia_high_fa_l4_cpi_top_compounds.png"), p4, width = 12, height = plot_height, dpi = 300)
}

message("\n=== Done ===")
message("Outputs in: ", normalizePath(micro_dir))
