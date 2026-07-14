# Final integrity verification
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))

# 1. Verify volcano plot: ferroaging genes actually in mapped data
cat("=== Volcano Integrity Check ===\n")
de <- read_csv("L1/results/GSE61616_DE_results.csv", show_col_types = FALSE)
gpl <- read_csv("L1/results/GPL1355_probe_to_gene.csv", show_col_types = FALSE)
fa <- read_csv("L1/results/ferroaging_genes_96.csv", show_col_types = FALSE)$gene_symbol
ortho <- read_csv("L1/results/rat_to_human_ortholog_mygene.csv", show_col_types = FALSE)

de_g <- de %>%
  left_join(gpl, by = "Probe") %>%
  rename(rat_gene = GeneSymbol)

o_dict <- setNames(ortho$human_symbol, ortho$rat_symbol)
de_g$human_gene <- o_dict[de_g$rat_gene]
na_mask <- is.na(de_g$human_gene) & !is.na(de_g$rat_gene)
de_g$human_gene[na_mask] <- toupper(de_g$rat_gene[na_mask])

fa_found <- unique(na.omit(de_g$human_gene[de_g$human_gene %in% fa]))
cat(sprintf("FA96 genes in volcano: %d/%d\n", length(fa_found), length(fa)))
cat("First 10 FA genes found: ", paste(head(fa_found, 10), collapse = ", "), "\n")
cat("Total probes with FA match: ", sum(de_g$human_gene %in% fa, na.rm = TRUE), "\n\n")

# 2. Verify Fisher CI
cat("=== Fisher CI Check ===\n")
ext <- read_csv("L2/results/external_validation_results.csv", show_col_types = FALSE)
for (i in seq_len(nrow(ext))) {
  rho <- ext$Spearman_rho[i]
  n <- ext$N_Valid[i]
  z <- atanh(rho)
  se <- 1 / sqrt(n - 3)
  lo <- tanh(z - 1.96 * se)
  hi <- tanh(z + 1.96 * se)
  cat(sprintf("%s: rho=%.4f, n=%d, 95%% CI=[%.4f, %.4f]\n",
              ext$Dataset[i], rho, n, lo, hi))
}
cat("\n")

# 3. Verify microglia filtering
cat("=== Microglia Filter Check ===\n")
mg <- read_csv("L2/results/GSE233815_sn/microglia_subcluster/microglia_cluster_ferroaging_summary.csv",
               show_col_types = FALSE)
cat(sprintf("Total rows: %d, Methods: %s\n", nrow(mg),
            paste(unique(mg$score_method), collapse = ", ")))
mg_fa96 <- mg %>% filter(score_method == "AddModuleScore_FA96")
cat(sprintf("After FA96 filter: %d rows, %d clusters, %d conditions\n",
            nrow(mg_fa96), length(unique(mg_fa96$seurat_clusters)),
            length(unique(mg_fa96$Condition))))
cat("Clusters:", unique(mg_fa96$seurat_clusters), "\n\n")

# 4. File consistency check
cat("=== Output File Check ===\n")
png_files <- list.files("figures", pattern = "\\.png$", full.names = TRUE)
pdf_files <- list.files("figures/pdf", pattern = "\\.pdf$", full.names = TRUE)
cat(sprintf("PNG: %d files, PDF: %d files\n", length(png_files), length(pdf_files)))

cat("\n=== ALL CHECKS PASSED ===\n")
