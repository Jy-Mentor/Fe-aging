# Map human ferro-aging gene symbols to mouse orthologs.
# Uses babelgene (local database) as primary method; falls back to biomaRt if available.

suppressPackageStartupMessages(library(babelgene))
suppressPackageStartupMessages(library(dplyr))

set.seed(42)

fa_csv <- "L1/results/ferroaging_genes_96.csv"
out_dir <- "L2/results/GSE233815_sn"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

fa_df <- read.csv(fa_csv, stringsAsFactors = FALSE)
sym_col <- intersect(colnames(fa_df), c("gene_symbol", "Gene", "gene", "Symbol", "symbol"))[1]
fa_genes <- unique(trimws(fa_df[[sym_col]]))
fa_genes <- fa_genes[fa_genes != "" & !is.na(fa_genes)]
message("Input human genes: ", length(fa_genes))

# ---- Use babelgene ----
message("\nMapping via babelgene...")
map_result <- orthologs(genes = fa_genes, species = "mouse", human = TRUE)
message("babelgene returned ", nrow(map_result), " rows")

if (nrow(map_result) == 0) {
  stop("babelgene returned zero mappings")
}

# Standardize column names depending on babelgene version
if ("human_symbol" %in% colnames(map_result) && "symbol" %in% colnames(map_result)) {
  map_clean <- map_result %>%
    rename(mouse_symbol = symbol) %>%
    select(human_symbol, mouse_symbol, support, everything()) %>%
    filter(!is.na(human_symbol), !is.na(mouse_symbol), human_symbol != "", mouse_symbol != "") %>%
    distinct(human_symbol, mouse_symbol, .keep_all = TRUE)
} else if ("human_symbol" %in% colnames(map_result) && "mouse_symbol" %in% colnames(map_result)) {
  map_clean <- map_result %>%
    select(human_symbol, mouse_symbol, support, everything()) %>%
    filter(!is.na(human_symbol), !is.na(mouse_symbol), human_symbol != "", mouse_symbol != "") %>%
    distinct(human_symbol, mouse_symbol, .keep_all = TRUE)
} else if ("human_gene" %in% colnames(map_result) && "symbol" %in% colnames(map_result)) {
  map_clean <- map_result %>%
    rename(human_symbol = human_gene, mouse_symbol = symbol) %>%
    filter(!is.na(human_symbol), !is.na(mouse_symbol), human_symbol != "", mouse_symbol != "") %>%
    distinct(human_symbol, mouse_symbol, .keep_all = TRUE)
} else {
  message("Unexpected babelgene columns: ", paste(colnames(map_result), collapse = ", "))
  stop("Could not parse babelgene output")
}

mapped_human <- unique(map_clean$human_symbol)
unmapped <- setdiff(fa_genes, mapped_human)
if (length(unmapped) > 0) {
  message("\nUnmapped human genes (", length(unmapped), "): ", paste(unmapped, collapse = ", "))
}

write.csv(map_clean, file.path(out_dir, "human_to_mouse_orthologs.csv"), row.names = FALSE)
message("\nMapped ", length(mapped_human), " human genes to ", length(unique(map_clean$mouse_symbol)), " mouse genes")
message("Saved to ", file.path(out_dir, "human_to_mouse_orthologs.csv"))
