#!/usr/bin/env Rscript
# ============================================================================
# Download human KEGG pathways via massdatabase and export to TSV/GMT
# ============================================================================
# Purpose:
#   1. Download all human KEGG pathways using massdatabase::download_kegg_pathway()
#   2. Parse pathway metadata and gene lists
#   3. Export:
#      - kegg_human_pathways_summary.tsv      (one row per pathway)
#      - kegg_human_pathway_genes.tsv         (long format: pathway-gene pairs)
#      - kegg_human_pathways.gmt              (GSEA-compatible GMT)
# ============================================================================

suppressPackageStartupMessages({
  library(data.table)
})

# ============================================================================
# Paths
# ============================================================================
project_root <- normalizePath(getwd())
out_dir <- file.path(project_root, "L2", "results", "kegg_pathways")
download_dir <- file.path(out_dir, "kegg_human_pathway_raw")

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(download_dir, showWarnings = FALSE, recursive = TRUE)

log_file <- file.path(out_dir, "download_kegg_pathways.log")
log_con <- file(log_file, open = "wt")

write_log <- function(msg) {
  ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  line <- sprintf("[%s] %s", ts, msg)
  cat(line, "\n")
  cat(line, "\n", file = log_con)
}

# ============================================================================
# Install / load massdatabase
# ============================================================================
write_log("Checking massdatabase package...")
if (!requireNamespace("massdatabase", quietly = TRUE)) {
  write_log("massdatabase not installed. Installing from tidymass/massdatabase...")
  if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes", repos = "https://cloud.r-project.org/")
  }
  remotes::install_github("tidymass/massdatabase", dependencies = TRUE, upgrade = "never")
}
suppressPackageStartupMessages(library(massdatabase))
write_log(sprintf("massdatabase version: %s", packageVersion("massdatabase")))

# ============================================================================
# Download human KEGG pathways
# ============================================================================
kegg_db_file <- file.path(download_dir, "kegg_pathway_database")

data_already_exists <- file.exists(kegg_db_file)
if (data_already_exists) {
  write_log(sprintf("Found existing KEGG database: %s", kegg_db_file))
  write_log("Skipping download. Remove this file to force re-download.")
} else {
  write_log("Downloading human KEGG pathways (organism = hsa)...")
  write_log(sprintf("Download directory: %s", download_dir))
  
  download_kegg_pathway(path = download_dir, sleep = 1, organism = "hsa")
  write_log("Download completed.")
}

# ============================================================================
# Read downloaded pathways
# ============================================================================
write_log("Reading downloaded pathway data...")
pw_list <- read_kegg_pathway(path = download_dir)
n_pw <- length(pw_list)
write_log(sprintf("Loaded %d pathways", n_pw))

if (n_pw == 0) {
  stop("No pathways loaded. Check download directory and network connectivity.")
}

# Inspect structure of the first pathway for logging
first <- pw_list[[1]]
write_log(sprintf(
  "First pathway example: %s | fields: %s",
  first$pathway_id,
  paste(names(first), collapse = "; ")
))

# Some entries may be overview/global maps or lack a proper pathway_id;
# report both total and unique-id counts for transparency.
unique_ids <- unique(sapply(pw_list, function(x) as.character(x$pathway_id)))
write_log(sprintf("Total pathway entries: %d | Unique pathway IDs: %d", n_pw, length(unique_ids)))

# ============================================================================
# Helper: extract gene symbol from massdatabase Gene.name column
#   Gene.name format: "SYMBOL; description [KO:...] [EC:...]"
# ============================================================================
extract_gene_symbol <- function(gene_name) {
  symbols <- as.character(gene_name)
  
  # Valid entries must have a symbol before the first semicolon
  has_semicolon <- grepl(";", symbols)
  symbols[!has_semicolon] <- NA_character_
  
  symbols[has_semicolon] <- sub("^([^;]+);.*$", "\\1", symbols[has_semicolon])
  symbols <- trimws(symbols)
  
  # Filter out KO-only / EC-only / empty / hsa: entries
  symbols <- ifelse(grepl("^\\[", symbols), NA_character_, symbols)
  symbols <- ifelse(grepl("^hsa:", symbols), NA_character_, symbols)
  symbols <- ifelse(symbols == "", NA_character_, symbols)
  symbols
}

# ============================================================================
# Build summary table (one row per pathway)
# ============================================================================
write_log("Building pathway summary table...")

summary_rows <- lapply(seq_along(pw_list), function(i) {
  pw <- pw_list[[i]]
  
  safe_char <- function(x) {
    if (is.null(x)) return(NA_character_)
    if (length(x) == 0) return(NA_character_)
    as.character(x)[1]
  }
  
  # Count unique valid gene symbols (consistent with the de-duplicated gene_long table)
  gene_count <- 0L
  if (!is.null(pw$gene_list) && is.data.frame(pw$gene_list) && "Gene.name" %in% names(pw$gene_list)) {
    symbols <- extract_gene_symbol(pw$gene_list$Gene.name)
    gene_count <- as.integer(length(unique(symbols[!is.na(symbols)])))
  }
  
  compound_count <- 0L
  if (!is.null(pw$compound_list) && is.data.frame(pw$compound_list)) {
    compound_count <- as.integer(nrow(pw$compound_list))
  }
  
  data.table(
    pathway_id = safe_char(pw$pathway_id),
    pathway_name = safe_char(pw$pathway_name),
    pathway_class = safe_char(pw$pathway_class),
    gene_count = gene_count,
    compound_count = compound_count
  )
})

summary_dt <- rbindlist(summary_rows, fill = TRUE, use.names = TRUE)
summary_file <- file.path(out_dir, "kegg_human_pathways_summary.tsv")
fwrite(summary_dt, summary_file, sep = "\t", quote = FALSE)
write_log(sprintf("Wrote summary: %s (%d rows)", summary_file, nrow(summary_dt)))

# ============================================================================
# Build gene membership table (long format)
# ============================================================================
write_log("Building pathway-gene membership table...")

gene_rows <- list()
n_with_genes <- 0L

for (i in seq_along(pw_list)) {
  pw <- pw_list[[i]]
  gl <- pw$gene_list
  if (is.null(gl) || !is.data.frame(gl) || nrow(gl) == 0) next
  
  n_with_genes <- n_with_genes + 1L
  gl_dt <- as.data.table(gl)
  
  if ("Gene.name" %in% names(gl_dt)) {
    gl_dt$gene_symbol <- extract_gene_symbol(gl_dt$Gene.name)
  } else {
    gl_dt$gene_symbol <- NA_character_
  }
  
  gl_dt$pathway_id <- as.character(pw$pathway_id)
  gl_dt$pathway_name <- as.character(pw$pathway_name)
  gl_dt$pathway_class <- as.character(pw$pathway_class)
  
  leading <- c("pathway_id", "pathway_name", "pathway_class", "gene_symbol")
  other <- setdiff(names(gl_dt), leading)
  gl_dt <- gl_dt[, c(leading, other), with = FALSE]
  
  gene_rows[[length(gene_rows) + 1]] <- gl_dt
}

gene_dt <- NULL
gene_file <- file.path(out_dir, "kegg_human_pathway_genes.tsv")
if (length(gene_rows) > 0) {
  gene_dt <- rbindlist(gene_rows, fill = TRUE, use.names = TRUE)
  
  # Deduplicate by pathway_id + gene_symbol (same gene may appear with
  # multiple KEGG IDs / isoform descriptions in KEGG)
  before_dedup <- nrow(gene_dt)
  gene_dt <- unique(gene_dt, by = c("pathway_id", "gene_symbol"))
  after_dedup <- nrow(gene_dt)
  n_dropped <- before_dedup - after_dedup
  if (n_dropped > 0) {
    write_log(sprintf("INFO: dropped %d duplicated pathway-gene pairs", n_dropped))
  }
  
  fwrite(gene_dt, gene_file, sep = "\t", quote = FALSE)
  write_log(sprintf(
    "Wrote gene membership: %s (%d rows, %d pathways with genes)",
    gene_file, nrow(gene_dt), n_with_genes
  ))
} else {
  write_log("WARNING: No gene_list found in any pathway.")
}

# ============================================================================
# Build GMT file (GSEA-compatible)
# ============================================================================
write_log("Building GMT file...")

gmt_lines <- character()
for (pw in pw_list) {
  gl <- pw$gene_list
  if (is.null(gl) || !is.data.frame(gl) || nrow(gl) == 0) next
  
  gl_dt <- as.data.table(gl)
  if ("Gene.name" %in% names(gl_dt)) {
    symbols <- extract_gene_symbol(gl_dt$Gene.name)
  } else {
    next
  }
  
  symbols <- symbols[!is.na(symbols) & symbols != ""]
  symbols <- unique(symbols)
  if (length(symbols) == 0) next
  
  desc <- ifelse(
    is.null(pw$pathway_name) || is.na(pw$pathway_name),
    as.character(pw$pathway_id),
    as.character(pw$pathway_name)
  )
  line <- paste(c(as.character(pw$pathway_id), desc, symbols), collapse = "\t")
  gmt_lines <- c(gmt_lines, line)
}

gmt_file <- file.path(out_dir, "kegg_human_pathways.gmt")
writeLines(gmt_lines, gmt_file)
write_log(sprintf("Wrote GMT: %s (%d pathways)", gmt_file, length(gmt_lines)))

# ============================================================================
# Self-check / validation
# ============================================================================
write_log("Running self-check...")

# 1. Summary table consistency
n_summary <- nrow(summary_dt)
n_unique_summary <- uniqueN(summary_dt$pathway_id)
if (n_summary != n_unique_summary) {
  stop(sprintf(
    "Summary table has duplicated pathway IDs: %d rows, %d unique IDs",
    n_summary, n_unique_summary
  ))
}
write_log(sprintf(
  "PASS: summary has %d rows and %d unique pathway IDs",
  n_summary, n_unique_summary
))

# 2. Gene table consistency
if (!is.null(gene_dt)) {
  n_gene_rows <- nrow(gene_dt)
  n_unique_gene_pw <- uniqueN(gene_dt$pathway_id)
  n_unique_symbols <- uniqueN(gene_dt$gene_symbol, na.rm = TRUE)
  n_na_symbols <- sum(is.na(gene_dt$gene_symbol))
  n_dup_pairs <- nrow(gene_dt) - uniqueN(gene_dt, by = c("pathway_id", "gene_symbol"))
  
  write_log(sprintf(
    "PASS: gene table has %d rows, %d pathways, %d unique symbols",
    n_gene_rows, n_unique_gene_pw, n_unique_symbols
  ))
  
  if (n_na_symbols > 0) {
    write_log(sprintf(
      "INFO: %d rows have missing gene_symbol (KO-only or unnamed entries)",
      n_na_symbols
    ))
  }
  
  if (n_dup_pairs > 0) {
    stop(sprintf("Gene table still has %d duplicated pathway-gene pairs", n_dup_pairs))
  }
  write_log("PASS: no duplicated pathway-gene pairs in gene table")
  
  # 3. GMT pathways must exist in gene table
  gmt_ids <- sapply(gmt_lines, function(line) strsplit(line, "\t")[[1]][1])
  missing_in_gene <- setdiff(gmt_ids, gene_dt$pathway_id)
  if (length(missing_in_gene) > 0) {
    stop(sprintf(
      "GMT pathways missing in gene table: %s",
      paste(missing_in_gene, collapse = ", ")
    ))
  }
  write_log(sprintf(
    "PASS: all %d GMT pathways are present in gene table",
    length(gmt_ids)
  ))
  
  # 4. GMT has no empty/NA/invalid symbols and no within-pathway duplicates
  gmt_problems <- character()
  for (line in gmt_lines) {
    parts <- strsplit(line, "\t")[[1]]
    symbols <- parts[-c(1, 2)]
    if (any(symbols == "" | is.na(symbols))) {
      gmt_problems <- c(gmt_problems, sprintf("%s: empty/NA symbol", parts[1]))
    }
    if (any(grepl("^\\[", symbols))) {
      gmt_problems <- c(gmt_problems, sprintf("%s: KO-only symbol leaked", parts[1]))
    }
    if (length(symbols) != length(unique(symbols))) {
      gmt_problems <- c(gmt_problems, sprintf("%s: duplicate symbols", parts[1]))
    }
  }
  if (length(gmt_problems) > 0) {
    stop(sprintf("GMT validation failed:\n%s", paste(head(gmt_problems, 10), collapse = "\n")))
  }
  write_log("PASS: GMT symbols are valid and unique within each pathway")
  
  # 5. summary gene_count matches gene_long counts
  actual_counts <- gene_dt[!is.na(gene_symbol), .(actual = .N), by = pathway_id]
  merged <- summary_dt[actual_counts, on = "pathway_id"]
  mismatched <- merged[gene_count != actual]
  if (nrow(mismatched) > 0) {
    stop(sprintf(
      "gene_count mismatch for %d pathways (first: %s, summary=%d, actual=%d)",
      nrow(mismatched), mismatched$pathway_id[1],
      mismatched$gene_count[1], mismatched$actual[1]
    ))
  }
  write_log("PASS: summary gene_count matches gene_long counts")
}

# 6. Expected output files exist and are readable
for (f in c(summary_file, gene_file, gmt_file)) {
  if (!file.exists(f)) {
    stop(sprintf("Missing output file: %s", f))
  }
  info <- file.info(f)
  if (info$size == 0) {
    stop(sprintf("Output file is empty: %s", f))
  }
}
write_log("PASS: all expected output files exist and are non-empty")

# 7. Sanity check on gene symbol pattern
if (!is.null(gene_dt)) {
  valid_symbols <- grepl("^[A-Za-z0-9_.\\-]+$", gene_dt$gene_symbol[!is.na(gene_dt$gene_symbol)])
  if (any(!valid_symbols)) {
    bad <- unique(gene_dt$gene_symbol[!is.na(gene_dt$gene_symbol)][!valid_symbols])
    stop(sprintf("Invalid gene_symbol patterns found: %s", paste(head(bad, 10), collapse = ", ")))
  }
  write_log("PASS: all gene_symbol values match expected pattern")
}

# ============================================================================
# Final report
# ============================================================================
write_log("=== Done ===")
write_log(sprintf("Total pathways: %d", n_pw))
write_log(sprintf("Pathways with genes: %d", n_with_genes))
if (!is.null(gene_dt)) {
  write_log(sprintf("Total pathway-gene pairs: %d", nrow(gene_dt)))
  write_log(sprintf("Unique gene symbols: %d", uniqueN(gene_dt$gene_symbol, na.rm = TRUE)))
}
write_log(sprintf("Output directory: %s", out_dir))

close(log_con)
