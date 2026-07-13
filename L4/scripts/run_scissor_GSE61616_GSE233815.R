# Scissor analysis: GSE61616 (rat bulk, MCAO vs Sham, 7d) + GSE233815 (mouse snRNA-seq)
# Steps:
# 1. affy::rma() on CEL files -> log2 RMA expression matrix
# 2. Probe annotation with GPL1355 / rat2302.db
# 3. Ortholog conversion rat -> human -> mouse
# 4. Build phenotype: MCAO=1, Sham=0 (binary logistic regression)
# 5. Subset scRNA-seq (7DPI first; fall back to all cells)
# 6. Run Scissor with family="binomial", reliability test first

suppressPackageStartupMessages({
  library(affy)
  library(rat2302.db)
  library(AnnotationDbi)
  library(babelgene)
  library(preprocessCore)
  library(Scissor)
  library(Seurat)
  library(dplyr)
})

set.seed(42)

# ---- Paths ----
proj_root <- "d:/铁衰老 绝不重蹈覆辙"
cel_dir <- file.path(proj_root, "L4/results/scissor_GSE61616_GSE233815")
seurat_path <- file.path(proj_root, "data/external/GSE233815/mendeley/Seurat_sn_MCAO_Zucha2023_QCFiltered.Rds")
meta_path <- file.path(proj_root, "L1/results/GSE61616_sample_meta.csv")
rat2human_path <- file.path(proj_root, "L1/results/rat_to_human_ortholog_mygene.csv")
human2mouse_path <- file.path(proj_root, "L2/results/GSE233815_sn/human_to_mouse_orthologs.csv")
out_dir <- file.path(proj_root, "L4/results/scissor_GSE61616_GSE233815")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

log_file <- file.path(out_dir, "scissor_run.log")
con <- file(log_file, open = "wt")
sink(con, type = "output")
sink(con, type = "message")
on.exit({
  sink(type = "message")
  sink(type = "output")
  close(con)
  message("Log file closed safely.")
}, add = TRUE)

message("Starting Scissor analysis at ", Sys.time())

# ---- 1. RMA normalization of CEL files ----
message("\n[1/6] Reading CEL files and running RMA normalization...")
cel_files <- list.files(cel_dir, pattern = "\\.CEL(\\.gz)?$", full.names = TRUE)
message("CEL files found: ", length(cel_files))
print(basename(cel_files))

affy_data <- ReadAffy(filenames = cel_files)
eset_rma <- rma(affy_data)
bulk_expr <- exprs(eset_rma)
message("Bulk expression matrix dimensions (probes x samples): ", paste(dim(bulk_expr), collapse = " x "))

# Clean sample names: GSMxxxxxx_Sham_1.CEL.gz -> Sham_1
sample_names <- gsub("\\.CEL\\.gz$", "", basename(colnames(bulk_expr)))
sample_names <- sub("^GSM[0-9]+_", "", sample_names)
colnames(bulk_expr) <- sample_names
message("Sample names: ", paste(sample_names, collapse = ", "))

# ---- 2. Probe annotation to rat gene symbol ----
message("\n[2/6] Annotating probes to rat gene symbols...")
probe_ids <- rownames(bulk_expr)
map <- AnnotationDbi::select(rat2302.db, keys = probe_ids, columns = c("SYMBOL"), keytype = "PROBEID")
message("Probes with annotation: ", length(unique(map$PROBEID)), "/", length(probe_ids))

# Collapse probes to gene symbol by mean
map <- map[!is.na(map$SYMBOL) & map$SYMBOL != "", ]
map$SYMBOL <- trimws(map$SYMBOL)

# Split multi-symbol probes (e.g., "A1i3 /// Cpamd8 /// ...")
map_long <- do.call(rbind, lapply(seq_len(nrow(map)), function(i) {
  symbols <- strsplit(map$SYMBOL[i], " /// ")[[1]]
  symbols <- trimws(symbols)
  symbols <- symbols[symbols != ""]
  symbols <- symbols[!grepl("^LOC", symbols)]
  if (length(symbols) == 0) return(NULL)
  data.frame(PROBEID = map$PROBEID[i], SYMBOL = symbols, stringsAsFactors = FALSE)
}))

bulk_df <- as.data.frame(bulk_expr)
bulk_df$PROBEID <- rownames(bulk_df)
merged <- merge(map_long, bulk_df, by = "PROBEID", all.x = TRUE)

# Compute mean per gene symbol across all probes
gene_cols <- setdiff(colnames(merged), c("PROBEID", "SYMBOL"))
bulk_gene <- merged %>%
  group_by(SYMBOL) %>%
  summarise(across(all_of(gene_cols), \(x) mean(x, na.rm = TRUE)), .groups = "drop") %>%
  as.data.frame()
rownames(bulk_gene) <- bulk_gene$SYMBOL
bulk_gene$SYMBOL <- NULL
bulk_gene <- as.matrix(bulk_gene)
message("Gene-level bulk expression matrix dimensions (genes x samples): ", paste(dim(bulk_gene), collapse = " x "))

# Save intermediate
write.csv(bulk_gene, file.path(out_dir, "GSE61616_bulk_gene_level_rma.csv"))

# ---- 3. Ortholog conversion rat -> human -> mouse ----
message("\n[3/6] Converting rat gene symbols to mouse orthologs...")

# Step 2 already collapsed probes to clean gene symbols via group_by(SYMBOL)
rat_genes_unique <- rownames(bulk_gene)
rat_genes_unique <- rat_genes_unique[!grepl("^LOC", rat_genes_unique)]
message("Unique rat gene symbols to map: ", length(rat_genes_unique))

# Map rat -> human -> mouse using babelgene
message("Mapping rat -> human...")
rat2human <- orthologs(genes = rat_genes_unique, species = "rat", human = FALSE)
message("Rat -> human mappings: ", nrow(rat2human))
rat2human <- rat2human %>%
  select(rat_symbol = symbol, human_symbol = human_symbol) %>%
  distinct(rat_symbol, .keep_all = TRUE)

message("Mapping human -> mouse...")
human_symbols_unique <- unique(rat2human$human_symbol)
human2mouse <- orthologs(genes = human_symbols_unique, species = "mouse", human = TRUE)
message("Human -> mouse mappings: ", nrow(human2mouse))
human2mouse <- human2mouse %>%
  select(human_symbol, mouse_symbol = symbol) %>%
  distinct(human_symbol, .keep_all = TRUE)

chain_map <- rat2human %>%
  inner_join(human2mouse, by = "human_symbol") %>%
  select(rat_symbol, human_symbol, mouse_symbol) %>%
  distinct(rat_symbol, .keep_all = TRUE)

# Filter to genes present in bulk matrix (direct mapping since rownames are clean)
chain_map <- chain_map[chain_map$rat_symbol %in% rownames(bulk_gene), ]
message("Rat -> Human -> Mouse chain mappings: ", nrow(chain_map))

# Subset bulk to mapped genes
bulk_mouse <- bulk_gene[chain_map$rat_symbol, , drop = FALSE]
rownames(bulk_mouse) <- chain_map$mouse_symbol

# Collapse duplicate mouse symbols by mean
bulk_mouse_df <- as.data.frame(bulk_mouse)
bulk_mouse_df$mouse_symbol <- rownames(bulk_mouse_df)
bulk_mouse_collapsed <- bulk_mouse_df %>%
  group_by(mouse_symbol) %>%
  summarise(across(everything(), \(x) mean(x, na.rm = TRUE)), .groups = "drop") %>%
  as.data.frame()
rownames(bulk_mouse_collapsed) <- bulk_mouse_collapsed$mouse_symbol
bulk_mouse_collapsed$mouse_symbol <- NULL
bulk_mouse <- as.matrix(bulk_mouse_collapsed)
message("Final bulk matrix (mouse orthologs x samples): ", paste(dim(bulk_mouse), collapse = " x "))

write.csv(bulk_mouse, file.path(out_dir, "GSE61616_bulk_mouse_ortholog.csv"))

# ---- 4. Build phenotype: MCAO=1, Sham=0 ----
message("\n[4/6] Building phenotype vector...")
sample_meta <- read.csv(meta_path, stringsAsFactors = FALSE)
# Keep only samples present in bulk matrix and groups MCAO/Sham
sample_meta <- sample_meta[sample_meta$sample %in% colnames(bulk_mouse), ]
sample_meta <- sample_meta[sample_meta$group %in% c("MCAO", "Sham"), ]
sample_meta <- sample_meta[match(colnames(bulk_mouse), sample_meta$sample), ]
sample_meta <- sample_meta[!is.na(sample_meta$sample), ]

bulk_mouse <- bulk_mouse[, sample_meta$sample, drop = FALSE]
phenotype <- ifelse(sample_meta$group == "MCAO", 1, 0)
names(phenotype) <- sample_meta$sample
message("Phenotype table:")
print(table(phenotype))
write.csv(data.frame(sample = names(phenotype), group = sample_meta$group, phenotype = phenotype),
          file.path(out_dir, "GSE61616_phenotype.csv"), row.names = FALSE)

# ---- 5. Load and subset single-cell data ----
message("\n[5/6] Loading single-cell data...")
seu <- readRDS(seurat_path)
message("Original Seurat object: ")
print(seu)
message("Conditions: ")
print(table(seu$Condition))
message("Cell types: ")
print(table(seu$cell_type_1))

# Try 7DPI first, fall back to all cells if too few
if ("7DPI" %in% seu$Condition) {
  seu_subset <- subset(seu, subset = Condition == "7DPI")
  n_7dpi <- ncol(seu_subset)
  message("7DPI cells: ", n_7dpi)
  if (n_7dpi < 200) {
    message("Too few 7DPI cells, falling back to all conditions.")
    seu_subset <- seu
  }
} else {
  message("No 7DPI condition found, using all cells.")
  seu_subset <- seu
}

message("Selected scRNA-seq cells: ", ncol(seu_subset))
message("Selected conditions: ")
print(table(seu_subset$Condition))

# Use RNA assay counts/data for Scissor
DefaultAssay(seu_subset) <- "RNA"

# Memory check before converting to dense matrix
sc_raw <- GetAssayData(seu_subset, layer = "data")
message("Single-cell data size: ", format(object.size(sc_raw), units = "MB"))

# Seurat v4/v5 compatible data extraction
get_seurat_data <- function(seu_obj, lay = "data") {
  if (packageVersion("Seurat") >= "5.0.0") {
    as.matrix(SeuratObject::LayerData(seu_obj, assay = "RNA", layer = lay))
  } else {
    as.matrix(Seurat::GetAssayData(seu_obj, slot = lay))
  }
}

# Always re-normalize from counts to ensure consistent log-normalized input
if (packageVersion("Seurat") >= "5.0.0") {
  has_counts <- !is.null(SeuratObject::LayerData(seu_subset, assay = "RNA", layer = "counts"))
} else {
  has_counts <- !is.null(Seurat::GetAssayData(seu_subset, slot = "counts"))
}
if (has_counts) {
  message("Normalizing from counts layer (log-normalized)...")
  seu_subset <- NormalizeData(seu_subset, verbose = FALSE)
}
sc_expr <- get_seurat_data(seu_subset, lay = "data")
message("Single-cell expression matrix (genes x cells): ", paste(dim(sc_expr), collapse = " x "))

# ---- Intersection of genes ----
message("\nIntersecting bulk and single-cell genes...")
common_genes <- intersect(rownames(bulk_mouse), rownames(sc_expr))
message("Common genes: ", length(common_genes))
if (length(common_genes) < 3000) {
  stop("Too few common genes (<3000) for reliable Scissor analysis.")
}

bulk_scissor <- bulk_mouse[common_genes, , drop = FALSE]
sc_scissor <- sc_expr[common_genes, , drop = FALSE]

# Save inputs for reproducibility
saveRDS(list(bulk = bulk_scissor, phenotype = phenotype, sc = sc_scissor),
        file.path(out_dir, "scissor_inputs.rds"))

# ---- 6. Run Scissor ----
message("\n[6/6] Running Scissor...")

# Compute cell-cell similarity network (SNN graph)
message("\nComputing cell-cell similarity network...")
seu_subset <- FindVariableFeatures(seu_subset, selection.method = "vst", verbose = FALSE)
seu_subset <- ScaleData(seu_subset, verbose = FALSE)
seu_subset <- RunPCA(seu_subset, features = VariableFeatures(seu_subset), verbose = FALSE)
seu_subset <- FindNeighbors(seu_subset, dims = 1:10, verbose = FALSE)

n_cells <- ncol(seu_subset)
message("Total cells in network: ", n_cells)
if (n_cells > 30000) {
  warning("High cell count may cause memory issues during dense network conversion.")
}

network <- as.matrix(seu_subset@graphs$RNA_snn)
diag(network) <- 0
message("Network dimensions: ", paste(dim(network), collapse = " x "))

# Compute correlation matrix for reliability.test
# reliability.test source (test_logit) expects:
#   X = correlation matrix (bulk_samples x cells), Y = phenotype vector (0/1)
# It does NOT do quantile normalization internally; that is done in Scissor()
message("\nComputing correlation matrix for reliability test...")
dataset0 <- cbind(bulk_scissor, sc_scissor)
dataset1 <- preprocessCore::normalize.quantiles(dataset0)
rownames(dataset1) <- rownames(dataset0)
colnames(dataset1) <- colnames(dataset0)
Expression_bulk <- dataset1[, 1:ncol(bulk_scissor), drop = FALSE]
Expression_cell <- dataset1[, (ncol(bulk_scissor) + 1):ncol(dataset1), drop = FALSE]
X_cor <- cor(Expression_bulk, Expression_cell)
message("Correlation matrix dimensions: ", paste(dim(X_cor), collapse = " x "))

# Reliability test: wrapped in tryCatch because APML1 C++ may crash with large networks.
# test_logit source expects X = correlation matrix, Y = phenotype vector.
# Returns $p field (not $pvalue).
message("\nRunning reliability.test (n=5, nfold=3 to reduce C++ crash risk)...")
reliability <- tryCatch({
  reliability.test(
    X = X_cor,
    Y = phenotype,
    network = network,
    alpha = 0.05,
    family = "binomial",
    cell_num = ncol(sc_scissor),
    n = 5,
    nfold = 3
  )
}, error = function(e) {
  warning("reliability.test R-level error: ", e$message, ". Proceeding without.")
  return(NULL)
})

if (is.null(reliability)) {
  warning("reliability.test returned NULL (likely C++ crash in APML1). Skipping significance check.")
} else {
  message("Reliability test results:")
  print(reliability)
  write.csv(as.data.frame(reliability), file.path(out_dir, "scissor_reliability_test.csv"))
  if (!is.null(reliability$p) && reliability$p > 0.05) {
    warning("Reliability test not significant (p = ", format(reliability$p, digits = 4),
            "). Results may be unreliable.")
  }
}

# Build proper Seurat v5 object with all layers for Scissor
# Use raw counts from seu_subset to avoid double-normalization
message("Building Seurat object for Scissor...")
sc_counts <- get_seurat_data(seu_subset, lay = "counts")
sc_counts <- sc_counts[common_genes, , drop = FALSE]
sc_seurat <- CreateSeuratObject(counts = sc_counts)
sc_seurat <- NormalizeData(sc_seurat, verbose = FALSE)
sc_seurat <- FindVariableFeatures(sc_seurat, selection.method = "vst", verbose = FALSE)
sc_seurat <- ScaleData(sc_seurat, verbose = FALSE)
sc_seurat <- RunPCA(sc_seurat, features = VariableFeatures(sc_seurat), verbose = FALSE)
sc_seurat <- FindNeighbors(sc_seurat, dims = 1:10, verbose = FALSE)
message("Seurat object ready: ", ncol(sc_seurat), " cells, ", nrow(sc_seurat), " genes")

# Patch Scissor for Seurat v5 + R >= 4.4 compatibility
# Modifies the original function body via text replacement
message("Patching Scissor function for Seurat v5 compatibility...")
Scissor_patched <- Scissor
body_text <- deparse(body(Scissor_patched))
body_text <- gsub("class\\(sc_dataset\\) == \"Seurat\"", "inherits(sc_dataset, \"Seurat\")", body_text)
body_text <- gsub("sc_dataset@assays\\$RNA@data", "SeuratObject::LayerData(sc_dataset, assay = \"RNA\", layer = \"data\")", body_text)
body(Scissor_patched) <- parse(text = body_text)[[1]]
environment(Scissor_patched) <- asNamespace("Scissor")
message("Scissor function patched. Using patched version directly.")

# Run Scissor using patched function with Seurat object
message("\nRunning Scissor with family='binomial', alpha=0.05...")
scissor_results <- Scissor_patched(
  bulk_dataset = bulk_scissor,
  sc_dataset = sc_seurat,
  phenotype = phenotype,
  tag = c("Sham", "MCAO"),
  alpha = 0.05,
  family = "binomial",
  Save_file = file.path(out_dir, "Scissor_results.rds")
)

message("\nScissor results:")
print(scissor_results)

# Annotate cells with Scissor classification
seu_subset$scissor <- NA
if (!is.null(scissor_results)) {
  if (length(scissor_results$Scissor_pos) > 0) {
    seu_subset$scissor[scissor_results$Scissor_pos] <- "Scissor+"
  }
  if (length(scissor_results$Scissor_neg) > 0) {
    seu_subset$scissor[scissor_results$Scissor_neg] <- "Scissor-"
  }
} else {
  warning("Scissor returned NULL — no cells selected.")
}

message("\nScissor cell counts:")
print(table(seu_subset$scissor, useNA = "ifany"))

# Remove SCT assay to avoid Seurat v5 SCTModel saveRDS incompatibility
if ("SCT" %in% names(seu_subset@assays)) {
  seu_subset[["SCT"]] <- NULL
  message("Removed SCT assay for safe serialization.")
}
saveRDS(seu_subset, file.path(out_dir, "seurat_with_scissor.rds"))
message("Seurat object saved to ", file.path(out_dir, "seurat_with_scissor.rds"))

message("\nAnalysis completed at ", Sys.time())
