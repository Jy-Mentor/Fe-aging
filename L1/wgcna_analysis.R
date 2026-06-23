# 模块: L1/wgcna_analysis.R
# 功能: WGCNA共表达网络构建与模块-性状关联分析
# 输入: L1/results/*_expression_matrix.csv, L1/results/*_sample_meta.csv
# 输出: L1/results/wgcna_*/ 下各数据集WGCNA结果
# 运行: Rscript L1/wgcna_analysis.R

suppressPackageStartupMessages({
  library(WGCNA)
  library(limma)
  library(edgeR)
})

# Allow multi-threading
allowWGCNAThreads(nThreads = 4)

# ============================================================
# 路径配置
# ============================================================
project_root <- getwd()
cat("Project root:", project_root, "\n")
result_dir <- file.path(project_root, "L1", "results")
dir.create(result_dir, showWarnings = FALSE, recursive = TRUE)

log_file <- file.path(project_root, "logs", "wgcna_analysis.log")
dir.create(dirname(log_file), showWarnings = FALSE, recursive = TRUE)

log_con <- file(log_file, open = "w")
sink(log_con, split = TRUE)

cat("========================================\n")
cat("Phase 1 Step 22: WGCNA Analysis\n")
cat("Start time:", format(Sys.time()), "\n")
cat("========================================\n\n")

# ============================================================
# WGCNA function
# ============================================================
run_wgcna <- function(exp_mat, trait_data, dataset_name, n_genes = 8000) {
  cat("\n", paste(rep("=", 50), collapse = ""), "\n")
  cat("WGCNA for", dataset_name, "\n")
  cat("Expression matrix dim:", dim(exp_mat), "\n")
  cat("Samples:", ncol(exp_mat), "\n")
  
  # Transpose: genes as columns, samples as rows
  datExpr <- t(exp_mat)
  
  # Check for missing values
  gsg <- goodSamplesGenes(datExpr, verbose = 3)
  if (!gsg$allOK) {
    cat("Removing", sum(!gsg$goodGenes), "genes with too many missing values\n")
    datExpr <- datExpr[, gsg$goodGenes]
  }
  
  # Filter to top variable genes
  if (ncol(datExpr) > n_genes) {
    var_genes <- apply(datExpr, 2, var, na.rm = TRUE)
    top_idx <- order(var_genes, decreasing = TRUE)[1:n_genes]
    datExpr <- datExpr[, top_idx]
    cat("Filtered to top", n_genes, "variable genes\n")
  }
  
  # Choose soft-thresholding power
  powers <- c(1:30)
  sft <- pickSoftThreshold(datExpr, powerVector = powers, verbose = 5,
                           networkType = "signed")
  
  # Find power with R^2 > 0.85
  best_power <- sft$powerEstimate
  if (is.na(best_power)) {
    # Find first power with R^2 > 0.8
    for (p in powers) {
      if (sft$fitIndices$SFT.R.sq[p] > 0.8) {
        best_power <- p
        break
      }
    }
    if (is.na(best_power)) best_power <- 6  # fallback
  }
  cat("Selected soft-thresholding power:", best_power, "\n")
  
  # Save SFT plot data
  sft_df <- sft$fitIndices
  write.csv(sft_df, file.path(result_dir, paste0("wgcna_", dataset_name, "_sft.csv")),
            row.names = FALSE)
  
  # One-step network construction
  net <- blockwiseModules(
    datExpr,
    power = best_power,
    networkType = "signed",
    TOMType = "signed",
    minModuleSize = 30,
    reassignThreshold = 0,
    mergeCutHeight = 0.25,
    numericLabels = TRUE,
    pamRespectsDendro = FALSE,
    saveTOMs = TRUE,
    saveTOMFileBase = file.path(result_dir, paste0("wgcna_", dataset_name, "_TOM")),
    verbose = 3,
    maxBlockSize = n_genes
  )
  
  # Module eigengenes
  MEs <- net$MEs
  moduleLabels <- net$colors
  moduleColors <- labels2colors(moduleLabels)
  
  cat("Number of modules:", length(unique(moduleColors)), "\n")
  cat("Module sizes:\n")
  print(table(moduleColors))
  
  # Module-trait correlations
  nSamples <- nrow(datExpr)
  
  # Correlate module eigengenes with traits
  moduleTraitCor <- cor(MEs, trait_data, use = "p")
  moduleTraitPvalue <- corPvalueStudent(moduleTraitCor, nSamples)
  
  # Save results
  wgcna_dir <- file.path(result_dir, paste0("wgcna_", dataset_name))
  dir.create(wgcna_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Gene-module assignments
  gene_module_df <- data.frame(
    Gene = colnames(datExpr),
    Module = moduleColors,
    ModuleLabel = moduleLabels,
    stringsAsFactors = FALSE
  )
  write.csv(gene_module_df, file.path(wgcna_dir, "gene_module_assignment.csv"),
            row.names = FALSE)
  
  # Module-trait correlations
  mt_df <- as.data.frame(moduleTraitCor)
  mt_df$Module <- rownames(mt_df)
  write.csv(mt_df, file.path(wgcna_dir, "module_trait_correlation.csv"),
            row.names = FALSE)
  
  mt_p_df <- as.data.frame(moduleTraitPvalue)
  mt_p_df$Module <- rownames(mt_p_df)
  write.csv(mt_p_df, file.path(wgcna_dir, "module_trait_pvalue.csv"),
            row.names = FALSE)
  
  # Gene significance (GS) and module membership (MM)
  # For each trait, compute GS
  gs_list <- list()
  for (trait_name in colnames(trait_data)) {
    gs <- as.numeric(cor(datExpr, trait_data[, trait_name], use = "p"))
    gs_list[[trait_name]] <- gs
  }
  
  mm_list <- list()
  # Use actual ME column names to avoid mismatches
  me_cols <- grep("^ME", colnames(MEs), value = TRUE)
  for (me_col in me_cols) {
    mod <- sub("^ME", "", me_col)
    if (mod != "grey") {
      me <- MEs[, me_col]
      mm <- as.numeric(cor(datExpr, me, use = "p"))
      mm_list[[mod]] <- mm
    }
  }
  
  # Save GS and MM
  gs_df <- as.data.frame(gs_list)
  gs_df$Gene <- colnames(datExpr)
  write.csv(gs_df, file.path(wgcna_dir, "gene_significance.csv"),
            row.names = FALSE)
  
  mm_df <- as.data.frame(mm_list)
  mm_df$Gene <- colnames(datExpr)
  write.csv(mm_df, file.path(wgcna_dir, "module_membership.csv"),
            row.names = FALSE)
  
  cat("WGCNA results saved to:", wgcna_dir, "\n")
  
  return(list(
    net = net,
    moduleColors = moduleColors,
    MEs = MEs,
    moduleTraitCor = moduleTraitCor,
    moduleTraitPvalue = moduleTraitPvalue,
    datExpr = datExpr,
    genes = colnames(datExpr)
  ))
}

# ============================================================
# 1. WGCNA on GSE104036 (Mouse RNA-seq, 27 samples)
# ============================================================
cat("\n\n========================================\n")
cat("1. WGCNA on GSE104036 (Mouse RNA-seq)\n")
cat("========================================\n")

gse104036_exp <- file.path(result_dir, "GSE104036_expression_matrix.csv")
gse104036_meta <- file.path(result_dir, "GSE104036_sample_meta.csv")

if (file.exists(gse104036_exp) && file.exists(gse104036_meta)) {
  counts <- as.matrix(read.csv(gse104036_exp, row.names = 1, check.names = FALSE))
  meta <- read.csv(gse104036_meta, stringsAsFactors = FALSE)
  
  cat("Counts dim:", dim(counts), "\n")
  
  # Select Sham and Ipsilateral samples
  sham_idx <- which(meta$group == "Sham")
  i24_idx <- which(meta$group == "Ipsilateral" & meta$time == "24hr")
  selected <- c(sham_idx, i24_idx)
  
  cat("Selected samples: Sham n=", length(sham_idx),
      ", Ipsilateral 24h n=", length(i24_idx), "\n")
  
  if (length(selected) >= 10) {
    sub_counts <- counts[, selected]
    
    # Filter low-expressed genes
    dge <- DGEList(counts = sub_counts)
    keep <- filterByExpr(dge, min.count = 10, min.total.count = 15)
    dge <- dge[keep, , keep.lib.sizes = FALSE]
    cat("Genes after filtering:", nrow(dge$counts), "\n")
    
    # Normalize: TMM + log2 CPM
    dge <- calcNormFactors(dge, method = "TMM")
    log_cpm <- cpm(dge, log = TRUE, prior.count = 1)
    
    # Create trait matrix
    group <- factor(c(rep("Sham", length(sham_idx)),
                      rep("Ipsi", length(i24_idx))))
    trait_data <- data.frame(
      CIRI = as.numeric(group == "Ipsi"),
      row.names = colnames(log_cpm)
    )
    
    cat("Trait data:\n")
    print(table(trait_data$CIRI))
    
    gse104036_wgcna <- run_wgcna(log_cpm, trait_data, "GSE104036", n_genes = 8000)
    
    cat("\nGSE104036 WGCNA completed.\n")
    cat("CIRI-correlated modules:\n")
    cor_val <- gse104036_wgcna$moduleTraitCor[, "CIRI"]
    p_val <- gse104036_wgcna$moduleTraitPvalue[, "CIRI"]
    sig_modules <- gsub("^ME", "", names(cor_val)[which(abs(cor_val) > 0.3 & p_val < 0.05)])
    cat("Significant modules (|cor|>0.3, p<0.05):", paste(sig_modules, collapse = ", "), "\n")
  } else {
    cat("WARNING: Not enough samples for GSE104036 WGCNA\n")
  }
} else {
  cat("WARNING: GSE104036 files not found\n")
}

# ============================================================
# 2. WGCNA on GSE16561 (Human Illumina, 63 samples)
# ============================================================
cat("\n\n========================================\n")
cat("2. WGCNA on GSE16561 (Human Illumina)\n")
cat("========================================\n")

gse16561_exp <- file.path(result_dir, "GSE16561_expression_matrix.csv")
gse16561_meta <- file.path(result_dir, "GSE16561_sample_meta.csv")

if (file.exists(gse16561_exp) && file.exists(gse16561_meta)) {
  exp_mat <- as.matrix(read.csv(gse16561_exp, row.names = 1, check.names = FALSE))
  meta <- read.csv(gse16561_meta, stringsAsFactors = FALSE)
  
  cat("Expression matrix dim:", dim(exp_mat), "\n")
  
  # Log2 transform if needed
  if (max(exp_mat, na.rm = TRUE) > 100) {
    cat("Applying log2 transformation\n")
    exp_mat <- log2(exp_mat + 1)
  }
  
  # Quantile normalize
  exp_norm <- normalizeBetweenArrays(exp_mat, method = "quantile")
  
  # Filter low-variance genes (keep top 8000)
  var_genes <- apply(exp_norm, 1, var, na.rm = TRUE)
  top_idx <- order(var_genes, decreasing = TRUE)[1:min(8000, length(var_genes))]
  exp_norm <- exp_norm[top_idx, ]
  cat("Filtered to", nrow(exp_norm), "variable genes\n")
  
  # Create trait matrix
  group <- factor(meta$group, levels = c("Control", "Stroke"))
  trait_data <- data.frame(
    Stroke = as.numeric(group == "Stroke"),
    row.names = colnames(exp_norm)
  )
  
  cat("Trait data:\n")
  print(table(trait_data$Stroke))
  
  gse16561_wgcna <- run_wgcna(exp_norm, trait_data, "GSE16561", n_genes = 8000)
  
  cat("\nGSE16561 WGCNA completed.\n")
  cat("Stroke-correlated modules:\n")
  cor_val <- gse16561_wgcna$moduleTraitCor[, "Stroke"]
  p_val <- gse16561_wgcna$moduleTraitPvalue[, "Stroke"]
  sig_modules <- gsub("^ME", "", names(cor_val)[which(abs(cor_val) > 0.3 & p_val < 0.05)])
  cat("Significant modules (|cor|>0.3, p<0.05):", paste(sig_modules, collapse = ", "), "\n")
} else {
  cat("WARNING: GSE16561 files not found\n")
}

cat("\n========================================\n")
cat("End time:", format(Sys.time()), "\n")
cat("========================================\n")

sink()