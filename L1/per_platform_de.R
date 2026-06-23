# 模块: L1/per_platform_de.R
# 功能: 对每个数据集独立进行差异表达分析，然后Robust Rank Aggregation荟萃分析
# 输入: Python解析后的表达矩阵CSV + CEL文件
# 输出: L1/results/ 下各数据集DE结果 + RRA整合结果
# 运行: Rscript L1/per_platform_de.R

# ============================================================
# 加载R包
# ============================================================
suppressPackageStartupMessages({
  library(limma)
  library(edgeR)
  library(affy)
  library(oligo)
  library(RobustRankAggreg)
  library(biomaRt)
})

# ============================================================
# 路径配置
# ============================================================
project_root <- getwd()
cat("Project root:", project_root, "\n")
data_dir <- file.path(project_root, "L1 数据集", "bulk")
result_dir <- file.path(project_root, "L1", "results")
dir.create(result_dir, showWarnings = FALSE, recursive = TRUE)

log_file <- file.path(project_root, "logs", "per_platform_de.log")
dir.create(dirname(log_file), showWarnings = FALSE, recursive = TRUE)

log_con <- file(log_file, open = "w")
sink(log_con, split = TRUE)

cat("========================================\n")
cat("Phase 1 Step 19: Per-platform DE + RRA\n")
cat("Start time:", format(Sys.time()), "\n")
cat("========================================\n\n")

# ============================================================
# 1. GSE104036 (Mouse RNA-seq) - edgeR DE
# ============================================================
cat("\n--- GSE104036: Mouse RNA-seq DE with edgeR ---\n")

gse104036_exp <- file.path(result_dir, "GSE104036_expression_matrix.csv")
gse104036_meta <- file.path(result_dir, "GSE104036_sample_meta.csv")

if (file.exists(gse104036_exp) && file.exists(gse104036_meta)) {
  counts <- as.matrix(read.csv(gse104036_exp, row.names = 1, check.names = FALSE))
  meta <- read.csv(gse104036_meta, stringsAsFactors = FALSE)
  
  cat("Counts matrix dim:", dim(counts), "\n")
  cat("Meta rows:", nrow(meta), "\n")
  
  # Select Sham vs Ipsilateral 24hr
  sham_idx <- which(meta$group == "Sham")
  i24_idx <- which(meta$group == "Ipsilateral" & meta$time == "24hr")
  selected <- c(sham_idx, i24_idx)
  
  cat("Sham samples:", meta$sample[sham_idx], "\n")
  cat("Ipsilateral 24hr samples:", meta$sample[i24_idx], "\n")
  
  if (length(sham_idx) >= 2 && length(i24_idx) >= 2) {
    sub_counts <- counts[, selected]
    group <- factor(c(rep("Sham", length(sham_idx)), rep("Ipsi", length(i24_idx))))
    
    # edgeR pipeline
    dge <- DGEList(counts = sub_counts, group = group)
    keep <- filterByExpr(dge)
    dge <- dge[keep, , keep.lib.sizes = FALSE]
    dge <- calcNormFactors(dge, method = "TMM")
    dge <- estimateDisp(dge)
    
    et <- exactTest(dge, pair = c("Sham", "Ipsi"))
    gse104036_de <- topTags(et, n = Inf)$table
    gse104036_de$Gene <- rownames(gse104036_de)
    gse104036_de$Dataset <- "GSE104036"
    gse104036_de$Species <- "Mouse"
    
    write.csv(gse104036_de, file.path(result_dir, "GSE104036_DE_results.csv"), row.names = FALSE)
    cat("GSE104036 DE: ", nrow(gse104036_de), " genes, ",
        sum(gse104036_de$FDR < 0.05), " significant (FDR<0.05)\n")
  } else {
    cat("WARNING: Not enough samples for GSE104036 DE\n")
    cat("Sham n=", length(sham_idx), ", Ipsi24hr n=", length(i24_idx), "\n")
  }
} else {
  cat("WARNING: GSE104036 files not found\n")
}

# ============================================================
# 2. GSE16561 (Human Illumina) - limma DE
# ============================================================
cat("\n--- GSE16561: Human Illumina DE with limma ---\n")

gse16561_exp <- file.path(result_dir, "GSE16561_expression_matrix.csv")
gse16561_meta <- file.path(result_dir, "GSE16561_sample_meta.csv")

if (file.exists(gse16561_exp) && file.exists(gse16561_meta)) {
  exp_mat <- as.matrix(read.csv(gse16561_exp, row.names = 1, check.names = FALSE))
  meta <- read.csv(gse16561_meta, stringsAsFactors = FALSE)
  
  cat("Expression matrix dim:", dim(exp_mat), "\n")
  
  # Select Stroke and Control samples
  group <- factor(meta$group, levels = c("Control", "Stroke"))
  
  cat("Groups:", table(group), "\n")
  
  if (sum(group == "Control") >= 2 && sum(group == "Stroke") >= 2) {
    # log2 transform if needed (Illumina data is usually already log2)
    if (max(exp_mat, na.rm = TRUE) > 100) {
      cat("Applying log2 transformation\n")
      exp_mat <- log2(exp_mat + 1)
    }
    
    # Quantile normalization
    exp_norm <- normalizeBetweenArrays(exp_mat, method = "quantile")
    
    # limma DE
    design <- model.matrix(~ group)
    fit <- lmFit(exp_norm, design)
    fit <- eBayes(fit, trend = TRUE)
    
    gse16561_de <- topTable(fit, coef = "groupStroke", number = Inf, sort.by = "none")
    gse16561_de$Probe <- rownames(gse16561_de)
    gse16561_de$Dataset <- "GSE16561"
    gse16561_de$Species <- "Human"
    
    write.csv(gse16561_de, file.path(result_dir, "GSE16561_DE_results.csv"), row.names = FALSE)
    cat("GSE16561 DE: ", nrow(gse16561_de), " probes, ",
        sum(gse16561_de$adj.P.Val < 0.05), " significant (adj.P<0.05)\n")
  }
} else {
  cat("WARNING: GSE16561 files not found\n")
}

# ============================================================
# 3. GSE37587 (Human Illumina non-norm) - limma DE
# NOTE: All samples are stroke patients, comparing baseline vs follow-up
# ============================================================
cat("\n--- GSE37587: Human Illumina non-norm ---\n")
cat("NOTE: GSE37587 has no healthy controls. All are stroke patients.\n")
cat("Checking for baseline vs follow-up comparison...\n")

gse37587_exp <- file.path(result_dir, "GSE37587_expression_matrix.csv")
gse37587_meta <- file.path(result_dir, "GSE37587_sample_meta.csv")

if (file.exists(gse37587_exp) && file.exists(gse37587_meta)) {
  # Read the series matrix to get time information
  series_file <- file.path(data_dir, "GSE37587", "GSE37587_series_matrix (1).txt.gz")
  if (file.exists(series_file)) {
    con <- gzfile(series_file, "rt", encoding = "UTF-8")
    series_lines <- readLines(con)
    close(con)
    
    # Extract time information from characteristics
    time_info <- NULL
    for (line in series_lines) {
      if (grepl("^!Sample_characteristics_ch1", line)) {
        if (grepl("time:", line, ignore.case = TRUE)) {
          parts <- strsplit(line, "\t")[[1]]
          time_info <- trimws(gsub('"', '', parts[-1]))
          break
        }
      }
    }
    
    if (!is.null(time_info)) {
      cat("Time info found:", table(time_info), "\n")
      
      exp_mat <- as.matrix(read.csv(gse37587_exp, row.names = 1, check.names = FALSE))
      cat("Expression matrix dim:", dim(exp_mat), "\n")
      
      # Filter to baseline and follow-up
      baseline_idx <- which(grepl("Baseline", time_info, ignore.case = TRUE))
      followup_idx <- which(grepl("Follow", time_info, ignore.case = TRUE))
      
      if (length(baseline_idx) >= 2 && length(followup_idx) >= 2) {
        cat("Baseline n=", length(baseline_idx), ", Follow-up n=", length(followup_idx), "\n")
        
        selected <- c(baseline_idx, followup_idx)
        sub_exp <- exp_mat[, selected]
        group <- factor(c(rep("Baseline", length(baseline_idx)), rep("Followup", length(followup_idx))))
        
        # log2 transform
        if (max(sub_exp, na.rm = TRUE) > 100) {
          cat("Applying log2 transformation\n")
          sub_exp <- log2(sub_exp + 1)
        }
        
        sub_norm <- normalizeBetweenArrays(sub_exp, method = "quantile")
        
        design <- model.matrix(~ group)
        fit <- lmFit(sub_norm, design)
        fit <- eBayes(fit, trend = TRUE)
        
        gse37587_de <- topTable(fit, coef = "groupFollowup", number = Inf, sort.by = "none")
        gse37587_de$Probe <- rownames(gse37587_de)
        gse37587_de$Dataset <- "GSE37587"
        gse37587_de$Species <- "Human"
        
        write.csv(gse37587_de, file.path(result_dir, "GSE37587_DE_results.csv"), row.names = FALSE)
        cat("GSE37587 DE: ", nrow(gse37587_de), " probes, ",
            sum(gse37587_de$adj.P.Val < 0.05), " significant (adj.P<0.05)\n")
      } else {
        cat("WARNING: Not enough baseline/follow-up samples for GSE37587\n")
      }
    } else {
      cat("WARNING: Could not find time information in GSE37587 series matrix\n")
    }
  }
} else {
  cat("WARNING: GSE37587 files not found\n")
}

# ============================================================
# 4. GSE61616 (Rat Affymetrix) - RMA normalization + limma DE
# ============================================================
cat("\n--- GSE61616: Rat Affymetrix DE with limma ---\n")

gse61616_cel <- file.path(result_dir, "GSE61616_CEL")
gse61616_meta <- file.path(result_dir, "GSE61616_sample_meta.csv")

if (dir.exists(gse61616_cel) && file.exists(gse61616_meta)) {
  cel_files <- list.files(gse61616_cel, pattern = "\\.CEL\\.gz$", full.names = TRUE)
  cat("CEL files found:", length(cel_files), "\n")
  
  if (length(cel_files) >= 4) {
    # Read CEL files using oligo (for newer Affymetrix arrays)
    # Note: Rat Genome 230 2.0 is a 3' IVT array, use affy package
    raw_data <- affy::ReadAffy(filenames = cel_files)
    cat("Raw data loaded:", length(sampleNames(raw_data)), "samples\n")
    
    # RMA normalization (affy::rma for 3' IVT arrays)
    eset <- affy::rma(raw_data)
    exp_mat <- exprs(eset)
    cat("RMA normalized matrix dim:", dim(exp_mat), "\n")
    
    meta <- read.csv(gse61616_meta, stringsAsFactors = FALSE)
    
    # Select Sham and MCAO samples
    sham_idx <- which(meta$group == "Sham")
    mcao_idx <- which(meta$group == "MCAO")
    
    if (length(sham_idx) >= 2 && length(mcao_idx) >= 2) {
      selected <- c(sham_idx, mcao_idx)
      sub_exp <- exp_mat[, selected]
      group <- factor(c(rep("Sham", length(sham_idx)), rep("MCAO", length(mcao_idx))),
                      levels = c("Sham", "MCAO"))
      
      design <- model.matrix(~ group)
      fit <- lmFit(sub_exp, design)
      fit <- eBayes(fit, trend = TRUE)
      
      gse61616_de <- topTable(fit, coef = 2, number = Inf, sort.by = "none")
      gse61616_de$Probe <- rownames(gse61616_de)
      gse61616_de$Dataset <- "GSE61616"
      gse61616_de$Species <- "Rat"
      
      write.csv(gse61616_de, file.path(result_dir, "GSE61616_DE_results.csv"), row.names = FALSE)
      cat("GSE61616 DE: ", nrow(gse61616_de), " probes, ",
          sum(gse61616_de$adj.P.Val < 0.05), " significant (adj.P<0.05)\n")
    } else {
      cat("WARNING: Not enough samples for GSE61616 DE\n")
    }
  }
} else {
  cat("WARNING: GSE61616 CEL files not found\n")
}

# ============================================================
# 5. GSE97537 (Rat Affymetrix, 24h) - RMA normalization + limma DE
# ============================================================
cat("\n--- GSE97537: Rat Affymetrix DE with limma ---\n")

gse97537_cel <- file.path(result_dir, "GSE97537_CEL")
gse97537_meta <- file.path(result_dir, "GSE97537_sample_meta.csv")

if (dir.exists(gse97537_cel) && file.exists(gse97537_meta)) {
  cel_files <- list.files(gse97537_cel, pattern = "\\.CEL\\.gz$", full.names = TRUE)
  cat("CEL files found:", length(cel_files), "\n")
  
  if (length(cel_files) >= 4) {
    raw_data <- affy::ReadAffy(filenames = cel_files)
    cat("Raw data loaded:", length(sampleNames(raw_data)), "samples\n")
    
    eset <- affy::rma(raw_data)
    exp_mat <- exprs(eset)
    cat("RMA normalized matrix dim:", dim(exp_mat), "\n")
    
    meta <- read.csv(gse97537_meta, stringsAsFactors = FALSE)
    
    sham_idx <- which(meta$group == "Sham")
    mcao_idx <- which(meta$group == "MCAO")
    
    if (length(sham_idx) >= 2 && length(mcao_idx) >= 2) {
      selected <- c(sham_idx, mcao_idx)
      sub_exp <- exp_mat[, selected]
      group <- factor(c(rep("Sham", length(sham_idx)), rep("MCAO", length(mcao_idx))),
                      levels = c("Sham", "MCAO"))
      
      design <- model.matrix(~ group)
      fit <- lmFit(sub_exp, design)
      fit <- eBayes(fit, trend = TRUE)
      
      gse97537_de <- topTable(fit, coef = 2, number = Inf, sort.by = "none")
      gse97537_de$Probe <- rownames(gse97537_de)
      gse97537_de$Dataset <- "GSE97537"
      gse97537_de$Species <- "Rat"
      
      write.csv(gse97537_de, file.path(result_dir, "GSE97537_DE_results.csv"), row.names = FALSE)
      cat("GSE97537 DE: ", nrow(gse97537_de), " probes, ",
          sum(gse97537_de$adj.P.Val < 0.05), " significant (adj.P<0.05)\n")
    } else {
      cat("WARNING: Not enough samples for GSE97537 DE\n")
    }
  }
} else {
  cat("WARNING: GSE97537 CEL files not found\n")
}

# ============================================================
# 6. Robust Rank Aggregation (RRA) - 荟萃分析
# ============================================================
cat("\n--- Robust Rank Aggregation ---\n")

# Collect all DE results
de_files <- list.files(result_dir, pattern = "_DE_results\\.csv$", full.names = TRUE)
cat("DE result files found:", length(de_files), "\n")

if (length(de_files) >= 2) {
  all_ranks <- list()
  
  for (f in de_files) {
    de <- read.csv(f, stringsAsFactors = FALSE)
    ds_name <- de$Dataset[1]
    
    # Determine the gene/Probe ID column and statistics
    if ("Gene" %in% colnames(de)) {
      gene_col <- "Gene"
    } else if ("Probe" %in% colnames(de)) {
      gene_col <- "Probe"
    } else {
      next
    }
    
    # Use logFC for ranking
    if ("logFC" %in% colnames(de) && "P.Value" %in% colnames(de)) {
      # Rank by signed p-value: -log10(p) * sign(logFC)
      ranks <- -log10(de$P.Value) * sign(de$logFC)
      names(ranks) <- de[[gene_col]]
      ranks <- ranks[!is.na(ranks) & is.finite(ranks)]
      all_ranks[[ds_name]] <- ranks
      cat(ds_name, ": ", length(ranks), " ranked genes/probes\n")
    }
  }
  
  if (length(all_ranks) >= 2) {
    # Convert ranks to rank lists
    rank_lists <- lapply(all_ranks, function(x) names(sort(x, decreasing = TRUE)))
    
    # Run RRA
    rra_result <- aggregateRanks(rank_lists, N = NA)
    rra_result$Gene <- rra_result$Name
    rra_result <- rra_result[order(rra_result$Score), ]
    
    write.csv(rra_result, file.path(result_dir, "RRA_integrated_DE_results.csv"), row.names = FALSE)
    cat("RRA results: ", nrow(rra_result), " genes\n")
    cat("Top 10 genes by RRA score:\n")
    print(head(rra_result, 10))
  }
} else {
  cat("WARNING: Not enough DE results for RRA\n")
}

cat("\n========================================\n")
cat("End time:", format(Sys.time()), "\n")
cat("========================================\n")

sink()