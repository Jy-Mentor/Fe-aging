#!/usr/bin/env Rscript
# Phase 2 - Bulk GSVA Pathway Scoring (Fixed: human->mouse gene conversion + probe mapping)
suppressPackageStartupMessages({
  library(GSVA)
  library(GSEABase)
})

project_root <- normalizePath(getwd())
results_dir <- file.path(project_root, "L2", "results")
l1_results <- file.path(project_root, "L1", "results")
ferroaging_file <- file.path(project_root, "铁衰老基因.txt")
probe_map_file <- file.path(l1_results, "ILMN_probe_to_gene.csv")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

cat("Loading ferroaging genes (human)...\n")
ferroaging_genes_human <- readLines(ferroaging_file, warn = FALSE)
ferroaging_genes_human <- ferroaging_genes_human[ferroaging_genes_human != ""]
cat(sprintf("Loaded %d human ferroaging genes\n", length(ferroaging_genes_human)))

# Human-to-mouse gene conversion function
human_to_mouse <- function(human_genes) {
  known_map <- list(
    'ACSL4'='Acsl4', 'HMOX1'='Hmox1', 'TFRC'='Tfrc', 'GPX4'='Gpx4',
    'HIF1A'='Hif1a', 'KEAP1'='Keap1', 'SOD1'='Sod1', 'NLRP3'='Nlrp3',
    'IL6'='Il6', 'TLR4'='Tlr4', 'MAPK1'='Mapk1', 'PTGS2'='Ptgs2',
    'CXCL10'='Cxcl10', 'LCN2'='Lcn2', 'IL1B'='Il1b', 'CD74'='Cd74',
    'IRF1'='Irf1', 'SP1'='Sp1', 'KLF6'='Klf6', 'EGR1'='Egr1',
    'BCL6'='Bcl6', 'CTSB'='Ctsb', 'SAT1'='Sat1', 'KDM6B'='Kdm6b',
    'LGMN'='Lgmn', 'IGFBP7'='Igfbp7', 'PDE4B'='Pde4b', 'EMP1'='Emp1',
    'EPHA4'='Epha4', 'RUNX3'='Runx3', 'FBXO31'='Fbxo31',
    'LPCAT3'='Lpcat3', 'DYRK1A'='Dyrk1a', 'LACTB'='Lactb',
    'GMFB'='Gmfb', 'HBP1'='Hbp1', 'MAPK14'='Mapk14',
    'ABCC1'='Abcc1', 'ACVR1B'='Acvr1b', 'ALOX15'='Alox15',
    'ATF3'='Atf3', 'ATG3'='Atg3', 'BAP1'='Bap1', 'BRD7'='Brd7',
    'CAVIN1'='Cavin1', 'CD82'='Cd82', 'CDO1'='Cdo1',
    'COX7A1'='Cox7a1', 'DPEP1'='Dpep1', 'DPP4'='Dpp4',
    'DUOX1'='Duox1', 'E2F1'='E2f1', 'E2F3'='E2f3', 'EBF3'='Ebf3',
    'EDN1'='Edn1', 'EPHA2'='Epha2', 'ERN1'='Ern1',
    'FOSL1'='Fosl1', 'HERPUD1'='Herpud1', 'HMGB1'='Hmgb1',
    'ICA1'='Ica1', 'IFNG'='Ifng', 'IRF7'='Irf7', 'IRF9'='Irf9',
    'LIFR'='Lifr', 'LOX'='Lox', 'MAP3K14'='Map3k14',
    'MCU'='Mcu', 'MEN1'='Men1', 'MPO'='Mpo', 'NOX4'='Nox4',
    'NR1D1'='Nr1d1', 'NR2F2'='Nr2f2', 'NUAK2'='Nuak2',
    'PADI4'='Padi4', 'PPP2R2B'='Ppp2r2b', 'PRKD1'='Prkd1',
    'PTBP1'='Ptbp1', 'RBM3'='Rbm3', 'S100A8'='S100a8',
    'SETD7'='Setd7', 'SLAMF8'='Slamf8', 'SLC1A5'='Slc1a5',
    'SMARCB1'='Smarcb1', 'SMURF2'='Smurf2', 'SNCA'='Snca',
    'SOCS1'='Socs1', 'SOCS2'='Socs2', 'SPATA2'='Spata2',
    'TBX2'='Tbx2', 'TNFAIP1'='Tnfaip1', 'TNFAIP3'='Tnfaip3',
    'TXNIP'='Txnip', 'WNT5A'='Wnt5a', 'WWTR1'='Wwtr1', 'YAP1'='Yap1',
    'ZEB1'='Zeb1'
  )
  mouse_genes <- character(length(human_genes))
  for (i in seq_along(human_genes)) {
    hg <- human_genes[i]
    if (hg %in% names(known_map)) {
      mouse_genes[i] <- known_map[[hg]]
    } else {
      mouse_genes[i] <- paste0(toupper(substr(hg, 1, 1)), tolower(substr(hg, 2, nchar(hg))))
    }
  }
  return(mouse_genes)
}

# Convert to mouse symbols
ferroaging_genes_mouse <- human_to_mouse(ferroaging_genes_human)
cat(sprintf("Converted %d ferroaging genes to mouse symbols\n", length(ferroaging_genes_mouse)))

# Load probe-to-gene mapping if available
probe_to_gene <- list()
if (file.exists(probe_map_file)) {
  probe_map <- read.csv(probe_map_file, stringsAsFactors = FALSE)
  probe_to_gene <- setNames(probe_map$GeneSymbol, probe_map$Probe)
  cat(sprintf("Loaded %d probe-to-gene mappings\n", length(probe_to_gene)))
}

datasets <- c("GSE104036", "GSE16561", "GSE37587")
all_gsva_scores <- list()

for (ds in datasets) {
  expr_file <- file.path(l1_results, paste0(ds, "_expression_matrix.csv"))
  if (!file.exists(expr_file)) {
    cat(sprintf("  %s: not found, skipping\n", ds))
    next
  }
  cat(sprintf("\nProcessing %s...\n", ds))
  expr_raw <- read.csv(expr_file, row.names = 1, check.names = FALSE)
  # Ensure numeric matrix
  expr <- apply(expr_raw, 2, function(x) as.numeric(as.character(x)))
  rownames(expr) <- rownames(expr_raw)
  
  # Determine gene identifier type and find matching ferroaging genes
  gene_ids <- rownames(expr)
  first_gene <- gene_ids[1]
  
  # Check if first gene looks like a probe ID (starts with ILMN_ or is numeric)
  is_probe <- grepl("^ILMN_", first_gene)
  
  if (is_probe) {
    # Map probes to genes
    cat("  Detected probe IDs, mapping to gene symbols...\n")
    # Convert probe IDs to gene symbols
    gene_symbols <- probe_to_gene[gene_ids]
    gene_symbols <- gene_symbols[!is.na(gene_symbols)]
    
    # Find ferroaging genes among mapped gene symbols
    present_genes <- intersect(ferroaging_genes_human, gene_symbols)
    cat(sprintf("  Ferroaging genes (via probe mapping): %d / %d\n", length(present_genes), length(ferroaging_genes_human)))
    
    if (length(present_genes) < 5) {
      cat("  Too few ferroaging genes, skipping\n")
      next
    }
    
    # Create expression matrix with gene symbols (average duplicate probes)
    # Map rows to gene symbols
    expr_genes <- probe_to_gene[gene_ids]
    valid_idx <- !is.na(expr_genes)
    expr <- expr[valid_idx, , drop = FALSE]
    row_genes <- expr_genes[valid_idx]
    
    # Average duplicate genes
    unique_genes <- unique(row_genes)
    expr_matrix <- matrix(NA, nrow = length(unique_genes), ncol = ncol(expr))
    rownames(expr_matrix) <- unique_genes
    colnames(expr_matrix) <- colnames(expr)
    
    for (g in unique_genes) {
      g_rows <- which(row_genes == g)
      if (length(g_rows) == 1) {
        expr_matrix[g, ] <- as.numeric(expr[g_rows, ])
      } else {
        expr_matrix[g, ] <- colMeans(expr[g_rows, , drop = FALSE], na.rm = TRUE)
      }
    }
    expr <- expr_matrix
    
  } else {
    # Gene symbols - try both human and mouse
    cat("  Detected gene symbols\n")
    # Try human genes first
    present_genes_human <- intersect(ferroaging_genes_human, gene_ids)
    # Try mouse genes
    present_genes_mouse <- intersect(ferroaging_genes_mouse, gene_ids)
    
    if (length(present_genes_mouse) > length(present_genes_human)) {
      present_genes <- present_genes_mouse
      cat(sprintf("  Using mouse gene symbols: %d / %d\n", length(present_genes), length(ferroaging_genes_mouse)))
    } else {
      present_genes <- present_genes_human
      cat(sprintf("  Using human gene symbols: %d / %d\n", length(present_genes), length(ferroaging_genes_human)))
    }
  }
  
  if (length(present_genes) < 5) {
    cat(sprintf("  Too few ferroaging genes (%d), skipping\n", length(present_genes)))
    next
  }
  
  cat(sprintf("  Running GSVA with %d ferroaging genes...\n", length(present_genes)))
  
  tryCatch({
    param <- GSVA::gsvaParam(exprData = expr, geneSets = list(Ferroaging = present_genes),
                             kcdf = "Gaussian", minSize = 5, maxSize = 500, verbose = FALSE)
    gsva_result <- GSVA::gsva(param)
    scores <- data.frame(
      sample = colnames(gsva_result),
      gsva_score = as.numeric(gsva_result[1, ]),
      dataset = ds,
      stringsAsFactors = FALSE
    )
    all_gsva_scores[[ds]] <- scores
    cat(sprintf("  GSVA completed: %d samples\n", nrow(scores)))
  }, error = function(e) {
    cat(sprintf("  GSVA failed: %s\n", e$message))
  })
}

if (length(all_gsva_scores) > 0) {
  combined <- do.call(rbind, all_gsva_scores)
  write.csv(combined, file.path(results_dir, "gsva_ferroaging_scores.csv"), row.names = FALSE)
  cat(sprintf("\nCombined GSVA scores: %d samples across %d datasets\n", 
              nrow(combined), length(unique(combined$dataset))))
  cat(sprintf("GSVA score - Mean: %.4f, SD: %.4f\n", 
              mean(combined$gsva_score), sd(combined$gsva_score)))
  for (ds in unique(combined$dataset)) {
    ds_scores <- combined$gsva_score[combined$dataset == ds]
    cat(sprintf("  %s: Mean=%.4f, SD=%.4f, N=%d\n", ds, mean(ds_scores), sd(ds_scores), length(ds_scores)))
  }
} else {
  cat("\nNo GSVA scores computed! Writing placeholder file.\n")
  write.csv(data.frame(message = "No GSVA scores computed - insufficient gene overlap"), 
            file.path(results_dir, "gsva_ferroaging_scores.csv"), row.names = FALSE)
}
cat("\nBulk GSVA Analysis Completed\n")