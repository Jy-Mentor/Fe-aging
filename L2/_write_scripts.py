#!/usr/bin/env python3
"""Helper script to write P2 R files."""

import os

def write_bulk_gsva():
    content = """#!/usr/bin/env Rscript
# Phase 2 - Bulk GSVA Pathway Scoring
suppressPackageStartupMessages({
  library(GSVA)
  library(GSEABase)
})

project_root <- normalizePath(getwd())
results_dir <- file.path(project_root, "L2", "results")
l1_results <- file.path(project_root, "L1", "results")
ferroaging_file <- file.path(project_root, "铁衰老基因.txt")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

cat("Loading ferroaging genes...\n")
ferroaging_genes <- readLines(ferroaging_file, warn = FALSE)
ferroaging_genes <- ferroaging_genes[ferroaging_genes != ""]
cat(sprintf("Loaded %d ferroaging genes\n", length(ferroaging_genes)))

datasets <- c("GSE104036", "GSE16561", "GSE37587", "GSE61616", "GSE97537")
all_gsva_scores <- list()

for (ds in datasets) {
  expr_file <- file.path(l1_results, paste0(ds, "_expression_matrix.csv"))
  if (!file.exists(expr_file)) {
    cat(sprintf("  %s: not found, skipping\n", ds))
    next
  }
  cat(sprintf("\nProcessing %s...\n", ds))
  expr <- as.matrix(read.csv(expr_file, row.names = 1, check.names = FALSE))
  present_genes <- intersect(ferroaging_genes, rownames(expr))
  cat(sprintf("  Ferroaging genes: %d / %d\n", length(present_genes), length(ferroaging_genes)))
  if (length(present_genes) < 5) next
  gene_set <- GSEABase::GeneSetCollection(list(GSEABase::GeneSet(present_genes, setName = "Ferroaging")))
  tryCatch({
    gsva_result <- GSVA::gsva(expr, gene_set, method = "gsva", kcdf = "Gaussian",
                               min.sz = 5, max.sz = 500, verbose = FALSE)
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
  cat(sprintf("\nCombined GSVA: %d samples\n", nrow(combined)))
  cat(sprintf("Mean: %.4f, SD: %.4f\n", mean(combined$gsva_score), sd(combined$gsva_score)))
} else {
  cat("\nNo GSVA scores computed!\n")
}
cat("\nBulk GSVA Analysis Completed\n")
"""
    with open(os.path.join('L2', 'bulk_gsva.R'), 'w', encoding='utf-8') as f:
        f.write(content)
    print("bulk_gsva.R written")


def write_immune_infiltration():
    content = """#!/usr/bin/env Rscript
# Phase 2 - Immune Infiltration Analysis
suppressPackageStartupMessages({
  library(immunedeconv)
})

project_root <- normalizePath(getwd())
results_dir <- file.path(project_root, "L2", "results")
l1_results <- file.path(project_root, "L1", "results")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

cat("Phase 2 - Immune Infiltration Analysis\n")

datasets <- c("GSE104036", "GSE16561", "GSE37587", "GSE61616", "GSE97537")
all_immune <- list()

for (ds in datasets) {
  expr_file <- file.path(l1_results, paste0(ds, "_expression_matrix.csv"))
  if (!file.exists(expr_file)) {
    cat(sprintf("  %s: not found, skipping\n", ds))
    next
  }
  cat(sprintf("\nProcessing %s...\n", ds))
  expr <- as.matrix(read.csv(expr_file, row.names = 1, check.names = FALSE))

  tryCatch({
    immune_result <- immunedeconv::deconvolute(expr, method = "quantiseq")
    immune_result$dataset <- ds
    all_immune[[ds]] <- immune_result
    cat(sprintf("  Immune deconvolution completed: %d cell types\n", nrow(immune_result)))
  }, error = function(e) {
    cat(sprintf("  Immune deconvolution failed for %s: %s\n", ds, e$message))
    cat(sprintf("  Trying MCP-counter...\n"))
    tryCatch({
      immune_result <- immunedeconv::deconvolute(expr, method = "mcp_counter")
      immune_result$dataset <- ds
      all_immune[[ds]] <- immune_result
      cat(sprintf("  MCP-counter completed: %d cell types\n", nrow(immune_result)))
    }, error = function(e2) {
      cat(sprintf("  MCP-counter also failed: %s\n", e2$message))
    })
  })
}

if (length(all_immune) > 0) {
  combined <- do.call(rbind, all_immune)
  write.csv(combined, file.path(results_dir, "immune_infiltration.csv"), row.names = FALSE)
  cat(sprintf("\nCombined immune infiltration: %d rows\n", nrow(combined)))
} else {
  cat("\nNo immune infiltration results computed!\n")
  cat("Writing placeholder file...\n")
  write.csv(data.frame(message = "Immune infiltration not available"), 
            file.path(results_dir, "immune_infiltration.csv"), row.names = FALSE)
}
cat("\nImmune Infiltration Analysis Completed\n")
"""
    with open(os.path.join('L2', 'immune_infiltration.R'), 'w', encoding='utf-8') as f:
        f.write(content)
    print("immune_infiltration.R written")


def write_integration():
    content = """#!/usr/bin/env python3
# Phase 2 - Multi-omics Integration for Posterior Explanation
import os, sys, logging, traceback, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'integrate_explanation.log')
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'L2', 'results')
L1_RESULTS = os.path.join(PROJECT_ROOT, 'L1', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    logger.info("=" * 60)
    logger.info("Phase 2: Multi-omics Integration for Posterior Explanation")
    logger.info("=" * 60)

    # Load core genes
    core_genes = pd.read_csv(os.path.join(L1_RESULTS, 'core_genes_final.csv'))
    gene_list = core_genes['GeneSymbol'].tolist()
    logger.info(f"Core target genes: {len(gene_list)}")

    # Build explanation matrix
    explanation = pd.DataFrame({'gene_symbol': gene_list})
    explanation = explanation.set_index('gene_symbol')

    # 1. PPI network topology
    ppi_nodes = pd.read_csv(os.path.join(L1_RESULTS, 'ppi_network_nodes.csv'))
    if 'gene_symbol' in ppi_nodes.columns:
        ppi_nodes = ppi_nodes.set_index('gene_symbol')
    elif 'name' in ppi_nodes.columns:
        ppi_nodes = ppi_nodes.set_index('name')
    if 'degree' in ppi_nodes.columns:
        explanation['ppi_degree'] = ppi_nodes['degree'].reindex(explanation.index)
    if 'betweenness' in ppi_nodes.columns:
        explanation['ppi_betweenness'] = ppi_nodes['betweenness'].reindex(explanation.index)

    # 2. RRA differential expression
    rra = pd.read_csv(os.path.join(L1_RESULTS, 'RRA_gene_level_integrated.csv'))
    if 'GeneSymbol' in rra.columns:
        rra = rra.set_index('GeneSymbol')
    if 'MedianRank' in rra.columns:
        explanation['rra_median_rank'] = rra['MedianRank'].reindex(explanation.index)
    if 'Direction' in rra.columns:
        explanation['rra_direction'] = rra['Direction'].reindex(explanation.index)

    # 3. Single-cell ferroaging scores
    sc_scores_file = os.path.join(RESULTS_DIR, 'sc_ferroaging_scores.csv')
    if os.path.exists(sc_scores_file):
        sc_scores = pd.read_csv(sc_scores_file)
        logger.info(f"Single-cell ferroaging scores loaded: {len(sc_scores)} cell types")

    # 4. Pseudobulk DE for core genes
    sc_de_file = os.path.join(RESULTS_DIR, 'sc_pseudobulk_de_core_genes.csv')
    if os.path.exists(sc_de_file):
        sc_de = pd.read_csv(sc_de_file)
        for gene in gene_list:
            gene_de = sc_de[sc_de['gene'] == gene]
            if len(gene_de) > 0:
                top_ct = gene_de.loc[gene_de['padj'].idxmin()]
                explanation.loc[gene, 'sc_top_cell_type'] = top_ct['cell_type']
                explanation.loc[gene, 'sc_log2FC'] = top_ct['log2FC']
                explanation.loc[gene, 'sc_padj'] = top_ct['padj']
        logger.info(f"Single-cell DE merged for core genes")

    # 5. GSVA scores
    gsva_file = os.path.join(RESULTS_DIR, 'gsva_ferroaging_scores.csv')
    if os.path.exists(gsva_file):
        gsva = pd.read_csv(gsva_file)
        logger.info(f"GSVA scores loaded: {len(gsva)} samples across {gsva['dataset'].nunique()} datasets")

    # 6. Immune infiltration
    immune_file = os.path.join(RESULTS_DIR, 'immune_infiltration.csv')
    if os.path.exists(immune_file):
        immune = pd.read_csv(immune_file)
        logger.info(f"Immune infiltration loaded: {len(immune)} rows")

    # 7. WGCNA module info
    wgcna_file = os.path.join(L1_RESULTS, 'wgcna_GSE16561', 'gene_module_assignment.csv')
    if os.path.exists(wgcna_file):
        wgcna = pd.read_csv(wgcna_file)
        if 'gene' in wgcna.columns and 'module' in wgcna.columns:
            wgcna = wgcna.set_index('gene')
            explanation['wgcna_module'] = wgcna['module'].reindex(explanation.index)

    # Save explanation matrix
    explanation = explanation.reset_index()
    explanation.to_csv(os.path.join(RESULTS_DIR, 'explanation_features.csv'), index=False)
    logger.info(f"Explanation features saved: {len(explanation)} genes x {len(explanation.columns)} columns")
    logger.info("NOTE: These features are for POSTERIOR EXPLANATION only, NOT for prediction model input")
    logger.info("=" * 60)
    logger.info("Integration completed")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
"""
    with open(os.path.join('L2', 'integrate_explanation_features.py'), 'w', encoding='utf-8') as f:
        f.write(content)
    print("integrate_explanation_features.py written")


if __name__ == '__main__':
    write_bulk_gsva()
    write_immune_infiltration()
    write_integration()
    print("\nAll P2 R/Python files written successfully!")