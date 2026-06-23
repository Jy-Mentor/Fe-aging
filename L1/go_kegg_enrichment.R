# L1/go_kegg_enrichment.R
suppressPackageStartupMessages({
  library(clusterProfiler)
  library(org.Hs.eg.db)
})

setwd('D:/铁衰老 绝不重蹈覆辙')
result_dir <- file.path('L1', 'results')
dir.create(result_dir, showWarnings = FALSE, recursive = TRUE)

core_lines <- readLines(file.path(result_dir, 'core_genes_final.csv'))
core_genes <- unique(toupper(trimws(sapply(core_lines[-1], function(x) strsplit(x, ',')[[1]][1]))))
cat('Core genes:', length(core_genes), '\n')

ego_bp <- enrichGO(gene = core_genes, OrgDb = org.Hs.eg.db, keyType = 'SYMBOL',
                   ont = 'BP', pAdjustMethod = 'BH', pvalueCutoff = 0.05, qvalueCutoff = 0.2)
cat('GO BP:', if(is.null(ego_bp)) 0 else nrow(ego_bp), '\n')
if (!is.null(ego_bp) && nrow(ego_bp) > 0) {
  bp_df <- as.data.frame(ego_bp)
  write.csv(bp_df, file.path(result_dir, 'go_bp_enrichment.csv'), row.names = FALSE)
  nshow <- min(10, nrow(bp_df))
  for (i in 1:nshow) {
    cat(sprintf('  %s (p.adj=%.2e)\n', bp_df[i, 'Description'], bp_df[i, 'p.adjust']))
  }
}

ego_mf <- enrichGO(gene = core_genes, OrgDb = org.Hs.eg.db, keyType = 'SYMBOL',
                   ont = 'MF', pAdjustMethod = 'BH', pvalueCutoff = 0.05, qvalueCutoff = 0.2)
cat('GO MF:', if(is.null(ego_mf)) 0 else nrow(ego_mf), '\n')
if (!is.null(ego_mf) && nrow(ego_mf) > 0) {
  write.csv(as.data.frame(ego_mf), file.path(result_dir, 'go_mf_enrichment.csv'), row.names = FALSE)
}

ego_cc <- enrichGO(gene = core_genes, OrgDb = org.Hs.eg.db, keyType = 'SYMBOL',
                   ont = 'CC', pAdjustMethod = 'BH', pvalueCutoff = 0.05, qvalueCutoff = 0.2)
cat('GO CC:', if(is.null(ego_cc)) 0 else nrow(ego_cc), '\n')
if (!is.null(ego_cc) && nrow(ego_cc) > 0) {
  write.csv(as.data.frame(ego_cc), file.path(result_dir, 'go_cc_enrichment.csv'), row.names = FALSE)
}

gene_entrez <- bitr(core_genes, fromType='SYMBOL', toType='ENTREZID', OrgDb='org.Hs.eg.db')
entrez_ids <- gene_entrez[['ENTREZID']]
cat('Trying KEGG...\n')
ekegg <- tryCatch({
  enrichKEGG(gene = entrez_ids, organism = 'hsa', pAdjustMethod = 'BH',
             pvalueCutoff = 0.05, qvalueCutoff = 0.2)
}, error = function(e) {
  cat('KEGG failed:', conditionMessage(e), '\n')
  NULL
})
cat('KEGG:', if(is.null(ekegg)) 0 else nrow(ekegg), '\n')
if (!is.null(ekegg) && nrow(ekegg) > 0) {
  write.csv(as.data.frame(ekegg), file.path(result_dir, 'kegg_enrichment.csv'), row.names = FALSE)
  kegg_df <- as.data.frame(ekegg)
  nshow <- min(10, nrow(kegg_df))
  for (i in 1:nshow) {
    cat(sprintf('  %s (p.adj=%.2e)\n', kegg_df[i, 'Description'], kegg_df[i, 'p.adjust']))
  }
}

cat(sprintf('Summary: GO BP=%d, MF=%d, CC=%d, KEGG=%d\n',
    if(is.null(ego_bp)) 0 else nrow(ego_bp),
    if(is.null(ego_mf)) 0 else nrow(ego_mf),
    if(is.null(ego_cc)) 0 else nrow(ego_cc),
    if(is.null(ekegg)) 0 else nrow(ekegg)))
cat('Done!\n')
