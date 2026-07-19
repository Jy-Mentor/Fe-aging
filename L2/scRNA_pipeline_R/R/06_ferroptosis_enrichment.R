# ============================================================================
# STEP 6: 铁衰老通路富集分析 (GO/KEGG)
# - 使用 clusterProfiler + org.Mm.eg.db
# - 输入: 每个细胞类型的显著 DEGs
# - 富集阈值: p.adj < 0.05
# - 额外: GSVA 在每个细胞类型上计算铁衰老通路活性
# ============================================================================

step06_ferroptosis_enrichment <- function(seu, cfg) {
  log_info("[Step6] Ferroptosis / Ferroaging pathway enrichment...")

  if (!requireNamespace("clusterProfiler", quietly = TRUE) ||
      !requireNamespace("org.Mm.eg.db", quietly = TRUE)) {
    stop("clusterProfiler / org.Mm.eg.db not installed. Run BiocManager::install().")
  }
  suppressPackageStartupMessages({
    library(clusterProfiler)
    library(org.Mm.eg.db)
    library(enrichplot)
  })

  deg_file <- file.path(cfg$project$tables_dir, "04_signif_degs.csv")
  if (!file.exists(deg_file)) {
    stop("DEG table not found: ", deg_file,
         ". Run step 4 (DE analysis) first.")
  }
  sig_degs <- read.csv(deg_file, stringsAsFactors = FALSE)

  cell_types <- unique(sig_degs$cell_type)
  log_info("[Step6] Cell types in DEG table: {length(cell_types)}")

  # 6.1 GO / KEGG 富集 - 每细胞类型 × 比较
  go_results_all <- list()
  kegg_results_all <- list()

  for (ct in cell_types) {
    for (cmp in unique(sig_degs$comparison[sig_degs$cell_type == ct])) {
      sub <- sig_degs[sig_degs$cell_type == ct & sig_degs$comparison == cmp, ]
      if (nrow(sub) < 10) next

      genes_up <- sub$gene[sub$direction == "up"]
      genes_dn <- sub$gene[sub$direction == "down"]

      ego_up <- tryCatch({
        enrichGO(gene = genes_up, OrgDb = org.Mm.eg.db,
                 keyType = "SYMBOL", ont = "BP",
                 pvalueCutoff = cfg$enrichment$pvalue_cutoff,
                 qvalueCutoff = cfg$enrichment$qvalue_cutoff,
                 pAdjustMethod = "BH")
      }, error = function(e) NULL)
      if (!is.null(ego_up) && nrow(as.data.frame(ego_up)) > 0) {
        df <- as.data.frame(ego_up)
        df$cell_type <- ct; df$comparison <- cmp; df$direction <- "up"
        go_results_all[[length(go_results_all) + 1]] <- df
      }

      ego_dn <- tryCatch({
        enrichGO(gene = genes_dn, OrgDb = org.Mm.eg.db,
                 keyType = "SYMBOL", ont = "BP",
                 pvalueCutoff = cfg$enrichment$pvalue_cutoff,
                 qvalueCutoff = cfg$enrichment$qvalue_cutoff,
                 pAdjustMethod = "BH")
      }, error = function(e) NULL)
      if (!is.null(ego_dn) && nrow(as.data.frame(ego_dn)) > 0) {
        df <- as.data.frame(ego_dn)
        df$cell_type <- ct; df$comparison <- cmp; df$direction <- "down"
        go_results_all[[length(go_results_all) + 1]] <- df
      }

      ekegg_up <- tryCatch({
        enrichKEGG(gene = bitr(genes_up, fromType = "SYMBOL",
                               toType = "ENTREZID",
                               OrgDb = org.Mm.eg.db)$ENTREZID,
                   organism = cfg$enrichment$kegg_organism,
                   pvalueCutoff = cfg$enrichment$pvalue_cutoff,
                   qvalueCutoff = cfg$enrichment$qvalue_cutoff)
      }, error = function(e) NULL)
      if (!is.null(ekegg_up) && nrow(as.data.frame(ekegg_up)) > 0) {
        df <- as.data.frame(ekegg_up)
        df$cell_type <- ct; df$comparison <- cmp; df$direction <- "up"
        kegg_results_all[[length(kegg_results_all) + 1]] <- df
      }
    }
  }

  if (length(go_results_all) > 0) {
    go_all <- do.call(rbind, go_results_all)
    save_table(go_all, "06_go_enrichment_all", cfg)
    log_info("[Step6] GO results: {nrow(go_all)} terms across {length(unique(go_all$cell_type))} cell types")

    # 6.2 Top GO terms dotplot (top cell types)
    top_go <- go_all[order(go_all$p.adjust), ]
    top_go <- top_go[!duplicated(top_go$Description), ]
    top_go <- head(top_go, 30)
    p_go <- ggplot(top_go, aes(x = cell_type, y = reorder(Description, -log10(p.adjust)),
                               color = p.adjust, size = Count)) +
      geom_point() +
      scale_color_gradient(low = "#B2182B", high = "#2166AC") +
      labs(title = "Top GO BP terms (signif DEGs)",
           x = "Cell type", y = "GO term",
           color = "p.adjust", size = "Gene count") +
      theme_pub(base_size = 9)
    save_figure(p_go, "06_go_top_dotplot", cfg, width = 13, height = 9)
  }

  if (length(kegg_results_all) > 0) {
    kegg_all <- do.call(rbind, kegg_results_all)
    save_table(kegg_all, "06_kegg_enrichment_all", cfg)
    log_info("[Step6] KEGG results: {nrow(kegg_all)} pathways")
  }

  # 6.3 铁死亡特异性富集: FerrDb 基因集 markers + drivers + suppressors
  # 基于文献 PMID: 33597951 (FerrDb) 整理的核心铁死亡基因
  ferroptosis_core_mouse <- c(
    "Gpx4", "Slc7a11", "Acsl4", "Alox15", "Tp53", "Nfe2l2", "Keap1",
    "Hmox1", "Sat1", "Slc3a2", "Ncoa4", "Fth1", "Ftl1", "Tfrc",
    "Steap3", "Bach1", "Pten", "Cd44", "Emt", "Mtor", "Hif1a",
    "Cs", "Rpl8", "Rps3", "Ireb2", "Lpcat3", "Acsl3", "Yap1",
    "Taz", "Atg5", "Atg7", "Bap1", "Ptgs2", "Chac1", "Alox5",
    "Pebp1", "Prnp", "Dpp4", "Map1lc3a", "Map1lc3b", "Gls2",
    "Slc1a5", "Cds2", "Pebp1", "Acox1", "Cpt1a", "Nrf2"
  )
  fa_genes <- load_ferroaging_genes(cfg)
  fa_mouse <- map_human_to_mouse(fa_genes)
  ferroptosis_combined <- unique(c(ferroptosis_core_mouse, fa_mouse))

  # 6.4 DEGs 与铁死亡基因集重叠 (Fisher's exact)
  all_degs <- read.csv(file.path(cfg$project$tables_dir, "04_all_degs.csv"),
                       stringsAsFactors = FALSE)
  background_genes <- unique(rownames(Seurat::GetAssayData(seu, assay = "RNA", layer = "data")))
  ferroptosis_in_bg <- intersect(ferroptosis_combined, background_genes)
  log_info("[Step6] Ferroptosis/ferroaging genes in expression matrix: {length(ferroptosis_in_bg)}/{length(ferroptosis_combined)}")

  overlap_results <- list()
  for (ct in unique(all_degs$cell_type)) {
    for (cmp in unique(all_degs$comparison[all_degs$cell_type == ct])) {
      sub <- all_degs[all_degs$cell_type == ct & all_degs$comparison == cmp, ]
      deg_set <- unique(sub$gene)
      n_deg <- length(deg_set)
      n_overlap <- length(intersect(deg_set, ferroptosis_in_bg))

      if (n_overlap == 0) next
      n_bg <- length(background_genes)
      n_fp <- length(ferroptosis_in_bg)

      fisher_mat <- matrix(c(n_overlap, n_fp - n_overlap,
                             n_deg - n_overlap,
                             n_bg - n_fp - (n_deg - n_overlap)),
                           nrow = 2)
      ft <- fisher.test(fisher_mat, alternative = "greater")
      overlap_results[[length(overlap_results) + 1]] <- data.frame(
        cell_type = ct, comparison = cmp,
        n_deg = n_deg, n_fp_bg = n_fp,
        n_overlap = n_overlap,
        expected = n_deg * n_fp / n_bg,
        fold_enrichment = (n_overlap / n_deg) / (n_fp / n_bg),
        p_value = ft$p.value
      )
    }
  }

  if (length(overlap_results) > 0) {
    overlap_df <- do.call(rbind, overlap_results)
    overlap_df$padj <- p.adjust(overlap_df$p_value, method = "BH")
    overlap_df$signif <- ifelse(overlap_df$padj < 0.05, "yes", "no")
    save_table(overlap_df, "06_ferroptosis_overlap_fisher", cfg)
    log_info("[Step6] Ferroptosis overlap: {sum(overlap_df$signif=='yes')} significant (FDR<0.05)")

    if (nrow(overlap_df) > 0) {
      overlap_df$comparison <- factor(overlap_df$comparison,
                                      levels = c("1DPI_vs_Ctrl",
                                                 "3DPI_vs_Ctrl",
                                                 "7DPI_vs_Ctrl"))
      p_overlap <- ggplot(overlap_df, aes(x = cell_type, y = fold_enrichment,
                                          fill = comparison)) +
        geom_col(position = position_dodge(width = 0.8)) +
        geom_hline(yintercept = 1, linetype = "dashed", color = "grey50") +
        scale_fill_manual(values = c("1DPI_vs_Ctrl" = "#E64B35",
                                     "3DPI_vs_Ctrl" = "#F39B7F",
                                     "7DPI_vs_Ctrl" = "#8491B4")) +
        labs(title = "Ferroptosis gene overlap (Fisher's exact)",
             x = "Cell type", y = "Fold enrichment") +
        theme_pub(base_size = 10) +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
      save_figure(p_overlap, "06_ferroptosis_overlap_barplot", cfg,
                  width = 12, height = 6)
    }
  }

  # 6.5 GSVA 铁衰老评分 - 按细胞类型 × 条件
  # 兼容 GSVA 1.48 (旧 API: gsva(expr, geneSets, method)) 与 1.50+ (新 API: GSVAParam)
  if (requireNamespace("GSVA", quietly = TRUE) &&
      requireNamespace("GSEABase", quietly = TRUE)) {
    log_info("[Step6] Computing GSVA ferroptosis/ferroaging activity per cell...")
    library(GSVA)
    library(GSEABase)
    expr_mat <- as.matrix(Seurat::GetAssayData(seu, assay = "RNA", layer = "data"))
    fa_avail <- intersect(fa_mouse, rownames(expr_mat))
    fp_avail <- intersect(ferroptosis_core_mouse, rownames(expr_mat))

    gene_sets <- list(
      Ferroaging = fa_avail,
      Ferroptosis_core = fp_avail,
      Ferroptosis_combined = unique(c(fa_avail, fp_avail))
    )

    # 抽样 2000 细胞以加速 GSVA
    set.seed(cfg$analysis$random_seed)
    n_sub <- min(ncol(expr_mat), 2000)
    expr_sub <- expr_mat[, sample(seq_len(ncol(expr_mat)), n_sub)]

    gsva_mat <- tryCatch({
      if ("GSVAParam" %in% getNamespaceExports("GSVA")) {
        gsva_params <- GSVA::GSVAParam(
          exprData = expr_sub,
          geneSets = gene_sets,
          kcdf = "Gaussian"
        )
        GSVA::gsva(gsva_params)
      } else {
        GSVA::gsva(expr_sub, gene_sets, method = "gsva", kcdf = "Gaussian",
                   parallel.sz = 1)
      }
    }, error = function(e) {
      log_warn("[Step6] GSVA failed: {conditionMessage(e)}")
      NULL
    })

    if (!is.null(gsva_mat)) {
      seu_sub_meta <- seu@meta.data[colnames(expr_sub), , drop = FALSE]
      seu_sub_meta$GSVA_Ferroaging <- gsva_mat["Ferroaging", ]
      seu_sub_meta$GSVA_Ferroptosis_core <- gsva_mat["Ferroptosis_core", ]
      seu_sub_meta$GSVA_Ferroptosis_combined <- gsva_mat["Ferroptosis_combined", ]

      gsva_long <- data.frame(
        Cell = colnames(expr_sub),
        Condition = seu_sub_meta[[cfg$analysis$condition_col]],
        CellType = seu_sub_meta[[cfg$analysis$celltype_col]],
        GSVA_Ferroaging = gsva_mat["Ferroaging", ],
        GSVA_Ferroptosis_core = gsva_mat["Ferroptosis_core", ],
        GSVA_Ferroptosis_combined = gsva_mat["Ferroptosis_combined", ],
        row.names = NULL,
        stringsAsFactors = FALSE
      )
      save_table(gsva_long, "06_gsva_scores_per_cell", cfg)
      log_info("[Step6] GSVA scores computed for {n_sub} cells (saved as table).")

      p_gsva <- ggplot(gsva_long, aes(x = CellType, y = GSVA_Ferroaging,
                                      fill = Condition)) +
        geom_boxplot(outlier.size = 0.3, outlier.alpha = 0.3) +
        scale_fill_manual(values = CONDITION_COLORS) +
        labs(title = "GSVA Ferroaging score by cell type x condition",
             x = "Cell type", y = "GSVA score") +
        theme_pub(base_size = 10) +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
      save_figure(p_gsva, "06_gsva_ferroaging_boxplot", cfg,
                  width = 13, height = 6)
    }
  } else {
    log_warn("[Step6] GSVA/GSEABase not installed; skip GSVA scoring.")
  }

  log_info("[Step6] Enrichment analysis done.")
  invisible(seu)
}

seu <- step06_ferroptosis_enrichment(seu, cfg)
