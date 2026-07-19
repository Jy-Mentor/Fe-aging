# ============================================================================
# STEP 03: L1 GSEA + WGCNA 共表达模块
# - 铁死亡/衰老/BCP 基因集 GSEA (基于 clusterProfiler + msigdbr)
# - WGCNA signed network + 模块-时间关联
# - 识别与铁衰老相关的 hub module + hub genes
# 参考:
#   - Yu G et al. 2012 OMICS (clusterProfiler)
#   - Langfelder P & Horvath S 2008 BMC Bioinformatics (WGCNA)
# ============================================================================

step03_bulk_gsea_wgcna <- function(dds, dea_list, cfg) {
  log_info("[Step03-L1] GSEA + WGCNA...")

  if (is.null(dea_list)) stop("DEA list is NULL. Run step 02 first.")

  require_packages(c("clusterProfiler", "msigdbr"),
                   install_hint = "BiocManager::install(c('clusterProfiler')); install.packages('msigdbr')")
  suppressPackageStartupMessages({
    library(clusterProfiler)
    library(msigdbr)
  })

  gene_sets <- build_gene_sets(cfg, organism = "mouse")

  # --------------------------------------------------------------------------
  # 3.1 GSEA: 铁死亡/衰老/BCP 基因集 × 时间序列
  # --------------------------------------------------------------------------
  log_info("[Step03] GSEA on custom ferroptosis/senescence/BCP gene sets...")

  # 构造 TERM2GENE 表
  t2g <- rbind(
    data.frame(term = "Ferroptosis",     gene = gene_sets$ferroptosis),
    data.frame(term = "Senescence",      gene = gene_sets$senescence),
    data.frame(term = "Ferroaging",      gene = gene_sets$ferroaging),
    data.frame(term = "BCP_Up",          gene = gene_sets$bcp_up),
    data.frame(term = "BCP_Down",        gene = gene_sets$bcp_down),
    data.frame(term = "Ferrosenescence", gene = gene_sets$ferrosenescence)
  )

  gsea_results_all <- list()
  nes_summary <- data.frame()

  for (cmp in names(dea_list)) {
    log_info("[Step03] GSEA for ", cmp)
    df <- dea_list[[cmp]]
    # 过滤 NA
    df <- df[!is.na(df$padj) & !is.na(df$log2FoldChange), ]
    gene_list <- df$log2FoldChange
    names(gene_list) <- df$gene
    gene_list <- sort(gene_list, decreasing = TRUE)

    gsea_res <- tryCatch({
      GSEA(geneList = gene_list,
           TERM2GENE = t2g,
           pvalueCutoff = cfg$bulk$gsea_padj_cutoff,
           pAdjustMethod = cfg$enrichment$p_adjust_method,
           minGSSize = cfg$bulk$gsea_min_gssize,
           maxGSSize = cfg$bulk$gsea_max_gssize,
           seed = TRUE,
           verbose = FALSE)
    }, error = function(e) {
      log_warn("[Step03] GSEA failed for ", cmp, ": ", conditionMessage(e))
      NULL
    })

    if (!is.null(gsea_res) && nrow(as.data.frame(gsea_res)) > 0) {
      gsea_df <- as.data.frame(gsea_res)
      gsea_df$comparison <- cmp
      gsea_results_all[[cmp]] <- gsea_df

      # NES 摘要
      nes_summary <- rbind(nes_summary, data.frame(
        comparison = cmp,
        term = rownames(gsea_df),
        NES = gsea_df$NES,
        pvalue = gsea_df$pvalue,
        p.adjust = gsea_df$p.adjust,
        stringsAsFactors = FALSE
      ))
    }
  }

  if (length(gsea_results_all) > 0) {
    gsea_all <- do.call(rbind, gsea_results_all)
    save_table(gsea_all, "03_bulk_gsea_all_terms", cfg)
    save_table(nes_summary, "03_bulk_gsea_nes_summary", cfg)

    # NES 随时间变化折线图 (铁死亡 vs 衰老对比)
    if (nrow(nes_summary) > 0) {
      nes_summary$comparison <- factor(nes_summary$comparison,
                                       levels = cfg$data$bulk_time_levels[-1])
      key_terms <- c("Ferroptosis", "Senescence", "Ferroaging",
                     "BCP_Up", "BCP_Down")
      nes_key <- nes_summary[nes_summary$term %in% key_terms, ]

      if (nrow(nes_key) > 0) {
        p_nes <- ggplot(nes_key, aes(x = comparison, y = NES,
                                      color = term, group = term)) +
          geom_line(linewidth = 1) +
          geom_point(size = 2.5) +
          geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
          scale_color_manual(values = c("Ferroptosis" = "#B2182B",
                                        "Senescence" = "#2166AC",
                                        "Ferroaging" = "#67001F",
                                        "BCP_Up" = "#4DBBD5",
                                        "BCP_Down" = "#F39B7F")) +
          labs(title = "NES trajectory: ferroptosis vs senescence (Bulk GSEA)",
               x = "Time point vs Control", y = "Normalized Enrichment Score",
               color = "Gene set") +
          theme_pub(base_size = 11)
        save_figure(p_nes, "03_bulk_gsea_nes_trajectory", cfg, width = 10, height = 6)
      }
    }
  } else {
    log_warn("[Step03] No GSEA results. Skipping NES plot.")
  }

  # --------------------------------------------------------------------------
  # 3.2 WGCNA 共表达网络
  # --------------------------------------------------------------------------
  log_info("[Step03] WGCNA co-expression network...")

  require_packages(c("WGCNA"),
                   install_hint = "install.packages('WGCNA')")
  suppressPackageStartupMessages({
    library(WGCNA)
  })

  # 取 VST 后的 top 变异基因
  vsd_path <- file.path(cfg$project$rds_dir, "02_bulk_vsd.rds")
  if (!file.exists(vsd_path)) {
    log_warn("[Step03] VSD RDS not found at ", vsd_path,
             ". Skipping WGCNA.")
    return(invisible(list(gsea = gsea_results_all, wgcna = NULL)))
  }
  vsd <- readRDS(vsd_path)
  expr_mat <- assay(vsd)

  # top 变异基因
  n_top <- cfg$bulk$wgcna_n_top_var_genes
  gene_vars <- rowVars(expr_mat)
  top_genes <- order(gene_vars, decreasing = TRUE)[seq_len(min(n_top, length(gene_vars)))]
  datExpr <- t(expr_mat[top_genes, ])
  log_info("[Step03] WGCNA input: ", nrow(datExpr), " samples x ",
           ncol(datExpr), " genes")

  # 软阈值选择
  powers <- 1:20
  sft <- pickSoftThreshold(datExpr, powerVector = powers,
                            networkType = cfg$bulk$wgcna_network_type,
                            RsquaredCut = cfg$bulk$wgcna_soft_threshold_r2,
                            verbose = 0)
  soft_power <- sft$powerEstimate
  log_info("[Step03] WGCNA soft threshold power = ", soft_power)

  if (is.na(soft_power)) {
    log_warn("[Step03] No soft threshold reached R²=", cfg$bulk$wgcna_soft_threshold_r2,
             "; falling back to power=12 (signed network typical)")
    soft_power <- 12
  }

  # 软阈值可视化
  p_sft <- ggplot(data.frame(power = powers,
                              R2 = -sign(sft$fitIndices[, 3]) * sft$fitIndices[, 2]),
                  aes(x = power, y = R2)) +
    geom_point(size = 2) +
    geom_hline(yintercept = cfg$bulk$wgcna_soft_threshold_r2,
               linetype = "dashed", color = "red") +
    labs(title = "WGCNA soft threshold selection",
         x = "Soft power", y = "Scale-free R² (signed)") +
    theme_pub()
  save_figure(p_sft, "03_bulk_wgcna_soft_threshold", cfg, width = 7, height = 5)

  # 构建网络
  net <- blockwiseModules(
    datExpr,
    power = soft_power,
    networkType = cfg$bulk$wgcna_network_type,
    TOMType = cfg$bulk$wgcna_network_type,
    corType = cfg$bulk$wgcna_cor_type,
    maxPOutliers = cfg$bulk$wgcna_max_p_outliers,
    maxBlockSize = cfg$bulk$wgcna_max_block_size,
    minModuleSize = cfg$bulk$wgcna_min_module_size,
    mergeCutHeight = cfg$bulk$wgcna_merge_cut_height,
    numericLabels = TRUE,
    saveTOMs = FALSE,
    randomSeed = cfg$reproducibility$r_seed,
    verbose = 0
  )

  module_colors <- labels2colors(net$colors)
  n_modules <- length(unique(net$colors))
  log_info("[Step03] WGCNA identified ", n_modules, " modules")

  # 模块-性状关联 (与时间点的相关性)
  time_numeric <- as.numeric(factor(colData(vsd)[[cfg$data$bulk_time_col]]))
  MEs <- net$MEs
  moduleTraitCor <- cor(MEs, time_numeric, use = "p")
  moduleTraitPvalue <- corPvalueStudent(moduleTraitCor, nrow(datExpr))

  module_trait_df <- data.frame(
    module = colnames(MEs),
    color = labels2colors(as.numeric(gsub("ME", "", colnames(MEs)))),
    cor_time = as.numeric(moduleTraitCor),
    p_value = as.numeric(moduleTraitPvalue),
    padj = p.adjust(as.numeric(moduleTraitPvalue), method = "BH")
  )
  module_trait_df <- module_trait_df[order(-abs(module_trait_df$cor_time)), ]
  save_table(module_trait_df, "03_bulk_wgcna_module_trait", cfg)

  # 识别最相关模块的 hub 基因与铁衰老基因交集
  top_module <- module_trait_df$module[1]
  module_genes <- colnames(datExpr)[net$colors == as.numeric(gsub("ME", "", top_module))]
  hub_fa_overlap <- intersect(module_genes, gene_sets$ferrosenescence)
  log_info("[Step03] Top module '", top_module, "' (color ",
           module_trait_df$color[1], "): ", length(module_genes), " genes, ",
           length(hub_fa_overlap), " overlap with ferroaging")

  # 模块-时间关联热图
  png(file.path(cfg$project$figures_dir, "03_bulk_wgcna_module_trait_heatmap.png"),
      width = 8, height = max(6, n_modules * 0.3), units = "in", res = 300)
  tryCatch({
    par(mar = c(6, 8, 3, 3))
    labeledHeatmap(
      Matrix = moduleTraitCor,
      xLabels = "Time (numeric)",
      yLabels = paste("ME", module_trait_df$color),
      ySymbols = colnames(MEs),
      colorLabels = FALSE,
      colors = blueWhiteRed(50),
      textMatrix = paste0("r=", signif(module_trait_df$cor_time, 2), "\n",
                          "p=", signif(module_trait_df$p_value, 2)),
      setStdMargins = FALSE,
      cex.text = 0.7,
      main = "Module-Trait correlation"
    )
  }, error = function(e) {
    log_warn("[Step03] module-trait heatmap failed: ", conditionMessage(e))
  }, finally = {
    try(dev.off(), silent = TRUE)
  })

  # 保存 WGCNA 结果
  wgcna_result <- list(
    datExpr = datExpr,
    net = net,
    module_colors = module_colors,
    MEs = MEs,
    moduleTraitCor = moduleTraitCor,
    moduleTraitPvalue = moduleTraitPvalue,
    module_trait_df = module_trait_df,
    top_module = top_module,
    top_module_genes = module_genes,
    hub_fa_overlap = hub_fa_overlap,
    soft_power = soft_power
  )
  save_rds(wgcna_result, "03_bulk_wgcna_result", cfg)

  log_info("[Step03] GSEA + WGCNA done.")
  invisible(list(gsea = gsea_results_all, wgcna = wgcna_result))
}
