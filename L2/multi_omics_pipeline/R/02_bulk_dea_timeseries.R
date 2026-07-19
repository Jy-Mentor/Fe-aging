# ============================================================================
# STEP 02: L1 Bulk DESeq2 差异表达 + LRT 时序整体效应
# - 各时间点 vs Control 的 Wald 检验 (12h/D1/D3/D7)
# - LRT (Likelihood Ratio Test) 检验时间整体效应
# - LFC shrinkage (apeglm 首选, ashr 多组备选)
# - VST 方差稳定变换用于下游 (WGCNA, 可视化)
# 参考:
#   - Love MI et al. 2014 Genome Biology (DESeq2)
#   - Zhu A et al. 2019 Bioinformatics (apeglm LFC shrinkage)
# ============================================================================

step02_bulk_dea_timeseries <- function(dds, cfg) {
  log_info("[Step02-L1] DESeq2 DEA + LRT time-series...")

  if (is.null(dds)) stop("DESeq2 object 'dds' is NULL. Run step 01 first.")
  require_packages(c("DESeq2"))
  suppressPackageStartupMessages(library(DESeq2))

  # --------------------------------------------------------------------------
  # 2.1 运行 DESeq() 主流程
  # --------------------------------------------------------------------------
  log_info("[Step02] Running DESeq()...")
  dds <- DESeq(dds)
  log_info("[Step02] DESeq() complete. Results names: ",
           paste(resultsNames(dds), collapse = ", "))

  # --------------------------------------------------------------------------
  # 2.2 各时间点 vs Control Wald 检验
  # --------------------------------------------------------------------------
  time_col <- cfg$data$bulk_time_col
  time_levels <- cfg$data$bulk_time_levels
  ref_level <- time_levels[1]   # "Control"
  comparisons <- setdiff(time_levels, ref_level)

  dea_list <- list()
  for (cmp in comparisons) {
    log_info("[Step02] Contrast: ", cmp, " vs ", ref_level)
    res <- results(dds, contrast = c(time_col, cmp, ref_level),
                   alpha = cfg$bulk$padj_threshold)

    # LFC shrinkage (apeglm 首选; 多组比较用 ashr)
    shrink_type <- cfg$bulk$lfc_shrink_type
    res_shrunk <- tryCatch({
      if (shrink_type == "apeglm") {
        # apeglm 需要 coef 参数; 通过 resultsNames 查找
        coef_name <- paste0(time_col, "_", cmp, "_vs_", ref_level)
        if (coef_name %in% resultsNames(dds)) {
          lfcShrink(dds, coef = coef_name, type = "apeglm")
        } else {
          log_warn("[Step02] coef '", coef_name, "' not in resultsNames; using ashr")
          lfcShrink(dds, contrast = c(time_col, cmp, ref_level), type = "ashr")
        }
      } else if (shrink_type == "ashr") {
        lfcShrink(dds, contrast = c(time_col, cmp, ref_level), type = "ashr")
      } else {
        log_warn("[Step02] Unknown shrink_type: ", shrink_type, "; skipping shrinkage")
        res
      }
    }, error = function(e) {
      log_warn("[Step02] LFC shrink failed for ", cmp, ": ", conditionMessage(e),
               "; using unshrunk results")
      res
    })

    res_df <- as.data.frame(res_shrunk)
    res_df$gene <- rownames(res_df)
    res_df$comparison <- paste0(cmp, "_vs_", ref_level)

    # 显著标记
    lfc_thr <- cfg$bulk$lfc_threshold
    padj_thr <- cfg$bulk$padj_threshold
    res_df$significant <- ifelse(!is.na(res_df$padj) &
                                   res_df$padj < padj_thr &
                                   abs(res_df$log2FoldChange) >= lfc_thr,
                                 "yes", "no")
    res_df$direction <- ifelse(res_df$log2FoldChange > 0, "up", "down")

    n_sig <- sum(res_df$significant == "yes", na.rm = TRUE)
    log_info("[Step02] ", cmp, " vs ", ref_level, ": ", n_sig, " significant DEGs")

    dea_list[[cmp]] <- res_df
  }

  # 合并所有 DE 结果
  deg_all <- do.call(rbind, dea_list)
  save_table(deg_all, "02_bulk_all_degs", cfg)

  deg_sig <- deg_all[deg_all$significant == "yes", ]
  save_table(deg_sig, "02_bulk_signif_degs", cfg)

  # --------------------------------------------------------------------------
  # 2.3 LRT (Likelihood Ratio Test) 时序整体效应
  # --------------------------------------------------------------------------
  log_info("[Step02] LRT: testing overall time effect...")
  reduced_formula <- as.formula(cfg$bulk$lrt_reduced_formula)
  dds_lrt <- DESeq(dds, test = "LRT", reduced = reduced_formula, quiet = TRUE)
  res_lrt <- results(dds_lrt)
  res_lrt_df <- as.data.frame(res_lrt)
  res_lrt_df$gene <- rownames(res_lrt_df)
  res_lrt_df$time_significant <- ifelse(!is.na(res_lrt_df$padj) &
                                          res_lrt_df$padj < cfg$bulk$padj_threshold,
                                        "yes", "no")
  save_table(res_lrt_df, "02_bulk_lrt_time_effect", cfg)

  n_time_sig <- sum(res_lrt_df$time_significant == "yes", na.rm = TRUE)
  log_info("[Step02] LRT: ", n_time_sig, " genes with significant time effect")

  # --------------------------------------------------------------------------
  # 2.4 VST 方差稳定变换 (用于 WGCNA, 可视化)
  # --------------------------------------------------------------------------
  log_info("[Step02] VST transformation...")
  vsd <- vst(dds_lrt, blind = FALSE)
  log_info("[Step02] VST matrix: ", nrow(vsd), " genes x ", ncol(vsd), " samples")
  save_rds(vsd, "02_bulk_vsd", cfg)

  # --------------------------------------------------------------------------
  # 2.5 可视化: 火山图 + DEG 数量条形图
  # --------------------------------------------------------------------------
  # 铁衰老基因高亮的火山图 (每个比较)
  gene_sets <- build_gene_sets(cfg, organism = "mouse")
  fa_genes <- gene_sets$ferroaging

  for (cmp in names(dea_list)) {
    p <- plot_volcano(dea_list[[cmp]],
                      logfc_col = "log2FoldChange",
                      padj_col = "padj",
                      gene_col = "gene",
                      highlight_genes = fa_genes,
                      title = paste0("Volcano: ", cmp, " vs ", ref_level))
    save_figure(p, sprintf("02_bulk_volcano_%s_vs_%s", cmp, ref_level), cfg,
                width = 8, height = 7)
  }

  # DEG 数量条形图 (上调/下调)
  deg_count_df <- do.call(rbind, lapply(names(dea_list), function(cmp) {
    df <- dea_list[[cmp]]
    data.frame(
      comparison = cmp,
      direction = c("up", "down"),
      count = c(sum(df$significant == "yes" & df$direction == "up", na.rm = TRUE),
                sum(df$significant == "yes" & df$direction == "down", na.rm = TRUE))
    )
  }))
  save_table(deg_count_df, "02_bulk_deg_counts", cfg)

  p_count <- ggplot(deg_count_df, aes(x = comparison, y = count, fill = direction)) +
    geom_col(position = position_dodge(width = 0.8)) +
    scale_fill_manual(values = c("up" = "#B2182B", "down" = "#2166AC")) +
    labs(title = "DEG counts per time point",
         x = "Comparison vs Control", y = "Number of DEGs",
         fill = "Direction") +
    theme_pub(base_size = 11) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
  save_figure(p_count, "02_bulk_deg_count_barplot", cfg, width = 8, height = 6)

  # 铁衰老基因的 log2FC 随时间变化热图
  fa_in_data <- intersect(fa_genes, rownames(dea_list[[1]]))
  if (length(fa_in_data) > 0) {
    lfc_mat <- do.call(cbind, lapply(dea_list, function(df) {
      df[fa_in_data, "log2FoldChange"]
    }))
    colnames(lfc_mat) <- names(dea_list)
    rownames(lfc_mat) <- fa_in_data
    save_table(data.frame(gene = fa_in_data, as.data.frame(lfc_mat)),
               "02_bulk_ferroaging_lfc_matrix", cfg)

    png(file.path(cfg$project$figures_dir, "02_bulk_ferroaging_lfc_heatmap.png"),
        width = 8, height = max(6, length(fa_in_data) * 0.18),
        units = "in", res = 300)
    tryCatch({
      plot_heatmap(lfc_mat, title = "Ferroaging genes log2FC (vs Control)")
    }, error = function(e) {
      log_warn("[Step02] Heatmap failed: ", conditionMessage(e))
    }, finally = {
      try(dev.off(), silent = TRUE)
    })
  }

  log_info("[Step02] DEA + LRT + VST done.")

  invisible(list(dds = dds_lrt, vsd = vsd, dea_list = dea_list,
                 lrt_result = res_lrt_df))
}
