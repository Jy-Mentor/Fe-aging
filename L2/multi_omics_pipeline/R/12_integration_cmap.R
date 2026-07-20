# ============================================================================
# STEP 12: L4 CMap 反证 (Connectivity Map 反向匹配)
# - 构建 BCP (β-caryophyllene) 处理后上调/下调 signature (从真实文献)
# - 与缺血时间序列的 DE 信号进行反向上调/下调匹配
# - 逆转比例 ≥ 阈值 视为 BCP 可能通过反相方向纠正缺血扰动
# - 输出: BCP 反证得分 × 时间点 × 通路
# 参考:
#   - Lamb J et al. 2006 Science (CMap, PMID: 17008526)
#   - Subramanian A et al. 2017 Cell (LINCS L1000, PMID: 29195078)
#   - Hu J et al. 2022 Phytomedicine (BCP-NRF2-CIR, PMID: 35550220)
#   - Li Y et al. 2024 J Agric Food Chem (PMID: 39088660)
#   - Wu Y et al. 2022 IJMS (PMID: 36555694)
#   - Rathod S et al. 2025 (PMID: 40410551)
#   - Khan A et al. 2024 (PMID: 39062016)
# ============================================================================

step12_integration_cmap <- function(bulk_dea_list, cfg) {
  log_info("[Step12-L4] CMap reversal analysis for BCP...")

  require_packages(c("fgsea", "BiocParallel"),
                   install_hint = "BiocManager::install(c('fgsea', 'BiocParallel'))")
  suppressPackageStartupMessages({
    library(fgsea)
    library(BiocParallel)
    library(ggplot2)
  })

  if (is.null(bulk_dea_list)) {
    stop("step12: bulk_dea_list is NULL. Run step02 first.")
  }

  # --------------------------------------------------------------------------
  # 12.1 加载 BCP signature (来自 config, 基于真实文献)
  # --------------------------------------------------------------------------
  gene_sets <- build_gene_sets(cfg, organism = "mouse")
  bcp_up <- gene_sets$bcp_signature$up
  bcp_down <- gene_sets$bcp_signature$down

  log_info("[Step12] BCP signature: ", length(bcp_up), " up, ",
           length(bcp_down), " down (mouse)")
  log_info("[Step12] BCP up:    ", paste(bcp_up, collapse = ", "))
  log_info("[Step12] BCP down:  ", paste(bcp_down, collapse = ", "))

  # 文献来源标注
  lit_pmids <- cfg$integration$cmap$bcp_signature_source_pmids
  log_info("[Step12] BCP signature source PMIDs: ",
           paste(lit_pmids, collapse = ", "))

  # --------------------------------------------------------------------------
  # 12.2 准备每个时间点的 LFC 排序表 (用于 fgsea 与反证计算)
  # --------------------------------------------------------------------------
  time_points <- names(bulk_dea_list)
  log_info("[Step12] Time points available: ", paste(time_points, collapse = ", "))

  cmap_results <- list()

  for (tp in time_points) {
    de_res <- bulk_dea_list[[tp]]
    if (is.null(de_res)) next

    # 提取 log2FC + padj
    de_df <- if (is.data.frame(de_res)) de_res else as.data.frame(de_res)
    lfc_col <- grep("^log2FoldChange$", colnames(de_df), value = TRUE)[1]
    padj_col <- grep("^padj$|^adj.P.Val$", colnames(de_df), value = TRUE)[1]
    if (!"gene" %in% colnames(de_df)) {
      de_df$gene <- rownames(de_df)
    }
    gene_col <- "gene"

    de_df <- de_df[!is.na(de_df[[lfc_col]]) & !is.na(de_df[[padj_col]]), ]
    log_info("[Step12] ", tp, ": ", nrow(de_df), " DE genes with valid LFC/padj")

    # 显著 DE 基因 (FDR<0.05)
    sig_de <- de_df[de_df[[padj_col]] < 0.05, ]

    # --------------------------------------------------------------------------
    # 12.3 反证 1: 显著 DE 中 BCP_up 基因方向反转
    # --------------------------------------------------------------------------
    sig_up <- sig_de[sig_de[[lfc_col]] > 0, gene_col]    # 缺血上调
    sig_dn <- sig_de[sig_de[[lfc_col]] < 0, gene_col]    # 缺血下调

    # 反证逻辑: BCP 处理后上调 → 缺血时应下调 (反相纠正)
    #           BCP 处理后下调 → 缺血时应上调 (反相纠正)
    bcp_up_in_isch_dn <- intersect(bcp_up, sig_dn)  # 反相匹配
    bcp_up_in_isch_up <- intersect(bcp_up, sig_up)  # 同向 (不纠正)
    bcp_dn_in_isch_up <- intersect(bcp_down, sig_up)  # 反相匹配
    bcp_dn_in_isch_dn <- intersect(bcp_down, sig_dn)  # 同向 (不纠正)

    bcp_up_in_data <- intersect(bcp_up, de_df[[gene_col]])
    bcp_dn_in_data <- intersect(bcp_down, de_df[[gene_col]])

    n_reversed <- length(bcp_up_in_isch_dn) + length(bcp_dn_in_isch_up)
    n_total <- length(bcp_up_in_data) + length(bcp_dn_in_data)
    reversal_score <- if (n_total > 0) n_reversed / n_total else NA_real_

    log_info(sprintf("[Step12] %s: reversal_score = %.3f (%d/%d)",
                     tp, reversal_score, n_reversed, n_total))

    # --------------------------------------------------------------------------
    # 12.4 反证 2: fgsea GSEA (BCP_up 应在缺血下调富集)
    # --------------------------------------------------------------------------
    gene_list <- de_df[[lfc_col]]
    names(gene_list) <- de_df[[gene_col]]
    gene_list <- sort(gene_list, decreasing = TRUE)

    pathways <- list(
      BCP_Up = intersect(bcp_up, names(gene_list)),
      BCP_Down = intersect(bcp_down, names(gene_list))
    )

    fgsea_res <- tryCatch({
      # fgsea 1.36+: 默认使用 fgseaMultilevel (解析 p-value)
      # Windows 上 BiocParallel 雪花 worker 找不到 C++ 函数 (fgseaMultilevelCpp/calcGseaStat)
      # 强制 SerialParam 避免并行 worker 命名空间问题
      fgsea(pathways = pathways, stats = gene_list,
            minSize = 5, maxSize = 500,
            BPPARAM = BiocParallel::SerialParam())
    }, error = function(e) {
      log_warn("[Step12] fgsea failed for ", tp, ": ", conditionMessage(e))
      NULL
    })

    if (!is.null(fgsea_res)) {
      fgsea_res$comparison <- tp
      save_table(as.data.frame(fgsea_res), paste0("12_fgsea_bcp_", tp), cfg)
    }

    # --------------------------------------------------------------------------
    # 12.5 反证 3: BCP signature 与时间序列 LFC 的 Spearman 相关
    # --------------------------------------------------------------------------
    bcp_up_lfc <- de_df[de_df[[gene_col]] %in% bcp_up, ]
    bcp_dn_lfc <- de_df[de_df[[gene_col]] %in% bcp_down, ]

    cor_up <- if (nrow(bcp_up_lfc) >= 3) {
      ct <- suppressWarnings(cor.test(bcp_up_lfc[[lfc_col]],
                                       rep(1, nrow(bcp_up_lfc)),
                                       method = "spearman"))
      data.frame(comparison = tp, set = "BCP_Up_vs_Ischemia",
                 n_genes = nrow(bcp_up_lfc),
                 rho = unname(ct$estimate), p_value = ct$p.value)
    } else NULL

    cor_dn <- if (nrow(bcp_dn_lfc) >= 3) {
      ct <- suppressWarnings(cor.test(bcp_dn_lfc[[lfc_col]],
                                       rep(-1, nrow(bcp_dn_lfc)),
                                       method = "spearman"))
      data.frame(comparison = tp, set = "BCP_Down_vs_Ischemia",
                 n_genes = nrow(bcp_dn_lfc),
                 rho = unname(ct$estimate), p_value = ct$p.value)
    } else NULL

    cor_df <- do.call(rbind, list(cor_up, cor_dn))

    # --------------------------------------------------------------------------
    # 12.6 记录结果
    # --------------------------------------------------------------------------
    cmap_results[[tp]] <- list(
      comparison = tp,
      n_sig_de = nrow(sig_de),
      n_bcp_up_in_data = length(bcp_up_in_data),
      n_bcp_dn_in_data = length(bcp_dn_in_data),
      bcp_up_isch_dn = list(bcp_up_in_isch_dn),
      bcp_dn_isch_up = list(bcp_dn_in_isch_up),
      n_reversed = n_reversed,
      n_total = n_total,
      reversal_score = reversal_score,
      fgsea = fgsea_res,
      correlation = cor_df
    )

    # 详细方向表 (当所有交集为空时, 仍能正确构造空 data.frame)
    n_total_directions <- length(bcp_up_in_isch_dn) + length(bcp_up_in_isch_up) +
                          length(bcp_dn_in_isch_up) + length(bcp_dn_in_isch_dn)
    if (n_total_directions > 0) {
      direction_df <- data.frame(
        gene = c(bcp_up_in_isch_dn, bcp_up_in_isch_up,
                  bcp_dn_in_isch_up, bcp_dn_in_isch_dn),
        bcp_direction = c(rep("up", length(bcp_up_in_isch_dn) + length(bcp_up_in_isch_up)),
                           rep("down", length(bcp_dn_in_isch_up) + length(bcp_dn_in_isch_dn))),
        ischemia_direction = c(rep("down", length(bcp_up_in_isch_dn)),
                                rep("up", length(bcp_up_in_isch_up)),
                                rep("up", length(bcp_dn_in_isch_up)),
                                rep("down", length(bcp_dn_in_isch_dn))),
        reversal = c(rep("reversed", length(bcp_up_in_isch_dn) + length(bcp_dn_in_isch_up)),
                      rep("same_direction", length(bcp_up_in_isch_up) + length(bcp_dn_in_isch_dn))),
        comparison = tp,
        stringsAsFactors = FALSE
      )
    } else {
      # 没有交集时, 仍输出空表 (列名一致)
      direction_df <- data.frame(
        gene = character(0),
        bcp_direction = character(0),
        ischemia_direction = character(0),
        reversal = character(0),
        comparison = character(0),
        stringsAsFactors = FALSE
      )
      log_warn("[Step12] ", tp, ": no BCP signature genes overlap with sig DE; empty direction table.")
    }
    save_table(direction_df, paste0("12_bcp_reversal_detail_", tp), cfg)
  }

  # --------------------------------------------------------------------------
  # 12.7 汇总表 + 可视化
  # --------------------------------------------------------------------------
  summary_df <- do.call(rbind, lapply(cmap_results, function(x) {
    data.frame(
      comparison = x$comparison,
      n_sig_de = x$n_sig_de,
      n_reversed = x$n_reversed,
      n_total = x$n_total,
      reversal_score = x$reversal_score
    )
  }))
  save_table(summary_df, "12_cmap_reversal_summary", cfg)

  p_reversal <- ggplot(summary_df, aes(x = comparison, y = reversal_score,
                                         fill = comparison)) +
    geom_col(width = 0.7) +
    geom_hline(yintercept = cfg$integration$cmap$reversal_score_threshold,
                linetype = "dashed", color = "darkred") +
    geom_text(aes(label = sprintf("%.2f (%d/%d)",
                                    reversal_score, n_reversed, n_total)),
              vjust = -0.5, size = 3.5) +
    scale_fill_manual(values = get_condition_colors(unique(summary_df$comparison))) +
    labs(title = "BCP reversal score across ischemia time points",
         subtitle = paste("Source PMIDs:", paste(lit_pmids, collapse = ", ")),
         x = "Time point vs Control",
         y = "Reversal score (BCP-corrected / total in signature)",
         fill = "Comparison") +
    theme_pub(base_size = 11) +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 30, hjust = 1))
  save_figure(p_reversal, "12_cmap_reversal_barplot", cfg, width = 10, height = 6)

  # GSEA NES 折线图
  fgsea_all <- do.call(rbind, lapply(cmap_results, function(x) {
    if (is.null(x$fgsea)) return(NULL)
    as.data.frame(x$fgsea)
  }))
  if (!is.null(fgsea_all) && nrow(fgsea_all) > 0) {
    fgsea_all$NES <- as.numeric(fgsea_all$NES)
    fgsea_all$padj <- as.numeric(fgsea_all$padj)
    save_table(fgsea_all, "12_fgsea_bcp_all_timepoints", cfg)

    p_gsea <- ggplot(fgsea_all, aes(x = comparison, y = NES,
                                      group = pathway, color = pathway)) +
      geom_line(linewidth = 1) +
      geom_point(size = 3) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
      scale_color_manual(values = c("BCP_Up" = "#B2182B",
                                     "BCP_Down" = "#2166AC")) +
      labs(title = "BCP signature GSEA NES over ischemia time course",
           subtitle = "Negative NES = BCP_up enriched in ischemia down (reversal pattern)",
           x = "Time point", y = "NES", color = "BCP signature") +
      theme_pub(base_size = 10)
    save_figure(p_gsea, "12_cmap_fgsea_nes_trajectory", cfg,
                width = 9, height = 6)
  }

  # 相关性汇总
  cor_all <- do.call(rbind, lapply(cmap_results, function(x) x$correlation))
  if (!is.null(cor_all) && nrow(cor_all) > 0) {
    cor_all$padj <- p.adjust(cor_all$p_value, method = "BH")
    save_table(cor_all, "12_bcp_ischemia_correlation", cfg)
  }

  log_info("[Step12] CMap reversal analysis done. ",
           length(cmap_results), " time points processed.")
  invisible(list(
    summary = summary_df,
    fgsea = fgsea_all,
    correlation = cor_all,
    per_timepoint = cmap_results
  ))
}
