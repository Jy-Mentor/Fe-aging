# ============================================================================
# STEP 5: CellChat 细胞通讯分析
# - 对每个条件 (Ctrl / 1DPI / 3DPI / 7DPI) 独立构建 CellChat 对象
# - 识别显著 L-R 对，重点关注铁衰老相关通路
# - 跨条件比较通讯强度变化
# 参考: Jin et al. 2021 Nature Communications (CellChat v2)
# ============================================================================

step05_cellchat_analysis <- function(seu, cfg) {
  log_info("[Step5] CellChat cell-cell communication analysis...")

  if (!requireNamespace("CellChat", quietly = TRUE)) {
    stop("CellChat package not installed. Install via:",
         " remotes::install_github('jinworks/CellChat')")
  }
  suppressPackageStartupMessages({
    library(CellChat)
    library(patchwork)
    library(NMF)
    library(ComplexHeatmap)
  })

  conditions <- cfg$analysis$condition_levels
  celltype_col <- cfg$analysis$celltype_col
  min_cells <- cfg$cellchat$min_cells_per_type

  cellchat_list <- list()
  expr_layers <- list()

  for (cond in conditions) {
    log_info("[Step5] Building CellChat for condition: {cond}")
    # 直接通过列名索引，避免 Seurat v5 subset 求值问题
    cells_cond <- colnames(seu)[seu@meta.data[[cfg$analysis$condition_col]] == cond]
    log_info("[Step5] {cond}: found {length(cells_cond)} cells")
    if (length(cells_cond) < 10) {
      log_warn("[Step5] {cond}: too few cells ({length(cells_cond)}), skip")
      next
    }
    seu_cond <- seu[, cells_cond]

    # 过滤过少细胞类型
    ct_counts <- table(seu_cond@meta.data[[celltype_col]])
    keep_ct <- names(ct_counts)[ct_counts >= min_cells]
    if (length(keep_ct) < 2) {
      log_warn("[Step5] {cond}: <2 cell types with >= {min_cells} cells, skip")
      next
    }
    cells_keep <- colnames(seu_cond)[seu_cond@meta.data[[celltype_col]] %in% keep_ct]
    seu_cond <- seu_cond[, cells_keep]
    log_info("[Step5] {cond}: {ncol(seu_cond)} cells, {length(keep_ct)} types")

    # 关键: 清理未使用的因子水平, 否则 CellChat identifyOverExpressedGenes 会报错
    seu_cond@meta.data[[celltype_col]] <- factor(seu_cond@meta.data[[celltype_col]],
                                                  levels = keep_ct)
    seu_cond@meta.data[[celltype_col]] <- droplevels(seu_cond@meta.data[[celltype_col]])
    Seurat::Idents(seu_cond) <- celltype_col

    # 使用 SCT 标准化数据 (若存在)，否则用 RNA data
    if ("SCT" %in% SeuratObject::Assays(seu_cond)) {
      data_use <- Seurat::GetAssayData(seu_cond, assay = "SCT", layer = "data")
      meta_use <- seu_cond@meta.data
    } else {
      data_use <- Seurat::GetAssayData(seu_cond, assay = "RNA", layer = "data")
      meta_use <- seu_cond@meta.data
    }

    cellchat <- CellChat::createCellChat(
      object = data_use,
      meta = meta_use,
      group.by = celltype_col
    )

    # 数据库: CellChatDB.mouse (Secretome DB)
    CellChatDB <- CellChat::CellChatDB.mouse
    cellchat@DB <- CellChatDB

    cellchat <- CellChat::subsetData(cellchat)
    cellchat <- CellChat::identifyOverExpressedGenes(cellchat, do.fast = FALSE)
    cellchat <- CellChat::identifyOverExpressedInteractions(cellchat)
    cellchat <- CellChat::computeCommunProb(cellchat,
                                            population.size = cfg$cellchat$population_size,
                                            nboot = cfg$cellchat$nboot)
    cellchat <- CellChat::filterCommunication(cellchat, min.cells = min_cells)
    cellchat <- CellChat::computeCommunProbPathway(cellchat)
    cellchat <- CellChat::aggregateNet(cellchat)

    cellchat_list[[cond]] <- cellchat
    rm(seu_cond, data_use, meta_use); gc(verbose = FALSE)
    log_info("[Step5] {cond}: {nrow(cellchat@net$count)}x{ncol(cellchat@net$count)} net")
  }

  if (length(cellchat_list) < 2) {
    log_warn("[Step5] Less than 2 conditions processed. Skip comparison.")
    saveRDS(cellchat_list, file.path(cfg$project$rds_dir, "cellchat_list.rds"))
    return(invisible(cellchat_list))
  }

  # 5.1 单条件可视化
  for (cond in names(cellchat_list)) {
    cc <- cellchat_list[[cond]]
    weight_mat <- cc@net$weight
    count_mat <- cc@net$count

    png(file.path(cfg$project$figures_dir,
                  sprintf("05_cellchat_network_%s.png", cond)),
        width = 10, height = 9, units = "in", res = 300)
    print(CellChat::netVisual_circle(weight_mat,
                                     weight.scale = TRUE,
                                     title.name = paste0(cond, " - Interaction strength")))
    dev.off()

    # 5.2 通路富集热图 (top 20)
    tryCatch({
      png(file.path(cfg$project$figures_dir,
                    sprintf("05_cellchat_pathway_heatmap_%s.png", cond)),
          width = 12, height = 8, units = "in", res = 300)
      h <- CellChat::netVisual_heatmap(cc, measure = "weight",
                                       title.name = paste0(cond, " - Pathway signaling"))
      draw(h)
      dev.off()
    }, error = function(e) {
      log_warn("[Step5] Heatmap failed for {cond}: {conditionMessage(e)}")
      if (exists("dev.list")) try(dev.off(), silent = TRUE)
    })

    # 5.3 通路整体强度条形图
    tryCatch({
      net_signaling <- CellChat::netAnalysis_signalingRole_scatter(cc)
      if (!is.null(net_signaling) && nrow(net_signaling$signalingContribution) > 0) {
        sig_df <- net_signaling$signalingContribution
      }
    }, error = function(e) {
      log_warn("[Step5] signalingRole_scatter failed for {cond}: {conditionMessage(e)}")
    })
  }

  # 5.4 跨条件比较
  log_info("[Step5] Comparing conditions across CellChat objects...")
  cellchat_merged <- CellChat::mergeCellChat(cellchat_list)

  tryCatch({
    png(file.path(cfg$project$figures_dir,
                  "05_cellchat_compare_count_total.png"),
        width = 14, height = 5, units = "in", res = 300)
    p1 <- CellChat::compareInteractions(cellchat_merged, show.legend = FALSE,
                                        group = names(cellchat_list))
    p2 <- CellChat::compareInteractions(cellchat_merged, show.legend = FALSE,
                                        group = names(cellchat_list),
                                        measure = "weight")
    print(p1 + p2)
    dev.off()
  }, error = function(e) {
    log_warn("[Step5] compareInteractions failed: {conditionMessage(e)}")
    try(dev.off(), silent = TRUE)
  })

  tryCatch({
    png(file.path(cfg$project$figures_dir,
                  "05_cellchat_compare_circle.png"),
        width = 16, height = 5, units = "in", res = 300)
    par(mfrow = c(1, length(cellchat_list)))
    for (cond in names(cellchat_list)) {
      CellChat::netVisual_diffInteraction(
        cellchat_merged, weight.scale = TRUE,
        comparison = c(1, which(names(cellchat_list) == cond)),
        title.name = paste0(cond, " vs Ctrl")
      )
    }
    dev.off()
  }, error = function(e) {
    log_warn("[Step5] netVisual_diffInteraction failed: {conditionMessage(e)}")
    try(dev.off(), silent = TRUE)
  })

  # 5.5 铁衰老相关通路 (TNF/IL1/IL6/TGFB/chemokine) 比较
  ferroaging_pathways <- c("TNF", "IL1", "IL6", "CCL", "CXCL", "TGFb",
                           "SPP1", "MIF", "VISFATIN", "COMPLEMENT",
                           "GALECTIN", "BTLA", "FASLG", "TRAIL")
  fa_pathways_avail <- intersect(ferroaging_pathways,
                                 unique(unlist(lapply(cellchat_list, function(cc) {
                                   cc@netP$pathways
                                 }))))
  log_info("[Step5] Ferroaging-related pathways available: {paste(fa_pathways_avail, collapse=', ')}")

  if (length(fa_pathways_avail) > 0) {
    for (pw in fa_pathways_avail) {
      tryCatch({
        conds_with_pw <- names(cellchat_list)[sapply(cellchat_list, function(cc) {
          pw %in% cc@netP$pathways
        })]
        if (length(conds_with_pw) < 2) next

        png(file.path(cfg$project$figures_dir,
                      sprintf("05_cellchat_pathway_%s.png", pw)),
            width = 4 * length(conds_with_pw), height = 4,
            units = "in", res = 300)
        par(mfrow = c(1, length(conds_with_pw)), xpd = TRUE)
        for (cond in conds_with_pw) {
          cc <- cellchat_list[[cond]]
          if (pw %in% cc@netP$pathways) {
            CellChat::netVisual_aggregate(cc, signaling = pw,
                                          layout = "circle",
                                          edge.weight.max = NULL,
                                          signaling.name = paste0(pw, " - ", cond))
          }
        }
        dev.off()
      }, error = function(e) {
        log_warn("[Step5] Pathway {pw} viz failed: {conditionMessage(e)}")
        try(dev.off(), silent = TRUE)
      })
    }

    # 通路总体比较信息流
    tryCatch({
      png(file.path(cfg$project$figures_dir,
                    "05_cellchat_ferroaging_information_flow.png"),
          width = 12, height = 7, units = "in", res = 300)
      p_info <- CellChat::compareInteractions(
        cellchat_merged, measure = "count",
        group = names(cellchat_list)
      )
      print(p_info)
      dev.off()
    }, error = function(e) {
      log_warn("[Step5] Information flow plot failed: {conditionMessage(e)}")
      try(dev.off(), silent = TRUE)
    })
  }

  saveRDS(cellchat_list, file.path(cfg$project$rds_dir, "cellchat_list.rds"))
  saveRDS(cellchat_merged, file.path(cfg$project$rds_dir, "cellchat_merged.rds"))
  log_info("[Step5] CellChat analysis done. Conditions: {paste(names(cellchat_list), collapse=', ')}")

  invisible(cellchat_list)
}

cellchat_list <- step05_cellchat_analysis(seu, cfg)
