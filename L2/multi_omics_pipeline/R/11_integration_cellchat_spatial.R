# ============================================================================
# STEP 11: L4 CellChat 空间细胞通讯分析
# - 用 CellChat v2 空间扩展 (create_spatial) 替代停更的 COMMOT
# - 基于 SPOTlight 投影的细胞类型比例, 推断空间 L-R 互作
# - 比较梗死核心 / 半暗带 / 健康区的通讯强度差异
# - 鉴别铁死亡/衰老相关 L-R 通路在空间上的富集
# 参考:
#   - Jin S et al. 2021 Nat Commun (CellChat, PMID: 33597522)
#   - Cang Z et al. 2023 (COMMOT 比较, 单细胞空间通讯 benchmark)
#   - CellChat v2 spatial vignette: https://htmlpreview.github.io/...
# ============================================================================

step11_integration_cellchat_spatial <- function(spatial_merged, cfg) {
  log_info("[Step11-L4] CellChat spatial cell-cell communication...")

  require_packages(c("CellChat", "patchwork"),
                   install_hint = paste("remotes::install_github('jinworks/CellChat')",
                                        "-> v2 spatial branch"))
  suppressPackageStartupMessages({
    library(CellChat)
    library(Seurat)
    library(ggplot2)
    library(patchwork)
  })

  if (is.null(spatial_merged)) {
    stop("step11: spatial_merged is NULL. Run step04-06 first.")
  }

  sp_cfg <- cfg$integration$cellchat_spatial
  if (!isTRUE(sp_cfg$enabled)) {
    log_info("[Step11] CellChat spatial disabled in config. Skip.")
    return(NULL)
  }

  # --------------------------------------------------------------------------
  # 11.1 检查 SPOTlight 投影得到的细胞类型比例
  # --------------------------------------------------------------------------
  prop_cols <- grep("^prop_", colnames(spatial_merged@meta.data), value = TRUE)
  if (length(prop_cols) == 0) {
    stop("step11: No SPOTlight proportion columns found. Run step10 first.")
  }
  cell_types <- gsub("^prop_", "", prop_cols)
  log_info("[Step11] Cell types in spatial object: ",
           paste(cell_types, collapse = ", "))

  # --------------------------------------------------------------------------
  # 11.2 按条件分别构建 CellChat 空间对象
  # --------------------------------------------------------------------------
  cellchat_list <- list()
  spatial_conds <- unique(spatial_merged$condition)

  for (cond in spatial_conds) {
    log_info("[Step11] Building spatial CellChat for condition: ", cond)
    sp_sub <- subset(spatial_merged, subset = condition == cond)
    if (ncol(sp_sub) < 50) {
      log_warn("[Step11] Too few spots for ", cond, " (n=", ncol(sp_sub),
               "); skip.")
      next
    }

    # 提取细胞类型标签 (取每 spot 主导细胞类型作为该 spot 标签)
    prop_mat <- as.matrix(sp_sub@meta.data[, prop_cols])
    spot_labels <- cell_types[max.col(prop_mat, ties.method = "first")]

    # 过滤 spot 数过少的细胞类型
    n_per_type <- table(spot_labels)
    keep_types <- names(n_per_type)[n_per_type >= sp_cfg$min_cells]
    keep_spots <- rownames(prop_mat)[spot_labels %in% keep_types]
    sp_sub <- sp_sub[, keep_spots]
    spot_labels <- spot_labels[keep_spots]
    log_info("[Step11] ", cond, ": ", length(keep_spots), " spots, ",
             length(unique(spot_labels)), " cell types")

    # 表达矩阵 (SCT 优先)
    sp_assay <- "SCT" %in% names(sp_sub@assays)
    expr_mat <- if (sp_assay) {
      as.matrix(GetAssayData(sp_sub, assay = "SCT", layer = "data"))
    } else {
      as.matrix(GetAssayData(sp_sub, assay = "Spatial", layer = "data"))
    }

    # 空间坐标
    spatial_coords <- sp_sub@images[[1]]@coordinates[, c("imagerow", "imagecol")]
    # Visium spot 之间间距换算成微米 (1 spot ≈ 100 μm center-to-center)
    image_scale <- sp_sub@images[[1]]@scale.factors$lowres
    spatial_coords_um <- spatial_coords * 100 / max(image_scale, 1e-6)

    # --------------------------------------------------------------------------
    # 11.3 create_cellchat + 空间扩展
    # --------------------------------------------------------------------------
    cellchat <- create_cellchat(object = expr_mat,
                                  meta = data.frame(labels = spot_labels,
                                                    row.names = colnames(expr_mat)),
                                  group.by = "labels")

    # 设置空间信息
    cellchat@images$spatial <- spatial_coords_um
    cellchat@.spatial.distance <- as.matrix(dist(spatial_coords_um))

    # CellChat 数据库 (小鼠)
    cellchat@DB <- CellChatDB.mouse
    cellchat <- subsetData(cellchat)

    # 空间通讯推断
    cellchat <- identifyOverExpressedGenes(cellchat)
    cellchat <- identifyOverExpressedInteractions(cellchat)

    # 空间感知的通讯概率计算
    # 参数:
    #   - distance.use: TRUE 使用空间距离衰减
    #   - interaction.range: 互作有效距离 (微米)
    #   - contact.dependent: 是否考虑接触依赖
    cellchat <- computeCommunProb(
      cellchat,
      type = sp_cfg$type,
      trim = sp_cfg$trim,
      distance.use = sp_cfg$distance_use,
      interaction.range = sp_cfg$interaction_range,
      contact.dependent = sp_cfg$contact_dependent,
      contact.range = sp_cfg$contact_range,
      population.size = sp_cfg$population_size
    )

    cellchat <- filterCommunication(
      cellchat,
      min.cells = sp_cfg$min_cells
    )

    cellchat <- computeCommunProbPathway(cellchat)
    cellchat <- aggregateNet(cellchat)

    cellchat_list[[cond]] <- cellchat
    save_rds(cellchat, paste0("11_cellchat_spatial_", cond), cfg)
  }

  if (length(cellchat_list) < 1) {
    log_error("[Step11] All conditions failed.")
    return(NULL)
  }

  # --------------------------------------------------------------------------
  # 11.4 单条件可视化
  # --------------------------------------------------------------------------
  for (cond in names(cellchat_list)) {
    cc <- cellchat_list[[cond]]
    log_info("[Step11] Visualizing: ", cond,
             " (", nrow(cc@LR$LRsig), " L-R pairs)")

    # 互作数量热图
    p_count <- netVisual_heatmap(cc, measure = "count") +
      ggtitle(paste(cond, "- interaction count"))
    save_figure(p_count, paste0("11_cellchat_heatmap_count_", cond), cfg,
                width = 8, height = 7)

    # 互作权重热图
    p_weight <- netVisual_heatmap(cc, measure = "weight") +
      ggtitle(paste(cond, "- interaction weight"))
    save_figure(p_weight, paste0("11_cellchat_heatmap_weight_", cond), cfg,
                width = 8, height = 7)

    # Circle plot (互作总数)
    tryCatch({
      png(file.path(cfg$project$figures_dir,
                     paste0("11_cellchat_circle_", cond, ".png")),
          width = 10, height = 10, units = "in", res = cfg$viz$figure_dpi)
      netVisual_circle(cc@net$count, vertex.weight = as.numeric(table(cc@idents)),
                        weight.scale = TRUE, title.edge = paste(cond, "count"))
      dev.off()
    }, error = function(e) {
      log_warn("[Step11] Circle plot failed for ", cond, ": ",
               conditionMessage(e))
    })
  }

  # --------------------------------------------------------------------------
  # 11.5 多条件比较 (至少 2 个条件)
  # --------------------------------------------------------------------------
  if (length(cellchat_list) >= 2) {
    log_info("[Step11] Comparing across ", length(cellchat_list), " conditions...")
    cc_merged <- mergeCellChat(cellchat_list, add.names = names(cellchat_list))

    save_rds(cc_merged, "11_cellchat_spatial_merged", cfg)

    # 1) 总互作数比较
    p_compare_count <- compareInteractions(cc_merged, show.legend = FALSE,
                                            group = c(1, length(cellchat_list))) +
      theme_pub(base_size = 10)
    save_figure(p_compare_count, "11_cellchat_compare_count", cfg,
                width = 8, height = 5)

    # 2) 互作强度比较
    p_compare_weight <- compareInteractions(cc_merged, show.legend = FALSE,
                                              measure = "weight",
                                              group = c(1, length(cellchat_list))) +
      theme_pub(base_size = 10)
    save_figure(p_compare_weight, "11_cellchat_compare_weight", cfg,
                width = 8, height = 5)

    # 3) 差异互作 (1D)
    p_diff <- netVisual_diffInteraction(cc_merged, weight.scale = TRUE,
                                          measure = "count") +
      ggtitle("Differential interactions (count)")
    save_figure(p_diff, "11_cellchat_diff_count", cfg, width = 9, height = 7)

    # 4) 信息流排名 (识别条件特异通路)
    p_rank <- rankNet(cc_merged, mode = "comparison",
                       stacked = TRUE, do.stat = TRUE) +
      theme_pub(base_size = 9) +
      theme(axis.text.y = element_text(size = 7))
    save_figure(p_rank, "11_cellchat_pathway_rank", cfg, width = 9, height = 14)

    # 5) 铁死亡/衰老相关通路提取
    fa_pathways <- c("SPP1", "TGFB", "CXCL", "CCL", "TNF", "IL6",
                      "GALECTIN", "MIF", "COMPLEMENT", "FLT3",
                      "GRN", "VISFATIN", "NRXN", "NCAM", "EPCAM",
                      "NOTCH", "WNT", "BMP", "FGF", "VEGF",
                      "PDGF", "EGF", "IFNII", "IL1", "IL2",
                      "IL4", "IL10", "IL12", "IL16", "IL17")

    available_pathways <- cc_merged@LR$LRsig$pathway
    available_pathways <- unique(available_pathways)
    fa_pathways_avail <- fa_pathways[fa_pathways %in% available_pathways]
    log_info("[Step11] Available pathways of interest: ",
             paste(fa_pathways_avail, collapse = ", "))

    if (length(fa_pathways_avail) > 0) {
      p_fa_heatmap <- netAnalysis_signalingRole_heatmap(
        cc_merged, pattern = "outgoing",
        signaling = fa_pathways_avail,
        width = 12, height = 8
      )
      save_figure(p_fa_heatmap, "11_cellchat_ferrosenescence_pathways_heatmap",
                  cfg, width = 12, height = 8)

      # 各通路 outgoing/incoming 得分
      for (pw in fa_pathways_avail) {
        tryCatch({
          p_pw <- netVisual_aggregate(cc_merged, signaling = pw,
                                       layout = "circle")
          save_figure(p_pw, paste0("11_cellchat_pathway_", pw), cfg,
                      width = 9, height = 7)
        }, error = function(e) {
          log_debug("[Step11] Pathway ", pw, " plot failed: ",
                    conditionMessage(e))
        })
      }
    }

    # 6) Outgoing / Incoming 通讯模式 (pathway-level)
    tryCatch({
      cc_merged <- computeNetVisual_Pairwise(cc_merged)
      cc_merged <- netAnalysis_computeCentrality(cc_merged)
      save_rds(cc_merged, "11_cellchat_spatial_merged_final", cfg)

      p_outgoing <- netAnalysis_signalingRole_scatter(
        cc_merged, slot.name = "netP", pattern = "outgoing") +
        theme_pub(base_size = 9)
      save_figure(p_outgoing, "11_cellchat_outgoing_pattern", cfg,
                  width = 9, height = 7)
    }, error = function(e) {
      log_warn("[Step11] netAnalysis compute failed: ", conditionMessage(e))
    })

    invisible(cc_merged)
  } else {
    invisible(cellchat_list[[1]])
  }
}
