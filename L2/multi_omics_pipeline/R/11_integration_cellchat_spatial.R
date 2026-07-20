# ============================================================================
# STEP 11: L4 CellChat 空间细胞通讯分析
# - 用 CellChat v2 空间扩展 (datatype = "spatial") 推断空间 L-R 互作
# - 基于 SPOTlight 投影的细胞类型比例, 推断空间 L-R 互作
# - 比较梗死核心 / 半暗带 / 健康区的通讯强度差异
# - 鉴别铁死亡/衰老相关 L-R 通路在空间上的富集
# 参考:
#   - Jin S et al. 2021 Nat Commun (CellChat v1, PMID: 33597522)
#   - CellChat v2 protocol (bioRxiv 2023.11.05.565674)
#   - 官方 vignette: tutorial/CellChat_analysis_of_spatial_transcriptomics_data.html
# API 核实 (2026-07-20):
#   - 函数名: createCellChat() (camelCase, 不是 create_cellchat)
#   - 空间信息: 创建时通过 coordinates= + spatial.factors= + datatype="spatial" 传入
#   - spatial.factors: data.frame(ratio=μm/pixel, tol=μm)
#   - 不能通过 cellchat@images$spatial 或 cellchat@.spatial.distance 直接赋值
# ============================================================================

step11_integration_cellchat_spatial <- function(spatial_merged, cfg) {
  log_info("[Step11-L4] CellChat spatial cell-cell communication...")

  require_packages(c("CellChat", "patchwork"),
                   install_hint = paste("remotes::install_github('jinworks/CellChat')",
                                        "-> v2 (master branch includes spatial)"))
  suppressPackageStartupMessages({
    library(CellChat)
    library(Seurat)
    library(ggplot2)
    library(patchwork)
  })

  if (is.null(spatial_merged)) {
    stop("step11: spatial_merged is NULL. Run step04-06 first.")
  }

  # 兼容性: 确保 spatial_merged 有 condition 列 (小写)
  if (!"condition" %in% colnames(spatial_merged@meta.data)) {
    if ("Condition" %in% colnames(spatial_merged@meta.data)) {
      spatial_merged$condition <- spatial_merged$Condition
      log_info("[Step11] Created 'condition' column from 'Condition' (case compatibility)")
    } else {
      stop("step11: spatial_merged has neither 'condition' nor 'Condition' column.")
    }
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
    # Seurat v5: 用 cells= 传递索引避免 .data[[]] 解析问题
    keep_cond <- spatial_merged@meta.data$condition == cond
    sp_sub <- subset(spatial_merged, cells = colnames(spatial_merged)[keep_cond])
    if (ncol(sp_sub) < 50) {
      log_warn("[Step11] Too few spots for ", cond, " (n=", ncol(sp_sub),
               "); skip.")
      next
    }

    # 提取细胞类型标签 (取每 spot 主导细胞类型作为该 spot 标签)
    # 关键: 必须给 spot_labels 设置 names, 否则后续用字符向量索引会全部变 NA
    prop_mat <- as.matrix(sp_sub@meta.data[, prop_cols])
    spot_labels <- cell_types[max.col(prop_mat, ties.method = "first")]
    names(spot_labels) <- rownames(prop_mat)

    # 过滤 spot 数过少的细胞类型 (用逻辑索引, 避免字符向量索引未命名向量)
    n_per_type <- table(spot_labels)
    keep_types <- names(n_per_type)[n_per_type >= sp_cfg$min_cells]
    keep_idx <- spot_labels %in% keep_types
    keep_spots <- names(spot_labels)[keep_idx]
    sp_sub <- sp_sub[, keep_spots]
    spot_labels <- spot_labels[keep_idx]
    log_info("[Step11] ", cond, ": ", length(keep_spots), " spots, ",
             length(unique(spot_labels)), " cell types; types=",
             paste(unique(spot_labels), collapse = ","))

    # 表达矩阵 (SCT 优先)
    sp_assay <- "SCT" %in% names(sp_sub@assays)
    expr_mat <- if (sp_assay) {
      as.matrix(GetAssayData(sp_sub, assay = "SCT", layer = "data"))
    } else {
      as.matrix(GetAssayData(sp_sub, assay = "Spatial", layer = "data"))
    }

    # ----------------------------------------------------------------------
    # 空间坐标 + spatial.factors 计算
    # CellChat v2 要求:
    #   coordinates: 像素单位的 data.frame (imagerow, imagecol)
    #   spatial.factors: data.frame(ratio = μm/pixel, tol = μm)
    # ----------------------------------------------------------------------
    # 用 Seurat::GetTissueCoordinates 获取坐标 (默认 lowres 像素)
    spatial_coords <- tryCatch(
      Seurat::GetTissueCoordinates(sp_sub, scale = NULL,
                                    cols = c("imagerow", "imagecol")),
      error = function(e) {
        log_warn("[Step11] GetTissueCoordinates failed: ",
                 conditionMessage(e), ". Falling back to @images[[1]]@coordinates.")
        img <- sp_sub@images[[1]]
        coords <- img@coordinates[, c("imagerow", "imagecol")]
        coords$cell <- rownames(coords)
        coords
      }
    )
    # 仅保留坐标列 (cell 名作为 rownames)
    coord_cols <- c("imagerow", "imagecol")
    spatial_coords <- spatial_coords[, coord_cols, drop = FALSE]
    # 对齐细胞名 (GetTissueCoordinates 可能含未在 expr_mat 中的 cell)
    common_cells <- intersect(rownames(spatial_coords), colnames(expr_mat))
    spatial_coords <- spatial_coords[common_cells, , drop = FALSE]
    expr_mat <- expr_mat[, common_cells, drop = FALSE]
    spot_labels <- spot_labels[common_cells]

    # 计算 spatial.factors (CellChat v2 官方 FAQ 推荐):
    #   ratio = conversion.factor = spot.size / spot_diameter_fullres
    #   tol   = spot.size / 2
    # 官方推荐 spot.size = 65 μm (10X Visium "理论 spot 大小", 含 55μm spot + 10μm gap)
    # 参考: jinworks/CellChat tutorial/FAQ_on_applying_CellChat_to_spatial_transcriptomics_data.Rmd
    #       Jin S et al. 2025 Nat Protoc PMID:39289562
    spot_size_um <- sp_cfg$spot_size_um   # 65 (config), 与 CellChat v2 官方一致
    tol_um <- spot_size_um / 2            # 32.5 μm (官方推荐)

    # 优先尝试从 spaceranger scalefactors_json.json 读取 spot_diameter_fullres
    # (CellChat v2 官方推荐方式)
    ratio_um_per_pixel <- NA_real_
    img_obj <- tryCatch(sp_sub@images[[1]], error = function(e) NULL)
    if (!is.null(img_obj)) {
      sf_json_path <- tryCatch(img_obj@scale.factors$json, error = function(e) NULL)
      # 部分旧版 Seurat Visium 对象未存 json 路径, 尝试从 BioServers/image.path 推断
      if (is.null(sf_json_path) || !file.exists(sf_json_path)) {
        # Seurat 5 Visium 对象 scale.factors 槽已含 lowres/hires/resolution
        # spot_diameter_fullres 通常未直接存储, 走回退逻辑
        ratio_um_per_pixel <- NA_real_
      } else {
        sf_data <- jsonlite::fromJSON(sf_json_path)
        if (!is.null(sf_data$spot_diameter_fullres) &&
            sf_data$spot_diameter_fullres > 0) {
          ratio_um_per_pixel <- spot_size_um / sf_data$spot_diameter_fullres
          log_info("[Step11] ", cond, ": ratio from scalefactors_json.json (",
                   "spot_diameter_fullres=", sf_data$spot_diameter_fullres, ")")
        }
      }
    }

    # 回退: 用最近邻 spot 间距估算 (Visium spot 中心间距 = 100 μm)
    if (!is.finite(ratio_um_per_pixel) || ratio_um_per_pixel <= 0) {
      if (nrow(spatial_coords) >= 2) {
        dmat <- as.matrix(dist(spatial_coords[, coord_cols, drop = FALSE]))
        diag(dmat) <- NA
        nn_dist <- apply(dmat, 1, min, na.rm = TRUE)
        median_nn_pixel <- median(nn_dist, na.rm = TRUE)
        if (!is.finite(median_nn_pixel) || median_nn_pixel <= 0) {
          log_warn("[Step11] Invalid NN distance for ", cond,
                   "; using scale.factors$lowres fallback.")
          img_sf <- tryCatch(sp_sub@images[[1]]@scale.factors$lowres,
                             error = function(e) 1)
          median_nn_pixel <- max(1 / max(img_sf, 1e-6), 1)
        }
      } else {
        median_nn_pixel <- 1
      }
      ratio_um_per_pixel <- 100 / median_nn_pixel
      log_info("[Step11] ", cond, ": ratio from NN estimate (median_nn_pixel=",
               round(median_nn_pixel, 2), ")")
    }

    spatial_factors <- data.frame(ratio = ratio_um_per_pixel, tol = tol_um)
    log_info(sprintf("[Step11] %s: %d spots, ratio=%.4f μm/pix, tol=%.1f μm",
                     cond, nrow(spatial_coords), ratio_um_per_pixel, tol_um))

    # ----------------------------------------------------------------------
    # 11.3 创建空间 CellChat 对象 (datatype = "spatial" 启用空间模式)
    # 官方 FAQ: meta 应包含 samples 列以支持跨 replicate 聚合分析
    # ----------------------------------------------------------------------
    cellchat <- createCellChat(
      object = expr_mat,
      meta = data.frame(labels = spot_labels,
                        samples = cond,
                        row.names = colnames(expr_mat)),
      group.by = "labels",
      datatype = "spatial",
      coordinates = as.matrix(spatial_coords),
      spatial.factors = spatial_factors
    )

    # CellChat 数据库 (小鼠)
    cellchat@DB <- CellChatDB.mouse
    cellchat <- subsetData(cellchat)

    # 空间通讯推断
    # do.fast=TRUE 需要 presto 包; 若未安装 presto, 显式 do.fast=FALSE 使用 base Wilcoxon
    has_presto <- requireNamespace("presto", quietly = TRUE)
    cellchat <- identifyOverExpressedGenes(cellchat,
                                           do.fast = has_presto)
    cellchat <- identifyOverExpressedInteractions(cellchat)

    # 空间感知的通讯概率计算
    cellchat <- computeCommunProb(
      cellchat,
      type = sp_cfg$type,
      trim = sp_cfg$trim,
      distance.use = sp_cfg$distance_use,
      interaction.range = sp_cfg$interaction_range,
      contact.dependent = sp_cfg$contact_dependent,
      contact.range = sp_cfg$contact_range,
      population.size = sp_cfg$population_size,
      nboot = sp_cfg$nboot
    )

    cellchat <- filterCommunication(
      cellchat,
      min.cells = sp_cfg$min_cells
    )

    cellchat <- computeCommunProbPathway(cellchat)
    cellchat <- aggregateNet(cellchat)
    # 在 single-object 上计算 centrality scores (mergeCellChat 后再调用会失败)
    # netAnalysis_signalingRole_heatmap/scatter 等均依赖 centrality
    cellchat <- netAnalysis_computeCentrality(cellchat)

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
                        weight.scale = TRUE, title.name = paste(cond, "count"))
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
    # group 参数需要与 condition 数量等长的向量 (每个 condition 一个 group label)
    # CellChat v2 compareInteractions 用 group 来分组/着色, 不接受索引向量
    cond_names <- names(cellchat_list)
    p_compare_count <- compareInteractions(cc_merged, show.legend = FALSE,
                                            group = cond_names) +
      theme_pub(base_size = 10)
    save_figure(p_compare_count, "11_cellchat_compare_count", cfg,
                width = 8, height = 5)

    # 2) 互作强度比较
    p_compare_weight <- compareInteractions(cc_merged, show.legend = FALSE,
                                              measure = "weight",
                                              group = cond_names) +
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
    # CellChat v2 mergeCellChat 后, cc@LR 是按 condition 分组的 list
    # 需要从每个 condition 的 LRsig$pathway_name 收集并集
    # 铁衰老/ferroptosis 相关 CellChat 通路列表 (基于 PubMed 文献验证 2026-07-20)
    # 通路名大小写敏感: CellChatDB.mouse 用 "TGFb" (不是 "TGFB"), "ApoE" (不是 "APOE"),
    #   "IFN-lII" (DB 中为 lowercase 'l', 不是 "IFN-III")
    # centrality scores 已在每个 condition 循环中计算 (mergeCellChat 后无法计算)
    #
    # 证据等级 (详见 PubMed 查询报告):
    # [A] 18 个强文献支持 (Jin 2021 PMID:33597522 CellChat 原文示例 + 铁死亡/SASP 文献)
    # [B] 8 个探索性通路 (CellChatDB 收录但无直接铁死亡/衰老文献证据, 论文需谨慎解读)
    # [C] 9 个文献支持补充 (TRAIL/FASLG 死亡受体-铁死亡交叉调控; IFN 干扰素-衰老;
    #     IGF/IGFBP 胰岛素-衰老经典通路; TWEAK 纤维化; ADIPONECTIN 代谢衰老)
    fa_pathways <- c(
      # [A] 强文献支持
      "SPP1", "TGFb", "CXCL", "CCL", "TNF", "IL6",
      "GALECTIN", "MIF", "COMPLEMENT", "GRN",
      "NOTCH", "WNT", "BMP", "FGF", "VEGF",
      "PDGF", "EGF", "IL1",
      # [B] 探索性通路 (CellChatDB 收录, 论文需标注为探索性)
      "FLT3", "VISFATIN", "NRXN", "NCAM",
      "IL2", "IL4", "IL10", "IL12", "IL16", "IL17",
      # [C] 文献支持补充 (PubMed 验证 2026-07-20)
      "TRAIL", "FASLG", "BTLA",
      "IFN-I", "IFN-II", "IFN-lII",
      "ApoE", "IGF", "IGFBP", "TWEAK", "ADIPONECTIN"
    )

    available_pathways <- character(0)
    for (cn in names(cc_merged@LR)) {
      pn <- unique(cc_merged@LR[[cn]]$LRsig$pathway_name)
      available_pathways <- union(available_pathways, pn)
    }
    fa_pathways_avail <- fa_pathways[fa_pathways %in% available_pathways]
    log_info("[Step11] Total pathways across conditions: ",
             length(available_pathways),
             "; Available pathways of interest: ",
             paste(fa_pathways_avail, collapse = ", "))

    if (length(fa_pathways_avail) > 0) {
      # netAnalysis_signalingRole_heatmap 只支持 single CellChat object
      # CellChat v2: merged 对象的 netP 是 list, 函数检查 slot(object, "netP")$centr 会失败
      # 必须对每个 condition 单独绘制, 然后用 ComplexHeatmap::draw 组合
      log_info("[Step11] Drawing per-condition signalingRole_heatmap...")
      ht_list <- list()
      for (cond in names(cellchat_list)) {
        tryCatch({
          cc_single <- cellchat_list[[cond]]
          pw_avail <- unique(cc_single@LR$LRsig$pathway_name)
          pw_use <- fa_pathways_avail[fa_pathways_avail %in% pw_avail]
          if (length(pw_use) == 0) next
          ht_list[[cond]] <- netAnalysis_signalingRole_heatmap(
            cc_single, pattern = "outgoing",
            signaling = pw_use,
            title = cond,
            width = 10, height = 8
          )
        }, error = function(e) {
          log_warn("[Step11] heatmap failed for ", cond, ": ",
                   conditionMessage(e))
        })
      }
      if (length(ht_list) > 0) {
        tryCatch({
          png(file.path(cfg$project$figures_dir,
                         "11_cellchat_ferrosenescence_pathways_heatmap.png"),
              width = 14, height = 3 * length(ht_list), units = "in",
              res = cfg$viz$figure_dpi)
          for (i in seq_along(ht_list)) {
            draw(ht_list[[i]], column_title = names(ht_list)[i])
          }
          dev.off()
          log_info("[Step11] Figure saved: 11_cellchat_ferrosenescence_pathways_heatmap.png")
        }, error = function(e) {
          log_warn("[Step11] heatmap combine failed: ",
                   conditionMessage(e))
          try(dev.off(), silent = TRUE)
        })
      }

      # 各通路 outgoing/incoming 得分
      # netVisual_aggregate 源码访问 object@LR$LRsig (single object 结构)
      # merged 对象的 @LR 是按 condition 分组的 list, 无法直接调用
      # 改为: 对每个 condition 的 single object 单独绘制 pathway circle plot
      # 只绘制前 6 个 pathway (避免 37 通路 × 5 condition = 185 张图)
      pw_to_plot <- head(fa_pathways_avail, 6)
      for (pw in pw_to_plot) {
        for (cond in names(cellchat_list)) {
          tryCatch({
            cc_single <- cellchat_list[[cond]]
            pw_avail <- unique(cc_single@LR$LRsig$pathway_name)
            if (!(pw %in% pw_avail)) next
            png(file.path(cfg$project$figures_dir,
                           paste0("11_cellchat_pathway_", pw, "_", cond, ".png")),
                width = 8, height = 8, units = "in", res = cfg$viz$figure_dpi)
            netVisual_aggregate(cc_single, signaling = pw, layout = "circle")
            dev.off()
            log_info("[Step11] Figure saved: 11_cellchat_pathway_", pw, "_", cond, ".png")
          }, error = function(e) {
            log_debug("[Step11] Pathway ", pw, " for ", cond, " failed: ",
                      conditionMessage(e))
            try(dev.off(), silent = TRUE)
          })
        }
      }
    }

    # 6) Outgoing / Incoming 通讯模式 (pathway-level)
    # CellChat v2: netAnalysis_signalingRole_scatter 源码访问 slot(object, "netP")$centr
    # merged 对象的 netP 是 list, 无法直接调用 - 改为 per-condition scatter + patchwork 组合
    # CellChat v2 已移除 computeNetVisual_Pairwise; centrality 已在循环中计算
    scatter_list <- list()
    for (cond in names(cellchat_list)) {
      tryCatch({
        p <- netAnalysis_signalingRole_scatter(
          cellchat_list[[cond]], slot.name = "netP") +
          ggtitle(cond) +
          theme_pub(base_size = 9)
        scatter_list[[cond]] <- p
      }, error = function(e) {
        log_warn("[Step11] scatter failed for ", cond, ": ",
                 conditionMessage(e))
      })
    }
    if (length(scatter_list) > 0) {
      p_combined <- wrap_plots(scatter_list, ncol = 3) +
        plot_annotation(title = "Outgoing vs Incoming interaction strength")
      save_figure(p_combined, "11_cellchat_outgoing_pattern", cfg,
                  width = 14, height = 4 * ceiling(length(scatter_list) / 3))
    }

    save_rds(cc_merged, "11_cellchat_spatial_merged_final", cfg)
    invisible(cc_merged)
  } else {
    invisible(cellchat_list[[1]])
  }
}
