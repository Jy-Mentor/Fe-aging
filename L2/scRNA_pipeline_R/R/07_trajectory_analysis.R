# ============================================================================
# STEP 7: 轨迹分析 (Monocle3)
# - 对 Microglia / OLs / OPCs / Astrocytes 进行伪时间轨迹推断
# - 关联铁衰老基因随伪时间的变化
# 参考: Trapnell et al. 2014 Nature Biotechnology; Qiu et al. 2017 Nature Methods
# ============================================================================

step07_trajectory_analysis <- function(seu, cfg) {
  log_info("[Step7] Trajectory analysis with monocle3...")

  if (!requireNamespace("monocle3", quietly = TRUE)) {
    log_warn("[Step7] monocle3 not installed. Trying slingshot fallback.")
    return(step07_trajectory_slingshot(seu, cfg))
  }
  suppressPackageStartupMessages({
    library(monocle3)
    library(SeuratWrappers)
  })

  target_types <- cfg$trajectory$cell_types_to_trajectory
  celltype_col <- cfg$analysis$celltype_col

  trajectory_results <- list()
  for (ct in target_types) {
    log_info("[Step7] Processing: ", ct)
    cells_ct <- Cells(seu)[seu[[celltype_col, drop = TRUE]] == ct]
    n_ct <- length(cells_ct)
    if (n_ct < 100) {
      log_warn("[Step7] ", ct, ": too few cells (", n_ct, "), skip")
      next
    }
    seu_sub <- subset(seu, cells = cells_ct)
    seu_sub$Condition <- factor(seu_sub$Condition,
                                levels = cfg$analysis$condition_levels)
    log_info("[Step7] ", ct, ": ", ncol(seu_sub),
             " cells, conditions: ",
             paste(levels(seu_sub$Condition), collapse = ","))

    # 7.1 Convert to monocle3 CDS via SeuratWrappers
    cds <- tryCatch({
      SeuratWrappers::as.cell_data_set(seu_sub)
    }, error = function(e) {
      log_warn("[Step7] as.cell_data_set failed for ", ct,
               ": ", conditionMessage(e))
      return(NULL)
    })
    if (is.null(cds)) next

    # 7.2 Re-cluster
    cds <- monocle3::cluster_cells(cds, reduction_method = "UMAP")
    cds <- monocle3::learn_graph(cds, use_partition = TRUE)

    # 7.3 选取 root nodes (Ctrl cells preferred)
    tryCatch({
      ctrl_cells <- rownames(cds@colData)[cds@colData$Condition == "Ctrl"]
      if (length(ctrl_cells) >= 10) {
        cds <- monocle3::order_cells(cds, root_cells = ctrl_cells)
      } else {
        cds <- monocle3::order_cells(cds)
      }
    }, error = function(e) {
      log_warn("[Step7] order_cells failed for ", ct, ": ", conditionMessage(e))
    })

    pseudotime <- monocle3::pseudotime(cds)
    cds@colData$pseudotime <- pseudotime
    log_info("[Step7] ", ct, ": pseudotime range [",
             round(min(pseudotime, na.rm = TRUE), 2), ", ",
             round(max(pseudotime, na.rm = TRUE), 2), "]")

    # 7.4 轨迹可视化
    tryCatch({
      p_traj <- monocle3::plot_cells(
        cds, color_cells_by = "pseudotime",
        label_groups_by_cluster = FALSE,
        label_leaves = FALSE, label_branch_points = FALSE,
        graph_label_size = 3
      ) + labs(title = paste0("Trajectory - ", ct))
      save_figure(p_traj, sprintf("07_trajectory_pseudotime_%s", ct), cfg,
                  width = 9, height = 7)
    }, error = function(e) {
      log_warn("[Step7] plot_cells pseudotime failed for ", ct,
               ": ", conditionMessage(e))
    })

    tryCatch({
      p_cond <- monocle3::plot_cells(
        cds, color_cells_by = "Condition",
        label_groups_by_cluster = FALSE,
        label_leaves = FALSE, label_branch_points = FALSE
      ) + labs(title = paste0(ct, " - Condition on trajectory"))
      save_figure(p_cond, sprintf("07_trajectory_condition_%s", ct), cfg,
                  width = 9, height = 7)
    }, error = function(e) {
      log_warn("[Step7] plot_cells condition failed for ", ct,
               ": ", conditionMessage(e))
    })

    # 7.5 铁衰老基因随伪时间变化
    fa_genes <- load_ferroaging_genes(cfg)
    fa_mouse <- map_human_to_mouse(fa_genes)
    expr_mat <- as.matrix(Seurat::GetAssayData(seu_sub, assay = "RNA", layer = "data"))
    fa_avail <- intersect(fa_mouse, rownames(expr_mat))

    if (length(fa_avail) > 0) {
      cds_subset <- cds[fa_avail, ]
      deg_pseudo <- monocle3::top_markers(cds_subset,
                                          group_cells_by = "pseudotime",
                                          reference_cells = 1000,
                                          verbose = FALSE)
      save_table(as.data.frame(deg_pseudo),
                 sprintf("07_pseudotime_degs_%s", ct), cfg)

      # 7.6 铁衰老评分 vs 伪时间
      if ("Ferroaging" %in% colnames(cds@colData)) {
        df_plot <- data.frame(
          pseudotime = pseudotime,
          Ferroaging = cds@colData$Ferroaging,
          Condition = cds@colData$Condition
        )
        df_plot <- df_plot[!is.na(df_plot$pseudotime), ]
        p_fa_pt <- ggplot(df_plot, aes(x = pseudotime, y = Ferroaging,
                                       color = Condition)) +
          geom_point(alpha = 0.5, size = 0.8) +
          geom_smooth(method = "loess", se = TRUE, alpha = 0.2) +
          scale_color_manual(values = CONDITION_COLORS) +
          labs(title = paste0("Ferroaging score vs pseudotime (", ct, ")"),
               x = "Pseudotime", y = "UCell Ferroaging") +
          theme_pub()
        save_figure(p_fa_pt, sprintf("07_ferroaging_vs_pseudotime_%s", ct), cfg,
                    width = 8, height = 6)
      }
    }

    trajectory_results[[ct]] <- list(
      cds = cds, n_cells = ncol(cds),
      pseudotime_range = range(pseudotime, na.rm = TRUE)
    )
    rm(seu_sub, cds); gc(verbose = FALSE)
  }

  saveRDS(trajectory_results,
          file.path(cfg$project$rds_dir, "trajectory_results.rds"))
  log_info("[Step7] Trajectory analysis done. Types processed: ",
           paste(names(trajectory_results), collapse = ", "))
  invisible(trajectory_results)
}

step07_trajectory_slingshot <- function(seu, cfg) {
  if (!requireNamespace("slingshot", quietly = TRUE)) {
    log_warn("[Step7] slingshot also not installed; skip trajectory.")
    return(invisible(NULL))
  }
  log_info("[Step7] Fallback: slingshot trajectory...")
  library(slingshot)

  target_types <- cfg$trajectory$cell_types_to_trajectory
  for (ct in target_types) {
    cells_ct <- Cells(seu)[seu[[cfg$analysis$celltype_col, drop = TRUE]] == ct]
    if (length(cells_ct) < 100) next
    seu_sub <- subset(seu, cells = cells_ct)
    pca_emb <- Seurat::Embeddings(seu_sub, "pca")[, 1:10]
    cluster_labels <- seu_sub$Condition
    sling <- slingshot(pca_emb, clusterLabels = cluster_labels,
                       start.clus = "Ctrl")
    log_info("[Step7] ", ct, ": slingshot pseudotime computed.")
  }
  invisible(NULL)
}
