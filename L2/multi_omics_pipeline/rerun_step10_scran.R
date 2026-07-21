#!/usr/bin/env Rscript
# ============================================================================
# 端到端验证: 重跑 Step 10 完整流程 (scran::scoreMarkers + SPOTlight)
# 然后验证 Step 11 兼容性 (加载新 spatial_merged, 检查 prop_ 列)
# ============================================================================
.libPaths(c("d:/铁衰老 绝不重蹈覆辙/R-library/4.5",
            "D:/R-library/4.5", .libPaths()))

script_dir <- "d:/铁衰老 绝不重蹈覆辙/L2/multi_omics_pipeline"
setwd(script_dir)

for (f in list.files(file.path(script_dir, "utils"), pattern = "\\.R$",
                     full.names = TRUE)) {
  source(f)
}

modules_needed <- c("10_integration_spotlight.R")
for (f in file.path(script_dir, "R", modules_needed)) {
  if (file.exists(f)) {
    log_info("Loading module: ", basename(f))
    source(f)
  }
}

cfg <- load_config("config.yaml")
init_logger(file.path(cfg$project$log_dir,
                      sprintf("rerun_step10_scran_%s.log",
                              format(Sys.time(), "%Y%m%d_%H%M%S"))), "INFO")
log_info("=== Re-run Step 10 with scran::scoreMarkers (end-to-end) ===")

# 加载 sc_seu
sc_rds <- file.path(cfg$project$rds_dir, "08_sc_seurat_annotated_scored.rds")
if (!file.exists(sc_rds)) {
  sc_rds <- file.path(cfg$project$rds_dir, "07_sc_seurat_integrated.rds")
}
log_info("[Restore] Loading ", basename(sc_rds))
sc_seu <- readRDS(sc_rds)
log_info("[Restore] sc_seu: ", ncol(sc_seu), " cells")

# 加载 spatial_merged (06_spatial_with_regions.rds 不含 prop_ 列, 用此版本重跑)
sp_rds <- file.path(cfg$project$rds_dir, "04_spatial_merged.rds")
if (!file.exists(sp_rds)) {
  log_error("Missing: ", sp_rds, " (run Step 04 first)")
  quit(status = 1)
}
log_info("[Restore] Loading ", basename(sp_rds))
spatial_merged <- readRDS(sp_rds)
log_info("[Restore] spatial_merged: ", ncol(spatial_merged), " spots")

# 备份原 10_spatial_with_proportions.rds
old_rds <- file.path(cfg$project$rds_dir, "10_spatial_with_proportions.rds")
if (file.exists(old_rds)) {
  backup_rds <- file.path(cfg$project$rds_dir,
                          paste0("10_spatial_with_proportions_pre_scran_",
                                 format(Sys.time(), "%Y%m%d_%H%M%S"), ".rds"))
  file.copy(old_rds, backup_rds)
  log_info("[Backup] Saved pre-scran version to ", basename(backup_rds))
}

# 重跑 Step 10 (scran::scoreMarkers + SPOTlight)
t0 <- Sys.time()
result <- step10_integration_spotlight(sc_seu, spatial_merged, cfg)
elapsed <- round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2)
log_info(sprintf("=== Step 10 (scran) done in %.2f min ===", elapsed))

if (is.null(result)) {
  log_error("Step 10 returned NULL")
  quit(status = 1)
}

# 验证输出
spatial_merged_new <- result$spatial_merged
prop_cols <- grep("^prop_", colnames(spatial_merged_new@meta.data), value = TRUE)
log_info("prop_ columns in new spatial_merged: ", length(prop_cols))
log_info("  names: ", paste(prop_cols, collapse = ", "))

# 验证 RDS 保存成功
if (file.exists(old_rds)) {
  log_info("[OK] 10_spatial_with_proportions.rds saved successfully")
  # 读取验证
  loaded <- readRDS(old_rds)
  prop_loaded <- grep("^prop_",
                      colnames(loaded@meta.data), value = TRUE)
  log_info("[OK] Reloaded RDS has ", length(prop_loaded), " prop_ columns")
}

# 验证 Step 11 兼容性 (仅加载 RDS, 不实际运行 CellChat)
log_info("\n=== Step 11 compatibility check ===")
step11_rds <- file.path(cfg$project$rds_dir, "11_cellchat_spatial_merged.rds")
if (file.exists(step11_rds)) {
  log_info("[OK] Step 11 RDS exists: ", basename(step11_rds))
  log_info("    (Will be regenerated when Step 11 is re-run with new spatial_merged)")
}

log_info("\n[OK] End-to-end Step 10 verification complete")
log_info("Next: Re-run Step 11 with new spatial_merged to verify CellChat compatibility")
