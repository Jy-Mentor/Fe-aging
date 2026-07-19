#!/usr/bin/env Rscript
# ============================================================================
# 四层递进"铁衰老"多组学证据链 - 主控脚本
# ============================================================================
# 数据集:
#   L1 Bulk RNA-seq:    GSE233811 (MCAO 12h/D1/D3/D7 时间序列)
#   L2 Spatial:         GSE233814 (10x Visium 空间切片)
#   L3 Single-cell:     GSE233518 / GSE233815 (scRNA-seq MCAO)
#   L4 Integration:     SPOTlight + CellChat spatial + CMap 反证
#
# 使用方法:
#   Rscript run_pipeline.R [step_numbers]
#   示例:
#     Rscript run_pipeline.R              # 运行所有步骤
#     Rscript run_pipeline.R 1 2 3        # 只运行 L1 三步
#     Rscript run_pipeline.R 4 5 6        # 只运行 L2 三步
#     Rscript run_pipeline.R 7 8 9        # 只运行 L3 三步
#     Rscript run_pipeline.R 10 11 12     # 只运行 L4 三步
#     Rscript run_pipeline.R 13           # 仅生成最终报告
# ============================================================================

# 加入 Bioc 包备用安装路径 (与 scRNA_pipeline_R 共享)
if (dir.exists("D:/R-library/4.5")) {
  .libPaths(c("D:/R-library/4.5", .libPaths()))
}

suppressPackageStartupMessages({
  library(Seurat)
})

# ----------------------------------------------------------------------------
# 加载配置与工具函数
# ----------------------------------------------------------------------------
script_dir <- dirname(sys.frame(1)$ofile %||% ".")

# 兼容: 当直接 Rscript 调用时, script_dir 可能取不到
if (is.null(script_dir) || script_dir == "." || !nzchar(script_dir)) {
  script_dir <- getwd()
}

# 加载 utils
for (f in list.files(file.path(script_dir, "utils"), pattern = "\\.R$",
                     full.names = TRUE)) {
  source(f)
}

# 加载所有模块 (R/ 下的所有 .R 文件)
r_files <- list.files(file.path(script_dir, "R"), pattern = "\\.R$",
                      full.names = TRUE)
for (f in sort(r_files)) {
  log_info("Loading module: ", basename(f))
  source(f)
}

# ----------------------------------------------------------------------------
# 解析命令行参数 (步骤号)
# ----------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  steps_to_run <- 1:13
} else {
  steps_to_run <- as.integer(args)
  steps_to_run <- steps_to_run[!is.na(steps_to_run)]
  if (length(steps_to_run) == 0) {
    stop("No valid step numbers provided. Examples: Rscript run_pipeline.R 1 2 3")
  }
}

log_info("Steps to run: ", paste(steps_to_run, collapse = ", "))

# ----------------------------------------------------------------------------
# 初始化
# ----------------------------------------------------------------------------
cfg <- load_config(file.path(script_dir, "config.yaml"))
set_seed_all(cfg$reproducibility$r_seed)

log_file <- file.path(cfg$project$log_dir,
                      sprintf("pipeline_%s.log",
                              format(Sys.time(), "%Y%m%d_%H%M%S")))
init_logger(log_file, level = "INFO")

log_info("============================================================")
log_info("铁衰老多组学证据链 Pipeline 启动")
log_info("项目根目录: ", cfg$project$root)
log_info("R 版本: ", R.version.string)
log_info("随机种子: ", cfg$reproducibility$r_seed)
log_info("步骤: ", paste(steps_to_run, collapse = ", "))
log_info("============================================================")

# 全局对象 (跨步骤传递)
bulk_dds        <- NULL      # L1: DESeq2 对象
bulk_vsd        <- NULL      # L1: VST 变换后矩阵
bulk_dea_list   <- NULL      # L1: 各时间点 DE 结果
wgcna_modules   <- NULL      # L1: WGCNA 模块
spatial_list    <- NULL      # L2: 各切片 Seurat 对象列表
spatial_merged  <- NULL      # L2: 合并后 Seurat 对象
sc_seu          <- NULL      # L3: 单细胞 Seurat 对象
sc_augur_res    <- NULL      # L3: Augur 结果
spotlight_res   <- NULL      # L4: SPOTlight 去卷积结果
cellchat_spatial<- NULL      # L4: 空间 CellChat 结果
cmap_result     <- NULL      # L4: CMap 反证结果

# ----------------------------------------------------------------------------
# 步骤调度
# ----------------------------------------------------------------------------
step_descriptions <- c(
  "01" = "[L1-1] Bulk 数据加载与验证",
  "02" = "[L1-2] Bulk DESeq2 差异 + LRT 时序",
  "03" = "[L1-3] Bulk GSEA + WGCNA 共表达",
  "04" = "[L2-1] Spatial 数据加载与 SCTransform",
  "05" = "[L2-2] Spatial 铁衰老评分与定位",
  "06" = "[L2-3] Spatial 半暗带识别与空间变量特征",
  "07" = "[L3-1] scRNA 数据加载 + Harmony 整合",
  "08" = "[L3-2] scRNA 细胞注释 + UCell 评分",
  "09" = "[L3-3] scRNA monocle3 拟时序 + Augur",
  "10" = "[L4-1] SPOTlight 空间去卷积",
  "11" = "[L4-2] CellChat 空间细胞通讯",
  "12" = "[L4-3] CMap BCP 反证分析",
  "13" = "[最终] 综合报告生成"
)

# 执行各步骤
for (step in steps_to_run) {
  step_name <- sprintf("%02d", step)
  desc <- step_descriptions[step_name]
  if (is.na(desc)) {
    log_warn("Unknown step: ", step, ". Skipping.")
    next
  }
  log_info("============================================================")
  log_info("STEP ", step_name, ": ", desc)
  log_info("============================================================")

  t0 <- Sys.time()
  tryCatch({
    switch(as.character(step),
      "1"  = bulk_dds <- step01_bulk_load_validate(cfg),
      "2"  = { res <- step02_bulk_dea_timeseries(bulk_dds, cfg)
               bulk_dds <- res$dds; bulk_vsd <- res$vsd; bulk_dea_list <- res$dea_list },
      "3"  = wgcna_modules <- step03_bulk_gsea_wgcna(bulk_dds, bulk_dea_list, cfg),
      "4"  = spatial_list <- step04_spatial_load_qc(cfg),
      "5"  = spatial_merged <- step05_spatial_module_score(spatial_list, cfg),
      "6"  = spatial_merged <- step06_spatial_penumbra(spatial_merged, cfg),
      "7"  = sc_seu <- step07_sc_load_integrate(cfg),
      "8"  = sc_seu <- step08_sc_annotate_score(sc_seu, cfg),
      "9"  = { res <- step09_sc_pseudotime_augur(sc_seu, cfg)
               sc_augur_res <- res$augur },
      "10" = spotlight_res <- step10_integration_spotlight(sc_seu, spatial_merged, cfg),
      "11" = cellchat_spatial <- step11_integration_cellchat_spatial(spatial_merged, cfg),
      "12" = cmap_result <- step12_integration_cmap(bulk_dea_list, cfg),
      "13" = step13_report_generation(cfg,
                                       bulk_dds = bulk_dds,
                                       bulk_dea_list = bulk_dea_list,
                                       wgcna_modules = wgcna_modules,
                                       spatial_merged = spatial_merged,
                                       sc_seu = sc_seu,
                                       sc_augur_res = sc_augur_res,
                                       spotlight_res = spotlight_res,
                                       cellchat_spatial = cellchat_spatial,
                                       cmap_result = cmap_result),
      stop("Unknown step: ", step)
    )
    elapsed <- difftime(Sys.time(), t0, units = "mins")
    log_info(sprintf("STEP %s completed in %.2f minutes", step_name, as.numeric(elapsed)))
  }, error = function(e) {
    log_error("STEP ", step_name, " failed: ", conditionMessage(e))
    log_error("Call stack: ", paste(deparse(sys.calls()[[1]]), collapse = ""))
    # 不中断, 继续执行下一步 (允许部分失败)
    if (step == 13) {
      log_warn("Final report will be generated with available results only.")
    }
  })
}

log_info("============================================================")
log_info("Pipeline 完成. 总耗时: ",
         round(as.numeric(difftime(Sys.time(), t0_global <- t0_global %||% Sys.time(),
                                   units = "mins")), 2), " minutes")
log_info("日志文件: ", log_file)
log_info("============================================================")

# 保存 session info
if (cfg$reproducibility$log_session_info) {
  sess_file <- file.path(cfg$project$log_dir,
                         sprintf("session_info_%s.txt",
                                 format(Sys.time(), "%Y%m%d_%H%M%S")))
  writeLines(capture.output(sessionInfo()), sess_file)
  log_info("Session info saved: ", sess_file)
}
