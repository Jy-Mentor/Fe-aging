#!/usr/bin/env Rscript
# ============================================================================
# 主控脚本: GSE233815 单细胞铁衰老分析 Pipeline
# R 4.5.2 + Seurat 5
# 数据: Zucha et al. 2024 PNAS MCAO 小鼠脑单核 RNA-seq (7414 细胞, PMID:39499634)
# ============================================================================

# 加入 Bioc 包备用安装路径 (clusterProfiler, monocle3, GSVA, fgsea 等均在此)
# 项目内 R-library/4.5 优先 (TRAE Sandbox 兼容), 然后 D:/R-library/4.5
proj_lib <- file.path(getwd(), "R-library/4.5")
.lib_paths_to_add <- character(0)
if (dir.exists(proj_lib)) .lib_paths_to_add <- c(.lib_paths_to_add, proj_lib)
if (dir.exists("D:/R-library/4.5")) .lib_paths_to_add <- c(.lib_paths_to_add, "D:/R-library/4.5")
if (length(.lib_paths_to_add) > 0) {
  .libPaths(c(.lib_paths_to_add, .libPaths()))
}

suppressPackageStartupMessages({
  library(Seurat)
})

set.seed(42)

script_dir <- tryCatch({
  arg_file <- grep("--file=", commandArgs(FALSE), value = TRUE, fixed = TRUE)
  if (length(arg_file) > 0) {
    dirname(normalizePath(sub("--file=", "", arg_file[1])))
  } else {
    cd <- getwd()
    if (file.exists(file.path(cd, "config.yaml"))) cd else "."
  }
}, error = function(e) ".")

if (!dir.exists(file.path(script_dir, "outputs"))) {
  script_dir <- "."
}

source(file.path(script_dir, "utils/io_helpers.R"))
source(file.path(script_dir, "utils/gene_sets.R"))
source(file.path(script_dir, "utils/plot_helpers.R"))

cfg <- load_config(file.path(script_dir, "config.yaml"))
ensure_dirs(cfg)
setup_logger(cfg)

log_info("==========================================================")
log_info("GSE233815 MCAO Ferroaging scRNA Pipeline")
log_info("R version: ", R.version.string)
log_info("Working dir: ", getwd())
log_info("Script dir: ", script_dir)
log_info("==========================================================")

args <- commandArgs(trailingOnly = TRUE)
steps_to_run <- if (length(args) == 0) 1:8 else as.integer(args)

step_files <- list(
  "1" = "R/01_data_loading_validation.R",
  "2" = "R/02_preprocessing_qc.R",
  "3" = "R/03_clustering_dimred.R",
  "4" = "R/04_de_analysis.R",
  "5" = "R/05_cellchat_analysis.R",
  "6" = "R/06_ferroptosis_enrichment.R",
  "7" = "R/07_trajectory_analysis.R",
  "8" = "R/08_report_generation.R"
)

seu <- NULL
step_results <- list()

# 如果跳过 Step 1, 自动加载最近可用的 Seurat RDS
if (!(1 %in% steps_to_run)) {
  candidate_rds <- c(
    file.path(cfg$project$rds_dir, "seurat_with_ferroaging.rds"),
    file.path(cfg$project$rds_dir, "seurat_loaded.rds")
  )
  loaded <- FALSE
  for (rds_path in candidate_rds) {
    if (file.exists(rds_path)) {
      log_info("Auto-loading Seurat from: ", rds_path)
      seu <- readRDS(rds_path)
      log_info("Loaded: ", nrow(seu), " genes x ", ncol(seu), " cells")
      loaded <- TRUE
      break
    }
  }
  if (!loaded) {
    stop("No Seurat RDS found. Run Step 1 first to load and validate the data.")
  }
}

for (step in steps_to_run) {
  step_str <- as.character(step)
  if (!step_str %in% names(step_files)) {
    log_warn("Unknown step: ", step, ". Skipping.")
    next
  }
  script_path <- file.path(script_dir, step_files[[step_str]])
  if (!file.exists(script_path)) {
    log_error("Step script not found: ", script_path)
    next
  }
  log_info(">>> STEP ", step, ": ", basename(script_path))
  t0 <- Sys.time()
  tryCatch({
    source(script_path, local = TRUE)
    elapsed <- difftime(Sys.time(), t0, units = "secs")
    log_info("<<< STEP ", step, " done (", round(as.numeric(elapsed), 1), " s)")
  }, error = function(e) {
    log_error("STEP ", step, " failed: ", conditionMessage(e))
    print(traceback())
    stop("Step ", step, " failed: ", conditionMessage(e))
  })
}

log_info("==========================================================")
log_info("Pipeline finished at ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
log_info("==========================================================")
