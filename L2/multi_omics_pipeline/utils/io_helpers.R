# ============================================================================
# IO 与日志辅助函数 (io_helpers.R)
# - 配置加载 + 路径解析
# - 日志记录 (同时输出到 console 和文件)
# - 通用工具函数
# ============================================================================

# ----------------------------------------------------------------------------
# 配置加载: 解析相对路径为绝对路径 (基于 project$root)
# ----------------------------------------------------------------------------
load_config <- function(config_path = "config.yaml") {
  if (!file.exists(config_path)) {
    stop("Config file not found: ", config_path)
  }
  cfg <- yaml::read_yaml(config_path)

  # 相对路径 → 绝对路径 (以 project$root 为基准)
  root <- cfg$project$root
  if (is.null(root)) {
    stop("config.yaml 缺少 project$root 字段")
  }
  root <- normalizePath(root, winslash = "/", mustWork = FALSE)

  # project 下的路径键
  project_path_keys <- c("outputs_dir", "log_dir", "figures_dir", "tables_dir",
                         "rds_dir")
  for (k in project_path_keys) {
    v <- cfg$project[[k]]
    if (!is.null(v) && is.character(v) && nchar(v) > 0 &&
        !startsWith(v, "/") && !(nchar(v) >= 2 && substr(v, 2, 2) == ":")) {
      cfg$project[[k]] <- normalizePath(file.path(root, v), winslash = "/",
                                         mustWork = FALSE)
    }
  }

  # data 路径 (相对于 root 的父目录, 即项目根 d:/铁衰老 绝不重蹈覆辙/)
  parent_root <- dirname(root)
  data_keys <- c("bulk_dir", "bulk_counts", "bulk_pheno", "spatial_dir",
                 "sc_dir", "sc_seurat_rds")
  for (k in data_keys) {
    v <- cfg$data[[k]]
    if (!is.null(v) && is.character(v) && nchar(v) > 0 &&
        !startsWith(v, "/") && !(nchar(v) >= 2 && substr(v, 2, 2) == ":")) {
      cfg$data[[k]] <- normalizePath(file.path(parent_root, v), winslash = "/",
                                      mustWork = FALSE)
    }
  }
  # spatial_samples + sc_samples 路径
  for (key in c("spatial_samples", "sc_samples")) {
    if (!is.null(cfg$data[[key]])) {
      for (sn in names(cfg$data[[key]])) {
        v <- cfg$data[[key]][[sn]]
        if (is.character(v) && nchar(v) > 0 &&
            !startsWith(v, "/") && !(nchar(v) >= 2 && substr(v, 2, 2) == ":")) {
          cfg$data[[key]][[sn]] <- normalizePath(file.path(parent_root, v),
                                                  winslash = "/", mustWork = FALSE)
        }
      }
    }
  }
  # ferroaging_file
  fa_file <- cfg$gene_sets$ferroaging_file
  if (!is.null(fa_file) && is.character(fa_file) && nchar(fa_file) > 0) {
    cfg$gene_sets$ferroaging_file <- normalizePath(file.path(parent_root, fa_file),
                                                    winslash = "/", mustWork = FALSE)
  }

  # 确保输出目录存在
  for (k in project_path_keys) {
    dir.create(cfg$project[[k]], recursive = TRUE, showWarnings = FALSE)
  }

  return(cfg)
}

# ----------------------------------------------------------------------------
# 日志系统: 同时输出到 console 和日志文件
# ----------------------------------------------------------------------------
LOG_LEVELS <- c(DEBUG = 10, INFO = 20, WARN = 30, ERROR = 40)
CURRENT_LOG_LEVEL <- "INFO"
LOG_FILE <- NULL

init_logger <- function(log_file, level = "INFO") {
  CURRENT_LOG_LEVEL <<- level
  LOG_FILE <<- log_file
  dir.create(dirname(log_file), recursive = TRUE, showWarnings = FALSE)
  cat(sprintf("[%s][%s] Logger initialized: %s\n",
              format(Sys.time(), "%Y-%m-%d %H:%M:%S"), level, log_file),
      file = log_file, append = FALSE)
  invisible(log_file)
}

log_msg <- function(level, ...) {
  if (LOG_LEVELS[level] < LOG_LEVELS[CURRENT_LOG_LEVEL]) return(invisible(NULL))
  msg <- paste0(...)
  line <- sprintf("[%s][%s] %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
                  level, msg)
  cat(line, "\n")
  if (!is.null(LOG_FILE)) {
    cat(line, "\n", file = LOG_FILE, append = TRUE)
  }
}

log_info  <- function(...) log_msg("INFO", ...)
log_warn  <- function(...) log_msg("WARN", ...)
log_error <- function(...) log_msg("ERROR", ...)
log_debug <- function(...) log_msg("DEBUG", ...)

# ----------------------------------------------------------------------------
# 通用工具
# ----------------------------------------------------------------------------
save_figure <- function(plot_obj, name, cfg, width = NULL, height = NULL,
                        dpi = NULL) {
  w <- if (is.null(width)) cfg$viz$figure_width else width
  h <- if (is.null(height)) cfg$viz$figure_height else height
  d <- if (is.null(dpi)) cfg$viz$figure_dpi else dpi
  path <- file.path(cfg$project$figures_dir, paste0(name, ".png"))
  ggplot2::ggsave(path, plot_obj, width = w, height = h, dpi = d,
                  units = "in", bg = "white")
  log_info("Figure saved: ", path)
  invisible(path)
}

save_table <- function(df, name, cfg, ext = "csv") {
  path <- file.path(cfg$project$tables_dir, paste0(name, ".", ext))
  if (ext == "csv") {
    write.csv(df, path, row.names = FALSE)
  } else if (ext == "tsv") {
    write.table(df, path, sep = "\t", row.names = FALSE, quote = FALSE)
  }
  log_info("Table saved: ", path, " (", nrow(df), " rows)")
  invisible(path)
}

save_rds <- function(obj, name, cfg) {
  path <- file.path(cfg$project$rds_dir, paste0(name, ".rds"))
  saveRDS(obj, path)
  log_info("RDS saved: ", path)
  invisible(path)
}

# 设置随机种子 (R + Python reticulate)
set_seed_all <- function(seed) {
  set.seed(seed)
  if (requireNamespace("reticulate", quietly = TRUE)) {
    tryCatch(reticulate::py_set_seed(seed), error = function(e) NULL)
  }
  log_info("Random seed set: ", seed)
}

# 检查必需包是否安装; 未安装则报错
require_packages <- function(pkgs, install_hint = NULL) {
  missing <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
  if (length(missing) > 0) {
    msg <- paste0("Missing required packages: ", paste(missing, collapse = ", "))
    if (!is.null(install_hint)) msg <- paste0(msg, "\nInstall hint: ", install_hint)
    stop(msg)
  }
  invisible(TRUE)
}
