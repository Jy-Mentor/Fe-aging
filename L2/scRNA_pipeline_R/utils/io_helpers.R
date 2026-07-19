# ============================================================================
# IO 与配置工具函数
# ============================================================================

suppressPackageStartupMessages({
  library(yaml)
  library(glue)
})

.LOG_FILE <- NULL

load_config <- function(config_path = "config.yaml") {
  if (!file.exists(config_path)) {
    stop("Config file not found: ", config_path)
  }
  cfg <- yaml::read_yaml(config_path)
  # 将相对路径解析为相对于 project$root 的绝对路径
  root <- cfg$project$root
  path_keys <- c("outputs_dir", "log_dir", "figures_dir", "tables_dir", "rds_dir",
                 "seurat_object", "seurat_object_raw", "bulk_expr", "bulk_pheno",
                 "ferroaging_genes", "core_genes", "pipeline_dir")
  for (k in path_keys) {
    v <- cfg$project[[k]]
    if (!is.null(v) && !is.null(root) && !nzchar(v)) next
    if (!is.null(v) && is.character(v) && !startsWith(v, "/") &&
        !(nchar(v) >= 2 && substr(v, 2, 2) == ":")) {
      cfg$project[[k]] <- normalizePath(file.path(root, v), winslash = "/", mustWork = FALSE)
    }
  }
  return(cfg)
}

ensure_dirs <- function(cfg) {
  dirs <- c(cfg$project$outputs_dir, cfg$project$log_dir,
            cfg$project$figures_dir, cfg$project$tables_dir,
            cfg$project$rds_dir)
  for (d in dirs) {
    if (!dir.exists(d)) dir.create(d, showWarnings = FALSE, recursive = TRUE)
  }
  invisible(cfg)
}

setup_logger <- function(cfg) {
  log_file <- file.path(cfg$project$log_dir,
                        sprintf("pipeline_%s.log",
                                format(Sys.time(), "%Y%m%d_%H%M%S")))
  assign(".LOG_FILE", log_file, envir = .GlobalEnv)
  log_info("Logger initialized. Log file: ", log_file)
  invisible(log_file)
}

fmt_msg <- function(..., .env = parent.frame()) {
  parts <- list(...)
  if (length(parts) == 0) return("")
  if (length(parts) == 1 && is.character(parts[[1]]) &&
      grepl("\\{", parts[[1]])) {
    return(glue::glue(parts[[1]], .envir = .env))
  }
  paste0(...)
}

log_info <- function(...) {
  msg <- fmt_msg(..., .env = parent.frame())
  ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  line <- sprintf("[INFO ] %s | %s", ts, msg)
  message(line)
  lf <- tryCatch(get(".LOG_FILE", envir = .GlobalEnv), error = function(e) NULL)
  if (!is.null(lf)) {
    try(cat(line, "\n", file = lf, append = TRUE), silent = TRUE)
  }
}

log_warn <- function(...) {
  msg <- fmt_msg(..., .env = parent.frame())
  ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  line <- sprintf("[WARN ] %s | %s", ts, msg)
  message(line)
  lf <- tryCatch(get(".LOG_FILE", envir = .GlobalEnv), error = function(e) NULL)
  if (!is.null(lf)) {
    try(cat(line, "\n", file = lf, append = TRUE), silent = TRUE)
  }
}

log_error <- function(...) {
  msg <- fmt_msg(..., .env = parent.frame())
  ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  line <- sprintf("[ERROR] %s | %s", ts, msg)
  message(line)
  lf <- tryCatch(get(".LOG_FILE", envir = .GlobalEnv), error = function(e) NULL)
  if (!is.null(lf)) {
    try(cat(line, "\n", file = lf, append = TRUE), silent = TRUE)
  }
}

save_figure <- function(plot, filename, cfg, width = NULL, height = NULL,
                        dpi = NULL) {
  w <- if (is.null(width)) cfg$report$figure_width else width
  h <- if (is.null(height)) cfg$report$figure_height else height
  d <- if (is.null(dpi)) cfg$report$figure_dpi else dpi
  fmt <- cfg$report$figure_format
  full_path <- file.path(cfg$project$figures_dir,
                         sprintf("%s.%s", filename, fmt))
  ggplot2::ggsave(full_path, plot = plot, width = w, height = h,
                  dpi = d, units = "in")
  log_info("Saved figure: ", full_path)
  invisible(full_path)
}

save_table <- function(df, filename, cfg) {
  full_path <- file.path(cfg$project$tables_dir,
                         sprintf("%s.csv", filename))
  utils::write.csv(df, full_path, row.names = FALSE)
  log_info("Saved table: ", full_path, " (", nrow(df), " rows)")
  invisible(full_path)
}

save_rds <- function(obj, filename, cfg) {
  full_path <- file.path(cfg$project$rds_dir, sprintf("%s.rds", filename))
  saveRDS(obj, full_path)
  log_info("Saved RDS: ", full_path)
  invisible(full_path)
}

check_packages <- function(pkgs) {
  missing <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
  if (length(missing) > 0) {
    log_warn("Missing packages: ", paste(missing, collapse = ", "))
    return(invisible(missing))
  }
  invisible(character(0))
}
