#!/usr/bin/env Rscript
# 安装缺失的 Bioc 包: DESeq2 (L1), Augur (L3), SPOTlight (L4)
# 安装位置: 项目内 R-library/4.5 (TRAE Sandbox 限制外部路径写入)
project_root <- "d:/铁衰老 绝不重蹈覆辙"
target_lib <- file.path(project_root, "R-library/4.5")
if (!dir.exists(target_lib)) {
  dir.create(target_lib, recursive = TRUE, showWarnings = FALSE)
}
.libPaths(c(target_lib, "D:/R-library/4.5", "D:/R-library", .libPaths()))

# 使用 Bioconductor 官方源 (镜像不稳定时回退到官方)
# 不设置 BioC_mirror, 让 BiocManager 使用默认 https://bioconductor.org
options(timeout = 600)  # 10 分钟超时, 避免大文件下载失败

# 验证 target_lib 可写
writable <- tryCatch({
  tf <- tempfile(tmpdir = target_lib)
  writeLines("test", tf)
  unlink(tf)
  TRUE
}, error = function(e) FALSE)
if (!writable) {
  stop("target_lib not writable: ", target_lib)
}

cat("Target lib:", target_lib, "\n")
cat(".libPaths():\n"); print(.libPaths())

to_install <- c("DESeq2", "SPOTlight")  # Bioc 包
github_pkgs <- c("Augur" = "neurorestore/Augur")  # GitHub 包

cat("=== Pre-install check ===\n")
for (p in c(to_install, names(github_pkgs))) {
  ok <- requireNamespace(p, quietly = TRUE)
  status <- if (ok) paste0("INSTALLED (v", as.character(packageVersion(p)), ")")
            else "MISSING -> will install"
  cat(sprintf("  %-12s: %s\n", p, status))
}

cat("\n=== Installing Bioc packages (this may take 10-30 minutes) ===\n")
for (p in to_install) {
  if (!requireNamespace(p, quietly = TRUE)) {
    cat(sprintf("\n[%s] Installing from Bioconductor...\n", p))
    flush.console()
    t0 <- Sys.time()
    tryCatch({
      BiocManager::install(p, lib = target_lib, update = FALSE,
                           ask = FALSE, force = FALSE)
      cat(sprintf("[%s] Done in %.1f min. Installed: %s\n", p,
                  as.numeric(difftime(Sys.time(), t0, units = "mins")),
                  requireNamespace(p, quietly = TRUE)))
    }, error = function(e) {
      cat(sprintf("[%s] FAILED: %s\n", p, conditionMessage(e)))
    })
    flush.console()
  } else {
    cat(sprintf("\n[%s] Already installed, skipping.\n", p))
  }
}

cat("\n=== Installing GitHub packages ===\n")
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes", lib = target_lib)
}
for (p in names(github_pkgs)) {
  if (!requireNamespace(p, quietly = TRUE)) {
    repo <- github_pkgs[[p]]
    cat(sprintf("\n[%s] Installing from GitHub: %s...\n", p, repo))
    flush.console()
    t0 <- Sys.time()
    tryCatch({
      remotes::install_github(repo, lib = target_lib, upgrade = "never",
                              dependencies = TRUE)
      cat(sprintf("[%s] Done in %.1f min. Installed: %s\n", p,
                  as.numeric(difftime(Sys.time(), t0, units = "mins")),
                  requireNamespace(p, quietly = TRUE)))
    }, error = function(e) {
      cat(sprintf("[%s] FAILED: %s\n", p, conditionMessage(e)))
    })
    flush.console()
  } else {
    cat(sprintf("\n[%s] Already installed, skipping.\n", p))
  }
}

cat("\n=== Post-install verification ===\n")
for (p in c(to_install, names(github_pkgs))) {
  ok <- requireNamespace(p, quietly = TRUE)
  v  <- if (ok) as.character(packageVersion(p)) else "NA"
  loc <- if (ok) {
    idx <- which(installed.packages()[, "Package"] == p)[1]
    installed.packages()[idx, "LibPath"]
  } else "NA"
  cat(sprintf("  %-12s: %s v%s @ %s\n", p, if (ok) "OK" else "MISSING", v, loc))
}
cat("\nDone.\n")
