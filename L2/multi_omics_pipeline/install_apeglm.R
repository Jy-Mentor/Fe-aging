#!/usr/bin/env Rscript
# 安装 apeglm + ashr (LFC shrinkage 依赖)
project_root <- "d:/铁衰老 绝不重蹈覆辙"
target_lib <- file.path(project_root, "R-library/4.5")
if (!dir.exists(target_lib)) dir.create(target_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(target_lib, "D:/R-library/4.5", "D:/R-library", .libPaths()))

options(timeout = 600)

to_install <- c("apeglm", "ashr")
cat("=== Pre-install check ===\n")
for (p in to_install) {
  ok <- requireNamespace(p, quietly = TRUE)
  status <- if (ok) paste0("INSTALLED (v", as.character(packageVersion(p)), ")")
            else "MISSING"
  cat(sprintf("  %-12s: %s\n", p, status))
}

cat("\n=== Installing ===\n")
for (p in to_install) {
  if (!requireNamespace(p, quietly = TRUE)) {
    cat(sprintf("\n[%s] Installing...\n", p))
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
  }
}

cat("\n=== Verification ===\n")
for (p in to_install) {
  ok <- requireNamespace(p, quietly = TRUE)
  v  <- if (ok) as.character(packageVersion(p)) else "NA"
  cat(sprintf("  %-12s: %s v%s\n", p, if (ok) "OK" else "MISSING", v))
}
