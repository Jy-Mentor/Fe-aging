# ===========================================================================
# verify_github_methodology.R
# 从 GitHub API 下载并验证方法学文件完整性
# ===========================================================================

library(httr)

OWNER <- "Jy-Mentor"
REPO  <- "Fe-aging"
BRANCH <- "main"
TOKEN <- Sys.getenv("GITHUB_TOKEN")

FILES <- c(
  "L2/multi_omics_pipeline/R/19_methodology_figures_v2.R",
  "L2/multi_omics_pipeline/output/methodology_figures/methodology_manuscript_v2.md",
  "L2/multi_omics_pipeline/output/methodology_figures/Figure1_cross_omics_integration.png",
  "L2/multi_omics_pipeline/output/methodology_figures/Figure1_cross_omics_integration.pdf"
)

TMP_DIR <- tempdir()
dir.create(TMP_DIR, showWarnings = FALSE, recursive = TRUE)

verify_file <- function(rel_path) {
  message(sprintf("[VERIFY] %s", rel_path))
  url <- sprintf("https://api.github.com/repos/%s/%s/contents/%s?ref=%s",
                 OWNER, REPO, URLencode(rel_path, reserved = TRUE), BRANCH)
  resp <- GET(url, add_headers(
    Authorization = paste("token", TOKEN),
    Accept = "application/vnd.github+json"
  ))

  if (status_code(resp) != 200) {
    stop(sprintf("Remote file not found: HTTP %d - %s",
                 status_code(resp), content(resp, "text")))
  }

  info <- content(resp, "parsed")
  remote_sha <- info$sha
  remote_size <- info$size
  message(sprintf("       remote sha=%s, size=%d bytes", remote_sha, remote_size))

  # 通过 API content 字段 (base64) 下载, 避免 raw.githubusercontent.com 连接问题
  b64 <- info$content
  b64 <- gsub("\\s", "", b64)  # 去除可能的换行
  raw <- base64enc::base64decode(b64)
  tmp_file <- file.path(TMP_DIR, basename(rel_path))
  writeBin(raw, tmp_file)

  # 对于文本文件, 检查大小与基本可读性
  if (grepl("\\.(R|md|r|txt)$", rel_path)) {
    local_size <- file.info(tmp_file)$size
    txt <- readChar(tmp_file, local_size)
    message(sprintf("       local text size=%d, nchars=%d", local_size, nchar(txt)))
    if (grepl("\\.R$", rel_path)) {
      # R 文件语法检查
      res <- tryCatch(parse(file = tmp_file), error = function(e) e)
      if (inherits(res, "error")) {
        stop(sprintf("R parse failed: %s", conditionMessage(res)))
      }
      message("       R parse OK")
    }
  }

  # 对于 PNG, 检查 magic bytes
  if (grepl("\\.png$", rel_path)) {
    magic <- readBin(tmp_file, "raw", 8)
    is_png <- all(magic[1:8] == as.raw(c(0x89, 0x50, 0x4e, 0x47,
                                          0x0d, 0x0a, 0x1a, 0x0a)))
    if (!is_png) stop("PNG magic bytes mismatch")
    message(sprintf("       PNG magic bytes OK, size=%d bytes",
                    file.info(tmp_file)$size))
  }

  # 对于 PDF, 检查 %PDF 头
  if (grepl("\\.pdf$", rel_path)) {
    header <- readChar(tmp_file, 4)
    if (header != "%PDF") stop("PDF header mismatch")
    message(sprintf("       PDF header OK, size=%d bytes",
                    file.info(tmp_file)$size))
  }

  return(TRUE)
}

for (rel_path in FILES) {
  verify_file(rel_path)
}

message("\n[Done] All remote methodology files verified.")
