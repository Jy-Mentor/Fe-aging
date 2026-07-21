# ===========================================================================
# push_methodology_to_github.R
# 使用 R httr + GitHub API 推送方法学段落与配图到 GitHub
# 绕过本地 git 端口 443 连接问题
# ===========================================================================

library(httr)

# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
OWNER <- "Jy-Mentor"
REPO  <- "Fe-aging"
BRANCH <- "main"
TOKEN <- Sys.getenv("GITHUB_TOKEN")
if (nchar(TOKEN) == 0) stop("GITHUB_TOKEN not set")

LOCAL_BASE <- "D:/铁衰老 绝不重蹈覆辙"

# (相对路径, 本地绝对路径 = file.path(LOCAL_BASE, rel_path))
FILES <- c(
  "L2/multi_omics_pipeline/R/19_methodology_figures_v2.R",
  "L2/multi_omics_pipeline/output/methodology_figures/methodology_manuscript_v2.md",
  "L2/multi_omics_pipeline/output/methodology_figures/Figure1_cross_omics_integration.png",
  "L2/multi_omics_pipeline/output/methodology_figures/Figure1_cross_omics_integration.pdf",
  "L2/multi_omics_pipeline/output/methodology_figures/Figure1A_pathway_axis_match_rate.pdf",
  "L2/multi_omics_pipeline/output/methodology_figures/Figure1B_kegg_top15.pdf",
  "L2/multi_omics_pipeline/output/methodology_figures/Figure1C_sat1_polyamine.pdf"
)

COMMIT_MESSAGE <- "docs(methodology): add methodology manuscript v2 + publication figures (Fig1 A/B/C)"

# --------------------------------------------------------------------------
# 辅助函数
# --------------------------------------------------------------------------
get_remote_sha <- function(rel_path) {
  url <- sprintf("https://api.github.com/repos/%s/%s/contents/%s?ref=%s",
                 OWNER, REPO, URLencode(rel_path, reserved = TRUE), BRANCH)
  resp <- GET(url, add_headers(
    Authorization = paste("token", TOKEN),
    Accept = "application/vnd.github+json"
  ))
  if (status_code(resp) == 200) {
    content(resp, "parsed")$sha
  } else if (status_code(resp) == 404) {
    NULL  # 文件不存在
  } else {
    stop(sprintf("Failed to get SHA for %s: HTTP %d - %s",
                 rel_path, status_code(resp), content(resp, "text")))
  }
}

push_file <- function(rel_path) {
  local_path <- file.path(LOCAL_BASE, rel_path)
  if (!file.exists(local_path)) {
    stop(sprintf("Local file not found: %s", local_path))
  }

  message(sprintf("[PUSH] %s", rel_path))
  sha <- get_remote_sha(rel_path)

  raw <- readBin(local_path, "raw", file.info(local_path)$size)
  b64 <- base64enc::base64encode(raw)

  url <- sprintf("https://api.github.com/repos/%s/%s/contents/%s",
                 OWNER, REPO, URLencode(rel_path, reserved = TRUE))
  body <- list(
    message = COMMIT_MESSAGE,
    content = b64,
    branch  = BRANCH
  )
  if (!is.null(sha)) {
    body$sha <- sha
    message(sprintf("       updating existing file (sha=%s)", substr(sha, 1, 7)))
  } else {
    message("       creating new file")
  }

  resp <- PUT(url, add_headers(
    Authorization = paste("token", TOKEN),
    Accept = "application/vnd.github+json"
  ), body = body, encode = "json")

  if (status_code(resp) %in% c(200, 201)) {
    result <- content(resp, "parsed")
    message(sprintf("       OK -> commit %s", substr(result$commit$sha, 1, 7)))
    return(result)
  } else {
    stop(sprintf("Failed to push %s: HTTP %d - %s",
                 rel_path, status_code(resp), content(resp, "text")))
  }
}

# --------------------------------------------------------------------------
# 批量推送
# --------------------------------------------------------------------------
for (rel_path in FILES) {
  tryCatch({
    push_file(rel_path)
  }, error = function(e) {
    message(sprintf("[ERROR] %s: %s", rel_path, conditionMessage(e)))
    stop(conditionMessage(e))
  })
}

message("\n[Done] All methodology files pushed to GitHub.")
