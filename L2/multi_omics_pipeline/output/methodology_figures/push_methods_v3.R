library(httr)

local_path <- "D:/铁衰老 绝不重蹈覆辙/L2/multi_omics_pipeline/output/methodology_figures/methodology_manuscript_v3.md"
repo_path <- "L2/multi_omics_pipeline/output/methodology_figures/methodology_manuscript_v3.md"
owner <- "Jy-Mentor"
repo <- "Fe-aging"
branch <- "main"
commit_msg <- "Add comprehensive methodology manuscript v3 covering all 18 steps (<=3000 words)"

token <- Sys.getenv("GITHUB_TOKEN")
if (nchar(token) == 0) stop("GITHUB_TOKEN not set")

content_raw <- readBin(local_path, "raw", file.info(local_path)$size)
content_b64 <- base64enc::base64encode(content_raw)

url <- sprintf("https://api.github.com/repos/%s/%s/contents/%s?ref=%s",
               owner, repo, repo_path, branch)
resp <- GET(url, add_headers(Authorization = paste("token", token),
                              Accept = "application/vnd.github+json"))
old_sha <- NULL
if (status_code(resp) == 200) {
  old_sha <- content(resp)$sha
  message("Current remote sha: ", old_sha)
} else {
  message("File does not exist on remote; creating new file.")
}

put_url <- sprintf("https://api.github.com/repos/%s/%s/contents/%s",
                   owner, repo, repo_path)
body <- list(message = commit_msg, content = content_b64, branch = branch)
if (!is.null(old_sha)) body$sha <- old_sha

resp2 <- PUT(put_url,
             add_headers(Authorization = paste("token", token),
                         Accept = "application/vnd.github+json"),
             body = body, encode = "json")
message("Status: ", status_code(resp2))
print(content(resp2))
