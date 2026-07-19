#!/usr/bin/env Rscript
# 临时脚本: 检查 GSE233815 两个 series_matrix 文件的平台与样本信息
src1 <- "C:/Users/Jy-Mentor-7/Desktop/申请书/原始数据/GSE233815-GPL19057_series_matrix.txt.gz"
src2 <- "C:/Users/Jy-Mentor-7/Desktop/申请书/原始数据/GSE233815-GPL24247_series_matrix.txt.gz"

for (src in c(src1, src2)) {
  cat("\n==================================================\n")
  cat("File:", basename(src), "\n")
  cat("==================================================\n")
  if (!file.exists(src)) {
    cat("FILE NOT FOUND\n"); next
  }
  con <- gzfile(src, "rt")
  lines <- readLines(con, n = 40)
  close(con)
  # 仅打印元数据头 (以 !series_matrix_table_begin 结束)
  for (ln in lines) {
    cat(ln, "\n")
    if (grepl("!series_matrix_table_begin", ln, fixed = TRUE)) break
  }
}
