# 安装 Bioconductor 包 (ComplexHeatmap, circlize 等)
options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", type = "win.binary")
}

BiocManager::install(c("ComplexHeatmap", "circlize"), ask = FALSE, update = FALSE, version = "3.18")
message("Bioconductor packages installed")