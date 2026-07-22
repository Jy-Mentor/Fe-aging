# 安装与当前 R 版本兼容的二进制包
options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))

pkgs <- c("rlang", "ggplot2", "tidyverse", "magrittr", "dplyr", "tidyr",
          "Seurat", "patchwork", "ComplexHeatmap", "circlize",
          "ggExtra", "ggpubr", "RColorBrewer", "grid", "cowplot", "viridis",
          "ggrepel", "scales", "svglite", "jsonlite", "ggridges",
          "stringr", "forcats")

for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    message("Installing ", p)
    install.packages(p, type = "win.binary", dependencies = TRUE)
  } else {
    message(p, " already installed")
  }
}

message("Done")
