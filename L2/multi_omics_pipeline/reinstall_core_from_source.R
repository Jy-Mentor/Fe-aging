# 强制从源码重新安装核心依赖, 解决二进制包与 R 4.3.3 不兼容问题
options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))

# 按依赖顺序排列
pkgs <- c(
  "Rcpp", "cli", "glue", "fansi", "utf8", "rlang", "vctrs",
  "lifecycle", "magrittr", "R6", "crayon", "pkgconfig", "pillar",
  "tibble", "tidyselect", "dplyr", "purrr", "tidyr", "stringr",
  "forcats", "ggplot2", "scales", "gtable", "isoband", "RColorBrewer",
  "farver", "labeling", "munsell", "generics", "cpp11", "readr",
  "hms", "bit64", "vroom", "tzdb", "progress", "jsonlite"
)

for (p in pkgs) {
  message("=== Installing ", p, " from source ===")
  tryCatch(
    install.packages(p, type = "source", dependencies = FALSE),
    error = function(e) message("ERROR installing ", p, ": ", conditionMessage(e))
  )
}

message("=== Done ===")