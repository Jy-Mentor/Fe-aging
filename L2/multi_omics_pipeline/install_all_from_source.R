# 从源码安装所有可能因二进制不兼容而无法加载的包
options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))

pkgs <- c(
  "cli", "glue", "fansi", "utf8", "lifecycle", "vctrs", "rlang",
  "pillar", "tibble", "dplyr", "tidyr", "ggplot2", "stringr", "forcats",
  "purrr", "readr", "tidyverse", "scales", "gtable", "isoband",
  "RColorBrewer", "farver", "labeling", "munsell", "Rcpp", "R6",
  "magrittr", "generics", "cpp11", "tzdb", "vroom", "hms", "bit64",
  "progress", "crayon", "withr", "pkgconfig", "tidyselect", "ellipsis",
  "sys", "askpass", "openssl", "curl", "xml2", "httr", "jsonlite"
)

for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    message("Installing ", p, " from source...")
    tryCatch(
      install.packages(p, type = "source", dependencies = FALSE),
      error = function(e) message("Failed to install ", p, ": ", conditionMessage(e))
    )
  } else {
    message(p, " already installed")
  }
}

message("Done")
