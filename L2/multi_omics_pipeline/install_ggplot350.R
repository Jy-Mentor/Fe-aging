# 安装与 rlang 1.1.5 兼容的 ggplot2 3.5.0
options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))

url <- "https://mirrors.tuna.tsinghua.edu.cn/CRAN/src/contrib/Archive/ggplot2/ggplot2_3.5.0.tar.gz"
dest <- tempfile(fileext = ".tar.gz")
download.file(url, dest, mode = "wb")
install.packages(dest, repos = NULL, type = "source")
message("ggplot2 3.5.0 install attempted")
