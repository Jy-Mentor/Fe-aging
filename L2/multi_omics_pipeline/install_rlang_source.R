# 尝试从源码安装新版 rlang
options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))
install.packages("rlang", type = "source")
message("rlang source install attempted")