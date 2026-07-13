# 铁衰老 GNN 项目 Docker 镜像
# 基于 NVIDIA CUDA 运行时，支持 RTX 50 系 GPU
#
# 构建:
#   docker build -t iron-aging-gnn:latest -f L4/Dockerfile .
#   docker build --build-arg TORCH_VERSION=2.5.0 --build-arg CUDA_VERSION=121 -t iron-aging-gnn:latest -f L4/Dockerfile .
#
# 运行:
#   docker-compose up train
#   docker run --gpus all -v ./data:/workspace/data iron-aging-gnn:latest python entry/train.py

# 构建参数
ARG TORCH_VERSION=2.4.0
ARG CUDA_VERSION=121
ARG PYTHON_VERSION=3.10

FROM nvidia/cuda:12.1-devel-ubuntu22.04

# 防止交互式提示
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    ca-certificates \
    bzip2 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libgl1-mesa-glx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda (Python 3.10)
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    && conda clean -afy

# 创建项目环境
WORKDIR /workspace

# 复制依赖文件
COPY L4/environment.yml /workspace/environment.yml
COPY L4/requirements.txt /workspace/requirements.txt
COPY L4/pyproject.toml /workspace/pyproject.toml

# 创建 Conda 环境
RUN conda env create -f environment.yml \
    && conda clean -afy

# 激活环境
SHELL ["conda", "run", "-n", "iron_aging_gnn", "/bin/bash", "-c"]

# 安装 PyTorch Geometric 相关包
# 使用构建参数支持不同 PyTorch/CUDA 版本组合
RUN pip install --no-cache-dir \
    torch-geometric \
    pyg_lib \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html \
    && pip cache purge

# 安装额外工具
RUN pip install --no-cache-dir \
    optuna \
    mlflow \
    && pip cache purge

# 复制项目代码
COPY L4/ /workspace/

# 安装项目（开发模式）
RUN pip install -e . --no-deps

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
ENV OMP_NUM_THREADS=4

# 设置默认入口
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "iron_aging_gnn", "python"]
CMD ["entry/train.py", "--help"]