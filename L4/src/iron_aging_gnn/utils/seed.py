"""随机种子设置工具
================
确保实验可复现：固定 Python、NumPy、PyTorch（CPU/GPU）随机种子，
并配置 CuDNN/cuBLAS 确定性行为。
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """设置全局随机种子以保证实验可复现。

    除固定 Python / NumPy / PyTorch 种子外，本函数还会：
      - 关闭 CuDNN benchmark，启用 deterministic 模式；
      - 设置 CUBLAS_WORKSPACE_CONFIG 以保证 cuBLAS 确定性；
      - 在支持的环境下启用 torch.use_deterministic_algorithms。

    Args:
        seed: 随机种子值，默认 42。
        deterministic: 是否强制确定性算法（可能降低训练速度，但提升可复现性）。

    Returns:
        None
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # CuDNN 确定性设置
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # cuBLAS 工作区配置，避免非确定性算法选择
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        # PyTorch 全局确定性算法（部分算子可能不支持，使用 warn_only=True）
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int, base_seed: int | None = None) -> None:
    """DataLoader worker 初始化函数，确保多进程加载时每个 worker 种子独立。

    建议与如下 DataLoader 参数配合使用：
        DataLoader(..., worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))

    Args:
        worker_id: DataLoader 自动传入的 worker 编号。
        base_seed: 基础随机种子；为 None 时使用当前 PyTorch 初始种子。
    """
    worker_seed = (base_seed if base_seed is not None else torch.initial_seed()) % (2**32)
    worker_seed = (worker_seed + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
