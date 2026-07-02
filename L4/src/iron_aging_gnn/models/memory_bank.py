"""Memory Bank — 存储跨 batch 蛋白嵌入，供全局困难负样本采样

异步更新：每次训练步骤后，将当前 batch 的蛋白嵌入入队，
旧嵌入出队。内存开销：K * out_dim * 4 bytes。

参考: He et al. (2020) "Momentum Contrast for Unsupervised
      Visual Representation Learning", CVPR.
      Wu et al. (2021) "Self-supervised Learning on Graphs: Contrastive"
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class MemoryBank:
    """存储最近 K 个 batch 的蛋白嵌入，供全局困难负样本采样

    异步更新：每次训练步骤后，将当前 batch 的蛋白嵌入入队，
    旧嵌入出队。内存开销：K * out_dim * 4 bytes。

    参考: He et al. (2020) "Momentum Contrast for Unsupervised
          Visual Representation Learning", CVPR.
          Wu et al. (2021) "Self-supervised Learning on Graphs: Contrastive"
    """

    def __init__(self, max_size: int = 8192, out_dim: int = 64, device: str = "cpu"):
        """初始化 MemoryBank。

        Args:
            max_size: 存储的最大嵌入数量。
            out_dim: 嵌入维度。
            device: 存储设备（"cpu" 或 "cuda"）。
        """
        self.max_size = max_size
        self.out_dim = out_dim
        self.device = device
        self.bank = torch.zeros(max_size, out_dim, device=device)
        self.ptr = 0
        self.full = False

    def update(self, embeddings: torch.Tensor):
        """将新嵌入入队（FIFO）"""
        n = embeddings.shape[0]
        if n == 0:
            return
        end = self.ptr + n
        if end > self.max_size:
            # 环绕
            first_part = self.max_size - self.ptr
            self.bank[self.ptr:] = embeddings[:first_part].detach()
            remaining = n - first_part
            if remaining > 0:
                self.bank[:remaining] = embeddings[first_part:].detach()
            self.full = True
        else:
            self.bank[self.ptr:end] = embeddings.detach()
            if end == self.max_size:
                self.full = True
        self.ptr = (self.ptr + n) % self.max_size

    def sample(self, n: int) -> torch.Tensor:
        """从 bank 中随机采样 n 个嵌入"""
        available = self.max_size if self.full else self.ptr
        if available == 0:
            return torch.zeros(0, self.out_dim, device=self.device)
        n_sample = min(n, available)
        indices = torch.randperm(available, device=self.device)[:n_sample]
        return self.bank[indices]

    def size(self) -> int:
        """返回当前 bank 中可用的嵌入数量。"""
        return self.max_size if self.full else self.ptr
