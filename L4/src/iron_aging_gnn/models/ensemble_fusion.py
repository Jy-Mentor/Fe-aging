"""可学习集成融合模块

基于 KLaR (PMID 42412844) 门控融合思想：
- GatedEnsembleFusion: 上下文感知门控融合，动态控制各视图贡献
- LearnableEnsembleFusion: softmax 权重学习（H2GnnDTI SAIF 风格）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedEnsembleFusion(nn.Module):
    """上下文感知门控集成融合（KLaR 风格，PMID 42412844）。

    动态计算每个分支的贡献权重，而非固定的 softmax 权重。
    门控信号由各分支嵌入拼接后通过 MLP 生成，实现上下文感知融合。
    """

    def __init__(self, n_branches: int = 3, emb_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.n_branches = n_branches
        self.gate_net = nn.Sequential(
            nn.Linear(n_branches * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_branches),
        )
        self.temperature = nn.Parameter(torch.tensor(0.5))

    def forward(self, *branch_embs):
        if len(branch_embs) != self.n_branches:
            raise ValueError(
                f"GatedEnsembleFusion 期望 {self.n_branches} 个分支，"
                f"但收到 {len(branch_embs)} 个"
            )
        stacked = torch.cat(branch_embs, dim=-1)
        gate_logits = self.gate_net(stacked)
        gates = F.softmax(gate_logits / self.temperature.abs(), dim=-1)
        result = torch.zeros_like(branch_embs[0])
        for k, emb in enumerate(branch_embs):
            result = result + gates[:, k:k + 1] * emb
        return result


class LearnableEnsembleFusion(nn.Module):
    """可学习集成融合层（H2GnnDTI SAIF 风格）。"""

    def __init__(self, n_branches: int = 3, temperature: float = 0.1):
        super().__init__()
        self.log_weights = nn.Parameter(torch.zeros(n_branches))
        self.temperature = temperature

    def forward(self, *branch_scores):
        if len(branch_scores) != self.n_branches:
            raise ValueError(
                f"LearnableEnsembleFusion 期望 {self.n_branches} 个分支，"
                f"但收到 {len(branch_scores)} 个"
            )
        weights = F.softmax(self.log_weights / self.temperature, dim=0)
        result = torch.zeros_like(branch_scores[0])
        for w, score in zip(weights, branch_scores):
            result = result + w * score
        return result


__all__ = ["GatedEnsembleFusion", "LearnableEnsembleFusion"]
