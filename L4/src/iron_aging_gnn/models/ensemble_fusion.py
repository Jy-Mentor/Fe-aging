"""可学习集成融合模块（H2GnnDTI SAIF 风格）

使用 softmax 权重替代固定加权平均，权重通过梯度下降学习。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableEnsembleFusion(nn.Module):
    """可学习集成融合层（H2GnnDTI SAIF 风格）

    使用 softmax 权重替代固定加权平均，权重通过梯度下降学习。

    用法::

        fusion = LearnableEnsembleFusion(n_branches=3, temperature=0.1)
        fused_scores = fusion(sage_scores, hgt_scores, simplehgn_scores)

    注意：在 TCM 预测阶段，AUPR 动态加权已提供有效的集成策略。
    此模块作为可选替代方案，可在需要端到端可学习融合时启用。
    """

    def __init__(self, n_branches: int = 3, temperature: float = 0.1):
        super().__init__()
        self.log_weights = nn.Parameter(torch.zeros(n_branches))
        self.temperature = temperature

    def forward(self, *branch_scores: torch.Tensor) -> torch.Tensor:
        """融合多个分支的预测分数。

        Args:
            *branch_scores: 每个分支的 (n_tcm, n_genes) 预测分数张量

        Returns:
            (n_tcm, n_genes) 融合后的分数
        """
        weights = F.softmax(self.log_weights / self.temperature, dim=0)
        result = torch.zeros_like(branch_scores[0])
        for w, score in zip(weights, branch_scores):
            result = result + w * score
        return result


__all__ = ["LearnableEnsembleFusion"]
