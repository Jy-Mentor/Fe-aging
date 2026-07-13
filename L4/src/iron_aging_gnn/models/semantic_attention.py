"""语义注意力机制 — 跨视图/跨模态语义级注意力融合

GHCDTI (2025) 提出语义注意力用于跨模态对齐：
  - 蛋白质嵌入表现出强选择性（高注意力集中在特定维度）
  - 药物嵌入呈现更分散的模式
  - 副作用和疾病节点呈现更平坦的分布

本模块实现两类核心组件：
  1. SemanticAttention: 对多视图节点表示学习语义级注意力权重并融合
  2. cross_view_infonce_loss: 跨视图 InfoNCE 对比损失

公式:
  omega_i = (1/|V|) Sigma alpha^T * tanh(W * h_i + b)
  beta_i = Softmax(omega_i)
  z = Sigma beta_i * h_i

参考:
  - GHCDTI (2025) "Semantic Attention for Cross-Modal Alignment"
  - Oord et al. (2018) "Representation Learning with Contrastive Predictive Coding", arXiv
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

EPS = 1e-8


class SemanticAttention(nn.Module):
    """跨视图语义注意力融合层。

    对来自不同视图/模态的节点表示，学习语义级别的注意力权重。
    参考 GHCDTI (2025) 的语义注意力机制。
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_views: int = 2,
        temperature: float = 0.1,
        use_bias: bool = True,
    ):
        """初始化语义注意力层。

        Args:
            hidden_dim: 节点嵌入维度。
            num_views: 视图数量（如不同 GNN 层、不同模态）。
            temperature: Softmax 温度系数，控制注意力分布的锐度。
            use_bias: 是否在 tanh 变换中使用偏置项。
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_views = num_views
        self.temperature = temperature

        self.W = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.alpha = nn.Parameter(torch.empty(hidden_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        if self.W.bias is not None:
            nn.init.zeros_(self.W.bias)
        nn.init.xavier_uniform_(self.alpha.view(1, -1))

    def forward(
        self,
        views: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播：计算语义注意力权重并融合多视图表示。

        Args:
            views: 多视图节点嵌入列表，每个元素形状为 (N, hidden_dim)。

        Returns:
            fused: (N, hidden_dim) 融合后的节点嵌入。
            attn_weights: (num_views,) 各视图的语义注意力权重 beta_i。
        """
        if len(views) != self.num_views:
            raise ValueError(
                f"期望 {self.num_views} 个视图，实际传入 {len(views)} 个"
            )

        omega = []
        for h in views:
            if h.dim() != 2 or h.shape[-1] != self.hidden_dim:
                raise ValueError(
                    f"视图张量形状必须为 (N, {self.hidden_dim})，实际 {h.shape}"
                )
            transformed = torch.tanh(self.W(h))
            scores = (self.alpha * transformed).sum(dim=-1)
            omega.append(scores.mean())

        omega = torch.stack(omega)
        beta = F.softmax(omega / self.temperature, dim=0)

        z = sum(beta[i] * views[i] for i in range(self.num_views))

        return z, beta

    def forward_with_node_attention(
        self,
        views: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回节点级注意力分数，用于可解释性分析。

        Args:
            views: 同 forward。

        Returns:
            fused: (N, hidden_dim) 融合后的节点嵌入。
            attn_weights: (num_views,) 语义级注意力权重 beta_i。
            node_attn: (num_views, N) 每个节点在每个视图上的原始注意力分数。
        """
        if len(views) != self.num_views:
            raise ValueError(
                f"期望 {self.num_views} 个视图，实际传入 {len(views)} 个"
            )

        omega = []
        node_scores = []
        for h in views:
            if h.dim() != 2 or h.shape[-1] != self.hidden_dim:
                raise ValueError(
                    f"视图张量形状必须为 (N, {self.hidden_dim})，实际 {h.shape}"
                )
            transformed = torch.tanh(self.W(h))
            scores = (self.alpha * transformed).sum(dim=-1)
            node_scores.append(scores)
            omega.append(scores.mean())

        omega = torch.stack(omega)
        beta = F.softmax(omega / self.temperature, dim=0)
        node_attn = torch.stack(node_scores, dim=0)

        z = sum(beta[i] * views[i] for i in range(self.num_views))

        return z, beta, node_attn

    def get_attention_entropy(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """计算注意力分布的熵，衡量注意力集中度。

        熵越低 -> 注意力越集中在少数视图（如蛋白质的强选择性模式）
        熵越高 -> 注意力越分散（如副作用/疾病的平坦分布）

        Args:
            attn_weights: (num_views,) 注意力权重，需已归一化。

        Returns:
            scalar: 注意力熵 H = -Sigma beta_i * log(beta_i)。
        """
        clamped = attn_weights.clamp(min=EPS)
        return -(clamped * clamped.log()).sum()


def cross_view_infonce_loss(
    view_embeddings: torch.Tensor,
    temperature: float = 0.07,
    symmetric: bool = True,
) -> torch.Tensor:
    """跨视图 InfoNCE 对比损失。

    正样本: 同一节点 n 在视图 i 和 j 中的表示 (h_i[n], h_j[n])
    负样本: 节点 n 在视图 i 中的表示与节点 m!=n 在视图 j 中的表示

    对所有视图对 (i, j), i!=j 计算 InfoNCE 并取平均。

    参考:
      - Oord et al. (2018) "Representation Learning with Contrastive
        Predictive Coding", arXiv
      - Chen et al. (2020) "A Simple Framework for Contrastive Learning
        of Visual Representations", ICML

    Args:
        view_embeddings: (num_views, N, hidden_dim) 多视图节点嵌入。
        temperature: InfoNCE 温度参数 tau（默认 0.07）。
        symmetric: 是否对每对视图计算对称损失 L(i,j) + L(j,i)。

    Returns:
        scalar loss。
    """
    num_views, N, hidden_dim = view_embeddings.shape
    if num_views < 2:
        raise ValueError(f"跨视图对比需要至少 2 个视图，实际 {num_views}")
    if N < 2:
        raise ValueError(f"跨视图对比需要至少 2 个节点用于负样本，实际 {N}")

    loss = torch.tensor(0.0, device=view_embeddings.device)
    pair_count = 0

    for i in range(num_views):
        for j in range(num_views):
            if i == j:
                continue
            h_i = F.normalize(view_embeddings[i], p=2, dim=-1)
            h_j = F.normalize(view_embeddings[j], p=2, dim=-1)

            sim = torch.matmul(h_i, h_j.T) / temperature

            labels = torch.arange(N, device=sim.device)
            loss_ij = F.cross_entropy(sim, labels)
            loss = loss + loss_ij
            pair_count += 1

            if symmetric:
                sim_t = sim.T
                loss_ji = F.cross_entropy(sim_t, labels)
                loss = loss + loss_ji
                pair_count += 1

    return loss / pair_count


def cross_view_infonce_loss_with_mask(
    view_embeddings: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 0.07,
) -> torch.Tensor:
    """带掩码的跨视图 InfoNCE 对比损失。

    支持掩码掉无效节点，仅对有效节点计算对比损失。

    Args:
        view_embeddings: (num_views, N, hidden_dim) 多视图节点嵌入。
        mask: (N,) bool 张量，True 表示有效节点。None 表示全部有效。
        temperature: InfoNCE 温度参数 tau。

    Returns:
        scalar loss。
    """
    num_views, N, hidden_dim = view_embeddings.shape
    if num_views < 2:
        raise ValueError(f"跨视图对比需要至少 2 个视图，实际 {num_views}")

    if mask is not None:
        valid_idx = mask.nonzero(as_tuple=True)[0]
        if valid_idx.numel() < 2:
            return torch.tensor(0.0, device=view_embeddings.device)
        view_embeddings = view_embeddings[:, valid_idx, :]

    return cross_view_infonce_loss(
        view_embeddings,
        temperature=temperature,
        symmetric=True,
    )


def compute_attention_diversity_loss(
    attn_weights: torch.Tensor,
    node_attn: torch.Tensor | None = None,
    target_entropy: float = 1.0,
    diversity_weight: float = 0.1,
    entropy_weight: float = 0.05,
) -> torch.Tensor:
    """注意力多样性正则化损失。

    两项正则：
      1. 多样性损失：阻止所有视图收敛到相同的注意力权重
      2. 熵正则：鼓励注意力分布接近目标熵值

    参考 GHCDTI (2025) 中不同节点类型呈现不同注意力模式。

    Args:
        attn_weights: (num_views,) 语义级注意力权重 beta。
        node_attn: (num_views, N) 节点级原始注意力分数，None 则跳过。
        target_entropy: 目标熵值。
        diversity_weight: 多样性损失权重。
        entropy_weight: 熵正则权重。

    Returns:
        scalar loss。
    """
    num_views = attn_weights.shape[0]
    loss = torch.tensor(0.0, device=attn_weights.device)

    if num_views > 1:
        diversity = attn_weights.var()
        loss = loss + diversity_weight * (-diversity)

    clamped = attn_weights.clamp(min=EPS)
    entropy = -(clamped * clamped.log()).sum()
    loss = loss + entropy_weight * (entropy - target_entropy) ** 2

    if node_attn is not None and node_attn.shape[1] > 1:
        node_beta = F.softmax(node_attn, dim=0)
        node_clamped = node_beta.clamp(min=EPS)
        node_entropy = -(node_clamped * node_clamped.log()).sum(dim=0).mean()
        loss = loss + entropy_weight * 0.5 * node_entropy

    return loss
