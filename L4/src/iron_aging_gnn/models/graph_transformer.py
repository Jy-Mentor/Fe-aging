"""Graph Transformer 编码器模块

基于 DHGT-DTI (2025) 的 Graph Transformer + 门控残差连接 + 语义注意力聚合。

核心机制:
  - 多层 TransformerConv 进行全局注意力消息传递
  - 门控残差连接（Gated Residual Connection）缓解过平滑
  - 语义注意力聚合（Semantic Attention Aggregation）融合多视角/元路径输出

参考:
  - DHGT-DTI (2025): Graph Transformer + 门控残差连接
  - Shi et al. (2021) "Masked Label Prediction: Unified Message Passing Model
    for Semi-Supervised Classification", IJCAI
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

logger = logging.getLogger(__name__)


class GatedResidual(nn.Module):
    """门控残差连接

    公式 (DHGT-DTI):
      z = Sigmoid(W_5 * h_prev + W_6 * h_new)
      h_out = z * h_new + (1 - z) * h_prev

    其中 h_prev = h_v^(k-1)（上一层输出），h_new = h_v^(k)（当前层聚合结果）。
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_5 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_6 = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, h_prev: torch.Tensor, h_new: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self.W_5(h_prev) + self.W_6(h_new))
        return z * h_new + (1 - z) * h_prev


class SemanticAttentionAggregation(nn.Module):
    """语义注意力聚合

    将多个元路径/视角的输出通过可学习注意力加权聚合。

    公式:
      ω_p = (1/|V|) Σ_{v∈V} α^T · tanh(W · h_v^p + b)
      β_p = Softmax(ω_p)
      z_v^global = Σ_p β_p · h_v^p
    """

    def __init__(self, hidden_dim: int, att_dim: int = 128):
        super().__init__()
        self.W = nn.Linear(hidden_dim, att_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(att_dim))
        self.alpha = nn.Parameter(torch.empty(att_dim, 1))
        nn.init.xavier_uniform_(self.alpha)

    def forward(self, meta_path_embeds: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            meta_path_embeds: 每个元路径的输出嵌入，形状均为 (N, hidden_dim)

        Returns:
            (N, hidden_dim) 注意力加权聚合后的全局嵌入
        """
        if len(meta_path_embeds) == 0:
            raise ValueError("meta_path_embeds 不能为空")
        if len(meta_path_embeds) == 1:
            return meta_path_embeds[0]

        stacked = torch.stack(meta_path_embeds, dim=0)
        P, N, D = stacked.shape

        flat = stacked.reshape(P * N, D)
        transformed = torch.tanh(self.W(flat) + self.b)
        scores = transformed @ self.alpha
        scores = scores.reshape(P, N, 1)

        omega = scores.mean(dim=1)
        beta = F.softmax(omega, dim=0)

        beta = beta.view(P, 1, 1)
        z_global = (beta * stacked).sum(dim=0)

        return z_global


class GraphTransformerEncoder(nn.Module):
    """Graph Transformer 编码器

    多层 TransformerConv + 可配置门控残差 + 语义注意力聚合。

    用法::

        encoder = GraphTransformerEncoder(
            in_dim=256, hidden_dim=128, output_dim=64,
            num_layers=3, num_heads=4, dropout=0.3,
            use_gated_residual=True,
        )
        # 单图编码
        embeds = encoder(x, edge_index)
        # 多视角（元路径）编码 + 语义注意力聚合
        global_embeds = encoder.forward_multi_view(x, [ei1, ei2, ei3])
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        use_gated_residual: bool = True,
        edge_dim: int | None = None,
    ):
        """
        Args:
            in_dim: 输入特征维度。
            hidden_dim: 隐藏层维度，必须能被 num_heads 整除。
            output_dim: 输出嵌入维度。
            num_layers: TransformerConv 层数。
            num_heads: 注意力头数。
            dropout: Dropout 概率。
            use_gated_residual: 是否使用门控残差连接。
            edge_dim: 边特征维度（可选，用于边特征增强的 TransformerConv）。
        """
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) 必须能被 num_heads ({num_heads}) 整除"
            )

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_gated_residual = use_gated_residual

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        head_dim = hidden_dim // num_heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=head_dim,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    concat=True,
                    bias=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            if use_gated_residual:
                self.gates.append(GatedResidual(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))

        self.output_proj = (
            nn.Identity()
            if hidden_dim == output_dim
            else nn.Linear(hidden_dim, output_dim, bias=False)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """单图前向传播。

        Args:
            x: (N, in_dim) 节点特征矩阵。
            edge_index: (2, E) 边索引。
            edge_attr: (E, edge_dim) 边特征，可选。

        Returns:
            (N, output_dim) 编码后的节点嵌入。
        """
        h = self.input_proj(x)
        h_prev = h

        for i, (conv, norm, drop) in enumerate(
            zip(self.convs, self.norms, self.dropouts, strict=False)
        ):
            h_new = conv(h, edge_index, edge_attr=edge_attr)

            if self.use_gated_residual and i < len(self.gates):
                h_new = self.gates[i](h_prev, h_new)
            else:
                h_new = h_new + h_prev

            h = norm(h_new)
            h = F.relu(h)
            h = drop(h)
            h_prev = h

        return self.output_proj(h)

    def forward_multi_view(
        self,
        x: torch.Tensor,
        edge_index_list: list[torch.Tensor],
        edge_attr_list: list[torch.Tensor | None] | None = None,
        att_dim: int = 128,
    ) -> torch.Tensor:
        """多视角（元路径）前向传播 + 语义注意力聚合。

        Args:
            x: (N, in_dim) 节点特征矩阵。
            edge_index_list: 每个元路径对应的边索引列表。
            edge_attr_list: 每个元路径对应的边特征列表，与 edge_index_list 等长。
            att_dim: 语义注意力聚合的隐藏维度。

        Returns:
            (N, output_dim) 语义注意力聚合后的全局嵌入。
        """
        if edge_attr_list is None:
            edge_attr_list = [None] * len(edge_index_list)
        if len(edge_index_list) != len(edge_attr_list):
            raise ValueError(
                f"edge_index_list 长度 ({len(edge_index_list)}) 与 "
                f"edge_attr_list 长度 ({len(edge_attr_list)}) 不一致"
            )

        meta_embeds: list[torch.Tensor] = []
        for ei, ea in zip(edge_index_list, edge_attr_list):
            embed = self.forward(x, ei, edge_attr=ea)
            meta_embeds.append(embed)

        aggregation = SemanticAttentionAggregation(
            hidden_dim=self.output_dim, att_dim=att_dim
        )
        return aggregation(meta_embeds)


__all__ = [
    "GraphTransformerEncoder",
    "GatedResidual",
    "SemanticAttentionAggregation",
]