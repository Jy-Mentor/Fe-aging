"""蛋白条件化非对称特征调制模块

基于 CFM-DTI (PMID 42341701) 核心思想：
- 蛋白特征作为条件信号，自适应校准药物特征表示
- 非对称融合：蛋白 → 条件参数 → 药物特征重校准
- 替代简单的对称拼接/加法融合，更好地捕获蛋白上下文对药物表示的影响

参考:
  - Li et al. (2026) "CFM-DTI: Protein-conditioned feature modulation for DTI", Comput Biol Chem
  - 在随机拆分下 AUC 0.8590, AUPR 0.8581
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProteinConditionedModulation(nn.Module):
    """蛋白条件化特征调制层。

    从蛋白嵌入生成 (gamma, beta) 调制参数，对药物嵌入进行
    逐特征的仿射变换，实现蛋白上下文驱动的药物表示自适应校准。

    Args:
        drug_dim: 药物嵌入维度
        prot_dim: 蛋白嵌入维度
        hidden_dim: 调制参数生成网络的隐藏层维度
        output_dim: 输出维度（None 则不投影）
        dropout: 调制参数生成网络中的 Dropout 比率
    """

    def __init__(
        self,
        drug_dim: int,
        prot_dim: int,
        hidden_dim: int = 128,
        output_dim: int | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.drug_dim = drug_dim
        self.prot_dim = prot_dim
        self.output_dim = output_dim or drug_dim

        self.gamma_net = nn.Sequential(
            nn.Linear(prot_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, drug_dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(prot_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, drug_dim),
        )

        self.output_proj = None
        if output_dim is not None and output_dim != drug_dim:
            self.output_proj = nn.Linear(drug_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        for net in [self.gamma_net, self.beta_net]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        self.gamma_net[-1].weight.data.normal_(0, 0.01)
        self.gamma_net[-1].bias.data.fill_(1.0)

        self.beta_net[-1].weight.data.normal_(0, 0.01)
        self.beta_net[-1].bias.data.zero_()

        if self.output_proj is not None:
            nn.init.xavier_uniform_(self.output_proj.weight)
            if self.output_proj.bias is not None:
                nn.init.zeros_(self.output_proj.bias)

    def forward(self, drug_emb: torch.Tensor, prot_emb: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_net(prot_emb)
        beta = self.beta_net(prot_emb)

        modulated = gamma * drug_emb + beta

        if self.output_proj is not None:
            modulated = self.output_proj(modulated)

        return modulated


class CrossModalGatedFusion(nn.Module):
    """跨模态门控融合层。

    基于 drGT (PMID 42304175) 注意力门控思想：
    使用跨模态注意力计算药物和蛋白的交互权重，
    实现更精细的多模态特征融合。

    Args:
        drug_dim: 药物嵌入维度
        prot_dim: 蛋白嵌入维度
        hidden_dim: 门控网络隐藏层维度
        dropout: Dropout 比率
    """

    def __init__(
        self,
        drug_dim: int,
        prot_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        num_heads = 4
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) 必须能被 num_heads ({num_heads}) 整除，"
                f"MultiheadAttention 要求 embed_dim 能被 num_heads 整除"
            )
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.prot_proj = nn.Linear(prot_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.output_proj = nn.Linear(hidden_dim, drug_dim)

    def forward(self, drug_emb: torch.Tensor, prot_emb: torch.Tensor) -> torch.Tensor:
        drug_h = self.drug_proj(drug_emb).unsqueeze(1)
        prot_h = self.prot_proj(prot_emb).unsqueeze(1)

        attn_out, _ = self.attn(drug_h, prot_h, prot_h)
        attn_out = attn_out.squeeze(1)

        gate_input = torch.cat([drug_h.squeeze(1), attn_out], dim=-1)
        gate_val = self.gate(gate_input)

        fused = gate_val * drug_h.squeeze(1) + (1 - gate_val) * attn_out
        return self.output_proj(fused)


__all__ = ["ProteinConditionedModulation", "CrossModalGatedFusion"]
