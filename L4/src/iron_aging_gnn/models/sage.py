"""GraphSAGE 编码器 + 可插拔解码器 — 同构图链接预测模型

特性:
  - 化合物/蛋白特征投影器，解耦高维特征与图卷积
  - 支持 MLP / Dot / Bilinear / ResidueAwareBilinear 解码器切换
  - 多任务联合训练（铁死亡表型分类头）

参考:
  - Hamilton et al. (2017) "GraphSAGE", NeurIPS
  - Veleiro et al. (2024) "GeNNius", Bioinformatics
  - Rives et al. (2021) "ESM-2", PNAS
  - GraphBAN (Nature Communications, 2025): 双线性注意力用于化合物-蛋白交互建模
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from .decoders import MLPDecoder, DotProductDecoder, BilinearDecoder, ResidueAwareBilinearDecoder

logger = logging.getLogger(__name__)

_PROT_PROJ_DROPOUT = 0.4          # 蛋白特征投影器外部 Dropout
_PROT_PROJ_INNER_DROPOUT = 0.3    # 蛋白特征投影器内部 Dropout
_PATHWAY_PROJ_DROPOUT = 0.3       # 通路投影器 Dropout
_PHENO_HEAD_DROPOUT = 0.3         # 表型分类头 Dropout
_TEMPERATURE = 5.0                # 温度参数 T（固定，不参与梯度更新）


class SAGELinkPredictor(nn.Module):
    """GraphSAGE 编码器 + 可插拔解码器 + 表型分类头"""

    def __init__(self, comp_feat_dim: int, prot_feat_dim: int, n_compounds: int,
                 hidden_dim: int = 64, out_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.5,
                 n_pathways: int = 0,
                 prot_proj_dropout: float = _PROT_PROJ_DROPOUT,
                 prot_proj_inner_dropout: float = _PROT_PROJ_INNER_DROPOUT,
                 pathway_proj_dropout: float = _PATHWAY_PROJ_DROPOUT,
                 pheno_head_dropout: float = _PHENO_HEAD_DROPOUT,
                 temperature: float = _TEMPERATURE,
                 decoder_type: str = "mlp"):
        """初始化 SAGE 链接预测模型。

        Args:
            comp_feat_dim: 化合物特征维度。
            prot_feat_dim: 蛋白特征维度（ESM-2 嵌入维度）。
            n_compounds: 化合物节点数量。
            hidden_dim: 隐藏层维度。
            out_dim: 输出嵌入维度。
            num_layers: SAGEConv 层数。
            dropout: Dropout 概率。
            n_pathways: 通路 one-hot 特征维度（0 表示无通路特征）。
            prot_proj_dropout: 蛋白投影器外部 Dropout。
            prot_proj_inner_dropout: 蛋白投影器内部 Dropout。
            pathway_proj_dropout: 通路投影器 Dropout。
            pheno_head_dropout: 表型头 Dropout。
            temperature: 解码温度系数 T。
            decoder_type: 解码器类型：mlp / dot / bilinear / residue_bilinear。
        """
        super().__init__()
        self.comp_feat_dim = comp_feat_dim
        self.prot_esm_dim = prot_feat_dim
        self.n_compounds = n_compounds
        self.out_dim = out_dim
        self.n_pathways = n_pathways
        self.decoder_type = decoder_type

        self.temperature = temperature

        # 化合物特征投影器
        self.comp_proj = nn.Sequential(
            nn.Linear(comp_feat_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
        )

        # 蛋白特征投影器
        self.prot_feat_proj = nn.Sequential(
            nn.Linear(prot_feat_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(prot_proj_inner_dropout),
            nn.Linear(256, hidden_dim),
        )

        # 通路特征独立投影器
        if n_pathways > 0:
            self.pathway_proj = nn.Sequential(
                nn.Linear(n_pathways, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(pathway_proj_dropout),
                nn.Linear(128, hidden_dim),
            )
        else:
            self.pathway_proj = None

        self.prot_dropout = nn.Dropout(prot_proj_dropout)

        # SAGEConv 层
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropouts.append(nn.Dropout(dropout))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))

        self.convs.append(SAGEConv(hidden_dim, out_dim))
        self.norms.append(nn.Identity())
        self.dropouts.append(nn.Dropout(dropout))

        # 可插拔解码器
        if decoder_type == "mlp":
            self.decoder = MLPDecoder(out_dim, hidden_dim=64, dropout=pheno_head_dropout)
        elif decoder_type == "dot":
            self.decoder = DotProductDecoder()
        elif decoder_type == "bilinear":
            self.decoder = BilinearDecoder(out_dim, rank=64)
        elif decoder_type == "residue_bilinear":
            self.decoder = ResidueAwareBilinearDecoder(
                comp_dim=out_dim, residue_dim=prot_feat_dim, rank=64, hidden_dim=128, dropout=pheno_head_dropout
            )
        else:
            raise ValueError(f"不支持的 decoder_type: {decoder_type}")

        # 铁死亡表型分类头
        self.pheno_head = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(pheno_head_dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                n_compounds: int = None, use_pathway: bool = True) -> torch.Tensor:
        """前向传播

        Args:
            x: (N, feat_dim) 节点特征矩阵
            edge_index: (2, E) 边索引
            n_compounds: 子图中化合物节点数
            use_pathway: 是否使用通路投影器

        Returns:
            (N, out_dim) 节点嵌入
        """
        if n_compounds is None:
            n_compounds = self.n_compounds

        comp_x = x[:n_compounds]
        prot_x = x[n_compounds:]
        comp_x_actual = comp_x[:, :self.comp_feat_dim]
        prot_esm = prot_x[:, :self.prot_esm_dim]
        prot_h = self.prot_feat_proj(prot_esm)

        if use_pathway and self.pathway_proj is not None and self.n_pathways > 0:
            prot_pathway = prot_x[:, self.prot_esm_dim:self.prot_esm_dim + self.n_pathways]
            prot_h = prot_h + self.pathway_proj(prot_pathway)

        comp_h = self.comp_proj(comp_x_actual)
        h = torch.cat([comp_h, prot_h], dim=0)

        prot_indices = slice(n_compounds, h.shape[0])
        h[prot_indices] = self.prot_dropout(h[prot_indices])

        for conv, norm, drop in zip(self.convs, self.norms, self.dropouts, strict=False):
            h_new = conv(h, edge_index)
            if h.shape[-1] == h_new.shape[-1]:
                h_new = h_new + h
            h = norm(h_new)
            h = F.relu(h)
            h = drop(h)
        return h

    def decode(self, comp_emb: torch.Tensor, prot_emb: torch.Tensor,
               prot_residue_indices: torch.Tensor | None = None) -> torch.Tensor:
        """解码化合物-蛋白交互分数。

        Args:
            comp_emb: (*, out_dim) 化合物嵌入
            prot_emb: (*, out_dim) 蛋白嵌入
            prot_residue_indices: 可选，蛋白全局索引，用于 residue_bilinear 解码器

        Returns:
            (*,) 预测 logits
        """
        if self.decoder_type == "residue_bilinear":
            return self.decoder(comp_emb, prot_emb, prot_residue_indices)
        return self.decoder(comp_emb, prot_emb)

    def set_residue_features(self, embeddings: torch.Tensor, offsets: torch.Tensor,
                             lengths: torch.Tensor, prot_to_residue_idx: torch.Tensor,
                             max_len: int = 1024, residue_device: str = "cpu") -> None:
        """注册蛋白残基级 ESM-2 特征（供 residue_bilinear 解码器使用）。

        Args:
            embeddings: (total_residues, residue_dim) 扁平化残基嵌入
            offsets: (n_proteins+1,) 每个蛋白在 embeddings 中的起始偏移
            lengths: (n_proteins,) 每个蛋白的残基数
            prot_to_residue_idx: (n_graph_proteins,) 图蛋白索引 -> residue 文件索引
            max_len: 单个蛋白最大残基数（截断/填充用）
            residue_device: 残基张量驻留设备，默认 "cpu"
        """
        self.residue_max_len = max_len
        if isinstance(self.decoder, ResidueAwareBilinearDecoder):
            self.decoder.register_residue_buffers(
                embeddings, offsets, lengths, max_len=max_len,
                prot_to_residue_idx=prot_to_residue_idx,
                residue_device=residue_device,
            )

    def encode_compound(self, x: torch.Tensor) -> torch.Tensor:
        """编码化合物特征（无图结构，仅投影+卷积）"""
        x_actual = x[:, :self.comp_feat_dim]
        h = self.comp_proj(x_actual)
        empty_edge = torch.zeros((2, 0), dtype=torch.long, device=h.device)
        for conv, norm, drop in zip(self.convs, self.norms, self.dropouts, strict=False):
            h_new = conv(h, empty_edge)
            if h.shape[-1] == h_new.shape[-1]:
                h_new = h_new + h
            h = norm(h_new)
            h = F.relu(h)
            h = drop(h)
        return h

    def predict_phenotype(self, compound_embeds: torch.Tensor) -> torch.Tensor:
        """预测化合物的铁死亡表型"""
        return self.pheno_head(compound_embeds)

    def free_residue_features(self) -> None:
        """释放 decoder 中的残基级 ESM-2 特征内存。

        训练主流程在 SAGE 训练结束后、HGT 训练初始化前调用，
        避免两份残基张量同时驻留导致 CPU OOM。
        """
        if isinstance(self.decoder, ResidueAwareBilinearDecoder):
            self.decoder.free_residue_features()
