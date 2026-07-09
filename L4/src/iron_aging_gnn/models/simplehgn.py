"""SimpleHGN 异质图编码器 + 可插拔解码器

SimpleHGN 是 HGT 的「外科手术式」替代方案：
  - 保留 HGT 的异质图注意力能力
  - 将 HGT 的类型特定参数矩阵替换为边类型嵌入（edge-type embedding）
  - 去掉门控（Gate）和复杂的消息归一化，数值稳定性好

核心公式（边类型感知注意力）：
  e_{ij} = (W_q h_i + r_{type}) · (W_k h_j + r_{type})

其中 r_{type} 是可学习的边类型向量，所有边类型共享投影矩阵 W_q, W_k。

实现方式：
  - HeteroConv + GATv2Conv(edge_dim=...) 替代 HGTConv
  - 边类型嵌入作为 edge_attr 传入 GATv2Conv

参考:
  - Hu et al. (2020) "Heterogeneous Graph Transformer", WWW
  - Brody et al. (2022) "How Attentive are Graph Attention Networks?", ICLR
  - Lv et al. (2021) "Are we really making much progress? Revisiting, benchmarking,
    and refining heterogeneous graph neural networks", KDD
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv

from .decoders import MLPDecoder, DotProductDecoder, BilinearDecoder, ResidueAwareBilinearDecoder

logger = logging.getLogger(__name__)

_PHENO_HEAD_DROPOUT = 0.3
_TEMPERATURE = 5.0


class SimpleHGNLinkPredictor(nn.Module):
    """SimpleHGN 异质图编码器 + 可插拔解码器 + 表型分类头

    用 HeteroConv + GATv2Conv(edge_dim) 替代 HGTConv，
    边类型参数从独立矩阵转为共享投影 + 边类型嵌入向量。
    无门控，无元路径注意力，稳定性远好于 HGT。
    """

    def __init__(self, hidden_dim: int = 128, out_dim: int = 128,
                 num_layers: int = 2, num_heads: int = 2, dropout: float = 0.5,
                 metadata=None, compound_feat_dim: int = 200,
                 node_feat_dims: dict[str, int] | None = None,
                 pheno_head_dropout: float = _PHENO_HEAD_DROPOUT,
                 temperature: float = _TEMPERATURE,
                 decoder_type: str = "mlp",
                 decoder_init_scheme: str = "xavier",
                 decoder_final_bias_init: float = -0.5,
                 decoder_max_residue_batch: int = 2):
        """初始化 SimpleHGN 链接预测模型。

        Args:
            hidden_dim: 隐藏层维度。
            out_dim: 输出嵌入维度。
            num_layers: HeteroConv 层数。
            num_heads: GATv2Conv 注意力头数。
            dropout: Dropout 概率。
            metadata: PyG 异质图元数据 (node_types, edge_types)。
            compound_feat_dim: 化合物输入特征维度。
            node_feat_dims: 各类节点特征维度。
            pheno_head_dropout: 表型头 Dropout。
            temperature: 解码温度系数 T。
            decoder_type: 解码器类型：mlp / dot / bilinear / residue_bilinear。
            decoder_init_scheme: 解码器权重初始化方案（仅在 residue_bilinear 时有效）。
            decoder_final_bias_init: 解码器末层偏置初始值（仅在 residue_bilinear 时有效）。
            decoder_max_residue_batch: 残基注意力最大 batch 大小（仅在 residue_bilinear 时有效）。
        """
        super().__init__()
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.decoder_type = decoder_type
        self.temperature = temperature

        # 通路嵌入
        n_pathways = node_feat_dims.get("pathway_count", 1) if node_feat_dims else 1
        self.pathway_embed = nn.Embedding(max(n_pathways, 1), hidden_dim)

        # 疾病节点嵌入
        n_diseases = node_feat_dims.get("disease_count", 0) if node_feat_dims else 0
        self.n_diseases = n_diseases
        self.disease_embed = nn.Embedding(max(n_diseases, 1), hidden_dim) if n_diseases > 0 else None

        self.comp_proj = nn.Sequential(
            nn.Linear(compound_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        prot_in_dim = node_feat_dims.get("protein", 640) if node_feat_dims else 640
        self.prot_in_dim = prot_in_dim
        self.prot_proj = nn.Sequential(
            nn.Linear(prot_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 构建边类型映射（包含 metadata 全部边类型 + 动态反向边）
        self._build_edge_type_mapping(metadata)

        # 边类型嵌入：每个边类型一个可学习向量，维度 = 每头维度
        edge_emb_dim = hidden_dim // num_heads
        self.edge_type_embed = nn.Embedding(self.num_relations, edge_emb_dim)

        # HeteroConv + GATv2Conv 层
        per_head_dim = hidden_dim // num_heads
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for et in self.all_edge_types:
                conv_dict[et] = GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=per_head_dim,
                    heads=num_heads,
                    edge_dim=edge_emb_dim,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=False,
                )
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        self.out_proj = nn.Identity() if hidden_dim == out_dim else nn.Linear(hidden_dim, out_dim, bias=False)

        # 可插拔解码器
        if decoder_type == "mlp":
            self.decoder = MLPDecoder(out_dim, hidden_dim=64, dropout=pheno_head_dropout)
        elif decoder_type == "dot":
            self.decoder = DotProductDecoder()
        elif decoder_type == "bilinear":
            self.decoder = BilinearDecoder(out_dim, rank=64)
        elif decoder_type == "residue_bilinear":
            self.decoder = ResidueAwareBilinearDecoder(
                comp_dim=out_dim, residue_dim=640, rank=64, hidden_dim=128,
                dropout=pheno_head_dropout,
                max_residue_batch=decoder_max_residue_batch,
                init_scheme=decoder_init_scheme,
                final_bias_init=decoder_final_bias_init,
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

    def _build_edge_type_mapping(self, metadata):
        """构建边类型 → 整数索引映射 + 全边类型列表（含动态反向边）。

        HeteroConv 需要为每个可能出现的边类型预创建 GATv2Conv 实例。
        """
        if metadata is None:
            self.node_types = []
            self.all_edge_types = []
            self.edge_type_to_idx = {}
            self.num_relations = 0
            return

        node_types, edge_types = metadata
        self.node_types = node_types

        # 按 metadata 顺序添加，再附加动态反向边
        # 用 set 去重的同时保持顺序
        seen = set()
        ordered = []
        for et in edge_types:
            if et not in seen:
                seen.add(et)
                ordered.append(et)
        dynamic_rev_edges = [
            ("protein", "rev_interacts", "compound"),
            ("protein", "rev_ppi", "protein"),
        ]
        for et in dynamic_rev_edges:
            if et not in seen:
                seen.add(et)
                ordered.append(et)

        self.all_edge_types = ordered
        self.edge_type_to_idx = {et: i for i, et in enumerate(ordered)}
        self.num_relations = len(ordered)
        logger.info(
            f"SimpleHGN 边类型映射: {self.num_relations} 种关系 "
            f"(metadata={len(edge_types)}, dynamic={self.num_relations - len(edge_types)})"
        )

    def encode_compound(self, x_comp: torch.Tensor) -> torch.Tensor:
        """编码化合物特征为输出空间嵌入（无图消息传递）。"""
        h = self.comp_proj(x_comp)
        return self.out_proj(h)

    def forward(self, x_dict, edge_index_dict, use_pathway: bool = True):
        """前向传播：投影 → 边类型嵌入 → HeteroConv 卷积 → 输出投影。

        Args:
            x_dict: {node_type: (N_i, feat_dim)} 节点特征字典
            edge_index_dict: {edge_type: (2, E)} 边索引字典
            use_pathway: 是否使用通路嵌入

        Returns:
            x_dict: {node_type: (N_i, out_dim)} 输出嵌入字典
        """
        x_dict = {k: v.clone() for k, v in x_dict.items()}

        # 投影到统一隐藏空间
        if "compound" in x_dict:
            x_dict["compound"] = self.comp_proj(x_dict["compound"])
        if "protein" in x_dict:
            actual_dim = x_dict["protein"].shape[-1]
            if actual_dim < self.prot_in_dim:
                raise ValueError(
                    f"蛋白输入维度 {actual_dim} < prot_in_dim {self.prot_in_dim}"
                )
            if actual_dim > self.prot_in_dim:
                x_dict["protein"] = self.prot_proj(x_dict["protein"][:, :self.prot_in_dim])
            else:
                x_dict["protein"] = self.prot_proj(x_dict["protein"])
        if "disease" in x_dict and self.disease_embed is not None:
            x_dict["disease"] = self.disease_embed(x_dict["disease"].squeeze(-1).long())
        if "pathway" in x_dict:
            if not use_pathway:
                x_dict["pathway"] = torch.zeros(
                    x_dict["pathway"].shape[0], self.hidden_dim,
                    device=x_dict["pathway"].device
                )
            elif x_dict["pathway"].shape[-1] != self.hidden_dim:
                x_dict["pathway"] = self.pathway_embed(
                    x_dict["pathway"].squeeze(-1).long().clamp(
                        0, self.pathway_embed.num_embeddings - 1
                    )
                )

        # 构建边特征字典（边类型嵌入，每条边分配其所属类型的嵌入向量）
        edge_attr_dict = {}
        for edge_key, ei in edge_index_dict.items():
            if ei.shape[1] == 0 or edge_key not in self.edge_type_to_idx:
                continue
            rel_idx = self.edge_type_to_idx[edge_key]
            edge_attr = self.edge_type_embed(
                torch.full((ei.shape[1],), rel_idx, device=ei.device, dtype=torch.long)
            )
            edge_attr_dict[edge_key] = edge_attr

        # HeteroConv + GATv2Conv 卷积层（无门控，直接残差连接）
        for conv, norm in zip(self.convs, self.norms):
            out = conv(x_dict, edge_index_dict, edge_attr_dict)
            # 补全无入边的节点类型
            for nt in x_dict:
                if nt not in out:
                    out[nt] = x_dict[nt]
            # 残差连接 + LayerNorm + ReLU
            for nt in out:
                if nt in x_dict and x_dict[nt].shape == out[nt].shape:
                    out[nt] = out[nt] + x_dict[nt]
            x_dict = {k: norm(v) for k, v in out.items()}
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = {k: self.dropout(v) for k, v in x_dict.items()}

        # 输出投影
        for nt in ["compound", "protein"]:
            if nt in x_dict:
                x_dict[nt] = self.out_proj(x_dict[nt])

        return x_dict

    def decode(self, comp_emb: torch.Tensor, prot_emb: torch.Tensor,
               prot_residue_indices: torch.Tensor | None = None) -> torch.Tensor:
        """解码化合物-蛋白交互分数。"""
        with torch.amp.autocast('cuda', enabled=False):
            comp_emb = comp_emb.float()
            prot_emb = prot_emb.float()
            if prot_residue_indices is not None:
                prot_residue_indices = prot_residue_indices.long()
            if self.decoder_type == "residue_bilinear":
                return self.decoder(comp_emb, prot_emb, prot_residue_indices)
            return self.decoder(comp_emb, prot_emb)

    def set_residue_features(self, embeddings: torch.Tensor, offsets: torch.Tensor,
                             lengths: torch.Tensor, prot_to_residue_idx: torch.Tensor,
                             max_len: int = 1024, residue_device: str = "cpu") -> None:
        """注册蛋白残基级 ESM-2 特征。"""
        self.residue_max_len = max_len
        if isinstance(self.decoder, ResidueAwareBilinearDecoder):
            self.decoder.register_residue_buffers(
                embeddings, offsets, lengths, max_len=max_len,
                prot_to_residue_idx=prot_to_residue_idx,
                residue_device=residue_device,
            )

    def predict_phenotype(self, compound_embeds: torch.Tensor) -> torch.Tensor:
        """预测化合物的铁死亡表型。"""
        with torch.amp.autocast('cuda', enabled=False):
            return self.pheno_head(compound_embeds.float())

    def free_residue_features(self) -> None:
        """释放 decoder 中的残基级 ESM-2 特征内存。"""
        if isinstance(self.decoder, ResidueAwareBilinearDecoder):
            self.decoder.free_residue_features()