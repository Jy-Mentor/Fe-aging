"""RGCN 异质图编码器 + 可插拔解码器

RGCN (Relational Graph Convolution) 替代 HGT，去除门控+多头注意力，
换回关系特定权重聚合。在已有 SAGE 基线 (AUPR 0.80) 基础上加边类型感知，
改动最小，风险最低。

特性:
  - 关系特定权重聚合，无门控，无多头注意力
  - 异质图 → 同质图统一转换，保留所有边类型语义
  - 支持 MLP / Dot / Bilinear / ResidueAwareBilinear 解码器切换
  - 多任务联合训练（铁死亡表型分类头）
  - 支持疾病节点嵌入（四模态异质图）

参考:
  - Schlichtkrull et al. (2018) "Modeling Relational Data with Graph Convolutional Networks", ESWC
  - Hamilton et al. (2017) "GraphSAGE", NeurIPS
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

from .decoders import MLPDecoder, DotProductDecoder, BilinearDecoder, ResidueAwareBilinearDecoder

logger = logging.getLogger(__name__)

_PHENO_HEAD_DROPOUT = 0.3
_TEMPERATURE = 5.0


class RGCNLinkPredictor(nn.Module):
    """RGCN 异质图编码器 + 可插拔解码器 + 表型分类头

    将 HGT 的异质图输入 (x_dict, edge_index_dict) 转换为统一节点矩阵，
    使用 RGCNConv 按边类型分参数聚合，再拆回 x_dict 输出。
    """

    def __init__(self, hidden_dim: int = 128, out_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.5,
                 metadata=None, compound_feat_dim: int = 200,
                 node_feat_dims: dict[str, int] | None = None,
                 pheno_head_dropout: float = _PHENO_HEAD_DROPOUT,
                 temperature: float = _TEMPERATURE,
                 decoder_type: str = "mlp",
                 decoder_init_scheme: str = "xavier",
                 decoder_final_bias_init: float = -0.5,
                 decoder_max_residue_batch: int = 2):
        """初始化 RGCN 链接预测模型。

        Args:
            hidden_dim: 隐藏层维度（建议 128，SAGE 用 64 后翻倍）。
            out_dim: 输出嵌入维度。
            num_layers: RGCNConv 层数（2 层足够，参数更多）。
            dropout: Dropout 概率。
            metadata: PyG 异质图元数据 (node_types, edge_types)。
            compound_feat_dim: 化合物输入特征维度。
            node_feat_dims: 各类节点特征维度，如 {"protein": 640, "pathway_count": N, "disease_count": D}。
            pheno_head_dropout: 表型头 Dropout。
            temperature: 解码温度系数 T。
            decoder_type: 解码器类型：mlp / dot / bilinear / residue_bilinear。
            decoder_init_scheme: residue_bilinear 初始化策略 (xavier/kaiming/orthogonal)。
            decoder_final_bias_init: residue_bilinear 最终打分层偏置初始值。
            decoder_max_residue_batch: residue_bilinear 残基路径前向分块大小。
        """
        super().__init__()
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
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

        # 构建边类型 → 整数索引映射（包含 metadata 全部边类型 + 动态反向边）
        self._build_relation_mapping(metadata)

        # RGCNConv 层 + 层归一化
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(RGCNConv(
                hidden_dim, hidden_dim,
                num_relations=self.num_relations,
            ))
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

    def _build_relation_mapping(self, metadata):
        """从 metadata 构建边类型 → 整数索引映射。

        包含全部 metadata 边类型 + 子图采样动态添加的反向边，
        确保训练时子图边类型全覆盖。
        """
        if metadata is None:
            self.node_types = []
            self.edge_type_to_idx = {}
            self.num_relations = 0
            return

        node_types, edge_types = metadata
        self.node_types = node_types

        self.edge_type_to_idx = {}
        idx = 0
        for et in edge_types:
            self.edge_type_to_idx[et] = idx
            idx += 1

        # 子图采样动态添加的反向边
        dynamic_rev_edges = [
            ("protein", "rev_interacts", "compound"),
            ("protein", "rev_ppi", "protein"),
        ]
        for et in dynamic_rev_edges:
            if et not in self.edge_type_to_idx:
                self.edge_type_to_idx[et] = idx
                idx += 1

        self.num_relations = idx
        logger.info(
            f"RGCN 边类型映射: {self.num_relations} 种关系 "
            f"(metadata={len(edge_types)}, dynamic={idx - len(edge_types)})"
        )

    def _hetero_to_homo(self, x_dict, edge_index_dict):
        """将异质图输入 (x_dict, edge_index_dict) 转换为同质图格式。

        Returns:
            x: (total_nodes, hidden_dim) 统一节点特征
            edge_index: (2, total_edges) 统一边索引
            edge_type: (total_edges,) 边类型整数索引
            node_offsets: {node_type: global_start_index}
        """
        # 1. 计算各节点类型偏移量
        node_offsets = {}
        total_nodes = 0
        for nt in self.node_types:
            node_offsets[nt] = total_nodes
            if nt in x_dict:
                total_nodes += x_dict[nt].shape[0]

        # 2. 拼接统一节点特征
        x_parts = []
        for nt in self.node_types:
            if nt in x_dict and x_dict[nt].shape[0] > 0:
                x_parts.append(x_dict[nt])
        if x_parts:
            x = torch.cat(x_parts, dim=0)
        else:
            x = torch.empty(0, self.hidden_dim, device=next(iter(x_dict.values())).device)

        # 3. 构建统一边索引 + 边类型
        edge_list = []
        type_list = []
        for edge_key, ei in edge_index_dict.items():
            if ei.shape[1] == 0 or edge_key not in self.edge_type_to_idx:
                continue
            src_type, _rel, dst_type = edge_key
            src_off = node_offsets.get(src_type, 0)
            dst_off = node_offsets.get(dst_type, 0)
            src_global = ei[0] + src_off
            dst_global = ei[1] + dst_off
            edge_list.append(torch.stack([src_global, dst_global], dim=0))
            rel_idx = self.edge_type_to_idx[edge_key]
            type_list.append(torch.full((ei.shape[1],), rel_idx, device=ei.device, dtype=torch.long))

        if edge_list:
            edge_index = torch.cat(edge_list, dim=1)
            edge_type = torch.cat(type_list)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=x.device)
            edge_type = torch.zeros(0, dtype=torch.long, device=x.device)

        return x, edge_index, edge_type, node_offsets

    def encode_compound(self, x_comp: torch.Tensor) -> torch.Tensor:
        """编码化合物特征为输出空间嵌入（无图消息传递）。"""
        h = self.comp_proj(x_comp)
        return self.out_proj(h)

    def forward(self, x_dict, edge_index_dict, use_pathway: bool = True):
        """前向传播：投影 → 异质转同质 → RGCN 卷积 → 拆回 x_dict。

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

        # 异质图 → 同质图转换
        x, edge_index, edge_type, node_offsets = self._hetero_to_homo(x_dict, edge_index_dict)

        # RGCN 卷积层
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index, edge_type)
            h = h + x  # 残差连接
            h = norm(h)
            h = F.relu(h)
            h = self.dropout(h)
            x = h

        # 同质图 → 异质图拆回
        out_dict = {}
        for nt in self.node_types:
            start = node_offsets.get(nt, 0)
            if nt in x_dict:
                end = start + x_dict[nt].shape[0]
                out_dict[nt] = self.out_proj(x[start:end])
            else:
                out_dict[nt] = torch.empty(0, self.out_dim, device=x.device)

        return out_dict

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