"""HGT 异质图编码器 + 可插拔解码器

特性:
  - 节点自适应门控，缓解过平滑
  - 多任务联合训练（铁死亡表型分类头）
  - 支持疾病节点嵌入（四模态异质图）
  - 支持 MLP / Dot / Bilinear / ResidueAwareBilinear 解码器切换

参考:
  - Hu et al. (2020) "Heterogeneous Graph Transformer", WWW
  - GraphBAN (Nature Communications, 2025): 双线性注意力用于化合物-蛋白交互建模
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv

from .decoders import MLPDecoder, DotProductDecoder, BilinearDecoder, ResidueAwareBilinearDecoder

logger = logging.getLogger(__name__)

_PHENO_HEAD_DROPOUT = 0.3         # 表型分类头 Dropout
_TEMPERATURE = 5.0                # 温度参数 T（固定，不参与梯度更新）


class HGTLinkPredictor(nn.Module):
    """HGT 异质图编码器 + 可插拔解码器 + 表型分类头"""

    def __init__(self, hidden_dim: int = 64, out_dim: int = 64,
                 num_heads: int = 2, num_layers: int = 2, dropout: float = 0.5,
                 metadata=None, compound_feat_dim: int = 200,
                 node_feat_dims: dict[str, int] | None = None,
                 pheno_head_dropout: float = _PHENO_HEAD_DROPOUT,
                 temperature: float = _TEMPERATURE,
                 decoder_type: str = "mlp"):
        """初始化 HGT 链接预测模型。

        Args:
            hidden_dim: 隐藏层维度。
            out_dim: 输出嵌入维度。
            num_heads: HGTConv 注意力头数。
            num_layers: HGTConv 层数。
            dropout: Dropout 概率。
            metadata: PyG 异质图元数据 (node_types, edge_types)。
            compound_feat_dim: 化合物输入特征维度。
            node_feat_dims: 各类节点特征维度，如 {"protein": 640, "pathway_count": N, "disease_count": D}。
            pheno_head_dropout: 表型头 Dropout。
            temperature: 解码温度系数 T。
            decoder_type: 解码器类型：mlp / dot / bilinear / residue_bilinear。
        """
        super().__init__()
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.decoder_type = decoder_type

        self.temperature = temperature

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

        self.convs = nn.ModuleList()
        self.gates = nn.ModuleList()
        if metadata:
            node_types, edge_types = metadata
            for _ in range(num_layers):
                self.convs.append(HGTConv(
                    dict.fromkeys(node_types, hidden_dim),
                    hidden_dim, metadata,
                    heads=num_heads,
                ))
                gate = nn.Linear(hidden_dim, 1)
                nn.init.constant_(gate.bias, 2.0)
                self.gates.append(gate)

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
                comp_dim=out_dim, residue_dim=640, rank=64, hidden_dim=128, dropout=pheno_head_dropout
            )
        else:
            raise ValueError(f"不支持的 decoder_type: {decoder_type}")

        self.dropout = nn.Dropout(dropout)

        # 铁死亡表型分类头
        self.pheno_head = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(pheno_head_dropout),
            nn.Linear(64, 1),
        )

    def encode_compound(self, x_comp: torch.Tensor) -> torch.Tensor:
        """编码化合物特征为输出空间嵌入（无图消息传递）。"""
        h = self.comp_proj(x_comp)
        return self.out_proj(h)

    def forward(self, x_dict, edge_index_dict, use_pathway: bool = True):
        """前向传播：执行 HGT 卷积、门控聚合与输出投影。

        HGT 将通路信息作为独立 pathway 节点处理（异质图结构），
        因此蛋白节点仅需 ESM-2 特征。输入特征维度必须与 prot_in_dim 严格一致，
        禁止截断或填充，不匹配时直接抛出 ValueError。
        """
        x_dict = {k: v.clone() for k, v in x_dict.items()}

        if "compound" in x_dict:
            x_dict["compound"] = self.comp_proj(x_dict["compound"])
        if "protein" in x_dict:
            actual_dim = x_dict["protein"].shape[-1]
            if actual_dim < self.prot_in_dim:
                raise ValueError(
                    f"蛋白输入维度 {actual_dim} < prot_in_dim {self.prot_in_dim}，"
                    f"特征维度不足，无法提取 ESM-2 嵌入"
                )
            if actual_dim > self.prot_in_dim:
                logger.debug(
                    f"蛋白输入维度 {actual_dim} > prot_in_dim {self.prot_in_dim}，"
                    f"取前 {self.prot_in_dim} 维作为 ESM-2 嵌入（其余为通路等附加特征，由异质图结构传递）"
                )
                x_dict["protein"] = self.prot_proj(x_dict["protein"][:, :self.prot_in_dim])
            else:
                x_dict["protein"] = self.prot_proj(x_dict["protein"])
        if "disease" in x_dict and self.disease_embed is not None:
            x_dict["disease"] = self.disease_embed(x_dict["disease"].squeeze(-1).long())
        if "pathway" in x_dict:
            if not use_pathway:
                x_dict["pathway"] = torch.zeros(x_dict["pathway"].shape[0], self.hidden_dim,
                                                device=x_dict["pathway"].device)
            elif x_dict["pathway"].shape[-1] != self.hidden_dim:
                x_dict["pathway"] = self.pathway_embed(
                    x_dict["pathway"].squeeze(-1).long().clamp(0, self.pathway_embed.num_embeddings - 1))

        for layer_idx, conv in enumerate(self.convs):
            out = conv(x_dict, edge_index_dict)
            for nt in x_dict:
                if nt not in out:
                    out[nt] = x_dict[nt]
            # 节点自适应门控，缓解过平滑
            gate = self.gates[layer_idx]
            for nt in out:
                if nt in x_dict and x_dict[nt].shape == out[nt].shape:
                    g = torch.sigmoid(gate(x_dict[nt]))
                    out[nt] = g * out[nt] + (1 - g) * x_dict[nt]
            x_dict = out
            x_dict = {k: self.dropout(v) for k, v in x_dict.items()}

        # 仅对 compound 和 protein 执行 out_proj
        for nt in ["compound", "protein"]:
            if nt in x_dict:
                x_dict[nt] = self.out_proj(x_dict[nt])

        return x_dict

    def decode(self, comp_emb: torch.Tensor, prot_emb: torch.Tensor,
               prot_residue_indices: torch.Tensor | None = None) -> torch.Tensor:
        """解码化合物-蛋白交互分数。

        Args:
            comp_emb: 化合物嵌入。
            prot_emb: 蛋白嵌入。
            prot_residue_indices: 可选，蛋白全局索引，用于 residue_bilinear 解码器。

        Returns:
            预测 logits（未经过 sigmoid）。
        """
        # AMP autocast 下 decoder 权重为 float32，输入可能为 float16，
        # 显式转 float32 并关闭 autocast，避免 mat1/mat2 dtype 不匹配。
        with torch.cuda.amp.autocast(enabled=False):
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

    def predict_phenotype(self, compound_embeds: torch.Tensor) -> torch.Tensor:
        """预测化合物的铁死亡表型（二分类：是否铁死亡调节剂）"""
        # 表型头权重为 float32，AMP 下输入可能为 float16，需统一 dtype。
        with torch.cuda.amp.autocast(enabled=False):
            return self.pheno_head(compound_embeds.float())

    def free_residue_features(self) -> None:
        """释放 decoder 中的残基级 ESM-2 特征内存。

        训练主流程在 SAGE 训练结束后、HGT 训练初始化前调用，
        避免两份残基张量同时驻留导致 CPU OOM。
        """
        if isinstance(self.decoder, ResidueAwareBilinearDecoder):
            self.decoder.free_residue_features()
