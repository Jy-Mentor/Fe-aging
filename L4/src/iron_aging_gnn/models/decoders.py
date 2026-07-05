"""DTI/CPI 解码器集合

参考:
  - GraphBAN (Nature Communications, 2025): 双线性注意力用于化合物-蛋白交互建模
"""

from __future__ import annotations

import gc
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MLPDecoder(nn.Module):
    """MLP 解码器：拼接化合物与蛋白嵌入后预测交互分数。"""

    def __init__(self, out_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, comp_emb: torch.Tensor, prot_emb: torch.Tensor) -> torch.Tensor:
        """Args: comp_emb (N, d), prot_emb (N, d). Returns: (N,) logits."""
        return self.net(torch.cat([comp_emb, prot_emb], dim=-1)).squeeze(-1)


class DotProductDecoder(nn.Module):
    """点积解码器：计算化合物与蛋白嵌入的内积。"""

    def forward(self, comp_emb: torch.Tensor, prot_emb: torch.Tensor) -> torch.Tensor:
        """Args: comp_emb (N, d), prot_emb (N, d). Returns: (N,) logits."""
        return (comp_emb * prot_emb).sum(dim=-1)


class BilinearDecoder(nn.Module):
    """低秩双线性解码器。

    对每对 (c, p)，计算:
        score = sum_k (u_k^T c) * (v_k^T p) + b
    其中 k=1..rank。rank=1 时退化为可学习投影后的点积；
    rank>1 时捕捉多通道交互，比 MLP 更轻量，比点积更表达力强。

    Args:
        out_dim: 嵌入维度
        rank: 低秩维度，默认 64
    """

    def __init__(self, out_dim: int, rank: int = 64):
        super().__init__()
        self.U = nn.Linear(out_dim, rank, bias=False)
        self.V = nn.Linear(out_dim, rank, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, comp_emb: torch.Tensor, prot_emb: torch.Tensor) -> torch.Tensor:
        """Args: comp_emb (N, d), prot_emb (N, d). Returns: (N,) logits."""
        cu = self.U(comp_emb)  # (N, rank)
        pv = self.V(prot_emb)  # (N, rank)
        scores = (cu * pv).sum(dim=-1)  # (N,)
        return scores + self.bias


class ResidueAwareBilinearDecoder(nn.Module):
    """残基感知双线性注意力解码器。

    输入为化合物全局嵌入 [N, d_comp] 与蛋白逐残基特征 [N, L, d_residue]，
    通过化合物嵌入作为 query，对蛋白残基做注意力加权，实现化合物-残基层面交互。

    基于 GraphBAN 双线性注意力思想，适配 packed 残基存储格式。

    Args:
        comp_dim: 化合物嵌入维度
        residue_dim: 残基特征维度（ESM-2 为 640）
        rank: 双线性低秩维度
        hidden_dim: 残基聚合后 MLP 隐藏维度
        dropout: Dropout 比率
        max_len: 单个蛋白最大残基数
        max_residue_batch: 分块前向的最大 batch 大小（防止 OOM）
    """

    def __init__(self, comp_dim: int, residue_dim: int = 640, rank: int = 64,
                 hidden_dim: int = 128, dropout: float = 0.3, max_len: int = 512,
                 max_residue_batch: int = 4):
        super().__init__()
        self.comp_dim = comp_dim
        self.residue_dim = residue_dim
        self.rank = rank
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.max_residue_batch = max_residue_batch

        # 化合物 query 投影到双线性秩空间
        self.U = nn.Linear(comp_dim, rank, bias=False)
        # 残基 key/value 投影到双线性秩空间
        self.V = nn.Linear(residue_dim, rank, bias=False)
        # 蛋白全局嵌入（GNN 输出）的投影，用于无残基索引时的快速双线性打分
        self.W = nn.Linear(comp_dim, rank, bias=False)

        # 残基聚合后非线性变换
        self.residue_agg = nn.Sequential(
            nn.Linear(residue_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 最终打分 MLP：聚合残基特征 + 最大双线性响应
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # 大残基张量以普通属性保存（不作为 nn buffer），避免 model.to(device) 将其搬到 GPU。
        self.residue_device = "cpu"
        self._residue_embeddings = torch.zeros(1, residue_dim)
        self._residue_offsets = torch.zeros(2, dtype=torch.long)
        self._residue_lengths = torch.zeros(1, dtype=torch.long)
        self._residue_is_mmap = False
        self._unknown_idx: int = -1
        self.register_buffer("_prot_to_residue_idx", torch.zeros(1, dtype=torch.long))

    def register_residue_buffers(self, embeddings: torch.Tensor, offsets: torch.Tensor,
                                 lengths: torch.Tensor, max_len: int = 1024,
                                 prot_to_residue_idx: torch.Tensor | None = None,
                                 residue_device: str = "cpu") -> None:
        """注册 packed 格式残基特征。

        v43: mmap 张量禁止 torch.cat 物化。unknown placeholder 通过独立索引
        _unknown_idx 标记，在 _gather_residues 中特殊处理。

        Args:
            embeddings: (total_residues, residue_dim) mmap 或普通张量
            offsets: (n_proteins+1,)
            lengths: (n_proteins,)
            max_len: 单个蛋白最大截断长度
            prot_to_residue_idx: (n_graph_proteins,) 图蛋白索引 -> residue 文件索引
            residue_device: 残基张量驻留设备，默认 "cpu"
        """
        self.max_len = max_len
        self.residue_device = residue_device

        is_mmap = getattr(embeddings, "is_mmap", False)
        if is_mmap:
            logger.info("检测到 mmap 残基张量，跳过 torch.cat 物化，"
                        "unknown placeholder 使用独立索引处理")
            self._residue_is_mmap = True
            self._residue_embeddings = embeddings
            self._residue_offsets = offsets
            self._residue_lengths = lengths
            self._unknown_idx = offsets.shape[0] - 1
        else:
            self._residue_is_mmap = False
            unknown_emb = torch.zeros(1, embeddings.shape[1], dtype=embeddings.dtype)
            embeddings = torch.cat([embeddings, unknown_emb], dim=0).to(residue_device)
            unknown_start = offsets[-1]
            unknown_end = unknown_start + 1
            offsets = torch.cat(
                [offsets, torch.tensor([unknown_start, unknown_end], dtype=offsets.dtype)],
                dim=0,
            ).to(residue_device)
            lengths = torch.cat(
                [lengths, torch.ones(1, dtype=lengths.dtype)],
                dim=0,
            ).to(residue_device)
            self._residue_embeddings = embeddings
            self._residue_offsets = offsets
            self._residue_lengths = lengths
            self._unknown_idx = offsets.shape[0] - 1

        if prot_to_residue_idx is None:
            prot_to_residue_idx = torch.zeros(1, dtype=torch.long)
        self._prot_to_residue_idx = prot_to_residue_idx.to(
            self._prot_to_residue_idx.device
        )

    def free_residue_features(self) -> None:
        """释放残基级 ESM-2 特征占用的 CPU 内存，避免大张量同时驻留导致 OOM。"""
        if hasattr(self, "_residue_embeddings"):
            self._residue_embeddings = torch.zeros(1, self.residue_dim)
        if hasattr(self, "_residue_offsets"):
            self._residue_offsets = torch.zeros(2, dtype=torch.long)
        if hasattr(self, "_residue_lengths"):
            self._residue_lengths = torch.zeros(1, dtype=torch.long)
        gc.collect()

    def _gather_residues(self, prot_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """根据蛋白全局索引收集残基特征并填充到 [N, L, d]。

        v43: mmap 模式下 unknown 蛋白直接返回全零嵌入，不触发 mmap 切片。

        Args:
            prot_indices: (N,) 蛋白全局索引（0-based，对应图蛋白节点）

        Returns:
            residue_feats: (N, max_len, residue_dim)
            residue_mask: (N, max_len) bool, True 表示有效残基
        """
        device = prot_indices.device
        n = prot_indices.shape[0]
        residue_feats = torch.zeros(
            n, self.max_len, self.residue_dim,
            dtype=self._residue_embeddings.dtype, device=device
        )
        residue_mask = torch.zeros(n, self.max_len, dtype=torch.bool, device=device)

        residue_idx = self._prot_to_residue_idx[prot_indices].cpu()
        n_total = self._residue_lengths.shape[0]

        # 检测 -1 索引（对应无残基特征的蛋白），标记为 unknown
        missing_mask = residue_idx < 0
        if missing_mask.any():
            missing_count = missing_mask.sum().item()
            logger.warning(
                "_gather_residues: %d/%d 蛋白索引无残基特征映射（值为 -1），"
                "已回退到 unknown placeholder",
                missing_count, residue_idx.shape[0],
            )
        residue_idx = residue_idx.clamp(0, n_total - 1)

        # 标记 unknown 蛋白（-1 或 _unknown_idx），跳过 mmap 切片
        unknown_mask = missing_mask | (residue_idx == self._unknown_idx)

        lengths = self._residue_lengths[residue_idx].clamp(1, self.max_len)
        offsets = self._residue_offsets[residue_idx]

        for i in range(n):
            if unknown_mask[i].item():
                continue
            start = int(offsets[i].item())
            length = int(lengths[i].item())
            end = start + length
            feats = self._residue_embeddings[start:end]
            if length > self.max_len:
                feats = feats[:self.max_len]
                length = self.max_len
            residue_feats[i, :length] = feats.to(device)
            residue_mask[i, :length] = True
        return residue_feats, residue_mask

    def _forward_chunk(self, comp_emb: torch.Tensor, prot_emb: torch.Tensor,
                       prot_indices: torch.Tensor) -> torch.Tensor:
        """对单个小批量残基样本执行前向（被 chunked forward 调用）。"""
        # 收集残基特征 [N, L, d_residue]
        residue_feats, residue_mask = self._gather_residues(prot_indices)

        # 双线性投影
        cu = self.U(comp_emb).unsqueeze(1)  # (N, 1, rank)
        pv = self.V(residue_feats)          # (N, L, rank)

        # 每残基双线性得分 [N, L]
        per_residue_scores = (cu * pv).sum(dim=-1)
        per_residue_scores = per_residue_scores.masked_fill(~residue_mask, -1e9)

        # 软注意力权重
        attn_weights = F.softmax(per_residue_scores, dim=-1)  # (N, L)
        attn_weights = attn_weights.masked_fill(~residue_mask, 0.0)

        # 加权聚合残基特征 [N, d_residue]
        agg_residue = (attn_weights.unsqueeze(-1) * residue_feats).sum(dim=1)
        agg_h = self.residue_agg(agg_residue)  # (N, hidden_dim)

        # 最大响应作为额外信号
        max_score = per_residue_scores.max(dim=-1).values.unsqueeze(-1)  # (N, 1)

        score = self.score_mlp(torch.cat([agg_h, max_score], dim=-1)).squeeze(-1)
        return score

    def forward(self, comp_emb: torch.Tensor, prot_emb: torch.Tensor,
                prot_indices: torch.Tensor | None = None) -> torch.Tensor:
        """前向传播。

        Args:
            comp_emb: (N, d_comp) 化合物嵌入
            prot_emb: (N, d_prot) 蛋白全局嵌入（作为辅助，可取自 GNN）
            prot_indices: (N,) 蛋白全局索引，必须提供以查找残基特征

        Returns:
            (N,) 预测 logits
        """
        if prot_indices is None:
            # 无残基索引时使用低秩双线性打分（共享 U 投影），替代全 pair-matrix 残基注意力
            cu = self.U(comp_emb)  # (N, rank)
            pw = self.W(prot_emb)  # (N, rank)
            return (cu * pw).sum(dim=-1)

        n = comp_emb.shape[0]
        if n == 0:
            return torch.zeros(0, device=comp_emb.device, dtype=comp_emb.dtype)

        # 按 max_residue_batch 分块前向，避免 N 过大导致 OOM
        if n <= self.max_residue_batch:
            return self._forward_chunk(comp_emb, prot_emb, prot_indices)

        outputs = []
        for i in range(0, n, self.max_residue_batch):
            outputs.append(self._forward_chunk(
                comp_emb[i:i + self.max_residue_batch],
                prot_emb[i:i + self.max_residue_batch],
                prot_indices[i:i + self.max_residue_batch],
            ))
        return torch.cat(outputs, dim=0)
