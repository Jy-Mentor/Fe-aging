"""DTI/CPI 解码器集合

参考:
  - Hadipour et al. (2025) "GraphBAN: An Inductive Graph-Based Approach for Enhanced Prediction of Compound-Protein Interactions", Nature Communications, DOI:10.1038/s41467-025-57536-9
"""

from __future__ import annotations

import gc
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MLPDecoder(nn.Module):
    """MLP 解码器：拼接化合物与蛋白嵌入后预测交互分数。"""

    def __init__(self, out_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        """初始化 MLP 解码器。

        Args:
            out_dim: 输入嵌入维度（化合物与蛋白拼接后维度为 2*out_dim）
            hidden_dim: 隐藏层维度
            dropout: Dropout 比率
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """受控初始化：隐藏层 Kaiming，输出层小增益，避免初始预测过度乐观。"""
        for i, m in enumerate(self.net.modules()):
            if isinstance(m, nn.Linear):
                if i == len(list(self.net.modules())) - 1:
                    # 输出层：小增益 + 零偏置
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_uniform_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

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
        """初始化低秩双线性解码器。

        Args:
            out_dim: 输入嵌入维度
            rank: 低秩交互维度
        """
        super().__init__()
        self.U = nn.Linear(out_dim, rank, bias=False)
        self.V = nn.Linear(out_dim, rank, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self) -> None:
        """双线性投影正交初始化，低秩交互更稳定。"""
        for linear in (self.U, self.V):
            if linear.weight is not None:
                nn.init.orthogonal_(linear.weight, gain=1.0)
        nn.init.zeros_(self.bias)

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

    基于 GraphBAN 双线性注意力思想（Hadipour et al., Nat. Commun. 2025, DOI:10.1038/s41467-025-57536-9），适配 packed 残基存储格式。

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
                 max_residue_batch: int = 4, init_scheme: str = "orthogonal",
                 final_bias_init: float = -0.5):
        """初始化残基感知双线性注意力解码器。

        Args:
            comp_dim: 化合物嵌入维度
            residue_dim: 蛋白残基特征维度（ESM-2 为 640）
            rank: 双线性低秩维度
            hidden_dim: 残基聚合后 MLP 隐藏维度
            dropout: Dropout 比率
            max_len: 单个蛋白最大残基数
            max_residue_batch: 分块前向的最大 batch 大小
            init_scheme: 初始化策略
            final_bias_init: 输出偏置初始值
        """
        super().__init__()
        self.comp_dim = comp_dim
        self.residue_dim = residue_dim
        self.rank = rank
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.max_residue_batch = max_residue_batch
        self.init_scheme = init_scheme
        self.final_bias_init = final_bias_init

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
        # _prot_to_residue_idx[i] 存储图蛋白局部索引 i 对应的 residue 文件索引；
        # -1 表示缺失，查询时会被替换为 unknown placeholder。
        self.register_buffer("_prot_to_residue_idx", torch.zeros(1, dtype=torch.long))

        self._init_weights()

    def _init_weights(self) -> None:
        """受控初始化：缓解训练初期数值不稳定，同时保留足够梯度信号。

        初始化策略:
          - U/V/W 双线性投影使用 init_scheme 指定初始化（xavier/orthogonal/kaiming），
            默认 orthogonal，使低秩交互矩阵更稳定。
          - residue_agg / score_mlp 隐藏层使用 Kaiming (ReLU) 或 Xavier，稳定残基变换。
          - score_mlp 最后一层固定使用小增益 (gain=0.1) 的 Xavier，与 scheme 无关，
            确保训练初期残基注意力路径的输出不会压过 fast bilinear 路径；
            同时 gain=0.1 保留足够梯度，使残基路径能够参与学习。
          - 最终打分偏置初始化为 small negative value（默认 -0.5），对类别不平衡
            （正样本极少）起到温和先验作用，避免初始预测过度乐观。
        """
        scheme = self.init_scheme.lower()

        def _init_linear(linear: nn.Linear, is_final: bool = False) -> None:
            if linear.weight is None:
                return
            if is_final:
                # 最终输出层：统一小增益，避免初始阶段残基路径信号过强
                nn.init.xavier_uniform_(linear.weight, gain=0.1)
            elif scheme == "xavier":
                nn.init.xavier_uniform_(linear.weight, gain=1.0)
            elif scheme == "kaiming":
                nn.init.kaiming_uniform_(linear.weight, a=0, mode="fan_in", nonlinearity="relu")
            elif scheme == "orthogonal":
                nn.init.orthogonal_(linear.weight, gain=1.0)
            else:
                raise ValueError(f"不支持的 init_scheme: {scheme}")
            if linear.bias is not None and not is_final:
                nn.init.zeros_(linear.bias)

        # 双线性投影：保持输入/输出方差平衡
        for linear in (self.U, self.V, self.W):
            _init_linear(linear, is_final=False)

        # 残基聚合 MLP
        for m in self.residue_agg.modules():
            if isinstance(m, nn.Linear):
                _init_linear(m, is_final=False)

        # score_mlp 中间层
        for m in self.score_mlp.modules():
            if isinstance(m, nn.Linear) and m is not self.score_mlp[-1]:
                _init_linear(m, is_final=False)

        # score_mlp 最后一层：小增益 + 负偏置
        final_linear = self.score_mlp[-1]
        if isinstance(final_linear, nn.Linear):
            _init_linear(final_linear, is_final=True)
            if final_linear.bias is not None:
                nn.init.constant_(final_linear.bias, self.final_bias_init)

    def load_pretrained_state(self, state_dict: dict, strict: bool = True) -> None:
        """加载预训练 decoder 权重（例如从已收敛的 bilinear 模型迁移）。

        用于支持 decoder 预训练权重加载方案：
          - 若预训练权重中缺少 score_mlp/residue_agg（从 bilinear 升级到 residue_bilinear），
            自动跳过这些层，保持其受控初始化。
          - 若预训练权重的 U/V/W 维度一致，则直接加载；维度不一致时记录警告。

        Args:
            state_dict: 预训练状态字典。
            strict: 是否严格匹配；对 residue_bilinear 扩展层建议传 strict=False。
        """
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        if missing:
            logger.warning(
                "ResidueAwareBilinearDecoder 加载预训练权重时缺失以下键（将保持受控初始化）: %s",
                missing,
            )
        if unexpected:
            logger.warning("ResidueAwareBilinearDecoder 加载预训练权重时遇到未知键: %s", unexpected)
        logger.info("ResidueAwareBilinearDecoder 预训练权重加载完成")

    def register_residue_buffers(self, embeddings: torch.Tensor, offsets: torch.Tensor,
                                 lengths: torch.Tensor, max_len: int = 1024,
                                 prot_to_residue_idx: torch.Tensor | None = None,
                                 residue_device: str = "cpu") -> None:
        """注册 packed 格式残基特征。

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

        n_residue_proteins = int(offsets.shape[0]) - 1
        self._unknown_idx = n_residue_proteins

        self._residue_embeddings = embeddings
        self._residue_is_mmap = getattr(embeddings, "is_mmap", False)

        unknown_start = int(offsets[-1].item())
        unknown_end = unknown_start + 1
        self._residue_offsets = torch.cat([
            offsets.to(residue_device),
            torch.tensor([unknown_start, unknown_end], dtype=offsets.dtype, device=residue_device),
        ])
        self._residue_lengths = torch.cat([
            lengths.to(residue_device),
            torch.ones(1, dtype=lengths.dtype, device=residue_device),
        ])

        if prot_to_residue_idx is None:
            prot_to_residue_idx = torch.zeros(1, dtype=torch.long)
        # 使用 .data.copy_() 保留 buffer 注册状态，确保 model.to(device) 自动迁移
        if self._prot_to_residue_idx.shape != prot_to_residue_idx.shape:
            self._prot_to_residue_idx.resize_(prot_to_residue_idx.shape)
        self._prot_to_residue_idx.data.copy_(prot_to_residue_idx.to(self._prot_to_residue_idx.device))

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
        """根据蛋白全局索引收集残基特征并填充到 [N, L, d]（逐蛋白循环，避免 CPU OOM）。

        Args:
            prot_indices: (N,) 蛋白全局索引（0-based，对应图蛋白节点）

        Returns:
            residue_feats: (N, max_len, residue_dim)
            residue_mask: (N, max_len) bool, True 表示有效残基
        """
        device = prot_indices.device
        n = prot_indices.shape[0]

        # 将缺失索引(-1/越界)替换为 unknown_idx，避免 _prot_to_residue_idx 索引异常。
        n_residue_proteins = self._prot_to_residue_idx.shape[0]
        safe_indices = prot_indices.clamp(0, n_residue_proteins - 1).long()
        residue_idx = self._prot_to_residue_idx[safe_indices].cpu()
        missing_mask = ((prot_indices < 0) | (prot_indices >= n_residue_proteins)).cpu() | (residue_idx < 0)
        if missing_mask.any():
            logger.warning(
                "_gather_residues: %d/%d 蛋白索引无残基特征映射，已回退到 unknown placeholder",
                missing_mask.sum().item(), residue_idx.shape[0],
            )
        residue_idx = residue_idx.clamp(0, self._unknown_idx)
        unknown_mask = (missing_mask | (residue_idx == self._unknown_idx)).cpu()

        residue_max_len = self.max_len
        residue_dim = self.residue_dim
        total_residues = self._residue_embeddings.shape[0]
        L = residue_max_len
        d = residue_dim

        residue_feats_list: list[torch.Tensor] = []
        residue_mask_list: list[torch.Tensor] = []

        for i in range(n):
            idx = int(residue_idx[i].item())
            prot_len = min(int(self._residue_lengths[idx].item()), residue_max_len)
            offset = int(self._residue_offsets[idx].item())
            end = min(offset + residue_max_len, total_residues)

            # 逐蛋白 gather，每次只分配 ~640KB 在 CPU 上，立即搬 GPU
            emb_slice = self._residue_embeddings[offset:end]
            if isinstance(emb_slice, torch.Tensor):
                feat = emb_slice.to(device)  # (L_actual, d)
            else:
                feat = torch.from_numpy(np.array(emb_slice)).to(device)  # memmap -> numpy -> tensor
            L_actual = feat.shape[0]
            if L_actual < L:
                pad = torch.zeros(L - L_actual, d, device=device, dtype=feat.dtype)
                feat = torch.cat([feat, pad], dim=0)

            mask = torch.zeros(L, dtype=torch.bool, device=device)
            mask[:min(prot_len, L)] = True

            if unknown_mask[i]:
                mask[:] = False
                mask[0] = True
                feat[1:] = 0.0

            residue_feats_list.append(feat)
            residue_mask_list.append(mask)

        residue_feats = torch.stack(residue_feats_list, dim=0)
        residue_mask = torch.stack(residue_mask_list, dim=0)
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
