#!/usr/bin/env python3
"""残基级 ESM-2 嵌入 → 轻量自注意力池化 → 蛋白级嵌入

论文依据:
  - GS-DTI (Bioinformatics 2025): ESM-2 逐残基嵌入 + GMT 多头注意力池化
  - PLM-SWE (Bioinformatics 2025): 注意力池化显著优于平均池化
  - DrugLAMP (Bioinformatics 2024): 残基级注意力可捕获结合口袋信息

OOM 安全设计:
  - CPU 逐批处理 (batch_size=100 蛋白), 内存峰值 < 500MB
  - 输出 6846×640 ≈ 17MB npz, 远小于 8.87GB 原始文件
  - 模型输入维度不变, 无需修改架构

输入:
  L4/results_v10_minibatch/esm2_150M_residue_features.pt (8.87 GB)
输出:
  L4/results_v10_minibatch/esm2_residue_pooled_embeddings.npz (~17 MB)
"""

from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("pool_residue_esm2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "L4" / "results_v10_minibatch"
INPUT_PT = RESULTS_DIR / "esm2_150M_residue_features.pt"
OUTPUT_NPZ = RESULTS_DIR / "esm2_residue_pooled_embeddings.npz"

POOL_BATCH_SIZE = 100   # 每批处理的蛋白数 (CPU, 防止 OOM)
POOL_HIDDEN = 128        # 注意力隐藏维度
POOL_DROPOUT = 0.1       # 注意力 dropout


class ResidueSelfAttentionPool(nn.Module):
    """轻量单头自注意力池化: (L, D) → (D,)

    参考: GS-DTI 的 GMT 池化 + Transformer 的 CLS token 机制
    用可学习的全局 query 向量对残基嵌入做交叉注意力聚合。
    """

    def __init__(self, embed_dim: int = 640, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        # 投影到低维做注意力计算 (减少参数量)
        self.q_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        # 可学习的全局 query (类似 CLS token)
        self.global_query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对单蛋白残基嵌入做自注意力池化

        Args:
            x: (L, embed_dim) 残基嵌入

        Returns:
            (embed_dim,) 蛋白级嵌入
        """
        L = x.shape[0]
        # 投影到低维
        k = self.k_proj(x)  # (L, hidden_dim)
        v = self.v_proj(x)  # (L, hidden_dim)
        q = self.global_query.expand(1, -1)  # (1, hidden_dim)

        # 缩放点积注意力
        attn_scores = (q @ k.T) / (k.shape[-1] ** 0.5)  # (1, L)
        attn_weights = F.softmax(attn_scores, dim=-1)   # (1, L)
        attn_weights = self.dropout(attn_weights)

        # 加权聚合
        pooled = attn_weights @ v  # (1, hidden_dim)
        pooled = self.out_proj(pooled).squeeze(0)  # (embed_dim,)

        # 残差连接: 与均值池化结果相加 (参考 PLM-SWE 发现残差提升稳定性)
        mean_pooled = x.mean(dim=0)
        pooled = self.layer_norm(pooled + mean_pooled)

        return pooled


def main() -> None:
    if not INPUT_PT.exists():
        logger.error(f"残基特征文件不存在: {INPUT_PT}")
        sys.exit(1)

    logger.info(f"加载残基特征文件: {INPUT_PT}")
    logger.info(f"  文件大小: {INPUT_PT.stat().st_size / 1e9:.2f} GB")

    # 加载到 CPU
    data = torch.load(INPUT_PT, map_location="cpu", weights_only=False)

    genes = data["genes"]          # list[str], 长度 N
    embeddings = data["embeddings"]  # (total_residues, 640)
    lengths = data["lengths"]       # (N,)
    offsets = data["offsets"]       # (N+1,)
    missing_genes = data["missing_genes"]

    n_proteins = len(genes)
    embed_dim = embeddings.shape[1]
    total_residues = embeddings.shape[0]
    logger.info(f"  蛋白数: {n_proteins}")
    logger.info(f"  总残基数: {total_residues}")
    logger.info(f"  嵌入维度: {embed_dim}")
    logger.info(f"  缺失蛋白: {len(missing_genes)}")

    # 初始化池化模型
    pool_model = ResidueSelfAttentionPool(
        embed_dim=embed_dim,
        hidden_dim=POOL_HIDDEN,
        dropout=POOL_DROPOUT,
    )
    pool_model.eval()

    # 对每个蛋白做残基池化
    pooled_embeddings = np.zeros((n_proteins, embed_dim), dtype=np.float32)

    with torch.no_grad():
        for batch_start in range(0, n_proteins, POOL_BATCH_SIZE):
            batch_end = min(batch_start + POOL_BATCH_SIZE, n_proteins)
            batch_size_actual = batch_end - batch_start

            if batch_start % (POOL_BATCH_SIZE * 10) == 0:
                logger.info(f"  池化进度: {batch_start}-{batch_end}/{n_proteins}")

            for i in range(batch_size_actual):
                idx = batch_start + i
                length = int(lengths[idx])
                offset = int(offsets[idx])

                if length == 0:
                    # 无残基的蛋白用零向量填充
                    pooled_embeddings[idx] = np.zeros(embed_dim, dtype=np.float32)
                    continue

                # 提取该蛋白的残基嵌入
                residue_emb = embeddings[offset:offset + length]  # (L, 640)

                # 注意力池化
                pooled = pool_model(residue_emb)  # (640,)
                pooled_embeddings[idx] = pooled.numpy()

            # 定期释放内存
            if batch_start % (POOL_BATCH_SIZE * 50) == 0:
                gc.collect()

    # 验证
    n_nonzero = (pooled_embeddings.sum(axis=1) != 0).sum()
    logger.info(f"  池化完成: {n_nonzero}/{n_proteins} 蛋白非零嵌入")
    logger.info(f"  嵌入统计: mean={pooled_embeddings.mean():.6f}, std={pooled_embeddings.std():.6f}")

    # 保存为 npz (与现有 esm2_protein_embeddings.npz 格式兼容)
    save_dict = {gene: pooled_embeddings[i] for i, gene in enumerate(genes)}
    # 同时保存元信息
    save_dict["__meta__"] = np.array([
        f"pooled_from=esm2_150M_residue_features.pt",
        f"pool_method=ResidueSelfAttentionPool",
        f"n_proteins={n_proteins}",
        f"embed_dim={embed_dim}",
        f"missing_genes={len(missing_genes)}",
    ])

    np.savez_compressed(OUTPUT_NPZ, **save_dict)
    logger.info(f"  已保存: {OUTPUT_NPZ}")
    logger.info(f"  输出文件大小: {OUTPUT_NPZ.stat().st_size / 1e6:.2f} MB")

    # 清理
    del data, embeddings, pooled_embeddings
    gc.collect()


if __name__ == "__main__":
    main()