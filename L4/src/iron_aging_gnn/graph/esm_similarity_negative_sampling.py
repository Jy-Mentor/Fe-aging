"""基于 ESM-2 结构相似度的难负样本挖掘。

利用蛋白级 ESM-2 嵌入的余弦相似度，为每个蛋白寻找结构上最相似
但无直接 PPI 交互的蛋白作为难负样本候选。这补充了仅基于 PPI 网络拓扑
的难负样本策略，捕获结构相似但拓扑关系缺失的蛋白对。

该模块与 topology_negative_sampling.py 互补：
- 拓扑难负样本：共同邻居多、Jaccard 高（网络结构相似）
- ESM-2 难负样本：余弦相似度高（序列/结构相似，可能结合相似配体）
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_esm_similarity_hard_neighbors(
    esm2_embeddings: dict[str, np.ndarray],
    gene_to_idx: dict[str, int],
    n_compounds: int,
    ppi_df: pd.DataFrame | None = None,
    source_col: str = "source",
    target_col: str = "target",
    active_genes: set[str] | None = None,
    top_k: int = 50,
    batch_size: int = 512,
) -> dict[int, set[int]]:
    """基于 ESM-2 余弦相似度构建难负样本邻居表。

    对每个蛋白，返回与其 ESM-2 嵌入余弦相似度最高、但无直接 PPI 交互
    的蛋白集合，作为课程负采样第三阶段的难负样本候选。

    接口与 build_topology_hard_neighbors 一致，可直接替换或叠加使用。

    Args:
        esm2_embeddings: {基因名: ESM-2 嵌入向量 (dim,)}。注意基因名必须已
            strip().upper()。
        gene_to_idx: 基因名 -> 全局节点索引。
        n_compounds: 化合物数量，用于将全局索引转换为蛋白局部索引。
        ppi_df: PPI 边表，用于排除直接交互蛋白。若为 None，不排除任何直接交互。
        source_col: PPI 边起点列名（默认 "source"）。
        target_col: PPI 边终点列名（默认 "target"）。
        active_genes: 若提供，仅针对这些正样本基因预计算邻居，减少计算量。
        top_k: 每个蛋白保留的难候选数量。
        batch_size: 批量计算余弦相似度的批次大小，控制内存峰值。

    Returns:
        {蛋白局部索引: set(ESM-2 相似难负样本局部索引)}
    """
    # 1. 收集 PPI 直接交互（用于排除）
    direct_interactors: dict[str, set[str]] = defaultdict(set)
    if ppi_df is not None:
        for _, row in ppi_df.iterrows():
            a = str(row[source_col]).strip().upper()
            b = str(row[target_col]).strip().upper()
            if a and b and a != b:
                direct_interactors[a].add(b)
                direct_interactors[b].add(a)

    # 2. 构建基因名列表和对应的嵌入矩阵
    genes = list(esm2_embeddings.keys())
    if not genes:
        logger.warning("ESM-2 嵌入为空，无法构建 ESM-2 相似度难负样本")
        return {}

    dim = esm2_embeddings[genes[0]].shape[0]
    emb_matrix = np.zeros((len(genes), dim), dtype=np.float32)
    for i, gene in enumerate(genes):
        emb_matrix[i] = esm2_embeddings[gene].astype(np.float32)

    # L2 归一化，使余弦相似度 = 内积
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_matrix = emb_matrix / norms

    # 3. 确定需要处理的基因集合
    genes_to_process = active_genes if active_genes else set(genes)
    # 过滤：仅保留在 gene_to_idx 中且映射为有效蛋白的基因
    valid_genes_to_process: set[str] = set()
    for gene in genes_to_process:
        if gene in gene_to_idx:
            g_local = gene_to_idx[gene] - n_compounds
            if g_local >= 0:
                valid_genes_to_process.add(gene)
    genes_to_process = valid_genes_to_process

    if not genes_to_process:
        logger.warning("ESM-2 相似度负样本：无有效正样本基因，跳过")
        return {}

    # 4. 构建基因名到嵌入矩阵索引的映射
    gene_to_emb_idx = {gene: i for i, gene in enumerate(genes)}

    # 处理基因列表（按 emb 索引排序以加速批量查询）
    process_list = sorted(genes_to_process, key=lambda g: gene_to_emb_idx.get(g, -1))

    logger.info(
        f"ESM-2 相似度难负样本: 处理 {len(process_list)} 个正样本基因, "
        f"dim={dim}, top_k={top_k}, batch_size={batch_size}"
    )

    result: dict[int, set[int]] = defaultdict(set)
    n_processed = 0
    n_missed = 0

    # 5. 批量计算余弦相似度
    for batch_start in range(0, len(process_list), batch_size):
        batch_genes = process_list[batch_start:batch_start + batch_size]
        batch_indices = []
        for gene in batch_genes:
            idx = gene_to_emb_idx.get(gene)
            if idx is not None:
                batch_indices.append(idx)

        if not batch_indices:
            continue

        batch_emb = emb_matrix[batch_indices]  # (B, dim)
        # 余弦相似度 = 内积（已 L2 归一化）
        sim_matrix = batch_emb @ emb_matrix.T  # (B, N)

        for i, gene in enumerate(batch_genes):
            emb_idx = gene_to_emb_idx.get(gene)
            if emb_idx is None:
                n_missed += 1
                continue

            g_local = gene_to_idx[gene] - n_compounds
            if g_local < 0:
                continue

            similarities = sim_matrix[i]  # (N,)
            # 排除自身
            similarities[emb_idx] = -1.0

            # 排除直接 PPI 交互蛋白
            for interactor in direct_interactors.get(gene, set()):
                if interactor in gene_to_emb_idx:
                    similarities[gene_to_emb_idx[interactor]] = -1.0

            # 取 top-k 最相似
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            for sim_idx in top_indices:
                if similarities[sim_idx] <= 0:
                    continue
                neg_gene = genes[sim_idx]
                if neg_gene not in gene_to_idx:
                    continue
                neg_local = gene_to_idx[neg_gene] - n_compounds
                if neg_local >= 0 and neg_local != g_local:
                    result[g_local].add(neg_local)

            n_processed += 1

    if n_missed > 0:
        logger.warning(f"ESM-2 相似度负样本: {n_missed} 个基因在嵌入矩阵中缺失")

    n_with_candidates = len(result)
    median_candidates = (
        int(np.median([len(v) for v in result.values()]))
        if result else 0
    )
    logger.info(
        f"ESM-2 similarity hard neighbors: {n_with_candidates} proteins with median "
        f"{median_candidates} candidates (processed={n_processed}, missed={n_missed})"
    )
    return result