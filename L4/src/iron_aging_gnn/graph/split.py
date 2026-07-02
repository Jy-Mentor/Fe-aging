"""数据拆分：头尾节点划分 + 训练/验证集拆分"""

from __future__ import annotations

import logging
import math
import random

logger = logging.getLogger(__name__)


def split_head_tail_nodes(
    train_compounds: list[int],
    compound_to_pos: dict[int, set],
    head_ratio: float = 0.2,
    lambda_hhi: float = 1.0,
    seed: int = 42,
    head_undersample_ratio: float = 0.6,
) -> tuple[list[int], list[int]]:
    """v19: 社区感知头尾节点划分 — 简化版 HHI 评分

    参考王煦 CTCL-DPI: Score_v = ln(d_v + 1) * (1 + lambda * HHI_v)。
    由于当前数据未显式给出二部图社区标签，这里用化合物度（已知靶标数）
    近似 HHI：度越低，HHI 越接近 1/degree（邻域越"单一"），越可能是尾节点。

    Returns:
        pretrain_compounds: 尾节点保留 + 头节点欠采样后的子图化合物列表
        tail_compounds: 尾节点化合物列表（用于日志/分析）
    """
    rng = random.Random(seed)
    scores = {}
    for c in train_compounds:
        pos_set = compound_to_pos.get(c, set())
        degree = len(pos_set)
        # 近似 HHI：靶标越少，邻域越集中，HHI 越高
        hhi = 1.0 / max(degree, 1)
        score = math.log(degree + 1) * (1.0 + lambda_hhi * hhi)
        scores[c] = score

    sorted_comps = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    n_head = max(1, int(len(sorted_comps) * head_ratio))
    head_compounds = [c for c, _ in sorted_comps[:n_head]]
    tail_compounds = [c for c, _ in sorted_comps[n_head:]]

    # 头节点欠采样至指定比例，尾节点全部保留
    n_head_keep = max(1, int(len(head_compounds) * head_undersample_ratio))
    rng.shuffle(head_compounds)
    head_kept = head_compounds[:n_head_keep]

    pretrain_compounds = tail_compounds + head_kept
    rng.shuffle(pretrain_compounds)
    return pretrain_compounds, tail_compounds


def split_train_val(
    all_compounds: list[int],
    all_proteins: list[int],
    cpi_proteins: set[int],
    val_compound_ratio: float = 0.15,
    val_protein_ratio: float = 0.20,
    seed: int = 42,
) -> tuple[list[int], list[int], set[int], set[int]]:
    """将化合物和蛋白按比例拆分为训练集和验证集

    化合物拆分: 随机打乱后按 val_compound_ratio 划分。
    蛋白拆分: 分层拆分 — 确保验证集包含足够有 CPI 交互的蛋白，
      避免验证集正样本蛋白过少导致评估失真。

    Args:
        all_compounds: 所有化合物全局索引列表
        all_proteins: 所有蛋白局部索引列表（0-based，相对于化合物数）
        cpi_proteins: 有 CPI 交互的蛋白局部索引集合
        val_compound_ratio: 验证集化合物比例 (默认 0.15)
        val_protein_ratio: 验证集蛋白比例 (默认 0.20)
        seed: 随机种子

    Returns:
        train_compounds: 训练集化合物列表
        val_compounds: 验证集化合物列表
        train_proteins: 训练集蛋白集合
        val_proteins: 验证集蛋白集合
    """
    rng = random.Random(seed)

    # 化合物拆分
    shuffled_compounds = list(all_compounds)
    rng.shuffle(shuffled_compounds)
    n_train_comp = int(len(shuffled_compounds) * (1.0 - val_compound_ratio))
    train_compounds = shuffled_compounds[:n_train_comp]
    val_compounds = shuffled_compounds[n_train_comp:]

    # 蛋白分层拆分
    cpi_proteins_list = list(cpi_proteins)
    non_cpi_proteins = [p for p in all_proteins if p not in cpi_proteins]

    rng.shuffle(cpi_proteins_list)
    rng.shuffle(non_cpi_proteins)

    n_val_cpi = max(1, int(len(cpi_proteins_list) * val_protein_ratio))
    n_train_cpi = len(cpi_proteins_list) - n_val_cpi
    n_val_non_cpi = max(1, int(len(non_cpi_proteins) * val_protein_ratio))
    n_train_non_cpi = len(non_cpi_proteins) - n_val_non_cpi

    train_proteins = set(cpi_proteins_list[:n_train_cpi]) | set(non_cpi_proteins[:n_train_non_cpi])
    val_proteins = set(cpi_proteins_list[n_train_cpi:]) | set(non_cpi_proteins[n_train_non_cpi:])

    n_val_cpi_actual = sum(1 for p in val_proteins if p in cpi_proteins)
    logger.info(f"冷启动拆分: {len(train_compounds)} train / {len(val_compounds)} val 化合物")
    logger.info(f"蛋白冷启动: {len(train_proteins)} train / {len(val_proteins)} val 蛋白 "
                f"(CPI蛋白: {len(cpi_proteins)} 总, {n_val_cpi_actual} 在验证集)")

    return train_compounds, val_compounds, train_proteins, val_proteins
