"""测试数据集拆分模块。"""

from __future__ import annotations

from L4.src.iron_aging_gnn.graph.split import (
    split_head_tail_nodes,
    split_train_val,
)


def test_split_head_tail_nodes_preserves_tail():
    """头尾节点划分应保留所有尾节点并欠采样头节点。"""
    train_compounds = list(range(100))
    compound_to_pos = {i: set(range(max(1, i % 10))) for i in train_compounds}

    pretrain, tails = split_head_tail_nodes(
        train_compounds, compound_to_pos, head_ratio=0.2, head_undersample_ratio=0.6, seed=42
    )

    # 尾节点应全部保留在预训练集中
    assert set(tails).issubset(set(pretrain))
    # 预训练集大小应小于原始集合（因为头节点被欠采样）
    assert len(pretrain) <= len(train_compounds)
    # 预训练集应非空
    assert len(pretrain) > 0


def test_split_train_val_is_disjoint():
    """训练集与验证集的化合物和蛋白应无交集。"""
    all_compounds = list(range(50))
    all_proteins = list(range(100))
    cpi_proteins = set(range(0, 100, 2))

    train_compounds, val_compounds, train_proteins, val_proteins = split_train_val(
        all_compounds, all_proteins, cpi_proteins, val_compound_ratio=0.2, val_protein_ratio=0.2, seed=42
    )

    assert not set(train_compounds) & set(val_compounds)
    assert not train_proteins & val_proteins
    assert len(val_compounds) > 0
    assert len(val_proteins) > 0
