#!/usr/bin/env python3
"""诊断 SAGE 蛋白冷启动 AUPR 异常高的原因"""
import sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from phase4_v10_minibatch import (
    build_graphs_and_adj, load_protein_features, load_cpi_data,
    load_ppi_network, load_kegg_pathways, load_tcm_pool,
    ALL_FERRORAGING_GENES, DEVICE
)


def main():
    # 加载数据（与主脚本一致）
    cpi_df = load_cpi_data()
    ppi_df = load_ppi_network()
    gene_to_pathways = load_kegg_pathways()
    prot_feat, gene_to_seq = load_protein_features()
    tcm_df = load_tcm_pool()
    warm_targets = sorted(set(cpi_df["gene"].unique()) & set(ALL_FERRORAGING_GENES))
    cpi_df = cpi_df[cpi_df["gene"].isin(warm_targets)].copy()

    graphs = build_graphs_and_adj(cpi_df, ppi_df, gene_to_pathways, prot_feat)

    n_compounds = graphs["n_compounds"]
    n_proteins = graphs["n_proteins"]
    all_proteins = list(range(n_proteins))
    n_train_prot = int(len(all_proteins) * 0.80)
    val_proteins = set(all_proteins[n_train_prot:])

    # 化合物冷启动拆分（与主脚本一致）
    all_smiles = list(graphs["smi_to_idx"].keys())
    np.random.seed(42)
    np.random.shuffle(all_smiles)
    n_val = max(1, int(len(all_smiles) * 0.15))
    val_smiles = set(all_smiles[:n_val])
    val_compounds = [graphs["smi_to_idx"][s] for s in val_smiles if s in graphs["smi_to_idx"]]
    val_comp_set = set(val_compounds)

    # 正样本
    compound_to_pos = defaultdict(set)
    for _, row in cpi_df.iterrows():
        smi = row["canonical_smiles"]
        gene = row["gene"]
        if smi in graphs["smi_to_idx"] and gene in graphs["gene_to_idx"]:
            compound_to_pos[graphs["smi_to_idx"][smi]].add(graphs["gene_to_idx"][gene])

    # 统计验证蛋白相关信息
    print(f"总蛋白数: {n_proteins}, 验证蛋白数: {len(val_proteins)}")

    # 哪些验证蛋白是正样本
    val_pos_proteins = set()
    val_pos_counts = defaultdict(int)
    for c in val_compounds:
        for p in compound_to_pos.get(c, set()):
            local_p = p - n_compounds
            if local_p in val_proteins:
                val_pos_proteins.add(local_p)
                val_pos_counts[local_p] += 1
    print(f"作为正样本出现的验证蛋白数: {len(val_pos_proteins)} / {len(val_proteins)}")
    print(f"每个正样本验证蛋白的平均正样本对数: {np.mean(list(val_pos_counts.values())) if val_pos_counts else 0:.2f}")

    # 验证蛋白的通路特征分布
    prot_matrix_arr = graphs["prot_matrix"]
    esm_dim = graphs["prot_esm_dim"]
    pathway_dim = graphs["n_pathways"]

    pos_prot_list = sorted(val_pos_proteins)
    neg_prot_list = sorted(val_proteins - val_pos_proteins)

    print(f"\n验证蛋白通路特征统计:")
    print(f"  正样本验证蛋白数: {len(pos_prot_list)}")
    print(f"  负样本验证蛋白数: {len(neg_prot_list)}")

    if pathway_dim > 0 and len(pos_prot_list) > 0 and len(neg_prot_list) > 0:
        pos_pathway = prot_matrix_arr[pos_prot_list, esm_dim:esm_dim+pathway_dim]
        neg_pathway = prot_matrix_arr[neg_prot_list, esm_dim:esm_dim+pathway_dim]
        pos_pathway_sum = pos_pathway.sum(axis=1)
        neg_pathway_sum = neg_pathway.sum(axis=1)
        print(f"  正样本验证蛋白平均通路数: {pos_pathway_sum.mean():.2f} ± {pos_pathway_sum.std():.2f}")
        print(f"  负样本验证蛋白平均通路数: {neg_pathway_sum.mean():.2f} ± {neg_pathway_sum.std():.2f}")

        # 哪些通路在正样本中富集
        pos_pathway_freq = pos_pathway.sum(axis=0)
        neg_pathway_freq = neg_pathway.sum(axis=0)
        fold_change = np.zeros(pathway_dim)
        for i in range(pathway_dim):
            p_freq = pos_pathway_freq[i] / max(len(pos_prot_list), 1)
            n_freq = neg_pathway_freq[i] / max(len(neg_prot_list), 1)
            if n_freq > 0:
                fold_change[i] = p_freq / n_freq
            elif p_freq > 0:
                fold_change[i] = np.inf
        top_pathways = np.argsort(fold_change)[-10:][::-1]
        print(f"\n  正样本最富集通路 (top 10):")
        for idx in top_pathways:
            print(f"    通路 {idx}: 正样本频率={pos_pathway_freq[idx]}/{len(pos_prot_list)}, "
                  f"负样本频率={neg_pathway_freq[idx]}/{len(neg_prot_list)}, 倍数={fold_change[idx]:.2f}")

    # 检查验证安全图中验证蛋白是否真正孤立
    from phase4_v10_minibatch import _build_val_safe_homo_edge_index
    homo_edge_index_val = _build_val_safe_homo_edge_index(
        graphs["homo_edge_index"], n_compounds, val_comp_set, val_proteins)

    val_prot_global = torch.tensor(sorted(p + n_compounds for p in val_proteins), dtype=torch.long)
    src_in_val = torch.isin(homo_edge_index_val[0], val_prot_global).sum().item()
    dst_in_val = torch.isin(homo_edge_index_val[1], val_prot_global).sum().item()
    print(f"\n验证安全图检查:")
    print(f"  验证蛋白作为源节点的边数: {src_in_val}")
    print(f"  验证蛋白作为目标节点的边数: {dst_in_val}")
    print(f"  期望: 都为 0")

    # 统计每个验证化合物的正负样本数
    n_pos_list = []
    n_neg_available = []
    for c in val_compounds:
        pos_set = compound_to_pos.get(c, set())
        valid_pos = [p - n_compounds for p in pos_set
                     if n_compounds <= p < n_compounds + n_proteins
                     and (p - n_compounds) in val_proteins]
        n_pos_list.append(len(valid_pos))
        n_neg_available.append(len(val_proteins) - len(valid_pos))

    print(f"\n验证化合物统计 (n_valid={len([x for x in n_pos_list if x > 0])}):")
    print(f"  平均正样本数: {np.mean([x for x in n_pos_list if x > 0]):.2f}")
    print(f"  平均可用负样本数: {np.mean(n_neg_available):.2f}")
    print(f"  正样本数分布: {dict(pd.Series(n_pos_list).value_counts().sort_index())}")


if __name__ == "__main__":
    main()
