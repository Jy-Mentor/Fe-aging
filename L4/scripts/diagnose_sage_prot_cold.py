#!/usr/bin/env python3
"""诊断 SAGE 蛋白冷启动 AUPR 异常高的原因"""
import sys
from pathlib import Path
import random
import numpy as np
import torch
import pandas as pd
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from phase4_v10_minibatch import (
    build_graphs_and_adj, load_protein_features, load_cpi_data,
    load_ppi_network, load_kegg_pathways, load_tcm_pool,
    ALL_FERRORAGING_GENES, DEVICE,
    _build_val_safe_homo_edge_index, _build_val_safe_hetero_data
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

    # 化合物冷启动拆分（与主脚本一致）
    all_compounds = sorted(graphs["smi_to_idx"].values())
    all_proteins = sorted(set(
        graphs["gene_to_idx"][g] - n_compounds
        for g in graphs["gene_to_idx"]
        if graphs["gene_to_idx"][g] >= n_compounds
    ))
    random.seed(42)
    random.shuffle(all_compounds)
    random.shuffle(all_proteins)

    n_train_comp = int(len(all_compounds) * 0.85)
    val_compounds = all_compounds[n_train_comp:]
    val_comp_set = set(val_compounds)

    # v18-fix: 蛋白冷启动分层拆分（与主脚本一致）
    cpi_proteins = set()
    for _, row in cpi_df.iterrows():
        gene = row["gene"]
        if gene in graphs["gene_to_idx"]:
            cpi_proteins.add(graphs["gene_to_idx"][gene] - n_compounds)
    non_cpi_proteins = [p for p in all_proteins if p not in cpi_proteins]

    n_val_cpi = max(1, int(len(cpi_proteins) * 0.20))
    n_val_non_cpi = max(1, int(len(non_cpi_proteins) * 0.20))

    cpi_proteins = list(cpi_proteins)
    random.shuffle(cpi_proteins)
    random.shuffle(non_cpi_proteins)

    val_proteins = set(cpi_proteins[len(cpi_proteins) - n_val_cpi:]) | set(
        non_cpi_proteins[len(non_cpi_proteins) - n_val_non_cpi:])

    # 正样本（全局蛋白索引）
    compound_to_pos = defaultdict(set)
    for _, row in cpi_df.iterrows():
        smi = row["canonical_smiles"]
        gene = row["gene"]
        if smi in graphs["smi_to_idx"] and gene in graphs["gene_to_idx"]:
            compound_to_pos[graphs["smi_to_idx"][smi]].add(graphs["gene_to_idx"][gene])

    # 统计验证蛋白相关信息
    print(f"总蛋白数: {n_proteins}, 验证蛋白数: {len(val_proteins)}")
    print(f"CPI蛋白数: {len(cpi_proteins)}, 非CPI蛋白数: {len(non_cpi_proteins)}")

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
    x_arr = graphs["x"].numpy()
    esm_dim = graphs["prot_esm_dim"]
    pathway_dim = graphs["n_pathways"]

    pos_prot_list = sorted(val_pos_proteins)
    neg_prot_list = sorted(val_proteins - val_pos_proteins)

    print(f"\n验证蛋白通路特征统计:")
    print(f"  正样本验证蛋白数: {len(pos_prot_list)}")
    print(f"  负样本验证蛋白数: {len(neg_prot_list)}")

    if pathway_dim > 0 and len(pos_prot_list) > 0 and len(neg_prot_list) > 0:
        pos_pathway = x_arr[n_compounds + np.array(pos_prot_list), esm_dim:esm_dim+pathway_dim]
        neg_pathway = x_arr[n_compounds + np.array(neg_prot_list), esm_dim:esm_dim+pathway_dim]
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
    homo_edge_index_val = _build_val_safe_homo_edge_index(
        graphs["homo_edge_index"], n_compounds, val_comp_set, set())  # 仅移除 val_comp 边
    homo_edge_index_prot_cold = _build_val_safe_homo_edge_index(
        graphs["homo_edge_index"], n_compounds, val_comp_set, val_proteins)  # 严格隔离

    val_prot_global = torch.tensor(sorted(p + n_compounds for p in val_proteins), dtype=torch.long)

    def count_val_prot_edges(edge_index, name):
        src_in_val = torch.isin(edge_index[0], val_prot_global).sum().item()
        dst_in_val = torch.isin(edge_index[1], val_prot_global).sum().item()
        print(f"\n{name}:")
        print(f"  验证蛋白作为源节点的边数: {src_in_val}")
        print(f"  验证蛋白作为目标节点的边数: {dst_in_val}")
        print(f"  期望 (蛋白冷启动): 都为 0")
        return src_in_val + dst_in_val

    print("\n验证安全图检查:")
    count_val_prot_edges(homo_edge_index_val, "化合物冷启动图 (仅移除 val_comp 边)")
    count_val_prot_edges(homo_edge_index_prot_cold, "蛋白冷启动图 (严格隔离 val_comp + val_prot)")

    # 异质图严格隔离检查
    hetero_data_prot_cold = _build_val_safe_hetero_data(
        graphs["hetero_data"], val_comp_set, val_proteins)
    print(f"\n异质蛋白冷启动图边数:")
    for et, ei in hetero_data_prot_cold.edge_index_dict.items():
        print(f"  {et}: {ei.shape[1]} 条边")

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

    # ============== 深入特征贡献分析 ==============
    print("\n" + "=" * 60)
    print("深入特征贡献分析")
    print("=" * 60)

    # 构建蛋白级别的标签：验证蛋白中哪些是 CPI 蛋白（即潜在正样本）
    val_prot_labels = np.zeros(len(val_proteins), dtype=int)
    val_prot_local_list = sorted(val_proteins)
    local_to_idx = {p: i for i, p in enumerate(val_prot_local_list)}
    for p in val_pos_proteins:
        val_prot_labels[local_to_idx[p]] = 1

    print(f"\n蛋白级二分类标签: 正样本={val_prot_labels.sum()}, 负样本={len(val_prot_labels) - val_prot_labels.sum()}")

    # 1. 仅通路特征的逻辑回归
    val_prot_indices = np.array(val_prot_local_list)
    pathway_features = x_arr[n_compounds + val_prot_indices, esm_dim:esm_dim + pathway_dim]
    esm2_features = x_arr[n_compounds + val_prot_indices, :esm_dim]

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    def eval_feature_set(X, y, name):
        if X.shape[1] == 0:
            print(f"{name}: 特征维度为 0，跳过")
            return
        # 使用 leave-one-out 风格的简单评估：在验证蛋白上训练、同数据预测（仅看特征可区分性）
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
        clf.fit(X, y)
        proba = clf.predict_proba(X)[:, 1]
        aupr = average_precision_score(y, proba)
        auc = roc_auc_score(y, proba)
        print(f"{name}: AUPR={aupr:.4f}, AUC={auc:.4f}")
        return aupr, auc

    print("\n蛋白级特征可区分性（同一数据集训练/预测，仅衡量特征信息量）:")
    eval_feature_set(pathway_features, val_prot_labels, "仅 KEGG 通路 one-hot")
    eval_feature_set(esm2_features, val_prot_labels, "仅 ESM-2 嵌入 (640-dim)")
    eval_feature_set(np.hstack([esm2_features, pathway_features]), val_prot_labels, "ESM-2 + 通路拼接")

    # 2. 随机打乱通路特征作为基线
    rng = np.random.RandomState(42)
    pathway_shuffled = pathway_features.copy()
    for col in range(pathway_shuffled.shape[1]):
        rng.shuffle(pathway_shuffled[:, col])
    eval_feature_set(pathway_shuffled, val_prot_labels, "随机打乱 KEGG 通路（基线）")

    # 3. 通路名映射与正样本富集通路列表
    print("\n正样本验证蛋白富集通路（KEGG ID）:")
    # 重建 pathway_to_idx 映射
    all_pathways = set()
    for pathways in gene_to_pathways.values():
        all_pathways.update(pathways)
    pathway_to_idx = {p: i for i, p in enumerate(sorted(all_pathways))}
    idx_to_pathway = {i: p for p, i in pathway_to_idx.items()}

    if pathway_dim > 0 and len(pos_prot_list) > 0:
        pos_pathway = x_arr[n_compounds + np.array(pos_prot_list), esm_dim:esm_dim + pathway_dim]
        neg_pathway = x_arr[n_compounds + np.array(neg_prot_list), esm_dim:esm_dim + pathway_dim]
        pos_pathway_freq = pos_pathway.sum(axis=0)
        neg_pathway_freq = neg_pathway.sum(axis=0)
        for idx in np.argsort(-pos_pathway_freq):
            if pos_pathway_freq[idx] > 0:
                kegg_id = idx_to_pathway.get(int(idx), f"idx-{idx}")
                print(f"  {kegg_id}: 正样本频率={int(pos_pathway_freq[idx])}/{len(pos_prot_list)}, "
                      f"负样本频率={int(neg_pathway_freq[idx])}/{len(neg_prot_list)}")

    # 4. 化合物-蛋白对级别：随机基线 AUPR（反映类别不平衡本身的影响）
    print("\n化合物-蛋白对级别基线（反映类别不平衡的随机排序 AUPR）:")
    pair_y = []
    for c in val_compounds:
        pos_set = compound_to_pos.get(c, set())
        valid_pos = [p - n_compounds for p in pos_set
                     if n_compounds <= p < n_compounds + n_proteins
                     and (p - n_compounds) in val_proteins]
        if len(valid_pos) == 0:
            continue
        for p_local in val_prot_local_list:
            label = 1 if p_local in valid_pos else 0
            pair_y.append(label)

    pair_y = np.array(pair_y)
    random_scores = np.random.RandomState(42).rand(len(pair_y))
    print(f"  总对数: {len(pair_y)}, 正样本: {pair_y.sum()}, 负样本: {len(pair_y) - pair_y.sum()}")
    if len(set(pair_y)) == 2:
        print(f"  随机分数 AUPR: {average_precision_score(pair_y, random_scores):.4f} (仅反映 1:10 不平衡)")
        print(f"  随机分数 AUC:  {roc_auc_score(pair_y, random_scores):.4f}")


if __name__ == "__main__":
    main()
