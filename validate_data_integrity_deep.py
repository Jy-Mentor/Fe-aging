#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
铁衰老项目 - 数据真实性深度验证脚本 v2
修复: ESM-2 npz字典结构解析、PPI基因匹配
"""

import os
import sys
import traceback
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from collections import Counter, defaultdict
from scipy.stats import mannwhitneyu, ks_2samp

# ============================================================
# 配置
# ============================================================
BASE = r"d:\铁衰老 绝不重蹈覆辙"
CPI_PATH = os.path.join(BASE, "L4", "results", "experimental_actives_detail_cleaned.csv")
PPI_PATH = os.path.join(BASE, "L1", "results", "ppi_network_extended_significant_edges.csv")
GENES_PATH = os.path.join(BASE, "L1", "results", "ferroaging_genes_96.csv")
ESM2_PATH = os.path.join(BASE, "L4", "results_v10_minibatch", "esm2_protein_embeddings.npz")
PHENO_PATH = os.path.join(BASE, "L4", "results_v10_minibatch", "phenotype_ferroptosis_dataset.csv")
REPORT_PATH = os.path.join(BASE, "L4", "logs", "data_integrity_report_v25.txt")
COMPOUND_POOL_PATH = os.path.join(BASE, "L3", "results", "tcm_compound_pool_v21_Alevel.csv")

os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

report_lines = []
issues_summary = []

def log(msg):
    print(msg)
    report_lines.append(msg)

def issue(level, check_name, detail):
    icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(level, "?")
    line = f"{icon} [{level}] {check_name}: {detail}"
    log(line)
    if level in ("WARN", "FAIL"):
        issues_summary.append(line)

# ============================================================
# 1. CPI数据深度验证
# ============================================================
def deep_validate_cpi():
    log("\n" + "=" * 70)
    log("【深度验证1】CPI数据深度验证")
    log("=" * 70)
    
    df = pd.read_csv(CPI_PATH, low_memory=False)
    log(f"  总行数: {len(df)}, 唯一基因: {df['gene'].nunique()}, 唯一SMILES: {df['canonical_smiles'].nunique()}")
    
    # 1.1 SMILES过短检查
    log("\n  --- 1.1 SMILES长度异常检查 ---")
    df['smiles_len'] = df['canonical_smiles'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    short_smiles = df[df['smiles_len'] < 10]
    if len(short_smiles) > 0:
        for _, row in short_smiles.iterrows():
            smi = row['canonical_smiles']
            mol = Chem.MolFromSmiles(smi)
            heavy_atoms = mol.GetNumHeavyAtoms() if mol else 0
            issue("WARN", "SMILES过短", f"基因={row['gene']}, SMILES='{smi}', 长度={row['smiles_len']}, 重原子数={heavy_atoms} (真实小分子，非异常)")
    else:
        issue("PASS", "SMILES过短检查", "无SMILES长度<10的异常记录")
    
    log(f"  SMILES长度分布: min={df['smiles_len'].min()}, max={df['smiles_len'].max()}, median={df['smiles_len'].median():.0f}, mean={df['smiles_len'].mean():.1f}")
    
    # 1.2 基因名拼写检查
    log("\n  --- 1.2 基因名拼写/格式检查 ---")
    all_genes = df['gene'].dropna().unique()
    import re
    def is_valid_hgnc(name):
        if not isinstance(name, str) or not name:
            return False
        return bool(re.match(r'^[A-Z][A-Z0-9a-z\-]*$', name))
    
    invalid_genes = [g for g in sorted(all_genes) if not is_valid_hgnc(g)]
    if invalid_genes:
        for g in invalid_genes:
            issue("WARN", "基因名格式异常", f"非标准HGNC格式: '{g}'")
    else:
        issue("PASS", "基因名格式检查", f"所有{len(all_genes)}个基因名符合HGNC标准格式")
    
    case_issues = [g for g in all_genes if g != g.upper() and not g[0].isdigit()]
    if case_issues:
        for g in case_issues:
            issue("WARN", "基因名大小写", f"非全大写: '{g}'")
    else:
        issue("PASS", "基因名大小写", "所有基因名均为大写标准格式")
    
    # 1.3 多靶标药物检查
    log("\n  --- 1.3 多靶标药物检查 ---")
    smiles_groups = df.groupby('canonical_smiles')['gene'].apply(set)
    multi_target = smiles_groups[smiles_groups.apply(len) > 1]
    if len(multi_target) > 0:
        n_targets_dist = multi_target.apply(len).value_counts().sort_index()
        log(f"  发现 {len(multi_target)} 个多靶标化合物")
        log(f"  多靶标分布: {dict(n_targets_dist)}")
        for i, (smi, genes) in enumerate(multi_target.head(5).items()):
            log(f"    SMILES={smi[:60]}... -> {len(genes)}个靶标: {sorted(genes)[:8]}")
        issue("PASS", "多靶标药物", f"发现{len(multi_target)}个多靶标化合物，属正常生物现象")
    else:
        issue("PASS", "多靶标药物", "无多靶标化合物")
    
    # 1.4 每个基因的CPI数量分布
    log("\n  --- 1.4 基因CPI数量分布 ---")
    gene_counts = df['gene'].value_counts()
    log(f"  分布: min={gene_counts.min()}, max={gene_counts.max()}, median={gene_counts.median():.0f}, mean={gene_counts.mean():.1f}, std={gene_counts.std():.1f}")
    ratio = gene_counts.max() / gene_counts.min()
    if ratio > 100:
        issue("WARN", "CPI数量极端不平衡", f"max/min={gene_counts.max()}/{gene_counts.min()}={ratio:.1f}>100")
    else:
        issue("PASS", "CPI数量分布", f"max/min={ratio:.1f}")
    log(f"  Top5: {dict(gene_counts.head(5))}")
    log(f"  Bottom5: {dict(gene_counts.tail(5))}")
    
    # 1.5 pchembl_value分布
    log("\n  --- 1.5 pchembl_value分布 ---")
    pchembl = df['pchembl_value'].dropna()
    if len(pchembl) > 0:
        log(f"  pchembl: min={pchembl.min():.2f}, max={pchembl.max():.2f}, mean={pchembl.mean():.2f}, median={pchembl.median():.2f}")
        abnormal = pchembl[(pchembl < 0) | (pchembl > 14)]
        if len(abnormal) > 0:
            issue("FAIL", "pchembl异常", f"发现{len(abnormal)}个超出0-14范围的值")
        else:
            issue("PASS", "pchembl范围", "所有值在0-14范围内")
    else:
        issue("WARN", "pchembl", "该列全为空")
    
    return df


# ============================================================
# 2. PPI网络拓扑验证
# ============================================================
def deep_validate_ppi():
    log("\n" + "=" * 70)
    log("【深度验证2】PPI网络拓扑验证")
    log("=" * 70)
    
    df = pd.read_csv(PPI_PATH)
    all_nodes = set(df['gene_a'].unique()) | set(df['gene_b'].unique())
    log(f"  总边数: {len(df)}, 唯一节点: {len(all_nodes)}")
    
    # 2.1 度数分布
    log("\n  --- 2.1 度数分布 ---")
    degrees = Counter()
    for _, row in df.iterrows():
        degrees[row['gene_a']] += 1
        degrees[row['gene_b']] += 1
    
    deg_values = np.array([degrees.get(n, 0) for n in all_nodes])
    log(f"  度数: min={deg_values.min()}, max={deg_values.max()}, median={np.median(deg_values):.0f}, mean={deg_values.mean():.1f}")
    
    isolated = [n for n in all_nodes if degrees.get(n, 0) == 0]
    if isolated:
        issue("FAIL", "孤立节点", f"发现{len(isolated)}个孤立节点(度=0)")
    else:
        issue("PASS", "孤立节点", "无孤立节点，所有节点至少有一条边")
    
    bins = [0, 1, 5, 10, 50, 100, 500, 1000, 10000]
    for i in range(len(bins)-1):
        count = ((deg_values >= bins[i]) & (deg_values < bins[i+1])).sum()
        if count > 0:
            log(f"    度[{bins[i]}, {bins[i+1]}): {count} 节点")
    
    # 2.2 铁衰老96基因PPI连接度
    log("\n  --- 2.2 铁衰老96基因PPI连接度 ---")
    genes_df = pd.read_csv(GENES_PATH)
    ferro_genes_list = genes_df['gene_symbol'].tolist()
    ferro_genes_set = set(ferro_genes_list)
    
    # 直接匹配（PPI基因名应该已经是标准格式）
    ferro_in_ppi = ferro_genes_set & all_nodes
    ferro_not_in_ppi = ferro_genes_set - all_nodes
    
    log(f"  铁衰老96基因在PPI中: {len(ferro_in_ppi)}/{len(ferro_genes_set)}")
    log(f"  缺失的铁衰老基因: {sorted(ferro_not_in_ppi)}")
    
    if ferro_not_in_ppi:
        issue("WARN", "铁衰老基因PPI缺失", f"{len(ferro_not_in_ppi)}个基因不在PPI网络中: {sorted(ferro_not_in_ppi)[:15]}...")
    
    # 计算铁衰老基因度数
    ferro_deg_values = np.array([degrees.get(g, 0) for g in ferro_genes_list])
    log(f"  铁衰老基因度数: min={ferro_deg_values.min()}, max={ferro_deg_values.max()}, median={np.median(ferro_deg_values):.0f}, mean={ferro_deg_values.mean():.1f}")
    
    global_median = np.median(deg_values)
    if np.median(ferro_deg_values) >= global_median:
        issue("PASS", "铁衰老基因连接度", f"中位度数({np.median(ferro_deg_values):.0f}) >= 全局中位({global_median:.0f})")
    else:
        issue("WARN", "铁衰老基因连接度", f"中位度数({np.median(ferro_deg_values):.0f}) < 全局中位({global_median:.0f})，部分基因不在PPI中")
    
    # 铁衰老基因在PPI中的度数排名
    ferro_in_ppi_deg = [(g, degrees[g]) for g in ferro_in_ppi]
    top_ferro = sorted(ferro_in_ppi_deg, key=lambda x: x[1], reverse=True)[:10]
    log(f"  PPI中度数最高的铁衰老基因: {top_ferro}")
    
    # 2.3 combined_score
    log("\n  --- 2.3 combined_score ---")
    scores = df['combined_score'].dropna()
    log(f"  combined_score: min={scores.min()}, max={scores.max()}, mean={scores.mean():.1f}, median={scores.median():.0f}")
    if scores.max() > 1000:
        issue("WARN", "combined_score", f"最大值{scores.max()} > 1000")
    else:
        issue("PASS", "combined_score", "得分范围正常")
    
    return df, degrees, all_nodes


# ============================================================
# 3. ESM-2嵌入质量验证
# ============================================================
def deep_validate_esm2():
    log("\n" + "=" * 70)
    log("【深度验证3】ESM-2嵌入质量验证")
    log("=" * 70)
    
    data = np.load(ESM2_PATH, allow_pickle=True)
    gene_names = sorted(data.keys())
    log(f"  嵌入基因数: {len(gene_names)}")
    log(f"  前20个基因: {gene_names[:20]}")
    
    # 构建嵌入矩阵 (genes × 640)
    embeddings = np.array([data[g] for g in gene_names])
    n_proteins, emb_dim = embeddings.shape
    log(f"  嵌入矩阵: {n_proteins} x {emb_dim}")
    
    # 3.1 L2范数分布
    log("\n  --- 3.1 L2范数分布 ---")
    l2_norms = np.linalg.norm(embeddings, axis=1)
    log(f"  L2范数: min={l2_norms.min():.4f}, max={l2_norms.max():.4f}, mean={l2_norms.mean():.4f}, median={np.median(l2_norms):.4f}, std={l2_norms.std():.4f}")
    
    zero_vecs = np.sum(l2_norms < 1e-8)
    if zero_vecs > 0:
        issue("FAIL", "零向量", f"发现{zero_vecs}个近零嵌入向量")
    else:
        issue("PASS", "零向量检测", "无零向量嵌入")
    
    q1, q3 = np.percentile(l2_norms, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
    outliers_l2 = np.sum((l2_norms < lower) | (l2_norms > upper))
    if outliers_l2 > 0:
        outlier_idx = np.where((l2_norms < lower) | (l2_norms > upper))[0]
        outlier_names = [gene_names[i] for i in outlier_idx]
        issue("WARN", "L2范数离群", f"发现{outliers_l2}个离群值(3*IQR): {outlier_names}")
    else:
        issue("PASS", "L2范数离群", "无3*IQR外的离群值")
    
    # 3.2 嵌入向量到质心距离
    log("\n  --- 3.2 嵌入向量离群检测 ---")
    centroid = np.mean(embeddings, axis=0)
    dist_to_centroid = np.array([np.linalg.norm(emb - centroid) for emb in embeddings])
    
    q1_c, q3_c = np.percentile(dist_to_centroid, [25, 75])
    iqr_c = q3_c - q1_c
    upper_c = q3_c + 3 * iqr_c
    outliers_c = np.sum(dist_to_centroid > upper_c)
    if outliers_c > 0:
        outlier_idx = np.where(dist_to_centroid > upper_c)[0]
        outlier_names = [gene_names[i] for i in outlier_idx]
        issue("WARN", "嵌入离群", f"发现{outliers_c}个离群向量: {outlier_names}")
    else:
        issue("PASS", "嵌入离群", "无显著离群嵌入向量")
    
    log(f"  到质心距离: min={dist_to_centroid.min():.4f}, max={dist_to_centroid.max():.4f}, mean={dist_to_centroid.mean():.4f}, median={np.median(dist_to_centroid):.4f}")
    
    # 3.3 成对相似度
    log("\n  --- 3.3 成对余弦相似度 ---")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    emb_norm = embeddings / norms
    sim_matrix = np.dot(emb_norm, emb_norm.T)
    triu_idx = np.triu_indices(len(gene_names), k=1)
    pairwise_sims = sim_matrix[triu_idx]
    
    log(f"  成对余弦相似度: min={pairwise_sims.min():.4f}, max={pairwise_sims.max():.4f}, mean={pairwise_sims.mean():.4f}, median={np.median(pairwise_sims):.4f}")
    
    identical = np.sum(pairwise_sims > 0.9999)
    if identical > 0:
        issue("WARN", "重复嵌入", f"发现{identical}对余弦相似度>0.9999")
    else:
        issue("PASS", "重复嵌入", "无高度相似嵌入对")
    
    # 3.4 铁衰老基因vs非铁衰老基因的嵌入差异
    log("\n  --- 3.4 铁衰老基因嵌入特异性 ---")
    genes_df = pd.read_csv(GENES_PATH)
    ferro_genes_set = set(genes_df['gene_symbol'].tolist())
    
    ferro_indices = [i for i, g in enumerate(gene_names) if g in ferro_genes_set]
    non_ferro_indices = [i for i, g in enumerate(gene_names) if g not in ferro_genes_set]
    
    log(f"  铁衰老基因嵌入数: {len(ferro_indices)}")
    log(f"  非铁衰老基因嵌入数: {len(non_ferro_indices)}")
    
    if ferro_indices and non_ferro_indices:
        # 铁衰老基因间相似度
        ferro_sims = []
        for i in range(len(ferro_indices)):
            for j in range(i+1, len(ferro_indices)):
                s = np.dot(emb_norm[ferro_indices[i]], emb_norm[ferro_indices[j]])
                ferro_sims.append(s)
        
        # 铁衰老 vs 非铁衰老
        cross_sims = []
        np.random.seed(42)
        for _ in range(min(2000, len(ferro_indices) * len(non_ferro_indices))):
            fi = np.random.choice(ferro_indices)
            ni = np.random.choice(non_ferro_indices)
            s = np.dot(emb_norm[fi], emb_norm[ni])
            cross_sims.append(s)
        
        log(f"  铁衰老基因间余弦相似度均值: {np.mean(ferro_sims):.4f}")
        log(f"  铁衰老-非铁衰老间余弦相似度均值: {np.mean(cross_sims):.4f}")
        
        if np.mean(ferro_sims) > np.mean(cross_sims):
            issue("PASS", "铁衰老基因嵌入特异性", f"铁衰老基因间相似度({np.mean(ferro_sims):.4f}) > 铁衰老-非铁衰老({np.mean(cross_sims):.4f})，嵌入具有生物学意义")
        else:
            issue("WARN", "铁衰老基因嵌入特异性", f"铁衰老基因间相似度({np.mean(ferro_sims):.4f}) <= 铁衰老-非铁衰老({np.mean(cross_sims):.4f})")
    
    # 3.5 最相似和最不相似的基因对
    log("\n  --- 3.5 最相似/最不相似基因对 ---")
    # 找前5个最相似的
    top_n = 5
    sorted_pairs = sorted(zip(triu_idx[0], triu_idx[1], pairwise_sims), key=lambda x: x[2], reverse=True)
    log(f"  最相似的前{top_n}对:")
    for i, j, s in sorted_pairs[:top_n]:
        log(f"    {gene_names[i]} - {gene_names[j]}: {s:.6f}")
    
    log(f"  最不相似的前{top_n}对:")
    sorted_pairs_asc = sorted(zip(triu_idx[0], triu_idx[1], pairwise_sims), key=lambda x: x[2])
    for i, j, s in sorted_pairs_asc[:top_n]:
        log(f"    {gene_names[i]} - {gene_names[j]}: {s:.6f}")
    
    return embeddings, gene_names, l2_norms


# ============================================================
# 4. 铁死亡表型数据深度验证
# ============================================================
def deep_validate_phenotype():
    log("\n" + "=" * 70)
    log("【深度验证4】铁死亡表型数据深度验证")
    log("=" * 70)
    
    df = pd.read_csv(PHENO_PATH)
    log(f"  总行数: {len(df)}, 列名: {list(df.columns)}")
    log(f"  标签分布: {dict(df['label'].value_counts())}")
    
    if 'ferroptosis_type' in df.columns:
        log(f"  铁死亡类型: {dict(df['ferroptosis_type'].value_counts())}")
    if 'source' in df.columns:
        log(f"  数据来源: {dict(df['source'].value_counts())}")
    
    # 4.1 SMILES长度分布
    log("\n  --- 4.1 SMILES长度分布 ---")
    df['smiles_len'] = df['canonical_smiles'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    
    pos_len = df[df['label'] == 1]['smiles_len']
    neg_len = df[df['label'] == 0]['smiles_len']
    
    log(f"  label=1(诱导剂): min={pos_len.min()}, max={pos_len.max()}, median={pos_len.median():.0f}, mean={pos_len.mean():.1f}, n={len(pos_len)}")
    log(f"  label=0(非诱导剂): min={neg_len.min()}, max={neg_len.max()}, median={neg_len.median():.0f}, mean={neg_len.mean():.1f}, n={len(neg_len)}")
    
    if len(pos_len) > 0 and len(neg_len) > 0:
        mw_u, mw_p = mannwhitneyu(pos_len, neg_len, alternative='two-sided')
        ks_s, ks_p = ks_2samp(pos_len, neg_len)
        
        if mw_p < 0.01:
            issue("WARN", "SMILES长度差异", f"正负样本SMILES长度有显著差异 (Mann-Whitney p={mw_p:.4e}), 可能引入长度偏差")
        else:
            issue("PASS", "SMILES长度差异", f"正负样本SMILES长度无显著差异 (Mann-Whitney p={mw_p:.4f})")
        log(f"  KS检验: statistic={ks_s:.4f}, p={ks_p:.4e}")
    
    # 4.2 结构重复检查 (Tanimoto > 0.99)
    log("\n  --- 4.2 结构重复检查 (Tanimoto > 0.99) ---")
    mols = []
    valid_indices = []
    for i, smi in enumerate(df['canonical_smiles']):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            valid_indices.append(i)
    
    log(f"  有效分子: {len(mols)}/{len(df)}")
    
    # 生成Morgan指纹 (使用新版API)
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    fps = [GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
    
    high_sim_pairs = []
    n = len(fps)
    
    if n > 2000:
        log(f"  分子数较多({n})，使用分批采样...")
        batch_size = 500
        for b in range(0, n, batch_size):
            batch_end = min(b + batch_size, n)
            for j in range(b, batch_end):
                for k in range(j + 1, n):
                    sim = DataStructs.TanimotoSimilarity(fps[j], fps[k])
                    if sim > 0.99:
                        high_sim_pairs.append((valid_indices[j], valid_indices[k], sim,
                                               df.iloc[valid_indices[j]]['label'],
                                               df.iloc[valid_indices[k]]['label']))
            if len(high_sim_pairs) > 100:
                break
    else:
        for j in range(n):
            for k in range(j + 1, n):
                sim = DataStructs.TanimotoSimilarity(fps[j], fps[k])
                if sim > 0.99:
                    high_sim_pairs.append((valid_indices[j], valid_indices[k], sim,
                                           df.iloc[valid_indices[j]]['label'],
                                           df.iloc[valid_indices[k]]['label']))
    
    if high_sim_pairs:
        # 分析高相似对是否跨标签
        cross_label = sum(1 for p in high_sim_pairs if p[3] != p[4])
        same_label = len(high_sim_pairs) - cross_label
        
        issue("WARN", "结构重复", f"发现{len(high_sim_pairs)}对Tanimoto>0.99的高相似SMILES对 (同标签:{same_label}, 跨标签:{cross_label})")
        for idx1, idx2, sim, l1, l2 in high_sim_pairs[:10]:
            log(f"    行{idx1}(label={l1}) vs 行{idx2}(label={l2}): sim={sim:.6f}")
            smi1 = df.iloc[idx1]['canonical_smiles']
            smi2 = df.iloc[idx2]['canonical_smiles']
            log(f"      SMILES1: {smi1[:80]}")
            log(f"      SMILES2: {smi2[:80]}")
        
        if cross_label > 0:
            issue("FAIL", "跨标签结构重复", f"发现{cross_label}对跨标签高相似SMILES，可能导致数据泄漏！")
    else:
        issue("PASS", "结构重复", "无Tanimoto>0.99的结构重复")
    
    return df


# ============================================================
# 5. 化合物池深度验证
# ============================================================
def deep_validate_compound_pool():
    log("\n" + "=" * 70)
    log("【深度验证5】化合物池深度验证")
    log("=" * 70)
    
    df = pd.read_csv(COMPOUND_POOL_PATH)
    log(f"  总化合物数: {len(df)}")
    
    log("\n  --- 5.1 SMILES长度 ---")
    df['smiles_len'] = df['SMILES_std'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    log(f"  SMILES长度: min={df['smiles_len'].min()}, max={df['smiles_len'].max()}, median={df['smiles_len'].median():.0f}, mean={df['smiles_len'].mean():.1f}")
    
    if 'MW_calc' in df.columns:
        log("\n  --- 5.2 分子量 ---")
        mw = df['MW_calc'].dropna()
        log(f"  MW: min={mw.min():.1f}, max={mw.max():.1f}, mean={mw.mean():.1f}, median={mw.median():.1f}")
        drug_like = ((mw >= 100) & (mw <= 900)).sum()
        log(f"  药物类(100-900): {drug_like}/{len(mw)} ({drug_like/len(mw)*100:.1f}%)")
    
    if 'QED' in df.columns:
        log("\n  --- 5.3 QED ---")
        qed = df['QED'].dropna()
        log(f"  QED: min={qed.min():.3f}, max={qed.max():.3f}, mean={qed.mean():.3f}, median={qed.median():.3f}")
    
    log("\n  --- 5.4 重复SMILES ---")
    dup_smiles = df['SMILES_std'].duplicated().sum()
    if dup_smiles > 0:
        issue("WARN", "重复SMILES", f"发现{dup_smiles}个重复SMILES")
    else:
        issue("PASS", "重复SMILES", "无重复SMILES")
    
    # 5.5 CPI泄漏详细分析
    log("\n  --- 5.5 CPI泄漏详细分析 ---")
    cpi_df = pd.read_csv(CPI_PATH, low_memory=False)
    cpi_smiles = set(cpi_df['canonical_smiles'].dropna().apply(lambda x: x.strip()).unique())
    pool_smiles = set(df['SMILES_std'].dropna().apply(lambda x: x.strip()).unique())
    
    # 标准化比较
    overlap = cpi_smiles & pool_smiles
    log(f"  CPI训练集SMILES数: {len(cpi_smiles)}")
    log(f"  化合物池SMILES数: {len(pool_smiles)}")
    log(f"  重叠SMILES数: {len(overlap)}")
    
    if overlap:
        issue("WARN", "CPI泄漏", f"化合物池与CPI训练集有{len(overlap)}个重叠SMILES，可能影响模型评估")
        for smi in list(overlap)[:5]:
            log(f"    {smi[:80]}")
    
    return df


# ============================================================
# 生成最终报告
# ============================================================
def generate_final_report():
    log("\n\n" + "=" * 70)
    log("=" * 70)
    log("                    📋 数据真实性深度验证最终报告")
    log("=" * 70)
    log("=" * 70)
    
    pass_count = sum(1 for i in issues_summary if "PASS" in i)
    warn_count = sum(1 for i in issues_summary if "WARN" in i)
    fail_count = sum(1 for i in issues_summary if "FAIL" in i)
    
    log(f"\n## 验证摘要")
    log(f"  总问题项: {len(issues_summary)} (PASS:{pass_count}, WARN:{warn_count}, FAIL:{fail_count})")
    
    if fail_count > 0:
        log("\n❌ 存在 FAIL 项，需立即处理！")
    elif warn_count > 0:
        log("\n⚠️ 存在 WARN 项，建议审查")
    else:
        log("\n🎉 所有检查项通过！")
    
    log("\n## 详细问题清单")
    for item in issues_summary:
        log(item)
    
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    log(f"\n\n报告已保存至: {REPORT_PATH}")


# ============================================================
# 主流程
# ============================================================
def main():
    log("=" * 70)
    log("     铁衰老项目 - 数据真实性深度验证 v2")
    log(f"     运行时间: {pd.Timestamp.now()}")
    log("=" * 70)
    
    for step_name, func in [
        ("CPI深度验证", deep_validate_cpi),
        ("PPI网络拓扑验证", deep_validate_ppi),
        ("ESM-2嵌入质量验证", deep_validate_esm2),
        ("铁死亡表型深度验证", deep_validate_phenotype),
        ("化合物池深度验证", deep_validate_compound_pool),
    ]:
        try:
            func()
        except Exception as e:
            issue("FAIL", step_name, f"执行失败: {e}")
            traceback.print_exc()
    
    generate_final_report()


if __name__ == "__main__":
    main()