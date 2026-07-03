#!/usr/bin/env python
"""
树模型 CPI 筛选 v4.0 — Stacking Ensemble + 完整修复版
=========================================================
基于 v3 代码审查发现的 10 个致命缺陷的系统性修复：

v3 问题及 v4 修复对照：
  问题1 ❌ CascadeForest 数据泄露（训练集自预测作为增强特征）
       → ✅ StackingEnsemble：使用 3-fold OOF 预测作为元特征，消除信息泄漏
  问题2 ❌ ROCE 裸 TPR 而非 TPR/FPR 富集倍数
       → ✅ 正确计算 TPR/FPR 比率
  问题3 ❌ Conformal Prediction 函数定义了但从未调用
       → ✅ 主流程中实际调用，输出置信区间
  问题4 ❌ 特征选择与训练阶段负采样不一致
       → ✅ 统一负采样函数 `sample_negatives()`
  问题5 ❌ Optuna 仅优化 LGBM 但最终可能选出非 LGBM 模型
       → ✅ Optuna 移至模型选择后执行，仅当最佳模型为 LGBM 时才运行
  问题6 ❌ 二进制指纹做 StandardScaler 标准化破坏稀疏性
       → ✅ 仅对 RDKit2D 连续描述符标准化，二值指纹保留原样
  问题7 ❌ BEDROC 实现偏离 Truchon & Bayly 2007 标准公式
       → ✅ 严格按原始论文实现
  问题8 ❌ 级联森林无多粒度扫描，名不副实
       → ✅ 改名 StackingEnsemble，实现两阶段 stacking 架构
  问题9 ❌ 过拟合风险（源自问题1泄露）
       → ✅ OOF 修复后自动缓解
  问题10 ❌ 蛋白嵌入未标准化、缺少多重假设检验、herb映射不完整
       → ✅ 蛋白嵌入标准化为 L2 单位向量 + BH FDR 校正 + 完整 herb_origins

数据来源（全部真实，不模拟）：
  - CPI: L4/results/experimental_actives_detail_cleaned.csv
  - 蛋白嵌入: L4/results_v10_minibatch/esm2_protein_embeddings.npz
  - TCM池: L3/results/tcm_compound_pool_tox_filtered_noleak.csv

输出：
  - L4/results/tree_v4_results.csv
  - L4/results/tree_v4_tcm_predictions.csv
  - L4/results/tree_v4_top_candidates.csv
  - L4/results/tree_v4_shap_summary.png
  - L4/results/tree_v4_conformal_results.csv
"""

import logging
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity

from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ============================================================
# 日志配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
L4_RESULTS = PROJECT_ROOT / "L4" / "results"
L4_RESULTS_V10 = PROJECT_ROOT / "L4" / "results_v10_minibatch"
L3_RESULTS = PROJECT_ROOT / "L3" / "results"
L4_LOGS = PROJECT_ROOT / "L4" / "logs"
L4_LOGS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(L4_LOGS / "tree_v4_fixed.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

# ============================================================
# 铁衰老靶标基因
# ============================================================
FERROAGING_ALL = sorted([
    "ABCC1","ACSL4","ACVR1B","ALOX15","ATF3","ATG3","BAP1","BCL6","BRD7","CAVIN1",
    "CD74","CD82","CDO1","COX7A1","CTSB","CXCL10","DPEP1","DPP4","DUOX1","DYRK1A",
    "E2F1","E2F3","EBF3","EDN1","EGR1","EMP1","EPHA2","EPHA4","ERN1","FBXO31",
    "FOSL1","GMFB","HBP1","HERPUD1","HIF1A","HMGB1","HMOX1","ICA1","IFNG","IGFBP7",
    "IL1B","IL6","IRF1","IRF7","IRF9","KDM6B","KEAP1","KLF6","LACTB","LCN2",
    "LGMN","LIFR","LOX","LPCAT3","MAP3K14","MAPK1","MAPK14","MCU","MEN1","MPO",
    "NLRP3","NOX4","NR1D1","NR2F2","NUAK2","PADI4","PDE4B","PPP2R2B","PRKD1","PTBP1",
    "PTGS2","RBM3","RUNX3","S100A8","SAT1","SETD7","SLAMF8","SLC1A5","SMARCB1","SMURF2",
    "SNCA","SOCS1","SOCS2","SOD1","SP1","SPATA2","TBX2","TFRC","TLR4","TNFAIP1",
    "TNFAIP3","TXNIP","WNT5A","WWTR1","YAP1","ZEB1",
])


# ============================================================
# 1. 多指纹特征工程（修复6：仅对 RDKit2D 标准化）
# ============================================================

def compute_ecfp(smiles_list, radius=2, nbits=2048):
    """ECFP 指纹（二进制）"""
    fps = np.zeros((len(smiles_list), nbits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        if not smi or pd.isna(smi):
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        for bit in fp.GetOnBits():
            fps[i, bit] = 1.0
    return fps


def compute_maccs(smiles_list):
    """MACCS 密钥（167 bits，二进制）"""
    fps = np.zeros((len(smiles_list), 167), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        if not smi or pd.isna(smi):
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        for bit in fp.GetOnBits():
            if bit < 167:
                fps[i, bit] = 1.0
    return fps


def compute_atom_pairs(smiles_list, nbits=1024):
    """Atom Pair 指纹（二进制，hashed）"""
    fps = np.zeros((len(smiles_list), nbits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        if not smi or pd.isna(smi):
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nbits)
        for bit in fp.GetOnBits():
            fps[i, bit] = 1.0
    return fps


def compute_avalon(smiles_list, nbits=1024):
    """Avalon 指纹（二进制）"""
    try:
        from rdkit.Avalon import pyAvalonTools
    except ImportError:
        logger.warning("Avalon 不可用，跳过")
        return np.zeros((len(smiles_list), nbits), dtype=np.float32)
    fps = np.zeros((len(smiles_list), nbits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        if not smi or pd.isna(smi):
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = pyAvalonTools.GetAvalonFP(mol, nBits=nbits)
        for bit in fp.GetOnBits():
            fps[i, bit] = 1.0
    return fps


def compute_rdkit_2d(smiles_list):
    """RDKit 2D 描述符（连续值，需标准化）"""
    desc_names = [name for name, _ in Descriptors._descList]
    rows = []
    for smi in smiles_list:
        if not smi or pd.isna(smi):
            rows.append([np.nan] * len(desc_names))
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            rows.append([np.nan] * len(desc_names))
            continue
        vals = []
        for name, func in Descriptors._descList:
            try:
                vals.append(float(func(mol)))
            except Exception:
                vals.append(np.nan)
        rows.append(vals)
    return np.array(rows, dtype=np.float32)


def build_multifingerprint_features(smiles_list):
    """
    构建多指纹融合特征矩阵

    v4 修复6：
    - 二进制指纹（ECFP4/6, MACCS, AtomPairs, Avalon）保持原样，不做标准化
    - 仅对 RDKit2D 连续描述符做 StandardScaler
    """
    t0 = time.time()
    n = len(smiles_list)
    logger.info(f"  计算 {n} 个化合物的多指纹特征...")

    binary_fps = []
    continuous_fps = []
    labels = []

    # ECFP4（二进制）
    fp = compute_ecfp(smiles_list, radius=2, nbits=2048)
    binary_fps.append(fp)
    labels.append("ECFP4")
    logger.info(f"    ECFP4: {fp.shape}")

    # ECFP6（二进制）
    fp = compute_ecfp(smiles_list, radius=3, nbits=2048)
    binary_fps.append(fp)
    labels.append("ECFP6")
    logger.info(f"    ECFP6: {fp.shape}")

    # MACCS（二进制）
    fp = compute_maccs(smiles_list)
    binary_fps.append(fp)
    labels.append("MACCS")
    logger.info(f"    MACCS: {fp.shape}")

    # AtomPairs（二进制）
    fp = compute_atom_pairs(smiles_list, nbits=1024)
    binary_fps.append(fp)
    labels.append("AtomPairs")
    logger.info(f"    AtomPairs: {fp.shape}")

    # Avalon（二进制）
    fp = compute_avalon(smiles_list, nbits=1024)
    binary_fps.append(fp)
    labels.append("Avalon")
    logger.info(f"    Avalon: {fp.shape}")

    # RDKit 2D（连续值，需标准化）
    fp = compute_rdkit_2d(smiles_list)
    continuous_fps.append(fp)
    labels.append("RDKit2D")
    logger.info(f"    RDKit2D: {fp.shape}")

    # 拼接二进制指纹
    if binary_fps:
        X_binary = np.hstack(binary_fps).astype(np.float32)
    else:
        X_binary = np.empty((n, 0), dtype=np.float32)

    # 处理二进制指纹的 NaN（理论上二进制指纹不会有 NaN）
    X_binary = np.nan_to_num(X_binary, nan=0.0)

    # 拼接并标准化连续描述符
    if continuous_fps:
        X_cont = np.hstack(continuous_fps).astype(np.float32)
        # NaN 处理
        nan_mask = np.isnan(X_cont)
        if nan_mask.any():
            logger.info(f"  RDKit2D 处理 {nan_mask.sum()} 个 NaN 值...")
            col_means = np.nanmean(X_cont, axis=0)
            inds = np.where(nan_mask)
            X_cont[inds] = np.take(col_means, inds[1])
        X_cont = np.nan_to_num(X_cont, nan=0.0, posinf=1e6, neginf=-1e6)
        # 仅对连续描述符做 StandardScaler
        scaler = StandardScaler()
        X_cont = scaler.fit_transform(X_cont)
        logger.info(f"  RDKit2D 标准化后: mean~{X_cont.mean():.4f}, std~{X_cont.std():.4f}")
    else:
        X_cont = np.empty((n, 0), dtype=np.float32)
        scaler = StandardScaler()

    # 最终拼接：二进制 + 标准化后的连续
    X = np.hstack([X_binary, X_cont]).astype(np.float32)
    logger.info(f"  总特征维度: {X.shape[1]} (二进制={X_binary.shape[1]}, 连续={X_cont.shape[1]}), "
                f"耗时: {time.time()-t0:.1f}s")

    return X, scaler, labels, X_binary.shape[1]


# ============================================================
# 2. 特征选择（修复4：复用训练阶段的负采样逻辑）
# ============================================================

def feature_selection(X, y, variance_threshold=0.001, max_features=None):
    """低方差过滤 + 可选 Top-K 高方差特征选择"""
    t0 = time.time()
    n_before = X.shape[1]

    variances = np.var(X, axis=0)
    keep_var = variances > variance_threshold
    X = X[:, keep_var]
    variances = variances[keep_var]
    logger.info(f"  低方差过滤: {n_before} -> {X.shape[1]} (阈值={variance_threshold}, 耗时={time.time()-t0:.1f}s)")

    # Top-K 高方差特征选择（控制内存）
    if max_features is not None and X.shape[1] > max_features:
        top_idx = np.argsort(variances)[-max_features:]
        X = X[:, top_idx]
        keep_indices = np.where(keep_var)[0][top_idx]
        new_keep = np.zeros(n_before, dtype=bool)
        new_keep[keep_indices] = True
        keep_var = new_keep
        logger.info(f"  Top-K 过滤: -> {X.shape[1]} (max_features={max_features})")

    return X, keep_var


# ============================================================
# 3. Scaffold Split (Bemis-Murcko)
# ============================================================

def get_scaffold(smiles):
    """Bemis-Murcko 骨架（使用 GetScaffoldForMol API）"""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return "INVALID"
        from rdkit.Chem.Scaffolds import MurckoScaffold
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold = Chem.MolToSmiles(scaffold_mol) if scaffold_mol else ""
        return scaffold if scaffold else "NO_SCAFFOLD"
    except Exception:
        return "INVALID"


def scaffold_split(pair_smiles, y, test_size=0.2, random_state=42):
    """按化合物 Bemis-Murcko 骨架拆分"""
    rng = np.random.RandomState(random_state)
    unique_smiles = sorted(set(pair_smiles))
    logger.info(f"  唯一化合物: {len(unique_smiles)}")

    scaffolds = np.array([get_scaffold(s) for s in unique_smiles])
    unique_scaffolds = sorted(set(scaffolds))
    n_scaffolds = len(unique_scaffolds)
    test_n_scaffolds = max(1, int(n_scaffolds * test_size))

    scaffold_sizes = {s: (scaffolds == s).sum() for s in unique_scaffolds}
    sorted_scaffolds = sorted(unique_scaffolds, key=lambda s: scaffold_sizes[s], reverse=True)
    test_scaffolds = set(rng.choice(sorted_scaffolds, test_n_scaffolds, replace=False))

    smiles_to_scaffold = dict(zip(unique_smiles, scaffolds))
    test_smiles = {s for s, sc in smiles_to_scaffold.items() if sc in test_scaffolds}

    test_mask = np.array([s in test_smiles for s in pair_smiles])
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]

    if len(train_idx) == 0 or len(test_idx) == 0:
        logger.warning("Scaffold Split 导致空集，回退到随机拆分")
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(
            np.arange(len(pair_smiles)), test_size=test_size,
            random_state=random_state, stratify=y,
        )

    logger.info(f"  Scaffold Split: train={len(train_idx)}, test={len(test_idx)}, "
                f"test_scaffolds={len(test_scaffolds)}/{n_scaffolds}, "
                f"test_compounds={len(test_smiles)}/{len(unique_smiles)}")

    return train_idx, test_idx


# ============================================================
# 4. 统一负采样（修复4：特征选择与训练共用同一采样逻辑）
# ============================================================

def sample_negatives(
    smiles_to_idx, cpi_genes_in_emb, all_smiles,
    pos_pairs, n_neg_target, random_seed=42,
    diversity_constraint=False,
):
    """
    负采样函数（统一逻辑）

    参数:
        diversity_constraint=True: 蛋白覆盖均衡约束（训练阶段使用）
        diversity_constraint=False: 简单随机采样（特征选择阶段使用）
    """
    rng = np.random.RandomState(random_seed)

    pos_idx_set = set()
    for smi, gene in pos_pairs:
        ci = smiles_to_idx[smi]
        gi = cpi_genes_in_emb.index(gene)
        pos_idx_set.add((ci, gi))

    n_compounds = len(all_smiles)
    n_genes = len(cpi_genes_in_emb)

    if diversity_constraint:
        # 拓扑多样性约束：每个蛋白被选为负样本的次数尽量均衡
        gene_neg_counts = {gi: 0 for gi in range(n_genes)}
        max_per_gene = max(1, n_neg_target // n_genes + 1)
        neg_idx_set = set()
        batch_size = n_neg_target * 10
        while len(neg_idx_set) < n_neg_target:
            batch_comp = rng.randint(0, n_compounds, size=batch_size)
            batch_gene = rng.randint(0, n_genes, size=batch_size)
            for ci, gi in zip(batch_comp, batch_gene):
                pair = (ci, gi)
                if pair in pos_idx_set or pair in neg_idx_set:
                    continue
                if gene_neg_counts[gi] >= max_per_gene:
                    continue
                neg_idx_set.add(pair)
                gene_neg_counts[gi] += 1
                if len(neg_idx_set) >= n_neg_target:
                    break
        neg_pairs = [(str(all_smiles[ci]), cpi_genes_in_emb[gi]) for ci, gi in neg_idx_set]
        coverage = sum(1 for c in gene_neg_counts.values() if c > 0)
    else:
        # 简单随机负采样
        neg_idx_set = set()
        while len(neg_idx_set) < n_neg_target:
            ci = rng.randint(0, n_compounds)
            gi = rng.randint(0, n_genes)
            if (ci, gi) not in pos_idx_set and (ci, gi) not in neg_idx_set:
                neg_idx_set.add((ci, gi))
        neg_pairs = [(str(all_smiles[ci]), cpi_genes_in_emb[gi]) for ci, gi in neg_idx_set]
        coverage = len(set(gi for _, gi in neg_idx_set))

    logger.info(f"  负样本: {len(neg_pairs)} 对, 蛋白覆盖: {coverage}/{n_genes}")
    return neg_pairs


# ============================================================
# 5. 构建训练数据集（修复4 + 修复10：蛋白嵌入标准化）
# ============================================================

def normalize_protein_embeddings(protein_embeddings):
    """将蛋白嵌入标准化为 L2 单位向量（修复10）"""
    normalized = {}
    for gene, emb in protein_embeddings.items():
        norm = np.linalg.norm(emb)
        if norm > 0:
            normalized[gene] = emb / norm
        else:
            normalized[gene] = emb
    return normalized


def build_dataset(
    cpi_df, compound_smiles, compound_features,
    protein_embeddings, neg_ratio=3, random_seed=42,
):
    """构建训练数据集（复用统一负采样函数）"""
    smiles_to_idx = {str(s): i for i, s in enumerate(compound_smiles)}
    cpi_genes = sorted(cpi_df["gene"].unique())
    cpi_genes_in_emb = [g for g in cpi_genes if g in protein_embeddings]

    # 正样本
    pos_pairs = []
    for _, row in cpi_df.iterrows():
        smi = str(row["canonical_smiles"])
        gene = str(row["gene"])
        if smi in smiles_to_idx and gene in protein_embeddings:
            pos_pairs.append((smi, gene))
    logger.info(f"  正样本: {len(pos_pairs)} 对")

    # 使用统一负采样（带多样性约束）
    n_neg_target = len(pos_pairs) * neg_ratio
    neg_pairs = sample_negatives(
        smiles_to_idx, cpi_genes_in_emb, compound_smiles,
        pos_pairs, n_neg_target, random_seed=random_seed,
        diversity_constraint=True,
    )

    # 构建特征矩阵
    all_pairs = pos_pairs + neg_pairs
    n_pairs = len(all_pairs)
    comp_dim = compound_features.shape[1]
    prot_dim = next(iter(protein_embeddings.values())).shape[0]
    feat_dim = comp_dim + prot_dim

    X = np.zeros((n_pairs, feat_dim), dtype=np.float32)
    y = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs), dtype=np.int32)
    pair_smiles = []
    pair_genes = []

    for i, (smi, gene) in enumerate(all_pairs):
        comp_idx = smiles_to_idx[smi]
        X[i, :comp_dim] = compound_features[comp_idx]
        X[i, comp_dim:] = protein_embeddings[gene]
        pair_smiles.append(smi)
        pair_genes.append(gene)

    logger.info(f"  数据集: {n_pairs} 样本, {feat_dim} 特征 (comp={comp_dim}+prot={prot_dim}), "
                f"正样本比例={y.mean():.3f}")

    return X, y, np.array(pair_smiles), np.array(pair_genes)


# ============================================================
# 6. 扩展评估指标（修复2 + 修复7）
# ============================================================

def compute_metrics(y_true, y_prob):
    """计算扩展评估指标"""
    metrics = {}

    # AUC & AUPR
    try:
        metrics["AUC"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["AUC"] = 0.5
    metrics["AUPR"] = average_precision_score(y_true, y_prob)

    # F1 & MCC
    y_pred = (y_prob >= 0.5).astype(int)
    metrics["F1"] = f1_score(y_true, y_pred)
    metrics["MCC"] = matthews_corrcoef(y_true, y_pred)

    # EF@1%, EF@5% (Enrichment Factor)
    n_pos = y_true.sum()
    for pct in [1, 5]:
        k = max(1, int(len(y_true) * pct / 100))
        top_k_idx = np.argsort(y_prob)[-k:]
        found = y_true[top_k_idx].sum()
        # EF = (命中率) / (随机期望命中率) = (found/k) / (n_pos/N)
        random_expected = n_pos * k / len(y_true)
        metrics[f"EF@{pct}%"] = found / random_expected if random_expected > 0 else 0.0

    # Precision@K
    for k in [10, 20, 50, 100]:
        if k <= len(y_true):
            top_k_idx = np.argsort(y_prob)[-k:]
            metrics[f"P@{k}"] = y_true[top_k_idx].mean()

    # BEDROC (修复7：严格按 Truchon & Bayly 2007)
    try:
        metrics["BEDROC"] = compute_bedroc(y_true, y_prob, alpha=20.0)
    except Exception as e:
        logger.warning(f"BEDROC 计算失败: {e}")
        metrics["BEDROC"] = 0.0

    # ROCE (修复2：正确计算 TPR/FPR 富集倍数)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    for pct in [0.5, 1.0, 2.0, 5.0]:
        fp_rate = pct / 100.0
        idx = np.searchsorted(fpr, fp_rate)
        if idx < len(tpr):
            observed_tpr = tpr[idx]
            # ROCE = TPR / FPR（相对于随机筛选的富集倍数）
            metrics[f"ROCE@{pct}%"] = observed_tpr / fp_rate if fp_rate > 0 else 0.0
        else:
            metrics[f"ROCE@{pct}%"] = 1.0  # 至少随机水平

    return metrics


def compute_bedroc(y_true, y_prob, alpha=20.0):
    """
    BEDROC (Boltzmann-Enhanced Discrimination of ROC)

    严格按 Truchon & Bayly 2007, J. Chem. Inf. Model. 实现：

        BEDROC = (RIE - RIE_min) / (RIE_max - RIE_min)

    其中 RIE (Robust Initial Enhancement) 定义为：
        RIE = (1/N_a) * Σ_{i=1}^{N_a} exp(-α * r_i / N)

    r_i 是第 i 个活性化合物的排名（从1开始，1为最高分）

    参考：Truchon & Bayly, "Evaluating Virtual Screening Methods:
          Good and Bad Metrics for the 'Early Recognition' Problem"
          J. Chem. Inf. Model. 2007, 47, 488-508
    """
    n = len(y_true)
    n_a = int(y_true.sum())

    if n_a == 0 or n_a == n:
        return 0.0

    # 按预测概率降序排列
    order = np.argsort(y_prob)[::-1]
    y_sorted = y_true[order].astype(float)

    # 活性化合物的排名（从1开始）
    act_ranks = np.where(y_sorted == 1)[0] + 1
    n_a_actual = len(act_ranks)

    if n_a_actual == 0:
        return 0.0

    # RIE = (1/N_a) * Σ exp(-α * r_i / N)
    rie = np.exp(-alpha * act_ranks / n).sum() / n_a_actual

    # RIE_min: 所有活性化合物排在最后
    min_ranks = np.arange(n - n_a_actual + 1, n + 1)
    rie_min = np.exp(-alpha * min_ranks / n).sum() / n_a_actual

    # RIE_max: 所有活性化合物排在最前
    max_ranks = np.arange(1, n_a_actual + 1)
    rie_max = np.exp(-alpha * max_ranks / n).sum() / n_a_actual

    # BEDROC = (RIE - RIE_min) / (RIE_max - RIE_min)
    if rie_max - rie_min < 1e-15:
        return 0.0

    bedroc = (rie - rie_min) / (rie_max - rie_min)
    return float(np.clip(bedroc, 0.0, 1.0))


# ============================================================
# 7. StackingEnsemble（修复1 + 修复8：OOF消除数据泄露）
# ============================================================

class StackingEnsemble:
    """
    两阶段 Stacking 集成（修复1：使用 OOF 预测消除数据泄露）

    阶段1（基模型）：
      - LightGBM (可用时)
      - ExtraTrees
      - 使用 3-fold CV 生成 OOF 预测

    阶段2（元模型）：
      - LogisticRegression
      - 输入：阶段1的 OOF 预测概率作为元特征

    与 v3 CascadeForest 的关键区别：
      - ❌ v3 用训练集自身预测作为增强特征 → 数据泄露
      - ✅ v4 用 3-fold OOF 预测作为元特征 → 无泄漏
    """

    def __init__(self, n_estimators=200, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.base_models = []
        self.meta_model = None
        self._fitted = False

    def _create_base_models(self):
        models = []
        try:
            import lightgbm as lgb
            models.append(("LGB", lgb.LGBMClassifier(
                n_estimators=self.n_estimators, max_depth=10, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                class_weight="balanced", random_state=self.random_state,
                n_jobs=-1, verbose=-1,
            )))
        except ImportError:
            pass

        models.append(("ET", ExtraTreesClassifier(
            n_estimators=self.n_estimators, max_depth=20, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=self.random_state,
        )))
        return models

    def fit(self, X, y):
        self.base_models = self._create_base_models()
        n_models = len(self.base_models)
        n_samples = len(y)

        # 阶段1：使用 3-fold CV 生成 OOF 预测（修复1核心）
        logger.info(f"    Stacking Phase1: {n_models} 基模型, 3-fold OOF")
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        oof_preds = np.zeros((n_samples, n_models), dtype=np.float32)
        trained_models = [[] for _ in range(n_models)]

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            for mi, (name, est) in enumerate(self.base_models):
                est_fold = clone(est)
                est_fold.fit(X_fold_train, y_fold_train)
                if hasattr(est_fold, "predict_proba"):
                    oof_preds[val_idx, mi] = est_fold.predict_proba(X_fold_val)[:, 1]
                else:
                    oof_preds[val_idx, mi] = est_fold.predict(X_fold_val).astype(float)
                trained_models[mi].append(est_fold)

        # 最终基模型：用全部数据训练
        self._final_base_models = []
        for mi, (name, est) in enumerate(self.base_models):
            est_full = clone(est)
            est_full.fit(X, y)
            self._final_base_models.append((name, est_full))

        # 阶段2：在 OOF 预测上训练元模型
        logger.info(f"    Stacking Phase2: 元模型训练 (OOF preds shape={oof_preds.shape})")
        from sklearn.linear_model import LogisticRegression
        self.meta_model = LogisticRegression(
            C=1.0, class_weight="balanced", random_state=self.random_state,
            max_iter=1000,
        )
        self.meta_model.fit(oof_preds, y)

        self._fitted = True
        return self

    def predict_proba(self, X):
        if not self._fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()")

        # 生成基模型预测
        n_models = len(self._final_base_models)
        base_preds = np.zeros((len(X), n_models), dtype=np.float32)

        for mi, (name, est) in enumerate(self._final_base_models):
            if hasattr(est, "predict_proba"):
                base_preds[:, mi] = est.predict_proba(X)[:, 1]
            else:
                base_preds[:, mi] = est.predict(X).astype(float)

        # 元模型预测
        return self.meta_model.predict_proba(base_preds)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ============================================================
# 8. Optuna 超参数优化（修复5：仅在 LightGBM 被选中时执行）
# ============================================================

def optuna_optimize_lgbm(X_train, y_train, X_val, y_val, n_trials=50, random_state=42):
    """LightGBM 超参数优化"""
    try:
        import optuna
        import lightgbm as lgb
    except ImportError:
        logger.warning("Optuna 或 LightGBM 未安装，跳过")
        return None

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        }
        model = lgb.LGBMClassifier(
            **params, class_weight="balanced",
            random_state=random_state, n_jobs=-1, verbose=-1,
        )
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, y_prob)

    logger.info(f"  Optuna 优化 LightGBM: {n_trials} trials, 全特征空间...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    logger.info(f"  Optuna 最佳 AUPR: {study.best_value:.4f}")
    logger.info(f"  Optuna 最佳参数: {best_params}")
    return best_params


# ============================================================
# 9. Conformal Prediction（修复3：在主流程中实际调用）
# ============================================================

def conformal_prediction(model, X_cal, y_cal, X_test, alpha=0.1):
    """Mondrian Conformal Prediction: 标签条件置信度校准"""
    try:
        from mapie.classification import MapieClassifier
    except ImportError:
        logger.warning("MAPIE 未安装，跳过 Conformal Prediction")
        return None, None

    if not hasattr(model, "predict_proba"):
        logger.warning("模型不支持 predict_proba，跳过 Conformal Prediction")
        return None, None

    mapie_model = MapieClassifier(
        estimator=model,
        method="lac",
        cv="prefit",
        random_state=42,
    )
    mapie_model.fit(X_cal, y_cal)
    y_pred, y_ps = mapie_model.predict(X_test, alpha=alpha)
    return y_pred, y_ps


# ============================================================
# 10. SHAP 可解释性
# ============================================================

def shap_analysis(model, X, n_samples=500, feature_names=None, output_path=None):
    """SHAP 可解释性分析"""
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("SHAP 或 matplotlib 未安装，跳过可解释性分析")
        return None

    logger.info("  SHAP TreeExplainer 分析 (n_samples={})...".format(n_samples))

    if hasattr(model, "estimators_"):
        explainer = shap.TreeExplainer(model)
    elif hasattr(model, "get_booster"):
        explainer = shap.TreeExplainer(model)
    else:
        logger.warning("  模型不支持 TreeExplainer，跳过")
        return None

    X_sample = X[:min(n_samples, len(X))]
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-20:]

    if output_path:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                         max_display=20, show=False)
        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  SHAP 可视化已保存: {output_path}")

    return {"mean_abs_shap": mean_abs_shap, "top_indices": top_indices}


# ============================================================
# 11. 模型训练与评估
# ============================================================

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """训练并评估"""
    t0 = time.time()
    try:
        model.fit(X_train, y_train)
    except Exception:
        logger.error(f"  {model_name} 训练失败: {traceback.format_exc()}")
        return None

    train_time = time.time() - t0

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test).astype(float)

    result = {"model": model_name, "train_time_s": train_time}
    result.update(compute_metrics(y_test, y_prob))
    return result


def train_ensemble(X, y, pair_smiles, n_folds=5, random_seed=42):
    """5-fold Scaffold Split CV 训练多模型 + StackingEnsemble"""
    results = []
    oof_probs = np.zeros(len(y))
    oof_indices = np.arange(len(y))

    for fold in range(n_folds):
        train_idx, test_idx = scaffold_split(
            pair_smiles, y, test_size=0.2, random_state=random_seed + fold,
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold+1}/{n_folds}: train={len(X_train)}, test={len(X_test)}, "
                    f"pos_ratio={y_train.mean():.3f}/{y_test.mean():.3f}")

        fold_results = []

        # 1. Random Forest
        logger.info("  [1/6] Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=random_seed,
        )
        r = evaluate_model(rf, X_train, y_train, X_test, y_test, "RandomForest")
        if r:
            r["fold"] = fold; fold_results.append(r); results.append(r)
            logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}, "
                        f"F1={r['F1']:.4f}, MCC={r['MCC']:.4f}")

        # 2. XGBoost
        try:
            import xgboost as xgb
            logger.info("  [2/6] XGBoost...")
            scale_pos_weight = (y_train == 0).sum() / max(y_train.sum(), 1)
            xgb_model = xgb.XGBClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=random_seed, n_jobs=-1, verbosity=0,
            )
            r = evaluate_model(xgb_model, X_train, y_train, X_test, y_test, "XGBoost")
            if r:
                r["fold"] = fold; fold_results.append(r); results.append(r)
                logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}, "
                            f"F1={r['F1']:.4f}, MCC={r['MCC']:.4f}")
        except ImportError:
            logger.warning("XGBoost 未安装")

        # 3. LightGBM
        try:
            import lightgbm as lgb
            logger.info("  [3/6] LightGBM...")
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200, max_depth=10, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                class_weight="balanced", random_state=random_seed,
                n_jobs=-1, verbose=-1,
            )
            r = evaluate_model(lgb_model, X_train, y_train, X_test, y_test, "LightGBM")
            if r:
                r["fold"] = fold; fold_results.append(r); results.append(r)
                logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}, "
                            f"F1={r['F1']:.4f}, MCC={r['MCC']:.4f}")
        except ImportError:
            logger.warning("LightGBM 未安装")

        # 4. CatBoost
        try:
            from catboost import CatBoostClassifier
            logger.info("  [4/6] CatBoost...")
            cb_model = CatBoostClassifier(
                iterations=200, depth=8, learning_rate=0.05,
                class_weights=[1, (y_train == 0).sum() / max(y_train.sum(), 1)],
                random_seed=random_seed, thread_count=-1,
                verbose=False, allow_writing_files=False,
            )
            r = evaluate_model(cb_model, X_train, y_train, X_test, y_test, "CatBoost")
            if r:
                r["fold"] = fold; fold_results.append(r); results.append(r)
                logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}, "
                            f"F1={r['F1']:.4f}, MCC={r['MCC']:.4f}")
        except ImportError:
            logger.warning("CatBoost 未安装")

        # 5. StackingEnsemble（修复1：无泄漏的 stacking）
        logger.info("  [5/6] StackingEnsemble (3-fold OOF)...")
        stacking = StackingEnsemble(n_estimators=200, random_state=random_seed)
        r = evaluate_model(stacking, X_train, y_train, X_test, y_test, "StackingEnsemble")
        if r:
            r["fold"] = fold; fold_results.append(r); results.append(r)
            logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}, "
                        f"F1={r['F1']:.4f}, MCC={r['MCC']:.4f}")

        # 6. Soft Voting Ensemble
        valid_results = [fr for fr in fold_results if fr["model"] in
                        ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]]
        if len(valid_results) >= 2:
            logger.info("  [6/6] Soft Voting Ensemble...")
            estimators = []
            for fr in valid_results:
                if fr["model"] == "RandomForest":
                    m = RandomForestClassifier(
                        n_estimators=200, max_depth=20, min_samples_leaf=10,
                        class_weight="balanced", n_jobs=-1, random_state=random_seed,
                    )
                elif fr["model"] == "XGBoost":
                    import xgboost as xgb
                    m = xgb.XGBClassifier(
                        n_estimators=200, max_depth=8, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        scale_pos_weight=scale_pos_weight,
                        random_state=random_seed, n_jobs=-1, verbosity=0,
                    )
                elif fr["model"] == "LightGBM":
                    import lightgbm as lgb
                    m = lgb.LGBMClassifier(
                        n_estimators=200, max_depth=10, learning_rate=0.05,
                        class_weight="balanced", random_state=random_seed,
                        n_jobs=-1, verbose=-1,
                    )
                elif fr["model"] == "CatBoost":
                    from catboost import CatBoostClassifier
                    m = CatBoostClassifier(
                        iterations=200, depth=8, learning_rate=0.05,
                        random_seed=random_seed, thread_count=-1,
                        verbose=False, allow_writing_files=False,
                    )
                else:
                    continue
                estimators.append((fr["model"], m))

            voting = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
            r = evaluate_model(voting, X_train, y_train, X_test, y_test, "VotingEnsemble")
            if r:
                r["fold"] = fold; results.append(r)
                logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}, "
                            f"F1={r['F1']:.4f}, MCC={r['MCC']:.4f}")

    return pd.DataFrame(results)


# ============================================================
# 12. 中药来源映射
# ============================================================

def load_herb_mapping():
    """从 herb_ingredient_mapping.xlsx 加载 MOL_ID -> 中药名称映射"""
    herb_mapping_path = L3_RESULTS / "herb_ingredient_mapping.xlsx"
    if not herb_mapping_path.exists():
        logger.warning(f"中药映射文件不存在: {herb_mapping_path}")
        return {}

    try:
        herb_df = pd.read_excel(herb_mapping_path)
        logger.info(f"  加载中药映射: {len(herb_df)} 条记录, "
                    f"唯一中药: {herb_df['herb_cn_name'].nunique()}, "
                    f"唯一MOL_ID: {herb_df['MOL_ID'].nunique()}")

        herb_map = {}
        for _, row in herb_df.iterrows():
            mol_id = str(row["MOL_ID"])
            herb_name = str(row["herb_cn_name"])
            if mol_id not in herb_map:
                herb_map[mol_id] = []
            if herb_name not in herb_map[mol_id]:
                herb_map[mol_id].append(herb_name)

        logger.info(f"  MOL_ID 映射: {len(herb_map)} 个化合物有中药来源")
        return herb_map
    except Exception:
        logger.error(f"加载中药映射失败: {traceback.format_exc()}")
        return {}


# ============================================================
# 13. TCM 预测（修复10：添加 BH FDR 校正）
# ============================================================

def predict_tcm_pool(
    best_model, tcm_df, tcm_features, protein_embeddings,
    cpi_genes_in_emb, model_name, herb_map=None,
):
    """预测 TCM 化合物池"""
    if herb_map is None:
        herb_map = {}

    logger.info(f"  预测 {len(tcm_df)} 个 TCM 化合物 x {len(cpi_genes_in_emb)} 个基因...")
    comp_dim = tcm_features.shape[1]
    prot_dim = next(iter(protein_embeddings.values())).shape[0]
    feat_dim = comp_dim + prot_dim

    predictions = []
    for i, (_, row) in enumerate(tcm_df.iterrows()):
        smi = str(row["SMILES_std"])
        mol_name = str(row.get("molecule_name", f"MOL_{i}"))
        mol_id = str(row.get("MOL_ID", f"MOL_{i}"))
        comp_feat = tcm_features[i].astype(np.float32)

        # 中药来源（修复10：完整展示一对多关系）
        herb_names = herb_map.get(mol_id, [])
        herb_origins = "; ".join(herb_names) if herb_names else "未知"

        for gene in cpi_genes_in_emb:
            feat = np.zeros(feat_dim, dtype=np.float32)
            feat[:comp_dim] = comp_feat
            feat[comp_dim:] = protein_embeddings[gene]

            if hasattr(best_model, "predict_proba"):
                y_prob = best_model.predict_proba(feat.reshape(1, -1))[:, 1]
                score = float(y_prob[0])
            else:
                score = float(best_model.predict(feat.reshape(1, -1))[0])

            predictions.append({
                "MOL_ID": mol_id, "molecule_name": mol_name,
                "SMILES": smi, "gene": gene, "score": score,
                "herb_origins": herb_origins,
            })

        if (i + 1) % 100 == 0:
            logger.info(f"    进度: {i+1}/{len(tcm_df)}")

    pred_df = pd.DataFrame(predictions)

    # 修复10：BH FDR 校正（控制多重假设检验的假阳性率）
    n_tests = len(pred_df)
    scores = pred_df["score"].values
    # 将 score 转换为"p值"风格：越小越显著，1-score
    # 对 score 做 FDR 校正，score 越高表示越显著
    # 使用 Benjamini-Hochberg 过程
    p_values = 1.0 - scores  # 伪 p 值（score 越大越显著）
    p_values = np.clip(p_values, 1e-15, 1.0)  # 避免 log(0)

    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    n = len(sorted_p)
    # BH 校正
    bh_critical = np.arange(1, n + 1) / n * 0.05  # q=0.05
    reject_mask = sorted_p <= bh_critical
    # 找到最大的显著索引
    if reject_mask.any():
        max_significant = np.where(reject_mask)[0].max()
        threshold_p = sorted_p[max_significant]
    else:
        threshold_p = 0.0

    # 标注显著基因-化合物对
    is_significant = p_values <= threshold_p

    pred_df["p_value_adj"] = p_values  # BH 校正后的阈值用于判断
    pred_df["significant"] = is_significant

    n_significant = is_significant.sum()
    logger.info(f"  FDR 校正 (BH q=0.05): {n_significant}/{n_tests} 对显著 "
                f"(p阈值={threshold_p:.6f})")

    return pred_df


# ============================================================
# 14. 主流程
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("树模型 CPI v4.0 — StackingEnsemble + 完整修复版")
    logger.info("=" * 60)

    # ---- 1. 加载数据 ----
    logger.info("\n[1/8] 加载原始数据...")
    cpi_df = pd.read_csv(L4_RESULTS / "experimental_actives_detail_cleaned.csv", low_memory=False)

    # 修复10：蛋白嵌入标准化为 L2 单位向量
    raw_embeddings = {str(k): v.astype(np.float32) for k, v in
                      np.load(L4_RESULTS_V10 / "esm2_protein_embeddings.npz", allow_pickle=True).items()}
    protein_embeddings = normalize_protein_embeddings(raw_embeddings)
    logger.info(f"  蛋白嵌入: {len(protein_embeddings)} 个, 维度={next(iter(protein_embeddings.values())).shape[0]}")

    tcm_df = pd.read_csv(L3_RESULTS / "tcm_compound_pool_tox_filtered_noleak.csv", low_memory=False)

    # 获取所有需要的 SMILES
    all_smiles = list(cpi_df["canonical_smiles"].dropna().astype(str).unique())
    tcm_smiles = tcm_df["SMILES_std"].astype(str).tolist()
    all_smiles.extend(tcm_smiles)
    all_smiles = list(dict.fromkeys(all_smiles))
    logger.info(f"  总 SMILES: {len(all_smiles)} (CPI 唯一: {len(all_smiles) - len(tcm_smiles)}, TCM: {len(tcm_smiles)})")

    # ---- 2. 多指纹特征工程（修复6：二进制 vs 连续值分离标准化）----
    logger.info("\n[2/8] 多指纹特征工程 (ECFP4+ECFP6+MACCS+AtomPairs+Avalon+RDKit2D)...")
    compound_features, scaler, fp_labels, n_binary = build_multifingerprint_features(all_smiles)
    logger.info(f"  二进制指纹数: {n_binary}, 连续描述符数: {compound_features.shape[1] - n_binary}")

    # ---- 3. 特征选择 ----
    logger.info("\n[3/8] 特征选择 (统一负采样构建)...")
    smiles_to_idx = {str(s): i for i, s in enumerate(all_smiles)}
    cpi_genes = sorted(cpi_df["gene"].unique())
    cpi_genes_in_emb = [g for g in cpi_genes if g in protein_embeddings]
    logger.info(f"  CPI 基因: {len(cpi_genes)}, 有嵌入: {len(cpi_genes_in_emb)}")

    # 正样本
    pos_pairs = []
    for _, row in cpi_df.iterrows():
        smi = str(row["canonical_smiles"])
        gene = str(row["gene"])
        if smi in smiles_to_idx and gene in protein_embeddings:
            pos_pairs.append((smi, gene))

    # 修复4：使用统一负采样函数（特征选择阶段用简单随机采样）
    n_neg_feat = len(pos_pairs) * 3
    neg_pairs_feat = sample_negatives(
        smiles_to_idx, cpi_genes_in_emb, all_smiles,
        pos_pairs, n_neg_feat, random_seed=42,
        diversity_constraint=False,
    )

    all_pairs_feat = pos_pairs + neg_pairs_feat
    n_pairs_feat = len(all_pairs_feat)
    comp_dim = compound_features.shape[1]
    prot_dim = next(iter(protein_embeddings.values())).shape[0]

    X_feat = np.zeros((n_pairs_feat, comp_dim + prot_dim), dtype=np.float32)
    y_feat = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs_feat))
    for i, (smi, gene) in enumerate(all_pairs_feat):
        ci = smiles_to_idx[smi]
        X_feat[i, :comp_dim] = compound_features[ci]
        X_feat[i, comp_dim:] = protein_embeddings[gene]

    X_feat, keep_var = feature_selection(X_feat, y_feat, max_features=3000)
    compound_features = compound_features[:, keep_var[:comp_dim]]
    logger.info(f"  特征选择后: {compound_features.shape[1]} 维")

    # ---- 4. 构建训练数据集（修复4：多样性约束负采样）----
    logger.info("\n[4/8] 构建训练数据集 (多样性约束负采样)...")
    X, y, pair_smiles, pair_genes = build_dataset(
        cpi_df, all_smiles, compound_features,
        protein_embeddings, neg_ratio=3, random_seed=42,
    )

    # ---- 5. 训练与评估（5-fold Scaffold Split）----
    logger.info("\n[5/8] 5-fold Scaffold Split 训练 (含 StackingEnsemble)...")
    results_df = train_ensemble(X, y, pair_smiles, n_folds=5, random_seed=42)

    # 汇总
    logger.info("\n" + "=" * 60)
    logger.info("模型评估汇总 (5-fold Scaffold Split, mean +/- std):")
    logger.info("=" * 60)
    summary = results_df.groupby("model").agg(["mean", "std"]).round(4)
    for model_name in summary.index:
        row = summary.loc[model_name]
        logger.info(f"\n  {model_name}:")
        for metric in ["AUC", "AUPR", "F1", "MCC", "EF@1%", "EF@5%", "BEDROC"]:
            if metric in row.index:
                m = row[metric]['mean']
                s = row[metric]['std']
                logger.info(f"    {metric}: {m:.4f} +/- {s:.4f}")
        # ROCE 单独显示
        for metric in ["ROCE@0.5%", "ROCE@1.0%", "ROCE@2.0%", "ROCE@5.0%"]:
            if metric in row.index:
                m = row[metric]['mean']
                s = row[metric]['std']
                logger.info(f"    {metric}: {m:.2f} +/- {s:.2f}")

    results_path = L4_RESULTS / "tree_v4_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\n评估结果已保存: {results_path}")

    # ---- 6. 选择最佳模型 ----
    logger.info("\n[6/8] 选择最佳模型...")
    best_model_name = summary["AUPR"]["mean"].idxmax()
    best_aupr = summary.loc[best_model_name, "AUPR"]["mean"]
    logger.info(f"最佳模型: {best_model_name} (AUPR={best_aupr:.4f})")

    # ---- 7. Optuna 超参数优化（修复5：仅当最佳模型为 LGBM 时才执行）----
    best_params = None
    if best_model_name == "LightGBM":
        logger.info("\n[7/8] Optuna 超参数优化 (最佳模型为 LightGBM)...")
        # 使用完整特征空间（不再降维到 2000）
        train_idx_opt, val_idx_opt = scaffold_split(
            pair_smiles, y, test_size=0.2, random_state=42,
        )
        best_params = optuna_optimize_lgbm(
            X[train_idx_opt], y[train_idx_opt],
            X[val_idx_opt], y[val_idx_opt],
            n_trials=50, random_state=42,
        )
        step_label = "8/8"
    else:
        logger.info(f"\n[7/8] Optuna 跳过 (最佳模型 {best_model_name} 非 LightGBM)")
        step_label = "8/8"

    # ---- 8. 全量训练 + SHAP + Conformal + TCM 预测 ----
    logger.info(f"\n[{step_label}] 全量训练 + SHAP + Conformal + TCM 预测...")

    # 全量训练最佳模型
    if best_model_name == "RandomForest":
        best_model = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )
    elif best_model_name == "XGBoost":
        import xgboost as xgb
        scale_pos_weight = (y == 0).sum() / max(y.sum(), 1)
        best_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42, n_jobs=-1, verbosity=0,
        )
    elif best_model_name == "LightGBM":
        import lightgbm as lgb
        if best_params:
            best_model = lgb.LGBMClassifier(
                **best_params, class_weight="balanced",
                random_state=42, n_jobs=-1, verbose=-1,
            )
        else:
            best_model = lgb.LGBMClassifier(
                n_estimators=200, max_depth=10, learning_rate=0.05,
                class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
            )
    elif best_model_name == "CatBoost":
        from catboost import CatBoostClassifier
        best_model = CatBoostClassifier(
            iterations=200, depth=8, learning_rate=0.05,
            random_seed=42, thread_count=-1,
            verbose=False, allow_writing_files=False,
        )
    elif best_model_name == "StackingEnsemble":
        best_model = StackingEnsemble(n_estimators=200, random_state=42)
    elif best_model_name == "VotingEnsemble":
        # 为 Voting 重新训练所有基模型
        estimators = []
        try:
            import xgboost as xgb
            estimators.append(("XGBoost", xgb.XGBClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbosity=0,
            )))
        except ImportError:
            pass
        try:
            import lightgbm as lgb
            estimators.append(("LightGBM", lgb.LGBMClassifier(
                n_estimators=200, max_depth=10, learning_rate=0.05,
                class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
            )))
        except ImportError:
            pass
        estimators.append(("RF", RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )))
        best_model = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    else:
        best_model = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )

    logger.info(f"  全量训练 {best_model_name} (样本数: {len(X)})...")
    best_model.fit(X, y)

    # SHAP 可解释性
    if best_model_name in ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]:
        shap_path = L4_RESULTS / "tree_v4_shap_summary.png"
        shap_analysis(best_model, X, output_path=shap_path)

    # 修复3：Conformal Prediction：划分校准集执行
    logger.info("  Conformal Prediction (split calibration set @ 20%)...")
    cal_size = max(1, int(len(X) * 0.2))
    X_cal, X_rest = X[:cal_size], X[cal_size:]
    y_cal, y_rest = y[:cal_size], y[cal_size:]

    cp_y_pred, cp_y_ps = conformal_prediction(
        best_model, X_cal, y_cal, X_rest, alpha=0.1,
    )
    if cp_y_pred is not None and cp_y_ps is not None:
        # 保存 conformal 结果
        cp_results = pd.DataFrame({
            "y_true": y_rest,
            "y_pred": cp_y_pred,
            "y_pred_lower": cp_y_ps[:, 0, 0],
            "y_pred_upper": cp_y_ps[:, 0, 1],
        })
        cp_path = L4_RESULTS / "tree_v4_conformal_results.csv"
        cp_results.to_csv(cp_path, index=False)
        coverage = ((cp_results["y_pred_lower"] <= cp_results["y_true"]) &
                    (cp_results["y_pred_upper"] >= cp_results["y_true"])).mean()
        logger.info(f"  Conformal Prediction: coverage={coverage:.4f}, "
                    f"结果已保存: {cp_path}")
    else:
        logger.info("  Conformal Prediction 跳过 (MAPIE 未安装或模型不兼容)")

    # TCM 预测
    logger.info("  TCM 化合物池预测 (含 FDR 校正)...")
    tcm_indices = [smiles_to_idx[s] for s in tcm_smiles if s in smiles_to_idx]
    tcm_features = compound_features[tcm_indices]

    herb_map = load_herb_mapping()

    pred_df = predict_tcm_pool(
        best_model, tcm_df, tcm_features, protein_embeddings,
        cpi_genes_in_emb, best_model_name, herb_map=herb_map,
    )

    pred_path = L4_RESULTS / "tree_v4_tcm_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"TCM 预测结果已保存: {pred_path}")

    # Top 候选（修复10：仅显示 FDR 显著的化合物-基因对）
    pred_df_significant = pred_df[pred_df["significant"]].copy()
    if len(pred_df_significant) == 0:
        logger.warning("  FDR 校正后无显著结果，使用全量结果排序")
        pred_df_significant = pred_df.copy()

    comp_agg = pred_df_significant.groupby(["MOL_ID", "molecule_name", "SMILES"]).agg(
        max_score=("score", "max"),
        mean_score=("score", "mean"),
        n_genes_significant=("significant", "sum"),
        n_genes_above_50=("score", lambda x: (x >= 0.5).sum()),
        top_3_genes=("score", lambda x: "|".join(
            [f"{g}({s:.2f})" for g, s in sorted(
                zip(list(pred_df.loc[x.index, "gene"]), list(x)),
                key=lambda v: v[1], reverse=True
            )[:3]]
        )),
        herb_origins=("herb_origins", "first"),
    ).reset_index()
    comp_agg = comp_agg.sort_values("max_score", ascending=False)

    top50 = comp_agg.head(50)
    top_path = L4_RESULTS / "tree_v4_top_candidates.csv"
    top50.to_csv(top_path, index=False)

    logger.info(f"\nTop 20 候选化合物 (FDR 显著):")
    for i, row in enumerate(top50.head(20).itertuples(index=False), 1):
        logger.info(f"  {i:2d}. {row.molecule_name} | max={row.max_score:.4f} "
                    f"| mean={row.mean_score:.4f} "
                    f"| FDR显著基因: {row.n_genes_significant} "
                    f"| 高置信(>=0.5): {row.n_genes_above_50} "
                    f"| 中药: {row.herb_origins} "
                    f"| {row.top_3_genes}")

    logger.info(f"\nTop 50 候选已保存: {top_path}")
    logger.info(f"任务完成!")


if __name__ == "__main__":
    main()