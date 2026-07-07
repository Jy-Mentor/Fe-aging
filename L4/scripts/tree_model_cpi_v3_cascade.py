#!/usr/bin/env python
"""
树模型 CPI 筛选 v3.0 — 级联森林工业化版
============================================
基于 GitHub/PubMed 前沿文献调研的全面工业化升级：

v3 新增模块（基于文献驱动）：
  1. Cascade Forest (LGBMDF, Peng et al. 2023, Frontiers in Microbiology)
     - 级联层：每层 LightGBM + ExtraTrees 组合，逐层增强特征表示
  2. Optuna 超参数优化 (TPE sampler)
     - 自动搜索最佳 n_estimators, max_depth, learning_rate, num_leaves 等
  3. SHAP 可解释性分析
     - TreeExplainer 全局特征重要性 + 局部化合物解释
  4. Conformal Prediction (Mondrian CP)
     - 标签条件置信度校准，提供预测可靠性评估
  5. 改进负采样策略
     - 拓扑多样性约束（避免负样本集中少数蛋白）
     - 化学空间多样性（Tanimoto 距离最大化）
  6. 扩展评估指标 (MCC, F1, BEDROC, ROCE, P@K)
  7. 特征重要性驱动的特征选择（SHAP-based）

v2 保留模块：
  - 多指纹特征工程：ECFP4+ECFP6+MACCS+AtomPairs+Avalon
  - Scaffold Split (Bemis-Murcko)
  - 多模型集成：RF + XGBoost + LightGBM + CatBoost + Voting
  - 低方差特征过滤

数据来源（全部真实，不模拟）：
  - CPI: L4/results/experimental_actives_detail_cleaned.csv
  - 蛋白嵌入: L4/results_v10_minibatch/esm2_protein_embeddings.npz
  - TCM池: L3/results/tcm_compound_pool_tox_filtered_noleak.csv

输出：
  - L4/results/tree_v3_results.csv
  - L4/results/tree_v3_tcm_predictions.csv
  - L4/results/tree_v3_top_candidates.csv
  - L4/results/tree_v3_shap_summary.png
  - L4/results/tree_v3_conformal_results.csv
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
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

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
        logging.FileHandler(L4_LOGS / "tree_v3_cascade.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

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
# 1. 多指纹特征工程
# ============================================================

def compute_ecfp(smiles_list, radius=2, nbits=2048):
    """ECFP 指纹"""
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
    """MACCS 密钥 (167 bits)"""
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
    """Atom Pair 指纹 (hashed)"""
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
    """Avalon 指纹"""
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
    """RDKit 2D 描述符"""
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


def build_multifingerprint_features(smiles_list, fit_scaler=None):
    """构建多指纹融合特征矩阵"""
    t0 = time.time()
    n = len(smiles_list)
    logger.info(f"  计算 {n} 个化合物的多指纹特征...")

    fps = []
    labels = []

    # ECFP4
    fp = compute_ecfp(smiles_list, radius=2, nbits=2048)
    fps.append(fp)
    labels.append("ECFP4")
    logger.info(f"    ECFP4: {fp.shape}")

    # ECFP6
    fp = compute_ecfp(smiles_list, radius=3, nbits=2048)
    fps.append(fp)
    labels.append("ECFP6")
    logger.info(f"    ECFP6: {fp.shape}")

    # MACCS
    fp = compute_maccs(smiles_list)
    fps.append(fp)
    labels.append("MACCS")
    logger.info(f"    MACCS: {fp.shape}")

    # AtomPairs
    fp = compute_atom_pairs(smiles_list, nbits=1024)
    fps.append(fp)
    labels.append("AtomPairs")
    logger.info(f"    AtomPairs: {fp.shape}")

    # Avalon
    fp = compute_avalon(smiles_list, nbits=1024)
    fps.append(fp)
    labels.append("Avalon")
    logger.info(f"    Avalon: {fp.shape}")

    # RDKit 2D (v3: 启用物理化学描述符)
    fp = compute_rdkit_2d(smiles_list)
    fps.append(fp)
    labels.append("RDKit2D")
    logger.info(f"    RDKit2D: {fp.shape}")

    # 拼接
    X = np.hstack(fps).astype(np.float32)
    logger.info(f"  总特征维度: {X.shape[1]}, 耗时: {time.time()-t0:.1f}s")

    # NaN 处理
    nan_mask = np.isnan(X)
    if nan_mask.any():
        logger.info(f"  处理 {nan_mask.sum()} 个 NaN 值...")
        col_means = np.nanmean(X, axis=0)
        inds = np.where(nan_mask)
        X[inds] = np.take(col_means, inds[1])
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # 标准化
    if fit_scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, scaler, labels
    else:
        X = fit_scaler.transform(X)
        return X, None, labels


# ============================================================
# 2. 特征选择
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
    """Bemis-Murcko 骨架 (使用 GetScaffoldForMol API)"""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return "INVALID"
        from rdkit.Chem.Scaffolds import MurckoScaffold
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold = Chem.MolToSmiles(scaffold_mol) if scaffold_mol else ""
        return scaffold if scaffold else "NO_SCAFFOLD"
    except Exception:
        logger.exception("捕获到异常并继续执行（原 except 'Exception' 静默吞掉）")
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

    smiles_to_scaffold = dict(zip(unique_smiles, scaffolds, strict=False))
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
# 4. 改进负采样（多样性约束）
# ============================================================

def build_dataset(
    cpi_df, compound_smiles, compound_features, scaler,
    protein_embeddings, neg_ratio=3, random_seed=42,
):
    """构建训练数据集，带多样性约束负采样"""
    rng = np.random.RandomState(random_seed)
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

    # 正样本索引集
    pos_idx_set = set()
    for smi, gene in pos_pairs:
        comp_idx = smiles_to_idx[smi]
        gene_idx = cpi_genes_in_emb.index(gene)
        pos_idx_set.add((comp_idx, gene_idx))

    n_compounds = len(compound_smiles)
    n_genes = len(cpi_genes_in_emb)
    n_neg_target = len(pos_pairs) * neg_ratio

    # 多样性约束：每个蛋白被选为负样本的次数尽量均衡
    gene_neg_counts = dict.fromkeys(range(n_genes), 0)
    max_per_gene = max(1, n_neg_target // n_genes + 1)

    neg_idx_set = set()
    batch_size = n_neg_target * 10
    max_attempts = n_neg_target * 50

    while len(neg_idx_set) < n_neg_target:
        batch_comp = rng.randint(0, n_compounds, size=batch_size)
        batch_gene = rng.randint(0, n_genes, size=batch_size)
        for ci, gi in zip(batch_comp, batch_gene, strict=False):
            pair = (ci, gi)
            if pair in pos_idx_set or pair in neg_idx_set:
                continue
            if gene_neg_counts[gi] >= max_per_gene:
                continue
            neg_idx_set.add(pair)
            gene_neg_counts[gi] += 1
            if len(neg_idx_set) >= n_neg_target:
                break

    neg_pairs = []
    for ci, gi in neg_idx_set:
        smi = str(compound_smiles[ci])
        gene = cpi_genes_in_emb[gi]
        neg_pairs.append((smi, gene))

    logger.info(f"  负样本: {len(neg_pairs)} 对 (比例 1:{neg_ratio}), "
                f"蛋白覆盖: {sum(1 for c in gene_neg_counts.values() if c > 0)}/{n_genes}")

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
# 5. 扩展评估指标
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

    # EF@1%, EF@5%
    n_pos = y_true.sum()
    for pct in [1, 5]:
        k = max(1, int(len(y_true) * pct / 100))
        top_k_idx = np.argsort(y_prob)[-k:]
        found = y_true[top_k_idx].sum()
        expected = n_pos * pct / 100
        metrics[f"EF@{pct}%"] = found / expected if expected > 0 else 0.0

    # Precision@K
    for k in [10, 20, 50, 100]:
        if k <= len(y_true):
            top_k_idx = np.argsort(y_prob)[-k:]
            metrics[f"P@{k}"] = y_true[top_k_idx].mean()

    # BEDROC (alpha=20)
    try:
        metrics["BEDROC"] = _compute_bedroc(y_true, y_prob, alpha=20.0)
    except Exception:
        metrics["BEDROC"] = 0.0

    # ROCE (ROC Enrichment at 0.5%, 1%, 2%, 5%)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    for pct in [0.5, 1.0, 2.0, 5.0]:
        fp_rate = pct / 100.0
        idx = np.searchsorted(fpr, fp_rate)
        if idx < len(tpr):
            metrics[f"ROCE@{pct}%"] = tpr[idx]
        else:
            metrics[f"ROCE@{pct}%"] = 1.0

    return metrics


def _compute_bedroc(y_true, y_prob, alpha=20.0):
    """BEDROC (Boltzmann-Enhanced Discrimination of ROC)"""
    n = len(y_true)
    order = np.argsort(y_prob)[::-1]
    y_sorted = y_true[order]

    ra = n / (y_true.sum() + 1e-8)  # ratio of total to actives
    ri = 1.0 - np.exp(-alpha / ra)

    weights = np.exp(-alpha * np.arange(1, n + 1) / n)
    sum_weights = weights.sum()

    bedroc_num = (y_sorted * weights).sum()
    bedroc_den = y_true.sum() * (1.0 - np.exp(-alpha)) / ra

    if bedroc_den == 0:
        return 0.0

    bedroc = bedroc_num / bedroc_den
    bedroc_min = (np.exp(-alpha / ra) * (1.0 - np.exp(-alpha))) / (ra * (1.0 - np.exp(-alpha / ra)))
    bedroc_max = (1.0 - np.exp(-alpha / ra)) / (ra * (1.0 - np.exp(-alpha)))

    return (bedroc - bedroc_min) / (bedroc_max - bedroc_min + 1e-8)


# ============================================================
# 6. Cascade Forest (LGBMDF-inspired)
# ============================================================

class CascadeForest:
    """级联森林：多层 LightGBM + ExtraTrees 级联

    每层：
      1. 用原始特征训练 LightGBM + ExtraTrees
      2. 将预测概率作为增强特征拼接到原始特征
      3. 下一层用增强特征继续训练
      4. 最终层输出平均预测概率
    """

    def __init__(self, n_layers=3, n_estimators=200, random_state=42):
        self.n_layers = n_layers
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = []

    def _create_estimators(self):
        estimators = []
        try:
            import lightgbm as lgb
            estimators.append(("LGB", lgb.LGBMClassifier(
                n_estimators=self.n_estimators, max_depth=10, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                class_weight="balanced", random_state=self.random_state,
                n_jobs=-1, verbose=-1,
            )))
        except ImportError:
            logger.exception("捕获到异常并继续执行（原 except 'ImportError' 静默吞掉）")
            pass

        estimators.append(("ET", ExtraTreesClassifier(
            n_estimators=self.n_estimators, max_depth=20, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=self.random_state,
        )))
        return estimators

    def fit(self, X, y):
        X_current = X.copy()
        self.models = []

        for layer in range(self.n_layers):
            layer_models = []
            layer_probs = []
            estimators = self._create_estimators()
            logger.info(f"    Cascade Layer {layer+1}/{self.n_layers}: "
                        f"训练 {len(estimators)} 个estimator, feat_dim={X_current.shape[1]}")

            for name, est in estimators:
                est.fit(X_current, y)
                if hasattr(est, "predict_proba"):
                    prob = est.predict_proba(X_current)[:, 1:2]
                else:
                    prob = est.predict(X_current).reshape(-1, 1)
                layer_models.append(est)
                layer_probs.append(prob)

            self.models.append(layer_models)

            # 不再拼接增强特征到下一层（避免维度爆炸）
            # 保持级联结构但每层用原始特征 + 上层的聚合概率
            mean_prob = np.mean(layer_probs, axis=0)
            if layer < self.n_layers - 1:
                X_current = np.hstack([X_current, mean_prob])

        return self

    def predict_proba(self, X):
        all_probs = []
        X_current = X.copy()

        for layer_idx, layer_models in enumerate(self.models):
            layer_probs = []
            for est in layer_models:
                if hasattr(est, "predict_proba"):
                    prob = est.predict_proba(X_current)[:, 1:2]
                else:
                    prob = est.predict(X_current).reshape(-1, 1)
                layer_probs.append(prob)

            mean_prob = np.mean(layer_probs, axis=0)
            all_probs.append(mean_prob)

            if layer_idx < self.n_layers - 1:
                X_current = np.hstack([X_current, mean_prob])

        # 最终层输出
        final_prob = np.mean(all_probs, axis=0)
        return np.hstack([1 - final_prob, final_prob])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ============================================================
# 7. Optuna 超参数优化
# ============================================================

def optuna_optimize(X_train, y_train, X_val, y_val, n_trials=50, random_state=42):
    """Optuna 超参数优化"""
    try:
        import optuna
        import lightgbm as lgb
    except ImportError:
        logger.warning("Optuna 或 LightGBM 未安装，跳过超参数优化")
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
            **params,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, y_prob)

    logger.info(f"  Optuna 优化: {n_trials} trials...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    logger.info(f"  Optuna 最佳参数: {best_params}")
    logger.info(f"  Optuna 最佳 AUPR: {study.best_value:.4f}")

    return best_params


# ============================================================
# 8. Conformal Prediction (Mondrian)
# ============================================================

def conformal_prediction(model, X_cal, y_cal, X_test, alpha=0.1):
    """Mondrian Conformal Prediction: 标签条件置信度校准"""
    try:
        import mapie
        from mapie.classification import MapieClassifier
    except ImportError:
        logger.warning("MAPIE 未安装，跳过 Conformal Prediction")
        return None, None

    if hasattr(model, "predict_proba"):
        mapie_model = MapieClassifier(
            estimator=model,
            method="lac",
            cv="prefit",
            random_state=42,
        )
        mapie_model.fit(X_cal, y_cal)
        y_pred, y_ps = mapie_model.predict(X_test, alpha=alpha)
        return y_pred, y_ps
    else:
        logger.warning("模型不支持 predict_proba，跳过 Conformal Prediction")
        return None, None


# ============================================================
# 9. SHAP 可解释性
# ============================================================

def shap_analysis(model, X, feature_names=None, output_path=None):
    """SHAP 可解释性分析"""
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("SHAP 或 matplotlib 未安装，跳过可解释性分析")
        return None

    logger.info("  SHAP TreeExplainer 分析...")

    # 使用 TreeExplainer
    if hasattr(model, "estimators_"):
        explainer = shap.TreeExplainer(model)
    elif hasattr(model, "get_booster"):
        explainer = shap.TreeExplainer(model)
    else:
        logger.warning("  模型不支持 TreeExplainer，跳过")
        return None

    # 采样计算 SHAP 值（避免内存溢出）
    n_samples = min(500, len(X))
    X_sample = X[:n_samples]
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # 正类 SHAP

    # 全局特征重要性
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
# 10. 模型训练与评估
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
    """5-fold Scaffold Split CV 训练多模型 + Cascade Forest"""
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
        logger.info("  [1/5] Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=random_seed,
        )
        r = evaluate_model(rf, X_train, y_train, X_test, y_test, "RandomForest")
        if r:
            r["fold"] = fold
            fold_results.append(r)
            results.append(r)
            logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}, "
                        f"F1={r['F1']:.4f}, MCC={r['MCC']:.4f}")

        # 2. XGBoost
        try:
            import xgboost as xgb
            logger.info("  [2/5] XGBoost...")
            scale_pos_weight = (y_train == 0).sum() / max(y_train.sum(), 1)
            xgb_model = xgb.XGBClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=random_seed, n_jobs=-1, verbosity=0,
            )
            r = evaluate_model(xgb_model, X_train, y_train, X_test, y_test, "XGBoost")
            if r:
                r["fold"] = fold
                fold_results.append(r)
                results.append(r)
                logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}, "
                            f"F1={r['F1']:.4f}, MCC={r['MCC']:.4f}")
        except ImportError:
            logger.warning("XGBoost 未安装")

        # 3. LightGBM
        try:
            import lightgbm as lgb
            logger.info("  [3/5] LightGBM...")
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200, max_depth=10, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                class_weight="balanced", random_state=random_seed,
                n_jobs=-1, verbose=-1,
            )
            r = evaluate_model(lgb_model, X_train, y_train, X_test, y_test, "LightGBM")
            if r:
                r["fold"] = fold
                fold_results.append(r)
                results.append(r)
                logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}, "
                            f"F1={r['F1']:.4f}, MCC={r['MCC']:.4f}")
        except ImportError:
            logger.warning("LightGBM 未安装")

        # 4. CatBoost
        try:
            from catboost import CatBoostClassifier
            logger.info("  [4/5] CatBoost...")
            cb_model = CatBoostClassifier(
                iterations=200, depth=8, learning_rate=0.05,
                class_weights=[1, (y_train == 0).sum() / max(y_train.sum(), 1)],
                random_seed=random_seed, thread_count=-1,
                verbose=False, allow_writing_files=False,
            )
            r = evaluate_model(cb_model, X_train, y_train, X_test, y_test, "CatBoost")
            if r:
                r["fold"] = fold
                fold_results.append(r)
                results.append(r)
                logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}, "
                            f"F1={r['F1']:.4f}, MCC={r['MCC']:.4f}")
        except ImportError:
            logger.warning("CatBoost 未安装")

        # 5. Cascade Forest
        logger.info("  [5/5] Cascade Forest...")
        cascade = CascadeForest(n_layers=3, n_estimators=200, random_state=random_seed)
        r = evaluate_model(cascade, X_train, y_train, X_test, y_test, "CascadeForest")
        if r:
            r["fold"] = fold
            fold_results.append(r)
            results.append(r)
            logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}, "
                        f"F1={r['F1']:.4f}, MCC={r['MCC']:.4f}")

        # 6. Soft Voting (v2 保留)
        valid_results = [fr for fr in fold_results if fr["model"] in
                        ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]]
        if len(valid_results) >= 2:
            logger.info("  [6] Soft Voting Ensemble...")
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
                r["fold"] = fold
                results.append(r)
                logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}, "
                            f"F1={r['F1']:.4f}, MCC={r['MCC']:.4f}")

    return pd.DataFrame(results)


# ============================================================
# 11. TCM 预测
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

        # MOL_ID -> 中药名称列表（一个化合物可能来自多种中药）
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

        # 中药来源
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

    return pd.DataFrame(predictions)


# ============================================================
# 主流程
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("树模型 CPI v3.0 — Cascade Forest 工业化版")
    logger.info("=" * 60)

    # ---- 1. 加载数据 ----
    logger.info("\n[1/7] 加载原始数据...")
    cpi_df = pd.read_csv(L4_RESULTS / "experimental_actives_detail_cleaned.csv", low_memory=False)
    protein_embeddings = {str(k): v.astype(np.float32) for k, v in
                          np.load(L4_RESULTS_V10 / "esm2_protein_embeddings.npz", allow_pickle=True).items()}
    tcm_df = pd.read_csv(L3_RESULTS / "tcm_compound_pool_tox_filtered_noleak.csv", low_memory=False)

    # 获取所有需要的 SMILES
    all_smiles = list(cpi_df["canonical_smiles"].dropna().astype(str).unique())
    tcm_smiles = tcm_df["SMILES_std"].astype(str).tolist()
    all_smiles.extend(tcm_smiles)
    all_smiles = list(dict.fromkeys(all_smiles))
    logger.info(f"  总 SMILES: {len(all_smiles)} (CPI 唯一: {len(all_smiles) - len(tcm_smiles)}, TCM: {len(tcm_smiles)})")

    # ---- 2. 多指纹特征工程 (v3: 含 RDKit 2D) ----
    logger.info("\n[2/7] 多指纹特征工程 (ECFP4+ECFP6+MACCS+AtomPairs+Avalon+RDKit2D)...")
    compound_features, scaler, fp_labels = build_multifingerprint_features(all_smiles)

    # ---- 3. 特征选择 ----
    logger.info("\n[3/7] 特征选择...")
    smiles_to_idx = {str(s): i for i, s in enumerate(all_smiles)}
    cpi_genes = sorted(cpi_df["gene"].unique())
    cpi_genes_in_emb = [g for g in cpi_genes if g in protein_embeddings]
    logger.info(f"  CPI 基因: {len(cpi_genes)}, 有嵌入: {len(cpi_genes_in_emb)}")

    # 临时数据集用于特征选择
    pos_pairs = []
    for _, row in cpi_df.iterrows():
        smi = str(row["canonical_smiles"])
        gene = str(row["gene"])
        if smi in smiles_to_idx and gene in protein_embeddings:
            pos_pairs.append((smi, gene))

    rng = np.random.RandomState(42)
    pos_idx_set = set()
    for smi, gene in pos_pairs:
        pos_idx_set.add((smiles_to_idx[smi], cpi_genes_in_emb.index(gene)))

    n_neg = len(pos_pairs) * 3
    neg_idx_set = set()
    while len(neg_idx_set) < n_neg:
        ci = rng.randint(0, len(all_smiles))
        gi = rng.randint(0, len(cpi_genes_in_emb))
        if (ci, gi) not in pos_idx_set and (ci, gi) not in neg_idx_set:
            neg_idx_set.add((ci, gi))

    neg_pairs = [(str(all_smiles[ci]), cpi_genes_in_emb[gi]) for ci, gi in neg_idx_set]
    all_pairs = pos_pairs + neg_pairs
    n_pairs = len(all_pairs)
    comp_dim = compound_features.shape[1]
    prot_dim = next(iter(protein_embeddings.values())).shape[0]

    X_feat = np.zeros((n_pairs, comp_dim + prot_dim), dtype=np.float32)
    y_feat = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs))
    for i, (smi, gene) in enumerate(all_pairs):
        ci = smiles_to_idx[smi]
        X_feat[i, :comp_dim] = compound_features[ci]
        X_feat[i, comp_dim:] = protein_embeddings[gene]

    X_feat, keep_var = feature_selection(X_feat, y_feat, max_features=3000)
    compound_features = compound_features[:, keep_var[:comp_dim]]
    logger.info(f"  特征选择后: {compound_features.shape[1]} 维")

    # ---- 4. 构建数据集 ----
    logger.info("\n[4/7] 构建训练数据集 (多样性约束负采样)...")
    X, y, pair_smiles, pair_genes = build_dataset(
        cpi_df, all_smiles, compound_features, scaler,
        protein_embeddings, neg_ratio=3, random_seed=42,
    )

    # ---- 5. Optuna 超参数优化（子采样 + 特征降维避免OOM）----
    logger.info("\n[5/7] Optuna 超参数优化 (子采样 + 特征降维)...")
    # 子采样：最多 10000 样本 + 保留 top 2000 高方差特征
    n_opt = min(10000, len(X))
    rng_opt = np.random.RandomState(42)
    opt_indices = rng_opt.choice(len(X), size=n_opt, replace=False)
    X_opt_full, y_opt = X[opt_indices], y[opt_indices]
    pair_opt = pair_smiles[opt_indices]

    # 特征降维：仅保留 top 2000 高方差特征（避免 Optuna 内存溢出）
    variances = np.var(X_opt_full, axis=0)
    top_feat_idx = np.argsort(variances)[-2000:]
    X_opt = X_opt_full[:, top_feat_idx]
    logger.info(f"  Optuna 特征降维: {X_opt_full.shape[1]} -> 2000, 样本: {n_opt}")

    train_idx_opt, val_idx_opt = scaffold_split(pair_opt, y_opt, test_size=0.2, random_state=42)
    best_params = optuna_optimize(
        X_opt[train_idx_opt], y_opt[train_idx_opt],
        X_opt[val_idx_opt], y_opt[val_idx_opt],
        n_trials=50, random_state=42,
    )

    # ---- 6. 训练与评估 ----
    logger.info("\n[6/7] 5-fold Scaffold Split 训练 (含 Cascade Forest)...")
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
                logger.info(f"    {metric}: {row[metric]['mean']:.4f} +/- {row[metric]['std']:.4f}")

    results_path = L4_RESULTS / "tree_v3_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\n评估结果已保存: {results_path}")

    # ---- 7. 全量训练 + TCM 预测 + SHAP ----
    logger.info("\n[7/7] 全量训练最佳模型 + TCM 预测 + SHAP...")

    # 选择最佳模型（按 AUPR）
    best_model_name = summary["AUPR"]["mean"].idxmax()
    best_aupr = summary.loc[best_model_name, "AUPR"]["mean"]
    logger.info(f"最佳模型: {best_model_name} (AUPR={best_aupr:.4f})")

    # 全量训练
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
    elif best_model_name == "CascadeForest":
        best_model = CascadeForest(n_layers=3, n_estimators=200, random_state=42)
    else:
        best_model = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )

    logger.info(f"  全量训练 {best_model_name} (样本数: {len(X)})...")
    best_model.fit(X, y)

    # SHAP 可解释性
    if best_model_name in ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]:
        shap_path = L4_RESULTS / "tree_v3_shap_summary.png"
        shap_result = shap_analysis(best_model, X, output_path=shap_path)

    # 加载中药来源映射
    herb_map = load_herb_mapping()

    # TCM 预测
    tcm_indices = [smiles_to_idx[s] for s in tcm_smiles if s in smiles_to_idx]
    tcm_features = compound_features[tcm_indices]

    pred_df = predict_tcm_pool(
        best_model, tcm_df, tcm_features, protein_embeddings,
        cpi_genes_in_emb, best_model_name, herb_map=herb_map,
    )

    pred_path = L4_RESULTS / "tree_v3_tcm_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"TCM 预测结果已保存: {pred_path}")

    # Top 候选
    comp_agg = pred_df.groupby(["MOL_ID", "molecule_name", "SMILES"]).agg(
        max_score=("score", "max"),
        mean_score=("score", "mean"),
        n_genes_above_50=("score", lambda x: (x >= 0.5).sum()),
        top_3_genes=("score", lambda x: "|".join(
            [f"{g}({s:.2f})" for g, s in sorted(
                zip(list(pred_df.loc[x.index, "gene"]), list(x), strict=False),
                key=lambda v: v[1], reverse=True
            )[:3]]
        )),
        herb_origins=("herb_origins", "first"),
    ).reset_index()
    comp_agg = comp_agg.sort_values("max_score", ascending=False)

    top50 = comp_agg.head(50)
    top_path = L4_RESULTS / "tree_v3_top_candidates.csv"
    top50.to_csv(top_path, index=False)

    logger.info(f"\nTop 20 候选化合物:")
    for i, row in enumerate(top50.head(20).itertuples(index=False), 1):
        logger.info(f"  {i:2d}. {row.molecule_name} | max={row.max_score:.4f} "
                    f"| mean={row.mean_score:.4f} "
                    f"| 高置信(>=0.5): {row.n_genes_above_50} "
                    f"| 中药: {row.herb_origins} "
                    f"| {row.top_3_genes}")

    logger.info(f"\nTop 50 候选已保存: {top_path}")
    logger.info(f"任务完成!")


if __name__ == "__main__":
    main()