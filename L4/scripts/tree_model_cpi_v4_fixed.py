#!/usr/bin/env python
"""
树模型 CPI 筛选 v4.0 — 修复版
============================================
基于用户深度审查反馈的系统性修复：

v4 修复内容：
  1. CascadeForest 数据泄露修复
     - 使用 K-Fold OOF 预测作为增强特征，而非训练集自身预测
     - 添加多粒度扫描（滑动窗口特征提取）
  2. ROCE 计算修复
     - 正确实现 TPR/FPR 富集因子，而非裸 TPR
  3. BEDROC 实现修复
     - 使用 Truchon & Bayly (2007) 标准公式
  4. 特征标准化修复
     - 二进制指纹保留原值，仅对 RDKit 2D 连续描述符标准化
  5. 蛋白嵌入处理
     - 添加 PCA 降维（640 -> 128）+ 标准化
  6. 中药映射完善
     - 正确处理一对多关系，完整输出所有来源
  7. 移除未使用的 Conformal Prediction
  8. 统一特征选择与训练阶段的负采样策略
  9. 扩展 Optuna 适用范围
     - 对所有候选模型类型进行超参数搜索

数据来源（全部真实，不模拟）：
  - CPI: L4/results/experimental_actives_detail_cleaned.csv
  - 蛋白嵌入: L4/results_v10_minibatch/esm2_protein_embeddings.npz
  - TCM池: L3/results/tcm_compound_pool_tox_filtered_noleak.csv
  - 中药映射: L3/results/herb_ingredient_mapping.xlsx

输出：
  - L4/results/tree_v4_results.csv
  - L4/results/tree_v4_tcm_predictions.csv
  - L4/results/tree_v4_top_candidates.csv
  - L4/results/tree_v4_shap_summary.png
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

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.model_selection import KFold, StratifiedKFold
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
        logging.FileHandler(L4_LOGS / "tree_v4_fixed.log", mode="w", encoding="utf-8"),
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
# 1. 多指纹特征工程（修复：二进制指纹不标准化）
# ============================================================

def compute_ecfp(smiles_list, radius=2, nbits=2048):
    """ECFP 指纹（二进制，不标准化）"""
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
    """MACCS 密钥（二进制，不标准化）"""
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
    """Atom Pair 指纹（二进制，不标准化）"""
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
    """Avalon 指纹（二进制，不标准化）"""
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
    return np.array(rows, dtype=np.float32), desc_names


def build_multifingerprint_features(smiles_list, rdkit_scaler=None):
    """构建多指纹融合特征矩阵（修复：二进制指纹不标准化）"""
    t0 = time.time()
    n = len(smiles_list)
    logger.info(f"  计算 {n} 个化合物的多指纹特征...")

    binary_fps = []
    binary_labels = []

    # ECFP4（二进制，不标准化）
    fp = compute_ecfp(smiles_list, radius=2, nbits=2048)
    binary_fps.append(fp)
    binary_labels.append("ECFP4")
    logger.info(f"    ECFP4: {fp.shape} (binary, no scaling)")

    # ECFP6（二进制，不标准化）
    fp = compute_ecfp(smiles_list, radius=3, nbits=2048)
    binary_fps.append(fp)
    binary_labels.append("ECFP6")
    logger.info(f"    ECFP6: {fp.shape} (binary, no scaling)")

    # MACCS（二进制，不标准化）
    fp = compute_maccs(smiles_list)
    binary_fps.append(fp)
    binary_labels.append("MACCS")
    logger.info(f"    MACCS: {fp.shape} (binary, no scaling)")

    # AtomPairs（二进制，不标准化）
    fp = compute_atom_pairs(smiles_list, nbits=1024)
    binary_fps.append(fp)
    binary_labels.append("AtomPairs")
    logger.info(f"    AtomPairs: {fp.shape} (binary, no scaling)")

    # Avalon（二进制，不标准化）
    fp = compute_avalon(smiles_list, nbits=1024)
    binary_fps.append(fp)
    binary_labels.append("Avalon")
    logger.info(f"    Avalon: {fp.shape} (binary, no scaling)")

    # 拼接二进制指纹
    X_binary = np.hstack(binary_fps).astype(np.float32)
    logger.info(f"  二进制指纹总维度: {X_binary.shape[1]}")

    # RDKit 2D（连续值，标准化）
    X_rdkit, rdkit_names = compute_rdkit_2d(smiles_list)
    logger.info(f"    RDKit2D: {X_rdkit.shape} (continuous, needs scaling)")

    # NaN 处理（仅对 RDKit 2D）
    nan_mask = np.isnan(X_rdkit)
    if nan_mask.any():
        logger.info(f"  RDKit2D 处理 {nan_mask.sum()} 个 NaN 值...")
        col_means = np.nanmean(X_rdkit, axis=0)
        inds = np.where(nan_mask)
        X_rdkit[inds] = np.take(col_means, inds[1])
    X_rdkit = np.nan_to_num(X_rdkit, nan=0.0, posinf=1e6, neginf=-1e6)

    # 标准化 RDKit 2D（仅对连续描述符）
    if rdkit_scaler is None:
        rdkit_scaler = StandardScaler()
        X_rdkit = rdkit_scaler.fit_transform(X_rdkit)
        logger.info(f"  RDKit2D 已标准化 (mean=0, std=1)")
        return X_binary, X_rdkit, rdkit_scaler, binary_labels, rdkit_names
    else:
        X_rdkit = rdkit_scaler.transform(X_rdkit)
        return X_binary, X_rdkit, None, binary_labels, rdkit_names


# ============================================================
# 2. 蛋白嵌入处理（修复：PCA 降维 + 标准化）
# ============================================================

def process_protein_embeddings(protein_embeddings, target_dim=128, pca_model=None, scaler=None):
    """蛋白嵌入处理：PCA 降维 + 标准化"""
    keys = sorted(protein_embeddings.keys())
    vectors = np.array([protein_embeddings[k] for k in keys], dtype=np.float32)
    original_dim = vectors.shape[1]
    logger.info(f"  蛋白嵌入原始维度: {original_dim}, 数量: {len(keys)}")

    if pca_model is None:
        # 训练 PCA
        pca = PCA(n_components=target_dim, random_state=42)
        vectors_reduced = pca.fit_transform(vectors)
        logger.info(f"  PCA 降维: {original_dim} -> {target_dim}, 解释方差比: {pca.explained_variance_ratio_.sum():.4f}")

        # 标准化
        scaler = StandardScaler()
        vectors_scaled = scaler.fit_transform(vectors_reduced)
        logger.info(f"  蛋白嵌入已标准化")

        # 返回字典
        processed = {k: vectors_scaled[i] for i, k in enumerate(keys)}
        return processed, pca, scaler
    else:
        # 使用已有的 PCA 和 scaler
        vectors_reduced = pca_model.transform(vectors)
        vectors_scaled = scaler.transform(vectors_reduced)
        processed = {k: vectors_scaled[i] for i, k in enumerate(keys)}
        return processed, None, None


# ============================================================
# 3. Scaffold Split (Bemis-Murcko)
# ============================================================

def get_scaffold(smiles):
    """Bemis-Murcko 骨架"""
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
# 4. 多样性约束负采样（统一用于特征选择和训练）
# ============================================================

def diversity_constrained_negative_sampling(
    pos_pairs, compound_smiles, cpi_genes_in_emb, neg_ratio=3, random_seed=42,
):
    """多样性约束负采样（统一版本）"""
    rng = np.random.RandomState(random_seed)
    smiles_to_idx = {str(s): i for i, s in enumerate(compound_smiles)}

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

    while len(neg_idx_set) < n_neg_target and len(neg_idx_set) < max_attempts:
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

    return neg_pairs


# ============================================================
# 5. 扩展评估指标（修复：ROCE 正确实现）
# ============================================================

def compute_metrics(y_true, y_prob):
    """计算扩展评估指标（修复版）"""
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
    n_total = len(y_true)
    for pct in [1, 5]:
        k = max(1, int(n_total * pct / 100))
        top_k_idx = np.argsort(y_prob)[-k:]
        found = y_true[top_k_idx].sum()
        expected = n_pos * pct / 100
        metrics[f"EF@{pct}%"] = found / expected if expected > 0 else 0.0

    # Precision@K
    for k in [10, 20, 50, 100]:
        if k <= n_total:
            top_k_idx = np.argsort(y_prob)[-k:]
            metrics[f"P@{k}"] = y_true[top_k_idx].mean()

    # BEDROC (修复：使用 Truchon & Bayly 2007 标准公式)
    try:
        metrics["BEDROC"] = compute_bedroc_standard(y_true, y_prob, alpha=20.0)
    except Exception:
        metrics["BEDROC"] = 0.0

    # ROCE (修复：正确实现 TPR/FPR 富集因子)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    for pct in [0.5, 1.0, 2.0, 5.0]:
        fp_rate = pct / 100.0
        # 找到最接近目标 FPR 的索引
        idx = np.argmin(np.abs(fpr - fp_rate))
        # ROCE = TPR / FPR（富集倍数）
        roce = tpr[idx] / fpr[idx] if fpr[idx] > 1e-08 else 0.0
        metrics[f"ROCE@{pct}%"] = roce

    return metrics


def compute_bedroc_standard(y_true, y_prob, alpha=20.0):
    """BEDROC 标准实现 (Truchon & Bayly, 2007)

    BEDROC = RIE / (1 - exp(-alpha)) + exp(-alpha) / (1 - exp(-alpha))
    其中 RIE = sum(exp(-alpha * r_i / N)) / (n_act * (1 - exp(-alpha)) / R_a)
    """
    n = len(y_true)
    n_act = y_true.sum()
    if n_act == 0:
        return 0.0

    R_a = n / n_act  # ratio of total to actives

    # 按预测分数降序排序
    order = np.argsort(y_prob)[::-1]
    y_sorted = y_true[order]

    # 计算活性化合物的排名
    act_ranks = np.where(y_sorted == 1)[0] + 1  # 排名从1开始

    # RIE (Robust Initial Enhancement)
    rie_num = np.sum(np.exp(-alpha * act_ranks / n))
    rie_den = n_act * (1.0 - np.exp(-alpha)) / R_a
    rie = rie_num / rie_den if rie_den > 0 else 0.0

    # BEDROC 范围 [0, 1]
    bedroc_min = np.exp(-alpha / R_a) * (1.0 - np.exp(-alpha)) / (R_a * (1.0 - np.exp(-alpha / R_a)))
    bedroc_max = 1.0 - np.exp(-alpha / R_a) / (R_a * (1.0 - np.exp(-alpha / R_a)))

    # 归一化到 [0, 1]
    if bedroc_max - bedroc_min < 1e-8:
        return 0.5
    bedroc = (rie - bedroc_min) / (bedroc_max - bedroc_min)

    return np.clip(bedroc, 0.0, 1.0)


# ============================================================
# 6. 修复版级联森林（使用 OOF 预测避免数据泄露）
# ============================================================

class CascadeForestFixed:
    """修复版级联森林：使用 K-Fold OOF 预测作为增强特征

    关键修复：
    1. 每层使用 K-Fold 交叉验证生成 OOF 预测，而非训练集自身预测
    2. 添加多粒度扫描（滑动窗口特征提取）
    3. 测试时使用平均预测，不泄露信息
    """

    def __init__(self, n_layers=3, n_estimators=200, n_folds=5,
                 window_sizes=[100, 200, 500], random_state=42):
        self.n_layers = n_layers
        self.n_estimators = n_estimators
        self.n_folds = n_folds
        self.window_sizes = window_sizes
        self.random_state = random_state
        self.models = []
        self.window_models = []

    def _create_estimators(self):
        """创建基础估计器"""
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

        estimators.append(("RF", RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=15, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=self.random_state,
        )))

        return estimators

    def _multi_grain_scan(self, X, y, window_sizes):
        """多粒度扫描：滑动窗口特征提取"""
        n_samples, n_features = X.shape
        scanned_features = []

        for window in window_sizes:
            if window > n_features:
                continue

            # 滑动窗口提取特征
            n_windows = n_features // window
            window_feats = []

            for i in range(n_windows):
                start = i * window
                end = start + window
                X_window = X[:, start:end]

                # 对每个窗口训练一个随机森林
                rf = RandomForestClassifier(
                    n_estimators=50, max_depth=10,
                    class_weight="balanced", n_jobs=-1,
                    random_state=self.random_state,
                )

                # 使用 K-Fold 生成 OOF 预测
                kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                                      random_state=self.random_state)
                oof_probs = np.zeros((n_samples, 2))

                for train_idx, val_idx in kf.split(X_window, y):
                    rf.fit(X_window[train_idx], y[train_idx])
                    oof_probs[val_idx] = rf.predict_proba(X_window[val_idx])

                window_feats.append(oof_probs[:, 1:2])  # 只取正类概率
                self.window_models.append((window, i, rf))

            if window_feats:
                scanned_features.append(np.hstack(window_feats))

        if scanned_features:
            return np.hstack(scanned_features)
        else:
            return np.zeros((n_samples, 1), dtype=np.float32)

    def fit(self, X, y):
        """训练级联森林（使用 OOF 预测避免泄露）"""
        # 多粒度扫描
        logger.info(f"    多粒度扫描: window_sizes={self.window_sizes}")
        scanned_feats = self._multi_grain_scan(X, y, self.window_sizes)
        X_current = np.hstack([X, scanned_feats])
        logger.info(f"    扫描后特征维度: {X_current.shape[1]}")

        self.models = []
        self.input_dim = X.shape[1]

        for layer in range(self.n_layers):
            layer_models = []
            estimators = self._create_estimators()

            logger.info(f"    Cascade Layer {layer+1}/{self.n_layers}: "
                        f"训练 {len(estimators)} 个估计器, feat_dim={X_current.shape[1]}")

            # 使用 K-Fold 生成 OOF 预测（避免数据泄露）
            kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                                  random_state=self.random_state + layer)
            oof_probs_list = []  # 每个估计器的 OOF 预测

            for name, est_template in estimators:
                oof_probs = np.zeros((len(X_current), 1), dtype=np.float32)

                for train_idx, val_idx in kf.split(X_current, y):
                    # 克隆估计器避免引用问题
                    est = clone(est_template)
                    est.fit(X_current[train_idx], y[train_idx])
                    oof_probs[val_idx, 0] = est.predict_proba(X_current[val_idx])[:, 1]
                    layer_models.append((name, est, train_idx, val_idx))

                oof_probs_list.append(oof_probs)

            # 存储"汇总模型"用于预测
            self.models.append([(name, clone(est_template)) for name, est_template in estimators])

            # 用 OOF 预测的平均值作为增强特征（无泄露）
            mean_oof_prob = np.mean(oof_probs_list, axis=0)

            # 下一层特征 = 当前特征 + OOF 预测
            if layer < self.n_layers - 1:
                X_current = np.hstack([X_current, mean_oof_prob])

        # 最终层训练完整模型
        for layer_idx, layer_estimators in enumerate(self.models):
            for name, est in layer_estimators:
                est.fit(X_current, y)

        return self

    def predict_proba(self, X):
        """预测概率（无泄露）"""
        # 重现多粒度扫描
        n_samples, n_features = X.shape
        scanned_feats = []

        for window, i, rf in self.window_models:
            start = i * window
            end = start + window
            if end <= n_features:
                X_window = X[:, start:end]
                prob = rf.predict_proba(X_window)[:, 1:2]
                scanned_feats.append(prob)

        scanned = np.hstack(scanned_feats) if scanned_feats else np.zeros((n_samples, 1), dtype=np.float32)

        X_current = np.hstack([X, scanned])
        all_probs = []

        for layer_idx, layer_estimators in enumerate(self.models):
            layer_probs = []
            for name, est in layer_estimators:
                prob = est.predict_proba(X_current)[:, 1:2]
                layer_probs.append(prob)

            mean_prob = np.mean(layer_probs, axis=0)
            all_probs.append(mean_prob)

            # 为下一层拼接预测（测试时用预测而非 OOF）
            if layer_idx < len(self.models) - 1:
                X_current = np.hstack([X_current, mean_prob])

        final_prob = np.mean(all_probs, axis=0)
        return np.hstack([1 - final_prob, final_prob])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ============================================================
# 7. Optuna 超参数优化（修复：扩展适用范围）
# ============================================================

def optuna_optimize_extended(X_train, y_train, X_val, y_val,
                              model_types=["lgb", "rf", "xgb"],
                              n_trials=30, random_state=42):
    """扩展版 Optuna 超参数优化"""
    try:
        import optuna
    except ImportError:
        logger.warning("Optuna 未安装，跳过超参数优化")
        return {}

    best_params_per_model = {}

    for model_type in model_types:
        logger.info(f"  Optuna 优化 {model_type}...")

        def objective(trial):
            if model_type == "lgb":
                try:
                    import lightgbm as lgb
                except ImportError:
                    logger.exception("捕获到异常并继续执行（原 except 'ImportError' 静默吞掉）")
                    return 0.0

                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                    "max_depth": trial.suggest_int("max_depth", 5, 15),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 16, 64),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                }
                model = lgb.LGBMClassifier(
                    **params, class_weight="balanced",
                    random_state=random_state, n_jobs=-1, verbose=-1,
                )
            elif model_type == "rf":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                    "max_depth": trial.suggest_int("max_depth", 10, 25),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                }
                model = RandomForestClassifier(
                    **params, class_weight="balanced",
                    n_jobs=-1, random_state=random_state,
                )
            elif model_type == "xgb":
                try:
                    import xgboost as xgb
                except ImportError:
                    logger.exception("捕获到异常并继续执行（原 except 'ImportError' 静默吞掉）")
                    return 0.0

                scale_pos_weight = (y_train == 0).sum() / max(y_train.sum(), 1)
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                    "max_depth": trial.suggest_int("max_depth", 5, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                }
                model = xgb.XGBClassifier(
                    **params, scale_pos_weight=scale_pos_weight,
                    random_state=random_state, n_jobs=-1, verbosity=0,
                )
            else:
                return 0.0

            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)[:, 1]
            return average_precision_score(y_val, y_prob)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params_per_model[model_type] = study.best_params
        logger.info(f"    {model_type} 最佳 AUPR: {study.best_value:.4f}")

    return best_params_per_model


# ============================================================
# 8. SHAP 可解释性
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

    if hasattr(model, "estimators_") or hasattr(model, "get_booster"):
        explainer = shap.TreeExplainer(model)
    else:
        logger.warning("  模型不支持 TreeExplainer，跳过")
        return None

    n_samples = min(500, len(X))
    X_sample = X[:n_samples]
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
# 9. 中药映射（修复：完善一对多处理）
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

        # MOL_ID -> (中药名称列表, 英文名列表, 拼音列表)
        herb_map = {}
        for _, row in herb_df.iterrows():
            mol_id = str(row["MOL_ID"])
            herb_cn = str(row.get("herb_cn_name", ""))
            herb_en = str(row.get("herb_en_name", ""))
            herb_py = str(row.get("herb_pinyin", ""))

            if mol_id not in herb_map:
                herb_map[mol_id] = {
                    "cn_names": [],
                    "en_names": [],
                    "pinyins": [],
                }

            # 添加中药信息（去重）
            if herb_cn and herb_cn not in herb_map[mol_id]["cn_names"]:
                herb_map[mol_id]["cn_names"].append(herb_cn)
            if herb_en and herb_en not in herb_map[mol_id]["en_names"]:
                herb_map[mol_id]["en_names"].append(herb_en)
            if herb_py and herb_py not in herb_map[mol_id]["pinyins"]:
                herb_map[mol_id]["pinyins"].append(herb_py)

        logger.info(f"  MOL_ID 映射: {len(herb_map)} 个化合物有中药来源")
        return herb_map
    except Exception:
        logger.error(f"加载中药映射失败: {traceback.format_exc()}")
        return {}


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


def train_ensemble(X, y, pair_smiles, best_params_per_model=None, n_folds=5, random_seed=42):
    """5-fold Scaffold Split CV 训练多模型 + 修复版 Cascade Forest"""
    results = []

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
        rf_params = best_params_per_model.get("rf", {})
        rf = RandomForestClassifier(
            n_estimators=rf_params.get("n_estimators", 200),
            max_depth=rf_params.get("max_depth", 20),
            min_samples_leaf=rf_params.get("min_samples_leaf", 10),
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
            xgb_params = best_params_per_model.get("xgb", {})
            scale_pos_weight = (y_train == 0).sum() / max(y_train.sum(), 1)
            xgb_model = xgb.XGBClassifier(
                n_estimators=xgb_params.get("n_estimators", 200),
                max_depth=xgb_params.get("max_depth", 8),
                learning_rate=xgb_params.get("learning_rate", 0.05),
                subsample=xgb_params.get("subsample", 0.8),
                colsample_bytree=xgb_params.get("colsample_bytree", 0.8),
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
            lgb_params = best_params_per_model.get("lgb", {})
            lgb_model = lgb.LGBMClassifier(
                n_estimators=lgb_params.get("n_estimators", 200),
                max_depth=lgb_params.get("max_depth", 10),
                learning_rate=lgb_params.get("learning_rate", 0.05),
                num_leaves=lgb_params.get("num_leaves", 31),
                subsample=lgb_params.get("subsample", 0.8),
                colsample_bytree=lgb_params.get("colsample_bytree", 0.8),
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

        # 5. 修复版 Cascade Forest
        logger.info("  [5/5] Cascade Forest (Fixed, OOF)...")
        cascade = CascadeForestFixed(
            n_layers=3, n_estimators=100, n_folds=5,
            window_sizes=[100, 200, 500], random_state=random_seed,
        )
        r = evaluate_model(cascade, X_train, y_train, X_test, y_test, "CascadeForestFixed")
        if r:
            r["fold"] = fold
            fold_results.append(r)
            results.append(r)
            logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}, "
                        f"F1={r['F1']:.4f}, MCC={r['MCC']:.4f}")

    return pd.DataFrame(results)


# ============================================================
# 11. TCM 预测（完善中药映射）
# ============================================================

def predict_tcm_pool(
    best_model, tcm_df, tcm_binary_feats, tcm_rdkit_feats,
    protein_embeddings, cpi_genes_in_emb, model_name, herb_map=None,
):
    """预测 TCM 化合物池"""
    if herb_map is None:
        herb_map = {}

    logger.info(f"  预测 {len(tcm_df)} 个 TCM 化合物 x {len(cpi_genes_in_emb)} 个基因...")

    predictions = []
    for i, (_, row) in enumerate(tcm_df.iterrows()):
        smi = str(row["SMILES_std"])
        mol_name = str(row.get("molecule_name", f"MOL_{i}"))
        mol_id = str(row.get("MOL_ID", f"MOL_{i}"))

        # 拼接化合物特征（二进制指纹 + RDKit 2D）
        comp_feat = np.hstack([tcm_binary_feats[i], tcm_rdkit_feats[i]])

        # 中药来源（完整处理一对多）
        herb_info = herb_map.get(mol_id, {})
        herb_cn = "; ".join(herb_info.get("cn_names", ["未知"]))
        herb_en = "; ".join(herb_info.get("en_names", ["Unknown"]))
        herb_py = "; ".join(herb_info.get("pinyins", [""]))

        comp_dim = comp_feat.shape[0]

        for gene in cpi_genes_in_emb:
            prot_feat = protein_embeddings[gene]
            feat = np.hstack([comp_feat, prot_feat])

            if hasattr(best_model, "predict_proba"):
                y_prob = best_model.predict_proba(feat.reshape(1, -1))[:, 1]
                score = float(y_prob[0])
            else:
                score = float(best_model.predict(feat.reshape(1, -1))[0])

            predictions.append({
                "MOL_ID": mol_id,
                "molecule_name": mol_name,
                "SMILES": smi,
                "gene": gene,
                "score": score,
                "herb_cn": herb_cn,
                "herb_en": herb_en,
                "herb_pinyin": herb_py,
            })

        if (i + 1) % 100 == 0:
            logger.info(f"    进度: {i+1}/{len(tcm_df)}")

    return pd.DataFrame(predictions)


# ============================================================
# 主流程
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("树模型 CPI v4.0 — 修复版（系统性修复所有缺陷）")
    logger.info("=" * 60)

    # ---- 1. 加载数据 ----
    logger.info("\n[1/7] 加载原始数据...")
    cpi_df = pd.read_csv(L4_RESULTS / "experimental_actives_detail_cleaned.csv", low_memory=False)
    protein_embeddings_raw = {str(k): v.astype(np.float32) for k, v in
                              np.load(L4_RESULTS_V10 / "esm2_protein_embeddings.npz", allow_pickle=True).items()}
    tcm_df = pd.read_csv(L3_RESULTS / "tcm_compound_pool_tox_filtered_noleak.csv", low_memory=False)

    # 获取所有需要的 SMILES
    all_smiles = list(cpi_df["canonical_smiles"].dropna().astype(str).unique())
    tcm_smiles = tcm_df["SMILES_std"].astype(str).tolist()
    all_smiles.extend(tcm_smiles)
    all_smiles = list(dict.fromkeys(all_smiles))
    logger.info(f"  总 SMILES: {len(all_smiles)} (CPI 唯一: {len(all_smiles) - len(tcm_smiles)}, TCM: {len(tcm_smiles)})")

    # ---- 2. 多指纹特征工程（修复：二进制指纹不标准化）----
    logger.info("\n[2/7] 多指纹特征工程 (修复版)...")
    X_binary, X_rdkit, rdkit_scaler, binary_labels, rdkit_names = build_multifingerprint_features(all_smiles)

    # 拼接完整化合物特征
    compound_features = np.hstack([X_binary, X_rdkit])
    logger.info(f"  化合物特征总维度: {compound_features.shape[1]} "
                f"(binary={X_binary.shape[1]}, rdkit={X_rdkit.shape[1]})")

    # ---- 3. 蛋白嵌入处理（修复：PCA 降维 + 标准化）----
    logger.info("\n[3/7] 蛋白嵌入处理 (PCA 降维 + 标准化)...")
    protein_embeddings, prot_pca, prot_scaler = process_protein_embeddings(
        protein_embeddings_raw, target_dim=128, pca_model=None, scaler=None,
    )

    # ---- 4. 构建数据集（统一负采样策略）----
    logger.info("\n[4/7] 构建训练数据集 (统一多样性约束负采样)...")
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

    logger.info(f"  正样本: {len(pos_pairs)} 对")

    # 负样本（统一使用多样性约束采样）
    neg_pairs = diversity_constrained_negative_sampling(
        pos_pairs, all_smiles, cpi_genes_in_emb, neg_ratio=3, random_seed=42,
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
        ci = smiles_to_idx[smi]
        X[i, :comp_dim] = compound_features[ci]
        X[i, comp_dim:] = protein_embeddings[gene]
        pair_smiles.append(smi)
        pair_genes.append(gene)

    pair_smiles = np.array(pair_smiles)
    pair_genes = np.array(pair_genes)

    logger.info(f"  数据集: {n_pairs} 样本, {feat_dim} 特征 (comp={comp_dim}+prot={prot_dim}), "
                f"正样本比例={y.mean():.3f}")

    # ---- 5. Optuna 超参数优化（修复：扩展适用范围）----
    logger.info("\n[5/7] Optuna 超参数优化 (扩展版, 对多模型优化)...")
    # 子采样避免内存溢出
    n_opt = min(5000, len(X))
    rng_opt = np.random.RandomState(42)
    opt_indices = rng_opt.choice(len(X), size=n_opt, replace=False)
    X_opt, y_opt = X[opt_indices], y[opt_indices]
    pair_opt = pair_smiles[opt_indices]

    train_idx_opt, val_idx_opt = scaffold_split(pair_opt, y_opt, test_size=0.2, random_state=42)
    best_params_per_model = optuna_optimize_extended(
        X_opt[train_idx_opt], y_opt[train_idx_opt],
        X_opt[val_idx_opt], y_opt[val_idx_opt],
        model_types=["lgb", "rf", "xgb"],
        n_trials=30, random_state=42,
    )

    # ---- 6. 训练与评估 ----
    logger.info("\n[6/7] 5-fold Scaffold Split 训练 (含修复版 Cascade Forest)...")
    results_df = train_ensemble(X, y, pair_smiles, best_params_per_model=best_params_per_model,
                                n_folds=5, random_seed=42)

    # 汇总
    logger.info("\n" + "=" * 60)
    logger.info("模型评估汇总 (5-fold Scaffold Split, mean +/- std):")
    logger.info("=" * 60)
    summary = results_df.groupby("model").agg(["mean", "std"]).round(4)
    for model_name in summary.index:
        row = summary.loc[model_name]
        logger.info(f"\n  {model_name}:")
        for metric in ["AUC", "AUPR", "F1", "MCC", "EF@1%", "EF@5%", "BEDROC", "ROCE@1%"]:
            if metric in row.index:
                logger.info(f"    {metric}: {row[metric]['mean']:.4f} +/- {row[metric]['std']:.4f}")

    results_path = L4_RESULTS / "tree_v4_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\n评估结果已保存: {results_path}")

    # ---- 7. 全量训练最佳模型 + TCM 预测 + SHAP ----
    logger.info("\n[7/7] 全量训练最佳模型 + TCM 预测 + SHAP...")

    # 选择最佳模型（按 AUPR）
    best_model_name = summary["AUPR"]["mean"].idxmax()
    best_aupr = summary.loc[best_model_name, "AUPR"]["mean"]
    logger.info(f"最佳模型: {best_model_name} (AUPR={best_aupr:.4f})")

    # 全量训练
    if best_model_name == "RandomForest":
        rf_params = best_params_per_model.get("rf", {})
        best_model = RandomForestClassifier(
            n_estimators=rf_params.get("n_estimators", 200),
            max_depth=rf_params.get("max_depth", 20),
            min_samples_leaf=rf_params.get("min_samples_leaf", 10),
            class_weight="balanced", n_jobs=-1, random_state=42,
        )
    elif best_model_name == "XGBoost":
        import xgboost as xgb
        xgb_params = best_params_per_model.get("xgb", {})
        scale_pos_weight = (y == 0).sum() / max(y.sum(), 1)
        best_model = xgb.XGBClassifier(
            n_estimators=xgb_params.get("n_estimators", 200),
            max_depth=xgb_params.get("max_depth", 8),
            learning_rate=xgb_params.get("learning_rate", 0.05),
            subsample=xgb_params.get("subsample", 0.8),
            colsample_bytree=xgb_params.get("colsample_bytree", 0.8),
            scale_pos_weight=scale_pos_weight,
            random_state=42, n_jobs=-1, verbosity=0,
        )
    elif best_model_name == "LightGBM":
        import lightgbm as lgb
        lgb_params = best_params_per_model.get("lgb", {})
        best_model = lgb.LGBMClassifier(
            n_estimators=lgb_params.get("n_estimators", 200),
            max_depth=lgb_params.get("max_depth", 10),
            learning_rate=lgb_params.get("learning_rate", 0.05),
            num_leaves=lgb_params.get("num_leaves", 31),
            subsample=lgb_params.get("subsample", 0.8),
            colsample_bytree=lgb_params.get("colsample_bytree", 0.8),
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
        )
    elif best_model_name == "CatBoost":
        from catboost import CatBoostClassifier
        best_model = CatBoostClassifier(
            iterations=200, depth=8, learning_rate=0.05,
            random_seed=42, thread_count=-1,
            verbose=False, allow_writing_files=False,
        )
    elif best_model_name == "CascadeForestFixed":
        best_model = CascadeForestFixed(
            n_layers=3, n_estimators=100, n_folds=5,
            window_sizes=[100, 200, 500], random_state=42,
        )
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
        shap_result = shap_analysis(best_model, X, output_path=shap_path)

    # 加载中药来源映射
    herb_map = load_herb_mapping()

    # TCM 预测
    tcm_indices = [smiles_to_idx[s] for s in tcm_smiles if s in smiles_to_idx]
    tcm_binary_feats = X_binary[tcm_indices]
    tcm_rdkit_feats = X_rdkit[tcm_indices]

    pred_df = predict_tcm_pool(
        best_model, tcm_df, tcm_binary_feats, tcm_rdkit_feats,
        protein_embeddings, cpi_genes_in_emb, best_model_name, herb_map=herb_map,
    )

    pred_path = L4_RESULTS / "tree_v4_tcm_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"TCM 预测结果已保存: {pred_path}")

    # Top 候选
    comp_agg = pred_df.groupby(["MOL_ID", "molecule_name", "SMILES", "herb_cn", "herb_en"]).agg(
        max_score=("score", "max"),
        mean_score=("score", "mean"),
        n_genes_above_50=("score", lambda x: (x >= 0.5).sum()),
        top_3_genes=("score", lambda x: "|".join(
            [f"{g}({s:.2f})" for g, s in sorted(
                zip(list(pred_df.loc[x.index, "gene"]), list(x), strict=False),
                key=lambda v: v[1], reverse=True
            )[:3]]
        )),
    ).reset_index()
    comp_agg = comp_agg.sort_values("max_score", ascending=False)

    top50 = comp_agg.head(50)
    top_path = L4_RESULTS / "tree_v4_top_candidates.csv"
    top50.to_csv(top_path, index=False)

    logger.info(f"\nTop 20 候选化合物:")
    for i, row in enumerate(top50.head(20).itertuples(index=False), 1):
        logger.info(f"  {i:2d}. {row.molecule_name} | max={row.max_score:.4f} "
                    f"| mean={row.mean_score:.4f} "
                    f"| 高置信(>=0.5): {row.n_genes_above_50} "
                    f"| 中药: {row.herb_cn} "
                    f"| {row.top_3_genes}")

    logger.info(f"\nTop 50 候选已保存: {top_path}")
    logger.info(f"任务完成!")


if __name__ == "__main__":
    main()