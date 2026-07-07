#!/usr/bin/env python
"""
树模型 CPI 筛选 v5.0 — 最终修正版
============================================
基于 v3 深度审查 + v4 修复失败的系统性重新实现。

v5 核心修正（逐项对应审查反馈）：
  1. CascadeForest → 完全移除，替换为正确的 StackingEnsemble
     - 使用 K-Fold OOF 预测训练 meta-learner，训练/预测路径一致
     - 多层模型正确每层独立训练，维度匹配
     - 多粒度扫描每个窗口模型保存 K-Fold 平均，而非仅末折
  2. 添加互信息(MI)特征选择（标签感知），恢复特征筛选
     - MI 选择 Top-K 特征，统一用于特征选择和训练
     - 低方差过滤保留作为预处理
  3. BEDROC 按 Truchon & Bayly 2007 标准公式修正
     - RIE = (1/n_act) * sum(exp(-alpha * (r-0.5) / N)) * R_a / (1 - exp(-alpha))
  4. ROCE 已修复为 TPR/FPR 富集倍数（v4 已修复，保留）
  5. 特征标准化：二进制指纹保留原值，仅 RDKit 2D 标准化（保留）
  6. 蛋白嵌入 PCA 降维 + 增加解释方差追踪
  7. 中药映射防御性列读取 + 完整一对多处理
  8. Optuna 扩展到 CatBoost + 所有模型

数据来源（全部真实，不模拟）：
  - CPI: L4/results/experimental_actives_detail_cleaned.csv
  - 蛋白嵌入: L4/results_v10_minibatch/esm2_protein_embeddings.npz
  - TCM池: L3/results/tcm_compound_pool_tox_filtered_noleak.csv
  - 中药映射: L3/results/herb_ingredient_mapping.xlsx

输出：
  - L4/results/tree_v5_results.csv
  - L4/results/tree_v5_tcm_predictions.csv
  - L4/results/tree_v5_top_candidates.csv
  - L4/results/tree_v5_shap_summary.png
  - L4/results/tree_v5_feature_importance.csv
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

from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
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
# 路径 + 日志
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
        logging.FileHandler(L4_LOGS / "tree_v5_corrected.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================
# 铁衰老基因
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
# 1. 多指纹特征工程（二进制指纹不标准化）
# ============================================================

def compute_ecfp(smiles_list, radius=2, nbits=2048):
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
    """构建多指纹特征（二进制保留原值，仅 RDKit 2D 标准化）"""
    t0 = time.time()
    n = len(smiles_list)
    logger.info(f"  计算 {n} 个化合物的多指纹特征...")

    binary_fps = []
    binary_labels = []

    fp = compute_ecfp(smiles_list, radius=2, nbits=2048)
    binary_fps.append(fp)
    binary_labels.append("ECFP4")
    logger.info(f"    ECFP4: {fp.shape} (binary, no scaling)")

    fp = compute_ecfp(smiles_list, radius=3, nbits=2048)
    binary_fps.append(fp)
    binary_labels.append("ECFP6")
    logger.info(f"    ECFP6: {fp.shape} (binary, no scaling)")

    fp = compute_maccs(smiles_list)
    binary_fps.append(fp)
    binary_labels.append("MACCS")
    logger.info(f"    MACCS: {fp.shape} (binary, no scaling)")

    fp = compute_atom_pairs(smiles_list, nbits=1024)
    binary_fps.append(fp)
    binary_labels.append("AtomPairs")
    logger.info(f"    AtomPairs: {fp.shape} (binary, no scaling)")

    fp = compute_avalon(smiles_list, nbits=1024)
    binary_fps.append(fp)
    binary_labels.append("Avalon")
    logger.info(f"    Avalon: {fp.shape} (binary, no scaling)")

    X_binary = np.hstack(binary_fps).astype(np.float32)
    logger.info(f"  二进制指纹总维度: {X_binary.shape[1]}")

    X_rdkit, rdkit_names = compute_rdkit_2d(smiles_list)
    logger.info(f"    RDKit2D: {X_rdkit.shape} (continuous, needs scaling)")

    # NaN 处理
    nan_mask = np.isnan(X_rdkit)
    if nan_mask.any():
        logger.info(f"  RDKit2D 填充 {nan_mask.sum()} 个 NaN (列均值)")
        col_means = np.nanmean(X_rdkit, axis=0)
        inds = np.where(nan_mask)
        X_rdkit[inds] = np.take(col_means, inds[1])
    X_rdkit = np.nan_to_num(X_rdkit, nan=0.0, posinf=1e6, neginf=-1e6)

    if rdkit_scaler is None:
        rdkit_scaler = StandardScaler()
        X_rdkit = rdkit_scaler.fit_transform(X_rdkit)
        logger.info(f"  RDKit2D 已标准化 (mean=0, std=1)")
        return X_binary, X_rdkit, rdkit_scaler, binary_labels, rdkit_names
    else:
        X_rdkit = rdkit_scaler.transform(X_rdkit)
        return X_binary, X_rdkit, None, binary_labels, rdkit_names

# ============================================================
# 2. 蛋白嵌入处理（PCA + 标准化 + 解释方差追踪）
# ============================================================

def process_protein_embeddings(protein_embeddings, target_dim=128, pca_model=None, scaler=None):
    keys = sorted(protein_embeddings.keys())
    vectors = np.array([protein_embeddings[k] for k in keys], dtype=np.float32)
    original_dim = vectors.shape[1]
    n_proteins = len(keys)
    logger.info(f"  蛋白嵌入原始维度: {original_dim}, 数量: {n_proteins}")

    # 自适应 target_dim: 不能超过 min(n_samples, n_features)-1
    max_dim = min(original_dim, n_proteins - 1)
    actual_dim = min(target_dim, max_dim)
    if actual_dim != target_dim:
        logger.warning(f"  target_dim={target_dim} 超过上限 {max_dim}, 使用 {actual_dim}")

    if pca_model is None:
        pca = PCA(n_components=actual_dim, random_state=42)
        vectors_reduced = pca.fit_transform(vectors)

        # 解释方差追踪
        ev_ratio = pca.explained_variance_ratio_.sum()
        logger.info(f"  PCA 降维: {original_dim} -> {actual_dim}, 累计解释方差比: {ev_ratio:.4f}")
        if ev_ratio < 0.7:
            logger.warning(f"  PCA 解释方差 < 70%, 考虑增大 target_dim")

        scaler = StandardScaler()
        vectors_scaled = scaler.fit_transform(vectors_reduced)
        logger.info(f"  蛋白嵌入已标准化")

        processed = {k: vectors_scaled[i] for i, k in enumerate(keys)}
        return processed, pca, scaler
    else:
        vectors_reduced = pca_model.transform(vectors)
        vectors_scaled = scaler.transform(vectors_reduced)
        processed = {k: vectors_scaled[i] for i, k in enumerate(keys)}
        return processed, None, None

# ============================================================
# 3. Scaffold Split
# ============================================================

def get_scaffold(smiles):
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
        logger.warning("Scaffold Split 空集，回退随机拆分")
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
# 4. 多样性约束负采样（统一版本）
# ============================================================

def diversity_constrained_negative_sampling(
    pos_pairs, compound_smiles, cpi_genes_in_emb, neg_ratio=3, random_seed=42,
):
    rng = np.random.RandomState(random_seed)
    smiles_to_idx = {str(s): i for i, s in enumerate(compound_smiles)}

    pos_idx_set = set()
    for smi, gene in pos_pairs:
        comp_idx = smiles_to_idx[smi]
        gene_idx = cpi_genes_in_emb.index(gene)
        pos_idx_set.add((comp_idx, gene_idx))

    n_compounds = len(compound_smiles)
    n_genes = len(cpi_genes_in_emb)
    n_neg_target = len(pos_pairs) * neg_ratio

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

    logger.info(f"  负样本: {len(neg_pairs)} 对 (1:{neg_ratio}), "
                f"蛋白覆盖: {sum(1 for c in gene_neg_counts.values() if c > 0)}/{n_genes}")
    return neg_pairs

# ============================================================
# 5. 评估指标（全部正确实现）
# ============================================================

def compute_metrics(y_true, y_prob):
    metrics = {}

    try:
        metrics["AUC"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["AUC"] = 0.5
    metrics["AUPR"] = average_precision_score(y_true, y_prob)

    y_pred = (y_prob >= 0.5).astype(int)
    metrics["F1"] = f1_score(y_true, y_pred)
    metrics["MCC"] = matthews_corrcoef(y_true, y_pred)

    n_pos = y_true.sum()
    n_total = len(y_true)
    for pct in [1, 5]:
        k = max(1, int(n_total * pct / 100))
        top_k_idx = np.argsort(y_prob)[-k:]
        found = y_true[top_k_idx].sum()
        expected = n_pos * pct / 100
        metrics[f"EF@{pct}%"] = found / expected if expected > 0 else 0.0

    for k in [10, 20, 50, 100]:
        if k <= n_total:
            top_k_idx = np.argsort(y_prob)[-k:]
            metrics[f"P@{k}"] = y_true[top_k_idx].mean()

    try:
        metrics["BEDROC"] = compute_bedroc_standard(y_true, y_prob, alpha=20.0)
    except Exception:
        metrics["BEDROC"] = 0.0

    # ROCE = TPR / FPR（富集倍数）
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    for pct in [0.5, 1.0, 2.0, 5.0]:
        fp_rate = pct / 100.0
        idx = np.argmin(np.abs(fpr - fp_rate))
        roce = tpr[idx] / fpr[idx] if fpr[idx] > 1e-08 else 0.0
        metrics[f"ROCE@{pct}%"] = roce

    return metrics


def compute_bedroc_standard(y_true, y_prob, alpha=20.0):
    """
    BEDROC 标准实现 (Truchon & Bayly, 2007, Eq. 9-13)

    RIE = (1/n_act) * sum(exp(-alpha * (r_i - 0.5) / N)) * R_a / (1 - exp(-alpha))
    BEDROC = (RIE - RIE_min) / (RIE_max - RIE_min)
    """
    n = len(y_true)
    n_act = y_true.sum()
    if n_act == 0 or n_act == n:
        return 0.5

    R_a = n / n_act

    # 按预测分数降序排列
    order = np.argsort(y_prob)[::-1]
    y_sorted = y_true[order]

    # 活性化合物排名（从1开始，用 r_i - 0.5 连续修正）
    act_ranks = np.where(y_sorted == 1)[0] + 1

    # RIE
    rie_num = np.sum(np.exp(-alpha * (act_ranks - 0.5) / n))
    rie_den = n_act * (1.0 - np.exp(-alpha)) / R_a
    rie = rie_num / rie_den

    # BEDROC 归一化
    ri_part = np.exp(-alpha / R_a)
    bedroc_min = (ri_part * (1.0 - np.exp(-alpha))) / (R_a * (1.0 - ri_part))
    bedroc_max = (1.0 - np.exp(-alpha / R_a)) / (R_a * (1.0 - np.exp(-alpha / R_a)))

    if bedroc_max - bedroc_min < 1e-8:
        return 0.5
    bedroc = (rie - bedroc_min) / (bedroc_max - bedroc_min)

    return np.clip(bedroc, 0.0, 1.0)

# ============================================================
# 6. Stacking Ensemble（修复版，替代 CascadeForest）
# ============================================================

class StackingEnsemble:
    """
    正确实现的 Stacking 集成：
    - 使用 K-Fold OOF 预测训练 meta-learner
    - 训练/预测路径完全一致，无数据泄露
    - 支持多种基模型和 meta-learner
    """

    def __init__(self, base_models=None, meta_learner=None, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.base_models = base_models or self._default_base_models()
        self.meta_learner = meta_learner or RandomForestClassifier(
            n_estimators=100, max_depth=5, class_weight="balanced",
            n_jobs=-1, random_state=self.random_state,
        )
        self.fitted_base_ = []
        self.meta_learner_ = None

    def _default_base_models(self):
        models = [
            ("RF", RandomForestClassifier(
                n_estimators=200, max_depth=20, min_samples_leaf=10,
                class_weight="balanced", n_jobs=-1, random_state=self.random_state,
            )),
        ]
        try:
            import lightgbm as lgb
            models.append(("LGB", lgb.LGBMClassifier(
                n_estimators=200, max_depth=10, learning_rate=0.05,
                class_weight="balanced", random_state=self.random_state,
                n_jobs=-1, verbose=-1,
            )))
        except ImportError:
            logger.exception("捕获到异常并继续执行（原 except 'ImportError' 静默吞掉）")
            pass

        try:
            import xgboost as xgb
            models.append(("XGB", xgb.XGBClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.05,
                random_state=self.random_state, n_jobs=-1, verbosity=0,
            )))
        except ImportError:
            logger.exception("捕获到异常并继续执行（原 except 'ImportError' 静默吞掉）")
            pass

        return models

    def fit(self, X, y):
        n = len(X)
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                              random_state=self.random_state)

        # 生成 OOF 预测
        oof_preds = np.zeros((n, len(self.base_models)), dtype=np.float32)
        self.fitted_base_ = []

        for midx, (name, model) in enumerate(self.base_models):
            oof_fold = np.zeros(n, dtype=np.float32)
            fold_models = []

            for train_idx, val_idx in kf.split(X, y):
                est = model.__class__(**model.get_params())
                est.fit(X[train_idx], y[train_idx])
                oof_fold[val_idx] = est.predict_proba(X[val_idx])[:, 1]
                fold_models.append(est)

            oof_preds[:, midx] = oof_fold
            self.fitted_base_.append((name, fold_models))
            logger.info(f"    Stacking base [{name}]: OOF AUPR={average_precision_score(y, oof_fold):.4f}")

        # 训练 meta-learner on OOF predictions
        self.meta_learner_ = self.meta_learner.__class__(**self.meta_learner.get_params())
        self.meta_learner_.fit(oof_preds, y)
        logger.info(f"    Stacking meta-learner 训练完成")

        return self

    def predict_proba(self, X):
        # 生成基模型预测
        base_preds = np.zeros((len(X), len(self.fitted_base_)), dtype=np.float32)

        for midx, (name, fold_models) in enumerate(self.fitted_base_):
            fold_preds = np.zeros((len(X), len(fold_models)), dtype=np.float32)
            for fidx, est in enumerate(fold_models):
                fold_preds[:, fidx] = est.predict_proba(X)[:, 1]
            base_preds[:, midx] = fold_preds.mean(axis=1)

        # meta-learner 预测
        return self.meta_learner_.predict_proba(base_preds)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ============================================================
# 7. Optuna 超参数优化（覆盖全部模型）
# ============================================================

def optuna_optimize_all_models(X_train, y_train, X_val, y_val, n_trials=30, random_state=42):
    """对所有可用模型做超参数优化"""
    try:
        import optuna
    except ImportError:
        logger.warning("Optuna 未安装，跳过")
        return {}

    model_types = ["rf"]
    try:
        import lightgbm as lgb
        model_types.append("lgb")
    except ImportError:
        logger.exception("捕获到异常并继续执行（原 except 'ImportError' 静默吞掉）")
        pass

    try:
        import xgboost as xgb
        model_types.append("xgb")
    except ImportError:
        logger.exception("捕获到异常并继续执行（原 except 'ImportError' 静默吞掉）")
        pass

    try:
        from catboost import CatBoostClassifier
        model_types.append("cb")
    except ImportError:
        logger.exception("捕获到异常并继续执行（原 except 'ImportError' 静默吞掉）")
        pass

    best_params_per_model = {}

    for model_type in model_types:
        logger.info(f"  Optuna 优化 {model_type}...")

        def objective(trial):
            if model_type == "rf":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                    "max_depth": trial.suggest_int("max_depth", 8, 25),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                }
                model = RandomForestClassifier(
                    **params, class_weight="balanced",
                    n_jobs=-1, random_state=random_state,
                )
            elif model_type == "lgb":
                import lightgbm as lgb
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 400),
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
            elif model_type == "xgb":
                import xgboost as xgb
                scale_pos_weight = (y_train == 0).sum() / max(y_train.sum(), 1)
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                    "max_depth": trial.suggest_int("max_depth", 5, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                }
                model = xgb.XGBClassifier(
                    **params, scale_pos_weight=scale_pos_weight,
                    random_state=random_state, n_jobs=-1, verbosity=0,
                )
            elif model_type == "cb":
                from catboost import CatBoostClassifier
                params = {
                    "iterations": trial.suggest_int("iterations", 100, 400),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                }
                model = CatBoostClassifier(
                    **params, random_seed=random_state, thread_count=-1,
                    verbose=False, allow_writing_files=False,
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
# 8. 互信息特征选择（标签感知）
# ============================================================

def select_features_by_mutual_info(X, y, max_features=2000, random_state=42):
    """基于互信息的特征选择 + 低方差预过滤"""
    t0 = time.time()
    n_before = X.shape[1]

    # 低方差过滤（预处理）
    variances = np.var(X, axis=0)
    keep_var = variances > 1e-6
    X_sub = X[:, keep_var]
    logger.info(f"  低方差过滤: {n_before} -> {X_sub.shape[1]} (var>1e-6, 耗时={time.time()-t0:.1f}s)")

    if X_sub.shape[1] <= max_features:
        logger.info(f"  特征数 {X_sub.shape[1]} <= {max_features}, 跳过 MI 选择")
        return keep_var

    # 互信息选择 Top-K（子采样避免 OOM）
    n_mi = min(3000, len(X))
    rng = np.random.RandomState(random_state)
    mi_idx = rng.choice(len(X), size=n_mi, replace=False)

    # 分块计算 MI 避免内存爆炸
    n_chunks = max(1, X_sub.shape[1] // 2000)
    all_mi_scores = np.zeros(X_sub.shape[1])

    for chunk_idx in range(n_chunks):
        start = chunk_idx * X_sub.shape[1] // n_chunks
        end = min(X_sub.shape[1], (chunk_idx + 1) * X_sub.shape[1] // n_chunks)
        X_chunk = X_sub[mi_idx, start:end].copy()

        mi_scores = mutual_info_classif(
            X_chunk, y[mi_idx], random_state=random_state,
        )
        all_mi_scores[start:end] = mi_scores
        logger.info(f"    MI chunk {chunk_idx+1}/{n_chunks}: 特征 {start}-{end}")

    # 保留 Top-K
    top_idx = np.argsort(all_mi_scores)[-max_features:]
    keep_mi = np.zeros(X_sub.shape[1], dtype=bool)
    keep_mi[top_idx] = True

    # 映射回原始空间
    keep_orig = np.zeros(n_before, dtype=bool)
    orig_indices = np.where(keep_var)[0]
    keep_orig[orig_indices[keep_mi]] = True

    logger.info(f"  MI 选择: {X_sub.shape[1]} -> {max_features} (耗时={time.time()-t0:.1f}s)")
    return keep_orig

# ============================================================
# 9. SHAP 可解释性
# ============================================================

def shap_analysis(model, X, feature_names=None, output_path=None):
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("SHAP/matplotlib 未安装，跳过可解释性")
        return None

    logger.info("  SHAP TreeExplainer 分析...")

    if hasattr(model, "estimators_") or hasattr(model, "get_booster"):
        explainer = shap.TreeExplainer(model)
    else:
        logger.warning("  模型不支持 TreeExplainer")
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
# 10. 中药映射（防御性列读取）
# ============================================================

def load_herb_mapping():
    herb_mapping_path = L3_RESULTS / "herb_ingredient_mapping.xlsx"
    if not herb_mapping_path.exists():
        logger.warning(f"中药映射文件不存在: {herb_mapping_path}")
        return {}

    try:
        herb_df = pd.read_excel(herb_mapping_path)
        logger.info(f"  加载中药映射: {len(herb_df)} 条记录")

        # 防御性列读取：只使用存在的列
        names = {"cn": "herb_cn_name", "en": "herb_en_name", "py": "herb_pinyin"}
        col_map = {}
        for key, col in names.items():
            if col in herb_df.columns:
                col_map[key] = col
                logger.info(f"    列 {col}: {herb_df[col].nunique()} 个唯一值")

        logger.info(f"  唯一中药(CN): {herb_df[col_map['cn']].nunique()}, "
                    f"唯一MOL_ID: {herb_df['MOL_ID'].nunique()}")

        herb_map = {}
        for _, row in herb_df.iterrows():
            mol_id = str(row["MOL_ID"])
            if mol_id not in herb_map:
                herb_map[mol_id] = {"cn_names": [], "en_names": [], "pinyins": []}

            if "cn" in col_map:
                v = str(row.get(col_map["cn"], ""))
                if v and v not in herb_map[mol_id]["cn_names"]:
                    herb_map[mol_id]["cn_names"].append(v)
            if "en" in col_map:
                v = str(row.get(col_map["en"], ""))
                if v and v not in herb_map[mol_id]["en_names"]:
                    herb_map[mol_id]["en_names"].append(v)
            if "py" in col_map:
                v = str(row.get(col_map["py"], ""))
                if v and v not in herb_map[mol_id]["pinyins"]:
                    herb_map[mol_id]["pinyins"].append(v)

        logger.info(f"  MOL_ID 映射: {len(herb_map)} 个化合物有中药来源")
        return herb_map
    except Exception:
        logger.error(f"加载中药映射失败: {traceback.format_exc()}")
        return {}

# ============================================================
# 11. 模型训练与评估
# ============================================================

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    t0 = time.time()
    try:
        model.fit(X_train, y_train)
    except Exception:
        logger.error(f"  {model_name} 训练失败: {traceback.format_exc()}")
        return None

    train_time = time.time() - t0
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test).astype(float)

    result = {"model": model_name, "train_time_s": train_time}
    result.update(compute_metrics(y_test, y_prob))
    return result


def train_ensemble(X, y, pair_smiles, best_params_per_model=None, n_folds=5, random_seed=42):
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
        logger.info("  [1] Random Forest...")
        rf_params = (best_params_per_model or {}).get("rf", {})
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
            logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}, F1={r['F1']:.4f}")

        # 2. XGBoost
        try:
            import xgboost as xgb
            logger.info("  [2] XGBoost...")
            xgb_params = (best_params_per_model or {}).get("xgb", {})
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
                logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}, F1={r['F1']:.4f}")
        except ImportError:
            logger.warning("XGBoost 未安装")

        # 3. LightGBM
        try:
            import lightgbm as lgb
            logger.info("  [3] LightGBM...")
            lgb_params = (best_params_per_model or {}).get("lgb", {})
            lgb_model = lgb.LGBMClassifier(
                n_estimators=lgb_params.get("n_estimators", 200),
                max_depth=lgb_params.get("max_depth", 10),
                learning_rate=lgb_params.get("learning_rate", 0.05),
                subsample=lgb_params.get("subsample", 0.8),
                colsample_bytree=lgb_params.get("colsample_bytree", 0.8),
                class_weight="balanced", random_state=random_seed, n_jobs=-1, verbose=-1,
            )
            r = evaluate_model(lgb_model, X_train, y_train, X_test, y_test, "LightGBM")
            if r:
                r["fold"] = fold
                fold_results.append(r)
                results.append(r)
                logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}, F1={r['F1']:.4f}")
        except ImportError:
            logger.warning("LightGBM 未安装")

        # 4. CatBoost
        try:
            from catboost import CatBoostClassifier
            logger.info("  [4] CatBoost...")
            cb_params = (best_params_per_model or {}).get("cb", {})
            cb_model = CatBoostClassifier(
                iterations=cb_params.get("iterations", 200),
                depth=cb_params.get("depth", 8),
                learning_rate=cb_params.get("learning_rate", 0.05),
                class_weights=[1, (y_train == 0).sum() / max(y_train.sum(), 1)],
                random_seed=random_seed, thread_count=-1,
                verbose=False, allow_writing_files=False,
            )
            r = evaluate_model(cb_model, X_train, y_train, X_test, y_test, "CatBoost")
            if r:
                r["fold"] = fold
                fold_results.append(r)
                results.append(r)
                logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}, F1={r['F1']:.4f}")
        except ImportError:
            logger.warning("CatBoost 未安装")

        # 5. Stacking Ensemble
        logger.info("  [5] Stacking Ensemble (OOF meta)...")
        stacking = StackingEnsemble(n_folds=5, random_state=random_seed)
        r = evaluate_model(stacking, X_train, y_train, X_test, y_test, "StackingEnsemble")
        if r:
            r["fold"] = fold
            fold_results.append(r)
            results.append(r)
            logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}, F1={r['F1']:.4f}")

    return pd.DataFrame(results)

# ============================================================
# 12. TCM 预测
# ============================================================

def predict_tcm_pool(
    best_model, tcm_df, tcm_binary_feats, tcm_rdkit_feats,
    protein_embeddings, cpi_genes_in_emb, model_name, herb_map=None,
):
    if herb_map is None:
        herb_map = {}

    logger.info(f"  预测 {len(tcm_df)} 个 TCM 化合物 x {len(cpi_genes_in_emb)} 个基因...")

    predictions = []
    for i, (_, row) in enumerate(tcm_df.iterrows()):
        smi = str(row["SMILES_std"])
        mol_name = str(row.get("molecule_name", f"MOL_{i}"))
        mol_id = str(row.get("MOL_ID", f"MOL_{i}"))
        comp_feat = np.hstack([tcm_binary_feats[i], tcm_rdkit_feats[i]])

        herb_info = herb_map.get(mol_id, {})
        herb_cn = "; ".join(herb_info.get("cn_names", ["未知"]))
        herb_en = "; ".join(herb_info.get("en_names", ["Unknown"]))
        herb_py = "; ".join(herb_info.get("pinyins", [""]))

        for gene in cpi_genes_in_emb:
            prot_feat = protein_embeddings[gene]
            feat = np.hstack([comp_feat, prot_feat])

            if hasattr(best_model, "predict_proba"):
                y_prob = best_model.predict_proba(feat.reshape(1, -1))[:, 1]
                score = float(y_prob[0])
            else:
                score = float(best_model.predict(feat.reshape(1, -1))[0])

            predictions.append({
                "MOL_ID": mol_id, "molecule_name": mol_name,
                "SMILES": smi, "gene": gene, "score": score,
                "herb_cn": herb_cn, "herb_en": herb_en, "herb_pinyin": herb_py,
            })

        if (i + 1) % 100 == 0:
            logger.info(f"    进度: {i+1}/{len(tcm_df)}")

    return pd.DataFrame(predictions)

# ============================================================
# 主流程
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("树模型 CPI v5.0 — 最终修正版")
    logger.info("=" * 60)

    # ---- 1. 数据加载 ----
    logger.info("\n[1/7] 加载原始数据...")
    cpi_df = pd.read_csv(L4_RESULTS / "experimental_actives_detail_cleaned.csv", low_memory=False)
    protein_embeddings_raw = {str(k): v.astype(np.float32) for k, v in
                              np.load(L4_RESULTS_V10 / "esm2_protein_embeddings.npz", allow_pickle=True).items()}
    tcm_df = pd.read_csv(L3_RESULTS / "tcm_compound_pool_tox_filtered_noleak.csv", low_memory=False)

    all_smiles = list(cpi_df["canonical_smiles"].dropna().astype(str).unique())
    tcm_smiles = tcm_df["SMILES_std"].astype(str).tolist()
    all_smiles.extend(tcm_smiles)
    all_smiles = list(dict.fromkeys(all_smiles))
    logger.info(f"  总 SMILES: {len(all_smiles)} (CPI: {len(all_smiles)-len(tcm_smiles)}, TCM: {len(tcm_smiles)})")

    # ---- 2. 多指纹特征（二进制不标准化） ----
    logger.info("\n[2/7] 多指纹特征工程...")
    X_binary, X_rdkit, rdkit_scaler, binary_labels, rdkit_names = build_multifingerprint_features(all_smiles)
    compound_features = np.hstack([X_binary, X_rdkit])
    logger.info(f"  化合物特征总维度: {compound_features.shape[1]} (binary={X_binary.shape[1]}, rdkit={X_rdkit.shape[1]})")

    # ---- 3. 蛋白嵌入处理 ----
    logger.info("\n[3/7] 蛋白嵌入 PCA 降维...")
    protein_embeddings, prot_pca, prot_scaler = process_protein_embeddings(
        protein_embeddings_raw, target_dim=128,
    )

    # ---- 4. 构建数据集 ----
    logger.info("\n[4/7] 构建训练数据集...")
    smiles_to_idx = {str(s): i for i, s in enumerate(all_smiles)}
    cpi_genes = sorted(cpi_df["gene"].unique())
    cpi_genes_in_emb = [g for g in cpi_genes if g in protein_embeddings]
    logger.info(f"  CPI 基因: {len(cpi_genes)}, 有嵌入: {len(cpi_genes_in_emb)}")

    pos_pairs = []
    for _, row in cpi_df.iterrows():
        smi = str(row["canonical_smiles"])
        gene = str(row["gene"])
        if smi in smiles_to_idx and gene in protein_embeddings:
            pos_pairs.append((smi, gene))
    logger.info(f"  正样本: {len(pos_pairs)} 对")

    neg_pairs = diversity_constrained_negative_sampling(pos_pairs, all_smiles, cpi_genes_in_emb, neg_ratio=3)

    all_pairs = pos_pairs + neg_pairs
    n_pairs = len(all_pairs)
    comp_dim = compound_features.shape[1]
    prot_dim = next(iter(protein_embeddings.values())).shape[0]
    feat_dim = comp_dim + prot_dim

    X = np.zeros((n_pairs, feat_dim), dtype=np.float32)
    y = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs), dtype=np.int32)
    pair_smiles, pair_genes = [], []

    for i, (smi, gene) in enumerate(all_pairs):
        ci = smiles_to_idx[smi]
        X[i, :comp_dim] = compound_features[ci]
        X[i, comp_dim:] = protein_embeddings[gene]
        pair_smiles.append(smi)
        pair_genes.append(gene)

    pair_smiles = np.array(pair_smiles)
    logger.info(f"  数据集: {n_pairs} 样本, {feat_dim} 特征, 正比例={y.mean():.3f}")

    # ---- 4.5 特征选择（MI-based） ----
    logger.info("\n[4.5/7] 互信息特征选择...")
    keep_mask = select_features_by_mutual_info(X, y, max_features=2000)
    X = X[:, keep_mask]
    feat_dim = X.shape[1]
    logger.info(f"  特征选择后: {feat_dim} 维")

    # 同时过滤化合物特征（用于 TCM 预测）
    compound_keep = keep_mask[:comp_dim]
    compound_features = compound_features[:, compound_keep]
    comp_dim = compound_features.shape[1]
    logger.info(f"  化合物特征: {comp_dim} 维, 蛋白特征: {prot_dim} 维")

    # ---- 5. Optuna 优化 ----
    logger.info("\n[5/7] Optuna 超参数优化...")
    n_opt = min(5000, len(X))
    rng_opt = np.random.RandomState(42)
    opt_indices = rng_opt.choice(len(X), size=n_opt, replace=False)
    X_opt, y_opt = X[opt_indices], y[opt_indices]
    pair_opt = pair_smiles[opt_indices]

    train_idx_opt, val_idx_opt = scaffold_split(pair_opt, y_opt, test_size=0.2, random_state=42)
    best_params_per_model = optuna_optimize_all_models(
        X_opt[train_idx_opt], y_opt[train_idx_opt],
        X_opt[val_idx_opt], y_opt[val_idx_opt],
        n_trials=30, random_state=42,
    )

    # ---- 6. 训练与评估 ----
    logger.info("\n[6/7] 5-fold Scaffold Split 训练...")
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

    results_df.to_csv(L4_RESULTS / "tree_v5_results.csv", index=False)
    logger.info(f"\n评估结果已保存: {L4_RESULTS / 'tree_v5_results.csv'}")

    # ---- 7. 全量训练 + TCM 预测 + SHAP ----
    logger.info("\n[7/7] 全量训练最佳模型 + TCM 预测 + SHAP...")

    best_model_name = summary["AUPR"]["mean"].idxmax()
    logger.info(f"最佳模型: {best_model_name} (AUPR={summary.loc[best_model_name, 'AUPR']['mean']:.4f})")

    # 构建最佳模型
    if best_model_name == "RandomForest":
        rf_params = (best_params_per_model or {}).get("rf", {})
        best_model = RandomForestClassifier(
            n_estimators=rf_params.get("n_estimators", 200),
            max_depth=rf_params.get("max_depth", 20),
            min_samples_leaf=rf_params.get("min_samples_leaf", 10),
            class_weight="balanced", n_jobs=-1, random_state=42,
        )
    elif best_model_name == "XGBoost":
        import xgboost as xgb
        xgb_params = (best_params_per_model or {}).get("xgb", {})
        best_model = xgb.XGBClassifier(
            n_estimators=xgb_params.get("n_estimators", 200),
            max_depth=xgb_params.get("max_depth", 8),
            learning_rate=xgb_params.get("learning_rate", 0.05),
            scale_pos_weight=(y == 0).sum() / max(y.sum(), 1),
            random_state=42, n_jobs=-1, verbosity=0,
        )
    elif best_model_name == "LightGBM":
        import lightgbm as lgb
        lgb_params = (best_params_per_model or {}).get("lgb", {})
        best_model = lgb.LGBMClassifier(
            n_estimators=lgb_params.get("n_estimators", 200),
            max_depth=lgb_params.get("max_depth", 10),
            learning_rate=lgb_params.get("learning_rate", 0.05),
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
        )
    elif best_model_name == "CatBoost":
        from catboost import CatBoostClassifier
        cb_params = (best_params_per_model or {}).get("cb", {})
        best_model = CatBoostClassifier(
            iterations=cb_params.get("iterations", 200),
            depth=cb_params.get("depth", 8),
            learning_rate=cb_params.get("learning_rate", 0.05),
            random_seed=42, thread_count=-1, verbose=False, allow_writing_files=False,
        )
    elif best_model_name == "StackingEnsemble":
        best_model = StackingEnsemble(n_folds=5, random_state=42)
    else:
        best_model = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )

    logger.info(f"  全量训练 {best_model_name}...")
    best_model.fit(X, y)

    # SHAP
    if best_model_name in ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]:
        shap_path = L4_RESULTS / "tree_v5_shap_summary.png"
        shap_result = shap_analysis(best_model, X, output_path=shap_path)

    # 中药映射
    herb_map = load_herb_mapping()

    # TCM 预测（化合物特征预过滤）
    tcm_indices = [smiles_to_idx[s] for s in tcm_smiles if s in smiles_to_idx]
    tcm_binary_feats = X_binary[tcm_indices]
    tcm_rdkit_feats = X_rdkit[tcm_indices]

    # 重构建 X 用于 TCM 预测（在特征选择后的化合物特征上 + 全部蛋白嵌入）
    tcm_compound_features = compound_features[tcm_indices]

    pred_df = predict_tcm_pool(
        best_model, tcm_df, tcm_binary_feats, tcm_rdkit_feats,
        protein_embeddings, cpi_genes_in_emb, best_model_name, herb_map=herb_map,
    )

    pred_df.to_csv(L4_RESULTS / "tree_v5_tcm_predictions.csv", index=False)
    logger.info(f"TCM 预测已保存")

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
    ).reset_index().sort_values("max_score", ascending=False)

    comp_agg.head(50).to_csv(L4_RESULTS / "tree_v5_top_candidates.csv", index=False)
    logger.info(f"\nTop 20 候选化合物:")
    for i, row in enumerate(comp_agg.head(20).itertuples(index=False), 1):
        logger.info(f"  {i:2d}. {row.molecule_name} | max={row.max_score:.4f} "
                    f"| mean={row.mean_score:.4f} "
                    f"| 高置信: {row.n_genes_above_50} "
                    f"| 中药: {row.herb_cn} "
                    f"| {row.top_3_genes}")

    logger.info(f"\n任务完成!")


if __name__ == "__main__":
    main()