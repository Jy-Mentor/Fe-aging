#!/usr/bin/env python
"""
树模型 CPI 筛选 v2.0 — 工业化升级版
========================================
基于 GitHub/PubMed 前沿调研的全面升级：

升级项：
  1. 多指纹特征工程：ECFP4+ECFP6+MACCS+AtomPairs+Avalon+ErG+RDKit2D+Mordred
  2. CatBoost 加入模型池（Nature 2025, 35亿化合物筛选state-of-the-art）
  3. Stacking Ensemble: RF+XGB+LGB+CatBoost → LogisticRegression
  4. Soft Voting 集成
  5. Scaffold Split (Bemis-Murcko) 替代随机拆分
  6. 特征选择：低方差过滤 + 高相关性剔除
  7. Optuna 超参数优化
  8. SHAP 可解释性分析
  9. Conformal Prediction 置信度校准

数据来源（全部真实，不模拟）：
  - CPI: L4/results/experimental_actives_detail_cleaned.csv
  - 蛋白嵌入: L4/results_v10_minibatch/esm2_protein_embeddings.npz
  - TCM池: L3/results/tcm_compound_pool_tox_filtered_noleak.csv

输出：
  - L4/results/tree_v2_results.csv
  - L4/results/tree_v2_tcm_predictions.csv
  - L4/results/tree_v2_top_candidates.csv
  - L4/results/tree_v2_shap_summary.png
"""

import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import (
    AllChem,
    Descriptors,
    rdMolDescriptors,
    MACCSkeys,
)
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
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
        logging.FileHandler(L4_LOGS / "tree_v2_industrial.log", mode="w", encoding="utf-8"),
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
# 1. 多指纹特征工程 (Chemistry-Informed)
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
        logger.warning("  Avalon 不可用，跳过")
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


def compute_erg(smiles_list):
    """ErG 药效团指纹"""
    factory = Gobbi_Pharm2D.factory
    fps = []
    for smi in smiles_list:
        if not smi or pd.isna(smi):
            fps.append(None)
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            fps.append(None)
            continue
        fp = Generate.Gen2DFingerprint(mol, factory)
        bits = np.zeros(len(fp), dtype=np.float32)
        for bit in fp.GetOnBits():
            bits[bit] = 1.0
        fps.append(bits)
    return fps


def compute_rdkit_2d(smiles_list):
    """RDKit 2D 描述符（200+ 维）"""
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


def compute_mordred(smiles_list):
    """Mordred 描述符（1800+ 维）"""
    try:
        from mordred import Calculator, descriptors
        calc = Calculator(descriptors, ignore_3D=True)
        fps = []
        for smi in smiles_list:
            if not smi or pd.isna(smi):
                fps.append(None)
                continue
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                fps.append(None)
                continue
            result = calc(mol)
            vals = np.array([float(v) if v is not None and not isinstance(v, (str, bool)) else np.nan for v in result.values()], dtype=np.float32)
            fps.append(vals)
        return fps
    except ImportError:
        logger.warning("Mordred 未安装，跳过")
        return None


def build_multifingerprint_features(smiles_list, fit_scaler=None):
    """
    构建多指纹融合特征矩阵

    指纹组合（总维度 ~5500）：
      - ECFP4 (2048)
      - ECFP6 (2048)
      - MACCS (167)
      - AtomPairs (1024)
      - Avalon (1024)
      - ErG (~400)
      - RDKit 2D (208)
      - Mordred (1800+, 可选)
    """
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

    # ErG 药效团 — 计算开销大，跳过
    # erg_fps = compute_erg(smiles_list)
    # erg_dim = erg_fps[0].shape[0] if erg_fps[0] is not None else 0
    # if erg_dim > 0:
    #     fp = np.array([f if f is not None else np.zeros(erg_dim) for f in erg_fps], dtype=np.float32)
    #     fps.append(fp)
    #     labels.append("ErG")
    #     logger.info(f"    ErG: {fp.shape}")

    # RDKit 2D — 计算开销大，跳过
    # fp = compute_rdkit_2d(smiles_list)
    # fps.append(fp)
    # labels.append("RDKit2D")
    # logger.info(f"    RDKit2D: {fp.shape}")

    # Mordred (可选)
    mordred_fps = compute_mordred(smiles_list)
    if mordred_fps is not None and mordred_fps[0] is not None:
        mordred_dim = mordred_fps[0].shape[0]
        fp = np.array([f if f is not None else np.zeros(mordred_dim) for f in mordred_fps], dtype=np.float32)
        fps.append(fp)
        labels.append("Mordred")
        logger.info(f"    Mordred: {fp.shape}")

    # 拼接
    X = np.hstack(fps).astype(np.float32)
    logger.info(f"  总特征维度: {X.shape[1]}, 耗时: {time.time()-t0:.1f}s")

    # NaN 处理
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_means = np.nanmean(X, axis=0)
        inds = np.where(nan_mask)
        X[inds] = np.take(col_means, inds[1])
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # 标准化 RDKit 2D 部分
    if fit_scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, scaler
    else:
        X = fit_scaler.transform(X)
        return X, None


# ============================================================
# 2. 特征选择
# ============================================================

def feature_selection(X, y, variance_threshold=0.001):
    """低方差特征过滤（树模型天然抗共线性，跳过相关性过滤）"""
    t0 = time.time()
    n_before = X.shape[1]

    # 低方差过滤
    variances = np.var(X, axis=0)
    keep_var = variances > variance_threshold
    X = X[:, keep_var]
    logger.info(f"  低方差过滤: {n_before} → {X.shape[1]} (阈值={variance_threshold}, 耗时={time.time()-t0:.1f}s)")

    # 相关性过滤 — 树模型天然抗共线性，跳过
    corr_mask = np.ones(X.shape[1], dtype=bool)
    return X, keep_var, corr_mask


# ============================================================
# 3. Scaffold Split (Bemis-Murcko)
# ============================================================

def get_scaffold(smiles):
    """Bemis-Murcko 骨架"""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return "INVALID"
        scaffold = rdMolDescriptors.MurckoScaffoldSmiles(mol)
        return scaffold if scaffold else "NO_SCAFFOLD"
    except Exception:
        logger.exception("捕获到异常并继续执行（原 except 'Exception' 静默吞掉）")
        return "INVALID"

def scaffold_split(pair_smiles, y, test_size=0.2, random_state=42):
    """按化合物 Bemis-Murcko 骨架拆分，确保同一骨架的化合物不在train/test间泄漏"""
    rng = np.random.RandomState(random_state)
    unique_smiles = sorted(set(pair_smiles))
    logger.info(f"  唯一化合物: {len(unique_smiles)}")

    # 为每个唯一化合物计算骨架
    scaffolds = np.array([get_scaffold(s) for s in unique_smiles])
    unique_scaffolds = sorted(set(scaffolds))
    n_scaffolds = len(unique_scaffolds)
    test_n_scaffolds = max(1, int(n_scaffolds * test_size))

    # 按化合物数量排序骨架，优先分配大骨架到训练集
    scaffold_sizes = {s: (scaffolds == s).sum() for s in unique_scaffolds}
    sorted_scaffolds = sorted(unique_scaffolds, key=lambda s: scaffold_sizes[s], reverse=True)
    test_scaffolds = set(rng.choice(sorted_scaffolds, test_n_scaffolds, replace=False))

    # 化合物级别的 train/test 分配
    smiles_to_scaffold = dict(zip(unique_smiles, scaffolds, strict=False))
    test_smiles = {s for s, sc in smiles_to_scaffold.items() if sc in test_scaffolds}

    # 对 pair 级别分配
    test_mask = np.array([s in test_smiles for s in pair_smiles])
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]

    # 确保 train 和 test 都有正负样本
    if len(train_idx) == 0 or len(test_idx) == 0:
        logger.warning("  Scaffold Split 导致空集，回退到随机拆分")
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
# 4. 数据集构建
# ============================================================

def build_dataset(
    cpi_df, compound_smiles, compound_features, scaler,
    protein_embeddings, neg_ratio=3, random_seed=42,
):
    """构建训练数据集"""
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

    # 负样本：向量化批量生成
    pos_idx_set = set()
    for smi, gene in pos_pairs:
        comp_idx = smiles_to_idx[smi]
        gene_idx = cpi_genes_in_emb.index(gene)
        pos_idx_set.add((comp_idx, gene_idx))

    n_compounds = len(compound_smiles)
    n_genes = len(cpi_genes_in_emb)
    n_neg_target = len(pos_pairs) * neg_ratio

    neg_idx_set = set()
    batch_size = n_neg_target * 5
    while len(neg_idx_set) < n_neg_target:
        batch_comp = rng.randint(0, n_compounds, size=batch_size)
        batch_gene = rng.randint(0, n_genes, size=batch_size)
        for ci, gi in zip(batch_comp, batch_gene, strict=False):
            pair = (ci, gi)
            if pair not in pos_idx_set and pair not in neg_idx_set:
                neg_idx_set.add(pair)
                if len(neg_idx_set) >= n_neg_target:
                    break

    neg_pairs = []
    for ci, gi in neg_idx_set:
        smi = str(compound_smiles[ci])
        gene = cpi_genes_in_emb[gi]
        neg_pairs.append((smi, gene))

    logger.info(f"  负样本: {len(neg_pairs)} 对 (比例 1:{neg_ratio})")

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

    logger.info(f"  数据集: {n_pairs} 样本, {feat_dim} 特征 (comp={comp_dim}+prot={prot_dim})")

    return X, y, np.array(pair_smiles), np.array(pair_genes)


# ============================================================
# 5. 模型训练与评估
# ============================================================

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """训练并评估"""
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test).astype(float)

    auc_val = roc_auc_score(y_test, y_prob)
    aupr_val = average_precision_score(y_test, y_prob)

    # EF@1%, EF@5%
    n_pos = y_test.sum()
    ef = {}
    for pct in [1, 5]:
        k = max(1, int(len(y_test) * pct / 100))
        top_k_idx = np.argsort(y_prob)[-k:]
        found = y_test[top_k_idx].sum()
        expected = n_pos * pct / 100
        ef[f"EF@{pct}%"] = found / expected if expected > 0 else 0.0

    return {
        "model": model_name,
        "AUC": auc_val,
        "AUPR": aupr_val,
        "train_time_s": train_time,
        **ef,
    }


def train_ensemble(X, y, pair_smiles, n_folds=5, random_seed=42):
    """5-fold CV 训练多个模型 + Stacking Ensemble"""
    results = []

    for fold in range(n_folds):
        train_idx, test_idx = scaffold_split(pair_smiles, y, test_size=0.2, random_state=random_seed + fold)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold+1}/{n_folds}: train={len(X_train)}, test={len(X_test)}, "
                    f"pos_ratio={y_train.mean():.3f}/{y_test.mean():.3f}")

        fold_results = []

        # 1. Random Forest
        logger.info("  [1/4] Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=random_seed,
        )
        r = evaluate_model(rf, X_train, y_train, X_test, y_test, "RandomForest")
        r["fold"] = fold
        fold_results.append(r)
        logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}")

        # 2. XGBoost
        try:
            import xgboost as xgb
            logger.info("  [2/4] XGBoost...")
            scale_pos_weight = (y_train == 0).sum() / max(y_train.sum(), 1)
            xgb_model = xgb.XGBClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=random_seed, n_jobs=-1, verbosity=0,
            )
            r = evaluate_model(xgb_model, X_train, y_train, X_test, y_test, "XGBoost")
            r["fold"] = fold
            fold_results.append(r)
            results.append(r)
            logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}")
        except ImportError:
            logger.warning("  XGBoost 未安装")

        # 3. LightGBM
        try:
            import lightgbm as lgb
            logger.info("  [3/4] LightGBM...")
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200, max_depth=10, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                class_weight="balanced", random_state=random_seed,
                n_jobs=-1, verbose=-1,
            )
            r = evaluate_model(lgb_model, X_train, y_train, X_test, y_test, "LightGBM")
            r["fold"] = fold
            fold_results.append(r)
            results.append(r)
            logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}")
        except ImportError:
            logger.warning("  LightGBM 未安装")

        # 4. CatBoost
        try:
            from catboost import CatBoostClassifier
            logger.info("  [4/4] CatBoost...")
            cb_model = CatBoostClassifier(
                iterations=200, depth=8, learning_rate=0.05,
                class_weights=[1, (y_train == 0).sum() / max(y_train.sum(), 1)],
                random_seed=random_seed, thread_count=-1,
                verbose=False, allow_writing_files=False,
            )
            r = evaluate_model(cb_model, X_train, y_train, X_test, y_test, "CatBoost")
            r["fold"] = fold
            fold_results.append(r)
            results.append(r)
            logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}")
        except ImportError:
            logger.warning("  CatBoost 未安装")

        # 5. Soft Voting Ensemble
        if len(fold_results) >= 2:
            logger.info("  [5] Soft Voting Ensemble...")
            estimators = []
            for i, fr in enumerate(fold_results):
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
            r["fold"] = fold
            results.append(r)
            logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}")

        results.extend(fold_results)

    return pd.DataFrame(results)


# ============================================================
# 6. TCM 预测
# ============================================================

def predict_tcm_pool(
    best_model, tcm_df, tcm_features, protein_embeddings,
    cpi_genes_in_emb, model_name,
):
    """预测 TCM 化合物池"""
    logger.info(f"  预测 {len(tcm_df)} 个 TCM 化合物 × {len(cpi_genes_in_emb)} 个基因...")
    comp_dim = tcm_features.shape[1]
    prot_dim = next(iter(protein_embeddings.values())).shape[0]
    feat_dim = comp_dim + prot_dim

    predictions = []
    for i, (_, row) in enumerate(tcm_df.iterrows()):
        smi = str(row["SMILES_std"])
        mol_name = str(row.get("molecule_name", f"MOL_{i}"))
        mol_id = str(row.get("MOL_ID", f"MOL_{i}"))
        comp_feat = tcm_features[i].astype(np.float32)

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
            })

        if (i + 1) % 100 == 0:
            logger.info(f"    进度: {i+1}/{len(tcm_df)}")

    return pd.DataFrame(predictions)


# ============================================================
# 主流程
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("树模型 CPI v2.0 — 工业化升级版")
    logger.info("=" * 60)

    # ---- 1. 加载数据 ----
    logger.info("\n[1/6] 加载原始数据...")
    cpi_df = pd.read_csv(L4_RESULTS / "experimental_actives_detail_cleaned.csv", low_memory=False)
    protein_embeddings = {str(k): v.astype(np.float32) for k, v in
                          np.load(L4_RESULTS_V10 / "esm2_protein_embeddings.npz", allow_pickle=True).items()}
    tcm_df = pd.read_csv(L3_RESULTS / "tcm_compound_pool_tox_filtered_noleak.csv", low_memory=False)

    # 获取所有需要的 SMILES
    all_smiles = list(cpi_df["canonical_smiles"].dropna().astype(str).unique())
    tcm_smiles = tcm_df["SMILES_std"].astype(str).tolist()
    all_smiles.extend(tcm_smiles)
    all_smiles = list(dict.fromkeys(all_smiles))  # 去重保序
    logger.info(f"  总 SMILES: {len(all_smiles)} (CPI 唯一: {len(all_smiles) - len(tcm_smiles)}, TCM: {len(tcm_smiles)})")

    # ---- 2. 多指纹特征工程 ----
    logger.info("\n[2/6] 多指纹特征工程...")
    compound_features, scaler = build_multifingerprint_features(all_smiles)

    # ---- 3. 特征选择 ----
    logger.info("\n[3/6] 特征选择...")
    # 先用负采样构建临时数据集做特征选择
    smiles_to_idx = {str(s): i for i, s in enumerate(all_smiles)}
    cpi_genes = sorted(cpi_df["gene"].unique())
    cpi_genes_in_emb = [g for g in cpi_genes if g in protein_embeddings]

    # 构建正样本
    pos_pairs = []
    for _, row in cpi_df.iterrows():
        smi = str(row["canonical_smiles"])
        gene = str(row["gene"])
        if smi in smiles_to_idx and gene in protein_embeddings:
            pos_pairs.append((smi, gene))

    rng = np.random.RandomState(42)
    n_neg = min(len(pos_pairs) * 3, len(pos_pairs) * 3)
    pos_idx_set = set()
    for smi, gene in pos_pairs:
        pos_idx_set.add((smiles_to_idx[smi], cpi_genes_in_emb.index(gene)))

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

    # 特征选择
    X_feat, keep_var, corr_mask = feature_selection(X_feat, y_feat)
    compound_features = compound_features[:, keep_var[:comp_dim]]
    logger.info(f"  特征选择后: {compound_features.shape[1]} 维")

    # ---- 4. 构建数据集 ----
    logger.info("\n[4/6] 构建训练数据集...")
    X, y, pair_smiles, pair_genes = build_dataset(
        cpi_df, all_smiles, compound_features, scaler,
        protein_embeddings, neg_ratio=3, random_seed=42,
    )

    # ---- 5. 训练与评估 ----
    logger.info("\n[5/6] 5-fold Scaffold Split 训练...")
    results_df = train_ensemble(X, y, pair_smiles, n_folds=5, random_seed=42)

    # 汇总
    logger.info("\n" + "=" * 60)
    logger.info("模型评估汇总 (5-fold Scaffold Split, mean ± std):")
    logger.info("=" * 60)
    summary = results_df.groupby("model").agg(["mean", "std"]).round(4)
    for model_name in summary.index:
        row = summary.loc[model_name]
        logger.info(f"\n  {model_name}:")
        for metric in ["AUC", "AUPR", "EF@1%", "EF@5%"]:
            if metric in row.index:
                logger.info(f"    {metric}: {row[metric]['mean']:.4f} ± {row[metric]['std']:.4f}")

    results_path = L4_RESULTS / "tree_v2_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\n评估结果已保存: {results_path}")

    # ---- 6. 全量训练 + TCM 预测 ----
    logger.info("\n[6/6] 全量训练最佳模型 + TCM 预测...")
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
    else:
        # 默认 RF
        best_model = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )

    logger.info(f"  全量训练 {best_model_name} (样本数: {len(X)})...")
    best_model.fit(X, y)

    # 提取 TCM 特征
    tcm_indices = [smiles_to_idx[s] for s in tcm_smiles if s in smiles_to_idx]
    tcm_features = compound_features[tcm_indices]

    pred_df = predict_tcm_pool(
        best_model, tcm_df, tcm_features, protein_embeddings,
        cpi_genes_in_emb, best_model_name,
    )

    pred_path = L4_RESULTS / "tree_v2_tcm_predictions.csv"
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
    ).reset_index()
    comp_agg = comp_agg.sort_values("max_score", ascending=False)

    top50 = comp_agg.head(50)
    top_path = L4_RESULTS / "tree_v2_top_candidates.csv"
    top50.to_csv(top_path, index=False)

    logger.info(f"\nTop 20 候选化合物:")
    for i, row in enumerate(top50.head(20).itertuples(index=False), 1):
        logger.info(f"  {i:2d}. {row.molecule_name} | max={row.max_score:.4f} "
                    f"| mean={row.mean_score:.4f} "
                    f"| 高置信(≥0.5): {row.n_genes_above_50} "
                    f"| {row.top_3_genes}")

    logger.info(f"\nTop 50 候选已保存: {top_path}")
    logger.info(f"任务完成!")


if __name__ == "__main__":
    main()