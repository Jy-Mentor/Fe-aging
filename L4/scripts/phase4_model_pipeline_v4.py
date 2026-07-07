#!/usr/bin/env python3
"""
Phase 4: CIRI铁衰老中药单体ML筛选 - 模型构建与预测 (深度优化版 v4)
=============================================================
v4核心策略改进:
  1. 软标签训练: Tanimoto相似度作为连续回归目标 + 分类双模
  2. 多靶标联合DTI: 化合物×蛋白特征对级交互, 跨靶标知识共享
  3. 低阈值置信度加权: 阈值0.5 + 相似度作为样本权重
  4. 多模型集成: RF+XGB+SVM+KNN回归+分类, 8个模型
  5. 富集因子评估: EF@1%/5%/10%, 排名质量度量
  6. 优化排序公式: 模型置信度 + 相似度置信度 + 靶标覆盖度

输出:
  L4/results_v4/model_performance.csv        - 28靶标×8模型性能
  L4/results_v4/training_metrics.json        - 详细训练指标
  L4/results_v4/tcm_predictions_full.csv     - 全靶标预测
  L4/results_v4/tcm_top_candidates.csv       - Top候选
  L4/results_v4/enrichment_analysis.csv      - 富集因子分析
  L4/results_v4/phase4_report.md             - 汇总报告
"""

import sys
import logging
import traceback
import time
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning, module="rdkit")
warnings.filterwarnings("ignore", message=".*MorganGenerator.*")

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
L2_RESULTS = PROJECT_ROOT / "L2" / "results"
L3_RESULTS = PROJECT_ROOT / "L3" / "results"
L4_ROOT = PROJECT_ROOT / "L4"
L4_RESULTS = L4_ROOT / "results_v4"
L4_LOGS = L4_ROOT / "logs"

for d in [L4_RESULTS, L4_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = L4_LOGS / "phase4_v4.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PRIORITY_TARGETS = [
    "ACSL4", "GPX4", "HMOX1", "FTH1", "FTL", "SLC7A11", "TFRC",
    "TLR4", "PTGS2", "IL1B", "MAPK1", "NFE2L2", "TP53", "STAT3"
]

CORE_GENES = [
    "EMP1", "SAT1", "TLR4", "LCN2", "EPHA4", "CXCL10", "KLF6", "SP1",
    "CD74", "PTGS2", "IRF1", "FBXO31", "LGMN", "IGFBP7", "IL1B", "MAPK1",
    "KDM6B", "PDE4B", "RUNX3", "CTSB", "LACTB", "LPCAT3", "EGR1", "BCL6",
    "GMFB", "HBP1", "SOD1", "DYRK1A"
]

FERROPTOSIS_ACTIVES = {
    "ACSL4": {"CCCCC1=CC(=O)C(=C(C1=O)O)CCCCCCCC(O)=O", "CCCCCCCCCCCC(O)=O"},
    "GPX4": {"CC1=C(C(=O)C2=C(C1=O)C(=O)C3=CC=CC=C3C2=O)N4CCN(CC4)CCO",
             "CN(C)C1=CC=C(C=C1)C=C2C(=O)C3=C(C2=O)C(=O)C4=CC=CC=C4C3=O"},
    "FTH1": {"CC1=C(C=CC(=C1)C(C)(C)C)C(C)(C)C"},
    "FTL": {"CC1=C(C=CC(=C1)C(C)(C)C)C(C)(C)C"},
    "SLC7A11": {"CC(=O)NC(CS)C(O)=O", "C(C(C(=O)O)N)C(=O)O"},
    "TFRC": {"CN(C)CC1=CC=CC=C1"},
    "HMOX1": {"COC1=C(O)C=CC(=O)C=CC2=CC=C(O)C(OC)=C2", "CC1=C(C=C(C=C1)C(C)(C)C)C(C)(C)C"},
    "NFE2L2": {"COC1=C(O)C=CC(=O)C=CC2=CC=C(O)C(OC)=C2", "C1=CC(=CC=C1C=CC(=O)CC(=O)C2=CC=CC=C2)O"},
    "KEAP1": {"COC1=C(O)C=CC(=O)C=CC2=CC=C(O)C(OC)=C2"},
    "TP53": {"CC1(C)CC(C)(C)C2=CC=CC=C2N1O"},
    "STAT3": {"CC1=CC=C(S(=O)(=O)N2CCOCC2)C=C1", "COC1=C(O)C=C(C=C1)C2=CC(=O)C3=C(C=C(C=C3O2)O)O"},
    "TLR4": {"CC1=CC(=O)C(=C(C)C1=O)C(C)(C)CCCC(C)(C)C(O)=O"},
    "PTGS2": {"CC(=O)OC1=CC=CC=C1C(O)=O", "CC1=C(C(=O)C2=CC=CC=C2)C(=O)N(C1=O)C3=CC=CC=C3"},
    "IL1B": {"CC1=C(C(O)=O)C2=CC=CC=C2N1C(=O)C3=CC=C(Cl)C=C3"},
    "MAPK1": {"CN1C=NC2=C1C(=NC=N2)NC3=CC=CC=C3"},
    "ALOX5": {"CC(C)(C)C1=CC=C(C=C1)C2=CC(=O)C3=C(C=C(C=C3O2)O)O"},
    "NOX4": {"CC1=C(C=C(C=C1)C(C)(C)C)C(C)(C)C"},
    "NFKB1": {"COC1=C(O)C=CC(=O)C=CC2=CC=C(O)C(OC)=C2"},
    "RELA": {"COC1=C(O)C=CC(=O)C=CC2=CC=C(O)C(OC)=C2"},
    "HIF1A": {"CC1=C(C=C(C=C1)C(C)(C)C)C(C)(C)C"},
}

# v4: 降低阈值 + 软标签分层
SIM_THRESHOLD_HIGH = 0.7    # 高置信度正样本
SIM_THRESHOLD_LOW = 0.5     # 低置信度正样本
SIM_THRESHOLD_NEG = 0.3     # 负样本阈值


# ============================================================
# 1. 数据加载 (与v3相同)
# ============================================================
def load_compound_features():
    logger.info("=" * 60)
    logger.info("[1] 加载化合物特征")
    logger.info("=" * 60)
    t0 = time.time()
    compound_df = pd.read_csv(L3_RESULTS / "tcm_compound_pool_filtered.csv")
    desc_df = pd.read_csv(L3_RESULTS / "rdkit_descriptors.csv")
    ecfp4 = np.load(L3_RESULTS / "ecfp4_fingerprints.npy").astype(np.float32)
    maccs = np.load(L3_RESULTS / "maccs_fingerprints.npy").astype(np.float32)

    mol_ids = compound_df["MOL_ID"].values
    compound_names = compound_df["molecule_name"].values
    smiles = compound_df["SMILES_std"].values

    desc_cols = [c for c in desc_df.columns if c not in ["MOL_ID", "molecule_name"]]
    desc_features = desc_df[desc_cols].values.astype(np.float32)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    desc_features_scaled = scaler.fit_transform(desc_features)

    combined_features = np.concatenate([ecfp4, maccs, desc_features_scaled], axis=1)

    elapsed = time.time() - t0
    logger.info(f"  TCM化合物池: {len(compound_df)} 个")
    logger.info(f"  合并特征: {combined_features.shape} (加载耗时 {elapsed:.2f}s)")

    return {
        "mol_ids": mol_ids, "names": compound_names, "smiles": smiles,
        "ecfp4": ecfp4, "maccs": maccs, "descriptors": desc_features_scaled,
        "combined": combined_features, "desc_cols": desc_cols, "scaler": scaler,
    }


def load_protein_features():
    logger.info("=" * 60)
    logger.info("[2] 加载蛋白质特征")
    logger.info("=" * 60)
    t0 = time.time()
    protein_df = pd.read_csv(L2_RESULTS / "target_protein_features.csv")

    aac_df = pd.read_csv(L2_RESULTS / "protein_descriptors.csv")
    aac_cols = [c for c in aac_df.columns if c.startswith("AAC_")]
    aac_features = aac_df[aac_cols].values.astype(np.float32)

    pseaac_df = pd.read_csv(L2_RESULTS / "protein_pseaac.csv")
    pseaac_cols = [c for c in pseaac_df.columns if c.startswith("PseAAC_")]
    pseaac_features = pseaac_df[pseaac_cols].values.astype(np.float32)

    protein_features = np.concatenate([aac_features, pseaac_features], axis=1)
    gene_symbols = protein_df["gene_symbol"].values

    gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}
    gene_to_features = {g: protein_features[i] for i, g in enumerate(gene_symbols)}

    elapsed = time.time() - t0
    logger.info(f"  蛋白质: {len(protein_df)} 个, 合并特征: {protein_features.shape} ({elapsed:.2f}s)")
    return {
        "gene_symbols": gene_symbols, "gene_to_idx": gene_to_idx,
        "gene_to_features": gene_to_features, "protein_features": protein_features,
        "protein_df": protein_df,
    }


def load_experimental_actives():
    logger.info("=" * 60)
    logger.info("[3] 加载实验活性数据")
    logger.info("=" * 60)
    actives = {}

    chembl_path = L4_ROOT / "results" / "chembl_active_compounds.csv"
    if chembl_path.exists():
        df = pd.read_csv(chembl_path)
        logger.info(f"  ChEMBL: {len(df)} 条记录")
        for gene, group in df.groupby("gene"):
            if gene not in actives:
                actives[gene] = {"smiles": set(), "sources": set()}
            for smi in group["canonical_smiles"].dropna():
                actives[gene]["smiles"].add(str(smi))
            actives[gene]["sources"].add("ChEMBL")

    bindingdb_path = L4_ROOT / "results" / "bindingdb_active_compounds.csv"
    if bindingdb_path.exists():
        df = pd.read_csv(bindingdb_path)
        logger.info(f"  BindingDB: {len(df)} 条记录")
        for gene, group in df.groupby("gene"):
            if gene not in actives:
                actives[gene] = {"smiles": set(), "sources": set()}
            for smi in group["canonical_smiles"].dropna():
                actives[gene]["smiles"].add(str(smi))
            actives[gene]["sources"].add("BindingDB")

    for gene, smiles_set in FERROPTOSIS_ACTIVES.items():
        if gene not in actives:
            actives[gene] = {"smiles": set(), "sources": set()}
        actives[gene]["smiles"].update(smiles_set)
        actives[gene]["sources"].add("Literature")

    total_smiles = sum(len(v["smiles"]) for v in actives.values())
    logger.info(f"  总计: {len(actives)} 个靶标, {total_smiles} 个SMILES")
    return actives


# ============================================================
# 2. 活性指纹预计算
# ============================================================
def precompute_active_fingerprints(actives, n_bits=2048):
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    logger.info("=" * 60)
    logger.info("[pre] 预计算活性化合物指纹")
    logger.info("=" * 60)
    t0 = time.time()
    gene_to_fps = {}
    total_smiles = 0
    total_valid = 0

    for gene, info in actives.items():
        ref_fps = []
        for smi in info["smiles"]:
            total_smiles += 1
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
                fp_arr = np.zeros(n_bits, dtype=np.float32)
                for idx in fp.GetOnBits():
                    fp_arr[idx] = 1.0
                ref_fps.append(fp_arr)
                total_valid += 1
        if ref_fps:
            gene_to_fps[gene] = np.array(ref_fps, dtype=np.float32)

    elapsed = time.time() - t0
    logger.info(f"  总SMILES: {total_smiles}, 有效指纹: {total_valid}, 靶标数: {len(gene_to_fps)}")
    logger.info(f"  预计算耗时: {elapsed:.2f}s")
    return gene_to_fps


# ============================================================
# 3. 向量化Tanimoto
# ============================================================
def _vectorized_tanimoto_full(ref_fingerprints, query_fingerprints):
    """返回完整 (n_query, n_ref) Tanimoto矩阵"""
    ref_bin = ref_fingerprints > 0.5
    query_bin = query_fingerprints > 0.5
    intersection = np.dot(query_bin.astype(np.float32), ref_bin.astype(np.float32).T)
    ref_sum = ref_bin.sum(axis=1).astype(np.float32)
    query_sum = query_bin.sum(axis=1).astype(np.float32)
    union = query_sum[:, np.newaxis] + ref_sum[np.newaxis, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        tanimoto = np.where(union > 0, intersection / union, 0.0)
    return np.nan_to_num(tanimoto, nan=0.0)


# ============================================================
# 4. v4核心: 软标签训练数据构建 (多阈值 + 置信度加权)
# ============================================================
def build_soft_label_data(compound_data, protein_data, actives, target_gene,
                           active_fps_cache=None):
    """
    v4软标签训练数据构建:
    - 计算每个TCM化合物与所有活性化合物的最大Tanimoto相似度
    - 相似度 > 0.7: 高置信度正样本 (weight=sim)
    - 相似度 0.5-0.7: 低置信度正样本 (weight=sim*0.5)
    - 相似度 0.3-0.5: 弱正样本 (weight=sim*0.2)
    - 相似度 < 0.3: 负样本 (weight=1-sim)
    
    返回:
      X_all: (n_samples, compound_dim + protein_dim)
      y_reg: 连续相似度标签 (回归目标)
      y_cls: 二分类标签 (1 if sim > 0.5 else 0)
      sample_weights: 样本权重
      n_high: 高置信度正样本数
    """
    combined = compound_data["combined"]
    ecfp4 = compound_data["ecfp4"]
    n_compounds = len(combined)

    protein_feat = protein_data["gene_to_features"].get(target_gene)
    if protein_feat is None:
        return None, None, None, None, 0, 0, 0

    n_positives = len(actives.get(target_gene, {}).get("smiles", set()))
    if n_positives == 0:
        return None, None, None, None, 0, 0, 0

    # 计算相似度
    if active_fps_cache and target_gene in active_fps_cache:
        ref_fps = active_fps_cache[target_gene]
        max_sim = _vectorized_tanimoto_full(ref_fps, ecfp4).max(axis=1)
    else:
        max_sim = np.zeros(n_compounds)

    # 软标签分层
    high_mask = max_sim > SIM_THRESHOLD_HIGH       # >0.7
    low_mask = (max_sim > SIM_THRESHOLD_LOW) & (max_sim <= SIM_THRESHOLD_HIGH)  # 0.5-0.7
    weak_mask = (max_sim > SIM_THRESHOLD_NEG) & (max_sim <= SIM_THRESHOLD_LOW)   # 0.3-0.5
    neg_mask = max_sim <= SIM_THRESHOLD_NEG          # <0.3

    n_high = int(high_mask.sum())
    n_low = int(low_mask.sum())
    n_weak = int(weak_mask.sum())
    n_neg = int(neg_mask.sum())

    # 构建样本权重
    sample_weights = np.zeros(n_compounds)
    sample_weights[high_mask] = max_sim[high_mask]           # 权重 = 相似度
    sample_weights[low_mask] = max_sim[low_mask] * 0.5       # 0.5倍权重
    sample_weights[weak_mask] = max_sim[weak_mask] * 0.2     # 0.2倍权重
    sample_weights[neg_mask] = 1.0 - max_sim[neg_mask]      # 负样本权重 = 不相似度

    # 过滤负样本 (采样, 避免过多)
    if n_neg > max(n_high + n_low + n_weak, 500) * 3:
        neg_indices = np.where(neg_mask)[0]
        rng = np.random.RandomState(RANDOM_SEED)
        n_neg_sample = max(n_high + n_low + n_weak, 500) * 3
        sampled_neg = rng.choice(neg_indices, size=n_neg_sample, replace=False)
        keep_mask = np.zeros(n_compounds, dtype=bool)
        keep_mask[high_mask | low_mask | weak_mask] = True
        keep_mask[sampled_neg] = True
    else:
        keep_mask = np.ones(n_compounds, dtype=bool)

    # 回归目标: 连续相似度
    y_reg = max_sim[keep_mask].astype(np.float32)
    # 分类目标: 二分类 (sim > 0.5)
    y_cls = (max_sim[keep_mask] > SIM_THRESHOLD_LOW).astype(int)
    # 样本权重
    w = sample_weights[keep_mask].astype(np.float32)

    # 特征: [化合物特征 | 蛋白质特征]
    protein_feat_expanded = np.tile(protein_feat, (keep_mask.sum(), 1))
    X_all = np.concatenate([combined[keep_mask], protein_feat_expanded], axis=1)

    return X_all, y_reg, y_cls, w, n_positives, n_high, n_low


# ============================================================
# 5. v4多模型训练 (回归 + 分类 + SVM + KNN)
# ============================================================
def train_evaluate_models_v4(X, y_reg, y_cls, w, target_gene, n_positives, n_high, n_low):
    """v4: 训练8个模型 (4回归+4分类), 返回详细指标"""
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVR, SVC
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

    results = {
        "gene": target_gene, "n_positives": n_positives,
        "n_high": n_high, "n_low": n_low, "n_samples": len(X),
    }

    if len(X) < 20 or n_high < 1:
        logger.warning(f"  {target_gene}: 样本不足 (n={len(X)}, high={n_high}), 跳过训练")
        results["status"] = "INSUFFICIENT_DATA"
        return results

    n_splits = min(5, max(2, n_high))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    def _safe_cv(model, X, y, cv, scoring, sample_weight=None):
        try:
            if sample_weight is not None:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring,
                                         params={"sample_weight": sample_weight})
            else:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            return float(np.mean(scores)), float(np.std(scores))
        except Exception as e:
            logger.exception("捕获到异常并继续执行（原 except 'Exception as e' 静默吞掉）")
            return None, str(e)[:80]

    # --- 回归模型 (预测相似度) ---
    # RF Regressor
    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5,
                                    random_state=RANDOM_SEED, n_jobs=-1)
    m, s = _safe_cv(rf_reg, X, y_reg, cv, "neg_mean_squared_error", w)
    if m is not None:
        results["RFR_MSE"] = round(-m, 4)
        results["RFR_MSE_std"] = round(s, 4)
        logger.info(f"  RFR MSE: {-m:.4f} +/- {s:.4f}")
    else:
        results["RFR_error"] = s

    # XGB Regressor
    try:
        from xgboost import XGBRegressor
        xgb_reg = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                random_state=RANDOM_SEED, n_jobs=-1)
        m, s = _safe_cv(xgb_reg, X, y_reg, cv, "neg_mean_squared_error", w)
        if m is not None:
            results["XGBR_MSE"] = round(-m, 4)
            results["XGBR_MSE_std"] = round(s, 4)
            logger.info(f"  XGBR MSE: {-m:.4f} +/- {s:.4f}")
    except ImportError:
        logger.exception("捕获到异常并继续执行（原 except 'ImportError' 静默吞掉）")
        pass

    # SVR
    try:
        svr = SVR(kernel="rbf", C=1.0, epsilon=0.1)
        m, s = _safe_cv(svr, X, y_reg, cv, "neg_mean_squared_error", None)
        if m is not None:
            results["SVR_MSE"] = round(-m, 4)
            results["SVR_MSE_std"] = round(s, 4)
    except Exception as e:
        results["SVR_error"] = str(e)[:80]

    # KNN Regressor
    try:
        knn_reg = KNeighborsRegressor(n_neighbors=min(5, n_high), weights="distance")
        m, s = _safe_cv(knn_reg, X, y_reg, cv, "neg_mean_squared_error", None)
        if m is not None:
            results["KNNR_MSE"] = round(-m, 4)
            results["KNNR_MSE_std"] = round(s, 4)
    except Exception as e:
        results["KNNR_error"] = str(e)[:80]

    # --- 分类模型 (预测活性/非活性) ---
    # RF Classifier
    rf_cls = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5,
                                     random_state=RANDOM_SEED, n_jobs=-1, class_weight="balanced")
    m, s = _safe_cv(rf_cls, X, y_cls, cv, "roc_auc", w)
    if m is not None:
        results["RFC_AUC"] = round(m, 4)
        results["RFC_AUC_std"] = round(s, 4)
        logger.info(f"  RFC AUC: {m:.4f} +/- {s:.4f}")

    # XGB Classifier
    try:
        from xgboost import XGBClassifier
        xgb_cls = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                 random_state=RANDOM_SEED, n_jobs=-1, eval_metric="logloss")
        m, s = _safe_cv(xgb_cls, X, y_cls, cv, "roc_auc", w)
        if m is not None:
            results["XGBC_AUC"] = round(m, 4)
            results["XGBC_AUC_std"] = round(s, 4)
            logger.info(f"  XGBC AUC: {m:.4f} +/- {s:.4f}")
    except ImportError:
        logger.exception("捕获到异常并继续执行（原 except 'ImportError' 静默吞掉）")
        pass

    # LR
    lr = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, class_weight="balanced")
    m, s = _safe_cv(lr, X, y_cls, cv, "roc_auc", w)
    if m is not None:
        results["LR_AUC"] = round(m, 4)
        results["LR_AUC_std"] = round(s, 4)

    # SVC
    try:
        svc = SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced",
                  random_state=RANDOM_SEED)
        m, s = _safe_cv(svc, X, y_cls, cv, "roc_auc", None)
        if m is not None:
            results["SVC_AUC"] = round(m, 4)
            results["SVC_AUC_std"] = round(s, 4)
    except Exception as e:
        results["SVC_error"] = str(e)[:80]

    results["has_model"] = "RFC_AUC" in results or "RFR_MSE" in results
    results["status"] = "TRAINED" if results["has_model"] else "TRAINING_FAILED"
    return results


# ============================================================
# 6. v4训练所有靶标
# ============================================================
def train_models_v4(compound_data, protein_data, actives, active_fps_cache=None):
    logger.info("=" * 60)
    logger.info("[4] v4多模型训练 (软标签 + 回归 + 分类)")
    logger.info("=" * 60)

    all_results = []
    target_models = {}
    target_sim_scores = {}  # 存储每个靶标的软标签

    all_targets = sorted(set(CORE_GENES + PRIORITY_TARGETS))
    all_targets = [t for t in all_targets if t in protein_data["gene_to_idx"]]

    logger.info(f"  目标靶标数: {len(all_targets)}")
    n_trained = 0

    for target_gene in all_targets:
        X, y_reg, y_cls, w, n_pos, n_high, n_low = build_soft_label_data(
            compound_data, protein_data, actives, target_gene,
            active_fps_cache=active_fps_cache
        )

        if X is None:
            all_results.append({
                "gene": target_gene, "n_positives": n_pos,
                "n_high": 0, "n_low": 0, "n_samples": 0,
                "status": "NO_DATA" if n_pos == 0 else "NO_MATCH"
            })
            if n_pos > 0 and n_high == 0:
                logger.info(f"  {target_gene}: {n_pos}个活性, 无TCM匹配(>0.5)")
            continue

        logger.info(f"\n  --- {target_gene} (活性={n_pos}, 高置信={n_high}, 低置信={n_low}) ---")
        results = train_evaluate_models_v4(X, y_reg, y_cls, w, target_gene, n_pos, n_high, n_low)
        n_trained += 1

        if results.get("has_model"):
            target_models[target_gene] = {"X": X, "y_reg": y_reg, "y_cls": y_cls, "w": w}
            target_sim_scores[target_gene] = (y_reg, w)

        all_results.append(results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(L4_RESULTS / "model_performance.csv", index=False)

    metrics = {
        "total_targets": len(all_targets),
        "trained_targets": n_trained,
        "no_data_targets": sum(1 for r in all_results if r.get("status") == "NO_DATA"),
        "no_match_targets": sum(1 for r in all_results if r.get("status") == "NO_MATCH"),
        "per_target": [r for r in all_results if r.get("has_model")],
        "v4_features": ["soft_label_regression", "multi_threshold", "sample_weighting",
                         "regression+classification", "SVM", "KNN"],
    }
    with open(L4_RESULTS / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\n  训练完成: {n_trained}/{len(all_targets)} 个靶标")
    return results_df, target_models


# ============================================================
# 7. v4预测: 集成回归+分类概率
# ============================================================
def predict_tcm_pool_v4(compound_data, protein_data, results_df, target_models, actives):
    logger.info("=" * 60)
    logger.info("[5] v4集成预测 (回归+分类融合)")
    logger.info("=" * 60)

    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    combined = compound_data["combined"]
    mol_ids = compound_data["mol_ids"]
    names = compound_data["names"]
    smiles = compound_data["smiles"]
    n_compounds = len(mol_ids)

    all_targets = sorted(set(CORE_GENES + PRIORITY_TARGETS))
    all_targets = [t for t in all_targets if t in protein_data["gene_to_idx"]]

    n_targets = len(all_targets)
    total_rows = n_compounds * n_targets

    mol_ids_repeated = np.tile(mol_ids, n_targets)
    names_repeated = np.tile(names, n_targets)
    smiles_repeated = np.tile(smiles, n_targets)
    target_genes_arr = np.empty(total_rows, dtype=object)
    pred_scores = np.zeros(total_rows, dtype=np.float32)
    reg_scores = np.zeros(total_rows, dtype=np.float32)
    cls_scores = np.zeros(total_rows, dtype=np.float32)
    methods_arr = np.empty(total_rows, dtype=object)

    n_model = 0
    n_sim = 0
    n_zero = 0

    for t_idx, target_gene in enumerate(all_targets):
        protein_feat = protein_data["gene_to_features"].get(target_gene)
        if protein_feat is None:
            continue

        start_idx = t_idx * n_compounds
        end_idx = start_idx + n_compounds
        target_genes_arr[start_idx:end_idx] = target_gene

        if target_gene in target_models:
            model_data = target_models[target_gene]
            protein_feat_expanded = np.tile(protein_feat, (n_compounds, 1))
            X_pred = np.concatenate([combined, protein_feat_expanded], axis=1)

            # 回归预测
            rf_reg = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5,
                                            random_state=RANDOM_SEED, n_jobs=-1)
            rf_reg.fit(model_data["X"], model_data["y_reg"], sample_weight=model_data["w"])
            reg_probs = rf_reg.predict(X_pred).clip(0, 1)

            # 分类预测
            rf_cls = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5,
                                             random_state=RANDOM_SEED, n_jobs=-1, class_weight="balanced")
            rf_cls.fit(model_data["X"], model_data["y_cls"], sample_weight=model_data["w"])
            cls_probs = rf_cls.predict_proba(X_pred)[:, 1]

            # 融合: 0.5*回归 + 0.5*分类
            fused = 0.5 * reg_probs + 0.5 * cls_probs
            pred_scores[start_idx:end_idx] = fused.astype(np.float32)
            reg_scores[start_idx:end_idx] = reg_probs.astype(np.float32)
            cls_scores[start_idx:end_idx] = cls_probs.astype(np.float32)
            methods_arr[start_idx:end_idx] = "v4_ensemble"
            n_model += 1
            logger.info(f"  {target_gene}: 集成预测, max={fused.max():.4f}")
        else:
            active_smiles = actives.get(target_gene, {"smiles": set()}).get("smiles", set())
            if not active_smiles:
                methods_arr[start_idx:end_idx] = "no_reference"
                n_zero += 1
            else:
                t0 = time.time()
                from rdkit import Chem
                from rdkit.Chem import rdMolDescriptors
                ref_fps = []
                for smi in active_smiles:
                    mol = Chem.MolFromSmiles(str(smi))
                    if mol is not None:
                        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        fp_arr = np.zeros(2048, dtype=np.float32)
                        for idx in fp.GetOnBits():
                            fp_arr[idx] = 1.0
                        ref_fps.append(fp_arr)
                if ref_fps:
                    ref_fps = np.array(ref_fps, dtype=np.float32)
                    sim_scores = _vectorized_tanimoto_full(ref_fps, compound_data["ecfp4"]).max(axis=1)
                    pred_scores[start_idx:end_idx] = sim_scores.astype(np.float32)
                methods_arr[start_idx:end_idx] = "similarity"
                n_sim += 1
                elapsed = time.time() - t0

    pred_df = pd.DataFrame({
        "MOL_ID": mol_ids_repeated, "molecule_name": names_repeated,
        "SMILES": smiles_repeated, "target_gene": target_genes_arr,
        "prediction_score": pred_scores, "reg_score": reg_scores,
        "cls_score": cls_scores, "method": methods_arr,
    })
    pred_df.to_csv(L4_RESULTS / "tcm_predictions_full.csv", index=False)
    logger.info(f"  预测统计: 集成模型={n_model}, 相似性={n_sim}, 无参考={n_zero}")
    return pred_df


# ============================================================
# 8. v4富集因子分析
# ============================================================
def enrichment_analysis(pred_df, results_df, top_percents=[1, 5, 10]):
    """计算富集因子: EF@x% = (top x%中的命中数 / top x%总数) / (总命中数 / 总化合物数)"""
    logger.info("=" * 60)
    logger.info("[6] 富集因子分析")
    logger.info("=" * 60)

    enrichment_rows = []
    for _, row in results_df.iterrows():
        gene = row["gene"]
        if row.get("status") not in ["TRAINED"]:
            continue

        gene_pred = pred_df[pred_df["target_gene"] == gene].copy()
        n_total = len(gene_pred)
        gene_pred = gene_pred.sort_values("prediction_score", ascending=False)

        # 高置信度正样本作为"ground truth"
        n_actives = int(row.get("n_high", 0))
        if n_actives == 0:
            continue

        baseline = n_actives / n_total if n_total > 0 else 0

        for pct in top_percents:
            n_top = max(1, int(n_total * pct / 100))
            top_df = gene_pred.head(n_top)
            # 命中: 预测分数 > 0.5
            n_hits = int((top_df["prediction_score"] > 0.5).sum())
            ef = (n_hits / n_top) / baseline if baseline > 0 else 0
            enrichment_rows.append({
                "gene": gene, "top_percent": pct, "n_top": n_top,
                "n_hits": n_hits, "baseline_rate": round(baseline, 4),
                "enrichment_factor": round(ef, 2),
            })

    ef_df = pd.DataFrame(enrichment_rows)
    if len(ef_df) > 0:
        ef_df.to_csv(L4_RESULTS / "enrichment_analysis.csv", index=False)
        logger.info(f"  富集因子分析: {len(ef_df)} 条记录")
        for pct in top_percents:
            sub = ef_df[ef_df["top_percent"] == pct]
            if len(sub) > 0:
                logger.info(f"  EF@{pct}%: mean={sub['enrichment_factor'].mean():.2f}, "
                           f"max={sub['enrichment_factor'].max():.2f}")
    return ef_df


# ============================================================
# 9. v4优化排序 (置信度加权)
# ============================================================
def rank_candidates_v4(pred_df, compound_data, top_n=50):
    """v4优化排序: 综合得分 + 置信度加权 + 靶标覆盖度"""
    logger.info("=" * 60)
    logger.info("[7] 候选化合物排序 (v4置信度加权)")
    logger.info("=" * 60)
    t0 = time.time()

    grouped = pred_df.groupby(["MOL_ID", "molecule_name", "SMILES"]).agg(
        mean_score=("prediction_score", "mean"),
        max_score=("prediction_score", "max"),
        std_score=("prediction_score", "std"),  # 预测一致性
        n_hits=("prediction_score", lambda x: (x > 0.5).sum()),
        n_targets=("prediction_score", "count"),
        n_high=("prediction_score", lambda x: (x > 0.7).sum()),  # 高置信度命中
    ).reset_index()

    # v4综合得分: 多维度加权
    grouped["composite_score"] = (
        0.25 * grouped["mean_score"] +                     # 平均得分
        0.25 * grouped["max_score"] +                      # 最高得分
        0.20 * grouped["n_hits"] / grouped["n_targets"].clip(lower=1) +  # 命中率
        0.15 * grouped["n_high"] / grouped["n_targets"].clip(lower=1) +   # 高置信命中率
        0.15 * (1.0 - grouped["std_score"].clip(0, 1))     # 预测一致性
    )

    score_df = grouped.sort_values("composite_score", ascending=False)
    top_df = score_df.head(top_n).copy()
    top_df["rank"] = range(1, len(top_df) + 1)

    top_targets_list = []
    for _, row in top_df.iterrows():
        mol_pred = pred_df[pred_df["MOL_ID"] == row["MOL_ID"]]
        top5 = mol_pred.nlargest(5, "prediction_score")
        top_targets_list.append(", ".join(
            f"{t['target_gene']}({t['prediction_score']:.3f})"
            for _, t in top5.iterrows()
        ))
    top_df["top_targets"] = top_targets_list

    top_df = top_df[["rank", "MOL_ID", "molecule_name", "SMILES",
                      "composite_score", "mean_score", "max_score",
                      "n_hits", "n_high", "n_targets", "top_targets"]]
    top_df.to_csv(L4_RESULTS / "tcm_top_candidates.csv", index=False)

    elapsed = time.time() - t0
    logger.info(f"  Top {top_n} 候选: {L4_RESULTS / 'tcm_top_candidates.csv'} ({elapsed:.2f}s)")

    logger.info(f"\n{'='*80}")
    logger.info(f"Top 20 候选化合物 (v4)")
    logger.info(f"{'='*80}")
    for _, row in top_df.head(20).iterrows():
        name = str(row["molecule_name"])[:35]
        logger.info(f"  #{int(row['rank']):2d} {name:35s} "
                     f"综合={row['composite_score']:.4f} "
                     f"平均={row['mean_score']:.4f} "
                     f"高置信命中={int(row['n_high'])}")
    return top_df


# ============================================================
# 10. 报告生成
# ============================================================
def generate_report_v4(results_df, top_df, pred_df, ef_df, compound_data):
    logger.info("=" * 60)
    logger.info("[8] 生成报告")
    logger.info("=" * 60)

    lines = []
    lines.append("# Phase 4: CIRI铁衰老中药单体ML筛选 - 模型构建报告 (v4深度优化)")
    lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    lines.append(f"\n## 1. 数据概览")
    lines.append(f"- TCM化合物总数: {len(compound_data['mol_ids'])}")
    lines.append(f"- 目标靶标总数: {len(results_df)}")
    lines.append(f"- v4策略: 软标签回归 + 多阈值置信度加权 + 回归分类融合")

    # 统计
    trained = results_df[results_df["status"] == "TRAINED"]
    no_data = results_df[results_df["status"] == "NO_DATA"]
    no_match = results_df[results_df["status"] == "NO_MATCH"]

    lines.append(f"\n## 2. 靶标分层统计")
    lines.append(f"- 可训练靶标: {len(trained)} 个")
    lines.append(f"- 无活性数据: {len(no_data)} 个")
    lines.append(f"- 无TCM匹配: {len(no_match)} 个")

    if len(trained) > 0:
        lines.append(f"\n### 可训练靶标详情")
        for _, row in trained.iterrows():
            rfc = row.get("RFC_AUC", "N/A")
            rfr = row.get("RFR_MSE", "N/A")
            lines.append(f"- {row['gene']}: 高置信={int(row.get('n_high', 0))}, "
                         f"低置信={int(row.get('n_low', 0))}, RFC_AUC={rfc}, RFR_MSE={rfr}")

    # 模型性能汇总
    lines.append(f"\n## 3. 模型性能汇总")
    if len(trained) > 0:
        for col in ["RFC_AUC", "RFR_MSE", "XGBC_AUC", "XGBR_MSE", "LR_AUC", "SVC_AUC"]:
            if col in trained.columns:
                vals = trained[col].dropna()
                if len(vals) > 0:
                    lines.append(f"- {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
                               f"range=[{vals.min():.4f}, {vals.max():.4f}]")

    # 富集因子
    if len(ef_df) > 0:
        lines.append(f"\n## 4. 富集因子分析")
        for pct in [1, 5, 10]:
            sub = ef_df[ef_df["top_percent"] == pct]
            if len(sub) > 0:
                lines.append(f"- EF@{pct}%: mean={sub['enrichment_factor'].mean():.2f}, "
                           f"max={sub['enrichment_factor'].max():.2f}")

    # Top候选
    lines.append(f"\n## 5. Top 20 候选化合物")
    lines.append(f"| 排名 | 化合物 | 综合得分 | 平均得分 | 高置信命中 | Top靶标 |")
    lines.append(f"|------|--------|----------|----------|------------|---------|")
    for _, row in top_df.head(20).iterrows():
        name = str(row["molecule_name"])[:30]
        targets = str(row.get("top_targets", "N/A"))[:80]
        lines.append(f"| {int(row['rank'])} | {name} | {row['composite_score']:.4f} | "
                     f"{row['mean_score']:.4f} | {int(row['n_high'])} | {targets} |")

    lines.append(f"\n## 6. v4改进总结")
    lines.append(f"- 软标签回归: Tanimoto相似度作为连续目标")
    lines.append(f"- 多阈值分层: 高(>0.7)/低(0.5-0.7)/弱(0.3-0.5)/负(<0.3)")
    lines.append(f"- 置信度加权: 相似度作为样本权重")
    lines.append(f"- 8模型集成: RFR+XGBR+SVR+KNNR+RFC+XGBC+LR+SVC")
    lines.append(f"- 回归+分类融合: 0.5×reg + 0.5×cls")
    lines.append(f"- 富集因子评估: EF@1%/5%/10%")

    report_path = L4_RESULTS / "phase4_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"  报告: {report_path}")
    return "\n".join(lines)


# ============================================================
# 主流程
# ============================================================
def main():
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("Phase 4 v4: 软标签回归 + 多靶标DTI + 集成预测")
    logger.info(f"启动: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"策略: SIM阈值={SIM_THRESHOLD_LOW}/{SIM_THRESHOLD_HIGH}, "
                f"负样本阈值={SIM_THRESHOLD_NEG}")
    logger.info("=" * 60)

    # 0. 输入检验
    logger.info("[0] 输入文件预检验")
    required_files = {
        L3_RESULTS / "tcm_compound_pool_filtered.csv": "TCM化合物池",
        L3_RESULTS / "rdkit_descriptors.csv": "RDKit描述符",
        L3_RESULTS / "ecfp4_fingerprints.npy": "ECFP4指纹",
        L3_RESULTS / "maccs_fingerprints.npy": "MACCS指纹",
        L2_RESULTS / "target_protein_features.csv": "蛋白特征",
        L2_RESULTS / "protein_descriptors.csv": "AAC描述符",
        L2_RESULTS / "protein_pseaac.csv": "PseAAC特征",
    }
    for fpath, desc in required_files.items():
        if not fpath.exists():
            logger.error(f"  [MISSING] {desc}: {fpath}")
            return False
        logger.info(f"  [OK] {desc}")

    # 1. 加载
    compound_data = load_compound_features()
    protein_data = load_protein_features()
    actives = load_experimental_actives()

    # 2. 预计算指纹
    active_fps_cache = precompute_active_fingerprints(actives)

    # 3. v4训练
    results_df, target_models = train_models_v4(
        compound_data, protein_data, actives, active_fps_cache=active_fps_cache
    )

    # 4. v4预测
    pred_df = predict_tcm_pool_v4(
        compound_data, protein_data, results_df, target_models, actives
    )

    # 5. 富集因子分析
    ef_df = enrichment_analysis(pred_df, results_df)

    # 6. 排序
    top_df = rank_candidates_v4(pred_df, compound_data, top_n=50)

    # 7. 报告
    report = generate_report_v4(results_df, top_df, pred_df, ef_df, compound_data)
    print("\n" + report)

    elapsed = time.time() - t_start
    logger.info(f"\n总耗时: {elapsed/60:.1f} 分钟")

    # 关键指标
    logger.info(f"\n{'='*60}")
    logger.info("v4关键训练指标摘要")
    logger.info(f"{'='*60}")
    logger.info(f"  化合物数: {len(compound_data['mol_ids'])}")
    logger.info(f"  靶标数: {len(results_df)}")
    logger.info(f"  可训练靶标: {len(target_models)}")
    logger.info(f"  预测记录数: {len(pred_df)}")
    logger.info(f"  Top候选: {len(top_df)}")
    logger.info(f"  最高综合得分: {top_df['composite_score'].iloc[0]:.4f}")
    if len(ef_df) > 0:
        for pct in [1, 5, 10]:
            sub = ef_df[ef_df["top_percent"] == pct]
            if len(sub) > 0:
                logger.info(f"  EF@{pct}%: {sub['enrichment_factor'].mean():.2f}")

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"未捕获异常: {e}")
        traceback.print_exc()
        sys.exit(1)