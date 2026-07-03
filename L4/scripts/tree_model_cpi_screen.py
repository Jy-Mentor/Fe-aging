#!/usr/bin/env python
"""树模型 CPI 筛选：铁衰老-CIRI 化合物发现

使用 Random Forest / XGBoost / LightGBM 预测化合物-蛋白互作 (CPI)，
从 TCM 化合物池中筛选潜在的铁衰老/CIRI 靶标活性化合物。

数据来源（全部真实，不模拟）：
  - CPI 正样本: L4/results/experimental_actives_detail_cleaned.csv
  - 化合物特征: L4/results_v10_minibatch/compound_features_v31.npz (ECFP4+MACCS+RDKit)
  - 蛋白特征: L4/results_v10_minibatch/esm2_protein_embeddings.npz (ESM-2 640维)
  - TCM 候选池: L3/results/tcm_compound_pool_tox_filtered_noleak.csv

输出：
  - L4/results/tree_model_cpi_results.csv: 模型评估指标
  - L4/results/tree_model_tcm_predictions.csv: TCM 化合物预测结果
  - L4/results/tree_model_top_candidates.csv: Top 候选化合物
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(__file__).resolve().parent.parent / "logs" / "tree_model_cpi.log",
            mode="w",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
L4_RESULTS = PROJECT_ROOT / "L4" / "results"
L4_RESULTS_V10 = PROJECT_ROOT / "L4" / "results_v10_minibatch"
L3_RESULTS = PROJECT_ROOT / "L3" / "results"
L4_LOGS = PROJECT_ROOT / "L4" / "logs"
L4_LOGS.mkdir(parents=True, exist_ok=True)

# ============================================================
# 铁衰老靶标基因
# ============================================================
FERROAGING_GENES = sorted([
    "ABCC1", "ACSL4", "ACVR1B", "ALOX15", "ATF3", "ATG3", "BAP1", "BCL6",
    "BRD7", "CAVIN1", "CD74", "CD82", "CDO1", "COX7A1", "CTSB", "CXCL10",
    "DPEP1", "DPP4", "DUOX1", "DYRK1A", "E2F1", "E2F3", "EBF3", "EDN1",
    "EGR1", "EMP1", "EPHA2", "EPHA4", "ERN1", "FBXO31", "FOSL1", "GMFB",
    "HBP1", "HERPUD1", "HIF1A", "HMGB1", "HMOX1", "ICA1", "IFNG", "IGFBP7",
    "IL1B", "IL6", "IRF1", "IRF7", "IRF9", "KDM6B", "KEAP1", "KLF6",
    "LACTB", "LCN2", "LGMN", "LIFR", "LOX", "LPCAT3", "MAP3K14", "MAPK1",
    "MAPK14", "MCU", "MEN1", "MPO", "NLRP3", "NOX4", "NR1D1", "NR2F2",
    "NUAK2", "PADI4", "PDE4B", "PPP2R2B", "PRKD1", "PTBP1", "PTGS2", "RBM3",
    "RUNX3", "S100A8", "SAT1", "SETD7", "SLAMF8", "SLC1A5", "SMARCB1", "SMURF2",
    "SNCA", "SOCS1", "SOCS2", "SOD1", "SP1", "SPATA2", "TBX2", "TFRC",
    "TLR4", "TNFAIP1", "TNFAIP3", "TXNIP", "WNT5A", "WWTR1", "YAP1", "ZEB1",
])

# ECFP4 参数
ECFP4_NBITS = 2048
RDKIT_DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "MolMR", "TPSA",
    "NumHAcceptors", "NumHDonors", "NumRotatableBonds",
    "HeavyAtomCount", "NumAromaticRings", "NumAliphaticRings",
    "NumHeteroatoms", "NumValenceElectrons", "NHOHCount", "NOCount",
    "RingCount", "FractionCSP3", "BalabanJ",
]


# ============================================================
# 数据加载
# ============================================================

def load_cpi_data() -> pd.DataFrame:
    """加载 CPI 正样本数据"""
    path = L4_RESULTS / "experimental_actives_detail_cleaned.csv"
    if not path.exists():
        logger.error(f"CPI 文件不存在: {path}")
        sys.exit(1)
    df = pd.read_csv(path, low_memory=False)
    df = df[df["canonical_smiles"].notna()].copy()
    df = df[df["canonical_smiles"].astype(str).str.strip() != ""].copy()
    logger.info(f"CPI 数据: {len(df)} 条, {df['gene'].nunique()} 基因, "
                f"{df['canonical_smiles'].nunique()} 唯一SMILES")
    return df


def load_compound_features() -> tuple[np.ndarray, np.ndarray]:
    """加载预计算化合物特征"""
    path = L4_RESULTS_V10 / "compound_features_v31.npz"
    if not path.exists():
        logger.error(f"化合物特征文件不存在: {path}")
        sys.exit(1)
    data = np.load(path, allow_pickle=True)
    features = data["features"].astype(np.float32)
    smiles = data["smiles"]
    logger.info(f"化合物特征: {features.shape}, {len(smiles)} SMILES")
    return features, smiles


def load_protein_embeddings() -> dict[str, np.ndarray]:
    """加载 ESM-2 蛋白嵌入"""
    path = L4_RESULTS_V10 / "esm2_protein_embeddings.npz"
    if not path.exists():
        logger.error(f"蛋白嵌入文件不存在: {path}")
        sys.exit(1)
    data = np.load(path, allow_pickle=True)
    embeddings = {str(k): v.astype(np.float32) for k, v in data.items()}
    logger.info(f"蛋白嵌入: {len(embeddings)} 蛋白, dim={next(iter(embeddings.values())).shape[0]}")
    return embeddings


def load_tcm_pool() -> pd.DataFrame:
    """加载 TCM 候选化合物池"""
    path = L3_RESULTS / "tcm_compound_pool_tox_filtered_noleak.csv"
    if not path.exists():
        logger.error(f"TCM 池文件不存在: {path}")
        sys.exit(1)
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"TCM 候选池: {len(df)} 化合物")
    return df


# ============================================================
# 特征工程（在线计算 TCM 化合物特征，复用预计算统计量）
# ============================================================

def compute_ecfp4(smiles_list: list[str]) -> np.ndarray:
    """计算 ECFP4 指纹"""
    fps = np.zeros((len(smiles_list), ECFP4_NBITS), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        if not smi or pd.isna(smi):
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=ECFP4_NBITS)
        for bit in fp.GetOnBits():
            fps[i, bit] = 1.0
    return fps


def compute_maccs(smiles_list: list[str]) -> np.ndarray:
    """计算 MACCS 密钥"""
    fps = np.zeros((len(smiles_list), 167), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        if not smi or pd.isna(smi):
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        for bit in fp.GetOnBits():
            if bit < 167:
                fps[i, bit] = 1.0
    return fps


def compute_rdkit_descriptors(smiles_list: list[str]) -> np.ndarray:
    """计算 RDKit 分子描述符"""
    desc_funcs = {name: getattr(Descriptors, name) for name in RDKIT_DESCRIPTOR_NAMES}
    rows = []
    for smi in smiles_list:
        if not smi or pd.isna(smi):
            rows.append([np.nan] * len(RDKIT_DESCRIPTOR_NAMES))
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            rows.append([np.nan] * len(RDKIT_DESCRIPTOR_NAMES))
            continue
        vals = []
        for name in RDKIT_DESCRIPTOR_NAMES:
            try:
                vals.append(float(desc_funcs[name](mol)))
            except Exception:
                vals.append(np.nan)
        rows.append(vals)
    return np.array(rows, dtype=np.float32)


def build_compound_features_online(
    smiles_list: list[str],
    desc_mean: np.ndarray,
    desc_std: np.ndarray,
    desc_col_mean: np.ndarray,
) -> np.ndarray:
    """在线计算化合物特征（复现训练集标准化）"""
    ecfp4 = compute_ecfp4(smiles_list)
    maccs = compute_maccs(smiles_list)
    desc = compute_rdkit_descriptors(smiles_list)

    # 用训练集统计量标准化
    inds = np.where(np.isnan(desc))
    desc[inds] = np.take(desc_col_mean, inds[1])
    desc = np.nan_to_num(desc, nan=0.0, posinf=1e6, neginf=-1e6)
    desc = (desc - desc_mean) / (desc_std + 1e-8)

    features = np.hstack([ecfp4, maccs, desc]).astype(np.float32)
    return features


# ============================================================
# 数据集构建
# ============================================================

def build_dataset(
    cpi_df: pd.DataFrame,
    compound_features: np.ndarray,
    compound_smiles: np.ndarray,
    protein_embeddings: dict[str, np.ndarray],
    neg_ratio: int = 3,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构建训练数据集（正样本 + 负样本）

    Args:
        cpi_df: CPI 正样本数据
        compound_features: 预计算化合物特征矩阵
        compound_smiles: 化合物 SMILES 列表
        protein_embeddings: 蛋白 ESM-2 嵌入
        neg_ratio: 负样本:正样本比例
        random_seed: 随机种子

    Returns:
        X: 特征矩阵 [compound_feat | protein_feat]
        y: 标签 (1=正, 0=负)
        pair_compounds: 化合物 SMILES 列表
        pair_genes: 基因名列表
    """
    rng = np.random.RandomState(random_seed)

    # 构建 SMILES → 特征索引映射
    smiles_to_idx = {str(s): i for i, s in enumerate(compound_smiles)}

    # 获取 CPI 中的基因集合
    cpi_genes = sorted(cpi_df["gene"].unique())
    cpi_genes_in_emb = [g for g in cpi_genes if g in protein_embeddings]
    missing_genes = set(cpi_genes) - set(cpi_genes_in_emb)
    if missing_genes:
        logger.warning(f"缺失蛋白嵌入的基因: {missing_genes}")

    # 正样本
    pos_pairs = []
    for _, row in cpi_df.iterrows():
        smi = str(row["canonical_smiles"])
        gene = str(row["gene"])
        if smi in smiles_to_idx and gene in protein_embeddings:
            pos_pairs.append((smi, gene))

    logger.info(f"正样本: {len(pos_pairs)} 对 (CPI 原始 {len(cpi_df)} 条)")

    # 负样本：向量化随机采样（高效批量生成）
    # 用 (compound_idx, gene_idx) 整数对代替字符串对，大幅加速
    logger.info("  构建正样本索引集...")
    pos_idx_set = set()
    for smi, gene in pos_pairs:
        comp_idx = smiles_to_idx[smi]
        gene_idx = cpi_genes_in_emb.index(gene)
        pos_idx_set.add((comp_idx, gene_idx))

    n_compounds = len(compound_smiles)
    n_genes = len(cpi_genes_in_emb)
    n_neg_target = len(pos_pairs) * neg_ratio

    logger.info(f"  生成 {n_neg_target} 个负样本 (候选空间: {n_compounds}×{n_genes}={n_compounds*n_genes})...")
    neg_idx_set = set()
    batch_size = n_neg_target * 5  # 每次生成 5x 目标，过滤后取足够

    while len(neg_idx_set) < n_neg_target:
        # 批量生成随机索引
        batch_comp = rng.randint(0, n_compounds, size=batch_size)
        batch_gene = rng.randint(0, n_genes, size=batch_size)
        for ci, gi in zip(batch_comp, batch_gene):
            pair = (ci, gi)
            if pair not in pos_idx_set and pair not in neg_idx_set:
                neg_idx_set.add(pair)
                if len(neg_idx_set) >= n_neg_target:
                    break

    # 转换回 (smiles, gene) 对
    neg_pairs = []
    for ci, gi in neg_idx_set:
        smi = str(compound_smiles[ci])
        gene = cpi_genes_in_emb[gi]
        neg_pairs.append((smi, gene))

    if len(neg_pairs) < n_neg_target:
        logger.warning(f"负样本不足: 目标 {n_neg_target}, 实际 {len(neg_pairs)}")

    logger.info(f"负样本: {len(neg_pairs)} 对 (比例 1:{neg_ratio})")

    # 构建特征矩阵
    all_pairs = pos_pairs + neg_pairs
    n_pairs = len(all_pairs)
    comp_dim = compound_features.shape[1]
    prot_dim = next(iter(protein_embeddings.values())).shape[0]
    feat_dim = comp_dim + prot_dim

    X = np.zeros((n_pairs, feat_dim), dtype=np.float32)
    y = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs), dtype=np.int32)
    pair_compounds = []
    pair_genes = []

    for i, (smi, gene) in enumerate(all_pairs):
        comp_idx = smiles_to_idx[smi]
        X[i, :comp_dim] = compound_features[comp_idx]
        X[i, comp_dim:] = protein_embeddings[gene]
        pair_compounds.append(smi)
        pair_genes.append(gene)

    logger.info(f"数据集: {n_pairs} 样本, {feat_dim} 特征 "
                f"(comp={comp_dim} + prot={prot_dim}), "
                f"正样本比例={y.mean():.3f}")

    return X, y, np.array(pair_compounds), np.array(pair_genes)


# ============================================================
# 模型训练与评估
# ============================================================

def evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> dict:
    """训练并评估单个模型"""
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # 预测
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test).astype(float)

    y_pred = (y_prob >= 0.5).astype(int)

    # 指标
    try:
        auc_val = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc_val = 0.5
    aupr_val = average_precision_score(y_test, y_prob)

    # Precision@K
    precision_at_k = {}
    for k in [10, 20, 50, 100]:
        if k <= len(y_test):
            top_k_idx = np.argsort(y_prob)[-k:]
            precision_at_k[f"P@{k}"] = y_test[top_k_idx].mean()

    # EF@1% and EF@5%
    n_pos = y_test.sum()
    n_total = len(y_test)
    ef = {}
    for pct in [1, 5]:
        k = max(1, int(n_total * pct / 100))
        top_k_idx = np.argsort(y_prob)[-k:]
        found = y_test[top_k_idx].sum()
        expected = n_pos * pct / 100
        ef[f"EF@{pct}%"] = found / expected if expected > 0 else 0.0

    result = {
        "model": model_name,
        "AUC": auc_val,
        "AUPR": aupr_val,
        "train_time_s": train_time,
        **precision_at_k,
        **ef,
    }
    return result


def train_evaluate_models(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """5-fold 交叉验证训练多个树模型"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold + 1}/{n_folds}: train={len(X_train)}, test={len(X_test)}")
        logger.info(f"  正样本比例: train={y_train.mean():.3f}, test={y_test.mean():.3f}")

        # 1. Random Forest
        logger.info("  训练 Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_seed,
        )
        rf_result = evaluate_model(rf, X_train, y_train, X_test, y_test, "RandomForest")
        rf_result["fold"] = fold
        results.append(rf_result)
        logger.info(f"  RF: AUC={rf_result['AUC']:.4f}, AUPR={rf_result['AUPR']:.4f}")

        # 2. XGBoost
        try:
            import xgboost as xgb
            logger.info("  训练 XGBoost...")
            scale_pos_weight = (y_train == 0).sum() / max(y_train.sum(), 1)
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=random_seed,
                n_jobs=-1,
                verbosity=0,
            )
            xgb_result = evaluate_model(
                xgb_model, X_train, y_train, X_test, y_test, "XGBoost"
            )
            xgb_result["fold"] = fold
            results.append(xgb_result)
            logger.info(f"  XGB: AUC={xgb_result['AUC']:.4f}, AUPR={xgb_result['AUPR']:.4f}")
        except ImportError:
            logger.warning("  XGBoost 未安装，跳过")

        # 3. LightGBM
        try:
            import lightgbm as lgb
            logger.info("  训练 LightGBM...")
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                random_state=random_seed,
                n_jobs=-1,
                verbose=-1,
            )
            lgb_result = evaluate_model(
                lgb_model, X_train, y_train, X_test, y_test, "LightGBM"
            )
            lgb_result["fold"] = fold
            results.append(lgb_result)
            logger.info(f"  LGB: AUC={lgb_result['AUC']:.4f}, AUPR={lgb_result['AUPR']:.4f}")
        except ImportError:
            logger.warning("  LightGBM 未安装，跳过")

    return pd.DataFrame(results)


# ============================================================
# TCM 候选池预测
# ============================================================

def predict_tcm_pool(
    best_model,
    tcm_df: pd.DataFrame,
    compound_features: np.ndarray,
    compound_smiles: np.ndarray,
    desc_mean: np.ndarray,
    desc_std: np.ndarray,
    desc_col_mean: np.ndarray,
    protein_embeddings: dict[str, np.ndarray],
    cpi_genes: list[str],
    model_name: str,
) -> pd.DataFrame:
    """对 TCM 化合物池预测所有靶标得分"""
    # 在线计算 TCM 化合物特征
    tcm_smiles = tcm_df["SMILES_std"].astype(str).tolist()
    logger.info(f"计算 {len(tcm_smiles)} 个 TCM 化合物特征...")
    tcm_feats = build_compound_features_online(
        tcm_smiles, desc_mean, desc_std, desc_col_mean
    )

    # 获取 ESM-2 可用的基因
    genes_in_emb = [g for g in cpi_genes if g in protein_embeddings]
    logger.info(f"预测靶标: {len(genes_in_emb)} 个基因")

    comp_dim = tcm_feats.shape[1]
    prot_dim = next(iter(protein_embeddings.values())).shape[0]
    feat_dim = comp_dim + prot_dim

    predictions = []
    for i, (_, row) in enumerate(tcm_df.iterrows()):
        smi = str(row["SMILES_std"])
        mol_name = str(row.get("molecule_name", f"MOL_{i}"))
        mol_id = str(row.get("MOL_ID", f"MOL_{i}"))

        comp_feat = tcm_feats[i]

        for gene in genes_in_emb:
            feat = np.zeros(feat_dim, dtype=np.float32)
            feat[:comp_dim] = comp_feat
            feat[comp_dim:] = protein_embeddings[gene]

            if hasattr(best_model, "predict_proba"):
                score = float(best_model.predict_proba(feat.reshape(1, -1))[:, 1])
            else:
                score = float(best_model.predict(feat.reshape(1, -1)))

            predictions.append({
                "MOL_ID": mol_id,
                "molecule_name": mol_name,
                "SMILES": smi,
                "gene": gene,
                "score": score,
            })

        if (i + 1) % 100 == 0:
            logger.info(f"  预测进度: {i + 1}/{len(tcm_df)}")

    pred_df = pd.DataFrame(predictions)
    logger.info(f"预测完成: {len(pred_df)} 条 (化合物×基因)")

    return pred_df


# ============================================================
# 主流程
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("树模型 CPI 筛选：铁衰老-CIRI 化合物发现")
    logger.info("=" * 60)

    # ---- 1. 加载数据 ----
    logger.info("\n[1/5] 加载数据...")
    cpi_df = load_cpi_data()
    compound_features, compound_smiles = load_compound_features()
    protein_embeddings = load_protein_embeddings()
    tcm_df = load_tcm_pool()

    # 获取预计算统计量
    stats_path = L4_RESULTS_V10 / "compound_features_v31.npz"
    stats_data = np.load(stats_path, allow_pickle=True)
    desc_mean = stats_data["mean"].astype(np.float32)
    desc_std = stats_data["std"].astype(np.float32)
    desc_col_mean = stats_data["col_mean"].astype(np.float32)

    # ---- 2. 构建数据集 ----
    logger.info("\n[2/5] 构建训练数据集...")
    X, y, pair_compounds, pair_genes = build_dataset(
        cpi_df, compound_features, compound_smiles, protein_embeddings,
        neg_ratio=3, random_seed=42,
    )

    # ---- 3. 训练与评估 ----
    logger.info("\n[3/5] 5-fold 交叉验证训练...")
    results_df = train_evaluate_models(X, y, n_folds=5, random_seed=42)

    # 汇总结果
    logger.info("\n" + "=" * 60)
    logger.info("模型评估汇总 (5-fold CV mean ± std):")
    logger.info("=" * 60)
    summary = results_df.groupby("model").agg(["mean", "std"]).round(4)
    for model_name in summary.index:
        row = summary.loc[model_name]
        logger.info(f"\n  {model_name}:")
        for metric in ["AUC", "AUPR", "P@10", "P@20", "P@50", "EF@1%", "EF@5%"]:
            if metric in row.index:
                logger.info(f"    {metric}: {row[metric]['mean']:.4f} ± {row[metric]['std']:.4f}")

    # 保存评估结果
    results_path = L4_RESULTS / "tree_model_cpi_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\n评估结果已保存: {results_path}")

    # ---- 4. 全量训练最佳模型并预测 TCM 池 ----
    logger.info("\n[4/5] 全量训练最佳模型并预测 TCM 候选池...")

    # 选择最佳模型（按 AUPR）
    best_model_name = summary["AUPR"]["mean"].idxmax()
    best_aupr = summary.loc[best_model_name, "AUPR"]["mean"]
    logger.info(f"最佳模型: {best_model_name} (AUPR={best_aupr:.4f})")

    # 全量训练最佳模型
    cpi_genes = sorted(cpi_df["gene"].unique())
    cpi_genes_in_emb = [g for g in cpi_genes if g in protein_embeddings]

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
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
        )
    else:
        logger.error(f"未知模型: {best_model_name}")
        sys.exit(1)

    logger.info(f"全量训练 {best_model_name} (样本数: {len(X)})...")
    best_model.fit(X, y)

    # 预测 TCM 池
    pred_df = predict_tcm_pool(
        best_model, tcm_df,
        compound_features, compound_smiles,
        desc_mean, desc_std, desc_col_mean,
        protein_embeddings, cpi_genes_in_emb,
        best_model_name,
    )

    # 保存完整预测
    pred_path = L4_RESULTS / "tree_model_tcm_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"TCM 预测结果已保存: {pred_path}")

    # ---- 5. 生成 Top 候选 ----
    logger.info("\n[5/5] 生成 Top 候选化合物...")

    # 按化合物聚合（取每个化合物的最高得分和平均得分）
    comp_agg = pred_df.groupby(["MOL_ID", "molecule_name", "SMILES"]).agg(
        max_score=("score", "max"),
        mean_score=("score", "mean"),
        top_gene=("gene", lambda x: list(x)[np.argmax(list(pred_df.loc[x.index, "score"]))]),
        n_genes_above_50=("score", lambda x: (x >= 0.5).sum()),
        top_3_genes=("score", lambda x: "|".join(
            [f"{g}({s:.2f})" for g, s in sorted(
                zip(list(x), list(pred_df.loc[x.index, "score"])),
                key=lambda v: v[1], reverse=True
            )[:3]]
        )),
    ).reset_index()

    comp_agg = comp_agg.sort_values("max_score", ascending=False)

    # Top 50
    top50 = comp_agg.head(50)
    top_path = L4_RESULTS / "tree_model_top_candidates.csv"
    top50.to_csv(top_path, index=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"Top 20 候选化合物 (按 max_score 排序):")
    logger.info(f"{'='*60}")
    for i, row in enumerate(top50.head(20).itertuples(index=False), 1):
        logger.info(f"  {i:2d}. {row.molecule_name} | max={row.max_score:.4f} "
                    f"| mean={row.mean_score:.4f} "
                    f"| 高置信靶标(≥0.5): {row.n_genes_above_50} "
                    f"| Top靶标: {row.top_3_genes}")

    logger.info(f"\nTop 50 候选已保存: {top_path}")
    logger.info(f"\n任务完成!")


if __name__ == "__main__":
    main()