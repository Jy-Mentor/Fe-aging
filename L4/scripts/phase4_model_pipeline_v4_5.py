#!/usr/bin/env python3
"""
Phase 4 v4.5: CIRI铁衰老中药单体ML筛选 - 模型构建与预测（优化版）
=====================================================================
核心策略：
  1. 真实活性样本 + 相似性扩展构建训练标签，相似性扩展样本不参与评估。
  2. 多模型集成（RF/XGB/LR/SVM/KNN）+ 5 折 CV，仅保留 AUC>0.6 的模型。
  3. 集成策略：Borda 排序融合、概率几何平均、AUC 驱动加权（带小样本收缩）。
  4. 跨靶标统一 CPI 模型：靶标配体集合指纹 + 元素乘积/Tanimoto 交互特征。
  5. 方法感知加权融合 per-target 预测与 cross-target 分数。
  6. 仅对真实出现在 TCM 库中的活性样本计算 EF@1%/5%/10%。
  7. 多维度候选化合物排序并显式记录权重。

输出：
  L4/results_v4_5/model_performance_v4_5.csv
  L4/results_v4_5/tcm_predictions_full_v4_5.csv
  L4/results_v4_5/tcm_top_candidates_v4_5.csv
  L4/results_v4_5/enrichment_analysis_v4_5.csv
  L4/results_v4_5/training_metrics_v4_5.json
  L4/results_v4_5/phase4_report_v4_5.md
  L4/logs/phase4_model_pipeline_v4_5.log
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# 仅抑制 RDKit 大量底层报错输出，不全局关闭所有 Deprecation/FutureWarning
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

# 模块级过滤：只忽略 rdkit 自身已弃用 API 的警告，保留其他库的诊断信息
warnings.filterwarnings("ignore", category=DeprecationWarning, module="rdkit")
warnings.filterwarnings("ignore", category=FutureWarning, module="rdkit")
warnings.filterwarnings("ignore", message=".*MorganGenerator.*")

PROJECT_ROOT = Path(__file__).parent.parent.parent
L2_RESULTS = PROJECT_ROOT / "L2" / "results"
L3_RESULTS = PROJECT_ROOT / "L3" / "results"
L4_ROOT = PROJECT_ROOT / "L4"
L4_RESULTS = L4_ROOT / "results_v4_5"
L4_LOGS = L4_ROOT / "logs"

for d in [L4_RESULTS, L4_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = L4_LOGS / "phase4_model_pipeline_v4_5.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

CORE_GENES = [
    # ============================================================
    # 铁衰老（CIRI）原始差异表达基因集
    # 来源：Phase 1 铁衰老课题实验验证差异表达基因
    # ============================================================
    "ABCC1", "ACVR1B", "ACSL4", "ALOX15", "ATF3", "ATG3",
    "BAP1", "BCL6", "BRD7",
    "CAVIN1", "CD74", "CD82", "CDO1", "COX7A1", "CTSB", "CXCL10",
    "DPEP1", "DPP4", "DUOX1", "DYRK1A",
    "E2F1", "E2F3", "EBF3", "EDN1", "EGR1", "EMP1", "EPHA2", "EPHA4", "ERN1",
    "FBXO31", "FOSL1",
    "GMFB",
    "HBP1", "HERPUD1", "HIF1A", "HMGB1", "HMOX1",
    "ICA1", "IFNG", "IGFBP7", "IL1B", "IL6", "IRF1", "IRF7", "IRF9",
    "KDM6B", "KEAP1", "KLF6",
    "LACTB", "LCN2", "LGMN", "LIFR", "LOX", "LPCAT3",
    "MAP3K14", "MAPK1", "MAPK14", "MCU", "MEN1", "MPO",
    "NLRP3", "NOX4", "NR1D1", "NR2F2", "NUAK2",
    "PADI4", "PDE4B", "PPP2R2B", "PRKD1", "PTBP1", "PTGS2",
    "RBM3", "RUNX3",
    "S100A8", "SAT1", "SETD7", "SLAMF8", "SLC1A5", "SMARCB1", "SMURF2", "SNCA",
    "SOCS1", "SOCS2", "SOD1", "SP1", "SPATA2",
    "TBX2", "TFRC", "TLR4", "TNFAIP1", "TNFAIP3", "TXNIP",
    "WNT5A", "WWTR1",
    "YAP1",
    "ZEB1",
]
PRIORITY_TARGETS = []
ALL_TARGET_GENES = sorted(set(CORE_GENES + PRIORITY_TARGETS))
logger.info(f"铁衰老差异表达基因靶标总数: {len(ALL_TARGET_GENES)}")

# 有效的活性数据类型
ACTIVITY_TYPES = {"IC50", "Ki", "Kd"}
SIM_THRESHOLD = 0.7
NEG_SIM_THRESHOLD = 0.3

# 跨靶标模型每靶标采样上限，用于控制内存与训练时间
MAX_POS_PER_TARGET_CROSS = 2000
MAX_NEG_PER_TARGET_CROSS = 2000

# CV 训练集大小上限，超大目标随机下采样以控制训练时间
MAX_CV_TRAIN_SIZE = 4000

# xgboost 可选
try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception as exc:
    HAS_XGB = False
    logger.warning(f"XGBoost 未安装或无法导入: {exc}")


# ============================================================
# 工具函数
# ============================================================
def _find_column(candidates, columns):
    """按候选子串（忽略大小写）查找第一个匹配列。"""
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        for lower_col, orig_col in cols_lower.items():
            if cand.lower() in lower_col:
                return orig_col
    return None


def _get_classifier_step_name(model):
    """返回 sklearn Pipeline 中具有 predict_proba 的分类器步骤名称。"""
    if not hasattr(model, "named_steps"):
        return None
    for name, step in model.named_steps.items():
        if hasattr(step, "predict_proba"):
            return name
    return None


def _compute_ecfp4(smiles_iter, n_bits=2048):
    """为 SMILES 列表计算 ECFP4 指纹，返回 (valid_mask, fingerprints)。"""
    fps = []
    valid = []
    for smi in smiles_iter:
        mol = None
        try:
            if pd.notna(smi):
                mol = Chem.MolFromSmiles(str(smi))
        except Exception as exc:
            logger.debug(f"SMILES 解析异常: {smi}, 错误: {exc}")
            mol = None
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.float32))
            valid.append(False)
            continue
        try:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
            arr = np.zeros(n_bits, dtype=np.float32)
            arr[list(fp.GetOnBits())] = 1.0
            fps.append(arr)
            valid.append(True)
        except Exception as exc:
            logger.debug(f"指纹生成异常: {smi}, 错误: {exc}")
            fps.append(np.zeros(n_bits, dtype=np.float32))
            valid.append(False)
    return np.array(valid, dtype=bool), np.array(fps, dtype=np.float32)


def _vectorized_tanimoto(ref_fps, query_fps):
    """计算 (n_query, n_ref) 的 Tanimoto 相似度矩阵。"""
    ref_bin = ref_fps > 0.5
    q_bin = query_fps > 0.5
    inter = q_bin.astype(np.float32) @ ref_bin.astype(np.float32).T
    ref_sum = ref_bin.sum(axis=1).astype(np.float32)
    q_sum = q_bin.sum(axis=1).astype(np.float32)
    union = q_sum[:, None] + ref_sum[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        tani = np.where(union > 0, inter / union, 0.0)
    return np.nan_to_num(tani, nan=0.0)


def _compute_scaffold(smiles):
    """计算 SMILES 的 Murcko 骨架；失败时返回原 SMILES 字符串。"""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return str(smiles)
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        logger.exception("捕获到异常并继续执行（原 except 'Exception' 静默吞掉）")
        return str(smiles)

# ============================================================
# 数据加载
# ============================================================
def load_compound_data():
    """加载 TCM 化合物池与预计算指纹。"""
    logger.info("=" * 60)
    logger.info("[1] 加载 TCM 化合物特征")
    logger.info("=" * 60)
    t0 = time.time()

    cp = pd.read_csv(L3_RESULTS / "tcm_compound_pool_filtered.csv")
    ecfp4 = np.load(L3_RESULTS / "ecfp4_fingerprints.npy").astype(np.float32)
    n_bits = ecfp4.shape[1]

    # 输入真实性校验
    required_cols = ["MOL_ID", "molecule_name", "SMILES_std"]
    missing_cols = [c for c in required_cols if c not in cp.columns]
    if missing_cols:
        raise ValueError(f"TCM化合物池缺失必需列: {missing_cols}")
    if len(cp) == 0:
        raise ValueError("TCM化合物池为空")
    if ecfp4.shape[0] != len(cp):
        raise ValueError(f"ECFP4行数({ecfp4.shape[0]}) != 化合物数({len(cp)})")
    if np.isnan(ecfp4).any() or np.isinf(ecfp4).any():
        raise ValueError("ECFP4指纹含NaN/Inf")
    zero_fp = (ecfp4.sum(axis=1) == 0).sum()
    if zero_fp > 0:
        logger.warning(f"  ECFP4含 {zero_fp} 条全零指纹")

    logger.info(f"  TCM化合物池: {len(cp)} 个, ECFP4维度: {n_bits}")
    logger.info(f"  耗时: {time.time() - t0:.2f}s")
    return {
        "mol_ids": cp["MOL_ID"].values,
        "names": cp["molecule_name"].values,
        "smiles": cp["SMILES_std"].values,
        "ecfp4": ecfp4,
        "n_bits": n_bits,
        "n_compounds": len(cp),
    }


def load_protein_data():
    """加载并合并蛋白 AAC 与 PseAAC 特征；缺失 PseAAC 用 0 填充。"""
    logger.info("=" * 60)
    logger.info("[2] 加载蛋白质特征")
    logger.info("=" * 60)
    t0 = time.time()

    prot = pd.read_csv(L2_RESULTS / "target_protein_features.csv")
    aac = pd.read_csv(L2_RESULTS / "protein_descriptors.csv")
    pseaac = pd.read_csv(L2_RESULTS / "protein_pseaac.csv")

    if "gene_symbol" not in prot.columns:
        raise ValueError("蛋白特征表缺失 gene_symbol 列")

    # 防御性去重与缺失清理
    prot = prot.dropna(subset=["gene_symbol"]).drop_duplicates(subset=["gene_symbol"], keep="first")
    aac = aac.dropna(subset=["gene_symbol"]).drop_duplicates(subset=["gene_symbol"], keep="first")
    pseaac = pseaac.dropna(subset=["gene_symbol"]).drop_duplicates(subset=["gene_symbol"], keep="first")

    if len(aac) != len(prot):
        raise ValueError(f"AAC行数({len(aac)}) != 蛋白数({len(prot)})")

    aac_cols = [c for c in aac.columns if c.startswith("AAC_")]
    pseaac_cols = [c for c in pseaac.columns if c.startswith("PseAAC_")]
    if not aac_cols:
        raise ValueError("AAC描述符为空")

    # 以 target_protein_features 为基准，合并 AAC 与 PseAAC
    df = prot[["gene_symbol"]].copy()
    df = df.merge(aac[["gene_symbol"] + aac_cols], on="gene_symbol", how="left")
    df = df.merge(pseaac[["gene_symbol"] + pseaac_cols], on="gene_symbol", how="left")

    missing_pseaac = df[pseaac_cols].isna().any().any() if pseaac_cols else False
    if missing_pseaac:
        n_missing = df[pseaac_cols].isna().any(axis=1).sum()
        logger.warning(f"  {n_missing} 个蛋白缺少 PseAAC，将用 0 填充")
        df[pseaac_cols] = df[pseaac_cols].fillna(0.0)

    features = df[aac_cols + pseaac_cols].values.astype(np.float32)
    gene_symbols = df["gene_symbol"].values
    gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}

    logger.info(f"  蛋白质: {len(gene_symbols)} 个, 合并特征维度: {features.shape[1]}")
    logger.info(f"  耗时: {time.time() - t0:.2f}s")
    return {
        "gene_symbols": gene_symbols,
        "gene_to_idx": gene_to_idx,
        "features": features,
        "n_features": features.shape[1],
    }


def load_activity_data(valid_genes):
    """读取 L4/results/ 下所有含基因列的活性 CSV，清洗并标记真实正样本。"""
    logger.info("=" * 60)
    logger.info("[3] 加载实验活性数据")
    logger.info("=" * 60)
    t0 = time.time()

    result_dir = L4_ROOT / "results"
    all_csv = sorted([p for p in result_dir.glob("*.csv") if p.is_file()])
    # 仅读取真实活性数据文件，避免混入预测结果/性能表/汇总表
    allowed_patterns = ("active", "actives", "experimental")
    excluded_patterns = ("prediction", "performance", "summary", "candidate", "report", "top")
    candidate_files = []
    for p in all_csv:
        stem_lower = p.stem.lower()
        if any(pat in stem_lower for pat in allowed_patterns) and not any(
            pat in stem_lower for pat in excluded_patterns
        ):
            candidate_files.append(p)

    frames = []
    for path in candidate_files:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            logger.warning(f"  跳过无法读取的文件 {path.name}: {exc}")
            continue
        gene_col = _find_column(["gene"], df.columns)
        if gene_col is None:
            logger.debug(f"  跳过无基因列文件: {path.name}")
            continue
        # 保留关键列，避免列名冲突
        smiles_col = _find_column(["canonical_smiles", "smiles"], df.columns)
        if smiles_col is None:
            logger.warning(f"  跳过无SMILES列文件: {path.name}（仅作name-only参考，不用于训练）")
            continue
        value_col = _find_column(["standard_value_nM", "value_nM"], df.columns)
        type_col = _find_column(["standard_type"], df.columns)
        relation_col = _find_column(["standard_relation"], df.columns)

        keep_cols = set()
        rename = {}
        if gene_col:
            keep_cols.add(gene_col)
            rename[gene_col] = "gene"
        if smiles_col:
            keep_cols.add(smiles_col)
            rename[smiles_col] = "canonical_smiles"
        if value_col:
            keep_cols.add(value_col)
            rename[value_col] = "standard_value_nM"
        if type_col:
            keep_cols.add(type_col)
            rename[type_col] = "standard_type"
        if relation_col:
            keep_cols.add(relation_col)
            rename[relation_col] = "standard_relation"
        # 若原表存在 source 列则保留，否则后续用文件名填充
        if "source" in df.columns:
            keep_cols.add("source")

        sub = df[list(keep_cols)].copy()
        sub = sub.rename(columns=rename)
        if "source" not in sub.columns:
            sub["source"] = path.stem
        frames.append(sub)

    if not frames:
        raise RuntimeError("未找到任何含基因列的活性数据文件")

    raw = pd.concat(frames, ignore_index=True)
    logger.info(f"  原始活性记录: {len(raw)} 条，来源文件数: {len(frames)}")

    # 仅保留蛋白表中的基因
    raw = raw[raw["gene"].isin(valid_genes)].copy()
    logger.info(f"  保留目标基因内记录: {len(raw)} 条")

    # 若存在 standard_relation，仅保留 '='
    if "standard_relation" in raw.columns:
        before = len(raw)
        raw = raw[raw["standard_relation"].fillna("=") == "="].copy()
        dropped = before - len(raw)
        if dropped:
            logger.info(f"  因 standard_relation != '=' 剔除 {dropped} 条")

    # 解析 SMILES
    valid_mask, _ = _compute_ecfp4(raw["canonical_smiles"].values)
    n_invalid = (~valid_mask).sum()
    if n_invalid:
        logger.warning(f"  无效 SMILES: {n_invalid} 条，已剔除")
    raw = raw[valid_mask].copy()

    # 标准化 canonical_smiles
    raw["canonical_smiles"] = raw["canonical_smiles"].astype(str)

    # 数值化活性值
    if "standard_value_nM" in raw.columns:
        raw["standard_value_nM"] = pd.to_numeric(raw["standard_value_nM"], errors="coerce")

    # 标记真实正样本：必须同时满足 (1) 活性类型属于 {IC50, Ki, Kd}；
    # (2) 有有效数值；(3) 数值 <= 10000 nM。缺少任一条件均不视为正样本，
    # 避免 EC50/%/Activity 等类型或无数值记录被错误标记。
    def _is_positive(row):
        val = row.get("standard_value_nM")
        typ = row.get("standard_type")
        if pd.isna(val) or pd.isna(typ):
            return False
        if typ not in ACTIVITY_TYPES:
            return False
        try:
            return float(val) <= 10000.0
        except (ValueError, TypeError):
            return False

    raw["is_positive"] = raw.apply(_is_positive, axis=1)
    n_positive = int(raw["is_positive"].sum())
    logger.info(f"  严格正样本标记: {n_positive}/{len(raw)} 条满足 IC50/Ki/Kd + <=10000 nM")

    # 仅保留正样本并去重
    positives = raw[raw["is_positive"]].copy()
    before_dedup = len(positives)
    positives = positives.drop_duplicates(subset=["gene", "canonical_smiles"], keep="first")
    logger.info(
        f"  真实正样本: 去重前 {before_dedup}，去重后 {len(positives)}，"
        f"覆盖基因数: {positives['gene'].nunique()}"
    )

    # 计算所有正样本指纹并缓存
    unique_smiles = positives["canonical_smiles"].unique()
    _, unique_fps = _compute_ecfp4(unique_smiles)
    smile_to_fp = dict(zip(unique_smiles, unique_fps, strict=False))
    positives["fp"] = positives["canonical_smiles"].map(smile_to_fp).apply(lambda x: x.tolist())
    positives["fp"] = positives["fp"].apply(np.array)

    logger.info(f"  耗时: {time.time() - t0:.2f}s")
    return positives


# ============================================================
# 标签构建
# ============================================================
def build_per_target_datasets(compound_data, protein_data, active_df):
    """为每个有真实活性的靶标构建训练集（真实正/负 + 相似性扩展正样本）。"""
    logger.info("=" * 60)
    logger.info("[4] 构建 per-target 训练数据集")
    logger.info("=" * 60)
    t0 = time.time()

    ecfp4 = compound_data["ecfp4"]
    rng = np.random.default_rng(RANDOM_SEED)

    target_genes = sorted(active_df["gene"].unique())
    datasets = {}

    for gene in target_genes:
        pos_df = active_df[active_df["gene"] == gene].reset_index(drop=True)
        pos_fps = np.vstack(pos_df["fp"].values)
        n_real_pos = len(pos_fps)
        pos_smiles = set(pos_df["canonical_smiles"].unique())

        # 相似性扩展：与真实正样本最大 Tanimoto > 0.7 的 TCM 化合物
        sim_mat = _vectorized_tanimoto(pos_fps, ecfp4)
        max_sim = sim_mat.max(axis=1)
        sim_mask = max_sim > SIM_THRESHOLD
        sim_idx = np.where(sim_mask)[0]
        sim_fps = ecfp4[sim_idx]
        n_sim = len(sim_idx)

        # 真实负样本：从其它靶标真实正样本中采样 5-10 个，并补充低相似 TCM 化合物
        other_df = active_df[active_df["gene"] != gene]
        n_other = len(other_df)
        neg_fps_list = []

        neg_smiles_list = []

        if n_other > 0:
            # 修复：当 n_other < 5 时，n_sample_other 不应超过 n_other，否则 sample() 抛 ValueError
            n_sample_other = min(10, n_other)
            sampled_other = other_df.sample(n=n_sample_other, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
            # 排除对当前靶标也为活性的分子
            sampled_other = sampled_other[~sampled_other["canonical_smiles"].isin(pos_smiles)]
            if len(sampled_other):
                neg_fps_list.append(np.vstack(sampled_other["fp"].values))
                neg_smiles_list.extend(sampled_other["canonical_smiles"].astype(str).tolist())

        # 目标负样本总量：至少 100，至多 2000，且不超过真实正样本数的两倍
        target_neg_count = min(max(100, min(n_real_pos * 2, 2000)), 2000)
        current_neg = sum(len(x) for x in neg_fps_list)
        if current_neg < target_neg_count:
            need = target_neg_count - current_neg
            # 候选：低相似且非相似性正样本的 TCM 化合物
            low_sim_mask = max_sim < NEG_SIM_THRESHOLD
            candidate_idx = np.where(low_sim_mask & (~sim_mask))[0]
            if len(candidate_idx) < need:
                # 若低相似不足，从全部非正样本中随机补
                all_non_pos = np.where(~sim_mask)[0]
                candidate_idx = all_non_pos
            if len(candidate_idx):
                chosen_idx = rng.choice(candidate_idx, size=min(need, len(candidate_idx)), replace=False)
                neg_fps_list.append(ecfp4[chosen_idx])
                neg_smiles_list.extend(compound_data["smiles"][chosen_idx].astype(str).tolist())

        neg_fps = np.vstack(neg_fps_list) if neg_fps_list else np.empty((0, compound_data["n_bits"]), dtype=np.float32)
        n_neg = len(neg_fps)

        # 构造特征：仅使用化合物指纹（避免蛋白特征作为靶标标识导致 shortcut learning）
        def _make_features(cf):
            return cf.astype(np.float32)

        X_pos = _make_features(pos_fps)
        X_neg = _make_features(neg_fps)
        X_sim = _make_features(sim_fps)

        X = np.vstack([X_pos, X_neg, X_sim])
        y = np.concatenate([
            np.ones(n_real_pos, dtype=int),
            np.zeros(n_neg, dtype=int),
            np.ones(n_sim, dtype=int),
        ])
        weights = np.concatenate([
            np.ones(n_real_pos, dtype=np.float32),
            np.ones(n_neg, dtype=np.float32),
            0.7 * np.ones(n_sim, dtype=np.float32),
        ])
        method = np.array(
            ["experimental"] * n_real_pos
            + ["experimental"] * n_neg
            + ["similarity"] * n_sim
        )

        # 真实样本索引用于 CV 评估；相似性样本只在训练集中出现
        real_idx = np.arange(0, n_real_pos + n_neg, dtype=int)
        sim_idx = np.arange(n_real_pos + n_neg, len(X), dtype=int)
        real_smiles = np.array(
            pos_df["canonical_smiles"].astype(str).tolist() + neg_smiles_list,
            dtype=object,
        )

        datasets[gene] = {
            "X": X,
            "y": y,
            "weights": weights,
            "method": method,
            "real_idx": real_idx,
            "sim_idx": sim_idx,
            "real_smiles": real_smiles,
            "n_real_pos": n_real_pos,
            "n_sim": n_sim,
            "n_neg": n_neg,
            "pos_fps": pos_fps,
            "pos_smiles": pos_smiles,
            "max_sim_to_pos": max_sim,
        }
        logger.info(
            f"  {gene}: 真实正样本={n_real_pos}, 相似扩展={n_sim}, 真实负样本={n_neg}"
        )

    logger.info(f"  构建完成靶标数: {len(datasets)}，耗时: {time.time() - t0:.2f}s")
    return datasets


# ============================================================
# 模型训练
# ============================================================
def _get_model_builders(n_neg):
    """返回模型名到构造器/超参的映射。"""
    builders = {}

    builders["RF"] = lambda: make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    )

    if HAS_XGB:
        builders["XGB"] = lambda scale_pos_weight=1.0: make_pipeline(
            StandardScaler(),
            XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
        )

    builders["LR"] = lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
    )

    builders["SVM"] = lambda: make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(
            LinearSVC(
                class_weight="balanced",
                max_iter=1000,
                tol=1e-3,
                dual=False,
                random_state=RANDOM_SEED,
            ),
            cv=2,
        ),
    )

    n_neighbors = min(5, max(1, n_neg))
    builders["KNN"] = lambda: make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", n_jobs=-1),
    )

    return builders


def _cv_evaluate(model, X, y, weights, real_idx, real_smiles, n_splits=5, use_scaffold_cv=True):
    """在真实样本上做 5 折 CV；相似性扩展样本不参与 CV，避免验证集信息泄漏。

    若提供 real_smiles 且 use_scaffold_cv=True，则使用 Murcko 骨架做
    StratifiedGroupKFold，防止相同骨架同时出现在训练/验证集。
    返回 (mean_auc, std_auc, mean_aupr, std_aupr)。
    """
    y_real = y[real_idx]
    classes = np.unique(y_real)
    if len(classes) < 2:
        logger.warning("  真实样本仅含单一类别，无法计算 AUC")
        return None, None, None, None

    n_splits = min(n_splits, int(min(np.bincount(y_real))))
    if n_splits < 2:
        logger.warning("  真实样本类别样本量不足，无法做 CV")
        return None, None, None, None

    # 选择 CV 策略：优先 StratifiedGroupKFold（按骨架分组）
    # 若骨架分组导致某折仅含单一类别，则回退到 StratifiedKFold
    def _make_split_iter():
        if use_scaffold_cv and real_smiles is not None and len(real_smiles) == len(real_idx):
            try:
                scaffolds = np.array([_compute_scaffold(s) for s in real_smiles])
                _, group_ids = np.unique(scaffolds, return_inverse=True)
                n_groups = len(np.unique(group_ids))
                if n_groups >= n_splits:
                    splitter = StratifiedGroupKFold(
                        n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED
                    )
                    return splitter.split(X[real_idx], y_real, groups=group_ids), "scaffold"
            except Exception as exc:
                logger.debug(f"  骨架分组失败，回退到 StratifiedKFold: {exc}")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        return splitter.split(X[real_idx], y_real), "stratified"

    split_iter, cv_strategy = _make_split_iter()

    # 预检查：确保每折验证集均含两类样本
    valid_split = True
    tmp_iter, cv_strategy = _make_split_iter()
    for _, val in tmp_iter:
        if len(np.unique(y_real[val])) < 2:
            valid_split = False
            break
    if not valid_split:
        logger.debug("  scaffold CV 某折仅含单一类别，回退到 StratifiedKFold")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        split_iter = splitter.split(X[real_idx], y_real)
        cv_strategy = "stratified"

    logger.debug(f"  CV 策略: {cv_strategy}")

    aucs = []
    auprs = []

    for train, val in split_iter:
        # CV 中只用真实训练样本，不用相似性扩展样本，防止验证集信息泄漏
        train_idx = real_idx[train]
        val_idx = real_idx[val]

        # 超大训练集下采样，控制 CV 时间，同时保持类别比例
        if len(train_idx) > MAX_CV_TRAIN_SIZE:
            y_train = y[train_idx]
            pos_idx = train_idx[y_train == 1]
            neg_idx = train_idx[y_train == 0]
            n_pos_max = max(1, int(MAX_CV_TRAIN_SIZE * len(pos_idx) / len(train_idx)))
            n_neg_max = MAX_CV_TRAIN_SIZE - n_pos_max
            rng = np.random.default_rng(RANDOM_SEED)
            if len(pos_idx) > n_pos_max:
                pos_idx = rng.choice(pos_idx, size=n_pos_max, replace=False)
            if len(neg_idx) > n_neg_max:
                neg_idx = rng.choice(neg_idx, size=n_neg_max, replace=False)
            train_idx = np.concatenate([pos_idx, neg_idx])

        m = clone(model)
        step_name = _get_classifier_step_name(m)
        try:
            if step_name is not None and weights is not None:
                m.fit(
                    X[train_idx],
                    y[train_idx],
                    **{f"{step_name}__sample_weight": weights[train_idx]},
                )
            else:
                m.fit(X[train_idx], y[train_idx])
        except TypeError:
            # 部分模型不接受 sample_weight
            m.fit(X[train_idx], y[train_idx])
        except Exception as exc:
            # 未知训练异常必须抛出，避免静默吞掉导致结果不可靠
            logger.exception(f"  CV 训练异常: {exc}")
            raise

        try:
            prob = m.predict_proba(X[val_idx])[:, 1]
            aucs.append(roc_auc_score(y[val_idx], prob))
            auprs.append(average_precision_score(y[val_idx], prob))
        except Exception as exc:
            # 评估异常同样不可静默忽略，必须抛出以暴露数据/模型问题
            logger.exception(f"  CV 评估异常: {exc}")
            raise

    if not aucs:
        return None, None, None, None
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(auprs)), float(np.std(auprs))


def _flag_suspicious_auc(cv_aucs, gene):
    """对 AUC 过高（>0.98）或过低（<0.6）的模型发出警告，提示可能存在泄漏或任务过简。"""
    if not cv_aucs:
        return
    for name, auc in cv_aucs.items():
        if auc > 0.98:
            logger.warning(
                f"  {gene}-{name} CV AUC={auc:.4f} 接近 1.0，可能存在数据泄漏、"
                f"负样本过易区分或评估设计过于乐观，建议在报告中说明"
            )
        elif auc < 0.6:
            logger.warning(
                f"  {gene}-{name} CV AUC={auc:.4f} 低于 0.6，模型区分能力弱"
            )


def train_per_target_models(datasets):
    """训练 per-target 多模型，仅保留 CV AUC > 0.6 的模型。"""
    logger.info("=" * 60)
    logger.info("[5] per-target 多模型训练与 CV 评估")
    logger.info("=" * 60)
    t0 = time.time()

    per_target = {}
    all_metrics = []

    for gene, ds in datasets.items():
        n_real_pos = ds["n_real_pos"]
        n_sim = ds["n_sim"]
        n_neg = ds["n_neg"]

        if n_real_pos < 5 or (n_real_pos + n_sim) < 10:
            logger.info(
                f"  {gene}: 样本不足（真实正样本={n_real_pos}, 总正样本={n_real_pos + n_sim}），"
                f"将使用相似性方法"
            )
            per_target[gene] = {
                "models": {},
                "cv_auc": {},
                "cv_aupr": {},
                "status": "SIMILARITY_FALLBACK",
                "n_real_pos": n_real_pos,
                "n_sim": n_sim,
                "n_neg": n_neg,
            }
            all_metrics.append({
                "gene": gene,
                "model": "N/A",
                "cv_auc": np.nan,
                "cv_aupr": np.nan,
                "status": "SIMILARITY_FALLBACK",
                "n_real_pos": n_real_pos,
                "n_sim": n_sim,
                "n_neg": n_neg,
            })
            continue

        builders = _get_model_builders(n_neg)
        kept_models = {}
        cv_aucs = {}
        cv_auprs = {}

        # XGB scale_pos_weight
        n_pos_total = n_real_pos + n_sim
        scale_pos_weight = max(1.0, n_neg / max(1, n_pos_total))

        for name, builder in builders.items():
            try:
                model = builder(scale_pos_weight=scale_pos_weight) if name == "XGB" else builder()
                auc, auc_std, aupr, aupr_std = _cv_evaluate(
                    model,
                    ds["X"],
                    ds["y"],
                    ds["weights"],
                    ds["real_idx"],
                    ds.get("real_smiles"),
                )
                if auc is None:
                    logger.info(f"  {gene}-{name}: CV 评估失败")
                    continue
                logger.info(
                    f"  {gene}-{name}: CV AUC={auc:.4f}±{auc_std:.4f}, "
                    f"AUPR={aupr:.4f}±{aupr_std:.4f}"
                )
                if auc > 0.6:
                    # 在全部数据上训练最终模型（此时可加入相似性扩展样本）
                    final_model = clone(model)
                    step_name = _get_classifier_step_name(final_model)
                    try:
                        if step_name is not None:
                            final_model.fit(
                                ds["X"],
                                ds["y"],
                                **{f"{step_name}__sample_weight": ds["weights"]},
                            )
                        else:
                            final_model.fit(ds["X"], ds["y"])
                    except TypeError:
                        final_model.fit(ds["X"], ds["y"])
                    kept_models[name] = final_model
                    cv_aucs[name] = auc
                    cv_auprs[name] = aupr
                    all_metrics.append({
                        "gene": gene,
                        "model": name,
                        "cv_auc": round(auc, 4),
                        "cv_auc_std": round(auc_std, 4),
                        "cv_aupr": round(aupr, 4),
                        "cv_aupr_std": round(aupr_std, 4),
                        "status": "KEPT",
                        "n_real_pos": n_real_pos,
                        "n_sim": n_sim,
                        "n_neg": n_neg,
                    })
                else:
                    all_metrics.append({
                        "gene": gene,
                        "model": name,
                        "cv_auc": round(auc, 4),
                        "cv_auc_std": round(auc_std, 4),
                        "cv_aupr": round(aupr, 4),
                        "cv_aupr_std": round(aupr_std, 4),
                        "status": "REJECTED",
                        "n_real_pos": n_real_pos,
                        "n_sim": n_sim,
                        "n_neg": n_neg,
                    })
            except (ValueError, TypeError) as exc:
                # 已知可能因样本过少或 sample_weight 不支持导致，记录后继续
                logger.warning(f"  {gene}-{name} 训练/评估失败: {exc}")
                all_metrics.append({
                    "gene": gene,
                    "model": name,
                    "cv_auc": np.nan,
                    "cv_aupr": np.nan,
                    "status": f"ERROR: {exc}",
                    "n_real_pos": n_real_pos,
                    "n_sim": n_sim,
                    "n_neg": n_neg,
                })
            except Exception:
                # 未知异常：记录后抛出，避免静默吞掉
                logger.exception(f"  {gene}-{name} 发生未预期异常")
                raise

        _flag_suspicious_auc(cv_aucs, gene)

        status = "TRAINED" if kept_models else "SIMILARITY_FALLBACK"
        per_target[gene] = {
            "models": kept_models,
            "cv_auc": cv_aucs,
            "cv_aupr": cv_auprs,
            "status": status,
            "n_real_pos": n_real_pos,
            "n_sim": n_sim,
            "n_neg": n_neg,
        }
        if kept_models:
            logger.info(
                f"  {gene}: 保留模型 {list(kept_models.keys())}, 平均 AUC="
                f"{np.mean(list(cv_aucs.values())):.4f}"
            )
        else:
            logger.info(f"  {gene}: 无模型通过 CV 阈值，回退到相似性方法")

    logger.info(f"  训练完成，耗时: {time.time() - t0:.2f}s")
    return per_target, all_metrics


# ============================================================
# 跨靶标统一 CPI 模型
# ============================================================
def _build_target_fingerprint(pos_fps):
    """根据真实正样本指纹构建靶标配体集合指纹（bit 在 >=1/3 正样本中出现则置 1）。"""
    n_pos = len(pos_fps)
    if n_pos == 0:
        return np.zeros(pos_fps.shape[1], dtype=np.float32)
    counts = pos_fps.sum(axis=0)
    threshold = n_pos / 3.0
    fp = (counts >= threshold).astype(np.float32)
    return fp


def _build_pair_features(compound_fps, target_fp):
    """构建 (化合物, 靶标) 交互特征（化合物指纹 + 靶标配体集指纹 + 元素乘积 + Tanimoto）。

    不拼接原始 AAC/PseAAC 蛋白描述符，避免跨靶标时 target identity 成为 shortcut。
    """
    n = len(compound_fps)
    target_fp_tile = np.tile(target_fp, (n, 1))
    elem = compound_fps * target_fp_tile
    tani = _vectorized_tanimoto(target_fp[None, :], compound_fps).flatten()
    return np.hstack([compound_fps, target_fp_tile, elem, tani[:, None]]).astype(np.float32)


def train_cross_target_model(datasets, compound_data, protein_data):
    """训练跨靶标统一 XGBClassifier。"""
    logger.info("=" * 60)
    logger.info("[6] 训练跨靶标统一 CPI 模型")
    logger.info("=" * 60)
    t0 = time.time()

    if not HAS_XGB:
        logger.warning("  XGBoost 不可用，跳过跨靶标模型")
        return None

    X_list = []
    y_list = []
    rng = np.random.default_rng(RANDOM_SEED)

    for gene, ds in datasets.items():
        n_real_pos = ds["n_real_pos"]
        if n_real_pos == 0:
            continue

        pos_fps = ds["pos_fps"]
        # 真实负样本来自 per-target 数据集中实验负样本部分
        neg_fps = ds["X"][ds["real_idx"]][ds["y"][ds["real_idx"]] == 0][:, : compound_data["n_bits"]]

        # 采样控制规模
        if len(pos_fps) > MAX_POS_PER_TARGET_CROSS:
            pos_idx = rng.choice(len(pos_fps), size=MAX_POS_PER_TARGET_CROSS, replace=False)
            pos_fps = pos_fps[pos_idx]
        if len(neg_fps) > MAX_NEG_PER_TARGET_CROSS:
            neg_idx = rng.choice(len(neg_fps), size=MAX_NEG_PER_TARGET_CROSS, replace=False)
            neg_fps = neg_fps[neg_idx]

        target_fp = _build_target_fingerprint(ds["pos_fps"])

        X_pos = _build_pair_features(pos_fps, target_fp)
        X_neg = _build_pair_features(neg_fps, target_fp)

        X_list.append(X_pos)
        X_list.append(X_neg)
        y_list.append(np.ones(len(X_pos), dtype=int))
        y_list.append(np.zeros(len(X_neg), dtype=int))

    if not X_list:
        logger.warning("  无可用于跨靶标训练的数据")
        return None

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    logger.info(f"  跨靶标训练样本: {len(X)}，特征维度: {X.shape[1]}，正样本: {int(y.sum())}")

    try:
        scale_pos_weight = max(1.0, (y == 0).sum() / max(1, (y == 1).sum()))
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        model.fit(X, y)
        logger.info(f"  跨靶标模型训练完成，耗时: {time.time() - t0:.2f}s")
        return model
    except Exception as exc:
        logger.error(f"  跨靶标模型训练失败: {exc}")
        traceback.print_exc()
        return None


def predict_cross_target_scores(cross_model, datasets, compound_data, protein_data, all_genes):
    """为所有有真实活性的靶标配对所有 TCM 化合物生成 cross_target_score。"""
    scores = {}
    if cross_model is None:
        return scores

    logger.info("[7] 跨靶标模型预测")
    t0 = time.time()
    for gene in all_genes:
        ds = datasets.get(gene)
        if ds is None or ds["n_real_pos"] == 0:
            continue
        target_fp = _build_target_fingerprint(ds["pos_fps"])
        X_pred = _build_pair_features(compound_data["ecfp4"], target_fp)
        try:
            prob = cross_model.predict_proba(X_pred)[:, 1]
            scores[gene] = prob.astype(np.float32)
        except Exception as exc:
            logger.warning(f"  {gene} cross-target 预测失败: {exc}")
    logger.info(f"  跨靶标预测完成，覆盖 {len(scores)} 个靶标，耗时: {time.time() - t0:.2f}s")
    return scores


# ============================================================
# per-target 预测与融合
# ============================================================
def predict_per_target_scores(compound_data, protein_data, per_target, datasets):
    """生成 per-target 预测分数（模型/相似性/无参考）。"""
    logger.info("=" * 60)
    logger.info("[8] per-target 预测与集成")
    logger.info("=" * 60)
    t0 = time.time()

    n_compounds = compound_data["n_compounds"]
    all_genes = protein_data["gene_symbols"]
    ecfp4 = compound_data["ecfp4"]

    rows = []
    for gene in all_genes:
        ds = datasets.get(gene)
        has_positives = ds is not None and ds["n_real_pos"] > 0

        method = "no_reference"
        model_score = np.full(n_compounds, np.nan, dtype=np.float32)
        sim_score = np.full(n_compounds, np.nan, dtype=np.float32)
        borda_score = np.full(n_compounds, np.nan, dtype=np.float32)
        geo_score = np.full(n_compounds, np.nan, dtype=np.float32)
        aucw_score = np.full(n_compounds, np.nan, dtype=np.float32)
        n_real_pos = 0
        n_sim_pos = 0

        if has_positives:
            n_real_pos = ds["n_real_pos"]
            n_sim_pos = ds["n_sim"]
            max_sim = ds["max_sim_to_pos"]
            sim_score = max_sim.astype(np.float32)

            pt = per_target[gene]
            if pt["status"] == "TRAINED" and pt["models"]:
                method = "model"
                probs = []
                model_names = []
                aucs = []
                for name, model in pt["models"].items():
                    try:
                        # per-target 特征仅使用化合物指纹
                        X_pred = ecfp4.astype(np.float32)
                        prob = model.predict_proba(X_pred)[:, 1].astype(np.float32)
                        probs.append(prob)
                        model_names.append(name)
                        aucs.append(pt["cv_auc"][name])
                    except Exception as exc:
                        logger.warning(f"  {gene}-{name} 预测失败: {exc}")

                if probs:
                    probs = np.vstack(probs)  # (n_models, n_compounds)
                    # Borda rank fusion（rank-by-median）
                    ranks = np.vstack([rankdata(-p) for p in probs])
                    median_ranks = np.median(ranks, axis=0)
                    borda_score = (1.0 - (median_ranks - 1) / max(1, n_compounds - 1)).astype(np.float32)
                    # 几何平均
                    geo_score = np.exp(np.mean(np.log(probs + 1e-12), axis=0)).astype(np.float32)
                    # AUC 加权（小样本收缩）
                    aucs_arr = np.array(aucs, dtype=np.float64)
                    shrink = min(1.0, n_real_pos / 30.0)
                    uniform = 1.0 / len(aucs_arr)
                    w = shrink * (aucs_arr / aucs_arr.sum()) + (1.0 - shrink) * uniform
                    aucw_score = (w @ probs).astype(np.float32)
                    # 主预测分数采用 AUC 加权
                    model_score = aucw_score.copy()
                else:
                    pt["status"] = "SIMILARITY_FALLBACK"
                    method = "similarity"
            else:
                method = "similarity"

        for i in range(n_compounds):
            rows.append({
                "MOL_ID": compound_data["mol_ids"][i],
                "molecule_name": compound_data["names"][i],
                "SMILES": compound_data["smiles"][i],
                "target_gene": gene,
                "method": method,
                "model_score": model_score[i],
                "sim_score": sim_score[i],
                "borda_score": borda_score[i],
                "geo_mean_score": geo_score[i],
                "auc_weighted_score": aucw_score[i],
                "n_real_pos": n_real_pos,
                "n_sim_pos": n_sim_pos,
            })

    pred_df = pd.DataFrame(rows)
    logger.info(f"  per-target 预测完成，记录数: {len(pred_df)}，耗时: {time.time() - t0:.2f}s")
    return pred_df


def combine_with_cross_target(pred_df, cross_scores):
    """将 per-target 分数与 cross-target 分数按方法感知加权融合。"""
    logger.info("[9] 融合 per-target 与 cross-target 分数")
    t0 = time.time()

    method_weights = {"model": 1.0, "similarity": 0.5, "no_reference": 0.0}

    def _combine(row):
        m = row["method"]
        if m == "no_reference":
            return np.nan
        per = row.get("model_score") if m == "model" else row.get("sim_score")
        if pd.isna(per):
            per = 0.0
        cross = row.get("cross_target_score")
        if pd.isna(cross):
            return float(per)
        w = method_weights[m]
        return float((w * per + 1.0 * cross) / (w + 1.0))

    pred_df["cross_target_score"] = np.nan
    for gene, scores in cross_scores.items():
        pred_df.loc[pred_df["target_gene"] == gene, "cross_target_score"] = scores

    pred_df["prediction_score"] = pred_df.apply(_combine, axis=1)
    logger.info(f"  融合完成，耗时: {time.time() - t0:.2f}s")
    return pred_df


# ============================================================
# 富集因子分析
# ============================================================
def run_enrichment_analysis(pred_df, active_df, compound_data):
    """仅对真实正样本出现在 TCM 库中的靶标计算 EF@1%/5%/10%。"""
    logger.info("=" * 60)
    logger.info("[10] 富集因子分析")
    logger.info("=" * 60)
    t0 = time.time()

    tcm_smiles = set(compound_data["smiles"].astype(str))
    n_total = compound_data["n_compounds"]
    rows = []

    for gene, group in active_df.groupby("gene"):
        pos_smiles = set(group["canonical_smiles"].unique())
        overlap_smiles = pos_smiles & tcm_smiles
        n_pos_tcm = len(overlap_smiles)
        if n_pos_tcm == 0:
            continue

        gene_pred = pred_df[pred_df["target_gene"] == gene].copy()
        gene_pred = gene_pred.sort_values("prediction_score", ascending=False)
        labels = gene_pred["SMILES"].astype(str).isin(overlap_smiles).astype(int).values
        baseline = n_pos_tcm / n_total

        for pct in [1, 5, 10]:
            n_top = max(1, int(n_total * pct / 100))
            top_labels = labels[:n_top]
            n_hits = int(top_labels.sum())
            ef = (n_hits / n_top) / baseline if baseline > 0 else 0.0
            rows.append({
                "gene": gene,
                "top_percent": pct,
                "n_top": n_top,
                "n_hits": n_hits,
                "n_pos_tcm": n_pos_tcm,
                "baseline_rate": round(baseline, 5),
                "enrichment_factor": round(ef, 2),
            })

    ef_df = pd.DataFrame(rows)
    ef_df.to_csv(L4_RESULTS / "enrichment_analysis_v4_5.csv", index=False)
    logger.info(f"  EF 记录数: {len(ef_df)}，耗时: {time.time() - t0:.2f}s")
    if len(ef_df) > 0:
        for pct in [1, 5, 10]:
            sub = ef_df[ef_df["top_percent"] == pct]
            if len(sub):
                logger.info(
                    f"  EF@{pct}%: mean={sub['enrichment_factor'].mean():.2f}, "
                    f"max={sub['enrichment_factor'].max():.2f}"
                )
    return ef_df


# ============================================================
# 候选化合物排序
# ============================================================
def rank_candidates(pred_df, compound_data, top_n=50):
    """多维度排序候选化合物，权重在报告中显式给出。"""
    logger.info("=" * 60)
    logger.info("[11] 候选化合物排序")
    logger.info("=" * 60)
    t0 = time.time()

    # 仅使用 model/similarity 结果；no_reference 不参与排序
    sub = pred_df[pred_df["method"] != "no_reference"].copy()

    grouped = sub.groupby(["MOL_ID", "molecule_name", "SMILES"]).agg(
        avg_score=("prediction_score", "mean"),
        max_score=("prediction_score", "max"),
        std_score=("prediction_score", "std"),
        n_targets=("prediction_score", "count"),
        n_hits=("prediction_score", lambda x: int((x > 0.5).sum())),
        n_high=("prediction_score", lambda x: int((x > 0.7).sum())),
    ).reset_index()

    grouped["std_score"] = grouped["std_score"].fillna(0.0)
    grouped["consistency"] = 1.0 - grouped["std_score"].clip(0.0, 1.0)

    # 显式启发式权重
    w_avg = 0.30
    w_max = 0.20
    w_hits = 0.20
    w_high = 0.20
    w_consistency = 0.10

    grouped["composite_score"] = (
        w_avg * grouped["avg_score"]
        + w_max * grouped["max_score"]
        + w_hits * (grouped["n_hits"] / grouped["n_targets"].clip(lower=1))
        + w_high * (grouped["n_high"] / grouped["n_targets"].clip(lower=1))
        + w_consistency * grouped["consistency"]
    )

    grouped = grouped.sort_values("composite_score", ascending=False).reset_index(drop=True)
    top_df = grouped.head(top_n).copy()
    top_df["rank"] = range(1, len(top_df) + 1)

    # 为每个 top 化合物记录其 top5 靶标
    top_targets = []
    for mol_id in top_df["MOL_ID"]:
        mol_pred = sub[sub["MOL_ID"] == mol_id].nlargest(5, "prediction_score")
        top_targets.append(
            ", ".join(f"{r.target_gene}({r.prediction_score:.3f})" for _, r in mol_pred.iterrows())
        )
    top_df["top_targets"] = top_targets

    top_df = top_df[[
        "rank", "MOL_ID", "molecule_name", "SMILES",
        "composite_score", "avg_score", "max_score",
        "n_hits", "n_high", "n_targets", "consistency", "top_targets",
    ]]
    top_df.to_csv(L4_RESULTS / "tcm_top_candidates_v4_5.csv", index=False)
    logger.info(f"  Top {top_n} 候选已输出，耗时: {time.time() - t0:.2f}s")

    logger.info(f"\n{'='*80}")
    logger.info("Top 20 候选化合物 (v4.5)")
    logger.info(f"{'='*80}")
    for _, row in top_df.head(20).iterrows():
        name = str(row["molecule_name"])[:35]
        logger.info(
            f"  #{int(row['rank']):2d} {name:35s} 综合={row['composite_score']:.4f} "
            f"平均={row['avg_score']:.4f} 高置信命中={int(row['n_high'])}"
        )
    return top_df


# ============================================================
# 报告与汇总
# ============================================================
def _compute_model_performance_df(per_target, all_metrics, ef_df):
    """整理 model_performance_v4_5.csv。"""
    rows = []
    for m in all_metrics:
        row = m.copy()
        gene = row["gene"]
        # 尝试追加该靶标的 EF
        if not ef_df.empty and "gene" in ef_df.columns:
            for pct in [1, 5, 10]:
                sub = ef_df[(ef_df["gene"] == gene) & (ef_df["top_percent"] == pct)]
                if not sub.empty:
                    row[f"EF@{pct}%"] = sub["enrichment_factor"].values[0]
                else:
                    row[f"EF@{pct}%"] = np.nan
        rows.append(row)

    # 添加 ensemble 汇总行
    for gene, pt in per_target.items():
        if pt["status"] == "TRAINED" and pt["models"]:
            row = {
                "gene": gene,
                "model": "ensemble",
                "cv_auc": round(np.mean(list(pt["cv_auc"].values())), 4),
                "cv_aupr": round(np.mean(list(pt["cv_aupr"].values())), 4),
                "status": "TRAINED",
                "n_real_pos": pt["n_real_pos"],
                "n_sim": pt["n_sim"],
                "n_neg": pt["n_neg"],
            }
            for pct in [1, 5, 10]:
                sub = ef_df[(ef_df["gene"] == gene) & (ef_df["top_percent"] == pct)]
                row[f"EF@{pct}%"] = sub["enrichment_factor"].values[0] if not sub.empty else np.nan
            rows.append(row)

    return pd.DataFrame(rows)


def generate_report(compound_data, protein_data, active_df, per_target, all_metrics, ef_df, top_df, t_total):
    """生成 Markdown 汇总报告。"""
    logger.info("=" * 60)
    logger.info("[12] 生成汇总报告")
    logger.info("=" * 60)

    lines = []
    lines.append("# Phase 4 v4.5: CIRI铁衰老中药单体ML筛选 - 模型构建报告")
    lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"总耗时: {t_total/60:.1f} 分钟")

    lines.append("\n## 1. 数据概览")
    lines.append(f"- TCM化合物总数: {compound_data['n_compounds']}")
    lines.append(f"- 蛋白靶标总数: {len(protein_data['gene_symbols'])}")
    lines.append(f"- 有真实活性数据的靶标数: {active_df['gene'].nunique()}")
    lines.append(f"- 真实正样本总数（去重后）: {len(active_df)}")

    trained = [g for g, pt in per_target.items() if pt["status"] == "TRAINED"]
    fallback = [g for g, pt in per_target.items() if pt["status"] == "SIMILARITY_FALLBACK"]
    noref = [g for g in protein_data["gene_symbols"] if g not in per_target]
    lines.append("\n## 2. 靶标分层")
    lines.append(f"- 集成模型可训练靶标: {len(trained)} 个")
    lines.append(f"- 回退到相似性方法: {len(fallback)} 个")
    lines.append(f"- 无参考数据: {len(noref)} 个")
    if trained:
        lines.append(f"- 可训练靶标列表: {', '.join(trained)}")

    lines.append("\n## 3. 模型性能汇总（CV 真实标签）")
    perf = _compute_model_performance_df(per_target, all_metrics, ef_df)
    if not perf.empty and "cv_auc" in perf.columns:
        kept_perf = perf[perf["status"] == "KEPT"]
        if not kept_perf.empty:
            lines.append(f"- 保留模型数: {len(kept_perf)}")
            for model_name in kept_perf["model"].unique():
                sub = kept_perf[kept_perf["model"] == model_name]
                auc_std = sub["cv_auc_std"].mean() if "cv_auc_std" in sub.columns else 0.0
                lines.append(
                    f"- {model_name}: 平均 AUC={sub['cv_auc'].mean():.4f}±{auc_std:.4f}, "
                    f"平均 AUPR={sub['cv_aupr'].mean():.4f}, 覆盖靶标={len(sub)}"
                )
        ens_perf = perf[perf["model"] == "ensemble"]
        if not ens_perf.empty:
            lines.append(
                f"- ensemble 平均 AUC={ens_perf['cv_auc'].mean():.4f}, "
                f"平均 AUPR={ens_perf['cv_aupr'].mean():.4f}"
            )
        # 高 AUC 警告
        high_auc = kept_perf[kept_perf["cv_auc"] > 0.98]
        if not high_auc.empty:
            lines.append(
                f"- ⚠️ 有 {len(high_auc)} 个模型 CV AUC > 0.98，"
                "建议检查负样本设计、骨架泄漏及结果可解释性。"
            )

    lines.append("\n## 4. 富集因子分析")
    if not ef_df.empty:
        for pct in [1, 5, 10]:
            sub = ef_df[ef_df["top_percent"] == pct]
            if not sub.empty:
                lines.append(
                    f"- EF@{pct}%: mean={sub['enrichment_factor'].mean():.2f}, "
                    f"max={sub['enrichment_factor'].max():.2f}"
                )
    else:
        lines.append("- 无靶标在 TCM 库中出现真实正样本，未计算 EF。")

    lines.append("\n## 5. Top 20 候选化合物")
    lines.append("| 排名 | 化合物 | 综合得分 | 平均得分 | 高置信命中 | Top 靶标 |")
    lines.append("|------|--------|----------|----------|------------|----------|")
    for _, row in top_df.head(20).iterrows():
        name = str(row["molecule_name"])[:30]
        targets = str(row.get("top_targets", "N/A"))[:80]
        lines.append(
            f"| {int(row['rank'])} | {name} | {row['composite_score']:.4f} | "
            f"{row['avg_score']:.4f} | {int(row['n_high'])} | {targets} |"
        )

    lines.append("\n## 6. 候选排序权重（显式记录）")
    lines.append("composite_score = 0.30 * avg_score + 0.20 * max_score + "
                 "0.20 * (n_hits / n_targets) + 0.20 * (n_high / n_targets) + "
                 "0.10 * consistency")
    lines.append("- avg_score: 该化合物在所有可预测靶标上的平均预测分")
    lines.append("- max_score: 最大预测分")
    lines.append("- n_hits: prediction_score > 0.5 的靶标数")
    lines.append("- n_high: prediction_score > 0.7 的靶标数")
    lines.append("- consistency: 1 - std(prediction_score)，衡量跨靶标预测一致性")

    lines.append("\n## 7. v4.5 改进总结")
    lines.append("- 真实活性 vs 相似性扩展标签分离，CV 评估仅使用真实标签。")
    lines.append("- CV 采用 Murcko 骨架分组的 StratifiedGroupKFold，防止相同骨架同时进入训练/验证集。")
    lines.append("- 多模型集成 + 5 折 CV，AUC>0.6 才保留，并记录 AUC/AUPR 标准差。")
    lines.append("- Borda 排序融合、概率几何平均、AUC 加权（小样本收缩）。")
    lines.append("- 跨靶标 CPI 模型共享负样本/蛋白信息。")
    lines.append("- per-target 与 cross-target 方法感知加权融合。")
    lines.append("- 仅对 TCM 中真实出现的活性样本计算 EF。")
    lines.append("- 新增 AUC  Sanity Check：对 AUC>0.98 或 <0.6 的模型发出警告。")

    lines.append("\n## 8. 已知局限性与使用建议")
    lines.append("- 当前真实负样本主要来自其他靶标活性分子，任务可能过简，导致 CV AUC 偏高。")
    lines.append("- 富集因子（EF）使用与训练相同的真实正样本计算，属于乐观估计，不可视为独立验证。")
    lines.append("- 如用于论文发表，建议补充：1) 外部独立测试集；2) 真实 inactive/ decoy 负样本；3) 时间切分或分子骨架切分验证。")

    report_path = L4_RESULTS / "phase4_report_v4_5.md"
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
    logger.info("Phase 4 v4.5: 中药单体ML筛选 - 模型构建与预测")
    logger.info(f"启动: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 1. 数据加载
    compound_data = load_compound_data()
    protein_data = load_protein_data()
    active_df = load_activity_data(set(protein_data["gene_symbols"]))

    # 2. per-target 训练集
    datasets = build_per_target_datasets(compound_data, protein_data, active_df)

    # 3. per-target 模型
    per_target, all_metrics = train_per_target_models(datasets)

    # 4. 跨靶标模型
    cross_model = train_cross_target_model(datasets, compound_data, protein_data)
    cross_scores = predict_cross_target_scores(
        cross_model, datasets, compound_data, protein_data, protein_data["gene_symbols"]
    )

    # 5. per-target 预测
    pred_df = predict_per_target_scores(compound_data, protein_data, per_target, datasets)
    pred_df = combine_with_cross_target(pred_df, cross_scores)

    # 6. 输出全预测表
    pred_df.to_csv(L4_RESULTS / "tcm_predictions_full_v4_5.csv", index=False)
    logger.info(f"  全预测表: {L4_RESULTS / 'tcm_predictions_full_v4_5.csv'}")

    # 7. EF 分析
    ef_df = run_enrichment_analysis(pred_df, active_df, compound_data)

    # 8. 候选排序
    top_df = rank_candidates(pred_df, compound_data, top_n=50)

    # 9. 模型性能表
    perf_df = _compute_model_performance_df(per_target, all_metrics, ef_df)
    perf_df.to_csv(L4_RESULTS / "model_performance_v4_5.csv", index=False)
    logger.info(f"  模型性能表: {L4_RESULTS / 'model_performance_v4_5.csv'}")

    # 10. 训练指标 JSON
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "n_compounds": int(compound_data["n_compounds"]),
        "n_genes": int(len(protein_data["gene_symbols"])),
        "n_active_genes": int(active_df["gene"].nunique()),
        "n_real_positives": int(len(active_df)),
        "per_target": {
            g: {
                "status": pt["status"],
                "n_real_pos": int(pt["n_real_pos"]),
                "n_sim": int(pt["n_sim"]),
                "n_neg": int(pt["n_neg"]),
                "models": list(pt["models"].keys()),
                "cv_auc": {k: round(v, 4) for k, v in pt["cv_auc"].items()},
            }
            for g, pt in per_target.items()
        },
        "all_model_metrics": all_metrics,
        "enrichment_summary": ef_df.groupby("top_percent")["enrichment_factor"].mean().round(2).to_dict()
        if not ef_df.empty else {},
    }
    with open(L4_RESULTS / "training_metrics_v4_5.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  训练指标 JSON: {L4_RESULTS / 'training_metrics_v4_5.json'}")

    # 11. 报告
    t_total = time.time() - t_start
    report = generate_report(
        compound_data, protein_data, active_df, per_target, all_metrics, ef_df, top_df, t_total
    )
    print("\n" + report)

    logger.info(f"\n总耗时: {t_total/60:.1f} 分钟")
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as exc:
        logger.exception(f"未捕获异常: {exc}")
        sys.exit(1)
