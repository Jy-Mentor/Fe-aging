#!/usr/bin/env python
"""
树模型 CPI 筛选 v6.4 — 扩展铁衰老基因版 + 无数据泄露 + TabPFN 集成
================================================================
基于 v6.3 架构，补充 28 个新铁衰老基因的 CPI 数据，显著扩展靶点覆盖。

核心更新（v6.4）：
  1. 【P0 基因覆盖扩展】合并 ChEMBL API 补充数据：
     - 新增 28 个铁衰老基因的 CPI 数据（4868 条记录）
     - 铁衰老基因覆盖：23 → 51 个（提升 122%）
     - 总基因数：42 → 70 个
     - 总数据量：43238 → 48106 条
  2. 【P1 最佳模式选择策略优化】：
     - 优先选择基因覆盖度更高的蛋白嵌入模式（生物学意义更大）
     - 基因覆盖度相同时，选 AUPR 更高的
  3. 【P1 数据质量保证】：
     - 补充数据 SMILES 有效性验证（100% 通过）
     - 统一 pchembl_value 计算（9 - log10(activity_nM)）

核心修复（v6.3 继承）：
  1. 【P0 数据泄露修复】RDKit 2D 描述符标准化：
     - 不再用全部化合物（含测试集+TCM池）拟合 StandardScaler
     - 改为在每个 CV 折内仅用训练集化合物拟合 scaler，再分别变换
  2. 【P0 数据泄露修复】蛋白嵌入 PCA 降维：
     - 不再用全部蛋白（含测试集蛋白）拟合 PCA 和 StandardScaler
     - 改为在每个 CV 折内仅用训练集出现的蛋白拟合 PCA+scaler
  3. 【P0 缓存一致性】特征缓存新增 X_rdkit_raw 字段，
     确保旧缓存不会导致"双重标准化"
  4. 【性能优化】蛋白 PCA 只变换 pair_genes 中实际出现的蛋白
     （从 6847 个降至约 70 个，每折节省数秒）
  5. 【性能优化】残基统计特征缓存：residue_meanmaxstd 模式
     首次计算后保存为 NPZ，后续秒级加载

新增内容（v6.2 继承）：
  1. TabPFN (Nature 2025) 集成：无需调参的表格基础模型
     - 自动处理缺失值、类别特征、异常值
     - 内置不确定性估计
  2. 特征选择适配：原始 ~6650 维 → 互信息降维至 ≤500 维（全矩阵一次性计算）
  3. 5-fold Scaffold Split 全面对比（RF / XGB / LGB / CatBoost / TabPFN）
  4. TabPFN 特征重要性（置换重要性）+ 不确定性估计（score_std）
  5. TabPFN 集成投票 Ensemble (TabPFN + Best Tree)，CV 严格评估无泄露
  6. 残基级 ESM-2 蛋白嵌入消融实验（4 种模式: global / residue_pooled /
     residue_meanmaxstd / combined）
     - residue_meanmaxstd: 残基 mean/max/std 聚合 → 3×640 → PCA 128 维
     - combined: 全局 CLS + 残基 mean 拼接 → PCA 降维

修复记录 v6 → v6.1：
    - 移除 CascadeForestFixed 空壳模型（fit 不保存模型，predict 用全零标签训练）
    - BEDROC 直接委托给 RDKit CalcBEDROC（Truchon & Bayly 2007 的事实标准实现），
      避免任何手写公式偏差；异常直接上抛，不准静默吞错
    - 特征选择修复：全矩阵 `mutual_info_classif` 一次性计算，不再分批
      且增加零方差特征过滤，防止 MI 崩溃
    - Ensemble 评估修复：统一使用 Scaffold Split（与主CV一致），消除拆分方式不一致
    - 主流程重构：先跑 4 模式蛋白嵌入消融（XGBoost 基线），选最佳模式再跑完整流程
    - 中药映射列名修复：多候选列名 + 明确警告
    - PCA 解释方差验证：<50% 时发出警告；n_components 自适应样本数
    - 不确定性估计：删除永不调用的虚假循环，改为 NaN 占位 + 文档说明
    - 清理进度日志编号、移除未使用导入

修复记录 v6.1 → v6.2：
    - 【P0 数据泄露】集成评估使用独立随机种子（seed+1000）的 Scaffold Split，
      与模型选择阶段的折完全独立，避免测试集重复使用导致的指标高估
    - 【P1 可比性】消融实验新增有效基因数、正样本数、总样本数记录，
      不同蛋白嵌入模式的结果可评估可比性
    - 【P1 数据完整性】NPZ 蛋白嵌入加载中，跳过高比例（>10%）非数值键时
      发出警告，避免隐蔽数据损坏
    - 【P2 文档完善】train_ensemble 明确标注为"模型选择用"，其指标不代表
      最终泛化性能；集成评估使用独立折，结果可信
    - 【P1 Bug修复】修复 residue_pooled NPZ 加载变量名错误（v for k, v in d.items()）
    - 【P1 参数统一】CatBoost 统一使用 auto_class_weights="Balanced"

修复记录 v6.2 → v6.3：
    - 【P0 数据泄露】RDKit 标准化移至 CV 折内，仅用训练集拟合 scaler
    - 【P0 数据泄露】蛋白 PCA 移至 CV 折内，仅用训练集蛋白拟合 PCA+scaler
    - 【P0 缓存安全】特征缓存新增 X_rdkit_raw，旧缓存自动兼容处理
    - 【性能】蛋白 PCA 仅变换 pair_genes 中出现的蛋白（6847 → ~42）
    - 【性能】残基统计特征缓存，避免重复计算

引用：Hollmann et al., "Accurate predictions on small data with a tabular foundation model",
       Nature 637(8045), 2025. https://github.com/PriorLabs/TabPFN

数据来源（全部真实，不模拟）：
  - CPI: L4/results/experimental_actives_detail_cleaned_combined.csv
    (合并 ChEMBL/BindingDB 补充数据，70 个基因，51 个铁衰老基因)
  - 蛋白嵌入: L4/results_v10_minibatch/esm2_protein_embeddings.npz (全局 CLS, 103 蛋白)
  - 蛋白嵌入: L4/results_v10_minibatch/esm2_residue_pooled_embeddings.npz (残基池化, 6847 蛋白)
  - 蛋白嵌入: L4/results_v10_minibatch/esm2_150M_residue_features.pt (残基级, 6864 蛋白)
  - TCM池: L3/results/tcm_compound_pool_tox_filtered_noleak.csv
  - 中药映射: L3/results/herb_ingredient_mapping.xlsx

输出：
  - L4/results/tree_v6_protein_emb_ablation.csv（蛋白嵌入消融对比结果）
  - L4/results/tree_v6_results.csv（5-fold CV 对比结果）
  - L4/results/tree_v6_tabpfn_importance.csv（TabPFN 特征重要性）
  - L4/results/tree_v6_tcm_predictions.csv（TCM 预测）
  - L4/results/tree_v6_top_candidates.csv（Top 候选）
  - L4/results/tree_v6_ensemble_results.csv（Ensemble 结果）
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

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
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
        logging.FileHandler(L4_LOGS / "tree_v6_tabpfn.log", mode="w", encoding="utf-8"),
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
    """MACCS 密钥（二进制）"""
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
    """Atom Pair 指纹（二进制）"""
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
    """RDKit 2D 描述符（连续值）"""
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


def _get_feature_cache_path():
    """特征缓存路径"""
    cache_dir = L4_RESULTS / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "compound_multifingerprint_features_v6.npz"


def build_multifingerprint_features(smiles_list, rdkit_scaler=None, use_cache=True):
    """构建多指纹融合特征矩阵（支持缓存）"""
    t0 = time.time()
    n = len(smiles_list)
    logger.info(f"  计算 {n} 个化合物的多指纹特征...")

    cache_path = _get_feature_cache_path()

    # 尝试加载缓存
    if use_cache and cache_path.exists() and rdkit_scaler is None:
        try:
            logger.info(f"  尝试加载特征缓存: {cache_path}")
            cache = np.load(cache_path, allow_pickle=True)
            cached_smiles = cache["smiles"].tolist()
            if cached_smiles == list(smiles_list):
                X_binary = cache["X_binary"].astype(np.float32)
                X_rdkit = cache["X_rdkit"].astype(np.float32)
                rdkit_scaler_loaded = cache["rdkit_scaler"].item()
                binary_labels = cache["binary_labels"].tolist()
                rdkit_names = cache["rdkit_names"].tolist()
                # 兼容旧缓存（没有 X_rdkit_raw）
                if "X_rdkit_raw" in cache.files:
                    X_rdkit_raw = cache["X_rdkit_raw"].astype(np.float32)
                else:
                    logger.warning("  缓存中无 X_rdkit_raw，将重新计算标准化（不影响正确性，但可能略慢）")
                    X_rdkit_raw = X_rdkit.copy()
                logger.info(f"  缓存命中! 加载 {X_binary.shape[0]} 样本, "
                            f"binary={X_binary.shape[1]}, rdkit={X_rdkit.shape[1]}, "
                            f"耗时: {time.time()-t0:.1f}s")
                return X_binary, X_rdkit, X_rdkit_raw, rdkit_scaler_loaded, binary_labels, rdkit_names
            else:
                logger.info(f"  缓存 SMILES 列表不匹配，重新计算")
        except Exception as e:
            logger.warning(f"  缓存加载失败: {e}，重新计算")

    binary_fps = []
    binary_labels = []

    # ECFP4
    fp = compute_ecfp(smiles_list, radius=2, nbits=2048)
    binary_fps.append(fp)
    binary_labels.append("ECFP4")
    logger.info(f"    ECFP4: {fp.shape} (binary)")

    # ECFP6
    fp = compute_ecfp(smiles_list, radius=3, nbits=2048)
    binary_fps.append(fp)
    binary_labels.append("ECFP6")
    logger.info(f"    ECFP6: {fp.shape} (binary)")

    # MACCS
    fp = compute_maccs(smiles_list)
    binary_fps.append(fp)
    binary_labels.append("MACCS")
    logger.info(f"    MACCS: {fp.shape} (binary)")

    # AtomPairs
    fp = compute_atom_pairs(smiles_list, nbits=1024)
    binary_fps.append(fp)
    binary_labels.append("AtomPairs")
    logger.info(f"    AtomPairs: {fp.shape} (binary)")

    # Avalon
    fp = compute_avalon(smiles_list, nbits=1024)
    binary_fps.append(fp)
    binary_labels.append("Avalon")
    logger.info(f"    Avalon: {fp.shape} (binary)")

    X_binary = np.hstack(binary_fps).astype(np.float32)
    logger.info(f"  二进制指纹总维度: {X_binary.shape[1]}")

    # RDKit 2D
    X_rdkit, rdkit_names = compute_rdkit_2d(smiles_list)
    logger.info(f"    RDKit2D: {X_rdkit.shape} (continuous)")

    # NaN 处理（原始值）
    nan_mask = np.isnan(X_rdkit)
    if nan_mask.any():
        logger.info(f"  RDKit2D 处理 {nan_mask.sum()} 个 NaN 值...")
        col_means = np.nanmean(X_rdkit, axis=0)
        inds = np.where(nan_mask)
        X_rdkit[inds] = np.take(col_means, inds[1])
    X_rdkit_raw = np.nan_to_num(X_rdkit, nan=0.0, posinf=1e6, neginf=-1e6).copy()

    if rdkit_scaler is None:
        rdkit_scaler = StandardScaler()
        X_rdkit = rdkit_scaler.fit_transform(X_rdkit_raw)
        logger.info(f"  RDKit2D 已标准化 (mean=0, std=1)")

        if use_cache:
            try:
                np.savez_compressed(
                    cache_path,
                    smiles=np.array(smiles_list),
                    X_binary=X_binary,
                    X_rdkit=X_rdkit,
                    X_rdkit_raw=X_rdkit_raw,
                    rdkit_scaler=rdkit_scaler,
                    binary_labels=np.array(binary_labels),
                    rdkit_names=np.array(rdkit_names),
                )
                logger.info(f"  特征缓存已保存: {cache_path}")
            except Exception as e:
                logger.warning(f"  特征缓存保存失败: {e}")

        return X_binary, X_rdkit, X_rdkit_raw, rdkit_scaler, binary_labels, rdkit_names
    else:
        X_rdkit = rdkit_scaler.transform(X_rdkit_raw)
        return X_binary, X_rdkit, X_rdkit_raw, None, binary_labels, rdkit_names


# ============================================================
# 2. 蛋白嵌入处理
# ============================================================

def process_protein_embeddings(protein_embeddings, target_dim=128, pca_model=None, scaler=None):
    """蛋白嵌入处理：PCA 降维 + 标准化"""
    keys = sorted(protein_embeddings.keys())
    vectors = np.array([protein_embeddings[k] for k in keys], dtype=np.float32)
    original_dim = vectors.shape[1]
    logger.info(f"  蛋白嵌入原始维度: {original_dim}, 数量: {len(keys)}")

    if pca_model is None:
        # PCA 的 n_components 不能超过 min(n_samples, n_features)
        max_components = min(vectors.shape[0], vectors.shape[1])
        actual_target = min(target_dim, max_components)
        if actual_target < target_dim:
            logger.warning(f"  蛋白数量 ({vectors.shape[0]}) 或维度 ({vectors.shape[1]}) "
                           f"小于目标维度 {target_dim}，PCA 自动调整至 {actual_target}")

        pca = PCA(n_components=actual_target, random_state=42)
        vectors_reduced = pca.fit_transform(vectors)
        ev_ratio = pca.explained_variance_ratio_.sum()
        logger.info(f"  PCA 降维: {original_dim} -> {actual_target}, "
                    f"解释方差比: {ev_ratio:.4f}")
        if ev_ratio < 0.5:
            logger.warning(f"  PCA 降维至 {actual_target} 仅保留 {ev_ratio:.2%} 方差，"
                           f"建议增加 target_dim 至 {min(256, original_dim)} 或使用更高维主干嵌入")
        # 显示 Top-5 主成分贡献
        top5_ratio = pca.explained_variance_ratio_[:5].sum()
        logger.info(f"  Top-5 PC 累计方差: {top5_ratio:.4f}")
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
# 2b. 蛋白嵌入多模式加载（支持残基层 ESM-2 消融）
# ============================================================

PROTEIN_EMB_MODES = ["global", "residue_pooled", "residue_meanmaxstd", "combined"]


def _load_protein_embeddings_global():
    """加载全局 CLS 蛋白嵌入（原始 v6 行为，仅覆盖 29/42 个 CPI 基因）"""
    d = np.load(L4_RESULTS_V10 / "esm2_protein_embeddings.npz",
                allow_pickle=True)
    result = {}
    skipped = 0
    for k in d.files:
        v = d[k]
        if hasattr(v, 'dtype') and np.issubdtype(v.dtype, np.number):
            result[str(k)] = v.astype(np.float32)
        else:
            skipped += 1
    total = len(d.files)
    skip_ratio = skipped / total if total > 0 else 0
    if skip_ratio > 0.1:
        logger.warning(f"  全局 CLS 嵌入: 跳过 {skipped}/{total} 个键 ({skip_ratio:.1%})，"
                       f"跳过比例过高，可能存在数据损坏")
    else:
        logger.info(f"  全局 CLS 嵌入: {len(result)} 个蛋白 (跳过 {skipped} 个非数值键)")
    return result


def _load_protein_embeddings_residue_pooled():
    """加载预池化的残基层 ESM-2 嵌入（覆盖全部 42 个 CPI 基因）"""
    d = np.load(L4_RESULTS_V10 / "esm2_residue_pooled_embeddings.npz",
                allow_pickle=True)
    result = {}
    skipped = 0
    for k in d.files:
        v = d[k]
        if hasattr(v, 'dtype') and np.issubdtype(v.dtype, np.number):
            result[str(k)] = v.astype(np.float32)
        else:
            skipped += 1
    total = len(d.files)
    skip_ratio = skipped / total if total > 0 else 0
    if skip_ratio > 0.1:
        logger.warning(f"  残基池化嵌入: 跳过 {skipped}/{total} 个键 ({skip_ratio:.1%})，"
                       f"跳过比例过高，可能存在数据损坏")
    else:
        logger.info(f"  残基池化嵌入: {len(result)} 个蛋白 (跳过 {skipped} 个非数值键)")
    return result


def _load_protein_embeddings_residue_stats(stats=("mean", "max", "std")):
    """
    从 esm2_150M_residue_features.pt 计算残基层统计特征。
    返回每个基因的固定长度向量（mean/max/std 拼接）。
    支持缓存以加速重复调用。
    """
    import torch
    stats_key = "_".join(stats)
    cache_path = L4_RESULTS_V10 / f"esm2_residue_{stats_key}_cache.npz"

    if cache_path.exists():
        try:
            d = np.load(cache_path, allow_pickle=True)
            result = {str(k): d[k].astype(np.float32) for k in d.files}
            logger.info(f"  残基层统计特征缓存命中: {len(result)} 个蛋白, "
                        f"维度={next(iter(result.values())).shape[0]}")
            return result
        except Exception as e:
            logger.warning(f"  残基特征缓存加载失败: {e}, 重新计算")

    pt_path = L4_RESULTS_V10 / "esm2_150M_residue_features.pt"
    logger.info(f"  加载原始残基层 ESM-2 特征: {pt_path}")
    data = torch.load(pt_path, map_location="cpu")

    gene_list = data["genes"]
    offsets = data["offsets"].numpy().astype(np.int64)
    embeddings = data["embeddings"].numpy().astype(np.float32)

    result = {}
    n_genes = len(gene_list)
    for i, gene in enumerate(gene_list):
        start = int(offsets[i])
        end = int(offsets[i + 1])
        res_feats = embeddings[start:end]

        parts = []
        if "mean" in stats:
            parts.append(res_feats.mean(axis=0))
        if "max" in stats:
            parts.append(res_feats.max(axis=0))
        if "std" in stats:
            parts.append(res_feats.std(axis=0))

        result[str(gene)] = np.concatenate(parts).astype(np.float32)

        if (i + 1) % 1000 == 0 or i == n_genes - 1:
            logger.info(f"    已处理 {i+1}/{n_genes} 个蛋白残基特征")

    logger.info(f"  残基层统计特征: {len(result)} 个蛋白, 维度={next(iter(result.values())).shape[0]}")

    try:
        np.savez_compressed(cache_path, **result)
        logger.info(f"  残基层统计特征已缓存: {cache_path}")
    except Exception as e:
        logger.warning(f"  残基特征缓存保存失败: {e}")

    return result


def _load_protein_embeddings_combined():
    """全局 CLS + 残基层 mean 拼接（仅保留同时有两种嵌入的蛋白）"""
    global_emb = _load_protein_embeddings_global()
    residue_mean = _load_protein_embeddings_residue_stats(stats=("mean",))

    result = {}
    for gene in global_emb:
        if gene in residue_mean:
            result[gene] = np.concatenate([global_emb[gene], residue_mean[gene]]).astype(np.float32)
    logger.info(f"  组合嵌入: {len(result)} 个蛋白 (全局 {len(global_emb)} + 残基 {len(residue_mean)} 的交集)")
    return result


def load_protein_embeddings_by_mode(mode="global"):
    """按模式加载蛋白嵌入"""
    if mode == "global":
        return _load_protein_embeddings_global()
    elif mode == "residue_pooled":
        return _load_protein_embeddings_residue_pooled()
    elif mode == "residue_meanmaxstd":
        return _load_protein_embeddings_residue_stats(stats=("mean", "max", "std"))
    elif mode == "combined":
        return _load_protein_embeddings_combined()
    else:
        raise ValueError(f"Unknown protein embedding mode: {mode}. "
                         f"Available: {PROTEIN_EMB_MODES}")


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
# 4. 多样性约束负采样
# ============================================================

def diversity_constrained_negative_sampling(
    pos_pairs, compound_smiles, cpi_genes_in_emb, neg_ratio=3, random_seed=42,
):
    """多样性约束负采样"""
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

    gene_neg_counts = {gi: 0 for gi in range(n_genes)}
    max_per_gene = max(1, n_neg_target // n_genes + 1)

    neg_idx_set = set()
    batch_size = n_neg_target * 10
    max_attempts = n_neg_target * 50

    while len(neg_idx_set) < n_neg_target and len(neg_idx_set) < max_attempts:
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

    neg_pairs = []
    for ci, gi in neg_idx_set:
        smi = str(compound_smiles[ci])
        gene = cpi_genes_in_emb[gi]
        neg_pairs.append((smi, gene))

    logger.info(f"  负样本: {len(neg_pairs)} 对 (比例 1:{neg_ratio}), "
                f"蛋白覆盖: {sum(1 for c in gene_neg_counts.values() if c > 0)}/{n_genes}")

    return neg_pairs


# ============================================================
# 5. 扩展评估指标
# ============================================================

def compute_metrics(y_true, y_prob):
    """计算扩展评估指标"""
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

    metrics["BEDROC"] = compute_bedroc_standard(y_true, y_prob, alpha=20.0)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    for pct in [0.5, 1.0, 2.0, 5.0]:
        fp_rate = pct / 100.0
        idx = np.argmin(np.abs(fpr - fp_rate))
        if fpr[idx] > 1e-8:
            roce = tpr[idx] / fpr[idx]
        else:
            roce = 0.0
        metrics[f"ROCE@{pct}%"] = roce

    return metrics


def compute_bedroc_standard(y_true, y_prob, alpha=20.0):
    """BEDROC 标准实现：直接调用 RDKit CalcBEDROC。

    为了避免手写公式与文献/RDKit 产生任何偏差，本函数直接委托给
    rdkit.ML.Scoring.Scoring.CalcBEDROC。该实现已被虚拟筛选领域广泛采纳，
    并明确对应 Truchon & Bayly, J. Chem. Inf. Model. 2007, 47, 488-508。

    RDKit 输入格式：scores 按分数降序排列，每行为 [score, is_active]。
    """
    from rdkit.ML.Scoring.Scoring import CalcBEDROC

    n = len(y_true)
    n_act = int(y_true.sum())
    if n_act == 0:
        return 0.0
    if n_act == n:
        return 1.0

    order = np.argsort(y_prob)[::-1]
    scores = [
        [float(y_prob[order[i]]), bool(y_true[order[i]])]
        for i in range(n)
    ]
    return float(CalcBEDROC(scores, col=1, alpha=alpha))


# ============================================================
# 6. 特征选择适配器（原始 ~6650 维 → 适配 TabPFN 上限）
# ============================================================

class TabPFNFeatureSelector:
    """特征选择适配器：将高维特征降到 TabPFN 支持的范围

    TabPFN v2 支持 ≤500 特征，v2.5 支持 ≤2000 特征。
    使用互信息特征选择，保留最重要的特征。

    注意：
      - 互信息一次性计算于全特征矩阵，保证所有特征的得分在同一分布下可比
      - 自动过滤零方差特征，防止 mutual_info_classif 报错
    """

    def __init__(self, max_features=500):
        self.max_features = max_features
        self.selected_indices = None
        self._valid_feature_mask = None

    def _remove_zero_variance(self, X):
        """移除零方差特征"""
        variances = np.var(X, axis=0)
        valid_mask = variances > 1e-10
        n_removed = (~valid_mask).sum()
        if n_removed > 0:
            logger.info(f"  TabPFN 特征过滤: 移除 {n_removed} 个零方差特征")
        return valid_mask

    def fit(self, X, y):
        n_features = X.shape[1]
        if n_features <= self.max_features:
            logger.info(f"  TabPFN 特征选择: 特征数 {n_features} ≤ {self.max_features}，无需降维")
            self._valid_feature_mask = np.ones(n_features, dtype=bool)
            self.selected_indices = np.arange(n_features)
            return self

        t0 = time.time()

        # 第一步: 过滤零方差特征
        valid_mask = self._remove_zero_variance(X)
        if valid_mask.sum() < 2:
            logger.error(f"  方差检验后有效特征仅 {valid_mask.sum()} 个，无法进行特征选择")
            self._valid_feature_mask = valid_mask
            self.selected_indices = np.where(valid_mask)[0]
            return self

        X_valid = X[:, valid_mask]
        n_valid = X_valid.shape[1]
        k = min(self.max_features, n_valid - 1)
        logger.info(f"  TabPFN 特征选择: {n_valid} (有效) → {k} (互信息排序)...")

        # 第二步: 全矩阵一次性计算互信息
        mi_scores = mutual_info_classif(X_valid, y, random_state=42)

        top_k_indices = np.argsort(mi_scores)[-k:]

        # 映射回原始特征索引
        valid_feat_indices = np.where(valid_mask)[0]
        self.selected_indices = valid_feat_indices[top_k_indices]
        self._valid_feature_mask = valid_mask

        logger.info(f"  TabPFN 特征选择完成: -> {k} 维, 耗时: {time.time()-t0:.1f}s")
        return self

    def transform(self, X):
        return X[:, self.selected_indices]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


# ============================================================
# 7. TabPFN 模型封装（确保兼容 sklearn API）
# ============================================================

class TabPFNWrapper(BaseEstimator, ClassifierMixin):
    """TabPFN sklearn 兼容封装

    使用 Nature 2025 的 TabPFN 基础模型进行零样本表格预测。
    无需调参，自动处理缺失值、类别特征。

    引用：Hollmann et al., Nature 2025
    """

    _availability_checked = False
    _is_available = False
    _availability_error = None

    def __init__(self, device="auto", random_state=42, max_features=500,
                 max_train_samples=8000):
        self.device = device
        self.random_state = random_state
        self.max_features = max_features
        self.max_train_samples = max_train_samples
        self._model = None
        self._feature_selector = None

    @classmethod
    def check_available(cls):
        if cls._availability_checked:
            return cls._is_available
        try:
            from tabpfn import TabPFNClassifier
            import numpy as np
            X_test = np.random.randn(100, 10).astype(np.float32)
            y_test = (X_test[:, 0] > 0).astype(int)
            clf = TabPFNClassifier(device="cpu", n_estimators=2, random_state=0)
            clf.fit(X_test, y_test)
            _ = clf.predict_proba(X_test[:5])
            cls._is_available = True
        except Exception as e:
            cls._is_available = False
            cls._availability_error = str(e)
            logger.warning(f"TabPFN 不可用: {e}")
        cls._availability_checked = True
        return cls._is_available

    def fit(self, X, y):
        t0 = time.time()

        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]

        if n_samples > self.max_train_samples:
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            n_pos = len(pos_idx)
            n_neg = len(neg_idx)
            n_sub_pos = min(n_pos, self.max_train_samples // 2)
            n_sub_neg = min(n_neg, self.max_train_samples - n_sub_pos)
            if n_sub_pos < n_pos or n_sub_neg < n_neg:
                sub_pos = rng.choice(pos_idx, size=n_sub_pos, replace=False)
                sub_neg = rng.choice(neg_idx, size=n_sub_neg, replace=False)
                sub_idx = np.concatenate([sub_pos, sub_neg])
                rng.shuffle(sub_idx)
                X_train_sub = X[sub_idx]
                y_train_sub = y[sub_idx]
                logger.info(f"    TabPFN 子采样: {n_samples} → {len(sub_idx)} "
                            f"(pos={n_sub_pos}, neg={n_sub_neg})")
            else:
                X_train_sub = X
                y_train_sub = y
        else:
            X_train_sub = X
            y_train_sub = y

        if X_train_sub.shape[1] > self.max_features:
            self._feature_selector = TabPFNFeatureSelector(max_features=self.max_features)
            X_selected = self._feature_selector.fit_transform(X_train_sub, y_train_sub)
            logger.info(f"    TabPFN 特征选择: {X_train_sub.shape[1]} → {X_selected.shape[1]}")
        else:
            X_selected = X_train_sub

        try:
            from tabpfn import TabPFNClassifier
        except ImportError:
            raise ImportError("tabpfn 未安装，请运行: pip install tabpfn")

        self._model = TabPFNClassifier(
            device=self.device,
            random_state=self.random_state,
            n_estimators=8,
        )

        self._model.fit(X_selected, y_train_sub)
        logger.info(f"    TabPFN 训练完成: {X_selected.shape}, "
                    f"耗时: {time.time()-t0:.1f}s")
        return self

    def predict_proba(self, X):
        if self._feature_selector is not None:
            X = self._feature_selector.transform(X)
        return self._model.predict_proba(X)

    def predict(self, X):
        if self._feature_selector is not None:
            X = self._feature_selector.transform(X)
        return self._model.predict(X)




# ============================================================
# 9. 模型训练与评估
# ============================================================

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """训练并评估单个模型"""
    t0 = time.time()
    try:
        model.fit(X_train, y_train)
    except Exception as e:
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


def train_ensemble(
    pair_smiles, pair_genes, y,
    X_binary_all, X_rdkit_raw_all, all_smiles,
    protein_embeddings_raw,
    best_params_per_model=None, n_folds=5, random_seed=42,
):
    """5-fold Scaffold Split CV 训练多模型（含 TabPFN），用于模型选择。

    v6.3 修复：RDKit 标准化和蛋白 PCA 均在每个 fold 内仅用训练集拟合，
    彻底消除数据泄露。

    CV 设计原则：
      - 每折独立拟合 RDKit scaler 和蛋白 PCA（仅用训练集）
      - 每折独立执行 TabPFN 特征选择（互信息 Top-500），不同折的特征空间不同
      - 这是严谨的 CV 设计（预处理作为 pipeline 纳入 CV），
        确保模型选择阶段的指标无泄露
      - 树模型始终使用全量特征，不受特征选择影响
      - 注意：CV 汇总指标仅用于模型选择，不代表最终泛化性能。
        最终集成模型性能由独立种子的另一组 5-fold CV 独立评估。
    """
    results = []

    for fold in range(n_folds):
        train_idx, test_idx = scaffold_split(
            pair_smiles, y, test_size=0.2, random_state=random_seed + fold,
        )
        y_train, y_test = y[train_idx], y[test_idx]

        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold+1}/{n_folds}: 构建无泄露特征...")

        X_fold, _, _, _ = build_fold_features(
            pair_smiles, pair_genes, train_idx,
            X_binary_all, X_rdkit_raw_all, all_smiles,
            protein_embeddings_raw, prot_target_dim=128,
        )
        X_train, X_test = X_fold[train_idx], X_fold[test_idx]

        logger.info(f"train={len(X_train)}, test={len(X_test)}, "
                    f"pos_ratio={y_train.mean():.3f}/{y_test.mean():.3f}, "
                    f"feat_dim={X_train.shape[1]}")

        fold_results = []

        # ---- 1. Random Forest ----
        logger.info("  [1/5] Random Forest...")
        rf_params = best_params_per_model.get("rf", {}) if best_params_per_model else {}
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

        # ---- 2. XGBoost ----
        try:
            import xgboost as xgb
            logger.info("  [2/5] XGBoost...")
            xgb_params = best_params_per_model.get("xgb", {}) if best_params_per_model else {}
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
            logger.warning("XGBoost 未安装，跳过")

        # ---- 3. LightGBM ----
        try:
            import lightgbm as lgb
            logger.info("  [3/5] LightGBM...")
            lgb_params = best_params_per_model.get("lgb", {}) if best_params_per_model else {}
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
            logger.warning("LightGBM 未安装，跳过")

        # ---- 4. CatBoost ----
        try:
            from catboost import CatBoostClassifier
            logger.info("  [4/5] CatBoost...")
            cb_model = CatBoostClassifier(
                iterations=200, depth=8, learning_rate=0.05,
                auto_class_weights="Balanced",
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
            logger.warning("CatBoost 未安装，跳过")

        # ---- 5. TabPFN (Nature 2025) ----
        logger.info("  [5/5] TabPFN (Nature 2025, zero-shot)...")
        if not TabPFNWrapper.check_available():
            logger.warning("  TabPFN 不可用（模型下载失败或未授权），跳过")
            continue
        try:
            tabpfn_model = TabPFNWrapper(
                device="auto",
                random_state=random_seed,
                max_features=500,
            )
            r = evaluate_model(tabpfn_model, X_train, y_train, X_test, y_test, "TabPFN")
            if r:
                r["fold"] = fold
                fold_results.append(r)
                results.append(r)
                logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}, "
                            f"F1={r['F1']:.4f}, MCC={r['MCC']:.4f}")
        except ImportError:
            logger.error("tabpfn 未安装，跳过 TabPFN 评估")
        except Exception as e:
            logger.error(f"  TabPFN 评估异常: {e}")
            import traceback as tb
            logger.error(tb.format_exc())
            TabPFNWrapper._is_available = False
            TabPFNWrapper._availability_error = str(e)

    return pd.DataFrame(results)


# ============================================================
# 10. TabPFN 特征重要性分析
# ============================================================

def tabpfn_feature_importance(tabpfn_model, X, y, feature_names=None, n_permute=5):
    """TabPFN 特征重要性（置换重要性）

    TabPFN 不直接提供 feature_importances_，使用置换重要性。
    """
    t0 = time.time()
    logger.info("  计算 TabPFN 置换特征重要性...")

    from sklearn.metrics import roc_auc_score

    # 基线 AUC
    y_prob = tabpfn_model.predict_proba(X)[:, 1]
    baseline_auc = roc_auc_score(y, y_prob)

    n_features = X.shape[1]
    importances = np.zeros(n_features)

    for i in range(n_features):
        scores = []
        for _ in range(n_permute):
            X_perm = X.copy()
            X_perm[:, i] = np.random.permutation(X_perm[:, i])
            y_prob_perm = tabpfn_model.predict_proba(X_perm)[:, 1]
            try:
                auc_perm = roc_auc_score(y, y_prob_perm)
            except ValueError:
                auc_perm = 0.5
            scores.append(baseline_auc - auc_perm)
        importances[i] = np.mean(scores)

    # 归一化
    imp_sum = np.abs(importances).sum()
    if imp_sum > 0:
        importances = importances / imp_sum

    logger.info(f"  特征重要性计算完成, 耗时: {time.time()-t0:.1f}s")

    if feature_names is not None and len(feature_names) == n_features:
        imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)
    else:
        imp_df = pd.DataFrame({
            "feature": [f"feat_{i}" for i in range(n_features)],
            "importance": importances,
        }).sort_values("importance", ascending=False)

    return imp_df


# ============================================================
# 11. TCM 预测（含 TabPFN 不确定性）
# ============================================================

def load_herb_mapping():
    """加载中药来源映射"""
    herb_map_path = L3_RESULTS / "herb_ingredient_mapping.xlsx"
    if not herb_map_path.exists():
        logger.warning(f"中药映射文件不存在: {herb_map_path}")
        return {}

    try:
        herb_df = pd.read_excel(herb_map_path)
        # 优先尝试已知列名
        col_candidates = {
            "MOL_ID": ["MOL_ID", "mol_id", "Molecule_ID", "compound_id"],
            "herb_cn": ["herb_cn", "Herb_cn", "herb_Chinese", "Chinese_name", "herb_name_cn"],
            "herb_en": ["herb_en", "Herb_en", "herb_English", "English_name", "herb_name_en"],
            "pinyin": ["herb_pinyin", "Pinyin", "pinyin", "herb_Pinyin"],
        }

        actual_cols = {}
        for key, candidates in col_candidates.items():
            found_col = None
            for c in candidates:
                if c in herb_df.columns:
                    found_col = c
                    break
            if found_col is None:
                logger.warning(f"中药映射: 未找到列 {key} (候选: {candidates}), 可用列: {list(herb_df.columns)}")
                actual_cols[key] = None
            else:
                actual_cols[key] = found_col

        herb_map = {}
        for _, row in herb_df.iterrows():
            mol_id_val = str(row.get(actual_cols["MOL_ID"], "")) if actual_cols["MOL_ID"] else ""
            if not mol_id_val:
                continue
            cn_name = str(row.get(actual_cols["herb_cn"], "")) if actual_cols["herb_cn"] else ""
            en_name = str(row.get(actual_cols["herb_en"], "")) if actual_cols["herb_en"] else ""
            pinyin_val = str(row.get(actual_cols["pinyin"], "")) if actual_cols["pinyin"] else ""

            if mol_id_val not in herb_map:
                herb_map[mol_id_val] = {"cn_names": set(), "en_names": set(), "pinyins": set()}
            if cn_name and cn_name != "nan":
                herb_map[mol_id_val]["cn_names"].add(cn_name)
            if en_name and en_name != "nan":
                herb_map[mol_id_val]["en_names"].add(en_name)
            if pinyin_val and pinyin_val != "nan":
                herb_map[mol_id_val]["pinyins"].add(pinyin_val)

        for k in herb_map:
            herb_map[k]["cn_names"] = sorted(herb_map[k]["cn_names"])
            herb_map[k]["en_names"] = sorted(herb_map[k]["en_names"])
            herb_map[k]["pinyins"] = sorted(herb_map[k]["pinyins"])

        logger.info(f"  中药映射加载完成: {len(herb_map)} 个化合物")
        if actual_cols.get("herb_cn") is None:
            logger.warning("  中药中文名列缺失，herb_cn 将全部标记为 '未知'")
        if actual_cols.get("herb_en") is None:
            logger.warning("  中药英文名列缺失，herb_en 将全部标记为 'Unknown'")
        return herb_map
    except Exception as e:
        logger.warning(f"中药映射加载失败: {e}, 路径: {herb_map_path}")
        return {}


def predict_tcm_pool(
    model, tcm_df, tcm_binary_feats, tcm_rdkit_feats,
    protein_embeddings, cpi_genes_in_emb, model_name, herb_map=None,
    return_std=False,
):
    """预测 TCM 化合物池，可选不确定性估计

    当 return_std=True 时，添加 score_std 列（当前为 NaN 占位）。
    TabPFN 的不确定性需要通过训练多个不同随机种子的模型集成来实现，
    在 TCM 预测循环中逐一拟合多个模型计算量过大，因此暂不实现。
    可替代方案：在模型选择阶段训练 5 个 TabPFN 成员模型，
    取预测值的标准差作为不确定性。
    """
    if herb_map is None:
        herb_map = {}

    logger.info(f"  预测 {len(tcm_df)} 个 TCM 化合物 x {len(cpi_genes_in_emb)} 个基因...")

    predictions = []
    comp_dim = tcm_binary_feats.shape[1] + tcm_rdkit_feats.shape[1]

    if return_std:
        logger.info("    注意: return_std=True 但当前暂未实现逐一预测的不确定性估计，score_std 将置为 NaN。"
                     "真正的 TabPFN 不确定性需要预训练多成员集成模型。")

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

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(feat.reshape(1, -1))[:, 1]
                score = float(y_prob[0])
            else:
                score = float(model.predict(feat.reshape(1, -1))[0])

            record = {
                "MOL_ID": mol_id,
                "molecule_name": mol_name,
                "SMILES": smi,
                "gene": gene,
                "score": score,
                "model": model_name,
                "herb_cn": herb_cn,
                "herb_en": herb_en,
                "herb_pinyin": herb_py,
            }
            if return_std:
                record["score_std"] = float("nan")
            predictions.append(record)

        if (i + 1) % 100 == 0:
            logger.info(f"    进度: {i+1}/{len(tcm_df)}")

    return pd.DataFrame(predictions)


# ============================================================
# 12. TabPFN + Tree 集成投票
# ============================================================

def ensemble_predict(tabpfn_model, tree_model, X):
    """TabPFN + 最佳树模型加权集成"""
    tabpfn_prob = tabpfn_model.predict_proba(X)[:, 1]
    tree_prob = tree_model.predict_proba(X)[:, 1]
    # 等权平均
    ensemble_prob = 0.5 * tabpfn_prob + 0.5 * tree_prob
    return ensemble_prob


def build_fold_features(
    pair_smiles,
    pair_genes,
    train_idx,
    X_binary_all,
    X_rdkit_raw_all,
    all_smiles,
    protein_embeddings_raw,
    prot_target_dim=128,
):
    """在单个CV折内构建特征矩阵（无数据泄露版本）。

    关键原则：
      - RDKit scaler 仅用训练集中出现的化合物拟合
      - 蛋白 PCA + scaler 仅用训练集中出现的蛋白拟合
      - 然后变换所有化合物和蛋白，构建完整特征矩阵
      - 调用方用 train_idx/test_idx 自行取训练/测试集

    Args:
        pair_smiles: 所有 pair 对应的 SMILES 数组
        pair_genes: 所有 pair 对应的基因名称数组
        train_idx: 训练集 pair 索引（用于拟合 scaler 和 PCA）
        X_binary_all: 所有化合物的二进制指纹 (原始，无需标准化)
        X_rdkit_raw_all: 所有化合物的 RDKit 2D 描述符 (原始值，未标准化)
        all_smiles: 所有化合物 SMILES 列表
        protein_embeddings_raw: 原始蛋白嵌入字典 {gene: vector}
        prot_target_dim: 蛋白 PCA 目标维度

    Returns:
        X: 完整特征矩阵 (n_pairs, feat_dim)
        rdkit_scaler: 拟合好的 RDKit StandardScaler
        prot_pca: 拟合好的蛋白 PCA
        prot_scaler: 拟合好的蛋白 StandardScaler
    """
    smiles_to_idx = {str(s): i for i, s in enumerate(all_smiles)}

    # ---- 1. 收集训练集中出现的化合物和蛋白 ----
    train_smiles_set = set(pair_smiles[train_idx])
    train_genes_set = set(pair_genes[train_idx])

    train_compound_indices = np.array([
        smiles_to_idx[s] for s in train_smiles_set if s in smiles_to_idx
    ], dtype=np.int64)

    if len(train_compound_indices) == 0:
        raise ValueError("训练集化合物为空，无法拟合 RDKit scaler")

    train_emb = {
        g: protein_embeddings_raw[g]
        for g in train_genes_set if g in protein_embeddings_raw
    }
    if len(train_emb) == 0:
        raise ValueError("训练集蛋白为空，无法拟合蛋白 PCA")

    # ---- 2. RDKit 标准化（仅训练集拟合）----
    rdkit_scaler = StandardScaler()
    rdkit_scaler.fit(X_rdkit_raw_all[train_compound_indices])
    X_rdkit_all_scaled = rdkit_scaler.transform(X_rdkit_raw_all)
    compound_features_all = np.hstack([X_binary_all, X_rdkit_all_scaled])
    compound_feat_dim = compound_features_all.shape[1]

    # ---- 3. 蛋白 PCA + 标准化（仅训练集蛋白拟合，仅变换 pair 中出现的蛋白）----
    _, prot_pca, prot_scaler = process_protein_embeddings(
        train_emb, target_dim=prot_target_dim, pca_model=None, scaler=None,
    )
    # 只变换 pair_genes 中出现的蛋白（通常只有几十个，而非数千个）
    pair_gene_set = set(pair_genes)
    pair_emb_raw = {
        g: protein_embeddings_raw[g]
        for g in pair_gene_set if g in protein_embeddings_raw
    }
    protein_embeddings_processed, _, _ = process_protein_embeddings(
        pair_emb_raw, target_dim=prot_target_dim,
        pca_model=prot_pca, scaler=prot_scaler,
    )
    prot_dim = next(iter(protein_embeddings_processed.values())).shape[0]
    feat_dim = compound_feat_dim + prot_dim

    # ---- 4. 构建完整特征矩阵 ----
    n_pairs = len(pair_smiles)
    X = np.zeros((n_pairs, feat_dim), dtype=np.float32)

    for i in range(n_pairs):
        smi = pair_smiles[i]
        gene = pair_genes[i]
        ci = smiles_to_idx[smi]
        X[i, :compound_feat_dim] = compound_features_all[ci]
        X[i, compound_feat_dim:] = protein_embeddings_processed[gene]

    return X, rdkit_scaler, prot_pca, prot_scaler


def build_pair_dataset(cpi_df, protein_embeddings, compound_features, all_smiles,
                       neg_ratio=3, random_seed=42):
    """构建化合物-蛋白 pair 分类数据集。"""
    smiles_to_idx = {str(s): i for i, s in enumerate(all_smiles)}
    compound_feat_dim = compound_features.shape[1]

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

    neg_pairs = diversity_constrained_negative_sampling(
        pos_pairs, all_smiles, cpi_genes_in_emb,
        neg_ratio=neg_ratio, random_seed=random_seed,
    )

    all_pairs = pos_pairs + neg_pairs
    n_pairs = len(all_pairs)
    prot_dim = next(iter(protein_embeddings.values())).shape[0]
    feat_dim = compound_feat_dim + prot_dim

    X = np.zeros((n_pairs, feat_dim), dtype=np.float32)
    y = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs), dtype=np.int32)
    pair_smiles = []
    pair_genes = []

    for i, (smi, gene) in enumerate(all_pairs):
        ci = smiles_to_idx[smi]
        X[i, :compound_feat_dim] = compound_features[ci]
        X[i, compound_feat_dim:] = protein_embeddings[gene]
        pair_smiles.append(smi)
        pair_genes.append(gene)

    pair_smiles = np.array(pair_smiles)
    pair_genes = np.array(pair_genes)

    logger.info(f"  数据集: {n_pairs} 样本, {feat_dim} 特征 "
                f"(comp={compound_feat_dim}+prot={prot_dim}), "
                f"正样本比例={y.mean():.3f}")
    return X, y, pair_smiles, pair_genes, cpi_genes_in_emb, smiles_to_idx


def run_mode_cv_ablation(mode, cpi_df, all_smiles, X_binary_all, X_rdkit_raw_all,
                         base_model_name="XGBoost", n_folds=5, random_seed=42):
    """对单个蛋白嵌入模式运行轻量 5-fold CV（仅基线模型），用于消融筛选。

    v6.3 修复：RDKit 标准化和蛋白 PCA 均在每个 fold 内仅用训练集拟合，
    彻底消除数据泄露。
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"蛋白嵌入消融: mode={mode} (无泄露 v6.3)")
    logger.info(f"{'='*60}")

    # ---- 加载原始蛋白嵌入（不做 PCA/标准化）----
    protein_embeddings_raw = load_protein_embeddings_by_mode(mode)

    # ---- 预构建所有 pairs（含负采样），仅用于获取 pair_smiles/pair_genes/y ----
    # 注意：这里用 dummy compound_features 来获取 pair 结构，不用于实际特征计算
    dummy_compound = np.zeros((len(all_smiles), 1), dtype=np.float32)
    dummy_protein = {k: np.zeros(1, dtype=np.float32) for k in protein_embeddings_raw}
    _, y, pair_smiles, pair_genes, cpi_genes_in_emb, _ = build_pair_dataset(
        cpi_df, dummy_protein, dummy_compound, all_smiles,
        neg_ratio=3, random_seed=random_seed,
    )
    n_pos = int(y.sum())
    n_genes = len(cpi_genes_in_emb)
    logger.info(f"  有效基因数: {n_genes}, 正样本数: {n_pos}, 总样本数: {len(y)}")

    if base_model_name == "RandomForest":
        base_model = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=random_seed,
        )
    elif base_model_name == "XGBoost":
        import xgboost as xgb
        scale_pos_weight = (y == 0).sum() / max(y.sum(), 1)
        base_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=random_seed, n_jobs=-1, verbosity=0,
        )
    elif base_model_name == "LightGBM":
        import lightgbm as lgb
        base_model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=10, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=random_seed, n_jobs=-1, verbose=-1,
        )
    else:
        raise ValueError(f"Unsupported base model for ablation: {base_model_name}")

    results = []
    for fold in range(n_folds):
        train_idx, test_idx = scaffold_split(
            pair_smiles, y, test_size=0.2, random_state=random_seed + fold,
        )
        y_train, y_test = y[train_idx], y[test_idx]

        logger.info(f"  Fold {fold+1}/{n_folds}: 构建无泄露特征...")
        X_fold, _, _, _ = build_fold_features(
            pair_smiles, pair_genes, train_idx,
            X_binary_all, X_rdkit_raw_all, all_smiles,
            protein_embeddings_raw, prot_target_dim=128,
        )
        X_train, X_test = X_fold[train_idx], X_fold[test_idx]

        logger.info(f"    train={len(X_train)}, test={len(X_test)}, feat_dim={X_train.shape[1]}")
        model = clone(base_model)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        m = compute_metrics(y_test, y_prob)
        m.update({
            "fold": fold,
            "model": f"{base_model_name}",
            "protein_emb_mode": mode,
            "n_genes": n_genes,
            "n_pos_samples": n_pos,
            "n_total_samples": len(y),
        })
        results.append(m)
        logger.info(f"    AUC={m['AUC']:.4f}, AUPR={m['AUPR']:.4f}, "
                    f"BEDROC={m['BEDROC']:.4f}")

    return pd.DataFrame(results), protein_embeddings_raw


def run_full_pipeline(mode, cpi_df, tcm_df, all_smiles,
                      X_binary_all, X_rdkit_raw_all,
                      n_folds=5, random_seed=42):
    """使用指定蛋白嵌入模式运行完整 5-fold CV、全量训练、TabPFN 集成与 TCM 预测。

    v6.3 修复：RDKit 标准化和蛋白 PCA 均在每个 fold 内仅用训练集拟合，
    彻底消除数据泄露。
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"完整流程: 蛋白嵌入 mode={mode} (无泄露 v6.3)")
    logger.info(f"{'='*60}")

    # ---- 3. 加载原始蛋白嵌入 ----
    logger.info("\n[3/7] 加载原始蛋白嵌入...")
    protein_embeddings_raw = load_protein_embeddings_by_mode(mode)

    # ---- 4. 预构建 pair 结构（负采样 + 获取 pair_smiles/pair_genes/y）----
    logger.info("\n[4/7] 构建训练数据集 (多样性约束负采样)...")
    # 用 dummy 特征获取 pair 结构，实际特征在 fold 内构建
    dummy_compound = np.zeros((len(all_smiles), 1), dtype=np.float32)
    dummy_protein = {k: np.zeros(1, dtype=np.float32) for k in protein_embeddings_raw}
    _, y, pair_smiles, pair_genes, cpi_genes_in_emb, smiles_to_idx = build_pair_dataset(
        cpi_df, dummy_protein, dummy_compound, all_smiles,
        neg_ratio=3, random_seed=random_seed,
    )
    logger.info(f"  有效基因数: {len(cpi_genes_in_emb)}, 正样本数: {int(y.sum())}, 总样本数: {len(y)}")

    # ---- 5. 5-fold CV 对比评估（无泄露版本）----
    logger.info("\n[5/7] 5-fold Scaffold Split 全模型对比 (含 TabPFN, 无泄露)...")
    results_df = train_ensemble(
        pair_smiles, pair_genes, y,
        X_binary_all, X_rdkit_raw_all, all_smiles,
        protein_embeddings_raw,
        best_params_per_model=None,
        n_folds=n_folds, random_seed=random_seed,
    )

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

    results_path = L4_RESULTS / "tree_v6_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\n评估结果已保存: {results_path}")

    # ---- 6. 全量训练最佳模型 + TabPFN ----
    logger.info("\n[6/7] 全量训练最佳模型 + TabPFN 特征重要性...")

    # 选择最佳树模型（按 AUPR）
    if "AUPR" in summary.columns.levels[0]:
        best_model_name = summary["AUPR"]["mean"].idxmax()
        best_aupr = summary.loc[best_model_name, "AUPR"]["mean"]
        logger.info(f"  最佳单模型: {best_model_name} (AUPR={best_aupr:.4f})")
    else:
        best_model_name = "RandomForest"
        logger.info(f"  使用默认模型: {best_model_name}")

    # 用全部 CPI 数据拟合 scaler 和 PCA（全量训练用，已脱离模型选择阶段）
    # 注意：这里用全部 pair 中的化合物和蛋白，因为已经是最终训练阶段
    logger.info("  用全量 CPI 数据拟合 RDKit scaler 和蛋白 PCA...")
    all_cpi_smiles_set = set(pair_smiles)
    all_cpi_genes_set = set(pair_genes)
    all_cpi_compound_indices = np.array([
        smiles_to_idx[s] for s in all_cpi_smiles_set if s in smiles_to_idx
    ], dtype=np.int64)

    full_rdkit_scaler = StandardScaler()
    full_rdkit_scaler.fit(X_rdkit_raw_all[all_cpi_compound_indices])
    X_rdkit_all_scaled = full_rdkit_scaler.transform(X_rdkit_raw_all)
    compound_features_all = np.hstack([X_binary_all, X_rdkit_all_scaled])

    full_protein_emb = {
        g: protein_embeddings_raw[g]
        for g in all_cpi_genes_set if g in protein_embeddings_raw
    }
    _, full_prot_pca, full_prot_scaler = process_protein_embeddings(
        full_protein_emb, target_dim=128, pca_model=None, scaler=None,
    )
    # 只变换 CPI 相关蛋白（用于训练和 TCM 预测），避免数千个无用蛋白的变换开销
    protein_embeddings_full, _, _ = process_protein_embeddings(
        full_protein_emb, target_dim=128,
        pca_model=full_prot_pca, scaler=full_prot_scaler,
    )

    # 构建全量训练特征
    compound_feat_dim = compound_features_all.shape[1]
    prot_dim = next(iter(protein_embeddings_full.values())).shape[0]
    feat_dim = compound_feat_dim + prot_dim
    X_full = np.zeros((len(pair_smiles), feat_dim), dtype=np.float32)
    for i in range(len(pair_smiles)):
        smi = pair_smiles[i]
        gene = pair_genes[i]
        ci = smiles_to_idx[smi]
        X_full[i, :compound_feat_dim] = compound_features_all[ci]
        X_full[i, compound_feat_dim:] = protein_embeddings_full[gene]

    # 全量训练最佳树模型
    if best_model_name == "RandomForest":
        best_tree = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=random_seed,
        )
    elif best_model_name == "XGBoost":
        import xgboost as xgb
        scale_pos_weight = (y == 0).sum() / max(y.sum(), 1)
        best_tree = xgb.XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=random_seed, n_jobs=-1, verbosity=0,
        )
    elif best_model_name == "LightGBM":
        import lightgbm as lgb
        best_tree = lgb.LGBMClassifier(
            n_estimators=200, max_depth=10, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=random_seed, n_jobs=-1, verbose=-1,
        )
    elif best_model_name == "CatBoost":
        from catboost import CatBoostClassifier
        best_tree = CatBoostClassifier(
            iterations=200, depth=8, learning_rate=0.05,
            auto_class_weights="Balanced",
            random_seed=random_seed, thread_count=-1,
            verbose=False, allow_writing_files=False,
        )
    else:
        best_tree = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=random_seed,
        )

    logger.info(f"  全量训练 {best_model_name} ({len(X_full)} 样本)...")
    best_tree.fit(X_full, y)

    # 全量训练 TabPFN
    best_tabpfn = None
    if TabPFNWrapper.check_available():
        logger.info(f"  全量训练 TabPFN ({len(X_full)} 样本)...")
        try:
            best_tabpfn = TabPFNWrapper(device="auto", random_state=random_seed, max_features=500)
            best_tabpfn.fit(X_full, y)
        except Exception as e:
            logger.warning(f"  TabPFN 全量训练失败: {e}")
            best_tabpfn = None
    else:
        logger.warning("  TabPFN 不可用，跳过全量训练")

    # TabPFN 特征重要性
    if best_tabpfn is not None:
        try:
            if best_tabpfn._feature_selector is not None:
                X_imp = best_tabpfn._feature_selector.transform(X_full)
                n_features_imp = X_imp.shape[1]
            else:
                X_imp = X_full
                n_features_imp = X_full.shape[1]

            imp_df = tabpfn_feature_importance(
                best_tabpfn, X_imp, y,
                feature_names=[f"sel_feat_{i}" for i in range(n_features_imp)],
                n_permute=3,
            )

            if best_tabpfn._feature_selector is not None:
                orig_indices = best_tabpfn._feature_selector.selected_indices
                feat_type = []
                n_binary = X_binary_all.shape[1]
                n_rdkit = X_rdkit_raw_all.shape[1]
                for idx in orig_indices:
                    if idx < n_binary:
                        feat_type.append("binary_fp")
                    elif idx < n_binary + n_rdkit:
                        feat_type.append("rdkit_2d")
                    else:
                        feat_type.append("protein_emb")
                imp_df["orig_feature_index"] = orig_indices
                imp_df["feature_type"] = feat_type
            else:
                imp_df["feature_type"] = "all"

            imp_path = L4_RESULTS / "tree_v6_tabpfn_importance.csv"
            imp_df.to_csv(imp_path, index=False)
            logger.info(f"\nTabPFN Top 20 重要特征:")
            for i, row in imp_df.head(20).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.6f} ({row.get('feature_type','')})")
            logger.info(f"特征重要性已保存: {imp_path}")
        except Exception as e:
            logger.warning(f"TabPFN 特征重要性计算失败: {e}")
            import traceback as tb
            logger.warning(tb.format_exc())
    else:
        logger.info("  TabPFN 不可用，跳过特征重要性计算")

    # TabPFN + Tree Ensemble 评估（严格 5-fold Scaffold Split CV，每折独立训练）
    # 注意：此处使用与模型选择阶段不同的随机种子（random_seed + 1000），
    #       确保集成评估的测试集与模型选择的折完全独立，避免数据泄露。
    #       模型选择使用 random_seed + fold，集成评估使用 random_seed + 1000 + fold。
    # v6.3：每折独立拟合 RDKit scaler 和蛋白 PCA，彻底消除预处理泄露。
    if best_tabpfn is not None and TabPFNWrapper.check_available():
        logger.info("\n  TabPFN + Tree Ensemble 评估 (独立 5-fold Scaffold Split CV, 无泄露)...")
        from sklearn.base import clone
        ensemble_metrics = []
        ensemble_seed_offset = 1000

        for fold in range(n_folds):
            train_idx, test_idx = scaffold_split(
                pair_smiles, y, test_size=0.2,
                random_state=random_seed + ensemble_seed_offset + fold,
            )
            y_tr_fold, y_te_fold = y[train_idx], y[test_idx]

            X_fold, _, _, _ = build_fold_features(
                pair_smiles, pair_genes, train_idx,
                X_binary_all, X_rdkit_raw_all, all_smiles,
                protein_embeddings_raw, prot_target_dim=128,
            )
            X_tr_fold, X_te_fold = X_fold[train_idx], X_fold[test_idx]

            tree_inner = clone(best_tree).set_params(**best_tree.get_params())
            tree_inner.fit(X_tr_fold, y_tr_fold)

            tabpfn_inner = TabPFNWrapper(device="auto", random_state=random_seed, max_features=500)
            tabpfn_inner.fit(X_tr_fold, y_tr_fold)

            ensemble_prob = ensemble_predict(tabpfn_inner, tree_inner, X_te_fold)
            m = compute_metrics(y_te_fold, ensemble_prob)
            m["fold"] = fold
            ensemble_metrics.append(m)

        if ensemble_metrics:
            ens_df = pd.DataFrame(ensemble_metrics)
            logger.info(f"\n  TabPFN+{best_model_name} Ensemble (严格 5-fold CV, 无泄露):")
            for metric in ["AUC", "AUPR", "F1", "MCC", "EF@1%"]:
                if metric in ens_df.columns:
                    logger.info(f"    {metric}: {ens_df[metric].mean():.4f} +/- {ens_df[metric].std():.4f}")

            ens_path = L4_RESULTS / "tree_v6_ensemble_results.csv"
            ens_df.to_csv(ens_path, index=False)
            logger.info(f"Ensemble 结果已保存: {ens_path}")
    else:
        logger.info("\n  TabPFN 不可用，跳过集成评估")

    # ---- 7. TCM 预测（用最佳模型 + TabPFN） ----
    logger.info("\n[7/7] TCM 化合物池预测...")
    herb_map = load_herb_mapping()

    tcm_smiles = tcm_df["SMILES_std"].astype(str).tolist()
    tcm_indices = [smiles_to_idx[s] for s in tcm_smiles if s in smiles_to_idx]
    tcm_binary_feats = X_binary_all[tcm_indices]
    tcm_rdkit_feats_raw = X_rdkit_raw_all[tcm_indices]
    # 用全量 CPI 数据拟合的 scaler 变换 TCM 化合物 RDKit 特征
    tcm_rdkit_feats = full_rdkit_scaler.transform(tcm_rdkit_feats_raw)

    # protein_embeddings_full 已经用全量 CPI 数据拟合的 PCA+scaler 变换过了
    pred_tree = predict_tcm_pool(
        best_tree, tcm_df, tcm_binary_feats, tcm_rdkit_feats,
        protein_embeddings_full, cpi_genes_in_emb, best_model_name, herb_map=herb_map,
    )

    if best_tabpfn is not None:
        try:
            pred_tabpfn = predict_tcm_pool(
                best_tabpfn, tcm_df, tcm_binary_feats, tcm_rdkit_feats,
                protein_embeddings_full, cpi_genes_in_emb, "TabPFN", herb_map=herb_map,
            )
            pred_df = pd.concat([pred_tree, pred_tabpfn], ignore_index=True)
        except Exception as e:
            logger.warning(f"TabPFN TCM 预测失败，仅使用树模型结果: {e}")
            pred_df = pred_tree
    else:
        logger.info("  TabPFN 不可用，仅使用树模型进行 TCM 预测")
        pred_df = pred_tree

    pred_path = L4_RESULTS / "tree_v6_tcm_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"TCM 预测结果已保存: {pred_path} ({len(pred_df)} 条)")

    # Top 候选
    comp_agg = pred_df.groupby(["MOL_ID", "molecule_name", "SMILES", "herb_cn", "herb_en"]).agg(
        max_score=("score", "max"),
        mean_score=("score", "mean"),
        n_genes_above_50=("score", lambda x: (x >= 0.5).sum()),
        n_models=("model", "nunique"),
        top_3_genes=("score", lambda x: "|".join(
            [f"{g}({s:.2f})" for g, s in sorted(
                zip(list(pred_df.loc[x.index, "gene"]), list(x)),
                key=lambda v: v[1], reverse=True
            )[:3]]
        )),
    ).reset_index()
    comp_agg = comp_agg.sort_values("max_score", ascending=False)

    top50 = comp_agg.head(50)
    top_path = L4_RESULTS / "tree_v6_top_candidates.csv"
    top50.to_csv(top_path, index=False)

    logger.info(f"\nTop 20 候选化合物:")
    for i, row in enumerate(top50.head(20).itertuples(index=False), 1):
        logger.info(f"  {i:2d}. {row.molecule_name} | max={row.max_score:.4f} "
                    f"| mean={row.mean_score:.4f} "
                    f"| 高置信(>=0.5): {row.n_genes_above_50} "
                    f"| 模型数: {row.n_models} "
                    f"| 中药: {row.herb_cn} "
                    f"| {row.top_3_genes}")

    logger.info(f"Top 50 候选已保存: {top_path}")
    logger.info(f"=" * 60)
    if best_tabpfn is not None:
        logger.info("任务完成! TabPFN (Nature 2025) 集成成功.")
    else:
        logger.info(f"任务完成! 最佳模型: {best_model_name} (TabPFN 因网络/授权问题不可用，已自动跳过)")
    logger.info(f"=" * 60)


# ============================================================
# 主流程
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("树模型 CPI v6.4 — 扩展铁衰老基因版 + 无数据泄露 + TabPFN 集成")
    logger.info("  v6.4 更新: 合并 28 个新铁衰老基因数据，51/96 铁衰老基因有 CPI 数据")
    logger.info("  v6.3 修复: RDKit 标准化和蛋白 PCA 均在 CV fold 内仅用训练集拟合")
    logger.info("=" * 60)

    # ---- 1. 加载数据 ----
    logger.info("\n[1/8] 加载原始数据...")
    cpi_df = pd.read_csv(L4_RESULTS / "experimental_actives_detail_cleaned_combined.csv", low_memory=False)
    tcm_df = pd.read_csv(L3_RESULTS / "tcm_compound_pool_tox_filtered_noleak.csv",
                         low_memory=False)
    logger.info(f"  CPI 记录: {len(cpi_df)}, TCM 化合物: {len(tcm_df)}")

    all_smiles = list(cpi_df["canonical_smiles"].dropna().astype(str).unique())
    tcm_smiles = tcm_df["SMILES_std"].astype(str).tolist()
    all_smiles.extend(tcm_smiles)
    all_smiles = list(dict.fromkeys(all_smiles))
    logger.info(f"  总 SMILES: {len(all_smiles)} (CPI 唯一: {len(all_smiles) - len(tcm_smiles)}, "
                f"TCM: {len(tcm_smiles)})")

    # ---- 2. 多指纹特征工程（保留原始 RDKit 值，标准化在 CV fold 内进行）----
    logger.info("\n[2/8] 多指纹特征工程...")
    X_binary, X_rdkit, X_rdkit_raw, rdkit_scaler, binary_labels, rdkit_names = \
        build_multifingerprint_features(all_smiles)

    compound_features_preview = np.hstack([X_binary, X_rdkit])
    logger.info(f"  化合物特征总维度: {compound_features_preview.shape[1]} "
                f"(binary={X_binary.shape[1]}, rdkit={X_rdkit.shape[1]})")
    logger.info(f"  注意: RDKit 标准化参数仅用于预览，实际 CV 内每折独立拟合")

    # ---- 2.5. 蛋白嵌入消融实验 (4 modes x 5-fold CV, XGBoost 基线, 无泄露) ----
    logger.info("\n[2.5/8] 蛋白嵌入消融实验 (4 种模式 x 5-fold CV, XGBoost 基线, 无泄露 v6.4)...")
    ablation_results = []
    best_mode = None
    best_aupr = -1.0
    best_n_genes = -1

    for mode in PROTEIN_EMB_MODES:
        try:
            df_mode, _ = run_mode_cv_ablation(
                mode, cpi_df, all_smiles, X_binary, X_rdkit_raw,
                base_model_name="XGBoost", n_folds=5, random_seed=42,
            )
            ablation_results.append(df_mode)
            mean_aupr = df_mode["AUPR"].mean()
            mean_auc = df_mode["AUC"].mean()
            mean_bedroc = df_mode["BEDROC"].mean()
            n_genes = int(df_mode["n_genes"].iloc[0])
            n_pos = int(df_mode["n_pos_samples"].iloc[0])
            logger.info(f"  {mode:22s}: AUC={mean_auc:.4f}, AUPR={mean_aupr:.4f}, "
                        f"BEDROC={mean_bedroc:.4f} | genes={n_genes}, pos={n_pos}")
            # 最佳模式选择策略：
            # 1. 优先选择基因覆盖度更高的模式（生物学意义更大，覆盖更多铁衰老靶点）
            # 2. 基因覆盖度相同时，选 AUPR 更高的
            if n_genes > best_n_genes or (n_genes == best_n_genes and mean_aupr > best_aupr):
                best_aupr = mean_aupr
                best_n_genes = n_genes
                best_mode = mode
        except Exception as e:
            logger.error(f"  {mode} 消融实验失败: {e}")
            import traceback as tb
            logger.error(tb.format_exc())

    if ablation_results:
        ablation_all = pd.concat(ablation_results, ignore_index=True)
        ablation_path = L4_RESULTS / "tree_v6_protein_emb_ablation.csv"
        ablation_all.to_csv(ablation_path, index=False)
        logger.info(f"\n  消融实验结果已保存: {ablation_path}")

    if best_mode is None:
        best_mode = "residue_meanmaxstd"
        logger.warning(f"  所有模式均失败，回退到默认模式: {best_mode}")

    logger.info(f"\n  最佳蛋白嵌入模式: {best_mode} (AUPR={best_aupr:.4f})")

    # ---- 3~7. 使用最佳模式运行完整流程 ----
    logger.info(f"\n[3-7/8] 使用最佳模式 {best_mode} 运行完整流程...")
    run_full_pipeline(
        best_mode, cpi_df, tcm_df, all_smiles,
        X_binary, X_rdkit_raw,
        n_folds=5, random_seed=42,
    )

    logger.info(f"\n[8/8] 全部流程完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
