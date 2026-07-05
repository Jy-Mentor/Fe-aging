#!/usr/bin/env python
"""
树模型 CPI 筛选 — 仅 XGBoost
==============================

基于化合物多指纹特征与蛋白 ESM-2 嵌入，训练 XGBoost 分类模型进行
化合物-蛋白相互作用（CPI）预测。采用 5-fold Scaffold Split 进行模型
选择，并在全量 CPI 数据上训练最终模型后对 TCM 化合物池打分。

数据来源（全部真实，不模拟）：
  - CPI: L4/results/experimental_actives_detail_cleaned_combined.csv
  - 蛋白嵌入: L4/results_v10_minibatch/esm2_protein_embeddings.npz (全局 CLS)
  - 蛋白嵌入: L4/results_v10_minibatch/esm2_residue_pooled_embeddings.npz (残基池化)
  - 蛋白嵌入: L4/results_v10_minibatch/esm2_150M_residue_features.pt (残基级)
  - TCM池: L3/results/tcm_compound_pool_tox_filtered_noleak.csv
  - 中药映射: L3/results/herb_ingredient_mapping.xlsx

输出：
  - L4/results/tree_v6_protein_emb_ablation.csv
  - L4/results/tree_v6_results.csv
  - L4/results/tree_v6_tcm_predictions_v7.csv
  - L4/results/tree_v6_top_candidates.csv
"""

import logging
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

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
        logging.FileHandler(L4_LOGS / "tree_v6_5.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

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


def compute_pharmacophore(smiles_list, nbits=1024):
    """Pharmacophore 指纹（Gobbi 2D 药效团）"""
    fps = np.zeros((len(smiles_list), nbits), dtype=np.float32)
    try:
        from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
        factory = Gobbi_Pharm2D.factory
        for i, smi in enumerate(smiles_list):
            if not smi or pd.isna(smi):
                continue
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                continue
            fp = Generate.Gen2DFingerprint(mol, factory)
            arr = np.zeros(nbits, dtype=np.float32)
            on_bits = list(fp.GetOnBits())
            for b in on_bits:
                if b < nbits:
                    arr[b] = 1.0
            fps[i] = arr
    except Exception as e:
        logger.warning(f"Pharmacophore 指纹计算失败: {e}，跳过")
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
            except Exception as e:
                vals.append(np.nan)
                logger.debug(f"RDKit 2D 描述符计算失败: {e}")
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
                logger.info("  缓存 SMILES 列表不匹配，重新计算")
        except Exception as e:
            logger.warning(f"  缓存加载失败: {e}，重新计算")

    binary_fps = []
    binary_labels = []

    fp = compute_ecfp(smiles_list, radius=2, nbits=2048)
    binary_fps.append(fp)
    binary_labels.append("ECFP4")
    logger.info(f"    ECFP4: {fp.shape} (binary)")

    fp = compute_ecfp(smiles_list, radius=3, nbits=2048)
    binary_fps.append(fp)
    binary_labels.append("ECFP6")
    logger.info(f"    ECFP6: {fp.shape} (binary)")

    fp = compute_maccs(smiles_list)
    binary_fps.append(fp)
    binary_labels.append("MACCS")
    logger.info(f"    MACCS: {fp.shape} (binary)")

    fp = compute_atom_pairs(smiles_list, nbits=1024)
    binary_fps.append(fp)
    binary_labels.append("AtomPairs")
    logger.info(f"    AtomPairs: {fp.shape} (binary)")

    fp = compute_avalon(smiles_list, nbits=1024)
    binary_fps.append(fp)
    binary_labels.append("Avalon")
    logger.info(f"    Avalon: {fp.shape} (binary)")

    fp = compute_pharmacophore(smiles_list, nbits=1024)
    binary_fps.append(fp)
    binary_labels.append("Pharmacophore")
    logger.info(f"    Pharmacophore: {fp.shape} (binary)")

    X_binary = np.hstack(binary_fps).astype(np.float32)
    logger.info(f"  二进制指纹总维度: {X_binary.shape[1]}")

    X_rdkit, rdkit_names = compute_rdkit_2d(smiles_list)
    logger.info(f"    RDKit2D: {X_rdkit.shape} (continuous)")

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
        logger.info("  RDKit2D 已标准化 (mean=0, std=1)")

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


def process_protein_embeddings(protein_embeddings, target_dim=128, pca_model=None, scaler=None):
    """蛋白嵌入处理：PCA 降维 + 标准化"""
    keys = sorted(protein_embeddings.keys())
    vectors = np.array([protein_embeddings[k] for k in keys], dtype=np.float32)
    original_dim = vectors.shape[1]
    logger.info(f"  蛋白嵌入原始维度: {original_dim}, 数量: {len(keys)}")

    if pca_model is None:
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
        top5_ratio = pca.explained_variance_ratio_[:5].sum()
        logger.info(f"  Top-5 PC 累计方差: {top5_ratio:.4f}")
        scaler = StandardScaler()
        vectors_scaled = scaler.fit_transform(vectors_reduced)
        logger.info("  蛋白嵌入已标准化")
        processed = {k: vectors_scaled[i] for i, k in enumerate(keys)}
        return processed, pca, scaler
    else:
        vectors_reduced = pca_model.transform(vectors)
        vectors_scaled = scaler.transform(vectors_reduced)
        processed = {k: vectors_scaled[i] for i, k in enumerate(keys)}
        return processed, None, None



PROTEIN_EMB_MODES = ["global", "residue_pooled", "residue_meanmaxstd", "combined"]


def _load_protein_embeddings_global():
    """加载全局 CLS 蛋白嵌入"""
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
    """加载预池化的残基层 ESM-2 嵌入"""
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
    except Exception as e:
        logger.debug(f"Scaffold 计算失败 (SMILES: {str(smiles)[:30]}...): {e}")
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
        train_idx, test_idx = train_test_split(
            np.arange(len(pair_smiles)), test_size=test_size,
            random_state=random_state, stratify=y,
        )

    logger.info(f"  Scaffold Split: train={len(train_idx)}, test={len(test_idx)}, "
                f"test_scaffolds={len(test_scaffolds)}/{n_scaffolds}, "
                f"test_compounds={len(test_smiles)}/{len(unique_smiles)}")

    return train_idx, test_idx


def diversity_constrained_negative_sampling(
    pos_pairs, compound_smiles, cpi_genes_in_emb, neg_ratio=3, random_seed=42,
    protein_embeddings=None, compound_ecfp4=None,
    hard_ratio=0.3, esm_hard_ratio=0.2,
):
    """多样性约束负采样：混合随机 + Tanimoto 难负 + ESM-2 结构难负。

    Args:
        pos_pairs: [(smiles, gene), ...] 正样本对
        compound_smiles: 所有化合物 SMILES
        cpi_genes_in_emb: 有嵌入的 CPI 基因列表
        neg_ratio: 负正样本比例
        random_seed: 随机种子
        protein_embeddings: {gene: vector} 蛋白 ESM-2 嵌入（用于 ESM-2 难负样本）
        compound_ecfp4: (n_compounds, 2048) ECFP4 指纹（用于 Tanimoto 难负样本）
        hard_ratio: Tanimoto 难负样本比例
        esm_hard_ratio: ESM-2 结构难负样本比例
    """
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

    n_random = int(n_neg_target * (1.0 - hard_ratio - esm_hard_ratio))
    n_tanimoto_hard = int(n_neg_target * hard_ratio)
    n_esm_hard = n_neg_target - n_random - n_tanimoto_hard

    gene_neg_counts = {gi: 0 for gi in range(n_genes)}
    max_per_gene = max(1, n_neg_target // n_genes + 1)

    neg_idx_set = set()

    # 1) Tanimoto 难负样本：选择与正样本化合物结构相似但无交互的化合物-蛋白对
    if n_tanimoto_hard > 0 and compound_ecfp4 is not None:
        from rdkit import DataStructs
        pos_comp_set = {smi for smi, _ in pos_pairs}
        pos_comp_indices = [smiles_to_idx[s] for s in pos_comp_set if s in smiles_to_idx]
        non_pos_comps = [i for i in range(n_compounds) if i not in pos_comp_indices]
        if non_pos_comps:
            rng.shuffle(non_pos_comps)
            non_pos_fps = compound_ecfp4[non_pos_comps] if compound_ecfp4 is not None else None
            pos_fps = compound_ecfp4[pos_comp_indices] if compound_ecfp4 is not None else None
            n_tani = 0
            for non_pos_i in non_pos_comps[:min(len(non_pos_comps), n_tanimoto_hard * 3)]:
                if n_tani >= n_tanimoto_hard:
                    break
                if non_pos_fps is None or pos_fps is None:
                    continue
                non_pos_fp = compound_ecfp4[non_pos_i]
                max_sim = 0.0
                for pos_i in pos_comp_indices[:min(len(pos_comp_indices), 100)]:
                    sim = DataStructs.TanimotoSimilarity(compound_ecfp4[pos_i], non_pos_fp)
                    if sim > max_sim:
                        max_sim = sim
                if max_sim >= 0.6:
                    gi = rng.randint(0, n_genes)
                    pair = (non_pos_i, int(gi))
                    if pair not in pos_idx_set and pair not in neg_idx_set:
                        if gene_neg_counts[gi] < max_per_gene:
                            neg_idx_set.add(pair)
                            gene_neg_counts[gi] += 1
                            n_tani += 1
            logger.info(f"  Tanimoto 难负样本: {n_tani}/{n_tanimoto_hard} (目标比例 {hard_ratio:.0%})")

    # 2) ESM-2 余弦相似度难负样本：选择蛋白结构相似但无交互的化合物-蛋白对
    if n_esm_hard > 0 and protein_embeddings is not None:
        gene_emb_list = np.array([protein_embeddings[g] for g in cpi_genes_in_emb], dtype=np.float32)
        gene_norms = np.linalg.norm(gene_emb_list, axis=1, keepdims=True) + 1e-8
        gene_emb_normed = gene_emb_list / gene_norms
        sim_matrix = np.dot(gene_emb_normed, gene_emb_normed.T)
        sim_matrix[np.diag_indices_from(sim_matrix)] = 0.0
        n_esm = 0
        for gi in range(n_genes):
            if n_esm >= n_esm_hard:
                break
            similar_genes = np.argsort(sim_matrix[gi])[::-1][:5]
            for sg in similar_genes:
                if sim_matrix[gi, sg] < 0.7:
                    continue
                ci = rng.randint(0, n_compounds)
                pair = (int(ci), int(sg))
                if pair not in pos_idx_set and pair not in neg_idx_set:
                    if gene_neg_counts[sg] < max_per_gene:
                        neg_idx_set.add(pair)
                        gene_neg_counts[sg] += 1
                        n_esm += 1
                        if n_esm >= n_esm_hard:
                            break
        logger.info(f"  ESM-2 结构难负样本: {n_esm}/{n_esm_hard} (目标比例 {esm_hard_ratio:.0%})")

    # 3) 随机负样本：填充剩余
    batch_size = n_random * 10
    max_attempts = n_random * 50
    n_random_collected = 0
    while n_random_collected < n_random and n_random_collected < max_attempts:
        batch_comp = rng.randint(0, n_compounds, size=batch_size)
        batch_gene = rng.randint(0, n_genes, size=batch_size)
        for ci, gi in zip(batch_comp, batch_gene):
            pair = (int(ci), int(gi))
            if pair in pos_idx_set or pair in neg_idx_set:
                continue
            if gene_neg_counts[gi] >= max_per_gene:
                continue
            neg_idx_set.add(pair)
            gene_neg_counts[gi] += 1
            n_random_collected += 1
            if n_random_collected >= n_random:
                break

    neg_pairs = []
    for ci, gi in neg_idx_set:
        smi = str(compound_smiles[ci])
        gene = cpi_genes_in_emb[gi]
        neg_pairs.append((smi, gene))

    logger.info(f"  负样本总计: {len(neg_pairs)} 对 (比例 1:{neg_ratio}), "
                f"蛋白覆盖: {sum(1 for c in gene_neg_counts.values() if c > 0)}/{n_genes}")

    return neg_pairs


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

    # Precision@K: 药物筛选场景核心指标，衡量Top-K推荐中真阳性比例
    for k in [10, 20, 50]:
        k_actual = min(k, n_total)
        top_k_idx = np.argsort(y_prob)[-k_actual:]
        hits = y_true[top_k_idx].sum()
        metrics[f"Precision@{k}"] = hits / k_actual

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


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """训练并评估单个模型"""
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


def train_ensemble(
    pair_smiles, pair_genes, y,
    X_binary_all, X_rdkit_raw_all, all_smiles,
    protein_embeddings_raw,
    best_params_per_model=None, n_folds=5, random_seed=42,
):
    """5-fold Scaffold Split CV 训练 XGBoost，用于模型选择。

    仅保留 XGBoost，RDKit 标准化和蛋白 PCA 均在每个 fold 内仅用训练集拟合，
    彻底消除数据泄露。

    CV 设计原则：
      - 每折独立拟合 RDKit scaler 和蛋白 PCA（仅用训练集）
      - 这是严谨的 CV 设计（预处理作为 pipeline 纳入 CV），
        确保模型选择阶段的指标无泄露
      - 注意：CV 汇总指标仅用于模型选择，不代表最终泛化性能。
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

        try:
            import xgboost as xgb
            logger.info("  [1/1] XGBoost...")
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
                results.append(r)
                logger.info(f"    AUC={r['AUC']:.4f}, AUPR={r['AUPR']:.4f}, "
                            f"F1={r['F1']:.4f}, MCC={r['MCC']:.4f}")
        except ImportError:
            logger.error("XGBoost 未安装，无法继续")
            raise

    return pd.DataFrame(results)


def load_herb_mapping():
    """加载中药来源映射"""
    herb_map_path = L3_RESULTS / "herb_ingredient_mapping.xlsx"
    if not herb_map_path.exists():
        logger.warning(f"中药映射文件不存在: {herb_map_path}")
        return {}

    try:
        herb_df = pd.read_excel(herb_map_path)
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
    不确定性估计需要通过预训练多成员集成模型实现，暂未在 TCM 预测循环中实现。
    """
    if herb_map is None:
        herb_map = {}

    logger.info(f"  预测 {len(tcm_df)} 个 TCM 化合物 x {len(cpi_genes_in_emb)} 个基因...")

    predictions = []

    if return_std:
        logger.info("    注意: return_std=True 但当前暂未实现逐一预测的不确定性估计，score_std 将置为 NaN。"
                     "不确定性需要预训练多成员集成模型。")

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

    rdkit_scaler = StandardScaler()
    rdkit_scaler.fit(X_rdkit_raw_all[train_compound_indices])
    X_rdkit_all_scaled = rdkit_scaler.transform(X_rdkit_raw_all)
    compound_features_all = np.hstack([X_binary_all, X_rdkit_all_scaled])
    compound_feat_dim = compound_features_all.shape[1]

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
                       neg_ratio=3, random_seed=42,
                       real_protein_embeddings=None, compound_ecfp4=None):
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
        protein_embeddings=real_protein_embeddings,
        compound_ecfp4=compound_ecfp4,
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
    """对单个蛋白嵌入模式运行轻量 5-fold CV（仅 XGBoost），用于消融筛选。

    RDKit 标准化和蛋白 PCA 均在每个 fold 内仅用训练集拟合，彻底消除数据泄露。
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"蛋白嵌入消融: mode={mode} (XGBoost-only)")
    logger.info(f"{'='*60}")

    protein_embeddings_raw = load_protein_embeddings_by_mode(mode)

    # 注意：这里用 dummy compound_features 来获取 pair 结构，不用于实际特征计算
    dummy_compound = np.zeros((len(all_smiles), 1), dtype=np.float32)
    dummy_protein = {k: np.zeros(1, dtype=np.float32) for k in protein_embeddings_raw}
    _, y, pair_smiles, pair_genes, cpi_genes_in_emb, _ = build_pair_dataset(
        cpi_df, dummy_protein, dummy_compound, all_smiles,
        neg_ratio=3, random_seed=random_seed,
        real_protein_embeddings=protein_embeddings_raw,
        compound_ecfp4=compute_ecfp(all_smiles, radius=2, nbits=2048),
    )
    n_pos = int(y.sum())
    n_genes = len(cpi_genes_in_emb)
    logger.info(f"  有效基因数: {n_genes}, 正样本数: {n_pos}, 总样本数: {len(y)}")

    import xgboost as xgb
    scale_pos_weight = (y == 0).sum() / max(y.sum(), 1)
    base_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=random_seed, n_jobs=-1, verbosity=0,
    )

    # GridSearchCV 超参数调优（仅在第一个 fold 上运行，避免过拟合）
    logger.info("  GridSearchCV 超参数调优 (fold 1, 3-fold 内交叉)...")
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [0.1, 1.0, 10.0],
        "min_child_weight": [1, 5, 10],
    }
    from sklearn.model_selection import GridSearchCV
    first_fold_train, first_fold_test = scaffold_split(
        pair_smiles, y, test_size=0.2, random_state=random_seed,
    )
    X_fold0, _, _, _ = build_fold_features(
        pair_smiles, pair_genes, first_fold_train,
        X_binary_all, X_rdkit_raw_all, all_smiles,
        protein_embeddings_raw, prot_target_dim=128,
    )
    X_gs_train, y_gs_train = X_fold0[first_fold_train], y[first_fold_train]
    grid_search = GridSearchCV(
        xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=random_seed, n_jobs=-1, verbosity=0,
        ),
        param_grid, cv=3, scoring="average_precision",
        n_jobs=-1, verbose=0,
    )
    grid_search.fit(X_gs_train, y_gs_train)
    best_params = grid_search.best_params_
    logger.info(f"  最优参数: {best_params}")
    logger.info(f"  最佳验证 AUPR: {grid_search.best_score_:.4f}")
    base_model = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        random_state=random_seed, n_jobs=-1, verbosity=0,
    )

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
    """使用指定蛋白嵌入模式运行完整 5-fold CV、全量训练与 TCM 预测。

    仅使用 XGBoost，RDKit 标准化和蛋白 PCA 均在每个 fold 内仅用训练集拟合，
    彻底消除数据泄露。
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"完整流程: 蛋白嵌入 mode={mode} (XGBoost-only)")
    logger.info(f"{'='*60}")

    logger.info("\n[3/7] 加载原始蛋白嵌入...")
    protein_embeddings_raw = load_protein_embeddings_by_mode(mode)

    logger.info("\n[4/7] 构建训练数据集 (多样性约束负采样)...")
    # 用 dummy 特征获取 pair 结构，实际特征在 fold 内构建
    dummy_compound = np.zeros((len(all_smiles), 1), dtype=np.float32)
    dummy_protein = {k: np.zeros(1, dtype=np.float32) for k in protein_embeddings_raw}
    _, y, pair_smiles, pair_genes, cpi_genes_in_emb, smiles_to_idx = build_pair_dataset(
        cpi_df, dummy_protein, dummy_compound, all_smiles,
        neg_ratio=3, random_seed=random_seed,
        real_protein_embeddings=protein_embeddings_raw,
        compound_ecfp4=compute_ecfp(all_smiles, radius=2, nbits=2048),
    )
    logger.info(f"  有效基因数: {len(cpi_genes_in_emb)}, 正样本数: {int(y.sum())}, 总样本数: {len(y)}")

    logger.info("\n[5/7] 5-fold Scaffold Split XGBoost 评估 (无泄露)...")
    results_df = train_ensemble(
        pair_smiles, pair_genes, y,
        X_binary_all, X_rdkit_raw_all, all_smiles,
        protein_embeddings_raw,
        best_params_per_model=None,
        n_folds=n_folds, random_seed=random_seed,
    )

    logger.info("\n" + "=" * 60)
    logger.info("XGBoost 评估汇总 (5-fold Scaffold Split, mean +/- std):")
    logger.info("=" * 60)
    summary = results_df.groupby("model").agg(["mean", "std"]).round(4)
    for model_name in summary.index:
        row = summary.loc[model_name]
        logger.info(f"\n  {model_name}:")
        for metric in ["AUC", "AUPR", "F1", "MCC", "Precision@10", "Precision@20", "Precision@50", "EF@1%", "EF@5%", "BEDROC", "ROCE@1%"]:
            if metric in row.index:
                logger.info(f"    {metric}: {row[metric]['mean']:.4f} +/- {row[metric]['std']:.4f}")

    results_path = L4_RESULTS / "tree_v6_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\n评估结果已保存: {results_path}")

    logger.info("\n[6/7] 全量训练 XGBoost...")

    best_model_name = "XGBoost"

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

    import xgboost as xgb
    scale_pos_weight = (y == 0).sum() / max(y.sum(), 1)
    best_tree = xgb.XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=random_seed, n_jobs=-1, verbosity=0,
    )

    logger.info(f"  全量训练 XGBoost ({len(X_full)} 样本)...")
    best_tree.fit(X_full, y)

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
    pred_df = pred_tree

    pred_path = L4_RESULTS / "tree_v6_tcm_predictions_v7.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"TCM 预测结果已保存: {pred_path} ({len(pred_df)} 条)")

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

    logger.info("\nTop 20 候选化合物:")
    for i, row in enumerate(top50.head(20).itertuples(index=False), 1):
        logger.info(f"  {i:2d}. {row.molecule_name} | max={row.max_score:.4f} "
                    f"| mean={row.mean_score:.4f} "
                    f"| 高置信(>=0.5): {row.n_genes_above_50} "
                    f"| 模型数: {row.n_models} "
                    f"| 中药: {row.herb_cn} "
                    f"| {row.top_3_genes}")

    logger.info(f"Top 50 候选已保存: {top_path}")
    logger.info("=" * 60)
    logger.info("任务完成.")
    logger.info("=" * 60)


def main():
    logger.info("=" * 60)
    logger.info("树模型 CPI — 仅 XGBoost")
    logger.info("  RDKit 标准化和蛋白 PCA 均在 CV fold 内仅用训练集拟合")
    logger.info("=" * 60)

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

    logger.info("\n[2/8] 多指纹特征工程...")
    X_binary, X_rdkit, X_rdkit_raw, rdkit_scaler, binary_labels, rdkit_names = \
        build_multifingerprint_features(all_smiles)

    compound_features_preview = np.hstack([X_binary, X_rdkit])
    logger.info(f"  化合物特征总维度: {compound_features_preview.shape[1]} "
                f"(binary={X_binary.shape[1]}, rdkit={X_rdkit.shape[1]})")
    logger.info("  注意: RDKit 标准化参数仅用于预览，实际 CV 内每折独立拟合")

    logger.info("\n[2.5/8] 蛋白嵌入消融实验 (4 种模式 x 5-fold CV, XGBoost 基线, 无泄露)...")
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

    logger.info(f"\n[3-7/8] 使用最佳模式 {best_mode} 运行完整流程...")
    run_full_pipeline(
        best_mode, cpi_df, tcm_df, all_smiles,
        X_binary, X_rdkit_raw,
        n_folds=5, random_seed=42,
    )

    logger.info("\n[8/8] 全部流程完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
