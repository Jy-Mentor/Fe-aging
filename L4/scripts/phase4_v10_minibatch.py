#!/usr/bin/env python3
"""Phase 4: Mini-Batch GNN 双分支 — 化合物冷启动候选化合物发现

SAGE + HGT 双分支集成：拓扑 (SAGEConv) + 语义 (HGTConv)，支持：
- 两阶段迁移学习 (BiMLPA/HHI 社区感知头尾划分)
- 课程负采样 (随机 → 中度通路邻近 → 极硬)
- PPI拓扑难负样本 + ESM-2结构相似性难负样本
- Focal Loss + BPR排序损失 + Memory Bank
- 铁死亡表型分类辅助任务 (多任务联合训练)
- 四模态异质图: 化合物-蛋白-通路-疾病 (GSE61616)
- ResidueAwareBilinearDecoder 残基-原子级交互解码器
- 化合物冷启动验证 (归纳式设定)
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
import random
import sys
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import HeteroData
from tqdm import tqdm

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="rdkit")
warnings.filterwarnings("ignore", category=FutureWarning, module="rdkit")

PROJECT_ROOT = Path(__file__).parent.parent.parent
L1_RESULTS = PROJECT_ROOT / "L1" / "results"
L2_RESULTS = PROJECT_ROOT / "L2" / "results"
L3_RESULTS = PROJECT_ROOT / "L3" / "results"
L4_ROOT = PROJECT_ROOT / "L4"
L4_SRC = L4_ROOT / "src"
L4_RESULTS = L4_ROOT / "results_v10_minibatch"
L4_LOGS = L4_ROOT / "logs"

sys.path.insert(0, str(L4_SRC))
from iron_aging_gnn.graph.sampling import sample_hetero_subgraph  # noqa: E402
from iron_aging_gnn.graph.topology_negative_sampling import (  # noqa: E402
    build_topology_hard_neighbors,
    build_topology_medium_neighbors,
)
from iron_aging_gnn.models import MemoryBank, SAGELinkPredictor, HGTLinkPredictor  # noqa: E402
from iron_aging_gnn.training.trainer import train_sage, train_hgt  # noqa: E402
from iron_aging_gnn.utils.config import Config, load_config  # noqa: E402

for d in [L4_RESULTS, L4_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = L4_LOGS / "phase4_v41_hgt_diag.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"设备: {DEVICE}")

_config_path = PROJECT_ROOT / "L4" / "configs" / "default.yaml"
try:
    _cfg = load_config(str(_config_path))
    logger.info(f"配置系统已加载: {_config_path}")
except Exception as _cfg_err:
    logger.warning(f"配置系统加载失败 ({_cfg_err})，将使用硬编码常量作为回退（向后兼容）")
    _cfg = None

FERRORAGING_GENES_CSV = L1_RESULTS / "ferroaging_genes_96.csv"
if FERRORAGING_GENES_CSV.exists():
    _df = pd.read_csv(FERRORAGING_GENES_CSV)
    ALL_FERRORAGING_GENES = sorted(_df["gene_symbol"].dropna().unique().tolist())
else:
    ALL_FERRORAGING_GENES = sorted([
        "ABCC1", "ACSL4", "ACVR1B", "ALOX15", "ATF3", "ATG3", "BAP1", "BCL6",
        "BRD7", "CAVIN1", "CD74", "CD82", "CDO1", "COX7A1", "CTSB", "CXCL10",
        "DPEP1", "DPP4", "DUOX1", "DYRK1A", "E2F1", "E2F3", "EBF3", "EDN1",
        "EGR1", "EMP1", "EPHA2", "EPHA4", "ERN1", "FBXO31", "FOSL1", "GMFB",
        "HBP1", "HERPUD1", "HIF1A", "HMGB1", "HMOX1", "ICA1", "IFNG", "IGFBP7",
        "IL1B", "IL6", "IRF1", "IRF7", "IRF9", "KDM6B", "KEAP1", "KLF6",
        "LACTB", "LCN2", "LGMN", "LIFR", "LOX", "LPCAT3", "MAP3K14", "MAPK1",
        "MAPK14", "MCU", "MEN1", "MPO", "NLRP3", "NOX4", "NR1D1", "NR2F2",
        "NUAK2", "PADI4", "PDE4B", "PPP2R2B", "PRKD1", "PTBP1", "PTGS2",
        "RBM3", "RUNX3", "S100A8", "SAT1", "SETD7", "SLAMF8", "SLC1A5",
        "SMARCB1", "SMURF2", "SNCA", "SOCS1", "SOCS2", "SOD1", "SP1", "SPATA2",
        "TBX2", "TFRC", "TLR4", "TNFAIP1", "TNFAIP3", "TXNIP", "WNT5A",
        "WWTR1", "YAP1", "ZEB1",
    ])
logger.info(f"铁衰老靶标: {len(ALL_FERRORAGING_GENES)} 个基因")

TARGET_PRIORITY = {
    # Tier 1: 核心铁死亡/铁衰老效应器 (weight=5.0)
    "ACSL4": 5.0, "HMOX1": 5.0, "TFRC": 5.0, "LPCAT3": 5.0, "PTGS2": 5.0,
    # Tier 2: 关键调控因子 (weight=3.0)
    "HIF1A": 3.0, "MAPK1": 3.0, "TLR4": 3.0, "NOX4": 3.0,
    "IL1B": 3.0, "IL6": 3.0, "IFNG": 3.0,
    "KEAP1": 3.0, "ALOX15": 2.5, "ATG3": 2.5,
    # Tier 3: 相关通路蛋白 (weight=2.0)
    "KDM6B": 2.0, "CTSB": 2.0, "CXCL10": 2.0, "SOD1": 2.0,
    "SAT1": 2.0, "CD74": 2.0, "IRF1": 2.0, "IRF7": 2.0, "IRF9": 2.0,
    "LGMN": 2.0, "DYRK1A": 2.0, "PDE4B": 2.0, "BCL6": 2.0,
    "EPHA4": 2.0, "LCN2": 2.0, "SP1": 2.0,
    # Tier 4: 其他铁衰老相关基因 (weight=1.5, 默认1.0)
    "MAPK14": 1.5, "NLRP3": 1.5, "MPO": 1.5, "HMGB1": 1.5,
    "TXNIP": 1.5, "S100A8": 1.5, "SNCA": 1.5,
    "WWTR1": 1.5, "YAP1": 1.5, "ZEB1": 1.5,
    "EGR1": 1.5, "FOSL1": 1.5,
    "CAVIN1": 1.5, "DPP4": 1.5, "ERN1": 1.5,
    "LOX": 1.5, "MCU": 1.5, "SMURF2": 1.5,
    "SOCS1": 1.5, "SOCS2": 1.5, "TNFAIP3": 1.5,
}
_DEFAULT_PRIORITY = 1.0

HIDDEN_DIM = _cfg.model.hidden_dim if _cfg else 64
OUT_DIM = _cfg.model.out_dim if _cfg else 64
NUM_LAYERS = _cfg.model.num_layers if _cfg else 2
NUM_HEADS = _cfg.model.num_heads if _cfg else 2
DROPOUT = _cfg.model.dropout if _cfg else 0.5
PROT_PROJ_DROPOUT = _cfg.model.prot_proj_dropout if _cfg else 0.4
PROT_PROJ_INNER_DROPOUT = _cfg.model.prot_proj_inner_dropout if _cfg else 0.3
PATHWAY_PROJ_DROPOUT = _cfg.model.pathway_proj_dropout if _cfg else 0.3
PHENO_HEAD_DROPOUT = _cfg.model.pheno_head_dropout if _cfg else 0.3
TEMPERATURE = _cfg.model.temperature if _cfg else 5.0

FOCAL_GAMMA = _cfg.loss.focal_gamma if _cfg else 2.0
FOCAL_ALPHA = _cfg.loss.focal_alpha if _cfg else 0.75
LABEL_SMOOTHING_POS = _cfg.loss.label_smoothing_pos if _cfg else 0.9
LABEL_SMOOTHING_NEG = _cfg.loss.label_smoothing_neg if _cfg else 0.1
BPR_WEIGHT = _cfg.loss.bpr_weight if _cfg else 0.4
CPI_LOSS_WEIGHT = _cfg.loss.bce_weight if _cfg else 0.6
INFONCE_WEIGHT = _cfg.loss.infonce_weight if _cfg else 0.1

LEARNING_RATE_SAGE = _cfg.sage.lr if _cfg else 5e-4
LEARNING_RATE_HGT = _cfg.hgt.lr if _cfg else 1e-3
PRETRAIN_LR_MULTIPLIER = _cfg.two_stage.pretrain_lr_multiplier if _cfg else 1.5
PRETRAIN_LR_DECAY = _cfg.two_stage.pretrain_lr_decay if _cfg else 0.5
WEIGHT_DECAY = _cfg.training.weight_decay if _cfg else 1e-4
GRAD_CLIP_NORM = _cfg.training.grad_clip_norm if _cfg else 1.0
WARMUP_RATIO = _cfg.training.warmup_ratio if _cfg else 0.05
DROPPEDGE_PPI = _cfg.training.dropedge_ppi if _cfg else 0.15
DROPPEDGE_PATHWAY = _cfg.training.dropedge_pathway if _cfg else 0.1
EPOCHS = _cfg.sage.epochs if _cfg else 15
PATIENCE = _cfg.sage.patience if _cfg else 5
PRETRAIN_EPOCHS = _cfg.sage.pretrain_epochs if _cfg else 10
PRETRAIN_LR_SAGE = _cfg.sage.pretrain_lr if _cfg else 7.5e-4
PRETRAIN_LR_HGT = _cfg.hgt.pretrain_lr if _cfg else 1.5e-3
SAGE_BATCH_SIZE = _cfg.sage.batch_size if _cfg else 256
HGT_BATCH_SIZE = _cfg.hgt.batch_size if _cfg else 128
SAGE_NUM_NEIGHBORS = _cfg.sage.num_neighbors if _cfg else [32, 16]
HGT_NUM_NEIGHBORS = _cfg.hgt.num_neighbors if _cfg else [32, 16]
VAL_FREQ = _cfg.validation.val_freq if _cfg else 2
PRETRAIN_VAL_FREQ = _cfg.validation.pretrain_val_freq if _cfg else 5
MEM_REFRESH_FREQ = _cfg.validation.mem_refresh_freq if _cfg else 5
PHENO_LAMBDA = _cfg.training.pheno_lambda if _cfg else 0.05

CURRICULUM_PHASE1 = _cfg.curriculum.random_ratio if _cfg else 0.3
CURRICULUM_PHASE2 = (_cfg.curriculum.random_ratio + _cfg.curriculum.moderate_ratio) if _cfg else 0.7
MEDIUM_NEG_RATIO = _cfg.curriculum.medium_neg_ratio if _cfg else 0.3
HARD_NEG_RATIO = _cfg.curriculum.hard_neg_ratio if _cfg else 0.1

USE_TOPOLOGY_NEG = _cfg.negative_sampling.use_topology_neg if _cfg else False
USE_ESM_SIMILARITY_NEG = _cfg.negative_sampling.use_esm_similarity_neg if _cfg else False
TOPO_NEIGHBORS_TOP_K = _cfg.negative_sampling.topo_neighbors_top_k if _cfg else 50
ESM_SIMILARITY_TOP_K = _cfg.negative_sampling.esm_similarity_top_k if _cfg else 50

FLAG_STEP = _cfg.training.flag_step if _cfg else 0.01
SCORE_CLAMP = _cfg.model.score_clamp if _cfg else 10
DECODER_TYPE = _cfg.model.decoder_type if _cfg else "mlp"

MEMORY_BANK_SIZE = _cfg.memory_bank.memory_bank_size if _cfg else 8192
INFONCE_WARMUP_RATIO = _cfg.two_stage.infonce_warmup_ratio if _cfg else 0.15
INFONCE_MEM_SAMPLE = _cfg.memory_bank.infonce_mem_sample if _cfg else 256
INFONCE_TEMPERATURE = _cfg.loss.infonce_temperature if _cfg else 0.07

HEAD_RATIO = _cfg.two_stage.head_ratio if _cfg else 0.2
LAMBDA_HHI = _cfg.two_stage.lambda_hhi if _cfg else 1.0
HEAD_UNDERSAMPLE_RATIO = _cfg.two_stage.head_undersample_ratio if _cfg else 0.6

COMPOUND_VAL_SPLIT = _cfg.validation.compound_split_ratio if _cfg else 0.85
PROTEIN_VAL_SPLIT = _cfg.validation.protein_cold_split_ratio if _cfg else 0.50

HARD_NEG_TOP_K = _cfg.validation.hard_neg_top_k if _cfg else 5
RAND_NEG_TOP_K = _cfg.validation.rand_neg_top_k if _cfg else 5
VAL_BATCH_SIZE = _cfg.validation.val_batch_size if _cfg else 512
HGT_VAL_BATCH_SIZE = _cfg.validation.hgt_val_batch_size if _cfg else 64
HGT_VAL_NUM_NEIGHBORS = _cfg.validation.hgt_val_num_neighbors if _cfg else [64, 32]

MC_SAMPLES = _cfg.validation.mc_samples if _cfg else 30
DIVERSITY_PENALTY = _cfg.validation.diversity_penalty if _cfg else 0.3
DEFAULT_AUPR = _cfg.validation.default_aupr if _cfg else 0.5
TOP_N_CANDIDATES = _cfg.prediction.top_n_candidates if _cfg else 500
WARM_TARGETS_TOP_N = _cfg.prediction.warm_targets_top_n if _cfg else 5
ZS_TARGETS_TOP_N = _cfg.prediction.zs_targets_top_n if _cfg else 3
COMPOSITE_AVG_WEIGHT = _cfg.prediction.composite_avg_weight if _cfg else 0.4
COMPOSITE_MAX_WEIGHT = _cfg.prediction.composite_max_weight if _cfg else 0.3
TREE_ENSEMBLE_WEIGHT = _cfg.prediction.tree_ensemble_weight if _cfg else 0.6
COMPOSITE_HITS_WEIGHT = _cfg.prediction.composite_hits_weight if _cfg else 0.3
FERRO_FACTOR_BASE = _cfg.prediction.ferro_factor_base if _cfg else 0.7
ZS_BONUS_MAX = _cfg.prediction.zs_bonus_max if _cfg else 0.05

ESM_MAX_LEN = _cfg.esm2.esm_max_len if _cfg else 1022
ESM_BATCH_SIZE = _cfg.esm2.esm_batch_size if _cfg else 4
ESM_MODEL_NAME = _cfg.esm2.model_name if _cfg else "facebook/esm2_t30_150M_UR50D"

MASK_VAL = _cfg.numerical.mask_val if _cfg else -1e9
EPS = _cfg.numerical.eps if _cfg else 1e-8
EPS_SMALL = _cfg.numerical.eps_small if _cfg else 1e-10


RDKIT_DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "MolMR", "TPSA",
    "NumHAcceptors", "NumHDonors", "NumRotatableBonds",
    "HeavyAtomCount", "NumAromaticRings", "NumAliphaticRings",
    "NumHeteroatoms", "NumValenceElectrons", "NHOHCount", "NOCount",
    "RingCount", "FractionCSP3", "BalabanJ",
]
ECFP4_NBITS = 2048


def _get_prot_feat_dim(prot_feat) -> int | None:
    """提取蛋白特征维度，兼容 numpy ndarray / torch Tensor / dict。"""
    if hasattr(prot_feat, "shape") and len(prot_feat.shape) >= 2:
        return int(prot_feat.shape[-1])
    if isinstance(prot_feat, dict):
        for v in prot_feat.values():
            if hasattr(v, "shape") and len(v.shape) >= 1:
                return int(v.shape[-1])
    return None


def _compute_ecfp4(smiles_iter: list[str]) -> np.ndarray:
    fps = np.zeros((len(smiles_iter), ECFP4_NBITS), dtype=np.float32)
    for i, smi in enumerate(smiles_iter):
        mol = None
        try:
            if pd.notna(smi):
                mol = Chem.MolFromSmiles(str(smi))
        except Exception as e:
            logger.warning(f"ECFP4 SMILES 解析失败 索引 {i}: {smi!r}, 错误: {e}")
            mol = None
        if mol is None:
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=ECFP4_NBITS)
            for bit in fp.GetOnBits():
                fps[i, bit] = 1.0
        except Exception as e:
            logger.warning(f"ECFP4 指纹生成失败 索引 {i}: {smi!r}, 错误: {e}")
    return fps


def _compute_maccs(smiles_iter: list[str]) -> np.ndarray:
    fps = []
    for i, smi in enumerate(smiles_iter):
        mol = None
        try:
            if pd.notna(smi):
                mol = Chem.MolFromSmiles(str(smi))
        except Exception as e:
            logger.warning(f"SMILES 解析失败: {smi!r}, 错误: {e}")
            mol = None
        if mol is None:
            fps.append(np.zeros(167, dtype=np.float32))
            continue
        try:
            fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            arr = np.zeros(167, dtype=np.float32)
            arr[list(fp.GetOnBits())] = 1.0
            fps.append(arr)
        except Exception as e:
            logger.warning(f"MACCS 指纹生成失败 索引 {i}: {smi!r}, 错误: {e}")
            fps.append(np.zeros(167, dtype=np.float32))
    return np.array(fps, dtype=np.float32)


def _compute_rdkit_descriptors(smiles_iter: list[str]) -> np.ndarray:
    desc_funcs = {name: getattr(Descriptors, name) for name in RDKIT_DESCRIPTOR_NAMES}
    rows = []
    for i, smi in enumerate(smiles_iter):
        mol = None
        try:
            if pd.notna(smi):
                mol = Chem.MolFromSmiles(str(smi))
        except Exception as e:
            logger.warning(f"SMILES 解析失败: {smi!r}, 错误: {e}")
            mol = None
        if mol is None:
            rows.append([np.nan] * len(RDKIT_DESCRIPTOR_NAMES))
            continue
        vals = []
        for name in RDKIT_DESCRIPTOR_NAMES:
            try:
                vals.append(float(desc_funcs[name](mol)))
            except Exception as e:
                logger.warning(f"RDKit 描述符计算失败 索引 {i} 描述符 {name}: {e}")
                vals.append(np.nan)
        rows.append(vals)
    return np.array(rows, dtype=np.float32)


def build_compound_features(
    smiles_list: list[str],
    stats: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    cache_path: Path | None = L4_RESULTS / "compound_features_v31.npz",
    cache_version: str = "v31",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构建化合物特征矩阵：ECFP4 + MACCS + RDKit 描述符（支持缓存）

    将 ECFP4 指纹（2048 bit）、MACCS 密钥（167 bit）和 RDKit 分子描述符
    拼接为统一特征向量，并对描述符列进行标准化（训练集拟合，测试集复用统计量）。

    仅在训练模式（stats is None）下读写缓存；验证/预测阶段传入 stats 时不使用缓存，
    避免不同数据拆分导致特征不一致。

    Args:
        smiles_list: SMILES 字符串列表
        stats: (mean, std, col_mean) 训练集标准化统计量，None 则从当前数据拟合
        cache_path: 训练模式下的缓存文件路径，None 则禁用缓存
        cache_version: 缓存版本标识，代码/配置变更后需更新以强制重新计算

    Returns:
        (features, mean, std, col_mean): 特征矩阵 (n, dim) 和标准化统计量
    """
    # 训练模式：尝试加载缓存
    if stats is None and cache_path is not None and cache_path.exists():
        try:
            logger.info(f"  从缓存加载化合物特征: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)

            # 多维度缓存校验（版本、长度、SMILES列表、特征维度）
            cached_version = str(data.get("version", ""))
            if cached_version != cache_version:
                logger.warning(
                    f"  缓存版本不匹配 (缓存 {cached_version!r} vs 当前 {cache_version!r})，"
                    f"但仍校验 SMILES 列表以决定复用"
                )

            cached_smiles = [str(s) for s in data["smiles"]]
            if len(cached_smiles) != len(smiles_list):
                logger.warning(
                    f"  缓存长度不匹配 (缓存 {len(cached_smiles)} vs 当前 {len(smiles_list)})，重新计算"
                )
            elif cached_smiles != list(smiles_list):
                logger.warning(
                    f"  缓存 SMILES 列表内容不匹配，重新计算"
                )
            else:
                features = data["features"].astype(np.float32)
                mean = data["mean"].astype(np.float32)
                std = data["std"].astype(np.float32)
                col_mean = data["col_mean"].astype(np.float32)
                
                logger.info(
                    f"  缓存命中: {features.shape[0]} compounds, dim={features.shape[1]}, "
                    f"version={cached_version or 'unknown'}"
                )
                return features, mean, std, col_mean
        except Exception as e:
            logger.warning(f"  加载化合物特征缓存失败: {e}，重新计算")

    logger.info(f"  computing ECFP4 ({len(smiles_list)} compounds)...")
    ecfp4 = _compute_ecfp4(smiles_list)
    logger.info(f"  computing MACCS ({len(smiles_list)} compounds)...")
    maccs = _compute_maccs(smiles_list)
    logger.info(f"  computing RDKit descriptors ({len(smiles_list)} compounds)...")
    desc = _compute_rdkit_descriptors(smiles_list)

    if stats is None:
        col_mean = np.nanmean(desc, axis=0)
        inds = np.where(np.isnan(desc))
        desc[inds] = np.take(col_mean, inds[1])
        desc = np.nan_to_num(desc, nan=0.0, posinf=1e6, neginf=-1e6)
        mean = desc.mean(axis=0)
        std = desc.std(axis=0) + EPS
        desc = (desc - mean) / std
    else:
        mean, std, col_mean = stats
        inds = np.where(np.isnan(desc))
        desc[inds] = np.take(col_mean, inds[1])
        desc = np.nan_to_num(desc, nan=0.0, posinf=1e6, neginf=-1e6)
        desc = (desc - mean) / (std + EPS)

    features = np.hstack([ecfp4, maccs, desc]).astype(np.float32)

    # 训练模式：保存缓存
    if stats is None and cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_path,
                features=features,
                mean=mean,
                std=std,
                col_mean=col_mean,
                smiles=np.array(smiles_list, dtype=object),
                version=cache_version,
            )
            logger.info(f"  化合物特征缓存已保存: {cache_path} (version={cache_version})")
        except Exception as e:
            logger.warning(f"  保存化合物特征缓存失败: {e}")

    return features, mean, std, col_mean


def compute_aac(sequences: list[str]) -> np.ndarray:
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
    aac_matrix = np.zeros((len(sequences), 20), dtype=np.float32)
    for i, seq in enumerate(sequences):
        if not seq or pd.isna(seq):
            continue
        seq = str(seq).upper().strip()
        total = len(seq)
        if total == 0:
            continue
        for aa in seq:
            if aa in aa_to_idx:
                aac_matrix[i, aa_to_idx[aa]] += 1
        aac_matrix[i] /= total
    return aac_matrix


def compute_esm2_embeddings(
    gene_to_seq: dict[str, str],
    cache_path: Path | None = None,
    model_name: str = ESM_MODEL_NAME,
    batch_size: int = 4,
) -> dict[str, np.ndarray]:
    """使用 ESM-2 预训练蛋白质语言模型计算 per-protein 嵌入

    对每个蛋白序列通过 ESM-2 前向传播，取序列位置（排除特殊 token）的
    均值池化作为蛋白嵌入。结果缓存到磁盘避免重复计算。

    参考: Rives et al. (2021) "Biological structure and function emerge from
          scaling unsupervised learning to 250 million protein sequences", PNAS.

    Args:
        gene_to_seq: {基因符号: 氨基酸序列}
        cache_path: 缓存文件路径（.npz），None 则不缓存
        model_name: HuggingFace ESM-2 模型名
        batch_size: 推理批次大小

    Returns:
        {基因符号: embedding (np.ndarray, shape=(esm_dim,))}
    """
    if cache_path is not None and cache_path.exists():
        logger.info(f"  从缓存加载 ESM-2 嵌入: {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        embeddings = {str(k): v.astype(np.float32) for k, v in cached.items()}
        logger.info(f"  ESM-2 嵌入已加载: {len(embeddings)} 蛋白, dim={next(iter(embeddings.values())).shape[0]}")
        return embeddings

    # 使用 HuggingFace 镜像解决国内网络不可达问题
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    from transformers import EsmModel, EsmTokenizer

    logger.info(f"  加载 ESM-2 模型: {model_name} (via HF_ENDPOINT={os.environ['HF_ENDPOINT']}) ...")
    tokenizer = EsmTokenizer.from_pretrained(model_name, local_files_only=True)
    model = EsmModel.from_pretrained(model_name, local_files_only=True).to(DEVICE)
    model.eval()
    esm_dim = model.config.hidden_size
    logger.info(f"  ESM-2 嵌入维度: {esm_dim}")

    genes = sorted(gene_to_seq.keys(), key=lambda g: len(gene_to_seq.get(g, "")), reverse=True)
    embeddings: dict[str, np.ndarray] = {}

    with torch.no_grad():
        for i in range(0, len(genes), batch_size):
            batch_genes = genes[i:i + batch_size]
            batch_seqs = [gene_to_seq[g] for g in batch_genes]

            # 截断过长序列（ESM-2 最大 1024 tokens，含特殊 token 则为 1022 aa）
            max_len = ESM_MAX_LEN
            truncated_seqs = [s[:max_len] for s in batch_seqs]

            inputs = tokenizer(
                truncated_seqs, return_tensors="pt", padding=True, truncation=True,
            ).to(DEVICE)

            outputs = model(**inputs)
            # last_hidden_state: (batch, seq_len, esm_dim)
            hidden = outputs.last_hidden_state

            # 均值池化：排除 [CLS] (pos 0) 和 [EOS] (最后一个有效 token)
            attention_mask = inputs["attention_mask"]
            # 将 [CLS] 和 [EOS] 位置 mask 掉
            for b in range(attention_mask.shape[0]):
                seq_len = attention_mask[b].sum().item()
                if seq_len > 1:
                    attention_mask[b, 0] = 0        # [CLS]
                    attention_mask[b, seq_len - 1] = 0  # [EOS]

            # 安全均值池化
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_emb = (hidden * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1)
            pooled = sum_emb / count

            for j, g in enumerate(batch_genes):
                embeddings[g] = pooled[j].cpu().numpy().astype(np.float32)

            if (i + batch_size) % 20 == 0 or i + batch_size >= len(genes):
                logger.info(f"  ESM-2 嵌入进度: {min(i + batch_size, len(genes))}/{len(genes)}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **embeddings)
        logger.info(f"  ESM-2 嵌入已缓存: {cache_path}")

    return embeddings


def load_cpi_data() -> pd.DataFrame:
    cpi_path = L4_ROOT / "results" / "experimental_actives_detail_cleaned_combined.csv"
    if not cpi_path.exists():
        logger.error(f"CPI 数据文件不存在: {cpi_path}")
        sys.exit(1)
    try:
        df = pd.read_csv(cpi_path, low_memory=False)
    except Exception:
        logger.error(f"CPI 数据文件读取失败: {cpi_path}", exc_info=True)
        raise
    required = ["gene", "canonical_smiles", "uniprot_id"]
    for col in required:
        if col not in df.columns:
            logger.error(f"CPI 数据缺少列: {col}")
            sys.exit(1)
    df = df[df["canonical_smiles"].notna()].copy()
    df = df[df["canonical_smiles"].astype(str).str.strip() != ""].copy()
    # SMILES 有效性校验，过滤无效SMILES
    valid_mask = df["canonical_smiles"].apply(
        lambda s: Chem.MolFromSmiles(str(s)) is not None
    )
    n_invalid_main = (~valid_mask).sum()
    if n_invalid_main > 0:
        logger.warning(f"主CPI数据中 {n_invalid_main} 条无效SMILES已过滤")
    df = df[valid_mask].copy()
    assert len(df) > 0, "主CPI数据 SMILES 有效性过滤后为空，请检查数据文件"
    n_main = len(df)
    logger.info(f"主CPI数据: {n_main} 条记录, {df['gene'].nunique()} 个基因, "
                f"{df['canonical_smiles'].nunique()} 个唯一 SMILES")

    # 加载补充CPI数据（统一合并文件，含版本追踪）
    merged_supplement_path = L4_RESULTS / "cpi_supplement_merged.csv"
    if merged_supplement_path.exists():
        try:
            supp_df = pd.read_csv(merged_supplement_path, low_memory=False)
            logger.info(f"加载补充CPI合并文件: {merged_supplement_path.name}, {len(supp_df)} 条记录")
            # 处理补充文件中的重复列名，避免后续选取/重命名时行为未定义
            if supp_df.columns.duplicated().any():
                dup_cols = supp_df.columns[supp_df.columns.duplicated()].unique().tolist()
                logger.warning(f"补充CPI数据存在重复列: {dup_cols}，保留首次出现列")
                supp_df = supp_df.loc[:, ~supp_df.columns.duplicated()]
            # 版本追踪日志：记录每个补充文件的来源版本和记录数
            if "source_version" in supp_df.columns:
                version_counts = supp_df["source_version"].value_counts().to_dict()
                for ver, cnt in sorted(version_counts.items()):
                    logger.info(f"  补充数据版本 {ver}: {cnt} 条记录")
            # 列名兼容：统一为 canonical_smiles / uniprot_id
            if "canonical_smiles" not in supp_df.columns and "smiles" in supp_df.columns:
                supp_df = supp_df.rename(columns={"smiles": "canonical_smiles"})
            if "uniprot_id" not in supp_df.columns and "uniprot" in supp_df.columns:
                supp_df = supp_df.rename(columns={"uniprot": "uniprot_id"})
            # 若新旧列名同时存在，优先保留标准列名
            drop_alt_cols = []
            if "canonical_smiles" in supp_df.columns and "smiles" in supp_df.columns:
                drop_alt_cols.append("smiles")
            if "uniprot_id" in supp_df.columns and "uniprot" in supp_df.columns:
                drop_alt_cols.append("uniprot")
            if drop_alt_cols:
                supp_df = supp_df.drop(columns=drop_alt_cols)
            # 重命名后再次去重，防止产生重复列
            if supp_df.columns.duplicated().any():
                dup_cols = supp_df.columns[supp_df.columns.duplicated()].unique().tolist()
                logger.warning(f"补充CPI数据重命名后仍存在重复列: {dup_cols}，保留首次出现列")
                supp_df = supp_df.loc[:, ~supp_df.columns.duplicated()]
            supp_required = ["gene", "canonical_smiles", "uniprot_id"]
            supp_missing = [c for c in supp_required if c not in supp_df.columns]
            if supp_missing:
                logger.warning(f"补充CPI数据缺少列: {supp_missing}，跳过合并")
            else:
                supp_df = supp_df[supp_df["canonical_smiles"].notna()].copy()
                supp_df = supp_df[supp_df["canonical_smiles"].astype(str).str.strip() != ""].copy()
                # SMILES 有效性校验，过滤无效SMILES
                valid_mask = supp_df["canonical_smiles"].apply(
                    lambda s: Chem.MolFromSmiles(str(s)) is not None
                )
                n_invalid = (~valid_mask).sum()
                if n_invalid > 0:
                    logger.warning(f"补充CPI数据中 {n_invalid} 条无效SMILES已过滤: "
                                   f"{supp_df.loc[~valid_mask, 'gene'].tolist()}")
                supp_df = supp_df[valid_mask].copy()
                if len(supp_df) > 0:
                    supp_df = supp_df[["gene", "canonical_smiles", "uniprot_id"]].copy()
                    n_supp = len(supp_df)
                    logger.info(f"补充CPI数据: {n_supp} 条有效记录, {supp_df['gene'].nunique()} 个基因")
                    df = pd.concat([df, supp_df], ignore_index=True)
                    before_dedup = len(df)
                    df = df.drop_duplicates(subset=["gene", "canonical_smiles"], keep="first")
                    logger.info(f"合并后CPI数据: {len(df)} 条记录 (去重移除 {before_dedup - len(df)} 条)")
                else:
                    logger.warning("补充CPI数据过滤后为空，跳过合并")
        except Exception:
            logger.warning(f"补充CPI数据读取失败: {merged_supplement_path}", exc_info=True)
    else:
        logger.info(f"补充CPI合并文件不存在: {merged_supplement_path}，跳过")
    assert len(df) > 0, "CPI 数据加载后为空，请检查数据文件内容"
    assert "gene" in df.columns, "CPI 数据缺少 gene 列"
    assert "canonical_smiles" in df.columns, "CPI 数据缺少 canonical_smiles 列"
    assert "uniprot_id" in df.columns, "CPI 数据缺少 uniprot_id 列"
    logger.info(f"CPI 数据总计: {len(df)} 条记录, {df['gene'].nunique()} 个基因, "
                f"{df['canonical_smiles'].nunique()} 个唯一 SMILES")
    return df


# [Ref: 14] STRING PPI: Szklarczyk et al. (2021) Nucleic Acids Research
def load_ppi_network() -> pd.DataFrame:
    # 优先使用补充后的PPI网络（覆盖全部96个铁衰老基因）
    supplemented_path = L1_RESULTS / "ppi_network_supplemented.csv"
    dedup_path = L1_RESULTS / "ppi_network_extended_significant_edges_dedup.csv"
    significant_path = L1_RESULTS / "ppi_network_extended_significant_edges.csv"
    extended_path = L1_RESULTS / "ppi_network_extended_edges.csv"

    ppi_path = None
    if supplemented_path.exists():
        ppi_path = supplemented_path
    elif dedup_path.exists():
        ppi_path = dedup_path
    elif significant_path.exists():
        ppi_path = significant_path
    elif extended_path.exists():
        ppi_path = extended_path

    if ppi_path is not None:
        try:
            df = pd.read_csv(ppi_path, low_memory=False)
        except Exception:
            logger.error(f"PPI 网络文件读取失败: {ppi_path}", exc_info=True)
            raise
        df = df.rename(columns={"gene_a": "source", "gene_b": "target", "combined_score": "weight"})
        if df["weight"].max() > 1.0:
            df["weight"] = df["weight"] / 1000.0
        df["source"] = df["source"].astype(str).str.upper()
        df["target"] = df["target"].astype(str).str.upper()
        network_type = "补充全覆盖" if ppi_path == supplemented_path else ("DEG 显著子网" if ppi_path == significant_path else "扩展")
        assert len(df) > 0, "PPI 网络数据加载后为空，请检查数据文件内容"
        assert "source" in df.columns and "target" in df.columns, "PPI 网络缺少 source/target 列"
        logger.info(f"PPI 网络（{network_type}）: {len(df)} 条边, "
                    f"{pd.concat([df['source'], df['target']]).nunique()} 个节点")
        return df

    logger.error("PPI 网络文件不存在")
    sys.exit(1)


# [Ref: 13] KEGG PATHWAY: Kanehisa et al. (2021) Nucleic Acids Research
def load_kegg_pathways() -> dict[str, list[str]]:
    kegg_path = L2_RESULTS / "kegg_pathways" / "kegg_human_pathway_genes.tsv"
    gene_to_pathways: dict[str, list[str]] = {}

    if kegg_path.exists():
        try:
            df = pd.read_csv(kegg_path, sep="\t", low_memory=False)
        except Exception:
            logger.error(f"KEGG 通路文件读取失败: {kegg_path}", exc_info=True)
            raise
        if {"pathway_id", "gene_symbol"}.issubset(df.columns):
            for _, row in df.iterrows():
                pid = str(row["pathway_id"]).strip()
                g = str(row["gene_symbol"]).strip().upper()
                if not pid or not g:
                    continue
                if g not in gene_to_pathways:
                    gene_to_pathways[g] = []
                if pid not in gene_to_pathways[g]:
                    gene_to_pathways[g].append(pid)
            logger.info(f"KEGG 通路（L2）: {len(gene_to_pathways)} 基因, "
                        f"{df['pathway_id'].nunique()} 通路")
            return gene_to_pathways

    logger.warning("KEGG 通路数据不可用")
    return {}


# [Ref: 4] ESM-2: Rives et al. (2021) PNAS — facebook/esm2_t30_150M_UR50D
def load_protein_features(use_esm2: bool = True) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """加载蛋白特征

    v17: 默认使用 ESM-2 预训练嵌入（640维），远程同源检测能力远超 AAC。
    若 ESM-2 不可用，自动降级为 AAC + PseAAC。

    Args:
        use_esm2: 是否使用 ESM-2 嵌入（默认 True）

    Returns:
        prot_feat: {基因符号: np.ndarray}
        gene_to_seq: {基因符号: 序列字符串}
    """
    pf_path = L2_RESULTS / "target_protein_features.csv"
    pseaac_path = L2_RESULTS / "protein_pseaac.csv"
    # 优先使用残基自注意力池化嵌入 (GS-DTI/PLM-SWE 论文方法)
    # 残基级 ESM-2 → 单头自注意力池化 → 蛋白级嵌入, 捕获结合口袋级别信息
    esm_residue_stats_cache = L4_RESULTS / "esm2_residue_mean_max_std_cache.npz"  # 1920D 统计池化
    esm_residue_cache = L4_RESULTS / "esm2_residue_pooled_embeddings.npz"
    esm_cache = L4_RESULTS / "esm2_protein_embeddings.npz"
    prot_feat: dict[str, np.ndarray] = {}
    gene_to_seq: dict[str, str] = {}

    if pf_path.exists():
        try:
            df = pd.read_csv(pf_path)
        except Exception:
            logger.error(f"蛋白特征文件读取失败: {pf_path}", exc_info=True)
            raise
        for _, row in df.iterrows():
            gene = str(row["gene_symbol"]).strip().upper()
            seq = str(row["sequence"]) if pd.notna(row["sequence"]) else ""
            gene_to_seq[gene] = seq

    # 优先使用残基级 ESM-2 统计池化特征 (mean/max/std, 1920D)
    # 避免加载损坏的 8.86GB .pt 文件，同时比 640D 池化嵌入保留更丰富的残基分布信息。
    esm2_embeddings = None
    if use_esm2 and esm_residue_stats_cache.exists():
        try:
            with np.load(esm_residue_stats_cache, allow_pickle=True) as data:
                esm2_embeddings = {str(k): data[k].astype(np.float32) for k in data.files if k != "__meta__"}
            logger.info(f"v40: 从残基统计池化 ESM-2 npz 加载 {len(esm2_embeddings)} 个蛋白嵌入, dim={next(iter(esm2_embeddings.values())).shape[0]}")
            for g in esm2_embeddings:
                if g not in gene_to_seq:
                    gene_to_seq[g] = ""
        except Exception as e:
            logger.warning(f"v40: 加载残基统计池化 ESM-2 npz 失败 ({e})，将回退")

    # 若统计池化未加载，则使用 640D 残基池化嵌入
    if use_esm2 and esm2_embeddings is None and esm_residue_cache.exists():
        try:
            with np.load(esm_residue_cache, allow_pickle=True) as data:
                # 残基池化缓存键为基因名，值为 640D 蛋白级嵌入
                esm2_embeddings = {str(k): data[k].astype(np.float32) for k in data.files if k != "__meta__"}
            logger.info(f"v40: 从残基池化 ESM-2 npz 加载 {len(esm2_embeddings)} 个蛋白嵌入")
            for g in esm2_embeddings:
                if g not in gene_to_seq:
                    gene_to_seq[g] = ""
        except Exception as e:
            logger.warning(f"v40: 加载残基池化 ESM-2 npz 失败 ({e})，将回退到标准 ESM-2")

    # 若尚未加载，则尝试标准 ESM-2 npz
    if use_esm2 and esm2_embeddings is None and esm_cache.exists():
        try:
            with np.load(esm_cache, allow_pickle=True) as data:
                esm2_embeddings = {str(k): data[k].astype(np.float32) for k in data.files if k != "__meta__"}
            logger.info(f"v25: 从 ESM-2 npz 加载 {len(esm2_embeddings)} 个蛋白嵌入")
            # 用 ESM-2 中的基因扩展 gene_to_seq（序列信息从pf_path中已有的保留）
            for g in esm2_embeddings:
                if g not in gene_to_seq:
                    gene_to_seq[g] = ""
        except Exception as e:
            logger.warning(f"v24: 加载 ESM-2 npz 失败 ({e})，将走原有流程")

    genes = list(gene_to_seq.keys())

    # 若尚未从 npz 加载 ESM-2 嵌入，则尝试计算/加载缓存
    if use_esm2 and esm2_embeddings is None:
        try:
            esm2_embeddings = compute_esm2_embeddings(
                gene_to_seq, cache_path=esm_cache,
                model_name=ESM_MODEL_NAME,
            )
        except Exception as e:
            logger.warning(f"ESM-2 嵌入计算失败 ({e})，降级为 AAC + PseAAC")

    if esm2_embeddings is not None:
        # 使用 ESM-2 嵌入作为蛋白特征
        esm_dim = next(iter(esm2_embeddings.values())).shape[0]
        missing_genes = set(genes) - set(esm2_embeddings.keys())
        if missing_genes:
            logger.warning(f"ESM-2 缺失 {len(missing_genes)} 个基因的嵌入，用随机初始化填充")
            rng = np.random.RandomState(42)
            for g in missing_genes:
                esm2_embeddings[g] = rng.randn(esm_dim).astype(np.float32) * 0.01

        prot_feat = esm2_embeddings
        logger.info(f"蛋白特征 (ESM-2): {len(prot_feat)} 基因, dim={esm_dim}")
    else:
        # ---- 降级: AAC + PseAAC ----
        seqs = [gene_to_seq[g] for g in genes]
        aac = compute_aac(seqs)

        pseaac_data: dict[str, np.ndarray] = {}
        if pseaac_path.exists():
            try:
                df_pseaac = pd.read_csv(pseaac_path)
            except Exception:
                logger.error(f"PseAAC 文件读取失败: {pseaac_path}", exc_info=True)
                raise
            if "Unnamed: 0" in df_pseaac.columns:
                df_pseaac = df_pseaac.drop(columns=["Unnamed: 0"])
            if "gene_symbol" in df_pseaac.columns:
                for _, row in df_pseaac.iterrows():
                    g = str(row["gene_symbol"]).strip().upper()
                    vals = row.drop("gene_symbol").values.astype(np.float32)
                    pseaac_data[g] = vals

        pseaac_dim = 0
        if pseaac_data:
            pseaac_dim = len(next(iter(pseaac_data.values())))

        for i, g in enumerate(genes):
            aac_vec = aac[i]
            if g in pseaac_data:
                prot_feat[g] = np.concatenate([aac_vec, pseaac_data[g]])
            elif pseaac_dim > 0:
                prot_feat[g] = np.concatenate([aac_vec, np.zeros(pseaac_dim, dtype=np.float32)])
            else:
                prot_feat[g] = aac_vec

        if not prot_feat:
            for i, g in enumerate(genes):
                prot_feat[g] = aac[i]

        logger.info(f"蛋白特征 (AAC+PseAAC): {len(prot_feat)} 基因, dim={next(iter(prot_feat.values())).shape[0]}")

    assert len(prot_feat) > 0, "蛋白特征加载后为空，至少需要一个蛋白有特征"
    return prot_feat, gene_to_seq


def load_residue_esm2_features(
    graphs: dict,
    residue_pt_path: Path | str = L4_RESULTS / "esm2_150M_residue_features.pt",
    max_len_cap: int = 512,
    residue_device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """v33: 加载残基级 ESM-2 特征并建立图蛋白索引到 residue 文件索引的映射。

    大残基张量默认驻留 CPU，仅在前向时按 batch 取到 GPU，避免 8GB 显存 OOM。

    Args:
        graphs: build_graphs_and_adj 返回的图字典，需包含 gene_to_idx / n_compounds / n_proteins。
        residue_pt_path: residue 特征 .pt 文件路径。
        max_len_cap: 单个蛋白最大残基数上限（截断/填充用）。
        residue_device: 残基张量驻留设备，默认 "cpu"。

    Returns:
        embeddings: (total_residues, residue_dim) 扁平化残基嵌入（CPU）。
        offsets: (n_residue_proteins+1,) 每个蛋白在 embeddings 中的起始偏移（CPU）。
        lengths: (n_residue_proteins,) 每个蛋白的残基数（CPU）。
        prot_to_residue_idx: (n_graph_proteins,) 图蛋白局部索引 -> residue 文件索引。
        max_len: 实际使用的最大残基数（不超过 max_len_cap）。
    """
    residue_pt_path = Path(residue_pt_path)
    if not residue_pt_path.exists():
        raise FileNotFoundError(f"残基级 ESM-2 特征文件不存在: {residue_pt_path}")

    logger.info(f">>> 加载残基级 ESM-2 特征: {residue_pt_path}")
    try:
        # 使用 mmap 避免一次性加载 8.86GB 到 RAM；
        # 保持 embeddings 为 mmap-backed CPU 张量，仅在 decoder 前向时按需切片。
        data = torch.load(
            residue_pt_path, map_location="cpu",
            mmap=True, weights_only=False)
    except Exception:
        logger.error(f"残基级 ESM-2 特征加载失败: {residue_pt_path}", exc_info=True)
        raise

    # 注意：mmap 张量必须保持驻留 CPU，调用 .to() 会触发深拷贝并撑爆内存。
    embeddings = data["embeddings"]
    offsets = data["offsets"]
    lengths = data["lengths"]
    residue_genes = data.get("genes", [])
    n_residue_proteins = len(residue_genes)
    logger.info(f"  残基特征: {n_residue_proteins} 个蛋白, "
                f"total_residues={embeddings.shape[0]}, dim={embeddings.shape[1]}, "
                f"device={embeddings.device}, is_mmapped={getattr(embeddings, 'is_mmap', 'unknown')}")

    max_len = min(int(lengths.max().item()), max_len_cap)
    if max_len < 1:
        max_len = 1

    gene_to_residue_idx = {gene: i for i, gene in enumerate(residue_genes)}
    unknown_idx = n_residue_proteins  # 对应 decoder 中追加的 unknown placeholder

    n_compounds = graphs["n_compounds"]
    n_proteins = graphs["n_proteins"]
    gene_to_idx = graphs["gene_to_idx"]

    # 按图蛋白局部索引顺序收集基因名
    all_genes = [None] * n_proteins
    for gene, g_idx in gene_to_idx.items():
        if g_idx >= n_compounds:
            all_genes[g_idx - n_compounds] = gene

    prot_to_residue_idx = torch.full((n_proteins,), unknown_idx, dtype=torch.long)
    n_matched = 0
    for local_idx, gene in enumerate(all_genes):
        if gene is None:
            continue
        if gene in gene_to_residue_idx:
            prot_to_residue_idx[local_idx] = gene_to_residue_idx[gene]
            n_matched += 1
        else:
            logger.warning(f"图蛋白 {gene} (local_idx={local_idx}) 在残基特征文件中缺失，"
                           f"将使用 unknown placeholder")

    logger.info(f"  图蛋白 -> residue 索引映射: {n_matched}/{n_proteins} 个匹配, "
                f"max_len={max_len}, device={residue_device}")
    return embeddings, offsets, lengths, prot_to_residue_idx, max_len


def load_tcm_pool() -> pd.DataFrame:
    v21_path = L3_RESULTS / "tcm_compound_pool_v21_Alevel.csv"
    noleak_path = L3_RESULTS / "tcm_compound_pool_tox_filtered_noleak.csv"
    original_path = L3_RESULTS / "tcm_compound_pool_tox_filtered.csv"

    if v21_path.exists():
        tcm_path = v21_path
        source_tag = "v21综合评分A级以上版"
    elif noleak_path.exists():
        tcm_path = noleak_path
        source_tag = "去泄漏版"
    else:
        tcm_path = original_path
        source_tag = "原始版"

    if not tcm_path.exists():
        logger.error(f"TCM 候选池文件不存在: {tcm_path}")
        sys.exit(1)
    try:
        df = pd.read_csv(tcm_path, low_memory=False)
    except Exception:
        logger.error(f"TCM 候选池文件读取失败: {tcm_path}", exc_info=True)
        raise
    assert len(df) > 0, "TCM 候选池加载后为空，请检查数据文件内容"
    # 确定SMILES列并验证有效性
    tcm_smiles_col = None
    for col in ["SMILES_std", "SMILES", "smiles", "canonical_smiles"]:
        if col in df.columns:
            tcm_smiles_col = col
            break
    assert tcm_smiles_col is not None, f"TCM 候选池中无有效SMILES列，可用列: {list(df.columns)}"
    valid_mask = df[tcm_smiles_col].apply(
        lambda s: pd.notna(s) and Chem.MolFromSmiles(str(s)) is not None
    )
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        logger.warning(f"TCM 候选池中 {n_invalid} 个无效SMILES已过滤")
    df = df[valid_mask].copy()
    assert len(df) > 0, "TCM 候选池 SMILES 有效性过滤后为空，请检查数据文件"
    logger.info(f"TCM 候选池（{source_tag}）: {len(df)} 个化合物")
    return df


def load_ferroptosis_library() -> pd.DataFrame | None:
    """加载铁死亡表型分类数据集（供消融实验等外部脚本调用）"""
    pheno_file = L4_RESULTS / "phenotype_ferroptosis_dataset_v25_clean.csv"
    if pheno_file.exists():
        try:
            df = pd.read_csv(pheno_file)
        except Exception:
            logger.error(f"铁死亡表型数据集读取失败: {pheno_file}", exc_info=True)
            return None
        assert "label" in df.columns, "铁死亡表型数据集缺少 label 列"
        logger.info(f"铁死亡表型数据集: {len(df)} 个化合物 (正={(df['label']==1).sum()}, 负={(df['label']==0).sum()})")
        return df
    logger.warning(f"铁死亡表型数据集不存在: {pheno_file}")
    return None


def load_disease_edges() -> pd.DataFrame | None:
    """加载疾病-基因边数据（供消融实验等外部脚本调用）"""
    disease_file = L4_RESULTS / "disease_gene_edges.csv"
    if disease_file.exists():
        try:
            df = pd.read_csv(disease_file)
        except Exception:
            logger.error(f"疾病-基因边数据读取失败: {disease_file}", exc_info=True)
            return None
        assert "gene_symbol" in df.columns, "疾病-基因边数据缺少 gene_symbol 列"
        assert "disease_name" in df.columns, "疾病-基因边数据缺少 disease_name 列"
        logger.info(f"疾病-基因边数据: {len(df)} 条边")
        return df
    logger.warning(f"疾病-基因边数据不存在: {disease_file}")
    return None


def build_pathway_neighbors(
    gene_to_pathways: dict[str, list[str]],
    gene_to_idx: dict[str, int],
    n_compounds: int,
) -> dict[int, set]:
    """预计算同通路蛋白邻居（用于中度负样本采样）

    Returns:
        prot_to_path_neighbors: {蛋白局部索引: set(同通路其他蛋白局部索引)}
    """
    pathway_to_genes: dict[str, set] = defaultdict(set)
    for gene, paths in gene_to_pathways.items():
        if gene not in gene_to_idx:
            continue
        g_idx = gene_to_idx[gene] - n_compounds
        if g_idx < 0:
            continue
        for p in paths:
            pathway_to_genes[p].add(g_idx)

    prot_to_path_neighbors: dict[int, set] = defaultdict(set)
    for genes in pathway_to_genes.values():
        for g in genes:
            prot_to_path_neighbors[g].update(genes - {g})

    return prot_to_path_neighbors


# [Ref: 1,2,4,13,14,15] GraphSAGE[1] + HGT[2] + ESM-2[4] + KEGG[13] + STRING[14] + GSE61616[15]
def build_graphs_and_adj(
    cpi_df: pd.DataFrame,
    ppi_df: pd.DataFrame,
    gene_to_pathways: dict[str, list[str]],
    prot_feat: dict[str, np.ndarray],
    disease_df: pd.DataFrame | None = None,
    use_topology_neg: bool = False,  # 是否预计算PPI拓扑负样本
    topo_neighbors_top_k: int = 50,
    use_esm_similarity_neg: bool = False,  # 是否预计算ESM-2余弦相似度难负样本
    esm_similarity_top_k: int = 50,
) -> dict:
    """构建同质图 + 异质图 + 邻接表（v24: 可选疾病节点；v41: 可选拓扑/ESM-2负样本）"""
    # 化合物索引
    all_smiles = sorted(cpi_df["canonical_smiles"].unique())
    smi_to_idx = {s: i for i, s in enumerate(all_smiles)}
    n_compounds = len(all_smiles)

    # 蛋白索引
    ppi_genes = set()
    for _, row in ppi_df.iterrows():
        ppi_genes.add(str(row["source"]).strip().upper())
        ppi_genes.add(str(row["target"]).strip().upper())
    # 确保铁衰老96基因全部作为蛋白节点，即使无CPI/PPI数据（用于zero-shot预测）
    ferro96_genes = set()
    ferro96_file = L1_RESULTS / "ferroaging_genes_96.csv"
    if ferro96_file.exists():
        ferro96_genes = set(pd.read_csv(ferro96_file)["gene_symbol"].dropna().astype(str).str.upper().unique())
        logger.info(f"v25: 铁衰老96基因集加载: {len(ferro96_genes)} 个")
    all_genes = sorted(set(cpi_df["gene"].str.upper().unique()) | set(prot_feat.keys()) | ppi_genes | ferro96_genes)
    gene_to_idx = {g: i + n_compounds for i, g in enumerate(all_genes)}
    n_proteins = len(all_genes)
    logger.info(f"v25: 总蛋白节点 = {n_proteins} (CPI={cpi_df['gene'].nunique()}, PPI网络={len(ppi_genes)}, 铁衰老96={len(ferro96_genes)})")

    # 化合物特征
    logger.info(f"  computing compound features ({n_compounds} compounds)...")
    comp_feat, _, _, _ = build_compound_features(all_smiles)

    # 蛋白特征
    prot_feat_dim = next(iter(prot_feat.values())).shape[0] if prot_feat else 20
    prot_esm_dim = prot_feat_dim  # 保存原始 ESM-2 维度（通路拼接前），供独立投影器使用
    prot_matrix = np.zeros((n_proteins, prot_feat_dim), dtype=np.float32)
    n_no_feat = 0
    for gene, idx_offset in gene_to_idx.items():
        idx = idx_offset - n_compounds
        if gene in prot_feat:
            prot_matrix[idx] = prot_feat[gene]
        else:
            seed = hash(gene) % (2**31)
            rng = np.random.RandomState(seed)
            prot_matrix[idx] = rng.randn(prot_feat_dim).astype(np.float32) * 0.01
            n_no_feat += 1
    if n_no_feat > 0:
        logger.info(f"  无蛋白特征基因（随机初始化）: {n_no_feat}")

    # ---- v17-ESM2: 通路隶属关系特征 ----
    # 为 SAGE 模型添加通路信息的独立投影器
    # 将每个蛋白的 KEGG 通路隶属关系编码为 one-hot 向量，拼接到 ESM-2 嵌入后
    all_pathways = sorted({pid for paths in gene_to_pathways.values() for pid in paths})
    pathway_to_idx = {p: i for i, p in enumerate(all_pathways)}
    n_pathways_feat = len(all_pathways)

    if n_pathways_feat > 0:
        pathway_feat = np.zeros((n_proteins, n_pathways_feat), dtype=np.float32)
        n_proteins_with_path = 0
        for gene, paths in gene_to_pathways.items():
            if gene in gene_to_idx:
                p_idx = gene_to_idx[gene] - n_compounds
                for pid in paths:
                    if pid in pathway_to_idx:
                        pathway_feat[p_idx, pathway_to_idx[pid]] = 1.0
                        n_proteins_with_path += 1
        # 去重计数（每个蛋白只计一次）
        n_proteins_with_path = int((pathway_feat.sum(axis=1) > 0).sum())
        logger.info(f"  通路特征 (one-hot): {n_proteins_with_path}/{n_proteins} 蛋白有通路信息, "
                    f"n_pathways={n_pathways_feat}")

        # 拼接通路特征到蛋白特征矩阵
        prot_matrix = np.concatenate([prot_matrix, pathway_feat], axis=1)
        prot_feat_dim = prot_matrix.shape[1]  # 更新蛋白特征维度
    else:
        logger.warning("  通路特征为空，跳过通路信息添加")

    # 统一维度
    feat_dim = max(comp_feat.shape[1], prot_feat_dim)
    if feat_dim != comp_feat.shape[1]:
        comp_feat = np.pad(comp_feat, ((0, 0), (0, feat_dim - comp_feat.shape[1])), mode="constant")
    if feat_dim != prot_feat_dim:
        prot_matrix = np.pad(prot_matrix, ((0, 0), (0, feat_dim - prot_feat_dim)), mode="constant")

    x = torch.from_numpy(np.vstack([comp_feat, prot_matrix]))
    logger.info(f"同质图: {n_compounds + n_proteins} 节点, feat_dim={feat_dim}, prot_raw_dim={prot_feat_dim}")

    homo_adj = defaultdict(list)  # node_idx -> [neighbor_indices]

    for _, row in cpi_df.iterrows():
        smi = row["canonical_smiles"]
        gene = row["gene"]
        if smi in smi_to_idx and gene in gene_to_idx:
            src = smi_to_idx[smi]
            dst = gene_to_idx[gene]
            homo_adj[src].append(dst)
            homo_adj[dst].append(src)

    n_ppi_edges = 0
    for _, row in ppi_df.iterrows():
        src = str(row["source"]).strip().upper()
        tgt = str(row["target"]).strip().upper()
        if src in gene_to_idx and tgt in gene_to_idx:
            si = gene_to_idx[src]
            ti = gene_to_idx[tgt]
            homo_adj[si].append(ti)
            homo_adj[ti].append(si)
            n_ppi_edges += 1

    logger.info(f"同质图邻接: {len(homo_adj)} 节点, {n_ppi_edges} PPI 边")

    # 异质图邻接表（用于 HGT 分支采样）
    hetero_adj = {
        ("compound", "interacts", "protein"): defaultdict(list),
        ("protein", "ppi", "protein"): defaultdict(list),
        ("protein", "belongs_to", "pathway"): defaultdict(list),
        ("protein", "associated_with", "disease"): defaultdict(list),
        ("disease", "involves", "protein"): defaultdict(list),
    }

    # 疾病节点（GSE61616 Ferroaging DEGs）
    disease_names = []
    disease_to_idx = {}
    n_diseases = 0
    if disease_df is not None and not disease_df.empty:
        disease_names = sorted(disease_df["disease_name"].unique())
        disease_to_idx = {d: i for i, d in enumerate(disease_names)}
        n_diseases = len(disease_names)
        logger.info(f"  疾病节点: {n_diseases} 个, 疾病-蛋白边来源={len(disease_df)} 条")

    for _, row in cpi_df.iterrows():
        smi = row["canonical_smiles"]
        gene = row["gene"]
        if smi in smi_to_idx and gene in gene_to_idx:
            hetero_adj[("compound", "interacts", "protein")][smi_to_idx[smi]].append(
                gene_to_idx[gene] - n_compounds)

    for _, row in ppi_df.iterrows():
        src = str(row["source"]).strip().upper()
        tgt = str(row["target"]).strip().upper()
        if src in gene_to_idx and tgt in gene_to_idx:
            hetero_adj[("protein", "ppi", "protein")][gene_to_idx[src] - n_compounds].append(
                gene_to_idx[tgt] - n_compounds)

    for gene, paths in gene_to_pathways.items():
        if gene in gene_to_idx:
            p_idx = gene_to_idx[gene] - n_compounds
            for pid in paths:
                hetero_adj[("protein", "belongs_to", "pathway")][p_idx].append(pid)

    # 疾病-蛋白边构建（蛋白局部索引 <-> 疾病整数索引）
    if disease_df is not None and not disease_df.empty:
        for _, row in disease_df.iterrows():
            gene = str(row["gene_symbol"]).strip().upper()
            d_name = str(row["disease_name"]).strip()
            if gene in gene_to_idx and d_name in disease_to_idx:
                p_idx = gene_to_idx[gene] - n_compounds
                d_idx = disease_to_idx[d_name]
                hetero_adj[("protein", "associated_with", "disease")][p_idx].append(d_idx)
                hetero_adj[("disease", "involves", "protein")][d_idx].append(p_idx)

    # 通路索引（已在特征构建阶段计算，此处复用）
    n_pathways = n_pathways_feat

    # 通路ID完全数值化 — 将邻接表中的字符串通路ID转为整数索引，消除字符串匹配开销
    new_pt_adj = defaultdict(list)
    for prot_idx, path_list in hetero_adj[("protein", "belongs_to", "pathway")].items():
        for pid in path_list:
            if pid in pathway_to_idx:
                new_pt_adj[prot_idx].append(pathway_to_idx[pid])
    hetero_adj[("protein", "belongs_to", "pathway")] = new_pt_adj
    logger.info(f"  通路ID数值化完成: {len(new_pt_adj)} 蛋白 → {n_pathways} 通路")

    # 预计算同通路蛋白邻居（用于中度负样本采样）
    prot_to_path_neighbors = build_pathway_neighbors(gene_to_pathways, gene_to_idx, n_compounds)
    logger.info(f"  同通路蛋白邻居: {len(prot_to_path_neighbors)} 蛋白")

    # 异质图数据（用于 HGT 全图验证）
    hetero_data = HeteroData()
    hetero_data["compound"].x = torch.from_numpy(comp_feat)
    hetero_data["protein"].x = torch.from_numpy(prot_matrix)
    hetero_data["pathway"].x = torch.zeros(max(n_pathways, 1), 1, dtype=torch.float32)
    hetero_data["pathway"].n_pathways = n_pathways

    # CPI 边（v40-fix: HGTConv 需要双向消息传递，添加 protein->compound 反向边）
    cpi_edges = [[], []]
    for src, dsts in hetero_adj[("compound", "interacts", "protein")].items():
        for dst in dsts:
            cpi_edges[0].append(src)
            cpi_edges[1].append(dst)
    hetero_data["compound", "interacts", "protein"].edge_index = torch.tensor(cpi_edges, dtype=torch.long)
    rev_cpi = [cpi_edges[1][:], cpi_edges[0][:]]
    hetero_data["protein", "rev_interacts", "compound"].edge_index = torch.tensor(rev_cpi, dtype=torch.long)

    # PPI 边（v40-fix: 无向 PPI 需要双向边以支持对称消息传递）
    ppi_edges = [[], []]
    for src, dsts in hetero_adj[("protein", "ppi", "protein")].items():
        for dst in dsts:
            ppi_edges[0].append(src)
            ppi_edges[1].append(dst)
    hetero_data["protein", "ppi", "protein"].edge_index = torch.tensor(ppi_edges, dtype=torch.long)
    rev_ppi = [ppi_edges[1][:], ppi_edges[0][:]]
    hetero_data["protein", "rev_ppi", "protein"].edge_index = torch.tensor(rev_ppi, dtype=torch.long)

    # 通路边（v12: 通路ID已数值化，dst 已是整数，无需再次转换）
    pt_edges = [[], []]
    for src, dsts in hetero_adj[("protein", "belongs_to", "pathway")].items():
        for dst in dsts:
            pt_edges[0].append(src)
            pt_edges[1].append(dst)
    hetero_data["protein", "belongs_to", "pathway"].edge_index = torch.tensor(pt_edges, dtype=torch.long)
    rev_pt = [pt_edges[1][:], pt_edges[0][:]]
    hetero_data["pathway", "includes", "protein"].edge_index = torch.tensor(rev_pt, dtype=torch.long)

    # 疾病边加入异质图
    if n_diseases > 0:
        pd_edges = [[], []]
        for src, dsts in hetero_adj[("protein", "associated_with", "disease")].items():
            for dst in dsts:
                pd_edges[0].append(src)
                pd_edges[1].append(dst)
        hetero_data["protein", "associated_with", "disease"].edge_index = torch.tensor(pd_edges, dtype=torch.long)
        rev_pd = [pd_edges[1][:], pd_edges[0][:]]
        hetero_data["disease", "involves", "protein"].edge_index = torch.tensor(rev_pd, dtype=torch.long)
        hetero_data["disease"].x = torch.zeros(n_diseases, 1, dtype=torch.float32)
        logger.info(f"v25: disease edges = {len(pd_edges[0])}")

    logger.info(f"异质图: compound({n_compounds}) protein({n_proteins}) pathway({n_pathways}) disease({n_diseases}) | "
                f"CPI={len(cpi_edges[0])} PPI={len(ppi_edges[0])} Pathway={len(pt_edges[0])}")

    # Opt1: 预计算全图同质边索引，验证/预测直接复用（速度提升 10x+）
    homo_edge_list = []
    for node in range(n_compounds + n_proteins):
        for nbr in homo_adj.get(node, []):
            homo_edge_list.append([node, nbr])
    homo_edge_index = torch.tensor(homo_edge_list, dtype=torch.long).t().contiguous() if homo_edge_list else torch.zeros((2, 0), dtype=torch.long)
    logger.info(f"预计算全图边索引: {homo_edge_index.shape[1]} 条边")

    # 预计算基于PPI拓扑的负样本邻居（可选，默认关闭以避免训练启动开销）
    prot_to_topo_medium_neighbors: dict[int, set] | None = None
    prot_to_topo_hard_neighbors: dict[int, set] | None = None
    if use_topology_neg:
        active_genes = {str(g).strip().upper() for g in cpi_df["gene"].dropna().unique()}
        logger.info(
            f"v23-topo: 预计算PPI拓扑负样本 (active_genes={len(active_genes)}, top_k={topo_neighbors_top_k}) ..."
        )
        sampler = None
        try:
            from iron_aging_gnn.graph.topology_negative_sampling import TopologyNegativeSampler
            sampler = TopologyNegativeSampler(ppi_df)
        except Exception:
            logger.exception("v23-topo: 初始化 TopologyNegativeSampler 失败，回退到无拓扑负样本")
        if sampler is not None:
            prot_to_topo_medium_neighbors = build_topology_medium_neighbors(
                ppi_df,
                gene_to_idx=gene_to_idx,
                n_compounds=n_compounds,
                active_genes=active_genes,
                top_k=topo_neighbors_top_k,
                sampler=sampler,
            )
            prot_to_topo_hard_neighbors = build_topology_hard_neighbors(
                ppi_df,
                gene_to_idx=gene_to_idx,
                n_compounds=n_compounds,
                active_genes=active_genes,
                top_k=topo_neighbors_top_k,
                sampler=sampler,
            )

    # 预计算基于 ESM-2 余弦相似度的难负样本邻居（可选，默认关闭）
    prot_to_esm_hard_neighbors: dict[int, set] | None = None
    if use_esm_similarity_neg:
        active_genes = {str(g).strip().upper() for g in cpi_df["gene"].dropna().unique()}
        logger.info(
            f"v41: 预计算ESM-2余弦相似度难负样本 "
            f"(active_genes={len(active_genes)}, top_k={esm_similarity_top_k}) ..."
        )
        try:
            from iron_aging_gnn.graph.esm_similarity_negative_sampling import (
                build_esm_similarity_hard_neighbors,
            )
            prot_to_esm_hard_neighbors = build_esm_similarity_hard_neighbors(
                esm2_embeddings=prot_feat,
                gene_to_idx=gene_to_idx,
                n_compounds=n_compounds,
                ppi_df=ppi_df,
                active_genes=active_genes,
                top_k=esm_similarity_top_k,
            )
        except ImportError as e:
            logger.warning(
                "v41: 初始化 ESM-2 相似度负样本失败，回退到无此负样本: %s", e
            )
        except Exception:
            logger.exception(
                "v41: 初始化 ESM-2 相似度负样本失败，回退到无此负样本"
            )

    return {
        "x": x,
        "feat_dim": feat_dim,
        "prot_feat_dim": prot_feat_dim,  # 蛋白特征总维度（ESM2 + 通路 one-hot），供 padding 计算
        "prot_esm_dim": prot_esm_dim,  # 原始 ESM-2 维度（640），供独立投影器使用
        "n_compounds": n_compounds,
        "n_proteins": n_proteins,
        "smi_to_idx": smi_to_idx,
        "gene_to_idx": gene_to_idx,
        "homo_adj": homo_adj,
        "homo_edge_index": homo_edge_index,
        "hetero_adj": hetero_adj,
        "hetero_data": hetero_data,
        "n_pathways": n_pathways,
        "n_diseases": n_diseases,
        "disease_to_idx": disease_to_idx,
        "prot_to_path_neighbors": prot_to_path_neighbors,  # 同通路蛋白邻居（中度负样本）
        "prot_to_topo_medium_neighbors": prot_to_topo_medium_neighbors,  # v23-topo
        "prot_to_topo_hard_neighbors": prot_to_topo_hard_neighbors,  # v23-topo
        "prot_to_esm_hard_neighbors": prot_to_esm_hard_neighbors,  # ESM-2余弦相似度难负样本
    }


def drop_edge(edge_index: torch.Tensor, p: float = 0.15) -> torch.Tensor:
    """DropEdge 正则化：随机丢弃 p 比例的边，缓解过拟合与过平滑

    参考: Rong et al. (2020) "DropEdge: Towards Deep Graph Neural Networks", ICLR
    """
    if p <= 0 or edge_index.shape[1] <= 1:
        return edge_index
    mask = torch.rand(edge_index.shape[1], device=edge_index.device) > p
    return edge_index[:, mask]


def sample_homo_subgraph(
    seed_compounds: list[int],
    homo_adj: dict[int, list[int]],
    num_neighbors: list[int] = None,
    seed: int | None = None,
):
    """GraphSAGE 风格邻居采样：固定每层邻居数，避免邻居爆炸（v12: 支持种子固定可复现）"""
    if num_neighbors is None:
        num_neighbors = [32, 16]
    if seed is not None:
        random.seed(seed)
    nodes = set(seed_compounds)
    frontier = set(seed_compounds)

    for hop_neighbors in num_neighbors:
        next_frontier = set()
        for node in frontier:
            nbrs = homo_adj.get(node, [])
            if len(nbrs) > hop_neighbors:
                nbrs = random.sample(nbrs, hop_neighbors)
            next_frontier.update(nbrs)
        nodes.update(next_frontier)
        frontier = next_frontier

    node_list = sorted(nodes)
    node_to_local = {n: i for i, n in enumerate(node_list)}

    # 子图边
    edge_list = []
    for node in node_list:
        for nbr in homo_adj.get(node, []):
            if nbr in node_to_local:
                edge_list.append([node_to_local[node], node_to_local[nbr]])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.zeros((2, 0), dtype=torch.long)

    return node_list, node_to_local, edge_index


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = FOCAL_GAMMA,
    alpha: float = FOCAL_ALPHA,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal Loss：自动降低易分类样本的损失权重，聚焦困难样本

    参考: Lin et al. (2017) "Focal Loss for Dense Object Detection", ICCV
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)  # p_t = sigmoid(logits) if target=1, else 1-sigmoid(logits)
    focal_weight = (1 - pt) ** gamma
    if alpha > 0:
        alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
        focal_weight = alpha_t * focal_weight
    loss = focal_weight * bce
    return loss.mean() if reduction == "mean" else loss.sum()


# MemoryBank 类已移至 iron_aging_gnn.models.memory_bank，从模块导入

def infonce_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    memory_scores: torch.Tensor | None = None,
    temperature: float = 0.07,
    memory_weight: float = 0.3,
) -> torch.Tensor:
    """InfoNCE 对比损失：最大化正样本对相似度，最小化负样本对相似度

    L = -log( exp(pos/τ) / (exp(pos/τ) + Σ exp(neg/τ) + Σ exp(mem/τ)) )

    当 memory_scores 不为 None 时，将 memory bank 中的全局负样本
    纳入分母，迫使决策边界更精确。

    参考:
      - Oord et al. (2018) "Representation Learning with Contrastive
        Predictive Coding", arXiv.
      - He et al. (2020) "MoCo", CVPR.

    Args:
        pos_scores: (N,) 正样本对得分
        neg_scores: (N, K) 或 (N,) 负样本对得分
        memory_scores: (N, M) 或 None，memory bank 全局负样本得分
        temperature: 温度参数 τ（默认 0.07）
        memory_weight: memory bank 负样本权重（0~1）

    Returns:
        scalar loss
    """
    pos = pos_scores / temperature
    neg = neg_scores / temperature

    if neg.dim() == 1:
        neg = neg.unsqueeze(1)

    # 分母：正样本 + 所有负样本
    denominator = torch.cat([pos.unsqueeze(1), neg], dim=1)

    if memory_scores is not None and memory_scores.numel() > 0:
        mem = memory_scores / temperature
        if mem.dim() == 1:
            mem = mem.unsqueeze(1)
        # 加权融合 memory bank 负样本
        denominator = torch.cat([denominator, memory_weight * mem], dim=1)

    loss = -pos + torch.logsumexp(denominator, dim=1)
    return loss.mean()


# _check_tensor_nan 和 _check_gradient_norm 与
#   src/iron_aging_gnn/training/trainer.py 重复定义。
# 主脚本保留内联版本供 train_sage/train_hgt 直接调用，
# src/ 模块版本供 trainer.py 内部使用。
# 后续完整迁移后应统一。
def _check_gpu_memory(min_free_gb: float = 1.0) -> bool:
    """检查 GPU 显存是否足够，返回 True/False"""
    if not torch.cuda.is_available():
        return True  # CPU 模式无需检查
    try:
        free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 3)  # 剩余显存 GB
        total_mem = torch.cuda.mem_get_info()[1] / (1024 ** 3)
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        if free_mem < min_free_gb:
            logger.warning(
                f"GPU 显存不足: 剩余 {free_mem:.1f}GB / 总 {total_mem:.1f}GB "
                f"(需要至少 {min_free_gb:.1f}GB), "
                f"已分配 {allocated:.2f}GB, 已预留 {reserved:.2f}GB")
            return False
        logger.info(
            f"GPU 显存: 剩余 {free_mem:.1f}GB / 总 {total_mem:.1f}GB, "
            f"已分配 {allocated:.2f}GB, 已预留 {reserved:.2f}GB")
        return True
    except Exception as e:
        logger.warning(f"无法获取 GPU 显存信息: {e}", exc_info=True)
        return True  # 获取失败时不阻塞执行，但记录详细信息


def _log_gpu_memory(tag: str = "") -> None:
    """记录当前 GPU 显存使用情况（用于关键步骤前后）"""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 3)
    logger.info(f"GPU显存 [{tag}]: 已分配={allocated:.2f}GB, 已预留={reserved:.2f}GB, 剩余={free_mem:.1f}GB")


def _log_step_time(start_time: float, step_name: str) -> float:
    """记录步骤耗时并返回当前时间（用于计时链）"""
    elapsed = time.time() - start_time
    if elapsed > 60:
        logger.info(f"  [TIMING] {step_name}: {elapsed / 60:.1f} 分钟")
    else:
        logger.info(f"  [TIMING] {step_name}: {elapsed:.1f} 秒")
    return time.time()


def _handle_oom_and_retry(
    fn, *args, fallback_fn=None, max_retries: int = 2, **kwargs
) -> Any:
    """OOM 降级重试：CUDA OOM 时自动清理缓存并重试，超过重试次数后调用 fallback。

    Args:
        fn: 主函数
        fallback_fn: 降级函数（如 mini-batch 版本的验证函数）
        max_retries: 最大重试次数
        *args, **kwargs: 传递给 fn 和 fallback_fn 的参数

    Returns:
        fn 或 fallback_fn 的返回值

    Raises:
        RuntimeError: 如果 fallback_fn 也为 None 且重试耗尽
    """
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and attempt < max_retries:
                logger.warning(
                    f"CUDA OOM 检测 (尝试 {attempt + 1}/{max_retries + 1}), "
                    f"清理缓存并重试...")
                torch.cuda.empty_cache()
                time.sleep(1)  # 等待 GPU 释放
                continue
            if fallback_fn is not None:
                logger.warning(
                    f"OOM 无法恢复，切换到降级方案: {fallback_fn.__name__}")
                torch.cuda.empty_cache()
                return fallback_fn(*args, **kwargs)
            logger.error(f"OOM 且无降级方案: {e}", exc_info=True)
            raise


def _check_tensor_nan(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """检查张量是否包含 NaN 或 Inf，返回 True 表示有问题"""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        logger.warning(f"张量 {name} 包含 NaN 或 Inf: "
                       f"NaN={torch.isnan(tensor).sum().item()}, "
                       f"Inf={torch.isinf(tensor).sum().item()}")
        return True
    return False


def _check_gradient_norm(model: nn.Module, warn_threshold: float = 100.0) -> float:
    """计算模型梯度总范数，若超过阈值则记录警告"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    if total_norm > warn_threshold:
        logger.warning(f"梯度范数异常: {total_norm:.1f} > {warn_threshold:.1f}")
    return total_norm


class _CpiLossState:
    """_compute_cpi_loss 运行期状态，替代模块级全局变量。"""

    def __init__(self) -> None:
        self.nan_batch_counter: int = 0
        self.pos_oom_counter: int = 0
        self.hard_neg_oom_counter: int = 0
        self.bpr_oom_counter: int = 0


# 默认共享状态：训练器不注入 _state 时仍可跨 batch 累计 OOM/NaN 次数。
_default_cpi_loss_state = _CpiLossState()


# [Ref: 3] Focal Loss: Lin et al. (2017) ICCV (α=0.75, γ=2.0)
def _compute_cpi_loss(
    model,
    comp_emb: torch.Tensor,
    prot_emb: torch.Tensor,
    pos_src: torch.Tensor,
    pos_dst: torch.Tensor,
    comp_sorted: list[int],
    prot_map: dict[int, int],
    precomputed_pos: dict[int, list[int]],
    n_compounds: int,
    prot_to_path_neighbors: dict[int, set] | None,
    epoch: int,
    stage_epochs: int,
    memory_bank: MemoryBank,
    compound_to_prot_locals: dict[int, list[int]] | None = None,  # 向量化mask预计算映射
    use_infonce: bool = False,  # 消融实验结论 — 移除 InfoNCE 提升 SAGE +75%, HGT +23%
    bpr_weight: float = 0.4,
    use_curriculum: bool = True,  # 消融实验开关
    use_topology_neg: bool = False,  # 是否使用PPI拓扑驱动的难负样本
    prot_to_topo_medium_neighbors: dict[int, set] | None = None,
    prot_to_topo_hard_neighbors: dict[int, set] | None = None,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.75,
    _state: _CpiLossState | None = None,
) -> torch.Tensor:
    """v21: 共享的 CPI 损失计算（Focal + BPR + 课程负采样）— InfoNCE 默认关闭
    v20: 新增 bpr_weight 参数支持消融实验
    v23-topo: 新增基于PPI拓扑的难负样本选项，可替代通路共现中度负样本。
    v44: OOM/NaN 状态改为 _CpiLossState 局部实例，避免模块级全局状态污染。

    统一 SAGE 与 HGT 训练循环中的负采样与损失计算逻辑，避免重复代码。

    Args:
        model: 拥有 decode() 与 temperature 属性的模型
        comp_emb: (n_batch_compounds, out_dim) 化合物嵌入
        prot_emb: (n_batch_proteins, out_dim) 蛋白嵌入
        pos_src: (n_pos,) 正样本对的化合物局部索引
        pos_dst: (n_pos,) 正样本对的蛋白局部索引
        comp_sorted: 局部化合物索引 -> 全局化合物索引
        prot_map: 局部蛋白索引 (p_global - n_compounds) -> batch 蛋白索引
        precomputed_pos: 全局化合物 -> 正样本蛋白全局索引集合
        n_compounds: 化合物总数，用于蛋白全局/局部索引转换
        prot_to_path_neighbors: 蛋白 -> 同通路蛋白局部索引集合
        epoch: 当前 epoch（用于课程阶段判定）
        stage_epochs: 当前阶段总 epoch 数
        memory_bank: MemoryBank 实例
        use_infonce: 是否启用 InfoNCE（预训练阶段可关闭）
        use_topology_neg: 为True时，优先使用PPI拓扑邻居替代通路邻居。
        prot_to_topo_medium_neighbors: 蛋白 -> 拓扑中度负样本局部索引集合。
        prot_to_topo_hard_neighbors: 蛋白 -> 拓扑难负样本局部索引集合。
        _state: 运行期状态实例；未提供时自动创建。

    Returns:
        loss 标量张量
    """
    state = _state if _state is not None else _default_cpi_loss_state
    n_batch_prots = prot_emb.shape[0]
    T = model.temperature

    # 诊断断言，确保 prot_map 键类型正确
    if prot_map:
        assert all(isinstance(k, int) for k in prot_map.keys()),             "_compute_cpi_loss: prot_map 键必须是整数局部蛋白索引"
        assert all(0 <= v < n_batch_prots for v in prot_map.values()),             f"_compute_cpi_loss: prot_map 值越界，应在 [0, {n_batch_prots}) 内"

    # 构建 batch 蛋白位置 -> 图蛋白局部索引的逆映射，用于 residue_bilinear 解码器
    prot_inv_map = {v: k for k, v in prot_map.items()} if prot_map else {}

    def _get_residue_indices(batch_positions: torch.Tensor) -> torch.Tensor | None:
        """将 batch 蛋白位置转换为图蛋白局部索引，供 residue_bilinear 查找残基特征。"""
        if not prot_inv_map:
            return None
        return torch.tensor(
            [prot_inv_map.get(p.item(), -1) for p in batch_positions],
            device=batch_positions.device, dtype=torch.long
        )

    # 正样本
    pos_residue_idx = _get_residue_indices(pos_dst)
    try:
        pos_score = model.decode(comp_emb[pos_src], prot_emb[pos_dst], prot_residue_indices=pos_residue_idx) / T
    except torch.cuda.OutOfMemoryError as e:
        # 正样本 OOM 降级为 fast bilinear
        state.pos_oom_counter += 1
        logger.warning(
            f"_compute_cpi_loss: pos 残基路径 OOM（连续 {state.pos_oom_counter} 次），"
            f"降级为 fast bilinear: {e}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pos_score = model.decode(comp_emb[pos_src], prot_emb[pos_dst], prot_residue_indices=None) / T
    pos_score = torch.clamp(pos_score, -SCORE_CLAMP, SCORE_CLAMP)
    pos_loss = focal_loss_with_logits(
        pos_score, torch.full_like(pos_score, LABEL_SMOOTHING_POS), gamma=focal_gamma, alpha=focal_alpha)

    unique_src = pos_src.unique()
    n_unique = len(unique_src)
    if n_unique == 0 or n_batch_prots <= 1:
        return pos_loss

    batch_comp_emb = comp_emb[unique_src]
    # 全蛋白 pair-matrix 规模极大，残基注意力会导致 8GB 显存图内存爆炸；
    # 此处退化为 GNN 全局嵌入点积，仅对正样本/难负样本启用残基解码器。
    # v46: 用 torch.no_grad() 包裹，all_scores 仅用于 hard neg 选择，不需要梯度
    with torch.no_grad():
        all_scores = model.decode(
            batch_comp_emb.unsqueeze(1).expand(-1, n_batch_prots, -1).reshape(-1, model.out_dim),
            prot_emb.repeat(n_unique, 1),
            prot_residue_indices=None,
        ).reshape(n_unique, n_batch_prots) / T

    # 向量化正样本mask构建 — 使用预计算 compound_to_prot_locals 映射表
    mask = torch.zeros(n_unique, n_batch_prots, device=DEVICE)
    src_indices = []
    dst_indices = []
    if compound_to_prot_locals is not None:
        for i, src_local in enumerate(unique_src):
            src_local_val = src_local.item()
            if src_local_val < len(comp_sorted):
                src_global = comp_sorted[src_local_val]
                if src_global >= 0 and src_global in compound_to_prot_locals:
                    for p_local in compound_to_prot_locals[src_global]:
                        if p_local in prot_map:
                            p_idx = prot_map[p_local]
                            if p_idx < n_batch_prots:
                                src_indices.append(i)
                                dst_indices.append(p_idx)
    else:
        # 回退：未提供预计算映射表时使用原始双循环
        for i, src_local in enumerate(unique_src):
            src_global = comp_sorted[src_local.item()] if src_local.item() < len(comp_sorted) else -1
            if src_global >= 0 and src_global in precomputed_pos:
                for p_global in precomputed_pos[src_global]:
                    p_local = p_global - n_compounds
                    if p_local in prot_map:
                        p_idx = prot_map[p_local]
                        if p_idx < n_batch_prots:
                            mask[i, p_idx] = MASK_VAL
    if src_indices:
        mask.index_put_(
            (torch.tensor(src_indices, device=DEVICE, dtype=torch.long),
             torch.tensor(dst_indices, device=DEVICE, dtype=torch.long)),
            torch.tensor(MASK_VAL, device=DEVICE)
        )

    # 课程阶段判定（按当前阶段总 epoch 计算）
    # use_curriculum=False 时始终使用随机负样本
    if use_curriculum:
        curriculum_phase = epoch / stage_epochs
        if curriculum_phase < CURRICULUM_PHASE1:
            n_medium = n_hard = 0
        elif curriculum_phase < CURRICULUM_PHASE2:
            n_medium = int(n_unique * MEDIUM_NEG_RATIO)
            n_hard = 0
        else:
            n_hard = int(n_unique * HARD_NEG_RATIO)
            n_medium = 0
    else:
        n_medium = n_hard = 0

    # 初始化 hard_neg_scores 为随机负样本（排除正样本 + 保护全零行）
    hard_neg_scores = torch.zeros(n_unique, device=DEVICE)
    valid_mask = (mask == 0).float()
    row_sum = valid_mask.sum(dim=1)
    safe_rows = row_sum > 0
    if safe_rows.any():
        valid_mask = valid_mask / (row_sum.unsqueeze(1) + EPS_SMALL)
        # 仅对 safe_rows 采样，避免全零行触发 RuntimeError
        rand_dst = torch.multinomial(valid_mask[safe_rows], 1).squeeze(-1)
        # 随机负样本尝试使用残基路径，OOM 时降级为 fast bilinear
        rand_residue_idx = _get_residue_indices(rand_dst)
        try:
            hard_neg_scores[safe_rows] = model.decode(
                comp_emb[unique_src[safe_rows]], prot_emb[rand_dst],
                prot_residue_indices=rand_residue_idx,
            ) / T
        except torch.cuda.OutOfMemoryError as e:
            state.hard_neg_oom_counter += 1
            logger.warning(
                f"_compute_cpi_loss: hard_neg 残基路径 OOM（连续 {state.hard_neg_oom_counter} 次），"
                f"降级为 fast bilinear: {e}"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            hard_neg_scores[safe_rows] = model.decode(
                comp_emb[unique_src[safe_rows]], prot_emb[rand_dst],
                prot_residue_indices=None,
            ) / T
        hard_neg_scores = torch.clamp(hard_neg_scores, -SCORE_CLAMP, SCORE_CLAMP)

    # Phase 2: 中度负样本
    # 支持基于PPI拓扑的中度负样本，与通路共现策略互斥
    medium_neighbor_dict = prot_to_path_neighbors
    if use_topology_neg and prot_to_topo_medium_neighbors is not None:
        medium_neighbor_dict = prot_to_topo_medium_neighbors

    if n_medium > 0 and medium_neighbor_dict is not None and n_batch_prots > 2:
        medium_neg_scores = hard_neg_scores.clone()
        medium_found = torch.zeros(n_unique, dtype=torch.bool, device=DEVICE)
        for i, src_local in enumerate(unique_src):
            src_global = comp_sorted[src_local.item()] if src_local.item() < len(comp_sorted) else -1
            if src_global < 0 or src_global not in precomputed_pos:
                continue
            topo_neighbors: set = set()
            for p_global in precomputed_pos[src_global]:
                p_local = p_global - n_compounds
                if p_local >= 0 and p_local in medium_neighbor_dict:
                    topo_neighbors.update(medium_neighbor_dict[p_local])
            if not topo_neighbors:
                continue
            batch_neighbor_positions = []
            for pn in topo_neighbors:
                if pn in prot_map:
                    pi = prot_map[pn]
                    if 0 <= pi < n_batch_prots and mask[i, pi] == 0:
                        batch_neighbor_positions.append(pi)
            if batch_neighbor_positions:
                bi_t = torch.tensor(batch_neighbor_positions, device=DEVICE)
                neighbor_scores = all_scores[i, bi_t]
                best_idx = neighbor_scores.argmax()
                medium_neg_scores[i] = torch.clamp(neighbor_scores[best_idx], -SCORE_CLAMP, SCORE_CLAMP)
                medium_found[i] = True

        medium_candidates = torch.where(medium_found)[0]
        if len(medium_candidates) > 0:
            n_actual = min(n_medium, len(medium_candidates))
            perm = torch.randperm(len(medium_candidates), device=DEVICE)
            hard_neg_scores[medium_candidates[perm[:n_actual]]] = medium_neg_scores[medium_candidates[perm[:n_actual]]]

    # Phase 3: 极硬负样本
    # 支持基于PPI拓扑的难负样本（共同邻居/高Jaccard）
    # v43: 额外引入 Memory Bank 全局蛋白嵌入作为跨 batch 的极硬负样本。
    hard_neg_from_memory = torch.zeros(n_unique, device=DEVICE)
    has_memory_neg = False
    if n_hard > 0 and memory_bank.size() > 0:
        n_mem = min(256, memory_bank.size())
        mem_emb = memory_bank.sample(n_mem)
        if mem_emb.shape[0] > 0:
            # 计算当前 batch 化合物与全局 memory bank 蛋白的得分
            mem_scores = model.decode(
                batch_comp_emb.unsqueeze(1).expand(-1, mem_emb.shape[0], -1).reshape(-1, model.out_dim),
                mem_emb.repeat(n_unique, 1),
                prot_residue_indices=None,
            ).reshape(n_unique, -1)
            mem_hard_scores, _ = mem_scores.max(dim=1)
            hard_neg_from_memory = torch.clamp(mem_hard_scores, -SCORE_CLAMP, SCORE_CLAMP)
            has_memory_neg = True

    if n_hard > 0:
        if use_topology_neg and prot_to_topo_hard_neighbors is not None:
            hard_neg_scores_topo = hard_neg_scores.clone()
            hard_found = torch.zeros(n_unique, dtype=torch.bool, device=DEVICE)
            for i, src_local in enumerate(unique_src):
                src_global = comp_sorted[src_local.item()] if src_local.item() < len(comp_sorted) else -1
                if src_global < 0 or src_global not in precomputed_pos:
                    continue
                topo_hard: set = set()
                for p_global in precomputed_pos[src_global]:
                    p_local = p_global - n_compounds
                    if p_local >= 0 and p_local in prot_to_topo_hard_neighbors:
                        topo_hard.update(prot_to_topo_hard_neighbors[p_local])
                if not topo_hard:
                    continue
                batch_hard_positions = []
                for pn in topo_hard:
                    if pn in prot_map:
                        pi = prot_map[pn]
                        if 0 <= pi < n_batch_prots and mask[i, pi] == 0:
                            batch_hard_positions.append(pi)
                if batch_hard_positions:
                    bi_t = torch.tensor(batch_hard_positions, device=DEVICE)
                    hard_neighbor_scores = all_scores[i, bi_t]
                    best_idx = hard_neighbor_scores.argmax()
                    hard_neg_scores_topo[i] = torch.clamp(hard_neighbor_scores[best_idx], -SCORE_CLAMP, SCORE_CLAMP)
                    hard_found[i] = True

            if has_memory_neg:
                # 取拓扑难负样本与 memory bank 难负样本中的更高分者
                hard_neg_scores_topo = torch.maximum(hard_neg_scores_topo, hard_neg_from_memory)
                hard_found = hard_found | (hard_neg_from_memory > hard_neg_scores)

            hard_candidates = torch.where(hard_found)[0]
            if len(hard_candidates) > 0:
                n_actual = min(n_hard, len(hard_candidates))
                perm = torch.randperm(len(hard_candidates), device=DEVICE)
                selected = hard_candidates[perm[:n_actual]]
                hard_neg_scores[selected] = hard_neg_scores_topo[selected]
        else:
            hard_neg_idx = (all_scores + mask).argmax(dim=1)
            hard_scores = all_scores[torch.arange(n_unique, device=DEVICE), hard_neg_idx]
            if has_memory_neg:
                hard_scores = torch.maximum(hard_scores, hard_neg_from_memory)
            hard_scores = torch.clamp(hard_scores, -SCORE_CLAMP, SCORE_CLAMP)
            hard_candidates = torch.randperm(n_unique, device=DEVICE)[:n_hard]
            hard_neg_scores[hard_candidates] = hard_scores[hard_candidates]

    # Focal Loss
    neg_loss = focal_loss_with_logits(
        hard_neg_scores, torch.full_like(hard_neg_scores, LABEL_SMOOTHING_NEG), gamma=focal_gamma, alpha=focal_alpha)

    # 正样本对 -> unique_src 位置映射（用于 BPR 和 InfoNCE）
    src_to_pos = {s.item(): i for i, s in enumerate(unique_src)}
    pos_indices = torch.tensor(
        [src_to_pos[s.item()] for s in pos_src],
        device=DEVICE, dtype=torch.long
    )

    # BPR 损失
    pair_mask = mask[pos_indices]
    bpr_valid_mask = (pair_mask == 0).float()
    bpr_row_sum = bpr_valid_mask.sum(dim=1)
    bpr_safe = bpr_row_sum > 0
    bpr_neg_scores = torch.zeros(len(pos_src), device=DEVICE)
    if bpr_safe.any():
        bpr_valid_mask[bpr_safe] = bpr_valid_mask[bpr_safe] / bpr_row_sum[bpr_safe].unsqueeze(1)
        # 仅对 bpr_safe 行采样，避免全零行触发 RuntimeError
        bpr_neg_dst = torch.multinomial(bpr_valid_mask[bpr_safe], 1).squeeze(-1)
        # BPR 负样本尝试使用残基注意力路径（与正样本一致以真正训练残基解码器），
        # 但在显存不足时降级为 fast bilinear（与 all_scores 路径一致），并明确打印警告。
        # 这是工程妥协而非掩盖错误：BPR 只占总损失的 bpr_weight*1.0(0.4)，部分降级不会毁掉训练。
        bpr_neg_residue_idx = _get_residue_indices(bpr_neg_dst)
        try:
            bpr_neg_scores[bpr_safe] = model.decode(
                comp_emb[pos_src[bpr_safe]], prot_emb[bpr_neg_dst],
                prot_residue_indices=bpr_neg_residue_idx,
            ) / T
        except torch.cuda.OutOfMemoryError as e:
            state.bpr_oom_counter += 1
            logger.warning(
                f"_compute_cpi_loss: BPR 残基路径 OOM（连续 {state.bpr_oom_counter} 次），"
                f"降级为 fast bilinear: {e}"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 降级为 fast bilinear：使用正样本与负样本的全局嵌入
            bpr_neg_scores[bpr_safe] = model.decode(
                comp_emb[pos_src[bpr_safe]], prot_emb[bpr_neg_dst],
                prot_residue_indices=None,
            ) / T
    # 对unsafe行使用(all_scores + mask)中最低分蛋白作为替代负样本
    # mask 已将正样本设为 MASK_VAL=-1e9，确保 min 不会选到正样本
    bpr_unsafe = ~bpr_safe
    if bpr_unsafe.any():
        bpr_neg_scores[bpr_unsafe] = (all_scores[pos_indices[bpr_unsafe]] + mask[pos_indices[bpr_unsafe]]).min(dim=1).values
    bpr_loss = -torch.log(torch.sigmoid(pos_score - bpr_neg_scores) + EPS).mean()

    loss = CPI_LOSS_WEIGHT * (pos_loss + neg_loss) + bpr_weight * bpr_loss

    # InfoNCE
    # 原 epoch > 50 在预训练(10) + 微调(15) 周期下永不触发，改为按阶段 epoch 比例触发
    infonce_warmup = max(2, int(stage_epochs * INFONCE_WARMUP_RATIO))
    if use_infonce and epoch > infonce_warmup and memory_bank.size() > 0 and len(pos_indices) > 0:
        n_mem = min(INFONCE_MEM_SAMPLE, memory_bank.size())
        mem_emb = memory_bank.sample(n_mem)
        if mem_emb.shape[0] > 0:
            pos_idx_sub = pos_indices[:len(pos_score)]
            mem_scores = model.decode(
                comp_emb[unique_src[pos_idx_sub]].unsqueeze(1).expand(-1, n_mem, -1).reshape(-1, model.out_dim),
                mem_emb.repeat(len(pos_idx_sub), 1),
                prot_residue_indices=None,  # Memory Bank 无对应蛋白索引，residue_bilinear 退化为点积
            ).reshape(len(pos_idx_sub), n_mem)
            infonce = infonce_loss(
                pos_score[:len(pos_idx_sub)] * T,
                hard_neg_scores[pos_idx_sub] * T,
                memory_scores=mem_scores, temperature=INFONCE_TEMPERATURE,
            )
            loss = loss + INFONCE_WEIGHT * infonce

    if torch.isnan(loss) or torch.isinf(loss):
        state.nan_batch_counter += 1
        if state.nan_batch_counter >= 5:
            raise RuntimeError(
                f"_compute_cpi_loss 连续 {state.nan_batch_counter} 个 batch 产生 NaN/Inf loss，"
                f"可能是梯度爆炸或数据异常，请检查学习率、模型初始化或输入数据")
        logger.warning(
            f"_compute_cpi_loss 产生 NaN/Inf loss（连续 {state.nan_batch_counter}/5），"
            f"返回零损失以保护训练")
        return torch.tensor(0.0, device=loss.device, requires_grad=False)
    else:
        state.nan_batch_counter = 0  # 正常 batch 重置计数器

    return loss


def _split_head_tail_nodes(
    train_compounds: list[int],
    compound_to_pos: dict[int, set],
    head_ratio: float = 0.2,
    lambda_hhi: float = 1.0,
    seed: int = 42,
    head_undersample_ratio: float = 0.6,
) -> tuple[list[int], list[int]]:
    """v19: 社区感知头尾节点划分 — 简化版 HHI 评分

    参考王煦 CTCL-DPI: Score_v = ln(d_v + 1) * (1 + lambda * HHI_v)。
    由于当前数据未显式给出二部图社区标签，这里用化合物度（已知靶标数）
    近似 HHI：度越低，HHI 越接近 1/degree（邻域越"单一"），越可能是尾节点。

    Returns:
        pretrain_compounds: 尾节点保留 + 头节点欠采样后的子图化合物列表
        tail_compounds: 尾节点化合物列表（用于日志/分析）
    """
    rng = random.Random(seed)
    scores = {}
    for c in train_compounds:
        pos_set = compound_to_pos.get(c, set())
        degree = len(pos_set)
        # 近似 HHI：靶标越少，邻域越集中，HHI 越高
        hhi = 1.0 / max(degree, 1)
        score = math.log(degree + 1) * (1.0 + lambda_hhi * hhi)
        scores[c] = score

    sorted_comps = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    n_head = max(1, int(len(sorted_comps) * head_ratio))
    head_compounds = [c for c, _ in sorted_comps[:n_head]]
    tail_compounds = [c for c, _ in sorted_comps[n_head:]]

    # 头节点欠采样至指定比例，尾节点全部保留
    n_head_keep = max(1, int(len(head_compounds) * head_undersample_ratio))
    rng.shuffle(head_compounds)
    head_kept = head_compounds[:n_head_keep]

    pretrain_compounds = tail_compounds + head_kept
    rng.shuffle(pretrain_compounds)
    return pretrain_compounds, tail_compounds


def _build_val_safe_homo_edge_index(
    homo_edge_index: torch.Tensor,
    n_compounds: int,
    val_comp_set: set,
    val_prot_set: set = None,
) -> torch.Tensor:
    """v18: 构建严格验证安全的同质图边索引

    为真实评估蛋白冷启动泛化能力，必须让验证蛋白在图中完全孤立：
      - 移除所有一端是验证集化合物的边
      - 移除所有一端是验证集蛋白的边（包括 PPI、CPI）

    Args:
        homo_edge_index: (2, E) 全图同质边索引
        n_compounds: 化合物节点数
        val_comp_set: 验证集化合物全局索引集合
        val_prot_set: 验证集蛋白局部索引集合（0-based，相对于 n_compounds）

    Returns:
        (2, E') 过滤后的边索引
    """
    if not val_comp_set and not val_prot_set:
        return homo_edge_index
    src = homo_edge_index[0]
    dst = homo_edge_index[1]

    # 全局索引集合转张量，用于向量化 isin
    val_comp_tensor = torch.tensor(sorted(val_comp_set), dtype=torch.long, device=homo_edge_index.device)

    # 移除所有涉及验证集化合物的边
    src_in_val_comp = torch.isin(src, val_comp_tensor)
    dst_in_val_comp = torch.isin(dst, val_comp_tensor)
    remove = src_in_val_comp | dst_in_val_comp

    # 移除所有涉及验证集蛋白的边（局部索引 -> 全局索引）
    if val_prot_set:
        val_prot_global = torch.tensor(
            sorted(p + n_compounds for p in val_prot_set),
            dtype=torch.long, device=homo_edge_index.device)
        src_in_val_prot = torch.isin(src, val_prot_global)
        dst_in_val_prot = torch.isin(dst, val_prot_global)
        remove = remove | src_in_val_prot | dst_in_val_prot

    mask = ~remove
    n_removed = (~mask).sum().item()
    logger.info(f"  严格验证安全同质图: 移除 {n_removed} 条边 (val_comp + val_prot 全部边), "
                f"保留 {mask.sum().item()} 条边")
    return homo_edge_index[:, mask]


def _build_val_safe_hetero_data(
    hetero_data,
    val_comp_set: set,
    val_prot_set: set = None,
) -> HeteroData:
    """v18: 构建严格验证安全的异质图

    移除所有涉及验证集化合物或验证集蛋白的边，确保验证蛋白在异质图中完全孤立。

    Args:
        hetero_data: 全图异质图数据
        val_comp_set: 验证集化合物全局索引集合
        val_prot_set: 验证集蛋白局部索引集合（0-based，相对于化合物数）

    Returns:
        过滤后的异质图数据
    """
    hetero_data_val = HeteroData()
    # 复制节点特征
    for node_type in hetero_data.node_types:
        hetero_data_val[node_type].x = hetero_data[node_type].x.clone()
        if node_type == "pathway" and hasattr(hetero_data["pathway"], "n_pathways"):
            hetero_data_val["pathway"].n_pathways = hetero_data["pathway"].n_pathways

    # 预计算过滤张量，避免每个 edge_type 重复 sorted+torch.tensor
    val_comp_tensor = torch.tensor(sorted(val_comp_set), dtype=torch.long) if val_comp_set else None
    val_prot_tensor = torch.tensor(sorted(val_prot_set), dtype=torch.long) if val_prot_set else None

    for edge_type in hetero_data.edge_types:
        edge_index = hetero_data[edge_type].edge_index
        src_type, rel, dst_type = edge_type
        keep_mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)

        # 1) 涉及验证集化合物的 CPI 边：compound -> protein
        if edge_type == ("compound", "interacts", "protein") and val_comp_tensor is not None:
            val_comp_dev = val_comp_tensor.to(edge_index.device)
            keep_mask = keep_mask & (~torch.isin(edge_index[0], val_comp_dev))

        # 2) 涉及验证集蛋白的边（PPI / protein-pathway / pathway-protein / protein-disease / disease-protein）
        if val_prot_tensor is not None:
            val_prot_dev = val_prot_tensor.to(edge_index.device)
            if src_type == "protein":
                keep_mask = keep_mask & (~torch.isin(edge_index[0], val_prot_dev))
            if dst_type == "protein":
                keep_mask = keep_mask & (~torch.isin(edge_index[1], val_prot_dev))

        n_removed = (~keep_mask).sum().item()
        if n_removed > 0:
            logger.info(f"  严格验证安全异质图: 移除 {edge_type} {n_removed} 条边, "
                        f"保留 {keep_mask.sum().item()} 条边")
        hetero_data_val[edge_type].edge_index = edge_index[:, keep_mask]

    return hetero_data_val


def _build_val_comp_cold_homo_edge_index(
    homo_edge_index: torch.Tensor,
    val_comp_set: set,
) -> torch.Tensor:
    """v18: 构建化合物冷启动验证同质图

    化合物冷启动评估中，验证化合物未在训练中出现，因此移除其所有 CPI 边。
    但蛋白（包括验证蛋白）之间的 PPI 边和通路边保留，因为蛋白侧不是冷启动对象。
    """
    if not val_comp_set:
        return homo_edge_index
    src = homo_edge_index[0]
    dst = homo_edge_index[1]
    # 预计算 val_comp_tensor，避免每次调用时 sorted+torch.tensor
    val_comp_tensor = torch.tensor(sorted(val_comp_set), dtype=torch.long, device=homo_edge_index.device)
    remove = torch.isin(src, val_comp_tensor) | torch.isin(dst, val_comp_tensor)
    mask = ~remove
    n_removed = (~mask).sum().item()
    logger.info(f"  化合物冷启动验证同质图: 移除 {n_removed} 条边 (仅 val_comp), 保留 {mask.sum().item()} 条边")
    return homo_edge_index[:, mask]


def _build_val_comp_cold_hetero_data(
    hetero_data,
    val_comp_set: set,
) -> HeteroData:
    """v18: 构建化合物冷启动验证异质图

    仅移除验证集化合物相关的 CPI 边，保留所有蛋白-蛋白/蛋白-通路/通路-蛋白边。
    """
    hetero_data_val = HeteroData()
    for node_type in hetero_data.node_types:
        hetero_data_val[node_type].x = hetero_data[node_type].x.clone()
        if node_type == "pathway" and hasattr(hetero_data["pathway"], "n_pathways"):
            hetero_data_val["pathway"].n_pathways = hetero_data["pathway"].n_pathways

    # 预计算 val_comp_tensor，避免每个 edge_type 重复 sorted+torch.tensor
    val_comp_tensor = torch.tensor(sorted(val_comp_set), dtype=torch.long) if val_comp_set else None

    for edge_type in hetero_data.edge_types:
        edge_index = hetero_data[edge_type].edge_index
        if edge_type == ("compound", "interacts", "protein") and val_comp_tensor is not None:
            val_comp_dev = val_comp_tensor.to(edge_index.device)
            keep_mask = ~torch.isin(edge_index[0], val_comp_dev)
        else:
            keep_mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
        n_removed = (~keep_mask).sum().item()
        if n_removed > 0:
            logger.info(f"  化合物冷启动验证异质图: 移除 {edge_type} {n_removed} 条边, 保留 {keep_mask.sum().item()} 条边")
        hetero_data_val[edge_type].edge_index = edge_index[:, keep_mask]
    return hetero_data_val


def _build_train_safe_homo_adj(
    homo_adj,
    n_compounds: int,
    val_comp_set: set,
    val_prot_set: set = None,
):
    """v18: 构建训练安全同质邻接表

    为真实蛋白冷启动评估，训练图必须对验证蛋白完全不可见：
      - 移除所有涉及验证集化合物的边
      - 移除所有涉及验证集蛋白的边（包括 CPI、PPI）

    这样验证蛋白在训练阶段从未参与任何消息传递，确保冷启动指标反映
    模型对全新蛋白的泛化能力。
    """
    train_adj = defaultdict(list)
    val_prot_global = {p + n_compounds for p in val_prot_set} if val_prot_set else set()
    val_nodes = set(val_comp_set) | val_prot_global
    for src, dsts in homo_adj.items():
        if src in val_nodes:
            continue
        for dst in dsts:
            if dst not in val_nodes:
                train_adj[src].append(dst)
    return train_adj


def _build_train_safe_hetero_adj(
    hetero_adj,
    n_compounds: int,
    val_comp_set: set,
    val_prot_set: set = None,
):
    """v18: 构建训练安全异质邻接表

    从训练图中彻底移除验证集化合物/蛋白相关的所有异质边。
    """
    train_adj = {}
    for et, adj in hetero_adj.items():
        new_adj = defaultdict(list)
        for src, dsts in adj.items():
            if et == ("compound", "interacts", "protein"):
                if src in val_comp_set:
                    continue
                for dst in dsts:
                    if val_prot_set is None or dst not in val_prot_set:
                        new_adj[src].append(dst)
            elif et == ("protein", "ppi", "protein"):
                if val_prot_set is not None and src in val_prot_set:
                    continue
                for dst in dsts:
                    if val_prot_set is None or dst not in val_prot_set:
                        new_adj[src].append(dst)
            elif et == ("protein", "belongs_to", "pathway"):
                if val_prot_set is not None and src in val_prot_set:
                    continue
                for dst in dsts:
                    new_adj[src].append(dst)
            elif et == ("protein", "associated_with", "disease"):
                # 验证蛋白的疾病边需移除（冷启动隔离）
                if val_prot_set is not None and src in val_prot_set:
                    continue
                for dst in dsts:
                    new_adj[src].append(dst)
            elif et == ("disease", "involves", "protein"):
                # 疾病反向边也需过滤验证蛋白，防止冷启动信息泄漏
                if val_prot_set is None:
                    new_adj[src].extend(dsts)
                else:
                    for dst in dsts:
                        if dst not in val_prot_set:
                            new_adj[src].append(dst)
            else:
                # 未知边类型不应静默处理，添加日志警告便于追踪图结构变更
                logger.warning(f"  _build_train_safe_hetero_adj: 未知边类型 {et}，未做验证集过滤，直接保留所有边")
                new_adj[src].extend(dsts)
        train_adj[et] = new_adj
    return train_adj


def _build_val_comp_cold_hetero_adj(
    hetero_adj,
    n_compounds: int,
    val_comp_set: set,
):
    """v31: 构建化合物冷启动验证异质邻接表

    仅移除验证集化合物相关的 CPI 边，保留蛋白侧所有拓扑（PPI / 通路 / 疾病）。
    用于 HGT mini-batch 化合物冷启动验证，与 _build_val_comp_cold_hetero_data 语义一致。
    """
    val_adj = {}
    for et, adj in hetero_adj.items():
        new_adj = defaultdict(list)
        for src, dsts in adj.items():
            if et == ("compound", "interacts", "protein") and src in val_comp_set:
                continue
            new_adj[src].extend(dsts)
        val_adj[et] = new_adj
    return val_adj


def _build_val_safe_hetero_adj(
    hetero_adj,
    n_compounds: int,
    val_comp_set: set,
    val_prot_set: set = None,
):
    """v18: 构建严格验证安全（蛋白冷启动）异质邻接表

    用于 HGT 蛋白冷启动 mini-batch 验证，彻底移除验证集化合物/蛋白相关边，
    确保验证蛋白在子图采样中完全孤立。
    """
    return _build_train_safe_hetero_adj(hetero_adj, n_compounds, val_comp_set, val_prot_set)


# 参考:
#   - Precision@K: 常用于推荐系统评估，衡量Top-K排序中正样本命中率
#   - Enrichment Factor (EF): 常用于虚拟筛选，EF = (正样本在Top X%命中率) / (全局正样本比例)
#     EF > 1 表示模型富集能力优于随机，EF 越大越好
#     Bender et al. (2021) "A practical guide to large-scale docking", Nature Protocols.
def _compute_ranking_metrics(score_matrix, valid_pos_list, ks=(10, 20, 50)):
    """从预计算得分矩阵中计算 Precision@K 和 Enrichment Factor (EF)。

    Args:
        score_matrix: (n_val, n_candidates) 得分矩阵（未经过 sigmoid 的原始分数）
        valid_pos_list: list of lists，每个元素为该化合物正样本的局部列索引
        ks: 用于 Precision@K 的 K 值列表

    Returns:
        dict: 包含 precision@K, ef@1%, ef@5% 指标
    """
    n_candidates = score_matrix.shape[1]
    precision_at_k = {k: [] for k in ks}
    ef_1pct_hits = 0
    ef_5pct_hits = 0
    total_positives = 0

    top_1pct_n = max(1, int(n_candidates * 0.01))
    top_5pct_n = max(1, int(n_candidates * 0.05))

    for idx, valid_pos in enumerate(valid_pos_list):
        if not valid_pos:
            continue
        scores = score_matrix[idx]
        _, sorted_indices = torch.sort(scores, descending=True)
        sorted_indices_cpu = sorted_indices.cpu().tolist()

        valid_pos_set = set(valid_pos)
        n_pos = len(valid_pos)
        total_positives += n_pos

        for k in ks:
            k_actual = min(k, n_candidates)
            top_k = sorted_indices_cpu[:k_actual]
            hits = sum(1 for p in top_k if p in valid_pos_set)
            precision_at_k[k].append(hits / k_actual)

        ef_1pct_hits += sum(1 for p in sorted_indices_cpu[:top_1pct_n] if p in valid_pos_set)
        ef_5pct_hits += sum(1 for p in sorted_indices_cpu[:top_5pct_n] if p in valid_pos_set)

    n_compounds = len([v for v in valid_pos_list if v])
    result = {}
    for k in ks:
        result[f"precision@{k}"] = float(np.mean(precision_at_k[k])) if precision_at_k[k] else 0.0

    if total_positives > 0 and n_candidates > 0:
        ef_1pct = (ef_1pct_hits / total_positives) / 0.01
        ef_5pct = (ef_5pct_hits / total_positives) / 0.05
        result["ef@1%"] = float(ef_1pct)
        result["ef@5%"] = float(ef_5pct)
    else:
        result["ef@1%"] = 1.0
        result["ef@5%"] = 1.0

    return result


def _compute_roce(y_true, y_score):
    """计算 ROCE (ROC Enrichment) — 早期富集评估指标。

    从二分类标签和预测分数中提取 ROC 曲线，计算在指定假阳性率
    (0.5%, 1.0%, 2.0%, 5.0%) 下的富集因子。

    Args:
        y_true: 真实标签 (0/1)
        y_score: 预测分数（概率值）

    Returns:
        dict: ROCE@0.5%, ROCE@1.0%, ROCE@2.0%, ROCE@5.0%
    """
    from sklearn.metrics import roc_curve
    result = {}
    fpr, tpr, _ = roc_curve(y_true, y_score)
    for pct in [0.5, 1.0, 2.0, 5.0]:
        fp_rate = pct / 100.0
        idx = np.argmin(np.abs(fpr - fp_rate))
        if fpr[idx] > 1e-8:
            roce = tpr[idx] / fpr[idx]
        else:
            roce = 0.0
        result[f"ROCE@{pct}%"] = float(roce)
    return result


def _compute_bedroc(y_true, y_score, alpha=20.0):
    """BEDROC (Boltzmann-Enhanced Discrimination of ROC) — 早期富集评估指标。

    Truchon & Bayly, J. Chem. Inf. Model. 2007, 47, 488-508.
    直接委托 RDKit CalcBEDROC 实现，避免手写公式偏差。

    Args:
        y_true: 真实标签 (0/1)
        y_score: 预测分数
        alpha: 早期富集权重 (默认 20.0)

    Returns:
        float: BEDROC 值
    """
    from rdkit.ML.Scoring.Scoring import CalcBEDROC

    n = len(y_true)
    n_act = int(np.sum(y_true))
    if n_act == 0:
        return 0.0
    if n_act == n:
        return 1.0

    order = np.argsort(y_score)[::-1]
    scores = [
        [float(y_score[order[i]]), bool(y_true[order[i]])]
        for i in range(n)
    ]
    return float(CalcBEDROC(scores, col=1, alpha=alpha))


def _validate_sage(model, x, homo_edge_index, val_compounds, all_compound_to_pos, n_compounds):
    """v18: SAGE 验证 — 批量 MLP 解码器评分，避免 Python 循环反复 forward
    
    论文引用:
      - GraphSAGE: Hamilton et al. (2017) "Inductive Representation Learning on Large Graphs", NeurIPS.
      - 药物筛选评估: Rifaioglu et al. (2021) "Recent applications of deep learning and machine
        intelligence on in silico drug discovery", Briefings in Bioinformatics."""
    with torch.no_grad():
        x_dev = x.to(DEVICE)
        edge_index = homo_edge_index.to(DEVICE)
        node_emb = model(x_dev, edge_index)  # n_compounds=None 使用 self.n_compounds
        prot_emb = node_emb[n_compounds:]
        comp_emb = node_emb[:n_compounds]
        T = model.temperature
        n_prots = prot_emb.shape[0]

        val_compounds_list = list(val_compounds)
        n_val = len(val_compounds_list)
        if n_val == 0:
            return {"auc": 0.5, "aupr": 0.5, "n_valid_compounds": 0}

        # 批量预计算 (n_val, n_prots) 得分矩阵，避免 Python 循环中反复调用 MLP
        # batch_size 512->128 降低显存峰值，避免 RTX 5060 8GB OOM 卡住
        comp_sub = comp_emb[val_compounds_list]  # (n_val, d)
        batch_size = 128
        score_chunks = []
        for start in range(0, n_val, batch_size):
            end = min(start + batch_size, n_val)
            sub_comp = comp_sub[start:end]
            sub_comp_exp = sub_comp.unsqueeze(1).expand(-1, n_prots, -1).reshape(-1, sub_comp.shape[-1])
            prot_exp = prot_emb.unsqueeze(0).expand(end - start, -1, -1).reshape(-1, prot_emb.shape[-1])
            # 验证阶段全 pair-matrix 使用快速双线性打分（prot_residue_indices=None），
            # 避免残基注意力逐对处理 n_val × n_prots 对导致严重减速/卡住。
            sub_scores = model.decode(
                sub_comp_exp, prot_exp, prot_residue_indices=None
            ).reshape(end - start, n_prots) / T
            score_chunks.append(sub_scores)
        score_matrix = torch.cat(score_chunks, dim=0)  # (n_val, n_prots)

        y_true, y_score = [], []
        valid_pos_list = []  # 收集 per-compound 正样本局部索引，用于排名指标
        n_valid = 0
        for idx, src in enumerate(val_compounds_list):
            pos_set = all_compound_to_pos.get(src, set())
            # pos_set 存全局索引，转为局部索引
            valid_pos = [p - n_compounds for p in pos_set if n_compounds <= p < n_compounds + n_prots]
            if not valid_pos:
                continue
            n_valid += 1
            valid_pos_list.append(valid_pos)

            scores = score_matrix[idx]

            # 正样本
            for p in valid_pos:
                y_true.append(1)
                y_score.append(torch.sigmoid(scores[p]).item())

            # 硬负样本（v13: 边界检查，n_prots 可能 < HARD_NEG_TOP_K）
            n_hard = min(HARD_NEG_TOP_K, n_prots - len(valid_pos))
            if n_hard > 0:
                mask = torch.zeros(n_prots, device=DEVICE)
                for p in valid_pos:
                    mask[p] = MASK_VAL
                _, hard_indices = (scores + mask).topk(n_hard)
                for hi in hard_indices:
                    if hi.item() < n_prots:
                        y_true.append(0)
                        y_score.append(torch.sigmoid(scores[hi]).item())

            # 随机负样本（v13: 边界检查）
            n_rand = min(RAND_NEG_TOP_K, n_prots - len(valid_pos))
            if n_rand > 0:
                rand_mask = torch.ones(n_prots, device=DEVICE)
                for p in valid_pos:
                    rand_mask[p] = 0
                # 排除已选中的硬负样本，避免重复采样导致 AUC/AUPR 虚高
                if n_hard > 0:
                    for hi in hard_indices:
                        if hi.item() < n_prots:
                            rand_mask[hi] = 0
                rand_candidates = torch.where(rand_mask > 0)[0]
                if len(rand_candidates) > 0:
                    n_sample = min(n_rand, len(rand_candidates))
                    rand_idx = rand_candidates[torch.randperm(len(rand_candidates), device=DEVICE)[:n_sample]]
                    for ri in rand_idx:
                        y_true.append(0)
                        y_score.append(torch.sigmoid(scores[ri]).item())

        if len(y_true) < 2 or len(set(y_true)) < 2:
            return {"auc": 0.5, "aupr": 0.5, "n_valid_compounds": n_valid}

        y_true_arr = np.array(y_true)
        y_score_arr = np.array(y_score)
        nan_mask = np.isnan(y_score_arr) | np.isinf(y_score_arr)
        if nan_mask.any():
            logger.warning(f"_validate_sage: 验证分数含 {nan_mask.sum()} 个 NaN/Inf，已过滤")
            valid_idx = ~nan_mask
            y_true_arr = y_true_arr[valid_idx]
            y_score_arr = y_score_arr[valid_idx]
            if len(y_true_arr) < 2 or len(set(y_true_arr)) < 2:
                return {"auc": 0.5, "aupr": 0.5, "n_valid_compounds": n_valid}

        # 计算排名指标（Precision@K, EF@1%, EF@5%）和 ROCE
        result = {
            "auc": float(roc_auc_score(y_true_arr, y_score_arr)),
            "aupr": float(average_precision_score(y_true_arr, y_score_arr)),
            "n_valid_compounds": n_valid,
        }
        if valid_pos_list:
            ranking = _compute_ranking_metrics(score_matrix, valid_pos_list)
            result.update(ranking)
        roce = _compute_roce(y_true_arr, y_score_arr)
        result.update(roce)
        bedroc = _compute_bedroc(y_true_arr, y_score_arr)
        result["BEDROC"] = bedroc
        return result


def _validate_hgt(
    model: HGTLinkPredictor,
    hetero_data,
    val_compounds: list[int],
    all_compound_to_pos: dict[int, set],
    n_compounds: int,
    n_proteins: int,
    hetero_adj: dict | None = None,
) -> dict[str, float]:
    """v31: HGT 强制 mini-batch 验证（禁止在完整异质图上执行全图前向计算）

    硬性约束: HGT 验证必须采用 mini-batch 子图采样，避免全图推理的 OOM 风险
    与蛋白嵌入不一致问题。该函数直接委托给 _validate_hgt_minibatch。
    """
    if hetero_adj is None:
        logger.error("  HGT mini-batch 验证失败: hetero_adj 未传入")
        return {"auc": 0.5, "aupr": 0.5, "n_valid_compounds": 0}

    return _validate_hgt_minibatch(
        model, hetero_data, hetero_adj, val_compounds,
        all_compound_to_pos, n_compounds, n_proteins)


def _validate_hgt_minibatch(
    model: HGTLinkPredictor,
    hetero_data,
    hetero_adj: dict,
    val_compounds: list[int],
    all_compound_to_pos: dict[int, set],
    n_compounds: int,
    n_proteins: int,
    num_neighbors: list[int] = None,
    val_batch_size: int = 64,
) -> dict[str, float]:
    """v16: HGT mini-batch 降级验证（OOM 时自动启用）

    对验证化合物分批采样异质子图，在各子图内计算得分后全局聚合。
    注意：降级模式下蛋白嵌入在不同子图间不一致，AUC 可能偏低。
    """
    if num_neighbors is None:
        num_neighbors = [64, 32]
    model.eval()
    with torch.no_grad():
        T = model.temperature
        all_y_true, all_y_score = [], []
        all_batch_ranking = []  # 收集每个 batch 的排名指标
        n_valid_compounds = 0

        for batch_start in range(0, len(val_compounds), val_batch_size):
            batch_seeds = val_compounds[batch_start:batch_start + val_batch_size]

            # 化合物冷启动验证中，验证化合物在 val_hetero_adj 中已移除 CPI 边，
            # 必须显式将正样本蛋白与随机负样本蛋白作为 seed_proteins 纳入子图，
            # 否则子图仅含孤立验证化合物，AUC/AUPR 会恒为 0.5。
            candidate_proteins = set()
            for s in batch_seeds:
                for p_global in all_compound_to_pos.get(s, set()):
                    p_local = p_global - n_compounds
                    if 0 <= p_local < n_proteins:
                        candidate_proteins.add(p_local)
            # 补充随机负样本，保证每个 batch 有足够候选蛋白（上限 1024）
            if candidate_proteins:
                all_prot_set = set(range(n_proteins))
                neg_pool = list(all_prot_set - candidate_proteins)
                if neg_pool:
                    rng = random.Random(42 + batch_start)
                    target_pool = 1024
                    n_neg_sample = min(target_pool - len(candidate_proteins), len(neg_pool))
                    candidate_proteins.update(rng.sample(neg_pool, n_neg_sample))
            seed_proteins = sorted(candidate_proteins)

            # 化合物冷启动验证中禁止临时添加 seed->candidate CPI 边。
            # 这些边会在 HGT 消息传递中造成信息泄漏（化合物嵌入吸收候选蛋白特征），
            # 导致模型在验证时变相“看到答案”，训练/验证分布不一致，AUC 被严重压低。
            # 保持化合物节点孤立，使其嵌入退化为 encode_compound(x)，才是公平的冷启动评估。
            sg, comp_sorted, prot_sorted, path_sorted, disease_sorted, comp_map, prot_map, disease_map = sample_hetero_subgraph(
                batch_seeds, hetero_adj, num_neighbors, seed=42, seed_proteins=seed_proteins,
                add_seed_cpi_edges=False)

            if not prot_sorted:
                continue

            sg["compound"].x = hetero_data["compound"].x[torch.tensor(comp_sorted)].to(DEVICE)
            sg["protein"].x = hetero_data["protein"].x[torch.tensor(prot_sorted)].to(DEVICE)
            if path_sorted:
                path_global_tensor = torch.tensor(sg._path_global, device=DEVICE)
                path_global_tensor = torch.clamp(path_global_tensor, min=0,
                                                  max=model.pathway_embed.num_embeddings - 1)
                sg["pathway"].x = model.pathway_embed(path_global_tensor)
            else:
                sg["pathway"].x = torch.zeros(0, model.pathway_embed.embedding_dim, device=DEVICE)
            # 疾病节点嵌入
            if disease_sorted:
                disease_global_tensor = torch.tensor(sg._disease_global, device=DEVICE).unsqueeze(-1)
                sg["disease"].x = disease_global_tensor
            else:
                sg["disease"].x = torch.zeros(0, 1, device=DEVICE)

            sg = sg.to(DEVICE)
            hgt_out = model(sg.x_dict, sg.edge_index_dict)
            prot_emb = hgt_out["protein"]
            comp_emb = hgt_out["compound"]
            n_batch_prots = prot_emb.shape[0]
            # 构建 局部索引 -> 全局蛋白索引 映射，供 prot_residue_indices 使用
            prot_inv_map_local = {v: k for k, v in prot_map.items()}

            batch_scores = []  # per-compound scores for ranking
            batch_valid_pos = []  # per-compound valid_pos for ranking

            for _bi, s in enumerate(batch_seeds):
                if s not in comp_map:
                    continue
                comp_local = comp_map[s]

                pos_set = all_compound_to_pos.get(s, set())
                valid_pos = []
                for p_global in pos_set:
                    p_local = p_global - n_compounds
                    if p_local in prot_map:
                        valid_pos.append(prot_map[p_local])
                if not valid_pos:
                    continue
                n_valid_compounds += 1

                # prot_residue_indices 必须使用全局蛋白索引（用于在残基级 ESM-2
                # 特征中查找正确的蛋白），而非子图局部索引。prot_sorted 是子图内部
                # 0~n_batch_prots-1 的局部索引，prot_map 才是 局部→全局 的映射。
                # 旧 v36 代码直接把 prot_sorted 传给 residue_indices，导致残基注意力
                # 路径查询了错误的蛋白身份，使 HGT 化合物冷启动 AUC/AUPR 被严重压低。
                prot_global_tensor = torch.tensor(
                    [prot_inv_map_local.get(p, -1) for p in range(n_batch_prots)],
                    device=DEVICE, dtype=torch.long
                )
                scores = model.decode(
                    comp_emb[comp_local:comp_local+1].expand(n_batch_prots, -1), prot_emb,
                    prot_residue_indices=prot_global_tensor,
                ) / T
                batch_scores.append(scores.cpu())  # 收集 per-compound 得分
                batch_valid_pos.append(valid_pos)  # 收集 per-compound 正样本索引
                valid_pos_tensor = torch.tensor(valid_pos, device=DEVICE, dtype=torch.long)
                for idx in valid_pos_tensor:
                    all_y_true.append(1)
                    all_y_score.append(torch.sigmoid(scores[idx]).item())

                n_hard = min(5, n_batch_prots - len(valid_pos))
                if n_hard > 0:
                    mask = torch.zeros(n_batch_prots, device=DEVICE)
                    for p in valid_pos:
                        mask[p] = -1e9
                    _, hard_indices = (scores + mask).topk(n_hard)
                    for hi in hard_indices:
                        if hi.item() < n_batch_prots:
                            all_y_true.append(0)
                            all_y_score.append(torch.sigmoid(scores[hi]).item())

                n_rand = min(5, n_batch_prots - len(valid_pos))
                if n_rand > 0:
                    rand_mask = torch.ones(n_batch_prots, device=DEVICE)
                    for p in valid_pos:
                        rand_mask[p] = 0
                    # 排除已选中的硬负样本，避免重复采样导致 AUC/AUPR 虚高
                    if n_hard > 0:
                        for hi in hard_indices:
                            if hi.item() < n_batch_prots:
                                rand_mask[hi] = 0
                    rand_candidates = torch.where(rand_mask > 0)[0]
                    if len(rand_candidates) > 0:
                        n_sample = min(n_rand, len(rand_candidates))
                        rand_idx = rand_candidates[torch.randperm(len(rand_candidates), device=DEVICE)[:n_sample]]
                        for ri in rand_idx:
                            all_y_true.append(0)
                            all_y_score.append(torch.sigmoid(scores[ri]).item())
                # v45 fix: 增加更多随机负样本以平衡硬负样本偏置
                n_rand_extra = min(20, n_batch_prots - len(valid_pos) - n_hard)
                if n_rand_extra > 0:
                    extra_rand_mask = torch.ones(n_batch_prots, device=DEVICE)
                    for p in valid_pos:
                        extra_rand_mask[p] = 0
                    if n_hard > 0:
                        for hi in hard_indices:
                            if hi.item() < n_batch_prots:
                                extra_rand_mask[hi] = 0
                    extra_candidates = torch.where(extra_rand_mask > 0)[0]
                    if len(extra_candidates) > 0:
                        n_sample = min(n_rand_extra, len(extra_candidates))
                        extra_rand_idx = extra_candidates[torch.randperm(len(extra_candidates), device=DEVICE)[:n_sample]]
                        for ri in extra_rand_idx:
                            all_y_true.append(0)
                            all_y_score.append(torch.sigmoid(scores[ri]).item())

            # 计算 per-batch 排名指标
            if batch_scores and batch_valid_pos:
                batch_score_matrix = torch.stack(batch_scores, dim=0)
                batch_ranking = _compute_ranking_metrics(batch_score_matrix, batch_valid_pos)
                all_batch_ranking.append(batch_ranking)

        if len(all_y_true) < 2 or len(set(all_y_true)) < 2:
            return {"auc": 0.5, "aupr": 0.5, "n_valid_compounds": n_valid_compounds}

        y_true_arr = np.array(all_y_true)
        y_score_arr = np.array(all_y_score)

        # HGT 验证 logit 分布诊断（用于排查 AUC≈0.5 / AUPR 异常低）
        try:
            pos_scores = y_score_arr[y_true_arr == 1]
            neg_scores = y_score_arr[y_true_arr == 0]
            if pos_scores.size and neg_scores.size:
                logger.info(
                    f"  [HGT val diag] n_pos={len(pos_scores)} n_neg={len(neg_scores)} "
                    f"pos={pos_scores.mean():.4f}±{pos_scores.std():.4f} "
                    f"neg={neg_scores.mean():.4f}±{neg_scores.std():.4f} "
                    f"gap={(pos_scores.mean() - neg_scores.mean()):.4f}"
                )
            else:
                logger.info(
                    f"  [HGT val diag] 样本缺失: n_pos={len(pos_scores)} n_neg={len(neg_scores)}"
                )
        except Exception as e:
            logger.warning(f"  [HGT val diag] 诊断打印异常: {e}")

        result = {
            "auc": float(roc_auc_score(y_true_arr, y_score_arr)),
            "aupr": float(average_precision_score(y_true_arr, y_score_arr)),
            "n_valid_compounds": n_valid_compounds,
        }

        # 计算排名指标（Precision@K, EF@1%, EF@5%）和 ROCE
        # HGT mini-batch 各 batch 蛋白候选集不同，排名指标按 batch 独立计算后平均
        if all_batch_ranking:
            precision_sums = {k: 0.0 for k in all_batch_ranking[0] if k.startswith("precision@")}
            ef_sums = {k: 0.0 for k in all_batch_ranking[0] if k.startswith("ef@")}
            n_batches = len(all_batch_ranking)
            for batch_r in all_batch_ranking:
                for k, v in batch_r.items():
                    if k in precision_sums:
                        precision_sums[k] += v
                    elif k in ef_sums:
                        ef_sums[k] += v
            for k, v in precision_sums.items():
                result[k] = v / n_batches
            for k, v in ef_sums.items():
                result[k] = v / n_batches
        roce = _compute_roce(y_true_arr, y_score_arr)
        result.update(roce)
        bedroc = _compute_bedroc(y_true_arr, y_score_arr)
        result["BEDROC"] = bedroc
        return result


def _predict_hgt_scores(
    hgt_model: HGTLinkPredictor,
    graphs: dict,
    tcm_feat: torch.Tensor,
    target_local_indices: torch.Tensor,
    n_targets: int,
) -> torch.Tensor:
    """HGT 目标蛋白评分：优先全图推理，OOM 时降级为 mini-batch。"""
    n_compounds = graphs["n_compounds"]
    n_pathways = graphs.get("n_pathways", 0)
    hetero_data = graphs["hetero_data"]
    hgt_data_dev = hetero_data.to(DEVICE)
    hgt_data_dev["pathway"].x = hgt_model.pathway_embed(
        torch.arange(max(n_pathways, 1), device=DEVICE))
    hgt_x_dict_full = {k: v.clone() for k, v in hgt_data_dev.x_dict.items()}

    hgt_model.eval()
    with torch.no_grad():
        try:
            hgt_out = hgt_model(hgt_x_dict_full, hgt_data_dev.edge_index_dict)
            hgt_prot_emb = hgt_out["protein"]
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            logger.warning("  HGT 全图预测 OOM，降级为 mini-batch 蛋白推理")
            hgt_prot_emb = torch.zeros(
                n_compounds + graphs["n_proteins"], hgt_model.out_dim, device=DEVICE)
            hgt_prot_emb[n_compounds:] = _predict_hgt_target_proteins_minibatch(
                hgt_model, graphs, target_local_indices)

        hgt_tcm_emb = hgt_model.encode_compound(tcm_feat)
        hgt_T = hgt_model.temperature
        n_tcm = hgt_tcm_emb.shape[0]
        hgt_tcm_exp = hgt_tcm_emb.unsqueeze(1).expand(-1, n_targets, -1).reshape(-1, hgt_tcm_emb.shape[-1])
        hgt_prot_exp = hgt_prot_emb[target_local_indices].unsqueeze(0).expand(n_tcm, -1, -1).reshape(-1, hgt_prot_emb.shape[-1])
        hgt_prot_residue_idx = target_local_indices.repeat(n_tcm)
        return torch.sigmoid(
            hgt_model.decode(hgt_tcm_exp, hgt_prot_exp, prot_residue_indices=hgt_prot_residue_idx) / hgt_T
        ).reshape(n_tcm, n_targets)


def _predict_hgt_target_proteins_minibatch(
    hgt_model: HGTLinkPredictor,
    graphs: dict,
    target_local_indices: torch.Tensor,
    num_neighbors: list[int] | None = None,
    batch_size: int = 16,
) -> torch.Tensor:
    """HGT mini-batch 目标蛋白嵌入推理（OOM 安全）"""
    if num_neighbors is None:
        num_neighbors = HGT_VAL_NUM_NEIGHBORS
    hetero_data = graphs["hetero_data"]
    hetero_adj = graphs["hetero_adj"]
    n_compounds = graphs["n_compounds"]
    n_proteins = graphs["n_proteins"]

    cpi_adj = hetero_adj[("compound", "interacts", "protein")]
    prot_to_compounds = defaultdict(list)
    for c_global, p_locals in cpi_adj.items():
        for p_local in p_locals:
            if 0 <= p_local < n_proteins:
                prot_to_compounds[p_local].append(c_global)

    full_prot_emb = torch.zeros(n_proteins, hgt_model.out_dim, device=DEVICE)
    target_list = target_local_indices.cpu().tolist()
    missing_targets = set()

    hgt_model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(target_list), batch_size):
            batch_targets = target_list[batch_start:batch_start + batch_size]
            seed_compounds = sorted({c for p in batch_targets for c in prot_to_compounds.get(p, [])})
            if not seed_compounds:
                seed_compounds = [0]
            try:
                sg, comp_sorted, prot_sorted, path_sorted, disease_sorted, comp_map, prot_map, disease_map = sample_hetero_subgraph(
                    seed_compounds, hetero_adj, num_neighbors=num_neighbors, seed=42,
                    seed_proteins=batch_targets, add_seed_cpi_edges=False,
                )
            except Exception as e:
                logger.warning(f"  HGT mini-batch 子图采样失败 (targets={batch_start}-{batch_start + len(batch_targets)}): {e}")
                missing_targets.update(batch_targets)
                continue
            if not prot_sorted:
                missing_targets.update(batch_targets)
                continue

            comp_tensor = torch.tensor(comp_sorted)
            if seed_compounds == [0] and 0 not in cpi_adj:
                sg["compound"].x = torch.zeros(len(comp_sorted), hetero_data["compound"].x.shape[1], device=DEVICE)
            else:
                sg["compound"].x = hetero_data["compound"].x[comp_tensor].to(DEVICE)
            sg["protein"].x = hetero_data["protein"].x[torch.tensor(prot_sorted)].to(DEVICE)
            if path_sorted:
                path_global_tensor = torch.tensor(sg._path_global, device=DEVICE)
                path_global_tensor = torch.clamp(path_global_tensor, min=0, max=hgt_model.pathway_embed.num_embeddings - 1)
                sg["pathway"].x = hgt_model.pathway_embed(path_global_tensor)
            else:
                sg["pathway"].x = torch.zeros(0, hgt_model.pathway_embed.embedding_dim, device=DEVICE)
            if disease_sorted:
                disease_global_tensor = torch.tensor(sg._disease_global, device=DEVICE).unsqueeze(-1)
                sg["disease"].x = disease_global_tensor
            else:
                sg["disease"].x = torch.zeros(0, 1, device=DEVICE)

            sg = sg.to(DEVICE)
            hgt_out = hgt_model(sg.x_dict, sg.edge_index_dict)
            batch_prot_emb = hgt_out["protein"]
            for p_local in batch_targets:
                if p_local in prot_map:
                    full_prot_emb[p_local] = batch_prot_emb[prot_map[p_local]]
                else:
                    missing_targets.add(p_local)

    if missing_targets:
        logger.warning(f"  HGT mini-batch 推理缺失 {len(missing_targets)} 个目标蛋白嵌入，已置零: {sorted(list(missing_targets))[:10]}")

    return full_prot_emb[target_local_indices]


def predict_tcm(
    sage_model: SAGELinkPredictor,
    hgt_model: HGTLinkPredictor | None,
    graphs: dict,
    tcm_smiles: list[str],
    target_genes: list[str],
    compound_stats: tuple,
    diversity_penalty: float = 0.3,  # 默认值 0.1→0.3 与全局 DIVERSITY_PENALTY 一致
    mc_samples: int = 0,
    tcm_feat_precomputed: torch.Tensor | None = None,
    tree_predictions: pd.DataFrame | None = None,  # 树模型预测分数
    tree_weight: float = 0.6,  # 树模型集成权重
) -> pd.DataFrame:
    """v40: SAGE + HGT + 树模型三方集成预测 — 等权集成 + 多样性约束 + MC Dropout

    Args:
        diversity_penalty: 余弦相似度惩罚系数（0~1，越大越惩罚相似预测）
        mc_samples: MC Dropout 采样次数（0=禁用，推荐30）
        tree_predictions: 树模型预测 DataFrame (MOL_ID, SMILES, gene, score)
        tree_weight: 树模型在最终集成中的权重（0~1）
    """
    n_iterations = max(1, mc_samples)
    use_mc = mc_samples > 0

    if use_mc:
        sage_model.train()  # 保持 Dropout 开启，无梯度
        if hgt_model is not None:
            hgt_model.train()
    else:
        sage_model.eval()
        if hgt_model is not None:
            hgt_model.eval()

    if tcm_feat_precomputed is not None:
        tcm_feat = tcm_feat_precomputed.to(DEVICE)
    else:
        tcm_feat_raw, _, _, _ = build_compound_features(tcm_smiles, stats=compound_stats)
        feat_dim = graphs["feat_dim"]
        if tcm_feat_raw.shape[1] < feat_dim:
            tcm_feat_raw = np.pad(tcm_feat_raw, ((0, 0), (0, feat_dim - tcm_feat_raw.shape[1])), mode="constant")
        tcm_feat = torch.from_numpy(tcm_feat_raw).to(DEVICE)

    x_dev = graphs["x"].to(DEVICE)
    n_compounds = graphs["n_compounds"]
    n_proteins = graphs["n_proteins"]
    gene_to_idx = graphs["gene_to_idx"]
    homo_edge_index = graphs["homo_edge_index"]

    # 等权集成 — 不依赖蛋白冷启动 AUPR
    sage_w = 0.5
    hgt_w = 0.5
    logger.info(f"  集成权重: SAGE={sage_w:.3f}, HGT={hgt_w:.3f}（等权集成）")

    # 预构建基因→蛋白局部索引映射
    gene_index_map = []  # [(gene, local_p_idx), ...]
    for gene in target_genes:
        if gene in gene_to_idx:
            p_idx = gene_to_idx[gene]
            local_p_idx = p_idx - n_compounds
            if 0 <= local_p_idx < n_proteins:
                gene_index_map.append((gene, local_p_idx))
            else:
                gene_index_map.append((gene, -1))
        else:
            gene_index_map.append((gene, -1))

    all_sage_scores_mc = []  # (n_iter, n_tcm, n_genes)
    all_hgt_scores_mc = []

    for _it in range(n_iterations):
        try:
            with torch.no_grad():
                # SAGE: 原生归纳式推理
                # 分别处理 — 全图编码蛋白嵌入 + encode_compound 编码 TCM 化合物
                edge_index = homo_edge_index.to(DEVICE)
                node_emb = sage_model(x_dev, edge_index)  # 全图（原化合物+蛋白）
                sage_prot_emb = node_emb[n_compounds:]
                sage_tcm_emb = sage_model.encode_compound(tcm_feat)  # TCM 化合物（无CPI边，仅投影+卷积）
                sage_T = sage_model.temperature

                # v45: 仅对目标基因对应的蛋白打分，避免 residue_bilinear 在全蛋白集上耗时爆炸。
                valid_gene_indices = [(j, lp) for j, (_, lp) in enumerate(gene_index_map) if lp >= 0]
                target_local_indices = torch.tensor(
                    [lp for _, lp in valid_gene_indices], dtype=torch.long, device=DEVICE)

                # SAGE 向量化评分: (n_tcm, n_target_prots)
                n_tcm_sage = sage_tcm_emb.shape[0]
                n_targets = target_local_indices.shape[0]
                sage_tcm_exp = sage_tcm_emb.unsqueeze(1).expand(-1, n_targets, -1).reshape(-1, sage_tcm_emb.shape[-1])
                sage_prot_exp = sage_prot_emb[target_local_indices].unsqueeze(0).expand(n_tcm_sage, -1, -1).reshape(-1, sage_prot_emb.shape[-1])
                sage_prot_residue_idx = target_local_indices.repeat(n_tcm_sage)
                sage_target_scores = torch.sigmoid(
                    sage_model.decode(sage_tcm_exp, sage_prot_exp, prot_residue_indices=sage_prot_residue_idx) / sage_T
                ).reshape(n_tcm_sage, n_targets)

                if hgt_model is not None:
                    hgt_target_scores = _predict_hgt_scores(
                        hgt_model, graphs, tcm_feat, target_local_indices, n_targets
                    )
                else:
                    hgt_target_scores = None

                # 提取目标基因的分数
                iter_sage = torch.full((len(tcm_smiles), len(target_genes)), 0.5, device=DEVICE)
                iter_hgt = torch.full((len(tcm_smiles), len(target_genes)), 0.5, device=DEVICE)
                for score_col, (target_col, _local_p_idx) in enumerate(valid_gene_indices):
                    iter_sage[:, target_col] = sage_target_scores[:, score_col]
                    if hgt_target_scores is not None:
                        iter_hgt[:, target_col] = hgt_target_scores[:, score_col]
                all_sage_scores_mc.append(iter_sage.cpu())
                all_hgt_scores_mc.append(iter_hgt.cpu())
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                raise
            logger.warning(f"  predict_tcm MC 迭代 {_it + 1}/{n_iterations} OOM: {e}")
            torch.cuda.empty_cache()
            if _it == 0:
                logger.error("  predict_tcm 首次迭代即 OOM，无法继续预测，请减小 mc_samples 或降级到 CPU")
                raise
            logger.warning(f"  predict_tcm OOM 降级: 跳过迭代 {_it + 1}，使用前 {len(all_sage_scores_mc)} 次结果")
            break

    # ---- 聚合 MC 迭代结果 ----
    if use_mc:
        sage_stack = torch.stack(all_sage_scores_mc, dim=0)  # (n_iter, n_tcm, n_genes)
        hgt_stack = torch.stack(all_hgt_scores_mc, dim=0)

        sage_mean = sage_stack.mean(dim=0)  # (n_tcm, n_genes)
        sage_std = sage_stack.std(dim=0)
        hgt_mean = hgt_stack.mean(dim=0)
        hgt_std = hgt_stack.std(dim=0)

        logger.info(f"  MC Dropout ({mc_samples} 次): SAGE 均值范围 [{sage_mean.min():.4f}, {sage_mean.max():.4f}], "
                    f"平均不确定度 {sage_std.mean():.4f}")
    else:
        sage_mean = all_sage_scores_mc[0]
        hgt_mean = all_hgt_scores_mc[0]
        sage_std = hgt_std = None

    # 多样性约束 — 在分支均值上应用
    # 原始公式 diversity_factor = 1 - penalty * (1 - delta) 会惩罚一致性、奖励分歧，与集成学习直觉相反。
    # 修正为：模型越一致（delta→0），越信任集成分数；越分歧（delta→1），越向 0.5 收缩表示不确定。
    delta = torch.abs(sage_mean - hgt_mean)  # (n_tcm, n_genes)
    # 使用函数参数 diversity_penalty 而非全局常量 DIVERSITY_PENALTY
    # 原代码直接引用全局常量，导致函数参数完全无效
    diversity_factor = 1.0 - diversity_penalty * delta
    weighted_scores = sage_w * sage_mean + hgt_w * hgt_mean
    final_scores = weighted_scores * diversity_factor + 0.5 * (1.0 - diversity_factor)

    # 树模型集成 — 将树模型预测与 GNN 集成分数加权融合
    tree_scores_tensor = None
    if tree_predictions is not None and len(tree_predictions) > 0:
        # 构建 (SMILES, gene) → score 的查找表
        tree_lookup = {}
        for _, row in tree_predictions.iterrows():
            raw_score = float(row["score"])
            # v44: 若树模型分数超出 [0,1]，视为 logit 进行 sigmoid 校准并告警。
            if raw_score < 0.0 or raw_score > 1.0:
                logger.warning(
                    f"树模型分数超出概率范围 (score={raw_score:.4f})，自动 sigmoid 校准"
                )
                raw_score = float(1.0 / (1.0 + math.exp(-raw_score)))
                raw_score = max(0.0, min(1.0, raw_score))
            tree_lookup[(str(row["SMILES"]), str(row["gene"]))] = raw_score
        # 构建与 final_scores 同形状的张量
        tree_scores_tensor = torch.full_like(final_scores, 0.5)
        tree_matched = 0
        for i, smi in enumerate(tcm_smiles):
            for j, (gene, _) in enumerate(gene_index_map):
                key = (smi, gene)
                if key in tree_lookup:
                    tree_scores_tensor[i, j] = tree_lookup[key]
                    tree_matched += 1
        logger.info(f"  树模型集成: 匹配 {tree_matched}/{final_scores.numel()} 对, "
                    f"权重 tree={tree_weight:.2f} GNN={1-tree_weight:.2f}")
        # 三方融合: GNN 集成分数 × (1-tree_weight) + 树模型分数 × tree_weight
        final_scores = (1 - tree_weight) * final_scores + tree_weight * tree_scores_tensor

    # 按基因维度计算余弦相似度并取均值，避免全局展平丢失基因特异性信息
    # sage_mean/hgt_mean: (n_tcm, n_genes)
    per_gene_cos = []
    for g in range(sage_mean.shape[1]):
        sg = sage_mean[:, g]
        hg = hgt_mean[:, g]
        per_gene_cos.append(F.cosine_similarity(sg.unsqueeze(0), hg.unsqueeze(0)).item())
    cos_sim = float(np.mean(per_gene_cos))
    logger.info(f"  分支余弦相似度: {cos_sim:.4f} (越低越好，表示分支互补性强)")

    # 构建结果 DataFrame
    results = []
    for i, smi in enumerate(tcm_smiles):
        row = {"MOL_ID": f"TCM_{i}", "molecule_name": "", "SMILES": smi}
        for j, (gene, _) in enumerate(gene_index_map):
            row[gene] = final_scores[i, j].item()
            if use_mc:
                # MC 不确定性：取两个分支标准差的均值作为该对的不确定度
                row[f"{gene}_uncertainty"] = ((sage_std[i, j] + hgt_std[i, j]) / 2).item()
        if use_mc:
            # 聚合不确定性指标
            pair_uncertainties = (sage_std[i] + hgt_std[i]) / 2
            row["mean_uncertainty"] = pair_uncertainties.mean().item()
            row["max_uncertainty"] = pair_uncertainties.max().item()
        results.append(row)

    return pd.DataFrame(results)


def pipeline_self_check(tcm_df, cpi_df, ppi_df, prot_feat, gene_to_pathways, warm_targets):
    """v25: 管线自检 — 训练前验证数据完整性和一致性

    检查项：
      - TCM 池 SMILES 列有效性
      - CPI 数据重复记录
      - TCM/训练集化合物重叠
      - 蛋白特征 NaN 值
      - 温靶标 CPI 边数量

    Args:
        tcm_df: TCM 候选池 DataFrame
        cpi_df: CPI 交互数据
        ppi_df: PPI 网络数据
        prot_feat: 蛋白特征字典
        gene_to_pathways: 基因-通路映射
        warm_targets: 温靶标基因列表

    Returns:
        dict: {"overall": "PASSED"|"PASSED_WITH_WARNINGS"|"FAILED", "errors": [...], "warnings": [...]}
    """
    logger.info("=" * 60)
    logger.info("开始管线自检...")
    results = {"errors": [], "warnings": []}

    # 检查SMILES列是否存在
    smiles_col = None
    for col in ["SMILES_std", "SMILES", "smiles", "canonical_smiles"]:
        if col in tcm_df.columns:
            smiles_col = col
            break
    if smiles_col is None:
        results["errors"].append("TCM池中无有效SMILES列（SMILES_std/SMILES/smiles/canonical_smiles）")
        for err in results["errors"]:
            logger.error(f"  [ERROR] {err}")
        for warn in results["warnings"]:
            logger.warning(f"  [WARN] {warn}")
        logger.info("管线自检完成")
        return results
    tcm_smiles = tcm_df[smiles_col].dropna().tolist()
    invalid_smiles = sum(1 for s in tcm_smiles if Chem.MolFromSmiles(str(s)) is None)
    if invalid_smiles:
        results["errors"].append(f"TCM池含 {invalid_smiles} 个无效SMILES")

    cpi_dupes = cpi_df.duplicated(subset=["gene", "canonical_smiles"]).sum()
    if cpi_dupes:
        results["warnings"].append(f"CPI数据含 {cpi_dupes} 条重复")

    tcm_smi_set = set(tcm_smiles)
    train_smi_set = set(cpi_df["canonical_smiles"].dropna().unique())
    overlap = tcm_smi_set & train_smi_set
    if overlap:
        # 详细记录重叠化合物信息，便于下游分析
        overlap_list = sorted(overlap)
        results["warnings"].append(f"TCM/训练集重叠: {len(overlap)} 个化合物（数据泄漏风险）")
        logger.warning(f"  [数据泄漏] TCM池与训练集有 {len(overlap)} 个重叠SMILES:")
        for i, smi in enumerate(overlap_list[:10]):  # 只展示前10个
            logger.warning(f"    [{i+1}] {smi}")
        if len(overlap_list) > 10:
            logger.warning(f"    ... 及其他 {len(overlap_list) - 10} 个")
        logger.warning("  这些化合物在预测时可能由于训练集过拟合而得分虚高，"
                       "建议在最终排序中标记为 'in_train' 以便人工审核")

    nan_genes = [g for g, v in prot_feat.items() if np.isnan(v).any()]
    if nan_genes:
        results["errors"].append(f"蛋白特征含NaN: {nan_genes}")

    warm_cpi = cpi_df[cpi_df["gene"].isin(warm_targets)]
    if len(warm_cpi) < 10:
        results["errors"].append(f"温靶标CPI边不足: {len(warm_cpi)}")

    overall = "FAILED" if results["errors"] else ("PASSED_WITH_WARNINGS" if results["warnings"] else "PASSED")
    logger.info(f"自检结果: {overall}")
    for e in results["errors"]:
        logger.error(f"  ERROR: {e}")
    for w in results["warnings"]:
        logger.warning(f"  WARNING: {w}")
    logger.info("=" * 60)

    return {"overall": overall, **results}


def main(decoder_type: str | None = None, skip_sage: bool = False, skip_hgt: bool = False,
         global_overrides: dict | None = None):
    """v27: Phase 4 主流程 — SAGE + HGT 双分支训练与 TCM 预测

    流程:
      1. 加载 CPI/PPI/KEGG/蛋白特征/TCM 池数据
      2. 构建同质图 + 异质图（可选疾病节点）
      3. 双重冷启动拆分（化合物 85/15 + 蛋白 80/20 分层）
      4. 训练 SAGE 分支（SAGEConv + 两阶段迁移学习）
      5. 训练 HGT 分支（HGTConv + 两阶段迁移学习）
      6. 动态集成权重预测 TCM 化合物-靶标得分
      7. 输出 v27 预测结果和性能指标

    Args:
        decoder_type: CLI 传入的解码器类型，覆盖配置中的 DECODER_TYPE。
    """
    global DECODER_TYPE, EPOCHS, PRETRAIN_EPOCHS, RANDOM_SEED
    if decoder_type is not None:
        DECODER_TYPE = decoder_type
        logger.info(f"CLI 覆盖 decoder_type = {DECODER_TYPE}")
    if global_overrides:
        for key, value in global_overrides.items():
            if key in globals():
                globals()[key] = value
                logger.info(f"CLI 覆盖 {key} = {value}")
            else:
                logger.warning(f"CLI 覆盖项 {key} 不是全局常量，已忽略")

    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Phase 4 v37: SAGE + HGT Mini-Batch — 工业级重构（配置系统/tqdm/GPU监控/类型注解）")
    logger.info("v21: 移除InfoNCE / 保留课程负采样&BPR&两阶段 / 双分支 / 蛋白冷启动")
    logger.info("v22: warm靶标扩展 / 靶标优先级加权 / zero-shot bonus")
    logger.info("v23: 铁死亡表型分类辅助任务 / 表型概率融合到composite_score")
    logger.info("v24: GSE61616疾病节点 / 铁衰老96基因全蛋白集 / 四模态异质图")
    logger.info("v25: 蛋白冷启动负采样仅从CPI蛋白采样 / 预训练checkpoint基于prot_aupr / 早停统一")
    logger.info("v26: PPI去重 / CPI补充SMILES修复 / HGT解码器统一MLP / DIVERSITY_PENALTY 0.1→0.3")
    logger.info("v27: 模型定义迁移至模块化文件 / warmup最小值2 / TCM重叠标记 / 代码注释完善")
    logger.info("v28: 配置系统集成 / tqdm进度条 / GPU显存监控 / 类型注解 / 前置断言 / OOM降级")
    logger.info("v37: HGT 蛋白冷启动验证负采样与 SAGE 对齐 (仅 CPI 蛋白) + 化合物冷启动验证 prot_residue_indices 改用全局索引")
    logger.info("v38: 彻底移除蛋白冷启动验证 / 早停仅基于 val_aupr / 等权集成 / 清理未定义变量")
    logger.info("v40: 集成树模型 v7 扩展CPI数据 (86基因/48K条) + 树模型预测集成")
    logger.info("v41: 修复 HGT 化合物冷启动验证信息泄漏，验证时禁用 seed->candidate 临时 CPI 边")
    logger.info("=" * 60)

    # ── 加固: 关键数据文件存在性检查 ──
    critical_files = {
        "CPI数据": L4_ROOT / "results" / "experimental_actives_detail_cleaned_combined.csv",
        "铁衰老基因列表": FERRORAGING_GENES_CSV,
        "ESM-2缓存": L4_RESULTS / "esm2_protein_embeddings.npz",
        "目标蛋白特征": L2_RESULTS / "target_protein_features.csv",
    }
    missing_critical = []
    for name, fpath in critical_files.items():
        if not fpath.exists():
            missing_critical.append(f"{name}: {fpath}")
    if missing_critical:
        logger.warning(f"以下关键文件不存在（可能导致降级）: {missing_critical}")

    # ── 加固: GPU 显存检查 ──
    _check_gpu_memory(min_free_gb=0.5)

    # 加载数据
    logger.info(">>> 加载数据")
    _t0 = time.time()
    cpi_df = load_cpi_data()
    ppi_df = load_ppi_network()
    gene_to_pathways = load_kegg_pathways()
    prot_feat, gene_to_seq = load_protein_features()  # 启用 ESM-2 预训练蛋白嵌入 (640维), hf-mirror.com 镜像下载
    tcm_df = load_tcm_pool()
    _t0 = _log_step_time(_t0, "数据加载完成")

    # 加载铁死亡表型分类数据集
    pheno_df = None
    pheno_file = L4_RESULTS / "phenotype_ferroptosis_dataset_v25_clean.csv"
    if pheno_file.exists():
        pheno_df = pd.read_csv(pheno_file)
        logger.info(f">>> 铁死亡表型数据集: {len(pheno_df)} 个化合物 (正={(pheno_df['label']==1).sum()}, 负={(pheno_df['label']==0).sum()})")
    else:
        logger.warning(f">>> 铁死亡表型数据集不存在: {pheno_file}，跳过表型辅助任务")

    # warm_targets 扩展为所有有 CPI 数据的基因（树模型 v7 扩展至 86基因，含非96列表基因）
    all_cpi_genes = sorted(set(cpi_df["gene"].unique()))
    warm_in_96 = sorted(set(all_cpi_genes) & set(ALL_FERRORAGING_GENES))
    warm_extra = sorted(set(all_cpi_genes) - set(ALL_FERRORAGING_GENES))
    warm_targets = all_cpi_genes  # 训练用全部 CPI 基因（86个）
    logger.info(f"温靶标: {len(warm_targets)} 个 (96列表内={len(warm_in_96)}, 额外核心={len(warm_extra)})")
    logger.info(f"  96列表内warm: {warm_in_96}")
    logger.info(f"  额外warm靶标: {warm_extra}")

    # 预测靶标 = 96个铁衰老列表基因 + 19个有CPI数据的额外核心靶标
    # 共计 115 个靶标，其中 42 个 warm（有训练数据），73 个 zero-shot（无训练数据）
    all_target_genes_pred = sorted(set(ALL_FERRORAGING_GENES) | set(warm_extra))
    zero_shot_pred = sorted(set(all_target_genes_pred) - set(warm_targets))
    logger.info(f"预测靶标: {len(all_target_genes_pred)} 个 (warm={len(warm_targets)}, zero-shot={len(zero_shot_pred)})")
    logger.info(f"  zero-shot 靶标数: {len(zero_shot_pred)}")

    check_results = pipeline_self_check(tcm_df, cpi_df, ppi_df, prot_feat, gene_to_pathways, warm_targets)
    if check_results["overall"] == "FAILED":
        logger.error("管线自检未通过，终止训练")
        sys.exit(1)

    cpi_df = cpi_df[cpi_df["gene"].isin(warm_targets)].copy()

    # 加载疾病节点边
    disease_df = None
    disease_file = L4_RESULTS / "disease_gene_edges.csv"
    if disease_file.exists():
        disease_df = pd.read_csv(disease_file)
        logger.info(f">>> 疾病节点: {len(disease_df)} 条疾病-蛋白边")
    else:
        logger.warning(f">>> 疾病节点文件不存在: {disease_file}")

    # 确保铁衰老96基因全部在蛋白特征中
    missing_ferro_in_prot_feat = [g for g in ALL_FERRORAGING_GENES if g not in prot_feat]
    if missing_ferro_in_prot_feat:
        logger.warning(f">>> 以下铁衰老96基因缺少ESM-2特征: {missing_ferro_in_prot_feat}")

    # 图数据缓存路径与缓存键（修复 HGT 化合物冷启动验证信息泄漏）
    GRAPH_CACHE_PATH = L4_RESULTS / "graph_cache_v41.pkl"
    GRAPH_CACHE_KEY = {
        "version": "v41",
        "random_seed": RANDOM_SEED,
        "compound_val_split": COMPOUND_VAL_SPLIT,
        "protein_val_split": PROTEIN_VAL_SPLIT,
        "n_cpi": len(cpi_df),
        "n_ppi": len(ppi_df),
        "n_pathway_genes": len(gene_to_pathways),
        "prot_feat_shape": _get_prot_feat_dim(prot_feat),
        "disease_file_exists": disease_df is not None,
    }

    # 尝试加载图数据缓存
    _cache_loaded = False
    if GRAPH_CACHE_PATH.exists():
        try:
            logger.info(f">>> 尝试加载图数据缓存: {GRAPH_CACHE_PATH}")
            with open(GRAPH_CACHE_PATH, "rb") as _f:
                _cached = pickle.load(_f)
            if _cached.get("key") == GRAPH_CACHE_KEY:
                graphs = _cached["graphs"]
                train_compounds = _cached["train_compounds"]
                val_compounds = _cached["val_compounds"]
                train_proteins = _cached["train_proteins"]
                val_proteins = _cached["val_proteins"]
                compound_to_pos = defaultdict(set, _cached["compound_to_pos"])
                pheno_train_indices = _cached.get("pheno_train_indices")
                pheno_train_labels = _cached.get("pheno_train_labels")
                warm_targets = _cached.get("warm_targets", warm_targets)
                all_target_genes_pred = _cached.get("all_target_genes_pred", all_target_genes_pred)
                _cache_loaded = True
                logger.info("  图数据缓存命中，跳过图构建与拆分")
            else:
                logger.info("  图数据缓存键不匹配，重新构建")
        except Exception as _e:
            logger.warning(f"  加载图数据缓存失败: {_e}，重新构建")

    if not _cache_loaded:
        # 构建图 & 邻接表
        logger.info(">>> 构建图 & 邻接表")
        _t0 = time.time()
        graphs = build_graphs_and_adj(
            cpi_df, ppi_df, gene_to_pathways, prot_feat, disease_df=disease_df,
            use_topology_neg=USE_TOPOLOGY_NEG,
            topo_neighbors_top_k=TOPO_NEIGHBORS_TOP_K,
            use_esm_similarity_neg=USE_ESM_SIMILARITY_NEG,
            esm_similarity_top_k=ESM_SIMILARITY_TOP_K,
        )
        _t0 = _log_step_time(_t0, "图构建完成")

        # ── 加固: 图结构完整性检查 ──
        n_nodes = graphs["n_compounds"] + graphs["n_proteins"]
        n_edges_homo = graphs["homo_edge_index"].shape[1]
        assert n_nodes > 0, "图节点数为0"
        assert graphs["n_compounds"] > 0, "化合物节点数为0"
        assert graphs["n_proteins"] > 0, "蛋白节点数为0"
        assert n_edges_homo > 0, "同质图边数为0"
        assert graphs["feat_dim"] > 0, "特征维度为0"
        logger.info(f"图结构完整性: {n_nodes} 节点 ({graphs['n_compounds']}c + {graphs['n_proteins']}p), "
                    f"{n_edges_homo} 边, feat_dim={graphs['feat_dim']}")

        # 双重冷启动拆分 — 化合物 + 蛋白
        all_compounds = sorted(graphs["smi_to_idx"].values())
        all_proteins = sorted({
            graphs["gene_to_idx"][g] - graphs["n_compounds"]
            for g in graphs["gene_to_idx"]
            if graphs["gene_to_idx"][g] >= graphs["n_compounds"]
        })
        random.shuffle(all_compounds)
        random.shuffle(all_proteins)

        # 化合物冷启动: 85% train / 15% val
        n_train_comp = int(len(all_compounds) * COMPOUND_VAL_SPLIT)
        train_compounds = all_compounds[:n_train_comp]
        val_compounds = all_compounds[n_train_comp:]

        # 蛋白训练/验证拆分 — 验证集蛋白在训练图中不可见（训练安全图），
        # 避免验证蛋白的PPI/通路边在训练时泄漏信息。
        cpi_proteins = set()
        for _, row in cpi_df.iterrows():
            gene = row["gene"]
            if gene in graphs["gene_to_idx"]:
                cpi_proteins.add(graphs["gene_to_idx"][gene] - graphs["n_compounds"])
        non_cpi_proteins = [p for p in all_proteins if p not in cpi_proteins]

        n_val_cpi = max(1, int(len(cpi_proteins) * PROTEIN_VAL_SPLIT))
        n_train_cpi = len(cpi_proteins) - n_val_cpi
        n_val_non_cpi = max(1, int(len(non_cpi_proteins) * PROTEIN_VAL_SPLIT))
        n_train_non_cpi = len(non_cpi_proteins) - n_val_non_cpi

        cpi_proteins = list(cpi_proteins)
        random.shuffle(cpi_proteins)
        random.shuffle(non_cpi_proteins)

        train_proteins = set(cpi_proteins[:n_train_cpi]) | set(non_cpi_proteins[:n_train_non_cpi])
        val_proteins = set(cpi_proteins[n_train_cpi:]) | set(non_cpi_proteins[n_train_non_cpi:])

        # 预计算正样本（v13: 统一使用全局蛋白索引 = gene_to_idx[gene]，不再存储 n_compounds 偏移）
        compound_to_pos = defaultdict(set)
        for _, row in cpi_df.iterrows():
            smi = row["canonical_smiles"]
            gene = row["gene"]
            if smi in graphs["smi_to_idx"] and gene in graphs["gene_to_idx"]:
                compound_to_pos[graphs["smi_to_idx"][smi]].add(
                    graphs["gene_to_idx"][gene])  # 全局索引，不做减法

    # 缓存命中时同样需要 cpi_proteins 用于后续统计
    if _cache_loaded:
        cpi_proteins = set()
        for _, row in cpi_df.iterrows():
            gene = row["gene"]
            if gene in graphs["gene_to_idx"]:
                cpi_proteins.add(graphs["gene_to_idx"][gene] - graphs["n_compounds"])
    # 表型化合物索引映射 — 只保留训练集中存在的化合物
    pheno_train_indices = None
    pheno_train_labels = None
    if pheno_df is not None:
        pheno_indices = []
        pheno_labels_list = []
        smi_col = None
        for col in ["SMILES", "smiles", "canonical_smiles", "SMILES_std"]:
            if col in pheno_df.columns:
                smi_col = col
                break
        if smi_col is None:
            logger.warning("  表型数据集中未找到 SMILES 列，跳过表型任务")
        else:
            for _, row in pheno_df.iterrows():
                smi = row[smi_col]
                if pd.isna(smi):
                    continue
                if smi in graphs["smi_to_idx"]:
                    idx = graphs["smi_to_idx"][smi]
                    pheno_indices.append(idx)
                    pheno_labels_list.append(int(row["label"]))
            logger.info(f"  表型化合物匹配: {len(pheno_indices)}/{len(pheno_df)} 个在训练图中找到")

            if len(pheno_indices) > 0:
                train_comp_set = set(train_compounds)
                train_pheno_idx = []
                train_pheno_lab = []
                val_pheno_idx = []
                val_pheno_lab = []
                for idx, lab in zip(pheno_indices, pheno_labels_list, strict=False):
                    if idx in train_comp_set:
                        train_pheno_idx.append(idx)
                        train_pheno_lab.append(lab)
                    else:
                        val_pheno_idx.append(idx)
                        val_pheno_lab.append(lab)

                if len(train_pheno_idx) < len(pheno_indices) * 0.5:
                    combined = list(zip(pheno_indices, pheno_labels_list, strict=False))
                    random.shuffle(combined)
                    n_train = max(1, int(len(combined) * 0.8))
                    train_combined = combined[:n_train]
                    val_combined = combined[n_train:]
                    train_pheno_idx = [x[0] for x in train_combined]
                    train_pheno_lab = [x[1] for x in train_combined]
                    val_pheno_idx = [x[0] for x in val_combined]
                    val_pheno_lab = [x[1] for x in val_combined]

                pheno_train_indices = train_pheno_idx
                pheno_train_labels = train_pheno_lab
                n_pos = sum(train_pheno_lab)
                logger.info(f"  表型训练集: {len(train_pheno_idx)} 个 (正={n_pos}, 负={len(train_pheno_lab)-n_pos})")
                logger.info(f"  表型验证集: {len(val_pheno_idx)} 个 (正={sum(val_pheno_lab)}, 负={len(val_pheno_lab)-sum(val_pheno_lab)})")
            else:
                logger.warning("  没有表型化合物能匹配到训练图，跳过表型任务")

    # 统计无效化合物
    n_val_no_pos = sum(1 for c in val_compounds if c not in compound_to_pos or len(compound_to_pos[c]) == 0)
    n_train_no_pos = sum(1 for c in train_compounds if c not in compound_to_pos or len(compound_to_pos[c]) == 0)
    logger.info(f"冷启动拆分: {len(train_compounds)} train ({n_train_no_pos} 无正样本) / "
                f"{len(val_compounds)} val ({n_val_no_pos} 无正样本) 化合物")
    n_val_cpi_actual = sum(1 for p in val_proteins if p in cpi_proteins)
    logger.info(f"蛋白拆分: {len(train_proteins)} train / {len(val_proteins)} val 蛋白 "
                f"(CPI蛋白: {len(cpi_proteins)} 总, {n_val_cpi_actual} 在验证集)")

    # 分离化合物冷启动与蛋白冷启动验证图
    val_comp_set = set(val_compounds)
    # 化合物冷启动验证图：仅移除验证集化合物的 CPI 边，保留蛋白侧拓扑
    graphs["homo_edge_index_val"] = _build_val_comp_cold_homo_edge_index(
        graphs["homo_edge_index"], val_comp_set)
    graphs["hetero_data_val"] = _build_val_comp_cold_hetero_data(
        graphs["hetero_data"], val_comp_set)
    graphs["hetero_adj_val"] = _build_val_comp_cold_hetero_adj(
        graphs["hetero_adj"], graphs["n_compounds"], val_comp_set)
    # 构建训练安全邻接表 — 训练阶段完全隐藏验证蛋白，杜绝 PPI 网络信息泄露
    graphs["homo_adj_train"] = _build_train_safe_homo_adj(
        graphs["homo_adj"], graphs["n_compounds"], val_comp_set, val_proteins)
    graphs["hetero_adj_train"] = _build_train_safe_hetero_adj(
        graphs["hetero_adj"], graphs["n_compounds"], val_comp_set, val_proteins)
    # Memory Bank 全局刷新也使用训练安全图，避免验证蛋白嵌入进入 bank
    graphs["homo_edge_index_train"] = _build_val_safe_homo_edge_index(
        graphs["homo_edge_index"], graphs["n_compounds"], val_comp_set, val_proteins)
    graphs["hetero_data_train"] = _build_val_safe_hetero_data(
        graphs["hetero_data"], val_comp_set, val_proteins)
    logger.info(f"  训练安全邻接表已构建: SAGE {sum(len(v) for v in graphs['homo_adj_train'].values())} 条边, "
                f"HGT {sum(len(v) for v in graphs['hetero_adj_train'].values())} 条边")

    # 保存图数据缓存（包含验证子图，避免每 epoch 重建）
    if not _cache_loaded:
        try:
            _cache_data = {
                "key": GRAPH_CACHE_KEY,
                "graphs": graphs,
                "train_compounds": train_compounds,
                "val_compounds": val_compounds,
                "train_proteins": train_proteins,
                "val_proteins": val_proteins,
                "compound_to_pos": dict(compound_to_pos),
                "pheno_train_indices": pheno_train_indices,
                "pheno_train_labels": pheno_train_labels,
                "warm_targets": warm_targets,
                "all_target_genes_pred": all_target_genes_pred,
            }
            with open(GRAPH_CACHE_PATH, "wb") as _f:
                pickle.dump(_cache_data, _f)
            logger.info(f"  图数据缓存已保存: {GRAPH_CACHE_PATH}")
        except Exception as _e:
            logger.warning(f"  保存图数据缓存失败: {_e}")

    # 加载残基级 ESM-2 特征（仅 residue_bilinear 解码器需要）
    residue_embeddings = None
    residue_offsets = None
    residue_lengths = None
    prot_to_residue_idx = None
    residue_max_len = None
    if DECODER_TYPE == "residue_bilinear":
        residue_pt_path = L4_RESULTS / "esm2_150M_residue_features.pt"
        if not residue_pt_path.exists():
            raise FileNotFoundError(
                f"decoder_type=residue_bilinear 但残基特征文件不存在: {residue_pt_path}"
            )
        try:
            (
                residue_embeddings,
                residue_offsets,
                residue_lengths,
                prot_to_residue_idx,
                residue_max_len,
            ) = load_residue_esm2_features(
                graphs, residue_pt_path=residue_pt_path, max_len_cap=ESM_MAX_LEN, residue_device="cpu"
            )
        except Exception as _e:
            logger.warning(
                f"v40: 残基特征文件加载失败 ({_e})，自动降级为 bilinear 解码器"
            )
            DECODER_TYPE = "bilinear"
            residue_embeddings = None

    sage_model_path = L4_RESULTS / "sage_best_v42.pt"
    if skip_sage and sage_model_path.exists():
        logger.info(f">>> 跳过 SAGE 训练，加载已有模型: {sage_model_path}")
        sage_model = SAGELinkPredictor(
            comp_feat_dim=graphs["feat_dim"], prot_feat_dim=graphs["prot_esm_dim"],
            n_compounds=graphs["n_compounds"],
            hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT,
            n_pathways=graphs["n_pathways"],
            prot_proj_dropout=PROT_PROJ_DROPOUT,
            prot_proj_inner_dropout=PROT_PROJ_INNER_DROPOUT,
            pathway_proj_dropout=PATHWAY_PROJ_DROPOUT,
            pheno_head_dropout=PHENO_HEAD_DROPOUT,
            temperature=TEMPERATURE,
            decoder_type=DECODER_TYPE)
        if DECODER_TYPE == "residue_bilinear" and residue_embeddings is not None:
            sage_model.set_residue_features(
                residue_embeddings, residue_offsets, residue_lengths,
                prot_to_residue_idx=prot_to_residue_idx,
                max_len=residue_max_len,
                residue_device="cpu",
            )
        sage_checkpoint = torch.load(sage_model_path, map_location=DEVICE, weights_only=False)
        sage_model.load_state_dict(sage_checkpoint["state_dict"])
        sage_model = sage_model.to(DEVICE)
        sage_history = []
        _t0 = time.time()
    else:
        logger.info(f">>> 训练 SAGE（v41: SAGEConv + {DECODER_TYPE} + 两阶段迁移学习 + FocalLoss + 课程负采样）")
        _log_gpu_memory("SAGE 训练前")
        _t0 = time.time()
        sage_model = SAGELinkPredictor(
            comp_feat_dim=graphs["feat_dim"], prot_feat_dim=graphs["prot_esm_dim"],  # 传 ESM-2 维度（640），通路独立投影
            n_compounds=graphs["n_compounds"],
            hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT,
            n_pathways=graphs["n_pathways"],
            prot_proj_dropout=PROT_PROJ_DROPOUT,
            prot_proj_inner_dropout=PROT_PROJ_INNER_DROPOUT,
            pathway_proj_dropout=PATHWAY_PROJ_DROPOUT,
            pheno_head_dropout=PHENO_HEAD_DROPOUT,
            temperature=TEMPERATURE,
            decoder_type=DECODER_TYPE)
        # 注册残基级 ESM-2 特征到 SAGE 解码器（大张量保留在 CPU）
        if DECODER_TYPE == "residue_bilinear" and residue_embeddings is not None:
            sage_model.set_residue_features(
                residue_embeddings, residue_offsets, residue_lengths,
                prot_to_residue_idx=prot_to_residue_idx,
                max_len=residue_max_len,
                residue_device="cpu",
            )
        sage_model, sage_history = train_sage(
        sage_model, graphs, train_compounds, val_compounds, compound_to_pos,
        device=DEVICE,
        val_proteins=val_proteins,
        epochs=EPOCHS, lr=LEARNING_RATE_SAGE, patience=PATIENCE, batch_size=SAGE_BATCH_SIZE,
        num_neighbors=SAGE_NUM_NEIGHBORS,
        prot_to_path_neighbors=graphs.get("prot_to_path_neighbors"),
        two_stage=True, pretrain_epochs=PRETRAIN_EPOCHS, pretrain_lr=PRETRAIN_LR_SAGE,
        random_seed=RANDOM_SEED,
        pheno_compound_indices=pheno_train_indices,
        pheno_labels=pheno_train_labels,
        pheno_lambda=PHENO_LAMBDA,
        bpr_weight=BPR_WEIGHT, weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_RATIO,
        dropedge_ppi=DROPPEDGE_PPI, dropedge_pathway=DROPPEDGE_PATHWAY,
        focal_gamma=FOCAL_GAMMA, focal_alpha=FOCAL_ALPHA,
        memory_bank_size=MEMORY_BANK_SIZE,
        head_ratio=HEAD_RATIO, lambda_hhi=LAMBDA_HHI,
        grad_clip_norm=GRAD_CLIP_NORM,
        pretrain_lr_multiplier=PRETRAIN_LR_MULTIPLIER, pretrain_lr_decay=PRETRAIN_LR_DECAY,
        use_topology_neg=USE_TOPOLOGY_NEG,
        _validate_sage_fn=_validate_sage, _compute_cpi_loss_fn=_compute_cpi_loss)

    try:
        torch.save({"state_dict": sage_model.state_dict(), "version": "v42", "hidden_dim": HIDDEN_DIM, "out_dim": OUT_DIM}, L4_RESULTS / "sage_best_v42.pt")
        logger.info("  SAGE 模型已保存到 sage_best_v42.pt")
    except Exception:
        logger.error("  SAGE 模型保存失败", exc_info=True)
        raise

    # train_sage 的早停加载最佳参数时会把参数 clone 到 CPU，
    # 预测前必须移回 DEVICE。
    sage_model = sage_model.to(DEVICE)

    _t0 = _log_step_time(_t0, "SAGE 训练完成")
    torch.cuda.empty_cache()
    _log_gpu_memory("SAGE 训练后 (cache cleared)")
    logger.info("  SAGE GPU 内存已释放")

    # 释放 SAGE 模型中的残基级 ESM-2 特征（~8.86GB），避免 HGT 初始化时
    # 与新加载的残基特征同时驻留 CPU 触发 OOM（系统总 RAM 仅 15.2GB）
    if DECODER_TYPE == "residue_bilinear":
        try:
            sage_model.free_residue_features()
            logger.info("  v37-fix: SAGE 残基特征已释放，释放 CPU 内存约 8.86GB")
        except Exception as e:
            logger.warning(f"  v37-fix: SAGE 残基特征释放失败: {e}")
        import gc
        gc.collect()

    if skip_hgt:
        logger.info(">>> 跳过 HGT 训练")
        hgt_history = []
    else:
        logger.info(f">>> 训练 HGT（v41: HGTConv + {DECODER_TYPE} + 两阶段迁移学习 + FocalLoss + 课程负采样）")
        _log_gpu_memory("HGT 训练前")
    if skip_hgt:
        hgt_model = None
        hgt_history = []
    else:
        hgt_node_feat_dims = {
            "compound": graphs["feat_dim"],
            "protein": graphs["prot_esm_dim"],  # 使用 ESM-2 维度（640），通路信息由异质图结构传递
            "pathway": 1,
            "pathway_count": graphs["n_pathways"],
            "disease_count": graphs.get("n_diseases", 0),
        }
        hgt_model = HGTLinkPredictor(
            hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
            dropout=DROPOUT, metadata=graphs["hetero_data"].metadata(),
            compound_feat_dim=graphs["feat_dim"], node_feat_dims=hgt_node_feat_dims,
            pheno_head_dropout=PHENO_HEAD_DROPOUT,
            temperature=TEMPERATURE,
            decoder_type=DECODER_TYPE)
        # 注册残基级 ESM-2 特征到 HGT 解码器（大张量保留在 CPU）
        if DECODER_TYPE == "residue_bilinear" and residue_embeddings is not None:
            hgt_model.set_residue_features(
                residue_embeddings, residue_offsets, residue_lengths,
                prot_to_residue_idx=prot_to_residue_idx,
                max_len=residue_max_len,
                residue_device="cpu",
            )
        hgt_model, hgt_history = train_hgt(
            hgt_model, graphs, train_compounds, val_compounds, compound_to_pos,
            device=DEVICE,
            val_proteins=val_proteins,
            epochs=EPOCHS, lr=LEARNING_RATE_HGT, patience=PATIENCE, batch_size=HGT_BATCH_SIZE,
            num_neighbors=HGT_NUM_NEIGHBORS,
            prot_to_path_neighbors=graphs.get("prot_to_path_neighbors"),
            two_stage=True, pretrain_epochs=PRETRAIN_EPOCHS, pretrain_lr=PRETRAIN_LR_HGT,
            random_seed=RANDOM_SEED,
            pheno_compound_indices=pheno_train_indices,
            pheno_labels=pheno_train_labels,
            pheno_lambda=PHENO_LAMBDA,
            bpr_weight=BPR_WEIGHT, weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_RATIO,
            dropedge_ppi=DROPPEDGE_PPI, dropedge_pathway=DROPPEDGE_PATHWAY,
            focal_gamma=FOCAL_GAMMA, focal_alpha=FOCAL_ALPHA,
            memory_bank_size=MEMORY_BANK_SIZE,
            head_ratio=HEAD_RATIO, lambda_hhi=LAMBDA_HHI,
            grad_clip_norm=GRAD_CLIP_NORM,
            pretrain_lr_multiplier=PRETRAIN_LR_MULTIPLIER, pretrain_lr_decay=PRETRAIN_LR_DECAY,
            use_topology_neg=USE_TOPOLOGY_NEG,
            _validate_hgt_fn=_validate_hgt, _compute_cpi_loss_fn=_compute_cpi_loss)

        try:
            torch.save({"state_dict": hgt_model.state_dict(), "version": "v42", "hidden_dim": HIDDEN_DIM, "out_dim": OUT_DIM}, L4_RESULTS / "hgt_best_v42.pt")
            logger.info("  HGT 模型已保存到 hgt_best_v42.pt")
        except Exception:
            logger.error("  HGT 模型保存失败", exc_info=True)
            raise

        # train_hgt 的早停加载最佳参数时会把参数 clone 到 CPU，预测前必须移回 DEVICE。
        hgt_model = hgt_model.to(DEVICE)

        _t0 = _log_step_time(_t0, "HGT 训练完成")
        torch.cuda.empty_cache()
        _log_gpu_memory("HGT 训练后 (cache cleared)")
        logger.info("  HGT GPU 内存已释放")

    # 蛋白冷启动评估已移除（被独立评估脚本替代）
    # 集成权重固定为等权
    sage_best_val_aupr = 0.5
    hgt_best_val_aupr = 0.5
    logger.info("  v42: 蛋白冷启动评估已移除，集成权重设为等权 (0.5/0.5)")

    # 加载树模型 v7 TCM 预测用于集成
    tree_pred_df = None
    tree_pred_path = L4_RESULTS / "tree_v6_tcm_predictions_v7.csv"
    if tree_pred_path.exists():
        tree_pred_df = pd.read_csv(tree_pred_path, low_memory=False)
        logger.info(f"v40: 树模型预测加载: {len(tree_pred_df)} 条 "
                    f"({tree_pred_df['MOL_ID'].nunique()} 化合物 x {tree_pred_df['gene'].nunique()} 基因)")
    else:
        logger.warning(f"v40: 树模型预测文件不存在: {tree_pred_path}，跳过树模型集成")

    # v45: predict_tcm 需要残基特征，SAGE 训练后可能已释放，重新注册。
    if DECODER_TYPE == "residue_bilinear" and residue_embeddings is not None:
        if sage_model is not None:
            sage_model.set_residue_features(
                residue_embeddings, residue_offsets, residue_lengths,
                prot_to_residue_idx=prot_to_residue_idx,
                max_len=residue_max_len,
                residue_device="cpu",
            )
            logger.info("  v45: SAGE 残基特征已重新注册，供预测使用")
        if hgt_model is not None:
            hgt_model.set_residue_features(
                residue_embeddings, residue_offsets, residue_lengths,
                prot_to_residue_idx=prot_to_residue_idx,
                max_len=residue_max_len,
                residue_device="cpu",
            )
            logger.info("  v45: HGT 残基特征已重新注册，供预测使用")

    logger.info(">>> 预测 TCM 化合物（v40: 86基因 + 树模型集成 + 铁死亡表型融合）")
    # 检查SMILES列
    tcm_smiles_col = "SMILES_std" if "SMILES_std" in tcm_df.columns else (
        "SMILES" if "SMILES" in tcm_df.columns else "canonical_smiles")
    tcm_smiles = tcm_df[tcm_smiles_col].dropna().tolist()
    all_train_smiles = list(graphs["smi_to_idx"].keys())
    _, cp_mean, cp_std, cp_col_mean = build_compound_features(all_train_smiles)
    compound_stats = (cp_mean, cp_std, cp_col_mean)

    # 预测靶标 = 全部 115 个铁衰老相关基因
    # warm = 42个（有CPI训练数据）, zero-shot = 73个（无CPI训练数据）
    # 排序仅基于 42 个 warm 靶标的加权得分，zero-shot 仅作参考
    all_target_genes = all_target_genes_pred
    zero_shot_genes = set(all_target_genes) - set(warm_targets)
    warm_genes_set = set(warm_targets)
    logger.info(f"  预测靶标: {len(all_target_genes)} 个 (warm={len(warm_targets)}, zero-shot={len(zero_shot_genes)})")
    logger.info(f"  zero-shot 靶标: {sorted(zero_shot_genes)}")

    # 预计算 TCM 化合物特征，避免 predict_tcm 和 pheno 预测各算一次
    tcm_feat_raw, _, _, _ = build_compound_features(tcm_smiles, stats=compound_stats)
    feat_dim = graphs["feat_dim"]
    if tcm_feat_raw.shape[1] < feat_dim:
        tcm_feat_raw = np.pad(tcm_feat_raw, ((0, 0), (0, feat_dim - tcm_feat_raw.shape[1])), mode="constant")
    tcm_feat_precomputed = torch.from_numpy(tcm_feat_raw)

    pred_df = predict_tcm(
        sage_model, hgt_model, graphs, tcm_smiles, all_target_genes,
        compound_stats,
        mc_samples=MC_SAMPLES,
        tcm_feat_precomputed=tcm_feat_precomputed,
        tree_predictions=tree_pred_df,  # 树模型预测集成
        tree_weight=TREE_ENSEMBLE_WEIGHT)  # 树模型权重

    # 铁死亡概率预测 — SAGE + HGT 双分支平均
    final_ferroptosis_prob = None
    if pheno_train_indices is not None and len(pheno_train_indices) > 0 and hgt_model is not None:
        logger.info(">>> 预测 TCM 化合物铁死亡概率（表型分类头）")
        sage_model.eval()
        hgt_model.eval()
        with torch.no_grad():
            tcm_feat_tensor = tcm_feat_precomputed.to(DEVICE)

            sage_tcm_emb = sage_model.encode_compound(tcm_feat_tensor)
            sage_ferro_logits = sage_model.predict_phenotype(sage_tcm_emb)
            sage_ferro_prob = torch.sigmoid(sage_ferro_logits).squeeze(-1).cpu().numpy()

            hgt_tcm_emb = hgt_model.encode_compound(tcm_feat_tensor)
            hgt_ferro_logits = hgt_model.predict_phenotype(hgt_tcm_emb)
            hgt_ferro_prob = torch.sigmoid(hgt_ferro_logits).squeeze(-1).cpu().numpy()

            final_ferroptosis_prob = (sage_ferro_prob + hgt_ferro_prob) / 2.0
            pred_df["ferroptosis_prob"] = final_ferroptosis_prob
            logger.info(f"  铁死亡概率范围: [{final_ferroptosis_prob.min():.4f}, {final_ferroptosis_prob.max():.4f}], "
                        f"均值={final_ferroptosis_prob.mean():.4f}")
    else:
        logger.info(">>> 无表型训练数据或跳过HGT，跳过铁死亡概率预测")

    if "MOL_ID" in tcm_df.columns and "molecule_name" in tcm_df.columns and "SMILES_std" in tcm_df.columns:
        name_map = dict(zip(tcm_df["SMILES_std"], tcm_df["molecule_name"], strict=False))
        mol_id_map = dict(zip(tcm_df["SMILES_std"], tcm_df["MOL_ID"], strict=False))
        pred_df["molecule_name"] = pred_df["SMILES"].map(name_map).fillna("")
        pred_df["MOL_ID"] = pred_df["SMILES"].map(mol_id_map).fillna("")

    # 中药来源 & 综合评分
    if "herb_origins" in tcm_df.columns and "SMILES_std" in tcm_df.columns:
        herb_map = dict(zip(tcm_df["SMILES_std"], tcm_df["herb_origins"], strict=False))
        pred_df["herb_origins"] = pred_df["SMILES"].map(herb_map).fillna("")
    if "n_herbs" in tcm_df.columns and "SMILES_std" in tcm_df.columns:
        nherb_map = dict(zip(tcm_df["SMILES_std"], tcm_df["n_herbs"], strict=False))
        pred_df["n_herbs"] = pred_df["SMILES"].map(nherb_map).fillna(0).astype(int)
    if "comprehensive_score" in tcm_df.columns and "SMILES_std" in tcm_df.columns:
        score_map = dict(zip(tcm_df["SMILES_std"], tcm_df["comprehensive_score"], strict=False))
        pred_df["tcm_pool_score"] = pred_df["SMILES"].map(score_map).fillna(0.0)
    if "tier" in tcm_df.columns and "SMILES_std" in tcm_df.columns:
        tier_map = dict(zip(tcm_df["SMILES_std"], tcm_df["tier"], strict=False))
        pred_df["tcm_pool_tier"] = pred_df["SMILES"].map(tier_map).fillna("")
    if "is_whitelist" in tcm_df.columns and "SMILES_std" in tcm_df.columns:
        wl_map = dict(zip(tcm_df["SMILES_std"], tcm_df["is_whitelist"], strict=False))
        pred_df["is_whitelist"] = pred_df["SMILES"].map(wl_map).fillna(False)
    # 标记TCM池中与训练集重叠的化合物（数据泄漏标记）
    train_smi_set = set(cpi_df["canonical_smiles"].dropna().unique())
    pred_df["in_train"] = pred_df["SMILES"].isin(train_smi_set)
    n_in_train = pred_df["in_train"].sum()
    if n_in_train > 0:
        logger.warning(f"  TCM池中 {n_in_train} 个化合物与训练集重叠（已标记为 in_train=True），"
                       f"建议人工审核这些化合物的预测得分")

    # 排序仅基于 42 个 warm 靶标的加权得分（zero-shot 不参与主排序）
    # 原因：zero-shot 预测分数校准性差，直接参与排序会引入噪声
    all_gene_cols = [g for g in all_target_genes if g in pred_df.columns]
    warm_gene_cols = [g for g in all_gene_cols if g in warm_genes_set]
    zs_gene_cols = [g for g in all_gene_cols if g not in warm_genes_set]
    logger.info(f"  排序用 warm 靶标: {len(warm_gene_cols)} 个")
    logger.info(f"  参考用 zero-shot 靶标: {len(zs_gene_cols)} 个")

    warm_scores = pred_df[warm_gene_cols].values

    # 构建 warm 靶标权重向量
    target_weights = np.array([TARGET_PRIORITY.get(g, _DEFAULT_PRIORITY) for g in warm_gene_cols])
    weights_sum = target_weights.sum()
    logger.info(f"  warm 靶标权重范围: [{target_weights.min():.1f}, {target_weights.max():.1f}], "
                f"总权重={weights_sum:.1f}")

    # 加权平均分
    weighted_avg = np.sum(warm_scores * target_weights, axis=1) / weights_sum

    # 加权最高分（相对权重缩放，核心靶标得分被放大）
    weighted_max = np.max(warm_scores * (target_weights / target_weights.max()), axis=1)

    # 加权命中数（核心靶标命中权重更高）
    weighted_hits = np.sum((warm_scores > 0.5).astype(float) * target_weights, axis=1) / weights_sum

    def _norm(x):
        return (x - x.min()) / (x.max() - x.min() + EPS)

    # 加权 composite，核心靶标权重更大
    composite = (
        COMPOSITE_AVG_WEIGHT * _norm(weighted_avg)
        + COMPOSITE_MAX_WEIGHT * _norm(weighted_max)
        + COMPOSITE_HITS_WEIGHT * _norm(weighted_hits)
    )

    # 不确定性调整 — 优先选择高分且低不确定度的化合物
    # v44: 改为温和惩罚，最大惩罚 50%，避免高不确定性高潜力分子被完全抑制。
    if "mean_uncertainty" in pred_df.columns:
        uncertainty = pred_df["mean_uncertainty"].values
        uncertainty_penalty = 1.0 - 0.5 * _norm(uncertainty)
        composite = composite * uncertainty_penalty
        pred_df["uncertainty_penalty"] = uncertainty_penalty
        logger.info(f"  不确定性调整: 惩罚范围 [{uncertainty_penalty.min():.4f}, {uncertainty_penalty.max():.4f}] (最大惩罚50%)")

    # 保留全部靶标的原始指标供参考（含 zero-shot）
    all_scores = pred_df[all_gene_cols].values
    raw_avg_all = np.nanmean(all_scores, axis=1)
    raw_max_all = np.nanmax(all_scores, axis=1)
    raw_n_hits_all = np.nansum(all_scores > 0.5, axis=1)

    # warm 靶标的原始指标
    raw_avg_warm = np.nanmean(warm_scores, axis=1)
    raw_max_warm = np.nanmax(warm_scores, axis=1)
    raw_n_hits_warm = np.nansum(warm_scores > 0.5, axis=1)

    # zero-shot 靶标参考指标（仅参考，不参与排序）
    if zs_gene_cols:
        zs_scores = pred_df[zs_gene_cols].values
        zs_avg = np.nanmean(zs_scores, axis=1)
        zs_max = np.nanmax(zs_scores, axis=1)
        zs_n_hits = np.nansum(zs_scores > 0.5, axis=1)
        pred_df["zs_avg_score"] = zs_avg
        pred_df["zs_max_score"] = zs_max
        pred_df["zs_n_hits"] = zs_n_hits
        pred_df["zs_n_targets"] = len(zs_gene_cols)
        # zero-shot bonus: 若 zero-shot 高分，给 composite 加一个小 bonus（不超过5%）
        zs_bonus = ZS_BONUS_MAX * _norm(zs_avg)
        composite = composite + zs_bonus
        pred_df["zs_bonus"] = zs_bonus
        logger.info(f"  zero-shot bonus: 范围 [{zs_bonus.min():.4f}, {zs_bonus.max():.4f}] (上限{int(ZS_BONUS_MAX*100)}%)")

    # 铁死亡概率融合 — 高铁死亡概率的化合物得分更高
    if final_ferroptosis_prob is not None:
        ferro_factor = FERRO_FACTOR_BASE + (1.0 - FERRO_FACTOR_BASE) * final_ferroptosis_prob
        composite = composite * ferro_factor
        pred_df["ferroptosis_factor"] = ferro_factor
        logger.info(f"  铁死亡融合因子: 范围 [{ferro_factor.min():.4f}, {ferro_factor.max():.4f}], "
                    f"均值={ferro_factor.mean():.4f}")

    pred_df["composite_score"] = composite
    pred_df["avg_score_all"] = raw_avg_all
    pred_df["max_score_all"] = raw_max_all
    pred_df["n_hits_all"] = raw_n_hits_all
    pred_df["n_targets_all"] = len(all_gene_cols)
    pred_df["avg_score_warm"] = raw_avg_warm
    pred_df["max_score_warm"] = raw_max_warm
    pred_df["n_hits_warm"] = raw_n_hits_warm
    pred_df["n_targets_warm"] = len(warm_gene_cols)
    pred_df["weighted_avg"] = weighted_avg
    pred_df["weighted_max"] = weighted_max
    pred_df["weighted_hits"] = weighted_hits

    # top targets: 显示 warm 靶标 top5 + zero-shot top3
    top_targets_list = []
    for i in range(len(pred_df)):
        warm_gs = [(g, warm_scores[i][j]) for j, g in enumerate(warm_gene_cols)]
        warm_gs.sort(key=lambda x: x[1], reverse=True)
        warm_top = [f"{g}({s:.3f})" for g, s in warm_gs[:WARM_TARGETS_TOP_N]]
        suffix = ""
        if zs_gene_cols:
            zs_gs = [(g, zs_scores[i][j]) for j, g in enumerate(zs_gene_cols)]
            zs_gs.sort(key=lambda x: x[1], reverse=True)
            zs_top = [f"{g}*({s:.3f})" for g, s in zs_gs[:ZS_TARGETS_TOP_N]]
            suffix = " | ZS: " + ", ".join(zs_top)
        top_targets_list.append(", ".join(warm_top) + suffix)
    pred_df["top_targets"] = top_targets_list

    pred_df = pred_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    pred_df["rank"] = range(1, len(pred_df) + 1)
    top_df = pred_df.head(TOP_N_CANDIDATES).copy()

    try:
        pred_df.to_csv(L4_RESULTS / "tcm_predictions_full_v41.csv", index=False)
        top_df.to_csv(L4_RESULTS / "tcm_top_candidates_v41.csv", index=False)
        logger.info(f"  预测结果已保存: tcm_predictions_full_v41.csv ({len(pred_df)} 行), "
                    f"tcm_top_candidates_v41.csv ({len(top_df)} 行)")
    except Exception:
        logger.error("  预测结果 CSV 保存失败", exc_info=True)
        raise

    # 性能
    # 扩展模型性能报告，包含排名指标、训练时间、GPU 显存
    perf_rows = []
    # 收集 GPU 显存峰值
    gpu_mem_peak_gb = 0.0
    if torch.cuda.is_available():
        gpu_mem_peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    train_time_min = (time.time() - start_time) / 60.0

    if sage_history:
        sage_best_auc = max(h["auc"] for h in sage_history)
        sage_best_aupr = max(h.get("aupr", 0) for h in sage_history)
        sage_row = {"model": "SAGE", "best_auc": sage_best_auc, "best_aupr": sage_best_aupr}
        sage_row["train_time_min"] = round(train_time_min, 1)
        sage_row["gpu_mem_peak_gb"] = round(gpu_mem_peak_gb, 2)
        perf_rows.append(sage_row)
    if hgt_history:
        hgt_best_auc = max(h["auc"] for h in hgt_history)
        hgt_best_aupr = max(h.get("aupr", 0) for h in hgt_history)
        hgt_row = {"model": "HGT", "best_auc": hgt_best_auc, "best_aupr": hgt_best_aupr}
        hgt_row["train_time_min"] = round(train_time_min, 1)
        hgt_row["gpu_mem_peak_gb"] = round(gpu_mem_peak_gb, 2)
        perf_rows.append(hgt_row)
    if perf_rows:
        try:
            pd.DataFrame(perf_rows).to_csv(L4_RESULTS / "model_performance_v41.csv", index=False)
            logger.info(f"  模型性能报告已保存: model_performance_v41.csv (训练时间={train_time_min:.1f}min, GPU峰值={gpu_mem_peak_gb:.2f}GB)")
        except Exception:
            logger.error("  模型性能 CSV 保存失败", exc_info=True)
            raise

    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Phase 4 v41 完成！总耗时 {total_time / 60:.1f} 分钟")
    if sage_history:
        logger.info(f"  SAGE best val_auc: {max(h['auc'] for h in sage_history):.4f}  val_aupr: {max(h.get('aupr', 0) for h in sage_history):.4f}")
    if hgt_history:
        logger.info(f"  HGT best val_auc: {max(h['auc'] for h in hgt_history):.4f}  val_aupr: {max(h.get('aupr', 0) for h in hgt_history):.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 v43: SAGE + HGT + 树模型集成训练与 TCM 预测")
    parser.add_argument(
        "--decoder_type",
        type=str,
        default=None,
        choices=["mlp", "dot", "bilinear", "residue_bilinear"],
        help="覆盖配置中的解码器类型（默认从 config 读取）",
    )
    parser.add_argument(
        "--skip_sage",
        action="store_true",
        help="跳过 SAGE 训练（用于快速测试 HGT 或 TCM 预测）",
    )
    parser.add_argument(
        "--skip_hgt",
        action="store_true",
        help="跳过 HGT 训练（用于快速测试 SAGE）",
    )
    parser.add_argument(
        "--sage_epochs",
        type=int,
        default=None,
        help="覆盖 SAGE 训练 epoch 数（快速测试用）",
    )
    parser.add_argument(
        "--hgt_epochs",
        type=int,
        default=None,
        help="覆盖 HGT 训练 epoch 数（快速测试用）",
    )
    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=None,
        help="覆盖预训练 epoch 数（快速测试用）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="覆盖随机种子",
    )
    args = parser.parse_args()
    # 应用 CLI 覆盖到全局常量
    global_overrides = {}
    if args.sage_epochs is not None:
        global_overrides["EPOCHS"] = args.sage_epochs
    if args.hgt_epochs is not None:
        global_overrides["EPOCHS"] = args.hgt_epochs
    if args.pretrain_epochs is not None:
        global_overrides["PRETRAIN_EPOCHS"] = args.pretrain_epochs
    if args.seed is not None:
        global_overrides["RANDOM_SEED"] = args.seed
    main(decoder_type=args.decoder_type, skip_sage=args.skip_sage, skip_hgt=args.skip_hgt,
         global_overrides=global_overrides)
