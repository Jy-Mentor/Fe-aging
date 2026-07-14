#!/usr/bin/env python3
"""Phase 4 v70 MODULARIZED: Mini-Batch GNN 三分支 — 化合物冷启动候选化合物发现

SAGE + HGT + SimpleHGN 三分支集成：拓扑 (SAGEConv) + 语义 (HGTConv) + 边类型感知 (SimpleHGN)，支持：
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
import json
import logging
import math
import os
import pickle
import random
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger

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
from iron_aging_gnn.models import SAGELinkPredictor, HGTLinkPredictor, SimpleHGNLinkPredictor  # noqa: E402
from iron_aging_gnn.training.trainer import train_sage, train_hgt, train_simplehgn  # noqa: E402
from iron_aging_gnn.utils.config import CompoundFeatureConfig, load_config  # noqa: E402

for d in [L4_RESULTS, L4_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = L4_LOGS / "phase4_v67_full_train.log"

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
# v60-fix: SAGE 与 HGT 解耦容量。SAGE 保持轻量避免收敛困难；HGT 使用 model 级 128/3 配置。
SAGE_HIDDEN_DIM = getattr(_cfg.sage, "hidden_dim", HIDDEN_DIM) if _cfg else HIDDEN_DIM
SAGE_OUT_DIM = getattr(_cfg.sage, "out_dim", OUT_DIM) if _cfg else OUT_DIM
SAGE_NUM_LAYERS = getattr(_cfg.sage, "num_layers", NUM_LAYERS) if _cfg else NUM_LAYERS
DROPOUT = _cfg.model.dropout if _cfg else 0.3
PROT_PROJ_DROPOUT = _cfg.model.prot_proj_dropout if _cfg else 0.4
PROT_PROJ_INNER_DROPOUT = _cfg.model.prot_proj_inner_dropout if _cfg else 0.3
PATHWAY_PROJ_DROPOUT = _cfg.model.pathway_proj_dropout if _cfg else 0.3
PHENO_HEAD_DROPOUT = _cfg.model.pheno_head_dropout if _cfg else 0.3
TEMPERATURE = _cfg.model.temperature if _cfg else 1.0

FOCAL_GAMMA = _cfg.loss.focal_gamma if _cfg else 1.0
FOCAL_ALPHA = _cfg.loss.focal_alpha if _cfg else 0.6
LABEL_SMOOTHING_POS = _cfg.loss.label_smoothing_pos if _cfg else 0.95
LABEL_SMOOTHING_NEG = _cfg.loss.label_smoothing_neg if _cfg else 0.05
BPR_WEIGHT = _cfg.loss.bpr_weight if _cfg else 0.4
CPI_LOSS_WEIGHT = _cfg.loss.bce_weight if _cfg else 0.6
INFONCE_WEIGHT = _cfg.loss.infonce_weight if _cfg else 0.0

LEARNING_RATE_SAGE = _cfg.sage.lr if _cfg else 5e-4
LEARNING_RATE_HGT = _cfg.hgt.lr if _cfg else 1e-3
PRETRAIN_LR_MULTIPLIER = _cfg.two_stage.pretrain_lr_multiplier if _cfg else 1.5
PRETRAIN_LR_DECAY = _cfg.two_stage.pretrain_lr_decay if _cfg else 0.5
FINETUNE_LR_MULTIPLIER = _cfg.sage.finetune_lr_multiplier if _cfg else 0.5
USE_PLATEAU_SCHEDULER_SAGE = _cfg.sage.use_plateau_scheduler if _cfg else True
PLATEAU_PATIENCE_SAGE = _cfg.sage.plateau_patience if _cfg else 2
PLATEAU_FACTOR_SAGE = _cfg.sage.plateau_factor if _cfg else 0.5
FINETUNE_LR_MULTIPLIER_HGT = _cfg.hgt.finetune_lr_multiplier if _cfg else 0.5
USE_PLATEAU_SCHEDULER_HGT = _cfg.hgt.use_plateau_scheduler if _cfg else True
PLATEAU_PATIENCE_HGT = _cfg.hgt.plateau_patience if _cfg else 2
PLATEAU_FACTOR_HGT = _cfg.hgt.plateau_factor if _cfg else 0.5
WEIGHT_DECAY = _cfg.training.weight_decay if _cfg else 1e-4
GRAD_CLIP_NORM = _cfg.training.grad_clip_norm if _cfg else 1.0
WARMUP_RATIO = _cfg.training.warmup_ratio if _cfg else 0.10
DROPPEDGE_PPI = _cfg.training.dropedge_ppi if _cfg else 0.05
DROPPEDGE_PATHWAY = _cfg.training.dropedge_pathway if _cfg else 0.05
DROPPEDGE_CPI = _cfg.training.dropedge_cpi if _cfg else 0.0
EPOCHS = _cfg.sage.epochs if _cfg else 15
EPOCHS_HGT = _cfg.hgt.epochs if _cfg else 15
PATIENCE = _cfg.sage.patience if _cfg else 5
PRETRAIN_EPOCHS = _cfg.sage.pretrain_epochs if _cfg else 10
PRETRAIN_EPOCHS_HGT = _cfg.hgt.pretrain_epochs if _cfg else 10
PRETRAIN_LR_SAGE = _cfg.sage.pretrain_lr if _cfg else 7.5e-4
PRETRAIN_LR_HGT = _cfg.hgt.pretrain_lr if _cfg else 1.5e-3
SAGE_BATCH_SIZE = _cfg.sage.batch_size if _cfg else 256
HGT_BATCH_SIZE = _cfg.hgt.batch_size if _cfg else 128
SAGE_NUM_NEIGHBORS = _cfg.sage.num_neighbors if _cfg else [32, 16]
HGT_NUM_NEIGHBORS = _cfg.hgt.num_neighbors if _cfg else [32, 16]
# SimpleHGN 超参（从 HGT 直接迁移，hidden_dim=128, 2 层, 2 头注意力）
EPOCHS_SIMPLEHGN = _cfg.simplehgn.epochs if _cfg else 15
PRETRAIN_EPOCHS_SIMPLEHGN = _cfg.simplehgn.pretrain_epochs if _cfg else 10
PRETRAIN_LR_SIMPLEHGN = _cfg.simplehgn.pretrain_lr if _cfg else 1.5e-3
LEARNING_RATE_SIMPLEHGN = _cfg.simplehgn.lr if _cfg else 1e-3
SIMPLEHGN_BATCH_SIZE = _cfg.simplehgn.batch_size if _cfg else 128
SIMPLEHGN_NUM_NEIGHBORS = _cfg.simplehgn.num_neighbors if _cfg else [32, 16]
SIMPLEHGN_VAL_NUM_NEIGHBORS = _cfg.simplehgn.val_num_neighbors if _cfg else [8, 4]
SIMPLEHGN_VAL_BATCH_SIZE = _cfg.simplehgn.val_batch_size if _cfg else 256
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

SCORE_CLAMP = _cfg.model.score_clamp if _cfg else 10
DECODER_TYPE = _cfg.model.decoder_type if _cfg else "mlp"
DECODER_INIT_SCHEME = _cfg.decoder.init_scheme if _cfg else "xavier"
DECODER_FINAL_BIAS_INIT = _cfg.decoder.final_bias_init if _cfg else -0.5
MAX_RESIDUE_BATCH = _cfg.decoder.max_residue_batch if _cfg else 2

MEMORY_BANK_SIZE = _cfg.memory_bank.memory_bank_size if _cfg else 4096
INFONCE_WARMUP_RATIO = _cfg.two_stage.infonce_warmup_ratio if _cfg else 0.08
INFONCE_MEM_SAMPLE = _cfg.memory_bank.infonce_mem_sample if _cfg else 128
INFONCE_TEMPERATURE = _cfg.loss.infonce_temperature if _cfg else 0.07

# v67: 辅助网络重建损失参数（DHGT-DTI/MHGNN-DTI 风格）
AUX_RECON_WEIGHT = _cfg.loss.aux_recon_weight if _cfg else 0.0
AUX_RECON_PPI_SAMPLES = _cfg.loss.aux_recon_ppi_samples if _cfg else 256
AUX_RECON_PATHWAY_SAMPLES = _cfg.loss.aux_recon_pathway_samples if _cfg else 128
AUX_RECON_DDI_SAMPLES = _cfg.loss.aux_recon_ddi_samples if _cfg else 128
AUX_RECON_DRUG_DISEASE_SAMPLES = _cfg.loss.aux_recon_drug_disease_samples if _cfg else 128
AUX_RECON_PROT_DISEASE_SAMPLES = _cfg.loss.aux_recon_protein_disease_samples if _cfg else 128
AUX_RECON_DRUG_SIDE_EFFECT_SAMPLES = _cfg.loss.aux_recon_drug_side_effect_samples if _cfg else 128
SEMANTIC_ATTN_WEIGHT = _cfg.loss.semantic_attn_weight if _cfg else 0.0
SEMANTIC_ATTN_TEMPERATURE = _cfg.loss.semantic_attn_temperature if _cfg else 0.5

# v67: 冷启动评估参数（GHCDTI 风格）
ENABLE_COLD_DRUG_EVAL = _cfg.validation.enable_cold_drug_eval if _cfg else False
ENABLE_COLD_TARGET_EVAL = _cfg.validation.enable_cold_target_eval if _cfg else False
COLD_DRUG_SPLIT_RATIO = _cfg.validation.cold_drug_split_ratio if _cfg else 0.2
COLD_TARGET_SPLIT_RATIO = _cfg.validation.cold_target_split_ratio if _cfg else 0.2

# v67: 元路径与 Graph Transformer 配置
META_PATH_ENABLED = _cfg.meta_path.enabled if _cfg else False
GT_ENABLED = _cfg.graph_transformer.enabled if _cfg else False
# v69: CrossModalGatedFusion 配置
USE_CROSS_MODAL_FUSION = getattr(_cfg.model, "use_cross_modal_fusion", False) if _cfg else False
FUSION_HIDDEN_DIM = getattr(_cfg.model, "fusion_hidden_dim", 64) if _cfg else 64

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
HGT_VAL_USE_RESIDUE_FOR_POS = getattr(_cfg.validation, "hgt_val_use_residue_for_pos", True) if _cfg else True

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
# v67 module imports — 所有功能从 iron_aging_gnn 包模块导入
from iron_aging_gnn.data.features import build_compound_features, load_protein_features, FeatureCache  # noqa: E402
from iron_aging_gnn.data.loader import load_cpi_data, load_ppi_network, load_kegg_pathways, load_tcm_pool  # noqa: E402
from iron_aging_gnn.data.self_check import pipeline_self_check  # noqa: E402
from iron_aging_gnn.graph.build import build_graphs_and_adj  # noqa: E402
from iron_aging_gnn.graph.validation_graphs import build_val_safe_homo_edge_index, build_val_safe_hetero_data, build_val_comp_cold_homo_edge_index, build_val_comp_cold_hetero_data, build_train_safe_homo_adj, build_train_safe_hetero_adj, build_val_comp_cold_hetero_adj, build_val_safe_hetero_adj  # noqa: E402
from iron_aging_gnn.models.losses import compute_cpi_loss, compute_auxiliary_reconstruction_loss  # noqa: E402
from iron_aging_gnn.pipeline.utils import get_prot_feat_dim, check_gpu_memory, log_gpu_memory, log_step_time, check_gradient_norm  # noqa: E402
from iron_aging_gnn.pipeline.validation import validate_sage, validate_hgt, validate_simplehgn  # noqa: E402
from iron_aging_gnn.pipeline.prediction import predict_tcm  # noqa: E402

# ============================================================================
# v67 Wrappers -- pass global constants to module functions
# ============================================================================

def _get_prot_feat_dim(prot_feat):
    return get_prot_feat_dim(prot_feat)


def _validate_sage(model, x, homo_edge_index, val_compounds, all_compound_to_pos, n_compounds,
                   n_proteins=None, return_embeddings=False):
    return validate_sage(model, x, homo_edge_index, val_compounds, all_compound_to_pos, n_compounds,
                         device=DEVICE, score_clamp=SCORE_CLAMP, hard_neg_top_k=HARD_NEG_TOP_K,
                         rand_neg_top_k=RAND_NEG_TOP_K, mask_val=MASK_VAL,
                         neg_ratio=100,
                         return_embeddings=return_embeddings)

def _validate_hgt(model, hetero_data, val_compounds, all_compound_to_pos, n_compounds,
                  n_proteins, **kwargs):
    return validate_hgt(model, hetero_data, val_compounds, all_compound_to_pos, n_compounds,
                        n_proteins, device=DEVICE, score_clamp=SCORE_CLAMP,
                        neg_ratio=100, **kwargs)


def _validate_simplehgn(model, hetero_data, val_compounds, all_compound_to_pos, n_compounds,
                        n_proteins, **kwargs):
    return validate_simplehgn(model, hetero_data, val_compounds, all_compound_to_pos, n_compounds,
                              n_proteins, device=DEVICE, score_clamp=SCORE_CLAMP, **kwargs)









def _build_val_safe_homo_edge_index(homo_edge_index, n_compounds, val_comp_set, val_prot_set=None):
    return build_val_safe_homo_edge_index(homo_edge_index, n_compounds, val_comp_set, val_prot_set)

def _build_val_safe_hetero_data(hetero_data, val_comp_set, val_prot_set=None):
    return build_val_safe_hetero_data(hetero_data, val_comp_set, val_prot_set)

def _build_val_comp_cold_homo_edge_index(homo_edge_index, val_comp_set):
    return build_val_comp_cold_homo_edge_index(homo_edge_index, val_comp_set)

def _build_val_comp_cold_hetero_data(hetero_data, val_comp_set):
    return build_val_comp_cold_hetero_data(hetero_data, val_comp_set)

def _build_train_safe_homo_adj(homo_adj, n_compounds, val_comp_set, val_prot_set=None):
    return build_train_safe_homo_adj(homo_adj, n_compounds, val_comp_set, val_prot_set)

def _build_train_safe_hetero_adj(hetero_adj, n_compounds, val_comp_set, val_prot_set=None):
    return build_train_safe_hetero_adj(hetero_adj, n_compounds, val_comp_set, val_prot_set)

def _build_val_comp_cold_hetero_adj(hetero_adj, n_compounds, val_comp_set, val_prot_set=None):
    return build_val_comp_cold_hetero_adj(hetero_adj, n_compounds, val_comp_set, val_prot_set)

def _build_val_safe_hetero_adj(hetero_adj, n_compounds, val_comp_set, val_prot_set=None):
    return build_val_safe_hetero_adj(hetero_adj, n_compounds, val_comp_set, val_prot_set)

def _check_gpu_memory(min_free_gb=1.0):
    return check_gpu_memory(min_free_gb)

def _log_gpu_memory(tag=''):
    log_gpu_memory(tag)

def _log_step_time(start_time, step_name):
    return log_step_time(start_time, step_name)

def _check_gradient_norm(model, warn_threshold=100.0):
    return check_gradient_norm(model, warn_threshold)


_compute_cpi_loss = compute_cpi_loss  # v67: trainer passes all args as kwargs, no wrapper needed

_compute_auxiliary_reconstruction_loss = compute_auxiliary_reconstruction_loss  # v67: trainer passes all args as kwargs
# -- Kept local functions --

def _check_tensor_nan(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """检查张量是否包含 NaN 或 Inf，返回 True 表示有问题"""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        logger.warning(f"张量 {name} 包含 NaN 或 Inf: "
                       f"NaN={torch.isnan(tensor).sum().item()}, "
                       f"Inf={torch.isinf(tensor).sum().item()}")
        return True
    return False

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
    # v61-fix: Windows + PyTorch 2.11 + cu128 在 WDDM 下对 mmap-backed torch.load
    # 会触发不可捕获的 ACCESS_VIOLATION (exit code -1073741819)；非 mmap 加载
    # 8.86GB 又超出可用 RAM。因此优先使用预转换的 numpy memmap 格式，该格式在
    # Windows 上稳定且按需分页。回退顺序：memmap -> torch mmap (Linux) -> torch 普通。
    memmap_dir = residue_pt_path.parent
    memmap_path = memmap_dir / (residue_pt_path.stem + ".memmap")
    offsets_path = memmap_dir / (residue_pt_path.stem + "_offsets.npy")
    lengths_path = memmap_dir / (residue_pt_path.stem + "_lengths.npy")
    genes_path = memmap_dir / (residue_pt_path.stem + "_genes.npy")
    meta_path = memmap_dir / (residue_pt_path.stem + "_meta.json")

    if memmap_path.exists() and offsets_path.exists() and lengths_path.exists() and genes_path.exists():
        logger.info(f">>> 加载残基级 ESM-2 特征 (numpy memmap): {memmap_path}")
        try:
            with open(meta_path, "r", encoding="utf-8") as _f:
                meta = json.load(_f)
            total_residues = int(meta["total_residues"])
            dim = int(meta["dim"])
            np_embeddings = np.memmap(memmap_path, dtype=np.float32, mode="r", shape=(total_residues, dim))
            offsets = torch.from_numpy(np.load(offsets_path))
            lengths = torch.from_numpy(np.load(lengths_path))
            residue_genes = list(np.load(genes_path, allow_pickle=True))
            # memmap 数组保持为 np.ndarray；decoder 前向时会 .to(device) 触发按需读取。
            embeddings = np_embeddings
            n_residue_proteins = len(residue_genes)
            logger.info(f"  残基特征: {n_residue_proteins} 个蛋白, "
                        f"total_residues={total_residues}, dim={dim}, "
                        f"backend=numpy_memmap")
        except Exception:
            logger.error(f"numpy memmap 加载失败: {memmap_path}", exc_info=True)
            raise
    else:
        if not residue_pt_path.exists():
            raise FileNotFoundError(f"残基级 ESM-2 特征文件不存在: {residue_pt_path} (也未找到 memmap)")

        logger.info(f">>> 加载残基级 ESM-2 特征: {residue_pt_path}")
        import platform
        _default_mmap = platform.system() != "Windows"
        _use_mmap = os.environ.get("IRON_AGING_RESIDUE_MMAP", str(_default_mmap)).lower() in ("1", "true", "yes")
        try:
            if _use_mmap:
                logger.info("  使用 mmap 模式加载残基特征（按需分页）")
                data = torch.load(
                    residue_pt_path, map_location="cpu",
                    mmap=True, weights_only=False)
            else:
                logger.info("  使用普通模式加载残基特征（Windows 默认禁用 mmap 以避免 ACCESS_VIOLATION）")
                data = torch.load(
                    residue_pt_path, map_location="cpu",
                    mmap=False, weights_only=False)
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


def main(decoder_type: str | None = None, skip_sage: bool = False, skip_hgt: bool = False,
         skip_simplehgn: bool = False,
         global_overrides: dict | None = None, reevaluate: bool = False):
    """v27: Phase 4 主流程 — SAGE + HGT + SimpleHGN 三分支训练与 TCM 预测

    流程:
      1. 加载 CPI/PPI/KEGG/蛋白特征/TCM 池数据
      2. 构建同质图 + 异质图（可选疾病节点）
      3. 双重冷启动拆分（化合物 85/15 + 蛋白 80/20 分层）
      4. 训练 SAGE 分支（SAGEConv + 两阶段迁移学习）
      5. 训练 HGT 分支（HGTConv + 两阶段迁移学习）
      6. 训练 SimpleHGN 分支（HeteroConv + GATv2Conv + 边类型嵌入）
      7. 动态集成权重预测 TCM 化合物-靶标得分
      8. 输出预测结果和性能指标

    Args:
        decoder_type: CLI 传入的解码器类型，覆盖配置中的 DECODER_TYPE。
    """
    global DECODER_TYPE, EPOCHS, PRETRAIN_EPOCHS, RANDOM_SEED
    if reevaluate and (not skip_sage or not skip_hgt or not skip_simplehgn):
        logger.warning("v60: reevaluate 模式要求同时跳过所有训练，已自动设置 skip_sage=skip_hgt=skip_simplehgn=True")
        skip_sage = True
        skip_hgt = True
        skip_simplehgn = True
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

    # 动态日志文件名
    log_parts = ["phase4_v70"]
    if skip_sage and skip_hgt and skip_simplehgn:
        log_parts.append("reevaluate" if reevaluate else "all_skip")
    elif skip_sage and not skip_hgt and not skip_simplehgn:
        log_parts.append("hgt_simplehgn_only")
    elif skip_hgt and not skip_sage and not skip_simplehgn:
        log_parts.append("sage_simplehgn_only")
    elif skip_simplehgn and not skip_sage and not skip_hgt:
        log_parts.append("sage_hgt_only")
    elif skip_sage and skip_hgt:
        log_parts.append("simplehgn_only")
    elif skip_sage and skip_simplehgn:
        log_parts.append("hgt_only")
    elif skip_hgt and skip_simplehgn:
        log_parts.append("sage_only")
    else:
        log_parts.append("full")
    log_parts.append(f"decoder_{DECODER_TYPE}")
    log_parts.append(f"seed_{RANDOM_SEED}")
    log_file = L4_LOGS / f"{'_'.join(log_parts)}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logger.info(f"日志已重定向到: {log_file}")

    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Phase 4 v70: SAGE + HGT Mini-Batch — 工业级重构（配置系统/tqdm/GPU监控/类型注解）")
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
    logger.info("v60: SAGE通路可训练嵌入 + BPR独立负采样 + HGT验证缓存清理 + 表型数据隔离")
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

    # v56: TCM 候选池与训练集完全隔离 — 从 CPI 中剔除所有 TCM 化合物
    tcm_smiles_col = "SMILES_std" if "SMILES_std" in tcm_df.columns else (
        "SMILES" if "SMILES" in tcm_df.columns else "canonical_smiles")
    tcm_smiles_set = set(tcm_df[tcm_smiles_col].dropna().astype(str))
    n_cpi_before = len(cpi_df)
    cpi_df = cpi_df[~cpi_df["canonical_smiles"].isin(tcm_smiles_set)].copy()
    n_cpi_removed = n_cpi_before - len(cpi_df)
    if n_cpi_removed > 0:
        logger.warning(f"  v56: 从 CPI 训练集剔除 {n_cpi_removed} 条 TCM 重叠记录（剩余 {len(cpi_df)} 条）")
    else:
        logger.info("  v56: TCM 与 CPI 训练集无重叠，无需剔除")

    # 加载铁死亡表型分类数据集
    pheno_df = None
    pheno_file = L4_RESULTS / "phenotype_ferroptosis_dataset_v25_clean.csv"
    if pheno_file.exists():
        pheno_df = pd.read_csv(pheno_file)
        # v59: 表型任务化合物必须与 CPI 训练集 / TCM 候选池隔离，防止信息泄漏
        pheno_smi_col = None
        for col in ["SMILES", "smiles", "canonical_smiles", "SMILES_std"]:
            if col in pheno_df.columns:
                pheno_smi_col = col
                break
        if pheno_smi_col is not None:
            train_smi_set = set(cpi_df["canonical_smiles"].dropna().astype(str))
            overlap_train = pheno_df[pheno_df[pheno_smi_col].astype(str).isin(train_smi_set)]
            overlap_tcm = pheno_df[pheno_df[pheno_smi_col].astype(str).isin(tcm_smiles_set)]
            n_before = len(pheno_df)
            pheno_df = pheno_df[
                ~pheno_df[pheno_smi_col].astype(str).isin(train_smi_set | tcm_smiles_set)
            ].copy()
            n_removed = n_before - len(pheno_df)
            if n_removed > 0:
                logger.warning(
                    f"  v59: 从表型数据集剔除 {n_removed} 条重叠记录 "
                    f"(CPI训练集={len(overlap_train)}, TCM池={len(overlap_tcm)})，剩余 {len(pheno_df)} 条"
                )
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
    GRAPH_CACHE_PATH = L4_RESULTS / "graph_cache_v67.pkl"
    GRAPH_CACHE_KEY = {
        "version": "v60",
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
        # v67: 化合物特征缓存管理器
        compound_cfg = _cfg.compound_feature if _cfg else CompoundFeatureConfig()
        if compound_cfg.enable_cache:
            compound_feature_cache = FeatureCache(
                PROJECT_ROOT / compound_cfg.cache_dir,
                version=compound_cfg.cache_version,
            )
            logger.info(f"化合物特征缓存已启用: {compound_feature_cache.cache_dir}")
        else:
            compound_feature_cache = None

        # 构建图 & 邻接表
        logger.info(">>> 构建图 & 邻接表")
        _t0 = time.time()
        graphs = build_graphs_and_adj(
            cpi_df, ppi_df, gene_to_pathways, prot_feat, disease_df=disease_df,
            use_topology_neg=USE_TOPOLOGY_NEG,
            topo_neighbors_top_k=TOPO_NEIGHBORS_TOP_K,
            use_esm_similarity_neg=USE_ESM_SIMILARITY_NEG,
            esm_similarity_top_k=ESM_SIMILARITY_TOP_K,
            compound_feature_cache=compound_feature_cache,
            use_meta_paths=META_PATH_ENABLED,
            meta_path_density_threshold=_cfg.meta_path.density_threshold if _cfg else 0.1,
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
        graphs["hetero_adj"], graphs["n_compounds"], val_comp_set, val_proteins)
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

    sage_model_path = L4_RESULTS / "sage_best_v67.pt"
    if skip_sage and sage_model_path.exists():
        logger.info(f">>> 跳过 SAGE 训练，加载已有模型: {sage_model_path}")
        sage_model = SAGELinkPredictor(
            comp_feat_dim=graphs["feat_dim"], prot_feat_dim=graphs["prot_esm_dim"],
            n_compounds=graphs["n_compounds"],
            hidden_dim=SAGE_HIDDEN_DIM, out_dim=SAGE_OUT_DIM, num_layers=SAGE_NUM_LAYERS, dropout=DROPOUT,
            n_pathways=graphs["n_pathways"],
            prot_proj_dropout=PROT_PROJ_DROPOUT,
            prot_proj_inner_dropout=PROT_PROJ_INNER_DROPOUT,
            pathway_proj_dropout=PATHWAY_PROJ_DROPOUT,
            pheno_head_dropout=PHENO_HEAD_DROPOUT,
            temperature=TEMPERATURE,
            decoder_type=DECODER_TYPE)
        sage_checkpoint = torch.load(sage_model_path, map_location=DEVICE, weights_only=False)
        # v59: 过滤掉 decoder 中动态注册的 residue 索引 buffer，避免旧 state_dict 的短 buffer 覆盖当前映射
        sage_state_dict = {k: v for k, v in sage_checkpoint["state_dict"].items() if "_prot_to_residue_idx" not in k}
        sage_model.load_state_dict(sage_state_dict, strict=False)
        sage_model = sage_model.to(DEVICE)
        # v59: 加载权重后再注册残基特征，确保使用当前图的正确蛋白-残基映射
        if DECODER_TYPE == "residue_bilinear" and residue_embeddings is not None:
            sage_model.set_residue_features(
                residue_embeddings, residue_offsets, residue_lengths,
                prot_to_residue_idx=prot_to_residue_idx,
                max_len=residue_max_len,
                residue_device="cpu",
            )
        sage_history = []
        _t0 = time.time()
    else:
        logger.info(f">>> 训练 SAGE（v60: SAGEConv + {DECODER_TYPE} + 两阶段迁移学习 + FocalLoss + 课程负采样）")
        _log_gpu_memory("SAGE 训练前")
        _t0 = time.time()
        sage_model = SAGELinkPredictor(
            comp_feat_dim=graphs["feat_dim"], prot_feat_dim=graphs["prot_esm_dim"],  # 传 ESM-2 维度（640），通路独立投影
            n_compounds=graphs["n_compounds"],
            hidden_dim=SAGE_HIDDEN_DIM, out_dim=SAGE_OUT_DIM, num_layers=SAGE_NUM_LAYERS, dropout=DROPOUT,
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
        use_infonce=INFONCE_WEIGHT > 0, use_bpr=True, use_curriculum=True,
        pheno_compound_indices=pheno_train_indices,
        pheno_labels=pheno_train_labels,
        pheno_lambda=PHENO_LAMBDA,
        bpr_weight=BPR_WEIGHT, weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_RATIO,
        dropedge_ppi=DROPPEDGE_PPI, dropedge_pathway=DROPPEDGE_PATHWAY, dropedge_cpi=DROPPEDGE_CPI,
        focal_gamma=FOCAL_GAMMA, focal_alpha=FOCAL_ALPHA,
        memory_bank_size=MEMORY_BANK_SIZE,
        head_ratio=HEAD_RATIO, lambda_hhi=LAMBDA_HHI, head_undersample_ratio=HEAD_UNDERSAMPLE_RATIO,
        grad_clip_norm=GRAD_CLIP_NORM,
        pretrain_lr_multiplier=PRETRAIN_LR_MULTIPLIER, pretrain_lr_decay=PRETRAIN_LR_DECAY,
        finetune_lr_multiplier=FINETUNE_LR_MULTIPLIER,
        use_plateau_scheduler=USE_PLATEAU_SCHEDULER_SAGE,
        plateau_patience=PLATEAU_PATIENCE_SAGE,
        plateau_factor=PLATEAU_FACTOR_SAGE,
        use_topology_neg=USE_TOPOLOGY_NEG,
        _validate_sage_fn=_validate_sage, _compute_cpi_loss_fn=_compute_cpi_loss,
        _compute_auxiliary_reconstruction_loss_fn=_compute_auxiliary_reconstruction_loss,
        aux_recon_weight=AUX_RECON_WEIGHT,
        aux_recon_ppi_samples=AUX_RECON_PPI_SAMPLES,
        aux_recon_ddi_samples=AUX_RECON_DDI_SAMPLES,
        hetero_adj=graphs.get("hetero_adj"),
        n_diseases=graphs.get("n_diseases", 0),
        use_amp=False, val_freq=VAL_FREQ)

    try:
        L4_RESULTS.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": sage_model.state_dict(), "version": "v60", "hidden_dim": SAGE_HIDDEN_DIM, "out_dim": SAGE_OUT_DIM}, L4_RESULTS / "sage_best_v67.pt")
        logger.info("  SAGE 模型已保存到 sage_best_v67.pt")
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
        logger.info(f">>> 训练 HGT（v60: HGTConv + {DECODER_TYPE} + 两阶段迁移学习 + FocalLoss + 课程负采样）")
        _log_gpu_memory("HGT 训练前")
    hgt_model_path = L4_RESULTS / "hgt_best_v67.pt"
    if skip_hgt and hgt_model_path.exists():
        logger.info(f">>> 跳过 HGT 训练，加载已有模型: {hgt_model_path}")
        hgt_node_feat_dims = {
            "compound": graphs["feat_dim"],
            "protein": graphs["prot_esm_dim"],
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
            decoder_type=DECODER_TYPE,
            use_graph_transformer=GT_ENABLED,
            gt_num_layers=_cfg.graph_transformer.num_layers if _cfg else 2,
            gt_num_heads=_cfg.graph_transformer.num_heads if _cfg else 4,
            gt_dropout=_cfg.graph_transformer.dropout if _cfg else 0.3,
            use_cross_modal_fusion=USE_CROSS_MODAL_FUSION,
            fusion_hidden_dim=FUSION_HIDDEN_DIM)
        hgt_checkpoint = torch.load(hgt_model_path, map_location=DEVICE, weights_only=False)
        # v59: 过滤掉 decoder 中动态注册的 residue 索引 buffer，避免旧 state_dict 的短 buffer 覆盖当前映射
        hgt_state_dict = {k: v for k, v in hgt_checkpoint["state_dict"].items() if "_prot_to_residue_idx" not in k}
        hgt_model.load_state_dict(hgt_state_dict, strict=False)
        hgt_model = hgt_model.to(DEVICE)
        # v59: 加载权重后再注册残基特征，确保使用当前图的正确蛋白-残基映射
        if DECODER_TYPE == "residue_bilinear" and residue_embeddings is not None:
            hgt_model.set_residue_features(
                residue_embeddings, residue_offsets, residue_lengths,
                prot_to_residue_idx=prot_to_residue_idx,
                max_len=residue_max_len,
                residue_device="cpu",
            )
        hgt_history = []
    elif skip_hgt:
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
            decoder_type=DECODER_TYPE,
            use_graph_transformer=GT_ENABLED,
            gt_num_layers=_cfg.graph_transformer.num_layers if _cfg else 2,
            gt_num_heads=_cfg.graph_transformer.num_heads if _cfg else 4,
            gt_dropout=_cfg.graph_transformer.dropout if _cfg else 0.3,
            use_cross_modal_fusion=USE_CROSS_MODAL_FUSION,
            fusion_hidden_dim=FUSION_HIDDEN_DIM)
        # 注册残基级 ESM-2 特征到 HGT 解码器（大张量保留在 CPU）
        if DECODER_TYPE == "residue_bilinear" and residue_embeddings is not None:
            hgt_model.set_residue_features(
                residue_embeddings, residue_offsets, residue_lengths,
                prot_to_residue_idx=prot_to_residue_idx,
                max_len=residue_max_len,
                residue_device="cpu",
            )
        # v69: 注册元路径边索引供 GraphTransformer 多视角编码
        if META_PATH_ENABLED:
            hgt_model.set_meta_path_edge_indices(graphs.get("meta_path_edge_indices"))
        hgt_model, hgt_history = train_hgt(
            hgt_model, graphs, train_compounds, val_compounds, compound_to_pos,
            device=DEVICE,
            val_proteins=val_proteins,
            epochs=EPOCHS_HGT, lr=LEARNING_RATE_HGT, patience=PATIENCE, batch_size=HGT_BATCH_SIZE,
            num_neighbors=HGT_NUM_NEIGHBORS,
            prot_to_path_neighbors=graphs.get("prot_to_path_neighbors"),
            two_stage=True, pretrain_epochs=PRETRAIN_EPOCHS_HGT, pretrain_lr=PRETRAIN_LR_HGT,
            random_seed=RANDOM_SEED,
            use_infonce=INFONCE_WEIGHT > 0, use_bpr=True, use_curriculum=True,
            pheno_compound_indices=pheno_train_indices,
            pheno_labels=pheno_train_labels,
            pheno_lambda=PHENO_LAMBDA,
            bpr_weight=BPR_WEIGHT, weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_RATIO,
            dropedge_ppi=DROPPEDGE_PPI, dropedge_pathway=DROPPEDGE_PATHWAY, dropedge_cpi=DROPPEDGE_CPI,
            focal_gamma=FOCAL_GAMMA, focal_alpha=FOCAL_ALPHA,
            semantic_attn_weight=SEMANTIC_ATTN_WEIGHT,
            memory_bank_size=MEMORY_BANK_SIZE,
            head_ratio=HEAD_RATIO, lambda_hhi=LAMBDA_HHI, head_undersample_ratio=HEAD_UNDERSAMPLE_RATIO,
            grad_clip_norm=GRAD_CLIP_NORM,
            pretrain_lr_multiplier=PRETRAIN_LR_MULTIPLIER, pretrain_lr_decay=PRETRAIN_LR_DECAY,
            finetune_lr_multiplier=FINETUNE_LR_MULTIPLIER_HGT,
            use_plateau_scheduler=USE_PLATEAU_SCHEDULER_HGT,
            plateau_patience=PLATEAU_PATIENCE_HGT,
            plateau_factor=PLATEAU_FACTOR_HGT,
            use_topology_neg=USE_TOPOLOGY_NEG,
            _validate_hgt_fn=_validate_hgt, _compute_cpi_loss_fn=_compute_cpi_loss,
            _compute_auxiliary_reconstruction_loss_fn=_compute_auxiliary_reconstruction_loss,
            aux_recon_weight=AUX_RECON_WEIGHT,
            aux_recon_ppi_samples=AUX_RECON_PPI_SAMPLES,
            aux_recon_ddi_samples=AUX_RECON_DDI_SAMPLES,
            aux_recon_prot_disease_samples=AUX_RECON_PROT_DISEASE_SAMPLES,
            hetero_adj=graphs.get("hetero_adj"),
            n_diseases=graphs.get("n_diseases", 0),
            use_amp=False, val_freq=VAL_FREQ)

        try:
            L4_RESULTS.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": hgt_model.state_dict(), "version": "v60", "hidden_dim": HIDDEN_DIM, "out_dim": OUT_DIM}, L4_RESULTS / "hgt_best_v67.pt")
            logger.info("  HGT 模型已保存到 hgt_best_v67.pt")
        except Exception:
            logger.error("  HGT 模型保存失败", exc_info=True)
            raise

        # train_hgt 的早停加载最佳参数时会把参数 clone 到 CPU，预测前必须移回 DEVICE。
        hgt_model = hgt_model.to(DEVICE)

        _t0 = _log_step_time(_t0, "HGT 训练完成")
        torch.cuda.empty_cache()
        _log_gpu_memory("HGT 训练后 (cache cleared)")
        logger.info("  HGT GPU 内存已释放")

    # ===== SimpleHGN 分支 =====
    if skip_simplehgn:
        logger.info(">>> 跳过 SimpleHGN 训练")
        simplehgn_history = []
    else:
        logger.info(f">>> 训练 SimpleHGN（HeteroConv + GATv2Conv + {DECODER_TYPE} + 两阶段迁移学习 + FocalLoss + 课程负采样）")
        _log_gpu_memory("SimpleHGN 训练前")
    simplehgn_model_path = L4_RESULTS / "simplehgn_best_v67.pt"
    if skip_simplehgn and simplehgn_model_path.exists():
        logger.info(f">>> 跳过 SimpleHGN 训练，加载已有模型: {simplehgn_model_path}")
        simplehgn_node_feat_dims = {
            "compound": graphs["feat_dim"],
            "protein": graphs["prot_esm_dim"],
            "pathway": 1,
            "pathway_count": graphs["n_pathways"],
            "disease_count": graphs.get("n_diseases", 0),
        }
        simplehgn_model = SimpleHGNLinkPredictor(
            hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
            dropout=DROPOUT, metadata=graphs["hetero_data"].metadata(),
            compound_feat_dim=graphs["feat_dim"], node_feat_dims=simplehgn_node_feat_dims,
            pheno_head_dropout=PHENO_HEAD_DROPOUT,
            temperature=TEMPERATURE,
            decoder_type=DECODER_TYPE,
            use_graph_transformer=GT_ENABLED,
            gt_num_layers=_cfg.graph_transformer.num_layers if _cfg else 2,
            gt_num_heads=_cfg.graph_transformer.num_heads if _cfg else 4,
            gt_dropout=_cfg.graph_transformer.dropout if _cfg else 0.3,
            use_cross_modal_fusion=USE_CROSS_MODAL_FUSION,
            fusion_hidden_dim=FUSION_HIDDEN_DIM)
        simplehgn_checkpoint = torch.load(simplehgn_model_path, map_location=DEVICE, weights_only=False)
        simplehgn_state_dict = {k: v for k, v in simplehgn_checkpoint["state_dict"].items() if "_prot_to_residue_idx" not in k}
        simplehgn_model.load_state_dict(simplehgn_state_dict, strict=False)
        simplehgn_model = simplehgn_model.to(DEVICE)
        if DECODER_TYPE == "residue_bilinear" and residue_embeddings is not None:
            simplehgn_model.set_residue_features(
                residue_embeddings, residue_offsets, residue_lengths,
                prot_to_residue_idx=prot_to_residue_idx,
                max_len=residue_max_len,
                residue_device="cpu",
            )
        simplehgn_history = []
    elif skip_simplehgn:
        simplehgn_model = None
        simplehgn_history = []
    else:
        simplehgn_node_feat_dims = {
            "compound": graphs["feat_dim"],
            "protein": graphs["prot_esm_dim"],
            "pathway": 1,
            "pathway_count": graphs["n_pathways"],
            "disease_count": graphs.get("n_diseases", 0),
        }
        simplehgn_model = SimpleHGNLinkPredictor(
            hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
            dropout=DROPOUT, metadata=graphs["hetero_data"].metadata(),
            compound_feat_dim=graphs["feat_dim"], node_feat_dims=simplehgn_node_feat_dims,
            pheno_head_dropout=PHENO_HEAD_DROPOUT,
            temperature=TEMPERATURE,
            decoder_type=DECODER_TYPE,
            use_graph_transformer=GT_ENABLED,
            gt_num_layers=_cfg.graph_transformer.num_layers if _cfg else 2,
            gt_num_heads=_cfg.graph_transformer.num_heads if _cfg else 4,
            gt_dropout=_cfg.graph_transformer.dropout if _cfg else 0.3,
            use_cross_modal_fusion=USE_CROSS_MODAL_FUSION,
            fusion_hidden_dim=FUSION_HIDDEN_DIM)
        if DECODER_TYPE == "residue_bilinear" and residue_embeddings is not None:
            simplehgn_model.set_residue_features(
                residue_embeddings, residue_offsets, residue_lengths,
                prot_to_residue_idx=prot_to_residue_idx,
                max_len=residue_max_len,
                residue_device="cpu",
            )
        # v69: 注册元路径边索引供 GraphTransformer 多视角编码
        if META_PATH_ENABLED:
            simplehgn_model.set_meta_path_edge_indices(graphs.get("meta_path_edge_indices"))
        simplehgn_model, simplehgn_history = train_simplehgn(
            simplehgn_model, graphs, train_compounds, val_compounds, compound_to_pos,
            device=DEVICE,
            val_proteins=val_proteins,
            epochs=EPOCHS_SIMPLEHGN, lr=LEARNING_RATE_SIMPLEHGN, patience=PATIENCE,
            batch_size=SIMPLEHGN_BATCH_SIZE,
            num_neighbors=SIMPLEHGN_NUM_NEIGHBORS,
            prot_to_path_neighbors=graphs.get("prot_to_path_neighbors"),
            two_stage=True, pretrain_epochs=PRETRAIN_EPOCHS_SIMPLEHGN,
            pretrain_lr=PRETRAIN_LR_SIMPLEHGN,
            random_seed=RANDOM_SEED,
            use_infonce=INFONCE_WEIGHT > 0, use_bpr=True, use_curriculum=True,
            pheno_compound_indices=pheno_train_indices,
            pheno_labels=pheno_train_labels,
            pheno_lambda=PHENO_LAMBDA,
            bpr_weight=BPR_WEIGHT, weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_RATIO,
            dropedge_ppi=DROPPEDGE_PPI, dropedge_pathway=DROPPEDGE_PATHWAY, dropedge_cpi=DROPPEDGE_CPI,
            focal_gamma=FOCAL_GAMMA, focal_alpha=FOCAL_ALPHA,
            semantic_attn_weight=SEMANTIC_ATTN_WEIGHT,
            memory_bank_size=MEMORY_BANK_SIZE,
            head_ratio=HEAD_RATIO, lambda_hhi=LAMBDA_HHI, head_undersample_ratio=HEAD_UNDERSAMPLE_RATIO,
            grad_clip_norm=GRAD_CLIP_NORM,
            pretrain_lr_multiplier=PRETRAIN_LR_MULTIPLIER, pretrain_lr_decay=PRETRAIN_LR_DECAY,
            finetune_lr_multiplier=FINETUNE_LR_MULTIPLIER_HGT,
            use_plateau_scheduler=USE_PLATEAU_SCHEDULER_HGT,
            plateau_patience=PLATEAU_PATIENCE_HGT,
            plateau_factor=PLATEAU_FACTOR_HGT,
            use_topology_neg=USE_TOPOLOGY_NEG,
            _validate_simplehgn_fn=_validate_simplehgn,
            _compute_cpi_loss_fn=_compute_cpi_loss,
            _compute_auxiliary_reconstruction_loss_fn=_compute_auxiliary_reconstruction_loss,
            aux_recon_weight=AUX_RECON_WEIGHT,
            aux_recon_ppi_samples=AUX_RECON_PPI_SAMPLES,
            aux_recon_ddi_samples=AUX_RECON_DDI_SAMPLES,
            aux_recon_prot_disease_samples=AUX_RECON_PROT_DISEASE_SAMPLES,
            hetero_adj=graphs.get("hetero_adj"),
            n_diseases=graphs.get("n_diseases", 0),
            use_amp=False, val_freq=VAL_FREQ)

        try:
            L4_RESULTS.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": simplehgn_model.state_dict(), "version": "v60", "hidden_dim": HIDDEN_DIM, "out_dim": OUT_DIM}, L4_RESULTS / "simplehgn_best_v67.pt")
            logger.info("  SimpleHGN 模型已保存到 simplehgn_best_v67.pt")
        except Exception:
            logger.error("  SimpleHGN 模型保存失败", exc_info=True)
            raise

        simplehgn_model = simplehgn_model.to(DEVICE)

        _t0 = _log_step_time(_t0, "SimpleHGN 训练完成")
        torch.cuda.empty_cache()
        _log_gpu_memory("SimpleHGN 训练后 (cache cleared)")
        logger.info("  SimpleHGN GPU 内存已释放")

    # v59: reevaluate 模式 — 使用已保存模型重新计算验证指标，避免依赖历史记录或硬编码值
    if reevaluate:
        logger.info(">>> v59: reevaluate 模式 — 重新计算 SAGE/HGT/SimpleHGN 验证指标")
        # SAGE 残基特征可能已在训练后释放，验证前重新注册
        if DECODER_TYPE == "residue_bilinear" and residue_embeddings is not None and sage_model is not None:
            sage_model.set_residue_features(
                residue_embeddings, residue_offsets, residue_lengths,
                prot_to_residue_idx=prot_to_residue_idx,
                max_len=residue_max_len,
                residue_device="cpu",
            )
            logger.info("  v59: SAGE 残基特征已重新注册，供 reevaluate 验证使用")
        if DECODER_TYPE == "residue_bilinear" and residue_embeddings is not None and hgt_model is not None:
            hgt_model.set_residue_features(
                residue_embeddings, residue_offsets, residue_lengths,
                prot_to_residue_idx=prot_to_residue_idx,
                max_len=residue_max_len,
                residue_device="cpu",
            )
            logger.info("  v59: HGT 残基特征已重新注册，供 reevaluate 验证使用")
        if DECODER_TYPE == "residue_bilinear" and residue_embeddings is not None and simplehgn_model is not None:
            simplehgn_model.set_residue_features(
                residue_embeddings, residue_offsets, residue_lengths,
                prot_to_residue_idx=prot_to_residue_idx,
                max_len=residue_max_len,
                residue_device="cpu",
            )
            logger.info("  v60: SimpleHGN 残基特征已重新注册，供 reevaluate 验证使用")
        sage_metrics = _validate_sage(
            sage_model, graphs["x"], graphs["homo_edge_index"],
            val_compounds, compound_to_pos, graphs["n_compounds"],
        )
        sage_history = [{"auc": sage_metrics.get("auc", 0.5), "aupr": sage_metrics.get("aupr", 0.5)}]
        logger.info(f"  SAGE 重新验证: auc={sage_metrics.get('auc', 0.5):.4f}, aupr={sage_metrics.get('aupr', 0.5):.4f}")

        hgt_metrics = _validate_hgt(
            hgt_model, graphs["hetero_data_val"], val_compounds, compound_to_pos,
            graphs["n_compounds"], graphs["n_proteins"],
        )
        hgt_history = [{"auc": hgt_metrics.get("auc", 0.5), "aupr": hgt_metrics.get("aupr", 0.5)}]
        logger.info(f"  HGT 重新验证: auc={hgt_metrics.get('auc', 0.5):.4f}, aupr={hgt_metrics.get('aupr', 0.5):.4f}")

        simplehgn_metrics = _validate_hgt(
            simplehgn_model, graphs["hetero_data_val"], val_compounds, compound_to_pos,
            graphs["n_compounds"], graphs["n_proteins"],
        )
        simplehgn_history = [{"auc": simplehgn_metrics.get("auc", 0.5), "aupr": simplehgn_metrics.get("aupr", 0.5)}]
        logger.info(f"  SimpleHGN 重新验证: auc={simplehgn_metrics.get('auc', 0.5):.4f}, aupr={simplehgn_metrics.get('aupr', 0.5):.4f}")

    # v56: 动态集成权重 — 基于验证 AUPR 加权
    # 训练时从训练历史提取；skip 时从已知最佳结果读取
    if skip_sage and skip_hgt and not reevaluate:
        sage_aupr = 0.7870  # SAGE v55 best_val_aupr（硬编码回退）
        hgt_aupr = 0.1251   # HGT v53 best_val_aupr（硬编码回退）
        simplehgn_aupr = 0.1251  # SimpleHGN 默认回退（与 HGT 同架构级别）
        logger.info(f"  v56: 动态集成权重（skip模式）— SAGE AUPR={sage_aupr:.4f}, HGT AUPR={hgt_aupr:.4f}, SimpleHGN AUPR={simplehgn_aupr:.4f}")
    else:
        sage_aupr = max((h.get("aupr", 0.5) for h in sage_history), default=0.5) if sage_history else 0.5
        hgt_aupr = max((h.get("aupr", 0.5) for h in hgt_history), default=0.5) if hgt_history else 0.5
        simplehgn_aupr = max((h.get("aupr", 0.5) for h in simplehgn_history), default=0.5) if simplehgn_history else 0.5
        logger.info(f"  v56: 动态集成权重（训练模式）— SAGE AUPR={sage_aupr:.4f}, HGT AUPR={hgt_aupr:.4f}, SimpleHGN AUPR={simplehgn_aupr:.4f}")
    total_aupr = sage_aupr + hgt_aupr + simplehgn_aupr
    sage_w = sage_aupr / total_aupr if total_aupr > 0 else 0.4
    hgt_w = hgt_aupr / total_aupr if total_aupr > 0 else 0.3
    simplehgn_w = simplehgn_aupr / total_aupr if total_aupr > 0 else 0.3
    logger.info(f"  v56: 集成权重 SAGE={sage_w:.4f}, HGT={hgt_w:.4f}, SimpleHGN={simplehgn_w:.4f}")

    # 加载树模型 v7 TCM 预测用于集成
    tree_pred_df = None
    tree_pred_path = L4_ROOT / "results" / "tree_v6_tcm_predictions_v7.csv"
    if tree_pred_path.exists():
        tree_pred_df = pd.read_csv(tree_pred_path, engine='python')
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
        if simplehgn_model is not None:
            simplehgn_model.set_residue_features(
                residue_embeddings, residue_offsets, residue_lengths,
                prot_to_residue_idx=prot_to_residue_idx,
                max_len=residue_max_len,
                residue_device="cpu",
            )
            logger.info("  v45: SimpleHGN 残基特征已重新注册，供预测使用")

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
        DEVICE,
        mc_samples=MC_SAMPLES,
        tcm_feat_precomputed=tcm_feat_precomputed,
        tree_predictions=tree_pred_df,  # 树模型预测集成
        tree_weight=TREE_ENSEMBLE_WEIGHT,  # 树模型权重
        sage_w=sage_w, hgt_w=hgt_w,  # v56: 动态集成权重
        simplehgn_model=simplehgn_model, simplehgn_w=simplehgn_w)  # SimpleHGN 分支

    # 铁死亡概率预测 — SAGE + HGT + SimpleHGN 三分支加权平均
    final_ferroptosis_prob = None
    if pheno_train_indices is not None and len(pheno_train_indices) > 0 and hgt_model is not None:
        logger.info(">>> 预测 TCM 化合物铁死亡概率（表型分类头）")
        sage_model.eval()
        hgt_model.eval()
        if simplehgn_model is not None:
            simplehgn_model.eval()
        with torch.no_grad():
            tcm_feat_tensor = tcm_feat_precomputed.to(DEVICE)

            sage_tcm_emb = sage_model.encode_compound(tcm_feat_tensor)
            sage_ferro_logits = sage_model.predict_phenotype(sage_tcm_emb)
            sage_ferro_prob = torch.sigmoid(sage_ferro_logits).squeeze(-1).cpu().numpy()

            hgt_tcm_emb = hgt_model.encode_compound(tcm_feat_tensor)
            hgt_ferro_logits = hgt_model.predict_phenotype(hgt_tcm_emb)
            hgt_ferro_prob = torch.sigmoid(hgt_ferro_logits).squeeze(-1).cpu().numpy()

            if simplehgn_model is not None:
                simplehgn_tcm_emb = simplehgn_model.encode_compound(tcm_feat_tensor)
                simplehgn_ferro_logits = simplehgn_model.predict_phenotype(simplehgn_tcm_emb)
                simplehgn_ferro_prob = torch.sigmoid(simplehgn_ferro_logits).squeeze(-1).cpu().numpy()
                # 三分支加权平均
                final_ferroptosis_prob = (sage_ferro_prob * sage_w
                                          + hgt_ferro_prob * hgt_w
                                          + simplehgn_ferro_prob * simplehgn_w) / (sage_w + hgt_w + simplehgn_w)
            else:
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

    # v56: TCM 训练集泄漏惩罚 — 重叠化合物降低 20% 排名分数
    if "in_train" in pred_df.columns:
        in_train_mask = pred_df["in_train"].values
        n_in_train_penalty = in_train_mask.sum()
        if n_in_train_penalty > 0:
            # 对 in_train 化合物应用 0.8 倍惩罚，避免训练集过拟合导致的虚高排名
            composite[in_train_mask] = composite[in_train_mask] * 0.8
            logger.warning(f"  v56: {n_in_train_penalty} 个 in_train 化合物 composite_score 已降权 20%")

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
        pred_df.to_csv(L4_RESULTS / "tcm_predictions_full_v67.csv", index=False)
        top_df.to_csv(L4_RESULTS / "tcm_top_candidates_v67.csv", index=False)
        logger.info(f"  预测结果已保存: tcm_predictions_full_v67.csv ({len(pred_df)} 行), "
                    f"tcm_top_candidates_v67.csv ({len(top_df)} 行)")
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
    if simplehgn_history:
        simplehgn_best_auc = max(h["auc"] for h in simplehgn_history)
        simplehgn_best_aupr = max(h.get("aupr", 0) for h in simplehgn_history)
        simplehgn_row = {"model": "SimpleHGN", "best_auc": simplehgn_best_auc, "best_aupr": simplehgn_best_aupr}
        simplehgn_row["train_time_min"] = round(train_time_min, 1)
        simplehgn_row["gpu_mem_peak_gb"] = round(gpu_mem_peak_gb, 2)
        perf_rows.append(simplehgn_row)
    if perf_rows:
        try:
            pd.DataFrame(perf_rows).to_csv(L4_RESULTS / "model_performance_v67.csv", index=False)
            logger.info(f"  模型性能报告已保存: model_performance_v67.csv (训练时间={train_time_min:.1f}min, GPU峰值={gpu_mem_peak_gb:.2f}GB)")
        except Exception:
            logger.error("  模型性能 CSV 保存失败", exc_info=True)
            raise

    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Phase 4 v70 完成！总耗时 {total_time / 60:.1f} 分钟")
    if sage_history:
        logger.info(f"  SAGE best val_auc: {max(h['auc'] for h in sage_history):.4f}  val_aupr: {max(h.get('aupr', 0) for h in sage_history):.4f}")
    if hgt_history:
        logger.info(f"  HGT best val_auc: {max(h['auc'] for h in hgt_history):.4f}  val_aupr: {max(h.get('aupr', 0) for h in hgt_history):.4f}")
    if simplehgn_history:
        logger.info(f"  SimpleHGN best val_auc: {max(h['auc'] for h in simplehgn_history):.4f}  val_aupr: {max(h.get('aupr', 0) for h in simplehgn_history):.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 v70: SAGE + HGT + 树模型集成训练与 TCM 预测")
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
        "--skip_simplehgn",
        action="store_true",
        help="跳过 SimpleHGN 训练",
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
        "--simplehgn_epochs",
        type=int,
        default=None,
        help="覆盖 SimpleHGN 训练 epoch 数（快速测试用）",
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
    parser.add_argument(
        "--reevaluate",
        action="store_true",
        help="v59: 跳过训练，加载已保存模型并重新计算验证指标（用于诊断指标可靠性）",
    )
    args = parser.parse_args()
    # 应用 CLI 覆盖到全局常量
    global_overrides = {}
    if args.sage_epochs is not None:
        global_overrides["EPOCHS"] = args.sage_epochs
    if args.hgt_epochs is not None:
        global_overrides["EPOCHS_HGT"] = args.hgt_epochs
    if args.simplehgn_epochs is not None:
        global_overrides["EPOCHS_SIMPLEHGN"] = args.simplehgn_epochs
    if args.pretrain_epochs is not None:
        global_overrides["PRETRAIN_EPOCHS"] = args.pretrain_epochs
    if args.seed is not None:
        global_overrides["RANDOM_SEED"] = args.seed
    main(decoder_type=args.decoder_type, skip_sage=args.skip_sage, skip_hgt=args.skip_hgt,
         skip_simplehgn=args.skip_simplehgn,
         global_overrides=global_overrides, reevaluate=args.reevaluate)
