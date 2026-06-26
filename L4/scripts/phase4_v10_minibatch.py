#!/usr/bin/env python3
"""
Phase 4 v18: Mini-Batch GNN 双分支 — 冷启动验证与训练稳定性修复
==========================================================================
v18 关键修复（基于 v17 评估与代码审查）:

  v18-F1 (必须): 分离常规验证与蛋白冷启动验证图
    - 常规化合物冷启动验证使用保留训练拓扑的全图
    - 蛋白冷启动验证使用严格隔离图（移除验证化合物/蛋白的所有 CPI/PPI/通路边）
    - 修复共用隔离图导致常规验证指标失真的问题

  v18-F2 (必须): 修复最终蛋白冷启动评估的信息泄露
    - main() 最终重新评估改用 homo_edge_index_prot_cold / hetero_data_prot_cold
    - 移除 _validate_hgt_protein_cold 内部错误置空所有 CPI 边的逻辑
    - 新增 _validate_hgt_protein_cold_minibatch 支持 OOM 降级

  v18-F3 (必须): 训练随机负样本排除正样本
    - 通过 mask 构建合法候选集，torch.multinomial 采样
    - 消除 50% 随机负样本中的噪声标签

  v18-F4 (必须): BPR 独立负采样
    - 每个正样本对独立采样负样本，避免多个正样本共享化合物级硬负样本
    - 排序损失信号更准确

  v18-F5 (高优): Focal Loss α 固定为 0.75
    - 移除动态 α = 1 - pos_frac，避免负样本权重过小

  v18-F6 (高优): 冗余代码清理
    - 移除未使用的 _compute_metrics、pathway_to_idx 等变量
    - 清理 pipeline_self_check 中未使用的局部变量
    - 合并重复的检查逻辑

  v18-F7 (中优): 验证边界检查增强
    - 课程负采样 multinomial 全零概率行保护
    - 蛋白冷启动负样本限制在 val_proteins 集合内
    - 通路嵌入 torch.clamp 防止 -1 索引越界

保留自 v17 的核心设计:
  - ESM-2 预训练蛋白嵌入 (facebook/esm2_t30_150M_UR50D, 640维)
  - SAGE + HGT 双分支：拓扑 (SAGEConv) + 语义 (HGTConv)
  - DropEdge / Focal Loss / 标签平滑 / AdamW / 课程负采样 / Memory Bank
  - 蛋白冷启动拆分 + MC Dropout 不确定性估计 + 动态集成权重
  - BCE + BPR 排序损失联合优化（6:4 加权）

关键参考:
  - GraphSAGE: Hamilton et al. (2017) NeurIPS.
  - HGT: Hu et al. (2020) WWW.
  - Focal Loss: Lin et al. (2017) ICCV.
  - ESM-2: Rives et al. (2021) PNAS.
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
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
from torch_geometric.nn import SAGEConv, HGTConv

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="rdkit")
warnings.filterwarnings("ignore", category=FutureWarning, module="rdkit")

PROJECT_ROOT = Path(__file__).parent.parent.parent
L1_RESULTS = PROJECT_ROOT / "L1" / "results"
L2_RESULTS = PROJECT_ROOT / "L2" / "results"
L3_RESULTS = PROJECT_ROOT / "L3" / "results"
L4_ROOT = PROJECT_ROOT / "L4"
L4_RESULTS = L4_ROOT / "results_v10_minibatch"
L4_LOGS = L4_ROOT / "logs"

for d in [L4_RESULTS, L4_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = L4_LOGS / "phase4_v10_minibatch.log"

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"设备: {DEVICE}")

# ============================================================
# 0. 铁衰老靶标列表
# ============================================================
FERRORAGING_GENES_CSV = L1_RESULTS / "ferroaging_genes_96.csv"
if FERRORAGING_GENES_CSV.exists():
    _df = pd.read_csv(FERRORAGING_GENES_CSV)
    ALL_FERRORAGING_GENES = sorted(_df["gene_symbol"].dropna().unique().tolist())
else:
    ALL_FERRORAGING_GENES = sorted([
        "ABCC1", "ACVR1B", "ACSL4", "ALOX15", "ATF3", "ATG3", "BAP1", "BCL6",
        "BRD7", "CD74", "CISD1", "CTSB", "CXCL10", "CYBB", "DYRK1A", "EGR1",
        "EMP1", "EPHA4", "FBXO31", "FTH1", "FTL", "GMFB", "GPX4", "HBP1",
        "HMOX1", "IGFBP7", "IL1B", "IRF1", "KDM6B", "KLF6", "LACTB", "LCN2",
        "LGMN", "LPCAT3", "MAP1LC3B", "MAPK1", "MTOR", "NFE2L2", "NOX4",
        "PDE4B", "PTGS2", "RELA", "RUNX3", "SAT1", "SLC3A2", "SLC7A11",
        "SOD1", "SP1", "SQSTM1", "STAT3", "TFRC", "TLR4", "TP53", "VDAC2",
        "VDAC3", "ACSL3", "ALOX5", "ATG7", "BECN1", "HIF1A", "KEAP1", "NFKB1",
    ])
logger.info(f"铁衰老靶标: {len(ALL_FERRORAGING_GENES)} 个基因")


# ============================================================
# 1. 化合物特征工程（复用 v9）
# ============================================================
RDKIT_DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "MolMR", "TPSA",
    "NumHAcceptors", "NumHDonors", "NumRotatableBonds",
    "HeavyAtomCount", "NumAromaticRings", "NumAliphaticRings",
    "NumHeteroatoms", "NumValenceElectrons", "NHOHCount", "NOCount",
    "RingCount", "FractionCSP3", "BalabanJ",
]
ECFP4_NBITS = 2048


def _compute_ecfp4(smiles_iter: List[str]) -> np.ndarray:
    fps = np.zeros((len(smiles_iter), ECFP4_NBITS), dtype=np.float32)
    for i, smi in enumerate(smiles_iter):
        mol = None
        try:
            if pd.notna(smi):
                mol = Chem.MolFromSmiles(str(smi))
        except Exception:
            mol = None
        if mol is None:
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=ECFP4_NBITS)
            for bit in fp.GetOnBits():
                fps[i, bit] = 1.0
        except Exception:
            pass
    return fps


def _compute_maccs(smiles_iter: List[str]) -> np.ndarray:
    fps = []
    for smi in smiles_iter:
        mol = None
        try:
            if pd.notna(smi):
                mol = Chem.MolFromSmiles(str(smi))
        except Exception:
            mol = None
        if mol is None:
            fps.append(np.zeros(167, dtype=np.float32))
            continue
        try:
            fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            arr = np.zeros(167, dtype=np.float32)
            arr[list(fp.GetOnBits())] = 1.0
            fps.append(arr)
        except Exception:
            fps.append(np.zeros(167, dtype=np.float32))
    return np.array(fps, dtype=np.float32)


def _compute_rdkit_descriptors(smiles_iter: List[str]) -> np.ndarray:
    desc_funcs = {name: getattr(Descriptors, name) for name in RDKIT_DESCRIPTOR_NAMES}
    rows = []
    for smi in smiles_iter:
        mol = None
        try:
            if pd.notna(smi):
                mol = Chem.MolFromSmiles(str(smi))
        except Exception:
            mol = None
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


def build_compound_features(
    smiles_list: List[str],
    stats: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        std = desc.std(axis=0) + 1e-8
        desc = (desc - mean) / std
    else:
        mean, std, col_mean = stats
        inds = np.where(np.isnan(desc))
        desc[inds] = np.take(col_mean, inds[1])
        desc = np.nan_to_num(desc, nan=0.0, posinf=1e6, neginf=-1e6)
        desc = (desc - mean) / (std + 1e-8)

    features = np.hstack([ecfp4, maccs, desc]).astype(np.float32)
    return features, mean, std, col_mean


# ============================================================
# 2. 蛋白特征（复用 v9）
# ============================================================
def compute_aac(sequences: List[str]) -> np.ndarray:
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
    gene_to_seq: Dict[str, str],
    cache_path: Optional[Path] = None,
    model_name: str = "facebook/esm2_t30_150M_UR50D",
    batch_size: int = 4,
) -> Dict[str, np.ndarray]:
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

    # v17: 使用 HuggingFace 镜像解决国内网络不可达问题
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    from transformers import EsmModel, EsmTokenizer

    logger.info(f"  加载 ESM-2 模型: {model_name} (via HF_ENDPOINT={os.environ['HF_ENDPOINT']}) ...")
    tokenizer = EsmTokenizer.from_pretrained(model_name, local_files_only=True)
    model = EsmModel.from_pretrained(model_name, local_files_only=True).to(DEVICE)
    model.eval()
    esm_dim = model.config.hidden_size
    logger.info(f"  ESM-2 嵌入维度: {esm_dim}")

    genes = sorted(gene_to_seq.keys(), key=lambda g: len(gene_to_seq.get(g, "")), reverse=True)
    embeddings: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        for i in range(0, len(genes), batch_size):
            batch_genes = genes[i:i + batch_size]
            batch_seqs = [gene_to_seq[g] for g in batch_genes]

            # 截断过长序列（ESM-2 最大 1024 tokens，含特殊 token 则为 1022 aa）
            max_len = 1022
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


# ============================================================
# 3. 数据加载（复用 v9）
# ============================================================
def load_cpi_data() -> pd.DataFrame:
    cpi_path = L4_ROOT / "results" / "experimental_actives_detail_cleaned.csv"
    if not cpi_path.exists():
        logger.error(f"CPI 数据文件不存在: {cpi_path}")
        sys.exit(1)
    df = pd.read_csv(cpi_path, low_memory=False)
    required = ["gene", "canonical_smiles", "uniprot_id"]
    for col in required:
        if col not in df.columns:
            logger.error(f"CPI 数据缺少列: {col}")
            sys.exit(1)
    df = df[df["canonical_smiles"].notna()].copy()
    df = df[df["canonical_smiles"].astype(str).str.strip() != ""].copy()
    logger.info(f"CPI 数据: {len(df)} 条记录, {df['gene'].nunique()} 个基因, "
                f"{df['canonical_smiles'].nunique()} 个唯一 SMILES")
    return df


def load_ppi_network() -> pd.DataFrame:
    significant_path = L1_RESULTS / "ppi_network_extended_significant_edges.csv"
    extended_path = L1_RESULTS / "ppi_network_extended_edges.csv"

    ppi_path = None
    if significant_path.exists():
        ppi_path = significant_path
    elif extended_path.exists():
        ppi_path = extended_path

    if ppi_path is not None:
        df = pd.read_csv(ppi_path, low_memory=False)
        df = df.rename(columns={"gene_a": "source", "gene_b": "target", "combined_score": "weight"})
        if df["weight"].max() > 1.0:
            df["weight"] = df["weight"] / 1000.0
        df["source"] = df["source"].astype(str).str.upper()
        df["target"] = df["target"].astype(str).str.upper()
        network_type = "DEG 显著子网" if ppi_path == significant_path else "扩展"
        logger.info(f"PPI 网络（{network_type}）: {len(df)} 条边, "
                    f"{pd.concat([df['source'], df['target']]).nunique()} 个节点")
        return df

    logger.error("PPI 网络文件不存在")
    sys.exit(1)


def load_kegg_pathways() -> Dict[str, List[str]]:
    kegg_path = L2_RESULTS / "kegg_pathways" / "kegg_human_pathway_genes.tsv"
    gene_to_pathways: Dict[str, List[str]] = {}

    if kegg_path.exists():
        df = pd.read_csv(kegg_path, sep="\t", low_memory=False)
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


def load_protein_features(use_esm2: bool = True) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
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
    esm_cache = L4_RESULTS / "esm2_protein_embeddings.npz"
    prot_feat: Dict[str, np.ndarray] = {}
    gene_to_seq: Dict[str, str] = {}

    if pf_path.exists():
        df = pd.read_csv(pf_path)
        for _, row in df.iterrows():
            gene = str(row["gene_symbol"]).strip().upper()
            seq = str(row["sequence"]) if pd.notna(row["sequence"]) else ""
            gene_to_seq[gene] = seq

    genes = list(gene_to_seq.keys())

    # ---- v17: 尝试 ESM-2 嵌入 ----
    esm2_embeddings = None
    if use_esm2:
        try:
            esm2_embeddings = compute_esm2_embeddings(
                gene_to_seq, cache_path=esm_cache,
                model_name="facebook/esm2_t30_150M_UR50D",
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

        pseaac_data: Dict[str, np.ndarray] = {}
        if pseaac_path.exists():
            df_pseaac = pd.read_csv(pseaac_path)
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

    return prot_feat, gene_to_seq


def load_tcm_pool() -> pd.DataFrame:
    noleak_path = L3_RESULTS / "tcm_compound_pool_tox_filtered_noleak.csv"
    original_path = L3_RESULTS / "tcm_compound_pool_tox_filtered.csv"
    tcm_path = noleak_path if noleak_path.exists() else original_path
    if not tcm_path.exists():
        logger.error(f"TCM 候选池文件不存在: {tcm_path}")
        sys.exit(1)
    df = pd.read_csv(tcm_path, low_memory=False)
    source_tag = "去泄漏版" if tcm_path == noleak_path else "原始版"
    logger.info(f"TCM 候选池（{source_tag}）: {len(df)} 个化合物")
    return df


# ============================================================
# 4. 图构建 & 邻接表预计算
# ============================================================
def build_pathway_neighbors(
    gene_to_pathways: Dict[str, List[str]],
    gene_to_idx: Dict[str, int],
    n_compounds: int,
) -> Dict[int, set]:
    """预计算同通路蛋白邻居（用于中度负样本采样）

    Returns:
        prot_to_path_neighbors: {蛋白局部索引: set(同通路其他蛋白局部索引)}
    """
    pathway_to_genes: Dict[str, set] = defaultdict(set)
    for gene, paths in gene_to_pathways.items():
        if gene not in gene_to_idx:
            continue
        g_idx = gene_to_idx[gene] - n_compounds
        if g_idx < 0:
            continue
        for p in paths:
            pathway_to_genes[p].add(g_idx)

    prot_to_path_neighbors: Dict[int, set] = defaultdict(set)
    for genes in pathway_to_genes.values():
        for g in genes:
            prot_to_path_neighbors[g].update(genes - {g})

    return prot_to_path_neighbors


def build_graphs_and_adj(
    cpi_df: pd.DataFrame,
    ppi_df: pd.DataFrame,
    gene_to_pathways: Dict[str, List[str]],
    prot_feat: Dict[str, np.ndarray],
):
    """构建同质图 + 异质图 + 邻接表"""
    # 化合物索引
    all_smiles = sorted(cpi_df["canonical_smiles"].unique())
    smi_to_idx = {s: i for i, s in enumerate(all_smiles)}
    n_compounds = len(all_smiles)

    # 蛋白索引
    ppi_genes = set()
    for _, row in ppi_df.iterrows():
        ppi_genes.add(str(row["source"]).strip().upper())
        ppi_genes.add(str(row["target"]).strip().upper())
    all_genes = sorted(set(cpi_df["gene"].unique()) | set(prot_feat.keys()) | ppi_genes)
    gene_to_idx = {g: i + n_compounds for i, g in enumerate(all_genes)}
    n_proteins = len(all_genes)

    # 化合物特征
    logger.info(f"  computing compound features ({n_compounds} compounds)...")
    comp_feat, _, _, _ = build_compound_features(all_smiles)

    # 蛋白特征
    prot_feat_dim = next(iter(prot_feat.values())).shape[0] if prot_feat else 20
    prot_esm_dim = prot_feat_dim  # v17-ESM2: 保存原始 ESM-2 维度（通路拼接前），供独立投影器使用
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
    # 为 SAGE 模型添加通路信息，弥补蛋白冷启动场景下拓扑缺失的不足
    # 将每个蛋白的 KEGG 通路隶属关系编码为 one-hot 向量，拼接到 ESM-2 嵌入后
    all_pathways = sorted(set(pid for paths in gene_to_pathways.values() for pid in paths))
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

    # ======== 邻接表构建 ========
    # 同质图邻接表（用于 GAT 分支采样）
    homo_adj = defaultdict(list)  # node_idx -> [neighbor_indices]
    homo_cpi_adj = defaultdict(list)  # compound_idx -> [protein_global_indices]

    for _, row in cpi_df.iterrows():
        smi = row["canonical_smiles"]
        gene = row["gene"]
        if smi in smi_to_idx and gene in gene_to_idx:
            src = smi_to_idx[smi]
            dst = gene_to_idx[gene]
            homo_adj[src].append(dst)
            homo_adj[dst].append(src)
            homo_cpi_adj[src].append(dst - n_compounds)

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
    }

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

    # 通路索引（已在特征构建阶段计算，此处复用）
    n_pathways = n_pathways_feat

    # v12: 通路ID完全数值化 — 将邻接表中的字符串通路ID转为整数索引，消除字符串匹配开销
    new_pt_adj = defaultdict(list)
    for prot_idx, path_list in hetero_adj[("protein", "belongs_to", "pathway")].items():
        for pid in path_list:
            if pid in pathway_to_idx:
                new_pt_adj[prot_idx].append(pathway_to_idx[pid])
    hetero_adj[("protein", "belongs_to", "pathway")] = new_pt_adj
    logger.info(f"  通路ID数值化完成: {len(new_pt_adj)} 蛋白 → {n_pathways} 通路")

    # v12: 预计算同通路蛋白邻居（用于中度负样本采样）
    prot_to_path_neighbors = build_pathway_neighbors(gene_to_pathways, gene_to_idx, n_compounds)
    logger.info(f"  同通路蛋白邻居: {len(prot_to_path_neighbors)} 蛋白")

    # 异质图数据（用于 HGT 全图验证）
    hetero_data = HeteroData()
    hetero_data["compound"].x = torch.from_numpy(comp_feat)
    hetero_data["protein"].x = torch.from_numpy(prot_matrix)
    hetero_data["pathway"].x = torch.zeros(max(n_pathways, 1), 1, dtype=torch.float32)
    hetero_data["pathway"].n_pathways = n_pathways

    # CPI 边
    cpi_edges = [[], []]
    for src, dsts in hetero_adj[("compound", "interacts", "protein")].items():
        for dst in dsts:
            cpi_edges[0].append(src)
            cpi_edges[1].append(dst)
    hetero_data["compound", "interacts", "protein"].edge_index = torch.tensor(cpi_edges, dtype=torch.long)

    # PPI 边
    ppi_edges = [[], []]
    for src, dsts in hetero_adj[("protein", "ppi", "protein")].items():
        for dst in dsts:
            ppi_edges[0].append(src)
            ppi_edges[1].append(dst)
    hetero_data["protein", "ppi", "protein"].edge_index = torch.tensor(ppi_edges, dtype=torch.long)

    # 通路边（v12: 通路ID已数值化，dst 已是整数，无需再次转换）
    pt_edges = [[], []]
    for src, dsts in hetero_adj[("protein", "belongs_to", "pathway")].items():
        for dst in dsts:
            pt_edges[0].append(src)
            pt_edges[1].append(dst)
    hetero_data["protein", "belongs_to", "pathway"].edge_index = torch.tensor(pt_edges, dtype=torch.long)
    rev_pt = [pt_edges[1][:], pt_edges[0][:]]
    hetero_data["pathway", "includes", "protein"].edge_index = torch.tensor(rev_pt, dtype=torch.long)

    logger.info(f"异质图: compound({n_compounds}) protein({n_proteins}) pathway({n_pathways}) | "
                f"CPI={len(cpi_edges[0])} PPI={len(ppi_edges[0])} Pathway={len(pt_edges[0])}")

    # Opt1: 预计算全图同质边索引，验证/预测直接复用（速度提升 10x+）
    homo_edge_list = []
    for node in range(n_compounds + n_proteins):
        for nbr in homo_adj.get(node, []):
            homo_edge_list.append([node, nbr])
    homo_edge_index = torch.tensor(homo_edge_list, dtype=torch.long).t().contiguous() if homo_edge_list else torch.zeros((2, 0), dtype=torch.long)
    logger.info(f"预计算全图边索引: {homo_edge_index.shape[1]} 条边")

    return {
        "x": x,
        "feat_dim": feat_dim,
        "prot_feat_dim": prot_feat_dim,  # v17: 蛋白特征总维度（ESM2 + 通路 one-hot），供 padding 计算
        "prot_esm_dim": prot_esm_dim,  # v17-ESM2: 原始 ESM-2 维度（640），供独立投影器使用
        "n_compounds": n_compounds,
        "n_proteins": n_proteins,
        "smi_to_idx": smi_to_idx,
        "gene_to_idx": gene_to_idx,
        "homo_adj": homo_adj,
        "homo_cpi_adj": homo_cpi_adj,
        "homo_edge_index": homo_edge_index,
        "hetero_adj": hetero_adj,
        "hetero_data": hetero_data,
        "n_pathways": n_pathways,
        "prot_to_path_neighbors": prot_to_path_neighbors,  # v12: 同通路蛋白邻居（中度负样本）
    }


# ============================================================
# 5. 手动邻居采样（GraphSAGE 风格）
# ============================================================
def drop_edge(edge_index: torch.Tensor, p: float = 0.15) -> torch.Tensor:
    """DropEdge 正则化：随机丢弃 p 比例的边，缓解过拟合与过平滑

    参考: Rong et al. (2020) "DropEdge: Towards Deep Graph Neural Networks", ICLR
    """
    if p <= 0 or edge_index.shape[1] <= 1:
        return edge_index
    mask = torch.rand(edge_index.shape[1], device=edge_index.device) > p
    return edge_index[:, mask]


def sample_homo_subgraph(
    seed_compounds: List[int],
    homo_adj: Dict[int, List[int]],
    num_neighbors: List[int] = [32, 16],
    seed: Optional[int] = None,
):
    """GraphSAGE 风格邻居采样：固定每层邻居数，避免邻居爆炸（v12: 支持种子固定可复现）"""
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


def sample_hetero_subgraph(
    seed_compounds: List[int],
    hetero_adj: dict,
    num_neighbors: List[int] = [32, 16],
    seed: Optional[int] = None,
    seed_proteins: Optional[List[int]] = None,
):
    """异质图手动邻居采样（v15: 通路ID已数值化，移除冗余 pathway_to_idx 参数）

    v18: 新增 seed_proteins 参数，允许将指定蛋白（如验证蛋白）作为孤立节点纳入子图，
    用于蛋白冷启动 OOM 降级 mini-batch 验证。
    """
    if seed is not None:
        random.seed(seed)
    compounds = set(seed_compounds)
    proteins = set(seed_proteins) if seed_proteins else set()
    pathways = set()

    # 1-hop: 化合物 → 蛋白
    cpi_adj = hetero_adj[("compound", "interacts", "protein")]
    for c in seed_compounds:
        if c in cpi_adj:
            nbrs = cpi_adj[c]
            if len(nbrs) > num_neighbors[0]:
                nbrs = random.sample(nbrs, num_neighbors[0])
            proteins.update(nbrs)

    # 2-hop: 蛋白 → 蛋白 + 蛋白 → 通路
    ppi_adj = hetero_adj[("protein", "ppi", "protein")]
    pt_adj = hetero_adj[("protein", "belongs_to", "pathway")]
    for p in list(proteins):
        if p in ppi_adj:
            nbrs = ppi_adj[p]
            if len(nbrs) > num_neighbors[1]:
                nbrs = random.sample(nbrs, num_neighbors[1])
            proteins.update(nbrs)
        if p in pt_adj:
            nbrs = pt_adj[p]
            if len(nbrs) > num_neighbors[1]:
                nbrs = random.sample(nbrs, num_neighbors[1])
            pathways.update(nbrs)

    comp_sorted = sorted(compounds)
    prot_sorted = sorted(proteins)
    path_sorted = sorted(pathways)  # v12: 已是整数通路ID

    comp_map = {c: i for i, c in enumerate(comp_sorted)}
    prot_map = {p: i for i, p in enumerate(prot_sorted)}
    path_map = {p: i for i, p in enumerate(path_sorted)}

    # v12: 通路ID已数值化，直接使用（无需 pathway_to_idx 转换）
    path_global = list(path_sorted)

    sg = HeteroData()
    sg._comp_sorted = comp_sorted
    sg._prot_map = prot_map
    sg._path_global = path_global  # v11: 存储全局通路索引

    def _build_edges(et, src_map, dst_map):
        sl, dl = [], []
        for s, ds in hetero_adj.get(et, {}).items():
            if s in src_map:
                for d in ds:
                    if d in dst_map:
                        sl.append(src_map[s])
                        dl.append(dst_map[d])
        if sl:
            return torch.tensor([sl, dl], dtype=torch.long)
        return torch.zeros((2, 0), dtype=torch.long)

    sg["compound", "interacts", "protein"].edge_index = _build_edges(
        ("compound", "interacts", "protein"), comp_map, prot_map)
    sg["protein", "ppi", "protein"].edge_index = _build_edges(
        ("protein", "ppi", "protein"), prot_map, prot_map)
    sg["protein", "belongs_to", "pathway"].edge_index = _build_edges(
        ("protein", "belongs_to", "pathway"), prot_map, path_map)

    # Bug1 修复: 手动构建反向边（通路→蛋白），_build_edges 无法处理反向映射
    sl_rev, dl_rev = [], []
    for p_global, pathway_ids in hetero_adj.get(("protein", "belongs_to", "pathway"), {}).items():
        if p_global in prot_map:
            for pid in pathway_ids:
                if pid in path_map:
                    sl_rev.append(path_map[pid])   # 通路为源
                    dl_rev.append(prot_map[p_global])  # 蛋白为目标
    if sl_rev:
        sg["pathway", "includes", "protein"].edge_index = torch.tensor(
            [sl_rev, dl_rev], dtype=torch.long)
    else:
        sg["pathway", "includes", "protein"].edge_index = torch.zeros((2, 0), dtype=torch.long)

    return sg, comp_sorted, prot_sorted, path_sorted, comp_map, prot_map


# ============================================================
# 6. GAT 分支模型（SAGEConv，v10 重构）
# ============================================================
class SAGELinkPredictor(nn.Module):
    """GraphSAGE 编码器 + 点积解码器

    v11: SAGEConv + 残差连接 + 手动邻居采样
    v17-ESM2: 添加化合物/蛋白特征投影器，解耦高维特征与图卷积
      - comp_proj: 化合物特征 → 256 → hidden_dim (含 LayerNorm, ReLU, Dropout)
      - prot_proj: 蛋白特征 (ESM-2 640维) → 256 → hidden_dim (含 LayerNorm, ReLU, 高 Dropout)
      - pathway_proj: 通路 one-hot 特征 → 128 → hidden_dim (独立投影，与 ESM-2 相加融合)
      - 先投影再卷积，避免 SAGEConv 直接从 2232 维压缩到 64 维的信息瓶颈
      - 蛋白投影器使用更高 Dropout(0.4) 迫使模型不依赖特定维度
      - 通路投影器独立于 ESM-2，避免稀疏 one-hot 被高维 ESM-2 稀释

    参考:
      - Hamilton et al. (2017) "GraphSAGE", NeurIPS
      - Veleiro et al. (2024) "GeNNius", Bioinformatics
      - Rives et al. (2021) "ESM-2", PNAS
    """

    def __init__(self, comp_feat_dim: int, prot_feat_dim: int, n_compounds: int,
                 hidden_dim: int = 64, out_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.5,
                 n_pathways: int = 0):
        super().__init__()
        self.comp_feat_dim = comp_feat_dim
        self.prot_esm_dim = prot_feat_dim  # v17-ESM2: prot_feat_dim 现为 ESM-2 维度（640），非总维度
        self.n_compounds = n_compounds
        self.out_dim = out_dim
        self.n_pathways = n_pathways

        # v17: 固定温度 T=5.0，不再参与梯度更新（防止优化到极端值使 sigmoid 饱和）
        self.temperature = 5.0

        # ---- v17-ESM2: 化合物特征投影器 ----
        # 将化合物特征（ECFP4+MACCS+Rdkit 描述符）从高维投影到 hidden_dim
        self.comp_proj = nn.Sequential(
            nn.Linear(comp_feat_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
        )

        # ---- v17-ESM2: 蛋白特征投影器 (仅 ESM-2 640维) ----
        # 将 ESM-2 预训练嵌入逐步降维，引入非线性，保留预训练知识
        # 使用更高 Dropout(0.4) 迫使模型不依赖特定维度，提升泛化
        # 注意：通路特征由独立的 pathway_proj 处理，避免稀疏 one-hot 被高维 ESM-2 稀释
        self.prot_feat_proj = nn.Sequential(
            nn.Linear(prot_feat_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),  # 蛋白特征高 Dropout，防止过拟合到特定维度
            nn.Linear(256, hidden_dim),
        )

        # ---- v17-ESM2: 通路特征独立投影器 ----
        # 将通路 one-hot 特征（稀疏，359维）独立投影到 hidden_dim
        # 与 ESM-2 投影结果相加融合，避免稀疏特征被高维 ESM-2 稀释
        # 若无通路数据（n_pathways=0），pathway_proj 为 None
        if n_pathways > 0:
            self.pathway_proj = nn.Sequential(
                nn.Linear(n_pathways, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, hidden_dim),
            )
        else:
            self.pathway_proj = None

        # ---- v17-ESM2: 蛋白特征独立 Dropout 增强 ----
        # 在投影后对蛋白节点嵌入施加额外 Dropout，迫使模型不依赖特定维度
        # 提升蛋白冷启动泛化能力，缓解信息瓶颈
        self.prot_dropout = nn.Dropout(0.4)

        # ---- SAGEConv 层 ----
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropouts.append(nn.Dropout(dropout))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))

        self.convs.append(SAGEConv(hidden_dim, out_dim))
        self.norms.append(nn.Identity())  # No LayerNorm on final layer
        self.dropouts.append(nn.Dropout(dropout))

        # ---- v18: MLP 解码器替换点积 ----
        # 点积解码器假设化合物与蛋白嵌入在同一空间直接可比，
        # 但 ESM-2 640→64 投影后此假设不成立，导致蛋白冷启动指标停滞在随机水平。
        # MLP 解码器允许学习化合物-蛋白间更复杂的非线性交互。
        # 参考: DrugBAN (2023) bilinear decoder; DeepPurpose (2020) MLP decoder
        self.decoder = nn.Sequential(
            nn.Linear(out_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                n_compounds: int = None) -> torch.Tensor:
        """前向传播

        Args:
            x: (N, feat_dim) 节点特征矩阵（含化合物和蛋白，padding 到统一维度）
            edge_index: (2, E) 边索引
            n_compounds: 子图中化合物节点数（None 则使用 self.n_compounds，用于全图推理）

        Returns:
            (N, out_dim) 节点嵌入
        """
        if n_compounds is None:
            n_compounds = self.n_compounds

        # ---- 分离化合物和蛋白特征 ----
        comp_x = x[:n_compounds]          # (n_compounds, comp_feat_dim+padded)
        prot_x = x[n_compounds:]          # (n_proteins, feat_dim)

        # 提取实际特征维度（排除 padding 零）
        comp_x_actual = comp_x[:, :self.comp_feat_dim]

        # ---- v17-ESM2: 独立 ESM-2 投影 ----
        # ESM-2 特征在前 self.prot_esm_dim 维（640）
        prot_esm = prot_x[:, :self.prot_esm_dim]
        prot_h = self.prot_feat_proj(prot_esm)     # (n_proteins, hidden_dim)

        # ---- v17-ESM2: 独立通路投影 ----
        # 通路 one-hot 紧随 ESM-2 之后（n_pathways 维），与 ESM-2 投影相加融合
        if self.pathway_proj is not None and self.n_pathways > 0:
            prot_pathway = prot_x[:, self.prot_esm_dim:self.prot_esm_dim + self.n_pathways]
            prot_h = prot_h + self.pathway_proj(prot_pathway)

        # ---- 化合物特征投影 ----
        comp_h = self.comp_proj(comp_x_actual)      # (n_compounds, hidden_dim)

        # ---- 拼接后通过 SAGEConv ----
        h = torch.cat([comp_h, prot_h], dim=0)

        # v17-ESM2: 对蛋白节点嵌入施加独立 Dropout，迫使模型不依赖特定维度
        # 提升蛋白冷启动泛化能力，缓解信息瓶颈
        prot_indices = slice(n_compounds, h.shape[0])
        h[prot_indices] = self.prot_dropout(h[prot_indices])

        for conv, norm, drop in zip(self.convs, self.norms, self.dropouts):
            h_new = conv(h, edge_index)
            # v11: 残差连接（维度匹配时）
            if h.shape[-1] == h_new.shape[-1]:
                h_new = h_new + h
            h = norm(h_new)
            h = F.relu(h)
            h = drop(h)
        return h

    def decode(self, comp_emb: torch.Tensor, prot_emb: torch.Tensor) -> torch.Tensor:
        """v18: MLP 解码器 — 化合物-蛋白非线性交互建模

        替换原有点积解码器 (comp_emb * prot_emb).sum(dim=1)，
        允许学习化合物与蛋白嵌入之间更复杂的交互关系。

        Args:
            comp_emb: (*, out_dim) 化合物嵌入
            prot_emb: (*, out_dim) 蛋白嵌入

        Returns:
            (*,) 预测 logits（未经过 sigmoid）
        """
        return self.decoder(torch.cat([comp_emb, prot_emb], dim=-1)).squeeze(-1)

    def encode_compound(self, x: torch.Tensor) -> torch.Tensor:
        """编码化合物特征（无图结构，仅投影+卷积）

        用于预测时编码 TCM 化合物（无 CPI 边），镜像 HGT.encode_compound 设计。
        """
        x_actual = x[:, :self.comp_feat_dim]
        h = self.comp_proj(x_actual)
        empty_edge = torch.zeros((2, 0), dtype=torch.long, device=h.device)
        for conv, norm, drop in zip(self.convs, self.norms, self.dropouts):
            h_new = conv(h, empty_edge)
            if h.shape[-1] == h_new.shape[-1]:
                h_new = h_new + h
            h = norm(h_new)
            h = F.relu(h)
            h = drop(h)
        return h


# ============================================================
# 7. HGT 分支模型（HGTConv，v10 优化）
# ============================================================
class HGTLinkPredictor(nn.Module):
    """HGT 异质图编码器 + 双线性解码器

    v11: 节点自适应门控，缓解过平滑
    """

    def __init__(self, hidden_dim: int = 64, out_dim: int = 64,
                 num_heads: int = 2, num_layers: int = 2, dropout: float = 0.5,
                 metadata=None, compound_feat_dim: int = 200,
                 node_feat_dims: Optional[Dict[str, int]] = None):
        super().__init__()
        self.out_dim = out_dim

        # v17: 固定温度 T=5.0
        self.temperature = 5.0

        n_pathways = node_feat_dims.get("pathway_count", 1) if node_feat_dims else 1
        self.pathway_embed = nn.Embedding(max(n_pathways, 1), hidden_dim)

        self.comp_proj = nn.Sequential(
            nn.Linear(compound_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        prot_in_dim = node_feat_dims.get("protein", 640) if node_feat_dims else 640
        self.prot_in_dim = prot_in_dim  # v17-ESM2: 存储实际蛋白特征维度，forward 中提取非 padding 部分
        self.prot_proj = nn.Sequential(
            nn.Linear(prot_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.convs = nn.ModuleList()
        # v12: 节点自适应门控（初始偏置1.0，前期更倾向于接收新信息）
        self.gates = nn.ModuleList()
        if metadata:
            node_types, edge_types = metadata
            for _ in range(num_layers):
                self.convs.append(HGTConv(
                    {nt: hidden_dim for nt in node_types},
                    hidden_dim, metadata,
                    heads=num_heads,
                ))
                gate = nn.Linear(hidden_dim, 1)
                nn.init.constant_(gate.bias, 1.0)  # v12: 偏置设为1，前期更新力度更强
                self.gates.append(gate)

        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.bilinear = nn.Bilinear(out_dim, out_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def encode_compound(self, x_comp: torch.Tensor) -> torch.Tensor:
        h = self.comp_proj(x_comp)
        return self.out_proj(h)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: v.clone() for k, v in x_dict.items()}

        if "compound" in x_dict:
            x_dict["compound"] = self.comp_proj(x_dict["compound"])
        if "protein" in x_dict:
            # v17-ESM2: 提取实际蛋白特征维度（排除 padding 零），避免 Linear 维度不匹配
            x_dict["protein"] = self.prot_proj(x_dict["protein"][:, :self.prot_in_dim])
        # v15: 通路嵌入统一由外部管理（每 epoch 刷新，模型内部不做双重判断）

        for layer_idx, conv in enumerate(self.convs):
            out = conv(x_dict, edge_index_dict)
            for nt in x_dict:
                if nt not in out:
                    out[nt] = x_dict[nt]
            # v11: 节点自适应门控，缓解过平滑
            gate = self.gates[layer_idx]
            for nt in out:
                if nt in x_dict and x_dict[nt].shape == out[nt].shape:
                    g = torch.sigmoid(gate(x_dict[nt]))
                    out[nt] = g * out[nt] + (1 - g) * x_dict[nt]
            x_dict = out
            x_dict = {k: self.dropout(v) for k, v in x_dict.items()}

        for nt in ["compound", "protein"]:
            if nt in x_dict:
                x_dict[nt] = self.out_proj(x_dict[nt])

        return x_dict

    def decode(self, comp_emb: torch.Tensor, prot_emb: torch.Tensor) -> torch.Tensor:
        return self.bilinear(comp_emb, prot_emb).squeeze(-1)


# ============================================================
# 8. 损失函数（v17: Focal Loss + 标签平滑）
# ============================================================
def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.75,
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


# ============================================================
# 9. Memory Bank + InfoNCE 对比损失（v17 改进）
# ============================================================
class MemoryBank:
    """存储最近 K 个 batch 的蛋白嵌入，供全局困难负样本采样

    异步更新：每次训练步骤后，将当前 batch 的蛋白嵌入入队，
    旧嵌入出队。内存开销：K * out_dim * 4 bytes。

    参考: He et al. (2020) "Momentum Contrast for Unsupervised
          Visual Representation Learning", CVPR.
          Wu et al. (2021) "Self-supervised Learning on Graphs: Contrastive"
    """

    def __init__(self, max_size: int = 8192, out_dim: int = 64, device: str = "cpu"):
        self.max_size = max_size
        self.out_dim = out_dim
        self.device = device
        self.bank = torch.zeros(max_size, out_dim, device=device)
        self.ptr = 0
        self.full = False

    def update(self, embeddings: torch.Tensor):
        """将新嵌入入队（FIFO）"""
        n = embeddings.shape[0]
        if n == 0:
            return
        end = self.ptr + n
        if end > self.max_size:
            # 环绕
            first_part = self.max_size - self.ptr
            self.bank[self.ptr:] = embeddings[:first_part].detach()
            remaining = n - first_part
            if remaining > 0:
                self.bank[:remaining] = embeddings[first_part:].detach()
            self.full = True
        else:
            self.bank[self.ptr:end] = embeddings.detach()
            if end == self.max_size:
                self.full = True
        self.ptr = (self.ptr + n) % self.max_size

    def sample(self, n: int) -> torch.Tensor:
        """从 bank 中随机采样 n 个嵌入"""
        available = self.max_size if self.full else self.ptr
        if available == 0:
            return torch.zeros(0, self.out_dim, device=self.device)
        n_sample = min(n, available)
        indices = torch.randperm(available, device=self.device)[:n_sample]
        return self.bank[indices]

    def size(self) -> int:
        return self.max_size if self.full else self.ptr


def infonce_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    memory_scores: Optional[torch.Tensor] = None,
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


# ============================================================
# 10. GAT 分支训练（mini-batch 邻居采样）
# ============================================================
def train_sage(
    model: SAGELinkPredictor,
    graphs: dict,
    train_compounds: List[int],
    val_compounds: List[int],
    compound_to_pos: Dict[int, set],
    val_proteins: set = None,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15,
    batch_size: int = 256,
    num_neighbors: List[int] = [32, 16],
    prot_to_path_neighbors: Optional[Dict[int, set]] = None,
    flag_step: float = 0.01,
) -> Tuple[SAGELinkPredictor, List[dict]]:
    """v17: GraphSAGE mini-batch 训练 — 三级负采样 + 向量化损失 + 蛋白冷启动验证 + Memory Bank"""

    model = model.to(DEVICE)
    for p in model.parameters():
        if p.dim() >= 2:
            nn.init.xavier_uniform_(p)

    x = graphs["x"].to(DEVICE)
    # v18: 使用训练安全邻接表，验证蛋白在训练阶段完全不可见
    homo_adj = graphs.get("homo_adj_train", graphs["homo_adj"])
    n_compounds = graphs["n_compounds"]
    all_compound_to_pos = compound_to_pos

    precomputed_pos = {src: sorted(pos_set) for src, pos_set in compound_to_pos.items() if pos_set}

    # v17: AdamW 解耦权重衰减与自适应学习率
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # v17: LR Warmup + Cosine Annealing（前5%线性预热，后95%余弦退火至1e-6）
    warmup_epochs = max(1, int(epochs * 0.05))
    def lr_lambda(e):
        if e < warmup_epochs:
            return e / warmup_epochs
        progress = (e - warmup_epochs) / (epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress)) * 1.0 + 1e-6
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # v17: Memory Bank — 存储跨 batch 蛋白嵌入，供全局困难负样本采样
    memory_bank = MemoryBank(max_size=8192, out_dim=model.out_dim, device=DEVICE)
    # v17: 多指标联合早停（AUC+AUPR+loss gap）
    best_val_auc = 0.0
    best_val_aupr = 0.0
    best_prot_aupr = 0.0  # v18: 蛋白冷启动 AUPR 用于早停
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        random.shuffle(train_compounds)
        for batch_start in range(0, len(train_compounds), batch_size):
            batch_seeds = train_compounds[batch_start:batch_start + batch_size]

            # 邻居采样 (v12: epoch+batch固定种子，保证可复现)
            node_list, node_to_local, edge_index = sample_homo_subgraph(
                batch_seeds, homo_adj, num_neighbors,
                seed=epoch * 10000 + batch_start)
            edge_index = edge_index.to(DEVICE)
            # v17: DropEdge — 随机丢弃 15% 边，缓解图稠密区域过拟合
            edge_index = drop_edge(edge_index, p=0.15)

            sub_x = x[torch.tensor(node_list, device=DEVICE)]
            # v17: Gaussian Feature Augmentation — 对化合物特征嵌入施加随机扰动
            if flag_step > 0:
                sub_x = sub_x + flag_step * torch.randn_like(sub_x)
                sub_x = sub_x.detach()
            # v17-ESM2: 计算子图中化合物节点数，供投影器拆分化合物/蛋白特征
            n_compounds_in_sub = sum(1 for n in node_list if n < n_compounds)
            node_emb = model(sub_x, edge_index, n_compounds=n_compounds_in_sub)

            # 局部索引映射
            seed_local = []
            batch_idx_to_comp_idx = {}  # batch_seeds 位置 → comp_emb 位置
            for bi, s in enumerate(batch_seeds):
                if s in node_to_local:
                    batch_idx_to_comp_idx[bi] = len(seed_local)
                    seed_local.append(node_to_local[s])
            if not seed_local:
                continue

            # 实际蛋白在子图中的位置
            prot_local_indices = [i for i, n in enumerate(node_list) if n >= n_compounds]

            comp_emb = node_emb[torch.tensor(seed_local, device=DEVICE)]
            if not prot_local_indices:
                continue

            prot_emb = node_emb[torch.tensor(prot_local_indices, device=DEVICE)]
            n_batch_prots = len(prot_local_indices)
            n_unique = len(seed_local)
            T = model.temperature

            # 构建 local_pos_in_node_list -> prot_emb 中的位置映射
            local_to_prot_pos = {local_pos: i for i, local_pos in enumerate(prot_local_indices)}

            # ---- v13: 向量化正样本损失（precomputed_pos 存全局索引，直接用 node_to_local 查找） ----
            pos_src, pos_dst = [], []
            for bi, s in enumerate(batch_seeds):
                ci = batch_idx_to_comp_idx.get(bi)
                if ci is None or s not in precomputed_pos:
                    continue
                for p_global in precomputed_pos[s]:  # v13: p_global 是真正的全局蛋白索引
                    if p_global in node_to_local:  # v13: 直接查找，无需 + n_compounds
                        local_pos = node_to_local[p_global]
                        if local_pos in local_to_prot_pos:
                            prot_pos = local_to_prot_pos[local_pos]
                            if 0 <= prot_pos < n_batch_prots:
                                pos_src.append(ci)
                                pos_dst.append(prot_pos)

            if not pos_src:
                continue

            pos_src_t = torch.tensor(pos_src, device=DEVICE)
            pos_dst_t = torch.tensor(pos_dst, device=DEVICE)

            # 向量化正样本分数
            pos_score = model.decode(comp_emb[pos_src_t], prot_emb[pos_dst_t]) / T

            # ---- v17: 课程负采样（先易后难三阶段） ----
            # Phase 1: 前30% epochs → 仅随机负样本
            # Phase 2: 中40% epochs → 随机70% + 中度30%
            # Phase 3: 后30% epochs → 随机90% + 极硬10%
            # 参考: CNS (2022) "Curriculum Negative Sampling"
            if n_unique > 0 and n_batch_prots > 1:
                # v18: MLP 解码器 — 向量化全对评分矩阵
                # (comp_emb @ prot_emb.T) 替换为 expand+cat+decode+reshape
                comp_exp = comp_emb.unsqueeze(1).expand(-1, n_batch_prots, -1).reshape(-1, comp_emb.shape[-1])
                prot_exp = prot_emb.unsqueeze(0).expand(n_unique, -1, -1).reshape(-1, prot_emb.shape[-1])
                all_scores = model.decode(comp_exp, prot_exp).reshape(n_unique, n_batch_prots) / T

                # 正样本 mask
                mask = torch.zeros(n_unique, n_batch_prots, device=DEVICE)
                for bi, s in enumerate(batch_seeds):
                    ci = batch_idx_to_comp_idx.get(bi)
                    if ci is None or s not in precomputed_pos:
                        continue
                    for p_global in precomputed_pos[s]:
                        if p_global in node_to_local:
                            local_pos = node_to_local[p_global]
                            if local_pos in local_to_prot_pos:
                                pi = local_to_prot_pos[local_pos]
                                if 0 <= pi < n_batch_prots:
                                    mask[ci, pi] = -1e9

                # 课程阶段判定
                curriculum_phase = epoch / epochs
                if curriculum_phase < 0.3:
                    # Phase 1: 100% 随机
                    n_rand = n_unique
                    n_medium = n_hard = 0
                elif curriculum_phase < 0.7:
                    # Phase 2: 70% 随机 + 30% 中度
                    n_medium = int(n_unique * 0.3)
                    n_rand = n_unique - n_medium
                    n_hard = 0
                else:
                    # Phase 3: 90% 随机 + 10% 极硬
                    n_hard = int(n_unique * 0.1)
                    n_rand = n_unique - n_hard
                    n_medium = 0

                # 初始化 hard_neg_scores 为随机负样本
                hard_neg_scores = torch.zeros(n_unique, device=DEVICE)
                rand_perm = torch.randperm(n_unique, device=DEVICE)
                # 所有化合物先用随机负样本填充（排除正样本 + 保护全零行）
                valid_mask = (mask == 0).float()
                row_sum = valid_mask.sum(dim=1)
                safe_rows = row_sum > 0
                if safe_rows.any():
                    valid_mask = valid_mask / (row_sum.unsqueeze(1) + 1e-10)
                    rand_dst = torch.multinomial(valid_mask, 1).squeeze(-1)
                    hard_neg_scores[safe_rows] = model.decode(comp_emb[safe_rows], prot_emb[rand_dst[safe_rows]]) / T
                    hard_neg_scores = torch.clamp(hard_neg_scores, -10, 10)
                # 全零行（所有蛋白均为正样本）保持 hard_neg_scores=0

                medium_idx = None
                hard_idx = None

                # Phase 2/3: 中度负样本
                if n_medium > 0 and prot_to_path_neighbors is not None and n_batch_prots > 2:
                    medium_neg_scores = hard_neg_scores.clone()
                    medium_found = torch.zeros(n_unique, dtype=torch.bool, device=DEVICE)
                    for bi, s in enumerate(batch_seeds):
                        ci = batch_idx_to_comp_idx.get(bi)
                        if ci is None or s not in precomputed_pos:
                            continue
                        path_neighbors: set = set()
                        for p_global in precomputed_pos[s]:
                            p_local = p_global - n_compounds
                            if p_local >= 0 and p_local in prot_to_path_neighbors:
                                path_neighbors.update(prot_to_path_neighbors[p_local])
                        if not path_neighbors:
                            continue
                        batch_neighbor_positions = []
                        for pn in path_neighbors:
                            pn_global = pn + n_compounds
                            if pn_global in node_to_local:
                                local_pos = node_to_local[pn_global]
                                if local_pos in local_to_prot_pos:
                                    pi = local_to_prot_pos[local_pos]
                                    if 0 <= pi < n_batch_prots and mask[ci, pi] == 0:
                                        batch_neighbor_positions.append(pi)
                        if batch_neighbor_positions:
                            bi_t = torch.tensor(batch_neighbor_positions, device=DEVICE)
                            neighbor_scores = all_scores[ci, bi_t]
                            best_idx = neighbor_scores.argmax()
                            medium_neg_scores[ci] = torch.clamp(neighbor_scores[best_idx], -10, 10)
                            medium_found[ci] = True

                    medium_candidates = torch.where(medium_found)[0]
                    if len(medium_candidates) > 0 and n_medium > 0:
                        n_actual_medium = min(n_medium, len(medium_candidates))
                        perm = torch.randperm(len(medium_candidates), device=DEVICE)
                        medium_idx = medium_candidates[perm[:n_actual_medium]]
                        hard_neg_scores[medium_idx] = medium_neg_scores[medium_idx]

                # Phase 3: 极硬负样本
                if n_hard > 0:
                    hard_neg_idx = (all_scores + mask).argmax(dim=1)
                    hard_scores = all_scores[torch.arange(n_unique, device=DEVICE), hard_neg_idx]
                    hard_scores = torch.clamp(hard_scores, -10, 10)
                    hard_candidates = torch.randperm(n_unique, device=DEVICE)[:n_hard]
                    hard_neg_scores[hard_candidates] = hard_scores[hard_candidates]

                # v18: 固定 Focal alpha = 0.75，避免动态 alpha 导致负样本权重过小
                pos_loss = focal_loss_with_logits(
                    pos_score, torch.full_like(pos_score, 0.9), alpha=0.75)
                neg_loss = focal_loss_with_logits(
                    hard_neg_scores, torch.full_like(hard_neg_scores, 0.1), alpha=0.75)

                # ---- v18: 向量化 BPR 损失 — 为每个正样本对独立采样负样本 ----
                pair_mask = mask[pos_src_t]  # (n_pos, n_batch_prots)
                bpr_valid_mask = (pair_mask == 0).float()
                bpr_row_sum = bpr_valid_mask.sum(dim=1)
                bpr_safe = bpr_row_sum > 0
                bpr_neg_scores = torch.zeros(len(pos_src_t), device=DEVICE)
                if bpr_safe.any():
                    bpr_valid_mask[bpr_safe] = bpr_valid_mask[bpr_safe] / bpr_row_sum[bpr_safe].unsqueeze(1)
                    bpr_neg_dst = torch.multinomial(bpr_valid_mask, 1).squeeze(-1)
                    bpr_neg_scores[bpr_safe] = all_scores[pos_src_t[bpr_safe], bpr_neg_dst[bpr_safe]]
                bpr_loss = -torch.log(torch.sigmoid(pos_score - bpr_neg_scores) + 1e-8).mean()

                loss = 0.6 * (pos_loss + neg_loss) + 0.4 * bpr_loss

                # ---- v17: Memory Bank InfoNCE 对比损失（epoch > 50 启用） ----
                if epoch > 50 and memory_bank.size() > 0:
                    n_mem = min(256, memory_bank.size())
                    mem_emb = memory_bank.sample(n_mem)
                    if mem_emb.shape[0] > 0:
                        # v18: InfoNCE — 所有 logits 统一使用 MLP 解码器输出
                        # pos_score/hard_neg_scores 已为 model.decode()/T，*T 还原为原始解码器输出
                        n_pos_mem = comp_emb[pos_src_t].shape[0]
                        n_mem_actual = mem_emb.shape[0]
                        comp_exp_mem = comp_emb[pos_src_t].unsqueeze(1).expand(-1, n_mem_actual, -1).reshape(-1, comp_emb.shape[-1])
                        mem_emb_exp = mem_emb.unsqueeze(0).expand(n_pos_mem, -1, -1).reshape(-1, mem_emb.shape[-1])
                        mem_scores = model.decode(comp_exp_mem, mem_emb_exp).reshape(n_pos_mem, n_mem_actual)
                        infonce = infonce_loss(
                            pos_score * T, hard_neg_scores[pos_src_t] * T,
                            memory_scores=mem_scores, temperature=0.07,
                        )
                        loss = loss + 0.1 * infonce
            else:
                loss = pos_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # v17: 更新 Memory Bank
            memory_bank.update(prot_emb.detach())

            total_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            continue

        avg_loss = total_loss / n_batches

        # v17: 验证 + 蛋白冷启动 + 多指标早停
        if epoch % 5 == 0 and val_compounds:
            model.eval()
            val_metrics = _validate_sage(model, x, graphs.get("homo_edge_index_val", graphs["homo_edge_index"]), val_compounds, all_compound_to_pos,
                                         n_compounds)
            m = val_metrics

            # v17: 蛋白冷启动验证
            prot_cold = _validate_sage_protein_cold(
                model, x, graphs.get("homo_edge_index_prot_cold", graphs["homo_edge_index"]), val_compounds, all_compound_to_pos,
                n_compounds, graphs["n_proteins"], val_proteins,
            )

            history.append({"epoch": epoch, "loss": avg_loss, **m,
                         "prot_auc": prot_cold["auc"], "prot_aupr": prot_cold["aupr"]})
            logger.info(f"  SAGE epoch {epoch:3d} | loss={avg_loss:.4f} | val_auc={m['auc']:.4f} | val_aupr={m['aupr']:.4f} | "
                        f"prot_auc={prot_cold['auc']:.4f} | prot_aupr={prot_cold['aupr']:.4f}")

            # v18: 蛋白冷启动早停 — 监控 prot_aupr（蛋白冷启动是核心泛化指标）
            # val_aupr 和 val_auc 仅记录参考，不参与早停决策
            if prot_cold["aupr"] > best_prot_aupr:
                best_prot_aupr = prot_cold["aupr"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if m["aupr"] > best_val_aupr:
                best_val_aupr = m["aupr"]
            if m["auc"] > best_val_auc:
                best_val_auc = m["auc"]

            # v18: 早停 — 连续 patience 次验证 prot_aupr 无提升 → 触发
            if patience_counter >= patience:
                logger.info(f"  SAGE 早停 (epoch {epoch}, patience_counter={patience_counter})")
                break

            # v17: Memory Bank 全局刷新 — 每5 epoch 全图前向，填充完整蛋白嵌入
            # v18: 使用训练安全图，并过滤掉验证蛋白，避免验证信息进入 bank
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    full_node_emb = model(x, graphs.get("homo_edge_index_train", graphs["homo_edge_index"]).to(DEVICE),
                                          n_compounds=n_compounds)
                    full_prot_emb = full_node_emb[n_compounds:]
                    # 排除验证蛋白嵌入
                    if val_proteins is not None and len(val_proteins) > 0:
                        train_prot_mask = torch.ones(full_prot_emb.shape[0], dtype=torch.bool, device=DEVICE)
                        train_prot_mask[list(val_proteins)] = False
                        full_prot_emb = full_prot_emb[train_prot_mask]
                    memory_bank = MemoryBank(max_size=8192, out_dim=model.out_dim, device=DEVICE)
                    memory_bank.update(full_prot_emb)
                logger.info(f"  SAGE Memory Bank 全局刷新: {memory_bank.size()} 训练蛋白嵌入")
                model.train()

        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    best_entry = max(history, key=lambda x: x["auc"]) if history else {"auc": 0.0}
    logger.info(f"  SAGE best val_auc={best_entry['auc']:.4f}")
    return model, history


# ============================================================
# 验证安全图构建（v17 冷启动数据泄露修复）
# ============================================================
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
) -> "HeteroData":
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

    for edge_type in hetero_data.edge_types:
        edge_index = hetero_data[edge_type].edge_index
        src_type, rel, dst_type = edge_type
        keep_mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)

        # 1) 涉及验证集化合物的 CPI 边：compound -> protein
        if edge_type == ("compound", "interacts", "protein") and val_comp_set:
            val_comp_tensor = torch.tensor(sorted(val_comp_set), dtype=torch.long, device=edge_index.device)
            keep_mask = keep_mask & (~torch.isin(edge_index[0], val_comp_tensor))

        # 2) 涉及验证集蛋白的边（PPI / protein-pathway / pathway-protein）
        if val_prot_set:
            val_prot_tensor = torch.tensor(sorted(val_prot_set), dtype=torch.long, device=edge_index.device)
            if src_type == "protein":
                keep_mask = keep_mask & (~torch.isin(edge_index[0], val_prot_tensor))
            if dst_type == "protein":
                keep_mask = keep_mask & (~torch.isin(edge_index[1], val_prot_tensor))

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
    val_comp_tensor = torch.tensor(sorted(val_comp_set), dtype=torch.long, device=homo_edge_index.device)
    remove = torch.isin(src, val_comp_tensor) | torch.isin(dst, val_comp_tensor)
    mask = ~remove
    n_removed = (~mask).sum().item()
    logger.info(f"  化合物冷启动验证同质图: 移除 {n_removed} 条边 (仅 val_comp), 保留 {mask.sum().item()} 条边")
    return homo_edge_index[:, mask]


def _build_val_comp_cold_hetero_data(
    hetero_data,
    val_comp_set: set,
) -> "HeteroData":
    """v18: 构建化合物冷启动验证异质图

    仅移除验证集化合物相关的 CPI 边，保留所有蛋白-蛋白/蛋白-通路/通路-蛋白边。
    """
    hetero_data_val = HeteroData()
    for node_type in hetero_data.node_types:
        hetero_data_val[node_type].x = hetero_data[node_type].x.clone()
        if node_type == "pathway" and hasattr(hetero_data["pathway"], "n_pathways"):
            hetero_data_val["pathway"].n_pathways = hetero_data["pathway"].n_pathways

    for edge_type in hetero_data.edge_types:
        edge_index = hetero_data[edge_type].edge_index
        if edge_type == ("compound", "interacts", "protein") and val_comp_set:
            val_comp_tensor = torch.tensor(sorted(val_comp_set), dtype=torch.long, device=edge_index.device)
            keep_mask = ~torch.isin(edge_index[0], val_comp_tensor)
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


def _build_train_safe_homo_cpi_adj(
    homo_cpi_adj,
    val_comp_set: set,
    val_prot_set: set = None,
):
    """v18: 构建训练安全 CPI 邻接表（蛋白局部索引）"""
    train_cpi_adj = defaultdict(list)
    for src, dsts in homo_cpi_adj.items():
        if src in val_comp_set:
            continue
        for dst in dsts:
            if val_prot_set is None or dst not in val_prot_set:
                train_cpi_adj[src].append(dst)
    return train_cpi_adj


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
            else:
                new_adj[src].extend(dsts)
        train_adj[et] = new_adj
    return train_adj


def _build_val_safe_hetero_adj(
    hetero_adj,
    n_compounds: int,
    val_comp_set: set,
    val_prot_set: set = None,
):
    """v18: 构建验证安全异质邻接表

    用于 HGT OOM 降级时的 mini-batch 验证，确保子图采样不引入验证蛋白边。
    与训练安全邻接表语义相同（均移除验证集化合物/蛋白相关边），但单独命名
    便于后续区分验证/训练采样策略。
    """
    # 验证安全邻接表与训练安全邻接表在当前设定下等价
    return _build_train_safe_hetero_adj(hetero_adj, n_compounds, val_comp_set, val_prot_set)


def _validate_sage(model, x, homo_edge_index, val_compounds, all_compound_to_pos, n_compounds):
    """v18: SAGE 验证 — 批量 MLP 解码器评分，避免 Python 循环反复 forward"""
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

        # v18: 批量预计算 (n_val, n_prots) 得分矩阵，避免 Python 循环中反复调用 MLP
        comp_sub = comp_emb[val_compounds_list]  # (n_val, d)
        batch_size = 512
        score_chunks = []
        for start in range(0, n_val, batch_size):
            end = min(start + batch_size, n_val)
            sub_comp = comp_sub[start:end]
            sub_comp_exp = sub_comp.unsqueeze(1).expand(-1, n_prots, -1).reshape(-1, sub_comp.shape[-1])
            prot_exp = prot_emb.unsqueeze(0).expand(end - start, -1, -1).reshape(-1, prot_emb.shape[-1])
            sub_scores = model.decode(sub_comp_exp, prot_exp).reshape(end - start, n_prots) / T
            score_chunks.append(sub_scores)
        score_matrix = torch.cat(score_chunks, dim=0)  # (n_val, n_prots)

        y_true, y_score = [], []
        n_valid = 0
        for idx, src in enumerate(val_compounds_list):
            pos_set = all_compound_to_pos.get(src, set())
            # v13: pos_set 存全局索引，转为局部索引
            valid_pos = [p - n_compounds for p in pos_set if n_compounds <= p < n_compounds + n_prots]
            if not valid_pos:
                continue
            n_valid += 1

            scores = score_matrix[idx]

            # 正样本
            for p in valid_pos:
                y_true.append(1)
                y_score.append(torch.sigmoid(scores[p]).item())

            # 硬负样本（v13: 边界检查，n_prots 可能 < 5）
            n_hard = min(5, n_prots - len(valid_pos))
            if n_hard > 0:
                mask = torch.zeros(n_prots, device=DEVICE)
                for p in valid_pos:
                    mask[p] = -1e9
                _, hard_indices = (scores + mask).topk(n_hard)
                for hi in hard_indices:
                    if hi.item() < n_prots:
                        y_true.append(0)
                        y_score.append(torch.sigmoid(scores[hi]).item())

            # 随机负样本（v13: 边界检查）
            n_rand = min(5, n_prots - len(valid_pos))
            if n_rand > 0:
                rand_mask = torch.ones(n_prots, device=DEVICE)
                for p in valid_pos:
                    rand_mask[p] = 0
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
        return {
            "auc": float(roc_auc_score(y_true_arr, y_score_arr)),
            "aupr": float(average_precision_score(y_true_arr, y_score_arr)),
            "n_valid_compounds": n_valid,
        }


def _validate_sage_protein_cold(
    model, x, homo_edge_index, val_compounds, all_compound_to_pos,
    n_compounds, n_proteins, val_proteins: set,
):
    """v18: SAGE 蛋白冷启动验证 — 批量 MLP 解码器评分

    仅评估对未见蛋白的预测能力。
    参考: DrugBAN (2023), DeepPurpose (2020)
    """
    with torch.no_grad():
        x_dev = x.to(DEVICE)
        edge_index = homo_edge_index.to(DEVICE)
        node_emb = model(x_dev, edge_index)  # n_compounds=None 使用 self.n_compounds
        prot_emb = node_emb[n_compounds:]
        comp_emb = node_emb[:n_compounds]
        T = model.temperature
        n_prots_actual = prot_emb.shape[0]

        val_compounds_list = list(val_compounds)
        val_prot_list = sorted(val_proteins)
        n_val_prots = len(val_prot_list)

        if len(val_compounds_list) == 0 or n_val_prots == 0:
            return {"auc": 0.5, "aupr": 0.5, "n_valid_compounds": 0}

        # v18: 批量预计算 (n_val, n_val_prots) 得分矩阵
        comp_sub = comp_emb[val_compounds_list]          # (n_val, d)
        prot_sub = prot_emb[val_prot_list]               # (n_val_prots, d)
        batch_size = 512
        score_chunks = []
        for start in range(0, len(val_compounds_list), batch_size):
            end = min(start + batch_size, len(val_compounds_list))
            sub_comp = comp_sub[start:end]
            sub_comp_exp = sub_comp.unsqueeze(1).expand(-1, n_val_prots, -1).reshape(-1, sub_comp.shape[-1])
            prot_exp = prot_sub.unsqueeze(0).expand(end - start, -1, -1).reshape(-1, prot_sub.shape[-1])
            sub_scores = model.decode(sub_comp_exp, prot_exp).reshape(end - start, n_val_prots) / T
            score_chunks.append(sub_scores)
        score_matrix = torch.cat(score_chunks, dim=0)    # (n_val, n_val_prots)

        # val_proteins 局部索引 → 子矩阵列索引
        val_prot_to_local = {p: i for i, p in enumerate(val_prot_list)}

        y_true, y_score = [], []
        n_valid = 0
        for idx, src in enumerate(val_compounds_list):
            pos_set = all_compound_to_pos.get(src, set())
            # 仅保留蛋白冷启动验证集中的蛋白（unseen proteins）
            valid_pos = [p - n_compounds for p in pos_set
                         if n_compounds <= p < n_compounds + n_proteins
                         and 0 <= (p - n_compounds) < n_prots_actual
                         and (p - n_compounds) in val_proteins]
            if not valid_pos:
                continue
            n_valid += 1

            scores = score_matrix[idx]

            for p in valid_pos:
                local_p = val_prot_to_local[p]
                y_true.append(1)
                y_score.append(torch.sigmoid(scores[local_p]).item())

            # 负样本：仅从 val_proteins 中采样（排除正样本 + 排除训练集蛋白）
            if n_val_prots > len(valid_pos):
                n_neg = min(10, n_val_prots - len(valid_pos))
                neg_candidates = [val_prot_to_local[p] for p in val_prot_list if p not in valid_pos]
                n_sample = min(n_neg, len(neg_candidates))
                rand_idx = torch.tensor(neg_candidates, device=DEVICE)[
                    torch.randperm(len(neg_candidates), device=DEVICE)[:n_sample]]
                for ri in rand_idx:
                    y_true.append(0)
                    y_score.append(torch.sigmoid(scores[ri]).item())

        if len(y_true) < 2 or len(set(y_true)) < 2:
            return {"auc": 0.5, "aupr": 0.5, "n_valid_compounds": n_valid}

        y_true_arr = np.array(y_true)
        y_score_arr = np.array(y_score)
        return {
            "auc": float(roc_auc_score(y_true_arr, y_score_arr)),
            "aupr": float(average_precision_score(y_true_arr, y_score_arr)),
            "n_valid_compounds": n_valid,
        }


# ============================================================
# 10. HGT 分支训练（mini-batch 邻居采样）
# ============================================================
def _validate_hgt(
    model: HGTLinkPredictor,
    hetero_data,
    val_compounds: List[int],
    all_compound_to_pos: Dict[int, set],
    n_compounds: int,
    n_proteins: int,
    hetero_adj: Optional[dict] = None,
) -> Dict[str, float]:
    """v16: HGT 全图验证 + OOM 自动降级 mini-batch

    v16 防御: 全图推理优先（保证蛋白嵌入一致性），若 CUDA OOM 则自动降级为
    mini-batch 子图采样验证。降级时蛋白嵌入不一致，但优于直接崩溃。

    参考:
      - Hu et al. (2020) "HGT" full-graph inference
    """
    model.eval()
    n_pathways = -1  # v16: 防御性初始化，防止 OOM 时机导致的 NameError
    with torch.no_grad():
        # ---- 尝试全图推理 ----
        try:
            hetero_data_dev = hetero_data.to(DEVICE)
            n_pathways = hetero_data_dev["pathway"].n_pathways
            hetero_data_dev["pathway"].x = model.pathway_embed(
                torch.arange(max(n_pathways, 1), device=DEVICE))

            x_dict_full = {k: v.clone() for k, v in hetero_data_dev.x_dict.items()}
            hgt_out = model(x_dict_full, hetero_data_dev.edge_index_dict)
            prot_emb = hgt_out["protein"]
            comp_emb = hgt_out["compound"]

            T = model.temperature

            y_true, y_score = [], []
            n_valid = 0
            for src in val_compounds:
                pos_set = all_compound_to_pos.get(src, set())
                valid_pos = [p - n_compounds for p in pos_set if n_compounds <= p < n_compounds + n_proteins]
                if not valid_pos:
                    continue
                n_valid += 1

                for p in valid_pos:
                    y_true.append(1)
                    y_score.append(torch.sigmoid(
                        model.decode(comp_emb[src:src+1], prot_emb[p:p+1]) / T
                    ).item())

                scores = model.decode(
                    comp_emb[src:src+1].expand(n_proteins, -1), prot_emb) / T

                n_hard = min(5, n_proteins - len(valid_pos))
                if n_hard > 0:
                    mask = torch.zeros(n_proteins, device=DEVICE)
                    for p in valid_pos:
                        mask[p] = -1e9
                    _, hard_indices = (scores + mask).topk(n_hard)
                    for hi in hard_indices:
                        if hi.item() < n_proteins:
                            y_true.append(0)
                            y_score.append(torch.sigmoid(scores[hi]).item())

                n_rand = min(5, n_proteins - len(valid_pos))
                if n_rand > 0:
                    rand_mask = torch.ones(n_proteins, device=DEVICE)
                    for p in valid_pos:
                        rand_mask[p] = 0
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
            return {
                "auc": float(roc_auc_score(y_true_arr, y_score_arr)),
                "aupr": float(average_precision_score(y_true_arr, y_score_arr)),
                "n_valid_compounds": n_valid,
            }

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # v16: OOM 降级 — 清除显存，改用 mini-batch 子图采样
            if "out of memory" not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
                raise
            logger.warning(f"  HGT 全图验证 OOM，降级为 mini-batch 验证 "
                           f"(graph: {n_compounds}c/{n_proteins}p/{n_pathways}w)")
            torch.cuda.empty_cache()

            if hetero_adj is None:
                logger.error("  HGT mini-batch 降级失败: hetero_adj 未传入")
                return {"auc": 0.5, "aupr": 0.5, "n_valid_compounds": 0}

            # ---- Mini-batch 降级验证 ----
            return _validate_hgt_minibatch(
                model, hetero_data, hetero_adj, val_compounds,
                all_compound_to_pos, n_compounds, n_proteins)


def _validate_hgt_minibatch(
    model: HGTLinkPredictor,
    hetero_data,
    hetero_adj: dict,
    val_compounds: List[int],
    all_compound_to_pos: Dict[int, set],
    n_compounds: int,
    n_proteins: int,
    num_neighbors: List[int] = [64, 32],
    val_batch_size: int = 64,
) -> Dict[str, float]:
    """v16: HGT mini-batch 降级验证（OOM 时自动启用）

    对验证化合物分批采样异质子图，在各子图内计算得分后全局聚合。
    注意：降级模式下蛋白嵌入在不同子图间不一致，AUC 可能偏低。
    """
    model.eval()
    with torch.no_grad():
        T = model.temperature
        all_y_true, all_y_score = [], []
        n_valid_compounds = 0

        for batch_start in range(0, len(val_compounds), val_batch_size):
            batch_seeds = val_compounds[batch_start:batch_start + val_batch_size]

            sg, comp_sorted, prot_sorted, path_sorted, comp_map, prot_map = sample_hetero_subgraph(
                batch_seeds, hetero_adj, num_neighbors, seed=42)

            if not prot_sorted:
                continue

            sg["compound"].x = hetero_data["compound"].x[torch.tensor(comp_sorted, device=DEVICE)]
            sg["protein"].x = hetero_data["protein"].x[torch.tensor(prot_sorted, device=DEVICE)]
            if path_sorted:
                path_global_tensor = torch.tensor(sg._path_global, device=DEVICE)
                path_global_tensor = torch.clamp(path_global_tensor, min=0,
                                                  max=model.pathway_embed.num_embeddings - 1)
                sg["pathway"].x = model.pathway_embed(path_global_tensor)
            else:
                sg["pathway"].x = torch.zeros(0, model.pathway_embed.embedding_dim, device=DEVICE)

            sg = sg.to(DEVICE)
            hgt_out = model(sg.x_dict, sg.edge_index_dict)
            prot_emb = hgt_out["protein"]
            comp_emb = hgt_out["compound"]
            n_batch_prots = prot_emb.shape[0]

            for bi, s in enumerate(batch_seeds):
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

                for p in valid_pos:
                    all_y_true.append(1)
                    all_y_score.append(torch.sigmoid(
                        model.decode(comp_emb[comp_local:comp_local+1], prot_emb[p:p+1]) / T
                    ).item())

                scores = model.decode(
                    comp_emb[comp_local:comp_local+1].expand(n_batch_prots, -1), prot_emb) / T

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
                    rand_candidates = torch.where(rand_mask > 0)[0]
                    if len(rand_candidates) > 0:
                        n_sample = min(n_rand, len(rand_candidates))
                        rand_idx = rand_candidates[torch.randperm(len(rand_candidates), device=DEVICE)[:n_sample]]
                        for ri in rand_idx:
                            all_y_true.append(0)
                            all_y_score.append(torch.sigmoid(scores[ri]).item())

        if len(all_y_true) < 2 or len(set(all_y_true)) < 2:
            return {"auc": 0.5, "aupr": 0.5, "n_valid_compounds": n_valid_compounds}

        y_true_arr = np.array(all_y_true)
        y_score_arr = np.array(all_y_score)
        return {
        "auc": float(roc_auc_score(y_true_arr, y_score_arr)),
        "aupr": float(average_precision_score(y_true_arr, y_score_arr)),
        "n_valid_compounds": n_valid_compounds,
    }


def _validate_hgt_protein_cold_minibatch(
    model: HGTLinkPredictor,
    hetero_data,
    hetero_adj: dict,
    val_compounds: List[int],
    all_compound_to_pos: Dict[int, set],
    n_compounds: int,
    n_proteins: int,
    val_proteins: set,
    num_neighbors: List[int] = [64, 32],
    val_batch_size: int = 64,
) -> Dict[str, float]:
    """v18: HGT 蛋白冷启动 OOM 降级 mini-batch 验证

    将验证蛋白作为孤立种子节点强制纳入子图，确保在严格隔离的蛋白冷启动图下
    仍能评估模型对未见蛋白的预测能力。
    """
    model.eval()
    val_prot_list = sorted(val_proteins)
    with torch.no_grad():
        T = model.temperature
        all_y_true, all_y_score = [], []
        n_valid_compounds = 0

        for batch_start in range(0, len(val_compounds), val_batch_size):
            batch_seeds = val_compounds[batch_start:batch_start + val_batch_size]

            sg, comp_sorted, prot_sorted, path_sorted, comp_map, prot_map = sample_hetero_subgraph(
                batch_seeds, hetero_adj, num_neighbors, seed=42,
                seed_proteins=val_prot_list)

            if not prot_sorted or not comp_sorted:
                continue

            sg["compound"].x = hetero_data["compound"].x[torch.tensor(comp_sorted, device=DEVICE)]
            sg["protein"].x = hetero_data["protein"].x[torch.tensor(prot_sorted, device=DEVICE)]
            if path_sorted:
                path_global_tensor = torch.tensor(sg._path_global, device=DEVICE)
                path_global_tensor = torch.clamp(path_global_tensor, min=0,
                                                  max=model.pathway_embed.num_embeddings - 1)
                sg["pathway"].x = model.pathway_embed(path_global_tensor)
            else:
                sg["pathway"].x = torch.zeros(0, model.pathway_embed.embedding_dim, device=DEVICE)

            sg = sg.to(DEVICE)
            hgt_out = model(sg.x_dict, sg.edge_index_dict)
            prot_emb = hgt_out["protein"]
            comp_emb = hgt_out["compound"]

            for s in batch_seeds:
                if s not in comp_map:
                    continue
                comp_local = comp_map[s]

                pos_set = all_compound_to_pos.get(s, set())
                valid_pos = []
                for p_global in pos_set:
                    p_local = p_global - n_compounds
                    # 仅评估验证蛋白
                    if p_local in val_proteins and p_local in prot_map:
                        valid_pos.append(prot_map[p_local])
                if not valid_pos:
                    continue
                n_valid_compounds += 1

                valid_pos_tensor = torch.tensor(valid_pos, device=DEVICE, dtype=torch.long)
                scores = model.decode(
                    comp_emb[comp_local:comp_local+1].expand(len(valid_pos), -1),
                    prot_emb[valid_pos_tensor]) / T
                for idx in range(len(valid_pos)):
                    all_y_true.append(1)
                    all_y_score.append(torch.sigmoid(scores[idx]).item())

                # 仅从验证蛋白中采样负样本
                n_neg = min(10, len(val_prot_list) - len(valid_pos))
                if n_neg > 0:
                    candidate_mask = torch.zeros(len(prot_sorted), dtype=torch.bool, device=DEVICE)
                    for vp in val_prot_list:
                        if vp in prot_map:
                            candidate_mask[prot_map[vp]] = True
                    for p in valid_pos:
                        candidate_mask[p] = False
                    rand_candidates = torch.where(candidate_mask)[0]
                    if len(rand_candidates) > 0:
                        n_sample = min(n_neg, len(rand_candidates))
                        rand_idx = rand_candidates[torch.randperm(len(rand_candidates), device=DEVICE)[:n_sample]]
                        scores = model.decode(
                            comp_emb[comp_local:comp_local+1].expand(len(rand_idx), -1),
                            prot_emb[rand_idx]) / T
                        for idx in range(len(rand_idx)):
                            all_y_true.append(0)
                            all_y_score.append(torch.sigmoid(scores[idx]).item())

        if len(all_y_true) < 2 or len(set(all_y_true)) < 2:
            return {"auc": 0.5, "aupr": 0.5, "n_valid_compounds": n_valid_compounds}

        y_true_arr = np.array(all_y_true)
        y_score_arr = np.array(all_y_score)
        return {
            "auc": float(roc_auc_score(y_true_arr, y_score_arr)),
            "aupr": float(average_precision_score(y_true_arr, y_score_arr)),
            "n_valid_compounds": n_valid_compounds,
        }


def _validate_hgt_protein_cold(
    model: HGTLinkPredictor,
    hetero_data,
    val_compounds: List[int],
    all_compound_to_pos: Dict[int, set],
    n_compounds: int,
    n_proteins: int,
    val_proteins: set,
    hetero_adj: Optional[dict] = None,
) -> Dict[str, float]:
    """v18: HGT 蛋白冷启动验证 — 仅评估对未见蛋白的预测能力"""
    model.eval()
    n_pathways = -1
    with torch.no_grad():
        try:
            hetero_data_dev = hetero_data.to(DEVICE)
            n_pathways = hetero_data_dev["pathway"].n_pathways
            hetero_data_dev["pathway"].x = model.pathway_embed(
                torch.arange(max(n_pathways, 1), device=DEVICE))
            x_dict = {k: v.clone() for k, v in hetero_data_dev.x_dict.items()}
            # v18-fix: 直接使用传入的严格隔离异质图（hetero_data_prot_cold）。
            # 该图已在外部移除验证化合物/验证蛋白相关的所有 CPI/PPI/通路边，
            # 同时保留训练化合物与训练蛋白之间的 CPI 边，避免内部再次置空导致信息损失。
            out = model(x_dict, hetero_data_dev.edge_index_dict)
            prot_emb = out["protein"]
            comp_emb = out["compound"]
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # v17: 仅捕获 OOM 相关异常，其他错误（如维度不匹配）应直接暴露
            if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                raise
            logger.warning(f"  HGT 蛋白冷启动 OOM (graph: {n_compounds}c/{n_proteins}p/{n_pathways}w): {e}")
            torch.cuda.empty_cache()

            # v18: OOM 降级 — 使用蛋白冷启动专用邻接表进行 mini-batch 采样
            if hetero_adj is not None:
                logger.info("  HGT 蛋白冷启动 OOM，降级为 mini-batch 验证")
                return _validate_hgt_protein_cold_minibatch(
                    model, hetero_data, hetero_adj, val_compounds,
                    all_compound_to_pos, n_compounds, n_proteins, val_proteins)
            return {"auc": 0.5, "aupr": 0.5, "n_valid_compounds": 0}

        T = model.temperature
        y_true, y_score = [], []
        n_valid = 0
        for src in val_compounds:
            if src >= n_compounds:
                continue
            pos_set = all_compound_to_pos.get(src, set())
            valid_pos = [p - n_compounds for p in pos_set
                         if n_compounds <= p < n_compounds + n_proteins
                         and (p - n_compounds) in val_proteins]
            if not valid_pos:
                continue
            n_valid += 1

            valid_pos_tensor = torch.tensor(valid_pos, device=DEVICE, dtype=torch.long)
            scores = model.decode(comp_emb[src:src+1].expand(len(valid_pos), -1), prot_emb[valid_pos_tensor]) / T
            for idx in range(len(valid_pos)):
                y_true.append(1)
                y_score.append(torch.sigmoid(scores[idx]).item())

            val_prot_list = sorted(val_proteins)
            if len(val_prot_list) > len(valid_pos):
                n_neg = min(10, len(val_prot_list) - len(valid_pos))
                # v17: 仅从 val_proteins 中采样负样本（排除正样本）
                candidate_mask = torch.zeros(n_proteins, dtype=torch.bool, device=DEVICE)
                for vp in val_proteins:
                    if 0 <= vp < n_proteins:
                        candidate_mask[vp] = True
                for p in valid_pos:
                    if 0 <= p < n_proteins:
                        candidate_mask[p] = False
                rand_candidates = torch.where(candidate_mask)[0]
                if len(rand_candidates) > 0:
                    n_sample = min(n_neg, len(rand_candidates))
                    rand_idx = rand_candidates[torch.randperm(len(rand_candidates), device=DEVICE)[:n_sample]]
                    scores = model.decode(comp_emb[src:src+1].expand(len(rand_idx), -1), prot_emb[rand_idx]) / T
                    for idx in range(len(rand_idx)):
                        y_true.append(0)
                        y_score.append(torch.sigmoid(scores[idx]).item())

    if len(y_true) < 2 or len(set(y_true)) < 2:
        return {"auc": 0.5, "aupr": 0.5, "n_valid_compounds": n_valid}

    y_true_arr = np.array(y_true)
    y_score_arr = np.array(y_score)
    return {
        "auc": float(roc_auc_score(y_true_arr, y_score_arr)),
        "aupr": float(average_precision_score(y_true_arr, y_score_arr)),
        "n_valid_compounds": n_valid,
    }


def train_hgt(
    model: HGTLinkPredictor,
    graphs: dict,
    train_compounds: List[int],
    val_compounds: List[int],
    compound_to_pos: Dict[int, set],
    val_proteins: set = None,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15,
    batch_size: int = 128,
    num_neighbors: List[int] = [32, 16],
    prot_to_path_neighbors: Optional[Dict[int, set]] = None,
    flag_step: float = 0.01,
) -> Tuple[HGTLinkPredictor, List[dict]]:
    """v17: HGT mini-batch 训练 — DropEdge + FocalLoss + 蛋白冷启动验证 + Memory Bank"""
    model = model.to(DEVICE)
    for p in model.parameters():
        if p.dim() >= 2:
            nn.init.xavier_uniform_(p)

    # v18: 使用训练安全异质邻接表，验证蛋白在训练阶段完全不可见
    hetero_adj = graphs.get("hetero_adj_train", graphs["hetero_adj"])
    # v18: Memory Bank 与 mini-batch 特征查找均使用训练安全图结构
    hetero_data = graphs.get("hetero_data_train", graphs["hetero_data"]).to(DEVICE)
    n_compounds = graphs["n_compounds"]
    n_proteins = graphs["n_proteins"]
    n_pathways = graphs["n_pathways"]

    logger.info(f"  HGT 通路嵌入: {max(n_pathways, 1)} 通路, dim={model.pathway_embed.embedding_dim}")

    all_compound_to_pos = compound_to_pos
    precomputed_pos = {src: sorted(pos_set) for src, pos_set in compound_to_pos.items() if pos_set}

    # v17: AdamW 解耦权重衰减与自适应学习率
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # v17: LR Warmup + Cosine Annealing（前5%线性预热，后95%余弦退火至1e-6）
    warmup_epochs = max(1, int(epochs * 0.05))
    def lr_lambda(e):
        if e < warmup_epochs:
            return e / warmup_epochs
        progress = (e - warmup_epochs) / (epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress)) * 1.0 + 1e-6
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # v17: Memory Bank — 存储跨 batch 蛋白嵌入，供全局困难负样本采样
    memory_bank = MemoryBank(max_size=8192, out_dim=model.out_dim, device=DEVICE)
    # v17: 多指标联合早停
    best_val_auc = 0.0
    best_val_aupr = 0.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        random.shuffle(train_compounds)
        for batch_start in range(0, len(train_compounds), batch_size):
            batch_seeds = train_compounds[batch_start:batch_start + batch_size]

            sg, comp_sorted, prot_sorted, path_sorted, comp_map, prot_map = sample_hetero_subgraph(
                batch_seeds, hetero_adj, num_neighbors,
                seed=epoch * 10000 + batch_start)

            if not prot_sorted:
                continue

            # 填充节点特征
            sg["compound"].x = hetero_data["compound"].x[torch.tensor(comp_sorted, device=DEVICE)]
            sg["protein"].x = hetero_data["protein"].x[torch.tensor(prot_sorted, device=DEVICE)]
            # v17: Gaussian Feature Augmentation
            if flag_step > 0:
                sg["compound"].x = sg["compound"].x + flag_step * torch.randn_like(sg["compound"].x)
                sg["protein"].x = sg["protein"].x + flag_step * torch.randn_like(sg["protein"].x)
                sg["compound"].x = sg["compound"].x.detach()
                sg["protein"].x = sg["protein"].x.detach()
            # v16: 通路嵌入 batch 级直接调用 model.pathway_embed，杜绝参数-特征不同步
            if path_sorted:
                path_global_tensor = torch.tensor(sg._path_global, device=DEVICE)
                path_global_tensor = torch.clamp(path_global_tensor, min=0,
                                                  max=model.pathway_embed.num_embeddings - 1)
                sg["pathway"].x = model.pathway_embed(path_global_tensor)
            else:
                sg["pathway"].x = torch.zeros(0, model.pathway_embed.embedding_dim, device=DEVICE)

            sg = sg.to(DEVICE)
            # v17: DropEdge 按边类型丢弃 — PPI边15%, 通路边10%, CPI边保留
            for et in list(sg.edge_index_dict.keys()):
                if "ppi" in str(et):
                    sg[et].edge_index = drop_edge(sg[et].edge_index, p=0.15)
                elif "pathway" in str(et) or "belongs_to" in str(et) or "includes" in str(et):
                    sg[et].edge_index = drop_edge(sg[et].edge_index, p=0.10)
                # CPI边 (interacts) 不丢弃，保留交互完整性

            optimizer.zero_grad()

            hgt_out = model(sg.x_dict, sg.edge_index_dict)
            prot_emb = hgt_out["protein"]
            comp_emb = hgt_out["compound"]

            if torch.isnan(prot_emb).any() or torch.isnan(comp_emb).any():
                continue

            cpi_ei = sg[("compound", "interacts", "protein")].edge_index
            if cpi_ei.shape[1] < 1:
                continue

            n_batch_prots = prot_emb.shape[0]
            T = model.temperature

            # 正样本
            pos_src = cpi_ei[0]
            pos_dst = cpi_ei[1]
            pos_score = model.decode(comp_emb[pos_src], prot_emb[pos_dst]) / T
            pos_score = torch.clamp(pos_score, -10, 10)

            # v17: 课程负采样（先易后难三阶段）
            # Phase 1: 前30% epochs → 仅随机负样本
            # Phase 2: 中40% epochs → 随机70% + 中度30%
            # Phase 3: 后30% epochs → 随机90% + 极硬10%
            unique_src = pos_src.unique()
            n_unique = len(unique_src)
            if n_unique > 0 and n_batch_prots > 1:
                batch_comp_emb = comp_emb[unique_src]
                all_scores = model.decode(
                    batch_comp_emb.unsqueeze(1).expand(-1, n_batch_prots, -1).reshape(-1, model.out_dim),
                    prot_emb.repeat(n_unique, 1)
                ).reshape(n_unique, n_batch_prots) / T

                # 正样本 mask
                mask = torch.zeros(n_unique, n_batch_prots, device=DEVICE)
                for i, src_local in enumerate(unique_src):
                    src_global = comp_sorted[src_local.item()] if src_local.item() < len(comp_sorted) else -1
                    if src_global >= 0 and src_global in precomputed_pos:
                        for p_global in precomputed_pos[src_global]:
                            p_local = p_global - n_compounds
                            if p_local in prot_map:
                                p_idx = prot_map[p_local]
                                if p_idx < n_batch_prots:
                                    mask[i, p_idx] = -1e9

                # 课程阶段判定
                curriculum_phase = epoch / epochs
                if curriculum_phase < 0.3:
                    n_medium = n_hard = 0
                elif curriculum_phase < 0.7:
                    n_medium = int(n_unique * 0.3)
                    n_hard = 0
                else:
                    n_hard = int(n_unique * 0.1)
                    n_medium = 0

                # 初始化 hard_neg_scores 为随机负样本（排除正样本 + 保护全零行）
                hard_neg_scores = torch.zeros(n_unique, device=DEVICE)
                valid_mask = (mask == 0).float()
                row_sum = valid_mask.sum(dim=1)
                safe_rows = row_sum > 0
                if safe_rows.any():
                    valid_mask = valid_mask / (row_sum.unsqueeze(1) + 1e-10)
                    rand_dst = torch.multinomial(valid_mask, 1).squeeze(-1)
                    hard_neg_scores[safe_rows] = model.decode(comp_emb[unique_src[safe_rows]], prot_emb[rand_dst[safe_rows]]) / T
                    hard_neg_scores = torch.clamp(hard_neg_scores, -10, 10)
                # 全零行保持 hard_neg_scores=0

                # Phase 2: 中度负样本
                if n_medium > 0 and prot_to_path_neighbors is not None and n_batch_prots > 2:
                    medium_neg_scores = hard_neg_scores.clone()
                    medium_found = torch.zeros(n_unique, dtype=torch.bool, device=DEVICE)
                    for i, src_local in enumerate(unique_src):
                        src_global = comp_sorted[src_local.item()] if src_local.item() < len(comp_sorted) else -1
                        if src_global < 0 or src_global not in precomputed_pos:
                            continue
                        path_neighbors: set = set()
                        for p_global in precomputed_pos[src_global]:
                            p_local = p_global - n_compounds
                            if p_local >= 0 and p_local in prot_to_path_neighbors:
                                path_neighbors.update(prot_to_path_neighbors[p_local])
                        if not path_neighbors:
                            continue
                        batch_neighbor_positions = []
                        for pn in path_neighbors:
                            if pn in prot_map:
                                pi = prot_map[pn]
                                if 0 <= pi < n_batch_prots and mask[i, pi] == 0:
                                    batch_neighbor_positions.append(pi)
                        if batch_neighbor_positions:
                            bi_t = torch.tensor(batch_neighbor_positions, device=DEVICE)
                            neighbor_scores = all_scores[i, bi_t]
                            best_idx = neighbor_scores.argmax()
                            medium_neg_scores[i] = torch.clamp(neighbor_scores[best_idx], -10, 10)
                            medium_found[i] = True

                    medium_candidates = torch.where(medium_found)[0]
                    if len(medium_candidates) > 0:
                        n_actual = min(n_medium, len(medium_candidates))
                        perm = torch.randperm(len(medium_candidates), device=DEVICE)
                        hard_neg_scores[medium_candidates[perm[:n_actual]]] = medium_neg_scores[medium_candidates[perm[:n_actual]]]

                # Phase 3: 极硬负样本
                if n_hard > 0:
                    hard_neg_idx = (all_scores + mask).argmax(dim=1)
                    hard_scores = all_scores[torch.arange(n_unique, device=DEVICE), hard_neg_idx]
                    hard_scores = torch.clamp(hard_scores, -10, 10)
                    hard_candidates = torch.randperm(n_unique, device=DEVICE)[:n_hard]
                    hard_neg_scores[hard_candidates] = hard_scores[hard_candidates]

                # v18: 固定 Focal alpha = 0.75，避免动态 alpha 导致负样本权重过小
                pos_loss = focal_loss_with_logits(
                    pos_score, torch.full_like(pos_score, 0.9), alpha=0.75)
                neg_loss = focal_loss_with_logits(
                    hard_neg_scores, torch.full_like(hard_neg_scores, 0.1), alpha=0.75)

                src_to_pos = {s.item(): i for i, s in enumerate(unique_src)}
                pos_indices = torch.tensor(
                    [src_to_pos[s.item()] for s in pos_src],
                    device=DEVICE, dtype=torch.long
                )

                # ---- v18: 向量化 BPR 损失 — 为每个正样本对独立采样负样本 ----
                pair_mask = mask[pos_indices]  # (n_pos, n_batch_prots)
                bpr_valid_mask = (pair_mask == 0).float()
                bpr_row_sum = bpr_valid_mask.sum(dim=1)
                bpr_safe = bpr_row_sum > 0
                bpr_neg_scores = torch.zeros(len(pos_src), device=DEVICE)
                if bpr_safe.any():
                    bpr_valid_mask[bpr_safe] = bpr_valid_mask[bpr_safe] / bpr_row_sum[bpr_safe].unsqueeze(1)
                    bpr_neg_dst = torch.multinomial(bpr_valid_mask, 1).squeeze(-1)
                    bpr_neg_scores[bpr_safe] = all_scores[pos_indices[bpr_safe], bpr_neg_dst[bpr_safe]]
                bpr_loss = -torch.log(torch.sigmoid(pos_score - bpr_neg_scores) + 1e-8).mean()

                loss = 0.6 * (pos_loss + neg_loss) + 0.4 * bpr_loss

                # ---- v17: Memory Bank InfoNCE 对比损失（epoch > 50 启用） ----
                if epoch > 50 and memory_bank.size() > 0 and len(pos_indices) > 0:
                    n_mem = min(256, memory_bank.size())
                    mem_emb = memory_bank.sample(n_mem)
                    if mem_emb.shape[0] > 0:
                        pos_idx_sub = pos_indices[:len(pos_score)]
                        # v17: InfoNCE 需要原始 logits（未缩放），内部 temperature=0.07 自行缩放
                        # pos_score/hard_neg_scores 已除以 T=5.0，需乘以 T 还原为原始 logits
                        mem_scores = model.decode(
                            comp_emb[unique_src[pos_idx_sub]].unsqueeze(1).expand(-1, n_mem, -1).reshape(-1, model.out_dim),
                            mem_emb.repeat(len(pos_idx_sub), 1)
                        ).reshape(len(pos_idx_sub), n_mem)  # raw, no /T
                        infonce = infonce_loss(
                            pos_score[:len(pos_idx_sub)] * T,
                            hard_neg_scores[pos_idx_sub] * T,
                            memory_scores=mem_scores, temperature=0.07,
                        )
                        loss = loss + 0.1 * infonce
            else:
                loss = pos_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # v17: 更新 Memory Bank
            memory_bank.update(prot_emb.detach())

            total_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            continue

        avg_loss = total_loss / n_batches

        # v18-short: 每 2 epoch 验证一次，短训练下更快观察指标
        if epoch % 2 == 0 and val_compounds:
            torch.cuda.empty_cache()
            # v17: 使用验证安全异质图（无验证集 CPI 边），防止冷启动信息泄露
            val_hetero = graphs.get("hetero_data_val")
            if val_hetero is not None:
                val_hetero = val_hetero.to(DEVICE)
            else:
                val_hetero = hetero_data
            # v18: 验证时使用验证安全邻接表，避免 OOM 降级采样引入训练边
            val_hetero_adj = graphs.get("hetero_adj_val", hetero_adj)
            val_metrics = _validate_hgt(
                model, val_hetero, val_compounds,
                all_compound_to_pos, n_compounds, n_proteins,
                hetero_adj=val_hetero_adj)
            val_auc = val_metrics["auc"]
            val_aupr = val_metrics["aupr"]

            # v17: 多指标联合早停（AUPR 为主）
            if val_aupr > best_val_aupr:
                best_val_aupr = val_aupr
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if val_auc > best_val_auc:
                best_val_auc = val_auc

            # v17: 蛋白冷启动验证
            if val_proteins is not None and len(val_proteins) > 0:
                # v18: 蛋白冷启动使用严格隔离图
                prot_hetero = graphs.get("hetero_data_prot_cold", val_hetero).to(DEVICE)
                prot_cold = _validate_hgt_protein_cold(
                    model, prot_hetero, val_compounds, all_compound_to_pos,
                    n_compounds, n_proteins, val_proteins,
                    hetero_adj=val_hetero_adj,
                )
                prot_auc = prot_cold["auc"]
                prot_aupr = prot_cold["aupr"]
            else:
                prot_auc = prot_aupr = 0.5

            history.append({"epoch": epoch, "loss": avg_loss, "auc": val_auc, "aupr": val_aupr,
                         "prot_auc": prot_auc, "prot_aupr": prot_aupr})
            logger.info(f"  HGT epoch {epoch:3d} | loss={avg_loss:.4f} | val_auc={val_auc:.4f} | val_aupr={val_aupr:.4f} | "
                        f"prot_auc={prot_auc:.4f} | prot_aupr={prot_aupr:.4f}")

            if patience_counter >= patience:
                logger.info(f"  HGT 早停 (epoch {epoch}, patience_counter={patience_counter})")
                break

            # v17: Memory Bank 全局刷新 — 每5 epoch 全图前向，填充完整蛋白嵌入
            # v18: 过滤掉验证蛋白，避免验证信息进入 bank
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    hetero_data_dev = hetero_data.to(DEVICE)
                    n_path = hetero_data_dev["pathway"].n_pathways
                    hetero_data_dev["pathway"].x = model.pathway_embed(
                        torch.arange(max(n_path, 1), device=DEVICE))
                    x_dict_full = {k: v.clone() for k, v in hetero_data_dev.x_dict.items()}
                    hgt_out = model(x_dict_full, hetero_data_dev.edge_index_dict)
                    full_prot_emb = hgt_out["protein"]
                    # 排除验证蛋白嵌入
                    if val_proteins is not None and len(val_proteins) > 0:
                        train_prot_mask = torch.ones(full_prot_emb.shape[0], dtype=torch.bool, device=DEVICE)
                        train_prot_mask[list(val_proteins)] = False
                        full_prot_emb = full_prot_emb[train_prot_mask]
                    memory_bank = MemoryBank(max_size=8192, out_dim=model.out_dim, device=DEVICE)
                    memory_bank.update(full_prot_emb)
                logger.info(f"  HGT Memory Bank 全局刷新: {memory_bank.size()} 训练蛋白嵌入")
                model.train()

        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    best_entry = max(history, key=lambda x: x["auc"]) if history else {"auc": 0.0}
    logger.info(f"  HGT best val_auc={best_entry['auc']:.4f}")
    return model, history


# ============================================================
# 11. 预测与集成
# ============================================================
def predict_tcm(
    sage_model: SAGELinkPredictor,
    hgt_model: Optional[HGTLinkPredictor],
    graphs: dict,
    tcm_smiles: List[str],
    target_genes: List[str],
    compound_stats: Tuple,
    sage_prot_aupr: float = 0.5,
    hgt_prot_aupr: float = 0.5,
    diversity_penalty: float = 0.1,
    mc_samples: int = 0,
) -> pd.DataFrame:
    """v17: SAGE + HGT 集成预测 — 动态权重 + 多样性约束 + MC Dropout

    v17 改进:
      - 基于蛋白冷启动 AUPR 动态调整 SAGE/HGT 权重
      - 余弦相似度多样性惩罚：鼓励两个分支利用不同信号
      - MC Dropout 不确定性估计：mc_samples>0 时保持 Dropout 开启，
        重复 mc_samples 次前向，输出均值 + 标准差
      - 参考: Zhou et al. (2021) "Diver";
              Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"

    Args:
        sage_prot_aupr: SAGE 蛋白冷启动 AUPR（v17: 来自最终模型重新评估）
        hgt_prot_aupr: HGT 蛋白冷启动 AUPR（v17: 来自最终模型重新评估）
        diversity_penalty: 余弦相似度惩罚系数（0~1，越大越惩罚相似预测）
        mc_samples: MC Dropout 采样次数（0=禁用，推荐30）
    """
    n_iterations = max(1, mc_samples)
    use_mc = mc_samples > 0

    if use_mc:
        sage_model.train()  # v17: 保持 Dropout 开启，无梯度
        if hgt_model is not None:
            hgt_model.train()
    else:
        sage_model.eval()
        if hgt_model is not None:
            hgt_model.eval()

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

    # v17: 动态集成权重 — 基于蛋白冷启动 AUPR
    total_aupr = sage_prot_aupr + hgt_prot_aupr
    if total_aupr > 0:
        sage_w = sage_prot_aupr / total_aupr
        hgt_w = hgt_prot_aupr / total_aupr
    else:
        sage_w = hgt_w = 0.5
    logger.info(f"  集成权重: SAGE={sage_w:.3f} (prot_aupr={sage_prot_aupr:.3f}), "
                f"HGT={hgt_w:.3f} (prot_aupr={hgt_prot_aupr:.3f})")

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

    valid_gene_indices = [j for j, (_, lp) in enumerate(gene_index_map) if lp >= 0]

    all_sage_scores_mc = []  # (n_iter, n_tcm, n_genes)
    all_hgt_scores_mc = []
    all_final_scores_mc = []  # (n_iter, n_tcm, n_genes) — diversity-adjusted

    for it in range(n_iterations):
        with torch.no_grad():
            # SAGE: 原生归纳式推理
            # v17-ESM2: 分别处理 — 全图编码蛋白嵌入 + encode_compound 编码 TCM 化合物
            edge_index = homo_edge_index.to(DEVICE)
            node_emb = sage_model(x_dev, edge_index)  # 全图（原化合物+蛋白）
            sage_prot_emb = node_emb[n_compounds:]
            sage_tcm_emb = sage_model.encode_compound(tcm_feat)  # TCM 化合物（无CPI边，仅投影+卷积）
            sage_T = sage_model.temperature

            # SAGE 向量化评分: (n_tcm, n_prots) — v18: MLP 解码器
            n_tcm_sage = sage_tcm_emb.shape[0]
            n_prots_all = sage_prot_emb.shape[0]
            sage_tcm_exp = sage_tcm_emb.unsqueeze(1).expand(-1, n_prots_all, -1).reshape(-1, sage_tcm_emb.shape[-1])
            sage_prot_exp = sage_prot_emb.unsqueeze(0).expand(n_tcm_sage, -1, -1).reshape(-1, sage_prot_emb.shape[-1])
            sage_all_scores = torch.sigmoid(
                sage_model.decode(sage_tcm_exp, sage_prot_exp) / sage_T
            ).reshape(n_tcm_sage, n_prots_all)

            # HGT: 全图推理
            if hgt_model is not None:
                hetero_data = graphs["hetero_data"].to(DEVICE)
                n_pathways = graphs["n_pathways"]
                hetero_data["pathway"].x = hgt_model.pathway_embed(
                    torch.arange(max(n_pathways, 1), device=DEVICE))
                x_dict_full = {k: v.clone() for k, v in hetero_data.x_dict.items()}
                hgt_out = hgt_model(x_dict_full, hetero_data.edge_index_dict)
                hgt_prot_emb = hgt_out["protein"]
                hgt_tcm_emb = hgt_model.encode_compound(tcm_feat)
                hgt_T = hgt_model.temperature

                # HGT 向量化双线性评分: (n_tcm, n_prots)
                n_tcm = hgt_tcm_emb.shape[0]
                n_prots_all = hgt_prot_emb.shape[0]
                hgt_tcm_exp = hgt_tcm_emb.unsqueeze(1).expand(-1, n_prots_all, -1).reshape(-1, hgt_tcm_emb.shape[-1])
                hgt_prot_exp = hgt_prot_emb.unsqueeze(0).expand(n_tcm, -1, -1).reshape(-1, hgt_prot_emb.shape[-1])
                hgt_all_scores = torch.sigmoid(
                    hgt_model.decode(hgt_tcm_exp, hgt_prot_exp) / hgt_T
                ).reshape(n_tcm, n_prots_all)
            else:
                hgt_all_scores = None

            # 提取目标基因的分数
            iter_sage = torch.full((len(tcm_smiles), len(target_genes)), 0.5, device=DEVICE)
            iter_hgt = torch.full((len(tcm_smiles), len(target_genes)), 0.5, device=DEVICE)
            for j, local_p_idx in [(j, lp) for j, (_, lp) in enumerate(gene_index_map) if lp >= 0]:
                iter_sage[:, j] = sage_all_scores[:, local_p_idx]
                if hgt_all_scores is not None:
                    iter_hgt[:, j] = hgt_all_scores[:, local_p_idx]
            all_sage_scores_mc.append(iter_sage.cpu())
            all_hgt_scores_mc.append(iter_hgt.cpu())

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

    # v17: 多样性约束 — 在分支均值上应用
    delta = torch.abs(sage_mean - hgt_mean)  # (n_tcm, n_genes)
    diversity_factor = 1.0 - diversity_penalty * (1.0 - delta)
    weighted_scores = sage_w * sage_mean + hgt_w * hgt_mean
    final_scores = weighted_scores * diversity_factor + 0.5 * (1.0 - diversity_factor)

    # v17: 全局分支余弦相似度报告
    sage_vec = sage_mean.flatten()
    hgt_vec = hgt_mean.flatten()
    cos_sim = F.cosine_similarity(sage_vec.unsqueeze(0), hgt_vec.unsqueeze(0)).item()
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


# ============================================================
# 12. 管线自检
# ============================================================
def pipeline_self_check(tcm_df, cpi_df, ppi_df, prot_feat, gene_to_pathways, warm_targets):
    logger.info("=" * 60)
    logger.info("开始管线自检...")
    results = {"errors": [], "warnings": []}

    tcm_smiles = tcm_df["SMILES_std"].dropna().tolist()
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
        results["warnings"].append(f"TCM/训练集重叠: {len(overlap)} 个化合物")

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


# ============================================================
# 13. 主流程
# ============================================================
def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Phase 4 v18: SAGE + HGT Mini-Batch — 拓扑-语义双视角互补融合")
    logger.info("v18: 分离验证图 / FocalLoss+LabelSmoothing / DropEdge / 蛋白冷启动 / 课程负采样")
    logger.info("=" * 60)

    # 加载数据
    logger.info(">>> 加载数据")
    cpi_df = load_cpi_data()
    ppi_df = load_ppi_network()
    gene_to_pathways = load_kegg_pathways()
    prot_feat, gene_to_seq = load_protein_features()  # v17: 启用 ESM-2 预训练蛋白嵌入 (640维), hf-mirror.com 镜像下载
    tcm_df = load_tcm_pool()

    warm_targets = sorted(set(cpi_df["gene"].unique()) & set(ALL_FERRORAGING_GENES))
    logger.info(f"温靶标: {len(warm_targets)} 个")

    check_results = pipeline_self_check(tcm_df, cpi_df, ppi_df, prot_feat, gene_to_pathways, warm_targets)
    if check_results["overall"] == "FAILED":
        logger.error("管线自检未通过，终止训练")
        sys.exit(1)

    cpi_df = cpi_df[cpi_df["gene"].isin(warm_targets)].copy()

    # 构建图 & 邻接表
    logger.info(">>> 构建图 & 邻接表")
    graphs = build_graphs_and_adj(cpi_df, ppi_df, gene_to_pathways, prot_feat)

    # v17: 双重冷启动拆分 — 化合物 + 蛋白
    all_compounds = sorted(graphs["smi_to_idx"].values())
    all_proteins = sorted(set(
        graphs["gene_to_idx"][g] - graphs["n_compounds"]
        for g in graphs["gene_to_idx"]
        if graphs["gene_to_idx"][g] >= graphs["n_compounds"]
    ))
    random.shuffle(all_compounds)
    random.shuffle(all_proteins)

    # 化合物冷启动: 85% train / 15% val
    n_train_comp = int(len(all_compounds) * 0.85)
    train_compounds = all_compounds[:n_train_comp]
    val_compounds = all_compounds[n_train_comp:]

    # v18-fix: 蛋白冷启动分层拆分 — 确保验证集包含足够有CPI交互的蛋白，
    # 避免验证集正样本蛋白过少导致 prot_aupr 评估失真（原随机拆分可能导致
    # 全部CPI蛋白落入训练集，使验证任务退化为检测单一异常蛋白）。
    cpi_proteins = set()
    for _, row in cpi_df.iterrows():
        gene = row["gene"]
        if gene in graphs["gene_to_idx"]:
            cpi_proteins.add(graphs["gene_to_idx"][gene] - graphs["n_compounds"])
    non_cpi_proteins = [p for p in all_proteins if p not in cpi_proteins]

    n_val_cpi = max(1, int(len(cpi_proteins) * 0.20))
    n_train_cpi = len(cpi_proteins) - n_val_cpi
    n_val_non_cpi = max(1, int(len(non_cpi_proteins) * 0.20))
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
                graphs["gene_to_idx"][gene])  # v13: 全局索引，不做减法

    # v17: 统计无效化合物
    n_val_no_pos = sum(1 for c in val_compounds if c not in compound_to_pos or len(compound_to_pos[c]) == 0)
    n_train_no_pos = sum(1 for c in train_compounds if c not in compound_to_pos or len(compound_to_pos[c]) == 0)
    logger.info(f"冷启动拆分: {len(train_compounds)} train ({n_train_no_pos} 无正样本) / "
                f"{len(val_compounds)} val ({n_val_no_pos} 无正样本) 化合物")
    n_val_cpi_actual = sum(1 for p in val_proteins if p in cpi_proteins)
    logger.info(f"蛋白冷启动: {len(train_proteins)} train / {len(val_proteins)} val 蛋白 "
                f"(CPI蛋白: {len(cpi_proteins)} 总, {n_val_cpi_actual} 在验证集)")

    # v18: 分离化合物冷启动与蛋白冷启动验证图
    val_comp_set = set(val_compounds)
    # 化合物冷启动验证图：仅移除验证集化合物的 CPI 边，保留蛋白侧拓扑
    graphs["homo_edge_index_val"] = _build_val_comp_cold_homo_edge_index(
        graphs["homo_edge_index"], val_comp_set)
    graphs["hetero_data_val"] = _build_val_comp_cold_hetero_data(
        graphs["hetero_data"], val_comp_set)
    # 蛋白冷启动验证图：严格移除验证集化合物 + 验证集蛋白的所有边
    graphs["homo_edge_index_prot_cold"] = _build_val_safe_homo_edge_index(
        graphs["homo_edge_index"], graphs["n_compounds"], val_comp_set, val_proteins)
    graphs["hetero_data_prot_cold"] = _build_val_safe_hetero_data(
        graphs["hetero_data"], val_comp_set, val_proteins)

    # v18: 构建训练安全邻接表 — 训练阶段完全隐藏验证蛋白，杜绝 PPI 网络信息泄露
    graphs["homo_adj_train"] = _build_train_safe_homo_adj(
        graphs["homo_adj"], graphs["n_compounds"], val_comp_set, val_proteins)
    graphs["homo_cpi_adj_train"] = _build_train_safe_homo_cpi_adj(
        graphs["homo_cpi_adj"], val_comp_set, val_proteins)
    graphs["hetero_adj_train"] = _build_train_safe_hetero_adj(
        graphs["hetero_adj"], graphs["n_compounds"], val_comp_set, val_proteins)
    # v18: 验证安全异质邻接表（用于 HGT OOM 降级 mini-batch 验证）
    graphs["hetero_adj_val"] = _build_val_safe_hetero_adj(
        graphs["hetero_adj"], graphs["n_compounds"], val_comp_set, val_proteins)
    # v18: 蛋白冷启动专用异质邻接表（最终重新评估与 OOM 降级均使用严格隔离图）
    graphs["hetero_adj_prot_cold"] = _build_val_safe_hetero_adj(
        graphs["hetero_adj"], graphs["n_compounds"], val_comp_set, val_proteins)
    # v18: Memory Bank 全局刷新也使用训练安全图，避免验证蛋白嵌入进入 bank
    graphs["homo_edge_index_train"] = _build_val_safe_homo_edge_index(
        graphs["homo_edge_index"], graphs["n_compounds"], val_comp_set, val_proteins)
    graphs["hetero_data_train"] = _build_val_safe_hetero_data(
        graphs["hetero_data"], val_comp_set, val_proteins)
    logger.info(f"  训练安全邻接表已构建: SAGE {sum(len(v) for v in graphs['homo_adj_train'].values())} 条边, "
                f"HGT {sum(len(v) for v in graphs['hetero_adj_train'].values())} 条边")

    # ======== 训练 SAGE ========
    logger.info(">>> 训练 SAGE（v18: SAGEConv + DropEdge + FocalLoss + 蛋白冷启动验证）")
    sage_model = SAGELinkPredictor(
        comp_feat_dim=graphs["feat_dim"], prot_feat_dim=graphs["prot_esm_dim"],  # v17-ESM2: 传 ESM-2 维度（640），通路独立投影
        n_compounds=graphs["n_compounds"],
        hidden_dim=64, out_dim=64, num_layers=2, dropout=0.5,
        n_pathways=graphs["n_pathways"])
    sage_model, sage_history = train_sage(
        sage_model, graphs, train_compounds, val_compounds, compound_to_pos,
        val_proteins=val_proteins,
        epochs=15, lr=5e-4, patience=5, batch_size=256, num_neighbors=[32, 16],  # v18-short: 短训练快速观察指标
        prot_to_path_neighbors=graphs.get("prot_to_path_neighbors"))

    torch.cuda.empty_cache()
    logger.info("  SAGE GPU 内存已释放")

    # ======== 训练 HGT ========
    logger.info(">>> 训练 HGT（v18: HGTConv + DropEdge + FocalLoss + 蛋白冷启动验证）")
    hgt_node_feat_dims = {
        "compound": graphs["feat_dim"],
        "protein": graphs["prot_esm_dim"],  # v17-ESM2: 使用 ESM-2 维度（640），通路信息由异质图结构传递
        "pathway": 1,
        "pathway_count": graphs["n_pathways"],
    }
    hgt_model = HGTLinkPredictor(
        hidden_dim=64, out_dim=64, num_heads=2, num_layers=2,
        dropout=0.5, metadata=graphs["hetero_data"].metadata(),
        compound_feat_dim=graphs["feat_dim"], node_feat_dims=hgt_node_feat_dims)
    hgt_model, hgt_history = train_hgt(
        hgt_model, graphs, train_compounds, val_compounds, compound_to_pos,
        val_proteins=val_proteins,
        epochs=15, lr=1e-3, patience=5, batch_size=128, num_neighbors=[32, 16],  # v18-short: 短训练快速观察指标
        prot_to_path_neighbors=graphs.get("prot_to_path_neighbors"))

    # v17: 动态集成权重 — 用最终 best_state 模型重新评估蛋白冷启动 AUPR
    # 而非从历史记录中取最大值，确保权重与最终模型状态对齐
    sage_best_prot_aupr = 0.5
    hgt_best_prot_aupr = 0.5

    if val_proteins and len(val_proteins) > 0:
        logger.info(">>> 重新评估蛋白冷启动 AUPR（最终 best_state 模型）")
        sage_model.eval()
        sage_final_prot = _validate_sage_protein_cold(
            sage_model, graphs["x"], graphs.get("homo_edge_index_prot_cold", graphs["homo_edge_index"]),
            val_compounds, compound_to_pos,
            graphs["n_compounds"], graphs["n_proteins"], val_proteins)
        sage_best_prot_aupr = sage_final_prot["aupr"]
        logger.info(f"  SAGE 最终蛋白冷启动 AUPR: {sage_best_prot_aupr:.4f} "
                    f"(n_valid={sage_final_prot['n_valid_compounds']})")

        hgt_model.eval()
        hgt_final_prot = _validate_hgt_protein_cold(
            hgt_model, graphs.get("hetero_data_prot_cold", graphs["hetero_data"]), val_compounds, compound_to_pos,
            graphs["n_compounds"], graphs["n_proteins"], val_proteins,
            hetero_adj=graphs.get("hetero_adj_prot_cold", graphs.get("hetero_adj")))
        hgt_best_prot_aupr = hgt_final_prot["aupr"]
        logger.info(f"  HGT 最终蛋白冷启动 AUPR: {hgt_best_prot_aupr:.4f} "
                    f"(n_valid={hgt_final_prot['n_valid_compounds']})")

    logger.info(f"动态集成权重 AUPR: SAGE={sage_best_prot_aupr:.4f}, HGT={hgt_best_prot_aupr:.4f}")

    # ======== 预测 TCM ========
    logger.info(">>> 预测 TCM 化合物（v18: MC Dropout ×30 不确定性估计）")
    tcm_smiles = tcm_df["SMILES_std"].dropna().tolist()
    all_train_smiles = list(graphs["smi_to_idx"].keys())
    _, cp_mean, cp_std, cp_col_mean = build_compound_features(all_train_smiles)
    compound_stats = (cp_mean, cp_std, cp_col_mean)

    pred_df = predict_tcm(
        sage_model, hgt_model, graphs, tcm_smiles, warm_targets,
        compound_stats,
        sage_prot_aupr=sage_best_prot_aupr, hgt_prot_aupr=hgt_best_prot_aupr,
        mc_samples=30)

    if "MOL_ID" in tcm_df.columns and "molecule_name" in tcm_df.columns:
        name_map = dict(zip(tcm_df["SMILES_std"], tcm_df["molecule_name"]))
        mol_id_map = dict(zip(tcm_df["SMILES_std"], tcm_df["MOL_ID"]))
        pred_df["molecule_name"] = pred_df["SMILES"].map(name_map).fillna("")
        pred_df["MOL_ID"] = pred_df["SMILES"].map(mol_id_map).fillna("")

    # 排序
    gene_cols = [g for g in warm_targets if g in pred_df.columns]
    scores = pred_df[gene_cols].values
    avg_score = np.nanmean(scores, axis=1)
    max_score = np.nanmax(scores, axis=1)
    n_hits = np.nansum(scores > 0.5, axis=1)

    def _norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    composite = 0.4 * _norm(avg_score) + 0.3 * _norm(max_score) + 0.3 * _norm(n_hits / len(gene_cols))

    # v17: 不确定性调整 — 优先选择高分且低不确定度的化合物
    if "mean_uncertainty" in pred_df.columns:
        uncertainty = pred_df["mean_uncertainty"].values
        uncertainty_penalty = 1.0 - _norm(uncertainty)
        composite = composite * uncertainty_penalty
        pred_df["uncertainty_penalty"] = uncertainty_penalty
        logger.info(f"  不确定性调整: 惩罚范围 [{uncertainty_penalty.min():.4f}, {uncertainty_penalty.max():.4f}]")

    pred_df["composite_score"] = composite
    pred_df["avg_score"] = avg_score
    pred_df["max_score"] = max_score
    pred_df["n_hits"] = n_hits
    pred_df["n_targets"] = len(gene_cols)

    top_targets_list = []
    for i in range(len(pred_df)):
        gene_scores = [(g, scores[i][j]) for j, g in enumerate(gene_cols)]
        gene_scores.sort(key=lambda x: x[1], reverse=True)
        top_targets_list.append(", ".join([f"{g}({s:.3f})" for g, s in gene_scores[:5]]))
    pred_df["top_targets"] = top_targets_list

    pred_df = pred_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    pred_df["rank"] = range(1, len(pred_df) + 1)
    top_df = pred_df.head(500).copy()

    pred_df.to_csv(L4_RESULTS / "tcm_predictions_full_v18.csv", index=False)
    top_df.to_csv(L4_RESULTS / "tcm_top_candidates_v18.csv", index=False)

    # 性能
    perf_rows = []
    if sage_history:
        bg = max(sage_history, key=lambda x: x.get("auc", 0))
        perf_rows.append({"model": "SAGE", "best_auc": bg["auc"], "best_aupr": bg.get("aupr", 0)})
    if hgt_history:
        bh = max(hgt_history, key=lambda x: x.get("auc", 0))
        perf_rows.append({"model": "HGT", "best_auc": bh["auc"], "best_aupr": bh.get("aupr", 0)})
    if perf_rows:
        pd.DataFrame(perf_rows).to_csv(L4_RESULTS / "model_performance_v18.csv", index=False)

    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Phase 4 v18 完成！总耗时 {total_time / 60:.1f} 分钟")
    if sage_history:
        logger.info(f"  SAGE best val_auc: {max(h['auc'] for h in sage_history):.4f}  val_aupr: {max(h.get('aupr', 0) for h in sage_history):.4f}")
    if hgt_history:
        logger.info(f"  HGT best val_auc: {max(h['auc'] for h in hgt_history):.4f}  val_aupr: {max(h.get('aupr', 0) for h in hgt_history):.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()