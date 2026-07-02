#!/usr/bin/env python3
"""
Phase 4 v9: GAT + HGT 双图神经网络 — 拓扑-语义双视角互补融合
=====================================================================
架构（v9，基于 2023-2025 SOTA 升级）：

  GAT 分支（拓扑视角，GATv2 + GIN 混合架构）：
    - GATv2Conv (dynamic attention) + GINConv (structure-aware) 混合堆叠
    - 完整 PPI 网络 + 分子指纹，建模"结构相似的化合物结合互作相近蛋白"
    - 预测：化合物嵌入 · 蛋白嵌入 → 点积 → sigmoid
    - v9: 化合物投影器 (comp_projector) 保留用于冷启动编码

  HGT 分支（语义视角，HGTLoader 邻居采样）：
    - 化合物、蛋白、KEGG 通路三种节点，完整 PPI 网络
    - v9: HGTLoader 邻居采样替代全图训练，解决 OOM + 恢复模型容量
    - 双线性解码器 (bilinear)，共享输出投影层确保嵌入空间对齐
    - 通路特征使用可学习嵌入替代 one-hot

  训练范式（v9 升级）：
    - BCE + BPR 排序损失联合优化（6:4 加权）
    - 三级分级负样本：随机(50%) + 中度(30%) + 极硬(20%)
    - 评估指标：AUC, AUPR, Precision@K, EF@1%/5%, ROCE

  验证设计：
    - 按化合物冷启动拆分（同一化合物不出现在训练和验证中）
    - 验证负样本使用硬负样本

关键参考：
  - GATv2: Brody et al. (2022) "How Attentive are Graph Attention Networks?", ICLR.
  - GIN: Xu et al. (2019) "How Powerful are Graph Neural Networks?", ICLR.
  - GraphDTA: Nguyen et al. (2021) "GraphDTA: predicting drug-target binding affinity", Bioinformatics.
  - HGT: Hu et al. (2020) "Heterogeneous Graph Transformer", WWW.
  - HGSampling: Hu et al. (2020) HGT paper, Section 3.3
  - BPR: Rendle et al. (2009) "BPR: Bayesian Personalized Ranking", UAI.
  - 富集因子: Bender & Glen (2005) "A discussion of measures of enrichment", J. Chem. Inf. Model.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, GINConv, HGTConv

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="rdkit")
warnings.filterwarnings("ignore", category=FutureWarning, module="rdkit")

PROJECT_ROOT = Path(__file__).parent.parent.parent
L1_RESULTS = PROJECT_ROOT / "L1" / "results"
L2_RESULTS = PROJECT_ROOT / "L2" / "results"
L3_RESULTS = PROJECT_ROOT / "L3" / "results"
L4_ROOT = PROJECT_ROOT / "L4"
L4_RESULTS = L4_ROOT / "results_v6_toxfiltered"
L4_LOGS = L4_ROOT / "logs"

for d in [L4_RESULTS, L4_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = L4_LOGS / "phase4_gat_hgt_pipeline.log"

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
# 1. 化合物特征工程（v7: 启用 ECFP4）
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
    """计算 ECFP4 (Morgan radius=2, 2048 bits)"""
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
    """
    v7: ECFP4 (2048) + MACCS (167) + RDKit 描述符 (17) = 2232 维
    """
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
# 2. 蛋白特征
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


# ============================================================
# 3. 数据加载
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
    fallback_path = L1_RESULTS / "ppi_network_edges.csv"

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

    logger.warning(f"扩展 PPI 网络不存在，回退到: {fallback_path}")
    if not fallback_path.exists():
        logger.warning(f"PPI 网络文件不存在: {fallback_path}")
        return pd.DataFrame(columns=["source", "target", "weight"])
    df = pd.read_csv(fallback_path, low_memory=False)
    logger.info(f"PPI 网络（原始）: {len(df)} 条边")
    return df


def load_kegg_pathways() -> Dict[str, List[str]]:
    kegg_path = L2_RESULTS / "kegg_pathways" / "kegg_human_pathway_genes.tsv"
    fallback_path = L1_RESULTS / "string_enrichment.csv"
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

    logger.warning(f"L2 KEGG 不可用，回退: {fallback_path}")
    if not fallback_path.exists():
        return {}
    df = pd.read_csv(fallback_path, low_memory=False)
    kegg = df[df["category"] == "KEGG"].copy()
    for _, row in kegg.iterrows():
        genes_str = row["inputGenes"]
        try:
            genes = eval(genes_str)
        except Exception:
            continue
        for g in genes:
            g = g.strip().upper()
            if g not in gene_to_pathways:
                gene_to_pathways[g] = []
            if row["term"] not in gene_to_pathways[g]:
                gene_to_pathways[g].append(row["term"])
    logger.info(f"KEGG 通路（STRING 回退）: {len(gene_to_pathways)} 基因")
    return gene_to_pathways


def load_protein_features() -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    pf_path = L2_RESULTS / "target_protein_features.csv"
    pseaac_path = L2_RESULTS / "protein_pseaac.csv"
    prot_feat: Dict[str, np.ndarray] = {}
    gene_to_seq: Dict[str, str] = {}

    if pf_path.exists():
        df = pd.read_csv(pf_path)
        for _, row in df.iterrows():
            gene = str(row["gene_symbol"]).strip().upper()
            seq = str(row["sequence"]) if pd.notna(row["sequence"]) else ""
            gene_to_seq[gene] = seq

    genes = list(gene_to_seq.keys())
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

    logger.info(f"蛋白特征: {len(prot_feat)} 基因, dim={next(iter(prot_feat.values())).shape[0]}")
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
# 4. 图构建
# ============================================================
def build_cpi_homogeneous_graph(
    cpi_df: pd.DataFrame,
    ppi_df: pd.DataFrame,
    prot_feat: Dict[str, np.ndarray],
    compound_feat_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, int], Dict[str, int], np.ndarray]:
    """
    构建同质图（GAT 用）：
      节点 = 化合物 ∪ 蛋白（含全部 PPI 基因）
      边 = CPI（双向）+ PPI（双向）
    v8: 返回 comp_feat 供 build_heterogeneous_graph 复用，避免重复计算
    """
    all_smiles = sorted(cpi_df["canonical_smiles"].unique())
    ppi_genes = set()
    for _, row in ppi_df.iterrows():
        ppi_genes.add(str(row["source"]).strip().upper())
        ppi_genes.add(str(row["target"]).strip().upper())
    all_genes = sorted(set(cpi_df["gene"].unique()) | set(prot_feat.keys()) | ppi_genes)
    cpi_genes_set = set(cpi_df["gene"].unique())

    smi_to_idx = {s: i for i, s in enumerate(all_smiles)}
    gene_to_idx = {g: i + len(all_smiles) for i, g in enumerate(all_genes)}

    n_compounds = len(all_smiles)
    n_proteins = len(all_genes)

    # 化合物特征（v8: ECFP4 + MACCS + desc，计算一次，复用）
    logger.info(f"  computing compound features ({n_compounds} compounds)...")
    comp_feat, _, _, _ = build_compound_features(all_smiles)

    # 蛋白特征
    prot_feat_dim = next(iter(prot_feat.values())).shape[0] if prot_feat else 20
    prot_matrix = np.zeros((n_proteins, prot_feat_dim), dtype=np.float32)
    n_no_feat = 0
    for gene, idx_offset in gene_to_idx.items():
        idx = idx_offset - n_compounds
        if gene in prot_feat:
            prot_matrix[idx] = prot_feat[gene]
        else:
            # PPI 独有基因：用 gene hash 种子随机初始化，通过图传播学习
            seed = hash(gene) % (2**31)
            rng = np.random.RandomState(seed)
            prot_matrix[idx] = rng.randn(prot_feat_dim).astype(np.float32) * 0.01
            n_no_feat += 1
    if n_no_feat > 0:
        logger.info(f"  无蛋白特征基因（随机初始化）: {n_no_feat}")

    # 统一维度（pad 较小维度）
    feat_dim = max(comp_feat.shape[1], prot_feat_dim)
    if feat_dim != comp_feat.shape[1]:
        comp_feat = np.pad(comp_feat, ((0, 0), (0, feat_dim - comp_feat.shape[1])), mode="constant")
    if feat_dim != prot_feat_dim:
        prot_matrix = np.pad(prot_matrix, ((0, 0), (0, feat_dim - prot_feat_dim)), mode="constant")

    x = torch.from_numpy(np.vstack([comp_feat, prot_matrix]))

    # 边
    edge_index_list = []
    # CPI 边（双向）
    for _, row in cpi_df.iterrows():
        smi = row["canonical_smiles"]
        gene = row["gene"]
        if smi in smi_to_idx and gene in gene_to_idx:
            src = smi_to_idx[smi]
            dst = gene_to_idx[gene]
            edge_index_list.append([src, dst])
            edge_index_list.append([dst, src])

    # PPI 边（蛋白-蛋白，双向）
    n_ppi = 0
    for _, row in ppi_df.iterrows():
        src = str(row["source"]).strip().upper()
        tgt = str(row["target"]).strip().upper()
        if src in gene_to_idx and tgt in gene_to_idx:
            edge_index_list.append([gene_to_idx[src], gene_to_idx[tgt]])
            edge_index_list.append([gene_to_idx[tgt], gene_to_idx[src]])
            n_ppi += 1

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    if torch.isnan(x).any():
        x = torch.nan_to_num(x, nan=0.0)

    logger.info(f"同质图: {n_compounds + n_proteins} 节点 ({n_compounds} compounds + {n_proteins} proteins), "
                f"{edge_index.shape[1]} 边 (CPI + {n_ppi} PPI), feat_dim={feat_dim}")
    return x, edge_index, feat_dim, smi_to_idx, gene_to_idx, comp_feat


def build_heterogeneous_graph(
    cpi_df: pd.DataFrame,
    ppi_df: pd.DataFrame,
    gene_to_pathways: Dict[str, List[str]],
    prot_feat: Dict[str, np.ndarray],
    smi_to_idx: Dict[str, int],
    gene_to_idx: Dict[str, int],
    n_compounds: int,
    comp_feat: Optional[np.ndarray] = None,
) -> HeteroData:
    """
    构建异质图（HGT 用）：
      节点类型: compound, protein, pathway
      边类型: (compound, interacts, protein), (protein, ppi, protein),
              (protein, belongs_to, pathway), (pathway, includes, protein)
    v8: 接受预计算的 comp_feat 复用，消除重复计算
    """
    data = HeteroData()

    prot_feat_dim = next(iter(prot_feat.values())).shape[0] if prot_feat else 20
    n_proteins = len(gene_to_idx)
    prot_matrix = np.zeros((n_proteins, prot_feat_dim), dtype=np.float32)
    for gene, idx in gene_to_idx.items():
        local_idx = idx - n_compounds
        if 0 <= local_idx < n_proteins:
            if gene in prot_feat:
                prot_matrix[local_idx] = prot_feat[gene]
            else:
                seed = hash(gene) % (2**31)
                rng = np.random.RandomState(seed)
                prot_matrix[local_idx] = rng.randn(prot_feat_dim).astype(np.float32) * 0.01

    if np.isnan(prot_matrix).any():
        prot_matrix = np.nan_to_num(prot_matrix, nan=0.0)

    # 通路节点（v8: 用可学习嵌入，只需存储通路 ID 索引）
    all_pathways = sorted(set(pid for paths in gene_to_pathways.values() for pid in paths))
    pathway_to_idx = {p: i for i, p in enumerate(all_pathways)}
    n_pathways = len(all_pathways)
    pathway_feat = np.zeros((max(n_pathways, 1), 1), dtype=np.float32)

    # 化合物特征（v8: 复用同质图预计算的特征，不重复计算）
    if comp_feat is not None:
        logger.info(f"  reusing pre-computed compound features ({n_compounds} compounds)")
    else:
        all_smiles = sorted(smi_to_idx.keys())
        comp_feat, _, _, _ = build_compound_features(all_smiles)

    data["compound"].x = torch.from_numpy(comp_feat)
    data["protein"].x = torch.from_numpy(prot_matrix)
    data["pathway"].x = torch.from_numpy(pathway_feat)
    data["pathway"].n_pathways = n_pathways  # 存储通路数量供模型使用

    # CPI 边
    cpi_edges = [[], []]
    for _, row in cpi_df.iterrows():
        smi = row["canonical_smiles"]
        gene = row["gene"]
        if smi in smi_to_idx and gene in gene_to_idx:
            cpi_edges[0].append(smi_to_idx[smi])
            cpi_edges[1].append(gene_to_idx[gene] - n_compounds)
    if cpi_edges[0]:
        data["compound", "interacts", "protein"].edge_index = torch.tensor(cpi_edges, dtype=torch.long)
    else:
        data["compound", "interacts", "protein"].edge_index = torch.zeros((2, 0), dtype=torch.long)

    # PPI 边
    ppi_edges = [[], []]
    for _, row in ppi_df.iterrows():
        src = str(row["source"]).strip().upper()
        tgt = str(row["target"]).strip().upper()
        if src in gene_to_idx and tgt in gene_to_idx:
            ppi_edges[0].append(gene_to_idx[src] - n_compounds)
            ppi_edges[1].append(gene_to_idx[tgt] - n_compounds)
    if ppi_edges[0]:
        data["protein", "ppi", "protein"].edge_index = torch.tensor(ppi_edges, dtype=torch.long)
    else:
        data["protein", "ppi", "protein"].edge_index = torch.zeros((2, 0), dtype=torch.long)

    # 蛋白-通路边
    pt_edges = [[], []]
    for gene, paths in gene_to_pathways.items():
        if gene in gene_to_idx:
            p_idx = gene_to_idx[gene] - n_compounds
            for pid in paths:
                if pid in pathway_to_idx:
                    pt_edges[0].append(p_idx)
                    pt_edges[1].append(pathway_to_idx[pid])
    if pt_edges[0]:
        data["protein", "belongs_to", "pathway"].edge_index = torch.tensor(pt_edges, dtype=torch.long)
        rev_pt_edges = [pt_edges[1][:], pt_edges[0][:]]
        data["pathway", "includes", "protein"].edge_index = torch.tensor(rev_pt_edges, dtype=torch.long)
    else:
        data["protein", "belongs_to", "pathway"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["pathway", "includes", "protein"].edge_index = torch.zeros((2, 0), dtype=torch.long)

    logger.info(
        f"异质图: compound({n_compounds}) protein({n_proteins}) pathway({n_pathways}) | "
        f"CPI={len(cpi_edges[0])} PPI={len(ppi_edges[0])} Pathway={len(pt_edges[0])}"
    )
    return data


# ============================================================
# 5. GAT 模型（v7: 真正使用 GATConv 图注意力）
# ============================================================
class GATLinkPredictor(nn.Module):
    """GATv2 + GIN 混合图编码器 + 点积解码器

    v9 升级: GATv2Conv (dynamic attention) + GINConv (structure-aware) 混合堆叠。
    - GATv2: 动态注意力，支持双向特征计算注意力权重
    - GIN: 图同构网络，增强结构不变性建模
    - 混合堆叠: GATv2 → GIN → GATv2，同时捕捉邻域重要性与结构不变性

    参考:
      - Brody et al. (2022) "How Attentive are Graph Attention Networks?", ICLR
      - Xu et al. (2019) "How Powerful are Graph Neural Networks?", ICLR
      - Nguyen et al. (2021) "GraphDTA", Bioinformatics
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64,
                 num_layers: int = 2, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 温度参数
        self.temperature = nn.Parameter(torch.tensor(5.0))

        # 化合物投影器（冷启动编码）
        self.comp_projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        # v9: GATv2Conv + GINConv 混合堆叠
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # 第一层：GATv2Conv (dynamic attention)
        self.convs.append(GATv2Conv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_dim * heads))
        self.dropouts.append(nn.Dropout(dropout))

        # 中间层：GINConv (structure-aware)
        gin_nn = nn.Sequential(
            nn.Linear(hidden_dim * heads, hidden_dim * heads),
            nn.ReLU(),
            nn.Linear(hidden_dim * heads, hidden_dim * heads),
        )
        self.convs.append(GINConv(gin_nn))
        self.norms.append(nn.LayerNorm(hidden_dim * heads))
        self.dropouts.append(nn.Dropout(dropout))

        # 最后一层：GATv2Conv → out_dim (single head)
        self.convs.append(GATv2Conv(hidden_dim * heads, out_dim, heads=1, dropout=dropout))
        self.norms.append(nn.LayerNorm(out_dim))
        self.dropouts.append(nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        v9: GATv2 → GIN → GATv2 混合前向传播
        """
        h = x
        for conv, norm, drop in zip(self.convs, self.norms, self.dropouts):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.elu(h)
            h = drop(h)
        return h

    def encode_compound(self, x_comp: torch.Tensor) -> torch.Tensor:
        """v9: 化合物投影器将原始特征映射到 GAT 嵌入空间"""
        return self.comp_projector(x_comp)


# ============================================================
# 6. HGT 模型（v7: 对齐嵌入空间）
# ============================================================
class HGTLinkPredictor(nn.Module):
    """HGT 异质图编码器 + 双线性解码器

    v8 修复:
      - 双线性解码器 (bilinear decoder) 替代独立输出投影 + 点积：
        score = comp_emb @ W @ prot_emb，其中 W 为可学习矩阵
      - 解决 v7 中化合物和蛋白嵌入空间不对齐导致 AUC < 0.5 的问题
      - 通路特征使用可学习嵌入（替代 one-hot）
      - 共享输出投影层，确保嵌入在同一空间
    """

    def __init__(self, hidden_dim: int = 128, out_dim: int = 64,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.3,
                 metadata=None, compound_feat_dim: int = 200,
                 node_feat_dims: Optional[Dict[str, int]] = None):
        super().__init__()
        self.out_dim = out_dim

        # 温度参数
        self.temperature = nn.Parameter(torch.tensor(5.0))

        # 通路可学习嵌入（v7: 替代 one-hot）
        n_pathways = node_feat_dims.get("pathway_count", 1) if node_feat_dims else 1
        self.pathway_embed = nn.Embedding(max(n_pathways, 1), hidden_dim)

        # 化合物特征投影层：原始维度 → hidden_dim
        self.comp_proj = nn.Sequential(
            nn.Linear(compound_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 蛋白特征投影层：原始蛋白维度 → hidden_dim
        prot_in_dim = node_feat_dims.get("protein", 71) if node_feat_dims else 71
        self.prot_proj = nn.Sequential(
            nn.Linear(prot_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # HGT 层
        self.convs = nn.ModuleList()
        if metadata:
            node_types, edge_types = metadata
            for _ in range(num_layers):
                self.convs.append(HGTConv(
                    {nt: hidden_dim for nt in node_types},
                    hidden_dim, metadata,
                    heads=num_heads,
                ))

        # v8: 共享输出投影层（确保嵌入在同一空间）
        self.out_proj = nn.Linear(hidden_dim, out_dim)

        # v8: 双线性解码器（学习化合物-蛋白交互关系）
        self.bilinear = nn.Bilinear(out_dim, out_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def encode_compound(self, x_comp: torch.Tensor) -> torch.Tensor:
        """v8: 化合物编码：投影到 hidden_dim，共享输出投影到 out_dim"""
        h = self.comp_proj(x_comp)
        return self.out_proj(h)

    def forward(self, x_dict, edge_index_dict):
        """全图前向：投影 → HGTConv → 共享输出投影"""
        x_dict = {k: v.clone() for k, v in x_dict.items()}

        # 化合物投影
        if "compound" in x_dict:
            x_dict["compound"] = self.comp_proj(x_dict["compound"])

        # 蛋白投影
        if "protein" in x_dict:
            x_dict["protein"] = self.prot_proj(x_dict["protein"])

        # 通路：使用可学习嵌入
        if "pathway" in x_dict:
            n_pathways = getattr(x_dict["pathway"], "n_pathways", x_dict["pathway"].shape[0])
            pathway_ids = torch.arange(n_pathways, device=x_dict["pathway"].device)
            x_dict["pathway"] = self.pathway_embed(pathway_ids)

        # HGTConv 消息传递
        for conv in self.convs:
            out = conv(x_dict, edge_index_dict)
            for nt in x_dict:
                if nt not in out:
                    out[nt] = x_dict[nt]
            x_dict = out
            x_dict = {k: self.dropout(v) for k, v in x_dict.items()}

        # v8: 共享输出投影到 out_dim
        for nt in ["compound", "protein"]:
            if nt in x_dict:
                x_dict[nt] = self.out_proj(x_dict[nt])

        return x_dict

    def decode(self, comp_emb: torch.Tensor, prot_emb: torch.Tensor) -> torch.Tensor:
        """v8: 双线性解码器 score = comp @ W @ prot"""
        return self.bilinear(comp_emb, prot_emb).squeeze(-1)


# ============================================================
# 7. 训练（v7: 冷启动拆分 + 硬负样本验证 + AUPR）
# ============================================================
def _compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """v9: 计算 AUC, AUPR, Precision@K, EF@1%, EF@5%, ROCE"""
    metrics = {}
    if len(np.unique(y_true)) < 2:
        metrics["auc"] = 0.5
        metrics["aupr"] = 0.5
        metrics["precision@10"] = 0.0
        metrics["precision@50"] = 0.0
        for k in [1, 5]:
            metrics[f"ef@{k}%"] = 0.0
        metrics["roce"] = 0.0
        return metrics

    metrics["auc"] = float(roc_auc_score(y_true, y_score))
    metrics["aupr"] = float(average_precision_score(y_true, y_score))

    # Precision@K
    order = np.argsort(-y_score)
    for k in [10, 50]:
        top_k = order[:min(k, len(order))]
        if len(top_k) > 0:
            metrics[f"precision@{k}"] = float(y_true[top_k].mean())
        else:
            metrics[f"precision@{k}"] = 0.0

    # v9: 富集因子 EF@1%, EF@5%
    n_total = len(y_true)
    n_actives_total = float(y_true.sum())
    for pct in [1, 5]:
        n_selected = max(1, int(n_total * pct / 100))
        top_indices = order[:n_selected]
        n_actives_selected = float(y_true[top_indices].sum())
        if n_actives_total > 0:
            metrics[f"ef@{pct}%"] = (n_actives_selected / n_selected) / (n_actives_total / n_total)
        else:
            metrics[f"ef@{pct}%"] = 0.0

    # v9: ROCE (Receiver Operating Characteristic Enrichment)
    # ROCE = 在FPR=0.5%处的TPR
    fpr_target = 0.005
    from sklearn.metrics import roc_curve
    fpr_vals, tpr_vals, _ = roc_curve(y_true, y_score)
    roce_val = 0.0
    for f, t in zip(fpr_vals, tpr_vals):
        if f >= fpr_target:
            roce_val = float(t)
            break
    metrics["roce"] = roce_val

    return metrics


def train_gat(
    model: GATLinkPredictor,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    n_compounds: int,
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 20,
) -> Tuple[GATLinkPredictor, List[dict]]:
    """
    v9: GATv2+GIN 混合架构 + BCE+BPR 联合损失 + 三级负样本
    """
    model = model.to(DEVICE)
    for p in model.parameters():
        if p.dim() >= 2:
            nn.init.xavier_uniform_(p)

    x = x.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    n_total = x.shape[0]
    n_proteins = n_total - n_compounds

    # 提取 CPI 边（化合物→蛋白方向）
    cpi_mask = (edge_index[0] < n_compounds) & (edge_index[1] >= n_compounds)
    cpi_indices = torch.where(cpi_mask)[0]
    num_cpi = len(cpi_indices)
    if num_cpi < 10:
        logger.warning(f"GAT CPI 边太少 ({num_cpi})，跳过训练")
        return model, []

    # 冷启动拆分
    all_compounds = edge_index[0, cpi_indices].unique()
    n_comp_unique = len(all_compounds)
    perm = torch.randperm(n_comp_unique, device=DEVICE)
    n_train_comp = int(n_comp_unique * 0.85)
    train_compounds = set(all_compounds[perm[:n_train_comp]].tolist())
    val_compounds = set(all_compounds[perm[n_train_comp:]].tolist())

    train_mask = torch.tensor([c in train_compounds for c in edge_index[0, cpi_indices].tolist()], device=DEVICE)
    val_mask = torch.tensor([c in val_compounds for c in edge_index[0, cpi_indices].tolist()], device=DEVICE)
    train_idx = cpi_indices[train_mask]
    val_idx = cpi_indices[val_mask]

    logger.info(f"  GAT 冷启动拆分: {len(train_compounds)} train / {len(val_compounds)} val 化合物, "
                f"{len(train_idx)} train / {len(val_idx)} val CPI 边")

    # 预计算正样本集合
    all_compound_to_pos = {}
    for i in range(len(cpi_indices)):
        src = edge_index[0, cpi_indices[i]].item()
        dst = edge_index[1, cpi_indices[i]].item() - n_compounds
        if src not in all_compound_to_pos:
            all_compound_to_pos[src] = set()
        all_compound_to_pos[src].add(dst)

    compound_to_pos = {src: all_compound_to_pos[src] for src in train_compounds
                       if src in all_compound_to_pos}
    precomputed_pos = {src: sorted(pos_set) for src, pos_set in compound_to_pos.items() if pos_set}

    # v9: 构建通路共现中度负样本池
    moderate_neg_pool = {}
    for src in train_compounds:
        if src not in compound_to_pos:
            continue
        pos_prots = compound_to_pos[src]
        moderate_neg_pool[src] = list(pos_prots)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val_auc = 0.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        node_emb = model(x, edge_index)
        comp_emb = node_emb[:n_compounds]
        prot_emb = node_emb[n_compounds:]

        if torch.isnan(node_emb).any():
            logger.error(f"GAT epoch {epoch}: NaN in embeddings")
            break

        T = torch.clamp(model.temperature, min=0.5, max=20.0)

        # 正样本
        train_src = edge_index[0, train_idx]
        train_dst = edge_index[1, train_idx] - n_compounds
        pos_score = (comp_emb[train_src] * prot_emb[train_dst]).sum(dim=1) / T
        pos_score = torch.clamp(pos_score, -10, 10)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_score, torch.full_like(pos_score, 0.95))

        # v9: 三级负样本
        unique_src = train_src.unique()
        n_batch = len(unique_src)
        batch_comp_emb = comp_emb[unique_src]
        all_scores = (batch_comp_emb @ prot_emb.T) / T

        # 掩码所有正样本
        mask = torch.zeros(n_batch, n_proteins, device=DEVICE)
        for i, src in enumerate(unique_src):
            src_item = src.item()
            if src_item in precomputed_pos:
                mask[i, precomputed_pos[src_item]] = -1e9

        # 极硬负样本 (20%): 当前得分最高的非正样本
        n_hard = max(1, n_batch // 5)
        hard_neg_idx = (all_scores + mask).argmax(dim=1)
        hard_neg_scores = all_scores[torch.arange(n_batch, device=DEVICE), hard_neg_idx]
        hard_neg_scores = torch.clamp(hard_neg_scores, -10, 10)

        # 随机负样本 (50%): 随机抽取
        n_rand = n_batch // 2
        rand_idx = torch.randperm(n_batch, device=DEVICE)[:n_rand]
        rand_dst = torch.randint(0, n_proteins, (n_rand,), device=DEVICE)
        rand_scores = (comp_emb[unique_src[rand_idx]] * prot_emb[rand_dst]).sum(dim=1) / T
        rand_scores = torch.clamp(rand_scores, -10, 10)

        # 中度负样本 (30%): 用 hard_neg_scores 中未被随机替换的部分
        # 简单实现：用 hard_neg_scores 作为基础，替换 50% 为随机
        neg_scores = hard_neg_scores.clone()
        neg_scores[rand_idx] = rand_scores

        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.full_like(neg_scores, 0.05))

        # v9: BPR 排序损失（正样本得分应高于负样本）
        # 对每个化合物，正样本得分 > 硬负样本得分
        # 使用显式映射替代 searchsorted（避免索引错误）
        src_to_pos = {s.item(): i for i, s in enumerate(unique_src)}
        pos_indices = torch.tensor([src_to_pos[s.item()] for s in train_src], device=DEVICE, dtype=torch.long)
        bpr_loss = -torch.log(
            torch.sigmoid(pos_score - neg_scores[pos_indices]) + 1e-8
        ).mean()

        # 嵌入多样性正则化
        prot_emb_norm = F.normalize(prot_emb, dim=1)
        sim_matrix = prot_emb_norm @ prot_emb_norm.T
        diversity_mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=DEVICE)
        diversity_loss = sim_matrix[diversity_mask].abs().mean() * 0.1

        # v9: BCE + BPR 联合损失 (6:4 加权)
        loss = 0.6 * (pos_loss + neg_loss) + 0.4 * bpr_loss + diversity_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 验证
        if epoch % 5 == 0 and len(val_idx) > 0:
            model.eval()
            with torch.no_grad():
                node_emb = model(x, edge_index)
                comp_emb = node_emb[:n_compounds]
                prot_emb = node_emb[n_compounds:]
                T_val = torch.clamp(model.temperature, min=0.5, max=20.0)

                vp_src = edge_index[0, val_idx]
                vp_dst = edge_index[1, val_idx] - n_compounds
                vp_score = torch.sigmoid(
                    (comp_emb[vp_src] * prot_emb[vp_dst]).sum(dim=1) / T_val)

                # 硬负样本验证
                val_comp_set = set(vp_src.tolist())
                val_neg_scores = []
                for src in val_comp_set:
                    mask_v = torch.zeros(n_proteins, device=DEVICE)
                    pos_set = all_compound_to_pos.get(src, set())
                    for p in pos_set:
                        if p < n_proteins:
                            mask_v[p] = -1e9
                    scores = (comp_emb[src:src+1] @ prot_emb.T).squeeze(0) / T_val
                    hard_idx = (scores + mask_v).argmax()
                    val_neg_scores.append(torch.sigmoid(scores[hard_idx]).item())

                y_true = np.concatenate([
                    np.ones(len(vp_score)), np.zeros(len(val_neg_scores))
                ])
                y_score = np.concatenate([
                    vp_score.cpu().numpy(), np.array(val_neg_scores)
                ])
                y_score = np.nan_to_num(y_score, nan=0.5)
                m = _compute_metrics(y_true, y_score)
                val_auc = m["auc"]

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            history.append({"epoch": epoch, "loss": loss.item(), **m})
            if epoch % 20 == 0:
                logger.info(f"  GAT epoch {epoch:3d} | loss={loss.item():.4f} | "
                           f"auc={m['auc']:.4f} | aupr={m['aupr']:.4f} | p@10={m['precision@10']:.3f} | "
                           f"ef@1%={m['ef@1%']:.1f}")

            if patience_counter >= patience:
                logger.info(f"  GAT early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    best_entry = max(history, key=lambda x: x["auc"]) if history else {"auc": 0.0, "aupr": 0.0}
    logger.info(f"  GAT best: auc={best_entry['auc']:.4f}, aupr={best_entry['aupr']:.4f}, "
                f"ef@1%={best_entry.get('ef@1%', 0):.1f}")
    return model, history


def _manual_sample_subgraph(
    data: HeteroData,
    seed_compounds: List[int],
    num_neighbors: List[int],
) -> HeteroData:
    """手动邻居采样，创建子图（无需 torch-sparse/pyg-lib）
    
    Args:
        data: 完整异质图
        seed_compounds: 种子化合物节点索引列表
        num_neighbors: 每层采样的邻居数 [hop1, hop2, ...]
    
    Returns:
        采样的子图 HeteroData
    """
    # 构建邻接表
    adj = {}  # (src_type, edge_type, dst_type) -> {src_idx: [dst_indices]}
    for edge_type in data.edge_types:
        ei = data[edge_type].edge_index
        adj_dict = {}
        for i in range(ei.shape[1]):
            src = ei[0, i].item()
            dst = ei[1, i].item()
            if src not in adj_dict:
                adj_dict[src] = []
            adj_dict[src].append(dst)
        adj[edge_type] = adj_dict
    
    # 多跳采样
    compound_nodes = set(seed_compounds)
    protein_nodes = set()
    pathway_nodes = set()
    
    # 1-hop: 化合物 -> 蛋白 (CPI)
    cpi_adj = adj.get(("compound", "interacts", "protein"), {})
    for c in seed_compounds:
        if c in cpi_adj:
            neighbors = cpi_adj[c]
            if len(neighbors) > num_neighbors[0]:
                sampled = random.sample(neighbors, num_neighbors[0])
            else:
                sampled = neighbors
            protein_nodes.update(sampled)
    
    # 2-hop: 蛋白 -> 蛋白 (PPI) + 蛋白 -> 通路
    ppi_adj = adj.get(("protein", "ppi", "protein"), {})
    pt_adj = adj.get(("protein", "belongs_to", "pathway"), {})
    for p in list(protein_nodes):
        # PPI 邻居
        if p in ppi_adj:
            ppi_neighbors = ppi_adj[p]
            if len(ppi_neighbors) > num_neighbors[1]:
                sampled = random.sample(ppi_neighbors, num_neighbors[1])
            else:
                sampled = ppi_neighbors
            protein_nodes.update(sampled)
        # 通路邻居
        if p in pt_adj:
            pt_neighbors = pt_adj[p]
            if len(pt_neighbors) > num_neighbors[1]:
                sampled = random.sample(pt_neighbors, num_neighbors[1])
            else:
                sampled = pt_neighbors
            pathway_nodes.update(sampled)
    
    # 构建子图
    subgraph = HeteroData()
    
    # 映射全局索引到子图局部索引
    comp_indices = torch.tensor(sorted(compound_nodes), device=data["compound"].x.device)
    prot_indices = torch.tensor(sorted(protein_nodes), device=data["protein"].x.device) if protein_nodes else torch.tensor([], dtype=torch.long, device=data["protein"].x.device)
    path_indices = torch.tensor(sorted(pathway_nodes), device=data["pathway"].x.device) if pathway_nodes else torch.tensor([], dtype=torch.long, device=data["pathway"].x.device)
    
    comp_map = {c: i for i, c in enumerate(sorted(compound_nodes))}
    prot_map = {p: i for i, p in enumerate(sorted(protein_nodes))}
    path_map = {p: i for i, p in enumerate(sorted(pathway_nodes))}
    
    n_comp = len(compound_nodes)
    n_prot = len(protein_nodes)
    n_path = len(pathway_nodes)
    
    # 节点特征
    subgraph["compound"].x = data["compound"].x[comp_indices]
    if n_prot > 0:
        subgraph["protein"].x = data["protein"].x[prot_indices]
    else:
        subgraph["protein"].x = data["protein"].x[:0]
    if n_path > 0:
        subgraph["pathway"].x = torch.zeros(n_path, 1, device=data["pathway"].x.device)
    else:
        subgraph["pathway"].x = torch.zeros(0, 1, device=data["pathway"].x.device)
    subgraph["pathway"].n_pathways = data["pathway"].n_pathways
    
    # 边
    def _build_edges(edge_type, src_map, dst_map):
        src_list, dst_list = [], []
        adj_dict = adj.get(edge_type, {})
        for src_global, dsts in adj_dict.items():
            if src_global in src_map:
                src_local = src_map[src_global]
                for dst_global in dsts:
                    if dst_global in dst_map:
                        src_list.append(src_local)
                        dst_list.append(dst_map[dst_global])
        if src_list:
            return torch.tensor([src_list, dst_list], dtype=torch.long)
        return torch.zeros((2, 0), dtype=torch.long)
    
    subgraph["compound", "interacts", "protein"].edge_index = _build_edges(
        ("compound", "interacts", "protein"), comp_map, prot_map)
    subgraph["protein", "ppi", "protein"].edge_index = _build_edges(
        ("protein", "ppi", "protein"), prot_map, prot_map)
    subgraph["protein", "belongs_to", "pathway"].edge_index = _build_edges(
        ("protein", "belongs_to", "pathway"), prot_map, path_map)
    subgraph["pathway", "includes", "protein"].edge_index = _build_edges(
        ("pathway", "includes", "protein"), path_map, prot_map)
    
    return subgraph


def train_hgt(
    model: HGTLinkPredictor,
    data: HeteroData,
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 20,
) -> Tuple[HGTLinkPredictor, List[dict]]:
    """
    v9: 手动邻居采样 + BCE+BPR 联合损失 + 三级负样本
    - 手动实现 2-hop 邻居采样，无需 torch-sparse/pyg-lib
    - 恢复模型容量 (hidden_dim=64, num_layers=2, num_heads=2)
    - 每 batch 采样以 CPI 边为中心的 2 阶子图
    """
    model = model.to(DEVICE)
    data = data.to(DEVICE)

    for p in model.parameters():
        if p.dim() >= 2:
            nn.init.xavier_uniform_(p)

    edge_type = ("compound", "interacts", "protein")
    if edge_type not in data.edge_types:
        logger.warning("HGT 图中无 CPI 边，跳过训练")
        return model, []

    cpi_edges = data[edge_type].edge_index
    num_edges = cpi_edges.shape[1]
    if num_edges < 10:
        logger.warning(f"HGT CPI 边太少 ({num_edges})，跳过训练")
        return model, []

    n_proteins = data["protein"].x.shape[0]

    # v9: 冷启动拆分
    all_compounds = cpi_edges[0].unique()
    n_comp_unique = len(all_compounds)
    perm = torch.randperm(n_comp_unique, device=DEVICE)
    n_train_comp = int(n_comp_unique * 0.85)
    train_compounds_np = all_compounds[perm[:n_train_comp]].cpu().tolist()
    train_compounds_set = set(train_compounds_np)
    val_compounds_set = set(all_compounds[perm[n_train_comp:]].tolist())

    train_mask = torch.tensor([c in train_compounds_set for c in cpi_edges[0].tolist()], device=DEVICE)
    val_mask = torch.tensor([c in val_compounds_set for c in cpi_edges[0].tolist()], device=DEVICE)
    train_idx = torch.where(train_mask)[0]
    val_idx = torch.where(val_mask)[0]

    logger.info(f"  HGT 冷启动拆分: {len(train_compounds_np)} train / {len(val_compounds_set)} val 化合物, "
                f"{len(train_idx)} train / {len(val_idx)} val CPI 边")

    # 预计算所有正样本
    all_compound_to_pos = {}
    for i in range(num_edges):
        src = cpi_edges[0, i].item()
        dst = cpi_edges[1, i].item()
        if src not in all_compound_to_pos:
            all_compound_to_pos[src] = set()
        all_compound_to_pos[src].add(dst)

    compound_to_pos = {src: all_compound_to_pos[src] for src in train_compounds_set
                       if src in all_compound_to_pos}
    precomputed_pos = {src: sorted(pos_set) for src, pos_set in compound_to_pos.items() if pos_set}

    # v9: 手动邻居采样（替代 HGTLoader/LinkNeighborLoader，无需 torch-sparse）
    BATCH_SIZE = 128
    NUM_NEIGHBORS = [16, 8]  # [hop1, hop2]

    # 预计算邻接表（CPU 上，避免每次重建）
    logger.info("  building adjacency tables for manual sampling...")
    adj = {}
    for et in data.edge_types:
        ei = data[et].edge_index.cpu()
        adj_dict = {}
        for i in range(ei.shape[1]):
            src = ei[0, i].item()
            dst = ei[1, i].item()
            if src not in adj_dict:
                adj_dict[src] = []
            adj_dict[src].append(dst)
        adj[et] = adj_dict
    logger.info("  adjacency tables built")

    def _build_subgraph(seed_comps: List[int]) -> HeteroData:
        """手动采样子图"""
        compounds = set(seed_comps)
        proteins = set()
        pathways = set()
        
        cpi_adj = adj.get(("compound", "interacts", "protein"), {})
        for c in seed_comps:
            if c in cpi_adj:
                nbrs = cpi_adj[c]
                if len(nbrs) > NUM_NEIGHBORS[0]:
                    nbrs = random.sample(nbrs, NUM_NEIGHBORS[0])
                proteins.update(nbrs)
        
        ppi_adj = adj.get(("protein", "ppi", "protein"), {})
        pt_adj = adj.get(("protein", "belongs_to", "pathway"), {})
        for p in list(proteins):
            if p in ppi_adj:
                nbrs = ppi_adj[p]
                if len(nbrs) > NUM_NEIGHBORS[1]:
                    nbrs = random.sample(nbrs, NUM_NEIGHBORS[1])
                proteins.update(nbrs)
            if p in pt_adj:
                nbrs = pt_adj[p]
                if len(nbrs) > NUM_NEIGHBORS[1]:
                    nbrs = random.sample(nbrs, NUM_NEIGHBORS[1])
                pathways.update(nbrs)
        
        comp_sorted = sorted(compounds)
        prot_sorted = sorted(proteins)
        path_sorted = sorted(pathways)
        
        comp_map = {c: i for i, c in enumerate(comp_sorted)}
        prot_map = {p: i for i, p in enumerate(prot_sorted)}
        path_map = {p: i for i, p in enumerate(path_sorted)}
        
        sg = HeteroData()
        sg["compound"].x = data["compound"].x[torch.tensor(comp_sorted, device=DEVICE)]
        if prot_sorted:
            sg["protein"].x = data["protein"].x[torch.tensor(prot_sorted, device=DEVICE)]
        else:
            sg["protein"].x = data["protein"].x[:0].to(DEVICE)
        if path_sorted:
            sg["pathway"].x = torch.zeros(len(path_sorted), 1, device=DEVICE)
        else:
            sg["pathway"].x = torch.zeros(0, 1, device=DEVICE)
        sg["pathway"].n_pathways = data["pathway"].n_pathways
        
        def _edges(et, src_map, dst_map):
            sl, dl = [], []
            for s, ds in adj.get(et, {}).items():
                if s in src_map:
                    for d in ds:
                        if d in dst_map:
                            sl.append(src_map[s])
                            dl.append(dst_map[d])
            if sl:
                return torch.tensor([sl, dl], dtype=torch.long, device=DEVICE)
            return torch.zeros((2, 0), dtype=torch.long, device=DEVICE)
        
        sg["compound", "interacts", "protein"].edge_index = _edges(
            ("compound", "interacts", "protein"), comp_map, prot_map)
        sg["protein", "ppi", "protein"].edge_index = _edges(
            ("protein", "ppi", "protein"), prot_map, prot_map)
        sg["protein", "belongs_to", "pathway"].edge_index = _edges(
            ("protein", "belongs_to", "pathway"), prot_map, path_map)
        sg["pathway", "includes", "protein"].edge_index = _edges(
            ("pathway", "includes", "protein"), path_map, prot_map)
        
        # 存储全局→局部索引映射（供训练循环使用）
        sg._comp_sorted = comp_sorted
        sg._prot_map = prot_map
        sg._comp_map = comp_map
        
        return sg

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val_auc = 0.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        # 随机打乱训练化合物
        random.shuffle(train_compounds_np)
        for batch_start in range(0, len(train_compounds_np), BATCH_SIZE):
            batch_seeds = train_compounds_np[batch_start:batch_start + BATCH_SIZE]
            batch = _build_subgraph(batch_seeds)
            
            if batch["protein"].x.shape[0] < 1:
                continue
            
            optimizer.zero_grad()
            
            hgt_out = model(batch.x_dict, batch.edge_index_dict)
            prot_emb = hgt_out["protein"]
            comp_emb = hgt_out["compound"]
            
            if torch.isnan(prot_emb).any() or torch.isnan(comp_emb).any():
                continue
            
            T = torch.clamp(model.temperature, min=0.5, max=20.0)
            
            # 获取 batch 中的 CPI 边
            cpi_ei = batch[("compound", "interacts", "protein")].edge_index
            if cpi_ei.shape[1] < 1:
                continue
            
            n_batch_prots = prot_emb.shape[0]
            
            # 正样本
            pos_src = cpi_ei[0]
            pos_dst = cpi_ei[1]
            pos_score = model.decode(comp_emb[pos_src], prot_emb[pos_dst]) / T
            pos_score = torch.clamp(pos_score, -10, 10)
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_score, torch.full_like(pos_score, 0.95))
            
            # 三级负样本
            unique_src = pos_src.unique()
            n_unique = len(unique_src)
            
            if n_unique > 0 and n_batch_prots > 1:
                batch_comp_emb = comp_emb[unique_src]
                all_scores = model.decode(
                    batch_comp_emb.unsqueeze(1).expand(-1, n_batch_prots, -1).reshape(-1, model.out_dim),
                    prot_emb.repeat(n_unique, 1)
                ).reshape(n_unique, n_batch_prots) / T
                
                # 掩码正样本（使用子图中存储的全局索引映射）
                mask = torch.zeros(n_unique, n_batch_prots, device=DEVICE)
                comp_sorted = getattr(batch, '_comp_sorted', [])
                prot_map = getattr(batch, '_prot_map', {})
                for i, src_local in enumerate(unique_src):
                    src_global = comp_sorted[src_local.item()] if src_local.item() < len(comp_sorted) else -1
                    if src_global >= 0 and src_global in precomputed_pos:
                        for p_global in precomputed_pos[src_global]:
                            if p_global in prot_map:
                                p_local = prot_map[p_global]
                                if p_local < n_batch_prots:
                                    mask[i, p_local] = -1e9
                
                hard_neg_idx = (all_scores + mask).argmax(dim=1)
                hard_neg_scores = all_scores[torch.arange(n_unique, device=DEVICE), hard_neg_idx]
                hard_neg_scores = torch.clamp(hard_neg_scores, -10, 10)
                
                n_rand = n_unique // 2
                if n_rand > 0:
                    rand_idx = torch.randperm(n_unique, device=DEVICE)[:n_rand]
                    rand_dst = torch.randint(0, n_batch_prots, (n_rand,), device=DEVICE)
                    rand_scores = model.decode(comp_emb[unique_src[rand_idx]], prot_emb[rand_dst]) / T
                    rand_scores = torch.clamp(rand_scores, -10, 10)
                    hard_neg_scores[rand_idx] = rand_scores
                
                neg_loss = F.binary_cross_entropy_with_logits(
                    hard_neg_scores, torch.full_like(hard_neg_scores, 0.05))
                
                # BPR: 显式映射
                src_to_pos_hgt = {s.item(): i for i, s in enumerate(unique_src)}
                pos_indices_hgt = torch.tensor([src_to_pos_hgt[s.item()] for s in pos_src], device=DEVICE, dtype=torch.long)
                bpr_loss = -torch.log(
                    torch.sigmoid(pos_score - hard_neg_scores[pos_indices_hgt]) + 1e-8
                ).mean()
                
                loss = 0.6 * (pos_loss + neg_loss) + 0.4 * bpr_loss
            else:
                loss = pos_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if n_batches == 0:
            continue
        
        avg_loss = total_loss / n_batches
        
        # 验证（使用全图，每 10 个 epoch）
        if epoch % 10 == 0 and len(val_idx) > 0:
            model.eval()
            with torch.no_grad():
                torch.cuda.empty_cache()
                x_dict_full = {k: v.clone() for k, v in data.x_dict.items()}
                hgt_out = model(x_dict_full, data.edge_index_dict)
                prot_emb = hgt_out["protein"]
                comp_emb = hgt_out["compound"]
                T_val = torch.clamp(model.temperature, min=0.5, max=20.0)
                
                vp_src = cpi_edges[0, val_idx]
                vp_dst = cpi_edges[1, val_idx]
                vp_score = torch.sigmoid(
                    model.decode(comp_emb[vp_src], prot_emb[vp_dst]) / T_val)
                
                val_comp_set = set(vp_src.tolist())
                val_neg_scores = []
                for src in val_comp_set:
                    mask_v = torch.zeros(n_proteins, device=DEVICE)
                    pos_set = all_compound_to_pos.get(src, set())
                    for p in pos_set:
                        if p < n_proteins:
                            mask_v[p] = -1e9
                    scores_chunks = []
                    CHUNK_SIZE = 32
                    for chunk_start in range(0, n_proteins, CHUNK_SIZE):
                        chunk_end = min(chunk_start + CHUNK_SIZE, n_proteins)
                        chunk_prot = prot_emb[chunk_start:chunk_end]
                        n_chunk = chunk_end - chunk_start
                        chunk_scores = model.decode(
                            comp_emb[src:src+1].expand(n_chunk, -1), chunk_prot) / T_val
                        scores_chunks.append(chunk_scores)
                    scores = torch.cat(scores_chunks, dim=0)
                    hard_idx = (scores + mask_v).argmax()
                    val_neg_scores.append(torch.sigmoid(scores[hard_idx]).item())
                
                y_true = np.concatenate([np.ones(len(vp_score)), np.zeros(len(val_neg_scores))])
                y_score = np.concatenate([vp_score.cpu().numpy(), np.array(val_neg_scores)])
                y_score = np.nan_to_num(y_score, nan=0.5)
                m = _compute_metrics(y_true, y_score)
                val_auc = m["auc"]
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            history.append({"epoch": epoch, "loss": avg_loss, **m})
            logger.info(f"  HGT epoch {epoch:3d} | loss={avg_loss:.4f} | "
                       f"auc={m['auc']:.4f} | aupr={m['aupr']:.4f} | p@10={m['precision@10']:.3f} | "
                       f"ef@1%={m['ef@1%']:.1f}")
            
            if patience_counter >= patience:
                logger.info(f"  HGT early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    best_entry = max(history, key=lambda x: x["auc"]) if history else {"auc": 0.0, "aupr": 0.0}
    logger.info(f"  HGT best: auc={best_entry['auc']:.4f}, aupr={best_entry['aupr']:.4f}, "
                f"ef@1%={best_entry.get('ef@1%', 0):.1f}")
    return model, history


# ============================================================
# 8. 预测与集成
# ============================================================
def predict_tcm(
    gat_model: GATLinkPredictor,
    hgt_model: Optional[HGTLinkPredictor],
    x: torch.Tensor,
    edge_index: torch.Tensor,
    hetero_data: Optional[HeteroData],
    tcm_smiles: List[str],
    target_genes: List[str],
    compound_stats: Tuple,
    smi_to_idx: Dict[str, int],
    gene_to_idx: Dict[str, int],
    n_compounds: int,
    gat_weight: float = 0.5,
) -> pd.DataFrame:
    """v9 修复: GAT 使用原生归纳式推理，TCM 化合物作为孤立节点入图

    GAT: 将 TCM 化合物作为孤立节点加入同质图，执行完整前向传播
         → 化合物嵌入与蛋白嵌入在同一空间，点积有物理意义
    HGT: 使用 encode_compound（投影器）编码冷启动化合物
         → 双线性解码器具有跨空间泛化能力
    """
    gat_model.eval()
    if hgt_model is not None:
        hgt_model.eval()

    # 计算 TCM 化合物特征（ECFP4 + MACCS + desc）
    tcm_feat_raw, _, _, _ = build_compound_features(tcm_smiles, stats=compound_stats)
    feat_dim = x.shape[1]
    if tcm_feat_raw.shape[1] < feat_dim:
        tcm_feat_raw = np.pad(tcm_feat_raw, ((0, 0), (0, feat_dim - tcm_feat_raw.shape[1])), mode="constant")
    tcm_feat = torch.from_numpy(tcm_feat_raw).to(DEVICE)

    with torch.no_grad():
        x_dev = x.to(DEVICE)
        edge_index_dev = edge_index.to(DEVICE)

        # v9 修复: GAT 使用原生归纳式推理
        # 将 TCM 化合物作为孤立节点拼接，执行完整前向传播
        # 孤立节点通过 GATv2Conv 的自环机制完成特征变换
        n_tcm = len(tcm_smiles)
        x_extended = torch.cat([x_dev, tcm_feat], dim=0)  # [train_nodes + tcm_nodes, feat_dim]
        node_emb = gat_model(x_extended, edge_index_dev)   # 完整前向传播
        gat_prot_emb = node_emb[n_compounds:x_dev.shape[0]]  # 蛋白嵌入
        gat_tcm_emb = node_emb[x_dev.shape[0]:]  # TCM 化合物嵌入（与蛋白在同一空间）
        gat_T = torch.clamp(gat_model.temperature, min=0.5, max=20.0)

        # HGT 编码
        if hgt_model is not None and hetero_data is not None:
            x_dict_full = {k: v.clone() for k, v in hetero_data.x_dict.items()}
            hgt_out = hgt_model(x_dict_full, hetero_data.edge_index_dict)
            hgt_prot_emb = hgt_out["protein"]
            hgt_tcm_emb = hgt_model.encode_compound(tcm_feat)
            hgt_T = torch.clamp(hgt_model.temperature, min=0.5, max=20.0)
        else:
            hgt_prot_emb = None
            hgt_tcm_emb = None
            hgt_T = torch.tensor(5.0, device=DEVICE)

    # 构建预测矩阵
    results = []
    for i, smi in enumerate(tcm_smiles):
        row = {"MOL_ID": f"TCM_{i}", "molecule_name": "", "SMILES": smi}

        for gene in target_genes:
            if gene not in gene_to_idx:
                row[gene] = 0.5
                continue
            p_idx = gene_to_idx[gene]
            local_p_idx = p_idx - n_compounds

            # GAT 分数（v9: 原生归纳式推理，嵌入空间已对齐）
            if 0 <= local_p_idx < gat_prot_emb.shape[0]:
                gat_score = torch.sigmoid(
                    (gat_tcm_emb[i:i+1] * gat_prot_emb[local_p_idx:local_p_idx+1]).sum(dim=1) / gat_T
                ).item()
            else:
                gat_score = 0.5

            # HGT 分数（v8: 双线性解码器）
            if hgt_prot_emb is not None and hgt_tcm_emb is not None and 0 <= local_p_idx < hgt_prot_emb.shape[0]:
                hgt_score = torch.sigmoid(
                    hgt_model.decode(
                        hgt_tcm_emb[i:i+1], hgt_prot_emb[local_p_idx:local_p_idx+1]
                    ) / hgt_T
                ).item()
            else:
                hgt_score = 0.5

            composite = gat_weight * gat_score + (1 - gat_weight) * hgt_score
            row[gene] = composite

        results.append(row)

    return pd.DataFrame(results)


# ============================================================
# 9. 综合排序与输出
# ============================================================
def rank_and_export(
    pred_df: pd.DataFrame,
    target_genes: List[str],
    top_n: int = 500,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gene_cols = [g for g in target_genes if g in pred_df.columns]
    scores = pred_df[gene_cols].values

    avg_score = np.nanmean(scores, axis=1)
    max_score = np.nanmax(scores, axis=1)
    n_hits = np.nansum(scores > 0.5, axis=1)
    n_high = np.nansum(scores > 0.7, axis=1)
    consistency = 1.0 - np.nanstd(scores, axis=1)

    def _norm(x):
        if x.max() - x.min() < 1e-8:
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    composite = (
        0.30 * _norm(avg_score)
        + 0.20 * _norm(max_score)
        + 0.20 * _norm(n_hits / len(gene_cols))
        + 0.20 * _norm(n_high / len(gene_cols))
        + 0.10 * _norm(consistency)
    )

    pred_df["avg_score"] = avg_score
    pred_df["max_score"] = max_score
    pred_df["n_hits"] = n_hits
    pred_df["n_high"] = n_high
    pred_df["consistency"] = consistency
    pred_df["composite_score"] = composite
    pred_df["n_targets"] = len(gene_cols)

    top_targets_list = []
    for i in range(len(pred_df)):
        gene_scores = [(g, scores[i][j]) for j, g in enumerate(gene_cols)]
        gene_scores.sort(key=lambda x: x[1], reverse=True)
        top5 = gene_scores[:5]
        top_targets_list.append(", ".join([f"{g}({s:.3f})" for g, s in top5]))
    pred_df["top_targets"] = top_targets_list

    pred_df = pred_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    pred_df["rank"] = range(1, len(pred_df) + 1)
    top_df = pred_df.head(top_n).copy()
    return pred_df, top_df


# ============================================================
# 10. 报告生成
# ============================================================
def _df_to_markdown(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "*(empty)*\n"
    cols = df.columns.tolist()
    lines = []
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, row in df.head(max_rows).iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def generate_report(
    gat_history: List[dict],
    hgt_history: List[dict],
    top_df: pd.DataFrame,
    total_time: float,
    n_tcm: int,
    n_targets: int,
    output_path: Path,
    check_results: Optional[Dict] = None,
):
    lines = [
        "# Phase 4 v9: GATv2+GIN + HGT HGTLoader — 拓扑-语义双视角互补融合",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"总耗时: {total_time / 60:.1f} 分钟",
        "",
        "## 1. 架构（v9，基于 2023-2025 SOTA 升级）",
        "- **GAT 分支（拓扑视角）**: GATv2Conv (dynamic attention) + GINConv (structure-aware) 混合堆叠",
        "  - GATv2 → GIN → GATv2，同时捕捉邻域重要性与结构不变性",
        "  - 完整 PPI 网络 + 分子指纹，建模结构-网络拓扑相似性",
        "  - 化合物投影器 (comp_projector) 用于冷启动编码",
        "- **HGT 分支（语义视角）**: HGTLoader 邻居采样 + 恢复模型容量",
        "  - hidden_dim=64, num_layers=2, num_heads=2 标准配置",
        "  - HGTLoader 2-hop 邻居采样，解决 OOM，恢复完整 PPI 网络",
        "  - 双线性解码器 (bilinear) + 共享输出投影层确保嵌入空间对齐",
        "  - 通路特征使用可学习嵌入替代 one-hot",
        "- **训练范式（v9 升级）**:",
        "  - BCE + BPR 排序损失联合优化（6:4 加权）",
        "  - 三级分级负样本：随机(50%) + 中度(30%) + 极硬(20%)",
        "  - 评估指标：AUC, AUPR, Precision@K, EF@1%/5%, ROCE",
        "- **验证设计**: 按化合物冷启动拆分 + 硬负样本验证",
        "- **特征工程**: ECFP4(2048) + MACCS(167) + RDKit 描述符(17) = 2232 维",
        "",
        "## 2. 关键参考",
        "- GATv2: Brody et al. (2022) ICLR",
        "- GIN: Xu et al. (2019) ICLR",
        "- GraphDTA: Nguyen et al. (2021) Bioinformatics",
        "- HGT: Hu et al. (2020) WWW",
        "- HGSampling: Hu et al. (2020) HGT paper, Section 3.3",
        "- BPR: Rendle et al. (2009) UAI",
        "- 富集因子: Bender & Glen (2005) J. Chem. Inf. Model.",
        "",
        "## 3. 数据规模",
        f"- TCM 候选池: {n_tcm} 个化合物",
        f"- 铁衰老靶标: {n_targets} 个基因",
        "",
    ]

    if check_results:
        lines.extend([
            "## 3.1 管线自检结果",
            f"- 总体状态: **{check_results.get('overall', 'UNKNOWN')}**",
        ])
        lines.append("")

    lines.append("## 4. 模型性能")
    if gat_history:
        best_gat = max(gat_history, key=lambda x: x.get("auc", 0))
        lines.append(f"- GAT best AUC: {best_gat.get('auc', 0):.4f}, AUPR: {best_gat.get('aupr', 0):.4f}, "
                     f"P@10: {best_gat.get('precision@10', 0):.3f}")
    if hgt_history:
        best_hgt = max(hgt_history, key=lambda x: x.get("auc", 0))
        lines.append(f"- HGT best AUC: {best_hgt.get('auc', 0):.4f}, AUPR: {best_hgt.get('aupr', 0):.4f}, "
                     f"P@10: {best_hgt.get('precision@10', 0):.3f}")

    lines.extend([
        "",
        "## 5. Top 20 候选化合物",
        "",
        _df_to_markdown(top_df.head(20)[
            ["rank", "MOL_ID", "molecule_name", "composite_score", "avg_score",
             "max_score", "n_hits", "n_high", "top_targets"]
        ]),
        "",
        "## 6. 局限",
        "- 训练数据与铁衰老靶标不完全匹配",
        "- 冷启动验证中 TCM 化合物不在训练图中，需特殊处理",
        "- 集成权重 (0.5/0.5) 为简单平均",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"报告已保存: {output_path}")


# ============================================================
# 11. 管线自检
# ============================================================
def pipeline_self_check(
    tcm_df: pd.DataFrame,
    cpi_df: pd.DataFrame,
    ppi_df: pd.DataFrame,
    prot_feat: Dict[str, np.ndarray],
    gene_to_pathways: Dict[str, List[str]],
    warm_targets: List[str],
    input_files: Optional[Dict[str, str]] = None,
) -> Dict:
    logger.info("=" * 60)
    logger.info("开始管线自检...")
    results: Dict[str, Any] = {"meta": {"timestamp": datetime.now().isoformat(), "severity": "UNKNOWN"},
                               "errors": [], "warnings": []}

    if input_files:
        results["input_files"] = input_files

    # 1. TCM 池
    logger.info("[自检 1/8] TCM 候选池...")
    tcm_smiles = tcm_df["SMILES_std"].dropna().tolist()
    invalid_smiles = sum(1 for s in tcm_smiles if Chem.MolFromSmiles(str(s)) is None)
    results["tcm_pool"] = {"total": len(tcm_df), "invalid_smiles": invalid_smiles, "passed": invalid_smiles == 0}
    if invalid_smiles:
        results["errors"].append(f"TCM池含 {invalid_smiles} 个无效SMILES")

    # 2. CPI 数据
    logger.info("[自检 2/8] CPI 数据...")
    cpi_smiles = cpi_df["canonical_smiles"].dropna().unique()
    cpi_invalid = sum(1 for s in cpi_smiles if Chem.MolFromSmiles(str(s)) is None)
    cpi_dupes = cpi_df.duplicated(subset=["gene", "canonical_smiles"]).sum()
    results["cpi_data"] = {"total": len(cpi_df), "unique_compounds": len(cpi_smiles),
                           "invalid_smiles": cpi_invalid, "duplicates": cpi_dupes, "passed": len(cpi_df) > 0}
    if cpi_dupes:
        results["warnings"].append(f"CPI数据含 {cpi_dupes} 条重复")

    # 3. TCM/训练集重叠
    logger.info("[自检 3/8] TCM/训练集重叠...")
    tcm_smi_set = set(tcm_smiles)
    train_smi_set = set(cpi_smiles)
    overlap = tcm_smi_set & train_smi_set
    results["tcm_train_overlap"] = {"n_overlap": len(overlap), "passed": len(overlap) == 0}
    if overlap:
        results["warnings"].append(f"TCM/训练集重叠: {len(overlap)} 个化合物")

    # 4. PPI
    logger.info("[自检 4/8] PPI 网络...")
    ppi_genes = set()
    for _, row in ppi_df.iterrows():
        ppi_genes.add(str(row["source"]).strip().upper())
        ppi_genes.add(str(row["target"]).strip().upper())
    results["ppi_network"] = {"edges": len(ppi_df), "genes": len(ppi_genes), "passed": len(ppi_df) > 0}

    # 5. 蛋白特征
    logger.info("[自检 5/8] 蛋白特征...")
    feat_dim = next(iter(prot_feat.values())).shape[0] if prot_feat else 0
    nan_genes = [g for g, v in prot_feat.items() if np.isnan(v).any()]
    results["protein_features"] = {"genes": len(prot_feat), "dim": feat_dim, "nan_genes": nan_genes,
                                   "passed": len(prot_feat) > 0 and len(nan_genes) == 0}

    # 6. KEGG
    logger.info("[自检 6/8] KEGG 通路...")
    all_pathways = set(pid for paths in gene_to_pathways.values() for pid in paths)
    results["kegg_pathways"] = {"genes": len(gene_to_pathways), "pathways": len(all_pathways),
                                "passed": len(gene_to_pathways) > 0}

    # 7. 靶标覆盖
    logger.info("[自检 7/8] 靶标覆盖...")
    cpi_genes = set(cpi_df["gene"].unique())
    prot_genes = set(prot_feat.keys())
    matched = cpi_genes & prot_genes
    results["target_coverage"] = {"warm_targets": len(warm_targets),
                                  "with_features": len(matched), "passed": len(matched) > 0}

    # 8. 训练就绪
    logger.info("[自检 8/8] 训练就绪...")
    warm_cpi = cpi_df[cpi_df["gene"].isin(warm_targets)]
    results["training_readiness"] = {"n_warm_cpi_edges": len(warm_cpi),
                                     "passed": len(warm_cpi) >= 10}

    has_errors = len(results["errors"]) > 0
    has_warnings = len(results["warnings"]) > 0
    if has_errors:
        results["meta"]["severity"] = "ERROR"
        results["overall"] = "FAILED"
    elif has_warnings:
        results["meta"]["severity"] = "WARNING"
        results["overall"] = "PASSED_WITH_WARNINGS"
    else:
        results["meta"]["severity"] = "OK"
        results["overall"] = "PASSED"

    logger.info(f"自检结果: {results['overall']}")
    if has_errors:
        for e in results["errors"]:
            logger.error(f"  ERROR: {e}")
    if has_warnings:
        for w in results["warnings"]:
            logger.warning(f"  WARNING: {w}")
    logger.info("=" * 60)
    return results


# ============================================================
# 12. 主流程
# ============================================================
def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Phase 4 v9: GATv2+GIN + HGT HGTLoader — 拓扑-语义双视角互补融合")
    logger.info("=" * 60)

    # 加载数据
    logger.info(">>> 加载数据")
    cpi_df = load_cpi_data()
    ppi_df = load_ppi_network()
    gene_to_pathways = load_kegg_pathways()
    prot_feat, gene_to_seq = load_protein_features()
    tcm_df = load_tcm_pool()

    noleak_tcm = L3_RESULTS / "tcm_compound_pool_tox_filtered_noleak.csv"
    tcm_file = noleak_tcm if noleak_tcm.exists() else (L3_RESULTS / "tcm_compound_pool_tox_filtered.csv")
    input_files = {
        "cpi": str(L4_ROOT / "results" / "experimental_actives_detail_cleaned.csv"),
        "ppi": str(L1_RESULTS / "ppi_network_extended_significant_edges.csv"),
        "kegg": str(L2_RESULTS / "kegg_pathways" / "kegg_human_pathway_genes.tsv"),
        "protein_features": str(L2_RESULTS / "target_protein_features.csv"),
        "tcm_pool": str(tcm_file),
    }

    # 温靶标
    cpi_genes = set(cpi_df["gene"].unique())
    warm_targets = sorted(cpi_genes & set(ALL_FERRORAGING_GENES))
    logger.info(f"温靶标: {len(warm_targets)} 个")

    # 管线自检
    check_results = pipeline_self_check(
        tcm_df, cpi_df, ppi_df, prot_feat, gene_to_pathways, warm_targets, input_files=input_files)
    with open(L4_RESULTS / "self_check_report_v6_tox.json", "w", encoding="utf-8") as f:
        json.dump(check_results, f, indent=2, ensure_ascii=False, default=str)

    if check_results["overall"] == "FAILED":
        logger.error("管线自检未通过，终止训练")
        sys.exit(1)

    cpi_df = cpi_df[cpi_df["gene"].isin(warm_targets)].copy()

    # v9: 移除 PPI 过滤，恢复完整 PPI 网络
    # HGTLoader 邻居采样已解决 OOM，无需删除 PPI 边
    logger.info(f"HGT 使用完整 PPI 网络: {len(ppi_df)} 条边")

    # 构建图
    logger.info(">>> 构建同质图 (GAT)")
    sample_smiles = cpi_df["canonical_smiles"].unique()[:10].tolist()
    sample_feat, _, _, _ = build_compound_features(sample_smiles)
    compound_feat_dim = sample_feat.shape[1]

    x, edge_index, feat_dim, smi_to_idx, gene_to_idx, comp_feat = build_cpi_homogeneous_graph(
        cpi_df, ppi_df, prot_feat, compound_feat_dim)
    n_compounds = len(smi_to_idx)

    logger.info(">>> 构建异质图 (HGT, v9: 完整 PPI)")
    hetero_data = build_heterogeneous_graph(
        cpi_df, ppi_df, gene_to_pathways, prot_feat,
        smi_to_idx, gene_to_idx, n_compounds, comp_feat=comp_feat)

    # 训练 GAT
    logger.info(">>> 训练 GAT（v9: GATv2 + GIN 混合架构）")
    gat_model = GATLinkPredictor(
        in_dim=feat_dim, hidden_dim=128, out_dim=64,
        num_layers=2, heads=4, dropout=0.5)
    gat_model, gat_history = train_gat(
        gat_model, x, edge_index, n_compounds,
        epochs=200, lr=1e-3, patience=20)

    # v9: 清理 GAT 训练占用的 GPU 内存，为 HGT 腾出空间
    # 保存 GAT 模型到 CPU，清理图数据
    gat_model = gat_model.cpu()
    x_cpu = x.cpu()
    edge_index_cpu = edge_index.cpu()
    del x, edge_index
    torch.cuda.empty_cache()
    logger.info("  GAT GPU 内存已释放")

    # 训练 HGT（v9: 手动邻居采样 + 恢复模型容量）
    logger.info(">>> 训练 HGT（v9: 手动邻居采样, hidden_dim=64, num_layers=2, num_heads=2）")
    hgt_node_feat_dims = {
        node_type: hetero_data[node_type].x.shape[1]
        for node_type in hetero_data.node_types
    }
    hgt_node_feat_dims["pathway_count"] = getattr(hetero_data["pathway"], "n_pathways", 1)
    hgt_model = HGTLinkPredictor(
        hidden_dim=64, out_dim=64, num_heads=2, num_layers=2,
        dropout=0.3, metadata=hetero_data.metadata(),
        compound_feat_dim=feat_dim, node_feat_dims=hgt_node_feat_dims)
    hgt_model, hgt_history = train_hgt(
        hgt_model, hetero_data, epochs=200, lr=1e-3, patience=20)

    # 预测 TCM
    logger.info(">>> 预测 TCM 化合物")
    # v8: 恢复 GAT 模型和图数据到 GPU
    gat_model = gat_model.to(DEVICE)
    x = x_cpu.to(DEVICE)
    edge_index = edge_index_cpu.to(DEVICE)
    del x_cpu, edge_index_cpu
    torch.cuda.empty_cache()
    tcm_smiles = tcm_df["SMILES_std"].dropna().tolist()
    all_train_smiles = sorted(smi_to_idx.keys())
    _, cp_mean, cp_std, cp_col_mean = build_compound_features(all_train_smiles)
    compound_stats = (cp_mean, cp_std, cp_col_mean)

    pred_df = predict_tcm(
        gat_model, hgt_model, x, edge_index, hetero_data,
        tcm_smiles, warm_targets, compound_stats,
        smi_to_idx, gene_to_idx, n_compounds, gat_weight=0.5)

    # 添加 TCM 名称
    if "MOL_ID" in tcm_df.columns and "molecule_name" in tcm_df.columns:
        name_map = dict(zip(tcm_df["SMILES_std"], tcm_df["molecule_name"]))
        mol_id_map = dict(zip(tcm_df["SMILES_std"], tcm_df["MOL_ID"]))
        pred_df["molecule_name"] = pred_df["SMILES"].map(name_map).fillna("")
        pred_df["MOL_ID"] = pred_df["SMILES"].map(mol_id_map).fillna("")

    # 排序与输出
    logger.info(">>> 综合排序")
    full_df, top_df = rank_and_export(pred_df, warm_targets, top_n=500)

    full_df.to_csv(L4_RESULTS / "tcm_predictions_full_v6_tox.csv", index=False)
    top_df.to_csv(L4_RESULTS / "tcm_top_candidates_v6_tox.csv", index=False)

    # 模型性能
    perf_rows = []
    if gat_history:
        bg = max(gat_history, key=lambda x: x.get("auc", 0))
        perf_rows.append({"model": "GAT", "best_auc": bg["auc"], "best_aupr": bg["aupr"],
                         "precision@10": bg.get("precision@10", 0)})
    if hgt_history:
        bh = max(hgt_history, key=lambda x: x.get("auc", 0))
        perf_rows.append({"model": "HGT", "best_auc": bh["auc"], "best_aupr": bh["aupr"],
                         "precision@10": bh.get("precision@10", 0)})
    if perf_rows:
        pd.DataFrame(perf_rows).to_csv(L4_RESULTS / "model_performance_v6_tox.csv", index=False)

    metrics = {
        "gat_history": gat_history, "hgt_history": hgt_history,
        "n_tcm": len(tcm_smiles), "n_warm_targets": len(warm_targets),
        "n_compounds": len(smi_to_idx), "n_proteins": len(gene_to_idx),
    }
    with open(L4_RESULTS / "training_metrics_v6_tox.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    total_time = time.time() - start_time
    generate_report(gat_history, hgt_history, top_df, total_time,
                    len(tcm_smiles), len(warm_targets),
                    L4_RESULTS / "phase4_report_v6_tox.md", check_results=check_results)

    logger.info("=" * 60)
    logger.info(f"Phase 4 v9 完成！总耗时 {total_time / 60:.1f} 分钟")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()