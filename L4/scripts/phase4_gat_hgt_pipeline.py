#!/usr/bin/env python3
"""
Phase 4 v6: GAT + HGT 双图神经网络集成 — 铁衰老靶标多靶标潜力粗筛
=====================================================================
架构：
  GAT 分支（同质图）：化合物 + 蛋白节点，边 = 实验 CPI，捕捉局部结合模式
  HGT 分支（异质图）：化合物、蛋白、KEGG 通路三种节点，通过 PPI 与通路边
                     捕捉多靶标协同模式（元路径：化合物→蛋白→通路→蛋白）
  集成：两模型预测概率加权平均 → composite_score 排序 → Top 500 候选分子

训练数据：
  ChEMBL / BindingDB 实验验证 CPI（不依赖铁衰老核心靶标已知配体）
  模型通过图结构（PPI、KEGG 通路）跨靶标泛化。

关键参考：
  - GAT: Velickovic et al. (2018) "Graph Attention Networks", ICLR.
    https://arxiv.org/abs/1710.10903
    PyG 实现: torch_geometric.nn.GATConv
  - HGT: Hu et al. (2020) "Heterogeneous Graph Transformer", WWW.
    https://arxiv.org/abs/2003.01332
    PyG 实现: torch_geometric.nn.HGTConv
    官方代码: https://github.com/acbull/pyHGT
  - PyTorch Geometric: https://github.com/pyg-team/pytorch_geometric
  - ECFP4: Rogers & Hahn (2010) J. Chem. Inf. Model. 50(5):742-754
  - MACCS keys: MDL Information Systems (now BIOVIA)
  - RDKit: Landrum G., https://github.com/rdkit/rdkit
  - STRING PPI: Szklarczyk et al. (2023) Nucleic Acids Res. 51(D1):D638-D646
  - ChEMBL: Mendez et al. (2019) Nucleic Acids Res. 47(D1):D930-D940
  - BindingDB: Gilson et al. (2016) Nucleic Acids Res. 44(D1):D1045-D1053

输出：
  L4/results_v6_toxfiltered/model_performance_v6_tox.csv
  L4/results_v6_toxfiltered/tcm_predictions_full_v6_tox.csv
  L4/results_v6_toxfiltered/tcm_top_candidates_v6_tox.csv
  L4/results_v6_toxfiltered/enrichment_analysis_v6_tox.csv
  L4/results_v6_toxfiltered/training_metrics_v6_tox.json
  L4/results_v6_toxfiltered/phase4_report_v6_tox.md
  L4/results_v6_toxfiltered/self_check_report_v6_tox.json
  L4/logs/phase4_gat_hgt_pipeline.log
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
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.metrics import roc_auc_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv

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
    # fallback
    ALL_FERRORAGING_GENES = sorted([
        "ABCC1", "ACVR1B", "ACSL4", "ALOX15", "ATF3", "ATG3", "BAP1", "BCL6",
        "BRD7", "CD74", "CISD1", "CTSB", "CXCL10", "CYBB", "DYRK1A", "EGR1",
        "EMP1", "EPHA4", "FBXO31", "FTH1", "FTL", "GMFB", "GPX4", "HBP1",
        "HMOX1", "IGFBP7", "IL1B", "IRF1", "KDM6B", "KLF6", "LACTB", "LCN2",
        "LGMN", "LPCAT3", "MAP1LC3B", "MAPK1", "MTOR", "NFE2L2", "NOX4",
        "PDE4B", "PTGS2", "RELA", "RUNX3", "SAT1", "SLC3A2", "SLC7A11",
        "SOD1", "SP1", "SQSTM1", "STAT3", "TFRC", "TLR4", "TP53", "VDAC2",
        "VDAC3", "ACSL3", "ALOX5", "ATG7", "BECN1", "HIF1A", "KEAP1",
        "NFKB1",
    ])
logger.info(f"铁衰老靶标: {len(ALL_FERRORAGING_GENES)} 个基因")

# ============================================================
# 1. 化合物特征工程
# ============================================================
RDKIT_DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "MolMR", "TPSA",
    "NumHAcceptors", "NumHDonors", "NumRotatableBonds",
    "HeavyAtomCount", "NumAromaticRings", "NumAliphaticRings",
    "NumHeteroatoms", "NumValenceElectrons", "NHOHCount", "NOCount",
    "RingCount", "FractionCSP3", "BalabanJ",
]


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
    ecfp4: Optional[np.ndarray] = None,
    stats: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ECFP4 + MACCS（保持 0/1 二值）+ RDKit 描述符（Z-score 标准化）。
    """
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

    if ecfp4 is not None:
        features = np.hstack([ecfp4, maccs, desc]).astype(np.float32)
    else:
        features = np.hstack([maccs, desc]).astype(np.float32)
    return features, mean, std, col_mean


# ============================================================
# 2. 蛋白特征：AAC (20 dims) + PseAAC (from L2)
# ============================================================
def compute_aac(sequences: List[str]) -> np.ndarray:
    """20 种氨基酸组成 (AAC)"""
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
    """加载 ChEMBL/BindingDB 实验验证 CPI 数据"""
    cpi_path = L4_ROOT / "results" / "experimental_actives_detail.csv"
    if not cpi_path.exists():
        logger.error(f"CPI 数据文件不存在: {cpi_path}")
        sys.exit(1)
    df = pd.read_csv(cpi_path)
    # 确保必需列存在
    required = ["gene", "canonical_smiles", "uniprot_id"]
    for col in required:
        if col not in df.columns:
            logger.error(f"CPI 数据缺少列: {col}")
            sys.exit(1)
    # 过滤无效 SMILES
    df = df[df["canonical_smiles"].notna()].copy()
    df = df[df["canonical_smiles"].str.strip() != ""].copy()
    logger.info(f"CPI 数据: {len(df)} 条记录, {df['gene'].nunique()} 个基因, "
                f"{df['canonical_smiles'].nunique()} 个唯一 SMILES")
    return df


def load_ppi_network() -> pd.DataFrame:
    """加载 PPI 网络"""
    ppi_path = L1_RESULTS / "ppi_network_edges.csv"
    if not ppi_path.exists():
        logger.warning(f"PPI 网络文件不存在: {ppi_path}")
        return pd.DataFrame(columns=["source", "target", "weight"])
    df = pd.read_csv(ppi_path)
    logger.info(f"PPI 网络: {len(df)} 条边")
    return df


def load_kegg_pathways() -> Dict[str, List[str]]:
    """加载 KEGG 通路注释 (gene -> pathway_ids)"""
    enrich_path = L1_RESULTS / "string_enrichment.csv"
    if not enrich_path.exists():
        logger.warning(f"KEGG 通路文件不存在: {enrich_path}")
        return {}
    df = pd.read_csv(enrich_path)
    kegg = df[df["category"] == "KEGG"].copy()
    gene_to_pathways: Dict[str, List[str]] = {}
    for _, row in kegg.iterrows():
        pathway_id = row["term"]
        genes_str = row["inputGenes"]
        try:
            genes = eval(genes_str)  # 解析 Python list 字符串
        except Exception:
            continue
        for g in genes:
            g = g.strip().upper()
            if g not in gene_to_pathways:
                gene_to_pathways[g] = []
            if pathway_id not in gene_to_pathways[g]:
                gene_to_pathways[g].append(pathway_id)
    logger.info(f"KEGG 通路: {len(gene_to_pathways)} 个基因有通路注释, "
                f"{kegg['term'].nunique()} 个唯一通路")
    return gene_to_pathways


def load_protein_features() -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """加载蛋白特征 (AAC + PseAAC) 和序列"""
    pf_path = L2_RESULTS / "target_protein_features.csv"
    pseaac_path = L2_RESULTS / "protein_pseaac.csv"

    prot_feat: Dict[str, np.ndarray] = {}
    gene_to_seq: Dict[str, str] = {}

    # 加载蛋白序列
    if pf_path.exists():
        df = pd.read_csv(pf_path)
        for _, row in df.iterrows():
            gene = str(row["gene_symbol"]).strip().upper()
            seq = str(row["sequence"]) if pd.notna(row["sequence"]) else ""
            gene_to_seq[gene] = seq

    # 计算 AAC
    genes = list(gene_to_seq.keys())
    seqs = [gene_to_seq[g] for g in genes]
    aac = compute_aac(seqs)

    # 加载 PseAAC
    pseaac_data: Dict[str, np.ndarray] = {}
    if pseaac_path.exists():
        df_pseaac = pd.read_csv(pseaac_path)
        # 排除 Unnamed: 0（pandas 保存的索引列）
        if "Unnamed: 0" in df_pseaac.columns:
            df_pseaac = df_pseaac.drop(columns=["Unnamed: 0"])
        if "gene_symbol" in df_pseaac.columns:
            for _, row in df_pseaac.iterrows():
                g = str(row["gene_symbol"]).strip().upper()
                vals = row.drop("gene_symbol").values.astype(np.float32)
                pseaac_data[g] = vals

    # 先确定全局 PseAAC 维度
    pseaac_dim = 0
    if pseaac_data:
        pseaac_dim = len(next(iter(pseaac_data.values())))

    # 拼接 AAC + PseAAC（统一维度）
    for i, g in enumerate(genes):
        aac_vec = aac[i]
        if g in pseaac_data:
            prot_feat[g] = np.concatenate([aac_vec, pseaac_data[g]])
        elif pseaac_dim > 0:
            prot_feat[g] = np.concatenate([aac_vec, np.zeros(pseaac_dim, dtype=np.float32)])
        else:
            prot_feat[g] = aac_vec

    if not prot_feat:
        # fallback: 纯 AAC
        for i, g in enumerate(genes):
            prot_feat[g] = aac[i]

    logger.info(f"蛋白特征: {len(prot_feat)} 个基因, dim={next(iter(prot_feat.values())).shape[0]}")
    return prot_feat, gene_to_seq


def load_tcm_pool() -> pd.DataFrame:
    """加载 TCM 候选化合物池（毒性过滤后）"""
    tcm_path = L3_RESULTS / "tcm_compound_pool_tox_filtered.csv"
    if not tcm_path.exists():
        logger.error(f"TCM 候选池文件不存在: {tcm_path}")
        sys.exit(1)
    df = pd.read_csv(tcm_path)
    logger.info(f"TCM 候选池: {len(df)} 个化合物")
    return df


# ============================================================
# 4. 图构建
# ============================================================
def build_cpi_homogeneous_graph(
    cpi_df: pd.DataFrame,
    ppi_df: pd.DataFrame,
    prot_feat: Dict[str, np.ndarray],
    compound_feat_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int], Dict[str, int]]:
    """
    构建同质图（GAT 用）：
      节点 = 化合物 ∪ 蛋白
      边 = 实验 CPI + PPI（蛋白-蛋白）
    """
    # 收集所有化合物 SMILES 和蛋白基因
    all_smiles = sorted(cpi_df["canonical_smiles"].unique())
    all_genes = sorted(set(cpi_df["gene"].unique()) | set(prot_feat.keys()))

    # 映射
    smi_to_idx = {s: i for i, s in enumerate(all_smiles)}
    gene_to_idx = {g: i + len(all_smiles) for i, g in enumerate(all_genes)}

    n_compounds = len(all_smiles)
    n_proteins = len(all_genes)
    n_total = n_compounds + n_proteins

    # 化合物特征
    logger.info(f"  computing compound features ({n_compounds} compounds)...")
    comp_feat, _, _, _ = build_compound_features(all_smiles)
    compound_feat_dim_actual = comp_feat.shape[1]

    # 蛋白特征
    prot_feat_dim = next(iter(prot_feat.values())).shape[0] if prot_feat else 20
    prot_matrix = np.zeros((n_proteins, prot_feat_dim), dtype=np.float32)
    for gene, idx_offset in gene_to_idx.items():
        idx = idx_offset - n_compounds
        if gene in prot_feat:
            prot_matrix[idx] = prot_feat[gene]

    # 投影到统一维度
    feat_dim = max(compound_feat_dim_actual, prot_feat_dim)
    if feat_dim != compound_feat_dim_actual:
        comp_feat = np.pad(comp_feat, ((0, 0), (0, feat_dim - compound_feat_dim_actual)), mode="constant")
    if feat_dim != prot_feat_dim:
        prot_matrix = np.pad(prot_matrix, ((0, 0), (0, feat_dim - prot_feat_dim)), mode="constant")

    # 节点特征
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

    if not edge_index_list:
        logger.error("无有效边！")
        sys.exit(1)

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    # NaN 检测
    if torch.isnan(x).any():
        nan_count = torch.isnan(x).sum().item()
        logger.error(f"同质图特征含 NaN: {nan_count} / {x.numel()} 个值")
        # 用 0 填充 NaN
        x = torch.nan_to_num(x, nan=0.0)

    logger.info(f"同质图: {n_total} 节点 ({n_compounds} compounds + {n_proteins} proteins), "
                f"{edge_index.shape[1]} 边 (CPI + {n_ppi} PPI), feat_dim={feat_dim}")
    return x, edge_index, torch.tensor(feat_dim), smi_to_idx, gene_to_idx


def build_heterogeneous_graph(
    cpi_df: pd.DataFrame,
    ppi_df: pd.DataFrame,
    gene_to_pathways: Dict[str, List[str]],
    prot_feat: Dict[str, np.ndarray],
    smi_to_idx: Dict[str, int],
    gene_to_idx: Dict[str, int],
    n_compounds: int,
    compound_feat_dim: int,
) -> HeteroData:
    """
    构建异质图（HGT 用）：
      节点类型: compound, protein, pathway
      边类型: (compound, interacts, protein), (protein, ppi, protein),
              (protein, belongs_to, pathway)
    """
    data = HeteroData()

    # 蛋白特征
    prot_feat_dim = next(iter(prot_feat.values())).shape[0] if prot_feat else 20
    n_proteins = len(gene_to_idx)
    prot_matrix = np.zeros((n_proteins, prot_feat_dim), dtype=np.float32)
    gene_list = [""] * n_proteins
    for gene, idx in gene_to_idx.items():
        local_idx = idx - n_compounds
        if 0 <= local_idx < n_proteins:
            gene_list[local_idx] = gene
            if gene in prot_feat:
                prot_matrix[local_idx] = prot_feat[gene]

    # 检测蛋白特征 NaN
    if np.isnan(prot_matrix).any():
        nan_count = np.isnan(prot_matrix).sum().item()
        nan_rows = np.where(np.isnan(prot_matrix).any(axis=1))[0]
        logger.error(f"异质图蛋白特征含 NaN: {nan_count} / {prot_matrix.size} 个值, "
                     f"受影响行: {nan_rows.tolist()}")
        # 填充 NaN
        prot_matrix = np.nan_to_num(prot_matrix, nan=0.0)

    # 通路节点
    all_pathways = sorted(set(
        pid for paths in gene_to_pathways.values() for pid in paths
    ))
    pathway_to_idx = {p: i for i, p in enumerate(all_pathways)}
    n_pathways = len(all_pathways)
    # 通路特征：one-hot
    pathway_feat = np.eye(n_pathways, dtype=np.float32) if n_pathways > 0 else np.zeros((1, 1), dtype=np.float32)

    # 化合物特征
    all_smiles = sorted(smi_to_idx.keys())
    comp_feat, _, _, _ = build_compound_features(all_smiles)

    data["compound"].x = torch.from_numpy(comp_feat)
    data["protein"].x = torch.from_numpy(prot_matrix)
    if n_pathways > 0:
        data["pathway"].x = torch.from_numpy(pathway_feat)
    else:
        data["pathway"].x = torch.zeros((1, 1), dtype=torch.float32)

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
        # 添加反向边：pathway → protein，使通路信息能回流到蛋白节点
        rev_pt_edges = [pt_edges[1][:], pt_edges[0][:]]
        data["pathway", "includes", "protein"].edge_index = torch.tensor(rev_pt_edges, dtype=torch.long)
    else:
        data["protein", "belongs_to", "pathway"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["pathway", "includes", "protein"].edge_index = torch.zeros((2, 0), dtype=torch.long)

    logger.info(
        f"异质图: compound({data['compound'].x.shape[0]}) "
        f"protein({data['protein'].x.shape[0]}) "
        f"pathway({data['pathway'].x.shape[0]}) | "
        f"CPI edges: {data['compound','interacts','protein'].edge_index.shape[1]} | "
        f"PPI edges: {data['protein','ppi','protein'].edge_index.shape[1]} | "
        f"Pathway edges: {data['protein','belongs_to','pathway'].edge_index.shape[1]}"
    )
    return data


# ============================================================
# 5. GAT 模型（同质图链接预测）
# ============================================================
class GATLinkPredictor(nn.Module):
    """MLP 编码器（化合物 + 蛋白独立 MLP）+ 点积解码器
    注意：虽然保持 GAT 命名，但蛋白编码器使用 MLP 而非图卷积，
    因为 PPI 网络过于稀疏（42 蛋白仅 53 条边），图卷积无实际增益。
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 化合物编码器：独立 MLP + LayerNorm
        self.comp_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        # 蛋白编码器：独立 MLP + LayerNorm
        self.prot_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def encode_compound(self, x_comp: torch.Tensor) -> torch.Tensor:
        return self.comp_encoder(x_comp)

    def encode_protein(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        """蛋白编码（edge_index 参数保留以兼容接口，实际不使用图结构）"""
        return self.prot_encoder(x)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        return self.encode_protein(x, edge_index)


# ============================================================
# 6. HGT 模型（异质图链接预测）
# ============================================================
class HGTLinkPredictor(nn.Module):
    """HGT 编码器 + 独立 MLP 化合物编码器 + 点积解码器

    HGTConv 内部 skip connection 要求所有节点类型维度一致，因此蛋白和化合物
    特征均先通过投影层统一到 hidden_dim，再送入 HGTConv。
    化合物编码器使用独立 MLP，确保新化合物（不在训练图结构中的 TCM 化合物）
    也能获得有效嵌入。
    """

    def __init__(self, hidden_dim: int = 256, out_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.3,
                 metadata=None, compound_feat_dim: int = 200,
                 node_feat_dims: Optional[Dict[str, int]] = None):
        super().__init__()
        self.out_dim = out_dim

        # 化合物编码器：独立 MLP（支持新化合物）
        self.comp_encoder = nn.Sequential(
            nn.Linear(compound_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        # 化合物特征投影层：将原始维度投影到 hidden_dim
        self.comp_proj = nn.Sequential(
            nn.Linear(compound_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # 蛋白特征投影层：将原始蛋白维度投影到 hidden_dim（关键修复）
        # 解决 HGTConv skip connection 维度不匹配导致的 NaN 问题
        self.prot_proj_in = nn.Sequential(
            nn.Linear(node_feat_dims.get("protein", 71) if node_feat_dims else 71, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # 通路特征投影层：将通路 one-hot 维度投影到 hidden_dim
        self.path_proj_in = nn.Sequential(
            nn.Linear(node_feat_dims.get("pathway", 156) if node_feat_dims else 156, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # HGT 层：所有节点类型统一为 hidden_dim
        self.convs = nn.ModuleList()
        self.node_types_for_hgt = []
        if metadata and node_feat_dims:
            node_types, edge_types = metadata
            self.node_types_for_hgt = node_types
            for layer_idx in range(num_layers):
                self.convs.append(HGTConv(
                    {nt: hidden_dim for nt in node_types},
                    hidden_dim, metadata,
                    heads=num_heads,
                ))

        # 蛋白投影层：将 HGT 输出的 hidden_dim 投影到 out_dim，与化合物编码器对齐
        self.prot_proj = nn.Linear(hidden_dim, out_dim)

        self.dropout = nn.Dropout(dropout)

    def encode_compound(self, x_comp: torch.Tensor) -> torch.Tensor:
        return self.comp_encoder(x_comp)

    def forward(self, x_dict, edge_index_dict):
        """全图前向：先投影所有节点类型到 hidden_dim，再 HGTConv 处理"""
        x_dict = {k: v for k, v in x_dict.items()}

        # 将所有节点类型投影到统一的 hidden_dim
        if "compound" in x_dict:
            x_dict["compound"] = self.comp_proj(x_dict["compound"])
        if "protein" in x_dict:
            x_dict["protein"] = self.prot_proj_in(x_dict["protein"])
        if "pathway" in x_dict:
            x_dict["pathway"] = self.path_proj_in(x_dict["pathway"])

        # 中间 NaN 检测
        for nt, feat in x_dict.items():
            if torch.isnan(feat).any():
                logger.error(f"HGT forward: NaN after projection in {nt}")

        for i, conv in enumerate(self.convs):
            out = conv(x_dict, edge_index_dict)
            # 中间 NaN 检测
            for nt, feat in out.items():
                if torch.isnan(feat).any():
                    logger.error(f"HGT forward: NaN after HGTConv layer {i} in {nt}")
            # HGTConv 只返回有消息流入的节点类型，手动保留未更新的节点
            for nt in x_dict:
                if nt not in out:
                    out[nt] = x_dict[nt]
            x_dict = out

        # 投影蛋白嵌入到 out_dim，与化合物编码器对齐
        if "protein" in x_dict:
            x_dict["protein"] = self.prot_proj(x_dict["protein"])

        return x_dict


# ============================================================
# 7. 训练
# ============================================================
def train_gat(
    model: GATLinkPredictor,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    n_compounds: int,
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 20,
) -> Tuple[GATLinkPredictor, List[float]]:
    """训练 GAT 链接预测（化合物 + 蛋白均为独立 MLP）"""
    model = model.to(DEVICE)

    # Xavier 初始化
    for p in model.parameters():
        if p.dim() >= 2:
            nn.init.xavier_uniform_(p)

    x = x.to(DEVICE)
    edge_index = edge_index.to(DEVICE)

    n_total = x.shape[0]
    n_proteins = n_total - n_compounds

    # 分割边（化合物→蛋白）
    cpi_mask = (edge_index[0] < n_compounds) & (edge_index[1] >= n_compounds)
    cpi_indices = torch.where(cpi_mask)[0]
    num_cpi = len(cpi_indices)
    if num_cpi < 10:
        logger.warning(f"GAT CPI 边太少 ({num_cpi})，跳过训练")
        return model, []

    perm = torch.randperm(num_cpi, device=DEVICE)
    n_train = int(num_cpi * 0.85)
    train_idx = cpi_indices[perm[:n_train]]
    val_idx = cpi_indices[perm[n_train:]]

    prot_x = x[n_compounds:].clone()
    comp_x = x[:n_compounds].clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val_auc = 0.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # 蛋白嵌入（MLP，不使用图结构）
        prot_emb = model.encode_protein(prot_x)
        # 化合物嵌入（MLP）
        comp_emb = model.encode_compound(comp_x)

        if torch.isnan(prot_emb).any() or torch.isnan(comp_emb).any():
            logger.error(f"GAT epoch {epoch}: NaN in embeddings! "
                         f"prot_emb NaN: {torch.isnan(prot_emb).any().item()}, "
                         f"comp_emb NaN: {torch.isnan(comp_emb).any().item()}")
            break

        # 正样本
        train_src = edge_index[0, train_idx]
        train_dst = edge_index[1, train_idx] - n_compounds
        pos_score = (comp_emb[train_src] * prot_emb[train_dst]).sum(dim=1)
        pos_score = torch.clamp(pos_score, -10, 10)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_score, torch.ones_like(pos_score)
        )

        # 负样本
        neg_dst = torch.randint(0, n_proteins, (train_src.shape[0],), device=DEVICE)
        neg_score = (comp_emb[train_src] * prot_emb[neg_dst]).sum(dim=1)
        neg_score = torch.clamp(neg_score, -10, 10)
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_score, torch.zeros_like(neg_score)
        )

        loss = pos_loss + neg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 验证
        if epoch % 5 == 0 and len(val_idx) > 0:
            model.eval()
            with torch.no_grad():
                prot_emb = model.encode_protein(prot_x)
                comp_emb = model.encode_compound(comp_x)

                vp_src = edge_index[0, val_idx]
                vp_dst = edge_index[1, val_idx] - n_compounds
                vp_score = (comp_emb[vp_src] * prot_emb[vp_dst]).sum(dim=1)

                vn_dst = torch.randint(0, n_proteins, (vp_src.shape[0],), device=DEVICE)
                vn_score = (comp_emb[vp_src] * prot_emb[vn_dst]).sum(dim=1)

                y_true = torch.cat([torch.ones(vp_score.shape[0]), torch.zeros(vn_score.shape[0])])
                y_score = torch.cat([vp_score, vn_score])

                y_score_np = torch.nan_to_num(
                    torch.sigmoid(y_score), nan=0.5, posinf=1.0, neginf=0.0
                ).cpu().numpy()
                if np.all(y_score_np == y_score_np[0]):
                    val_auc = 0.5
                else:
                    val_auc = roc_auc_score(y_true.cpu().numpy(), y_score_np)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            history.append({"epoch": epoch, "loss": loss.item(), "val_auc": val_auc})
            if epoch % 20 == 0:
                logger.info(f"  GAT epoch {epoch:3d} | loss={loss.item():.4f} | val_auc={val_auc:.4f}")

            if patience_counter >= patience:
                logger.info(f"  GAT early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info(f"  GAT best val_auc={best_val_auc:.4f}")
    return model, history


def train_hgt(
    model: HGTLinkPredictor,
    data: HeteroData,
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 20,
) -> Tuple[HGTLinkPredictor, List[float]]:
    """训练 HGT 链接预测（化合物用 MLP，蛋白用 HGT 图卷积）"""
    model = model.to(DEVICE)
    data = data.to(DEVICE)

    # Xavier 初始化
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

    perm = torch.randperm(num_edges, device=DEVICE)
    n_train = int(num_edges * 0.85)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    n_proteins = data["protein"].x.shape[0]
    compound_x = data["compound"].x.clone()

    # 完整异质图（HGT 需要所有节点类型来做消息传递）
    x_dict_full = {k: v.clone() for k, v in data.x_dict.items()}
    logger.info(f"  HGT x_dict keys: {list(x_dict_full.keys())}, "
                f"shapes: {[(k, v.shape) for k, v in x_dict_full.items()]}")
    logger.info(f"  HGT metadata node_types: {data.metadata()[0]}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val_auc = 0.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        hgt_out = model(x_dict_full, data.edge_index_dict)
        prot_emb = hgt_out["protein"]
        comp_emb = model.encode_compound(compound_x)

        if torch.isnan(prot_emb).any() or torch.isnan(comp_emb).any():
            logger.error(f"HGT epoch {epoch}: NaN in embeddings! "
                         f"prot_emb NaN: {torch.isnan(prot_emb).any().item()}, "
                         f"comp_emb NaN: {torch.isnan(comp_emb).any().item()}")
            break

        train_src = cpi_edges[0, train_idx]
        train_dst = cpi_edges[1, train_idx]
        pos_score = (comp_emb[train_src] * prot_emb[train_dst]).sum(dim=1)
        pos_score = torch.clamp(pos_score, -10, 10)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_score, torch.ones_like(pos_score)
        )

        neg_dst = torch.randint(0, n_proteins, (train_src.shape[0],), device=DEVICE)
        neg_score = (comp_emb[train_src] * prot_emb[neg_dst]).sum(dim=1)
        neg_score = torch.clamp(neg_score, -10, 10)
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_score, torch.zeros_like(neg_score)
        )

        loss = pos_loss + neg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 5 == 0 and len(val_idx) > 0:
            model.eval()
            with torch.no_grad():
                hgt_out = model(x_dict_full, data.edge_index_dict)
                prot_emb = hgt_out["protein"]
                comp_emb = model.encode_compound(compound_x)

                vp_src = cpi_edges[0, val_idx]
                vp_dst = cpi_edges[1, val_idx]
                vp_score = (comp_emb[vp_src] * prot_emb[vp_dst]).sum(dim=1)

                vn_dst = torch.randint(0, n_proteins, (vp_src.shape[0],), device=DEVICE)
                vn_score = (comp_emb[vp_src] * prot_emb[vn_dst]).sum(dim=1)

                y_true = torch.cat([torch.ones(vp_score.shape[0]), torch.zeros(vn_score.shape[0])])
                y_score = torch.cat([vp_score, vn_score])

                y_score_np = torch.nan_to_num(
                    torch.sigmoid(y_score), nan=0.5, posinf=1.0, neginf=0.0
                ).cpu().numpy()
                if np.all(y_score_np == y_score_np[0]):
                    val_auc = 0.5
                else:
                    val_auc = roc_auc_score(y_true.cpu().numpy(), y_score_np)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            history.append({"epoch": epoch, "loss": loss.item(), "val_auc": val_auc})
            if epoch % 20 == 0:
                logger.info(f"  HGT epoch {epoch:3d} | loss={loss.item():.4f} | val_auc={val_auc:.4f}")

            if patience_counter >= patience:
                logger.info(f"  HGT early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info(f"  HGT best val_auc={best_val_auc:.4f}")
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
    tcm_ecfp4: Optional[np.ndarray],
    target_genes: List[str],
    compound_stats: Tuple,
    smi_to_idx: Dict[str, int],
    gene_to_idx: Dict[str, int],
    n_compounds: int,
    gat_weight: float = 0.5,
) -> pd.DataFrame:
    """
    预测 TCM 化合物对铁衰老靶标的活性概率。
    GAT: MLP(compound) · GAT(protein)
    HGT: MLP(compound) · HGT(protein+pathway)
    集成: p = gat_weight * p_gat + (1-gat_weight) * p_hgt
    """
    gat_model.eval()
    if hgt_model is not None:
        hgt_model.eval()

    # 计算 TCM 化合物特征（与训练时一致：MACCS + desc，不使用 ECFP4）
    tcm_feat_raw, _, _, _ = build_compound_features(tcm_smiles, ecfp4=None, stats=compound_stats)
    feat_dim = x.shape[1]
    if tcm_feat_raw.shape[1] < feat_dim:
        tcm_feat_raw = np.pad(tcm_feat_raw, ((0, 0), (0, feat_dim - tcm_feat_raw.shape[1])), mode="constant")
    tcm_feat = torch.from_numpy(tcm_feat_raw).to(DEVICE)

    # GAT 编码
    with torch.no_grad():
        prot_x = x[n_compounds:].to(DEVICE)
        gat_prot_emb = gat_model.encode_protein(prot_x)
        gat_tcm_emb = gat_model.encode_compound(tcm_feat)

    # HGT 编码
    hgt_prot_emb = None
    if hgt_model is not None and hetero_data is not None:
        with torch.no_grad():
            x_dict_full = {k: v.clone() for k, v in hetero_data.x_dict.items()}
            hgt_out = hgt_model(x_dict_full, hetero_data.edge_index_dict)
            hgt_prot_emb = hgt_out["protein"]
            hgt_tcm_emb = hgt_model.encode_compound(tcm_feat)
    else:
        hgt_tcm_emb = torch.zeros((len(tcm_smiles), 1), device=DEVICE)
        hgt_prot_emb = torch.zeros((1, 1), device=DEVICE)

    # 构建预测矩阵
    results = []
    for i, smi in enumerate(tcm_smiles):
        row = {"MOL_ID": f"TCM_{i}", "molecule_name": "", "SMILES": smi}
        tcm_gat = gat_tcm_emb[i:i + 1]
        tcm_hgt = hgt_tcm_emb[i:i + 1] if hgt_tcm_emb.shape[0] > i else torch.zeros((1, 1), device=DEVICE)

        for gene in target_genes:
            if gene not in gene_to_idx:
                row[gene] = 0.5
                continue
            p_idx = gene_to_idx[gene]
            local_p_idx = p_idx - n_compounds

            # GAT 分数
            if 0 <= local_p_idx < gat_prot_emb.shape[0]:
                gat_score = torch.sigmoid((tcm_gat * gat_prot_emb[local_p_idx:local_p_idx + 1]).sum(dim=1)).item()
            else:
                gat_score = 0.5

            # HGT 分数
            if hgt_prot_emb is not None and 0 <= local_p_idx < hgt_prot_emb.shape[0]:
                hgt_vec = hgt_prot_emb[local_p_idx:local_p_idx + 1]
                # 对齐维度
                min_dim = min(tcm_hgt.shape[1], hgt_vec.shape[1])
                hgt_score = torch.sigmoid((tcm_hgt[:, :min_dim] * hgt_vec[:, :min_dim]).sum(dim=1)).item()
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
    """综合评分排序，输出 Top N"""
    # 计算综合得分
    gene_cols = [g for g in target_genes if g in pred_df.columns]
    scores = pred_df[gene_cols].values

    avg_score = np.nanmean(scores, axis=1)
    max_score = np.nanmax(scores, axis=1)
    n_hits = np.nansum(scores > 0.5, axis=1)
    n_high = np.nansum(scores > 0.7, axis=1)
    consistency = 1.0 - np.nanstd(scores, axis=1)

    # 归一化
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

    # 添加 top_targets
    top_targets_list = []
    for i in range(len(pred_df)):
        gene_scores = [(g, scores[i][j]) for j, g in enumerate(gene_cols)]
        gene_scores.sort(key=lambda x: x[1], reverse=True)
        top5 = gene_scores[:5]
        top_targets_list.append(
            ", ".join([f"{g}({s:.3f})" for g, s in top5])
        )
    pred_df["top_targets"] = top_targets_list

    # 排序
    pred_df = pred_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    pred_df["rank"] = range(1, len(pred_df) + 1)

    top_df = pred_df.head(top_n).copy()

    return pred_df, top_df


# ============================================================
# 10. 报告生成
# ============================================================
def _df_to_markdown(df: pd.DataFrame, max_rows: int = 30) -> str:
    """生成 Markdown 表格，不依赖 tabulate"""
    if df.empty:
        return "*(empty)*\n"
    cols = df.columns.tolist()
    lines = []
    # header
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    # rows
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
    """生成 Markdown 报告"""
    lines = [
        "# Phase 4 v6: GAT + HGT 双图神经网络集成 — 训练报告（毒性过滤后重训练）",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"总耗时: {total_time / 60:.1f} 分钟",
        "",
        "## 1. 架构",
        "- **GAT 分支**: 同质图（化合物 + 蛋白），化合物和蛋白均为独立 MLP 编码器，边 = 实验 CPI",
        "- **HGT 分支**: 异质图（化合物、蛋白、KEGG 通路），HGTConv 捕捉多靶标协同模式",
        "- **集成**: 加权平均 (GAT 0.5 + HGT 0.5)",
        "- **训练数据**: ChEMBL / BindingDB 实验验证 CPI",
        "- **预测范围**: 铁衰老温靶标",
        "- **TCM 候选池**: 经 L3 毒性过滤（剔除致癌物/致突变物）后的化合物池",
        "",
        "## 2. 关键参考",
        "- GAT: Velickovic et al. (2018) ICLR, https://arxiv.org/abs/1710.10903",
        "- HGT: Hu et al. (2020) WWW, https://arxiv.org/abs/2003.01332",
        "  - 官方代码: https://github.com/acbull/pyHGT",
        "  - PyG 实现: torch_geometric.nn.HGTConv",
        "- PyTorch Geometric: https://github.com/pyg-team/pytorch_geometric",
        "- ECFP4: Rogers & Hahn (2010) J. Chem. Inf. Model. 50(5):742-754",
        "- MACCS keys: MDL Information Systems (now BIOVIA)",
        "- RDKit: Landrum G., https://github.com/rdkit/rdkit",
        "- STRING PPI: Szklarczyk et al. (2023) Nucleic Acids Res. 51(D1):D638-D646",
        "- ChEMBL: Mendez et al. (2019) Nucleic Acids Res. 47(D1):D930-D940",
        "- BindingDB: Gilson et al. (2016) Nucleic Acids Res. 44(D1):D1045-D1053",
        "",
        "## 3. 数据规模",
        f"- TCM 候选池: {n_tcm} 个化合物（已剔除毒性/致癌物）",
        f"- 铁衰老靶标: {n_targets} 个基因",
        "",
    ]

    # 自检结果摘要
    if check_results:
        lines.extend([
            "## 3.1 管线自检结果",
            f"- 总体状态: **{check_results.get('overall', 'UNKNOWN')}**",
            f"- 严重性: {check_results.get('meta', {}).get('severity', 'UNKNOWN')}",
        ])
        errors = check_results.get("errors", [])
        warnings_list = check_results.get("warnings", [])
        if errors:
            lines.append(f"- ERRORS: {len(errors)} 条")
            for e in errors[:5]:
                lines.append(f"  - {e}")
        if warnings_list:
            lines.append(f"- WARNINGS: {len(warnings_list)} 条")
            for w in warnings_list[:5]:
                lines.append(f"  - {w}")
        lines.append("")

    lines.append("## 4. 模型性能")
    if gat_history:
        best_gat = max(gat_history, key=lambda x: x["val_auc"])
        lines.append(f"- GAT best val_auc: {best_gat['val_auc']:.4f}")
    if hgt_history:
        best_hgt = max(hgt_history, key=lambda x: x["val_auc"])
        lines.append(f"- HGT best val_auc: {best_hgt['val_auc']:.4f}")

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
        "- 训练数据与铁衰老靶标不完全匹配，跨靶标泛化可能有偏差",
        "- 未使用真实 inactive/decoy 负样本，负样本为随机采样",
        "- 通路特征使用 one-hot 编码，信息量有限",
        "- 集成权重 (0.5/0.5) 为简单平均，未优化",
        f"- TCM 候选池（经毒性过滤后）包含 {n_tcm} 个唯一 SMILES",
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
    """管线自检：验证输入数据完整性和一致性

    检查项：
      1. TCM 池完整性（SMILES 有效性、分子属性分布）
      2. CPI 数据质量（无效 SMILES、重复、每靶标记录数）
      3. TCM/训练集化合物重叠（防数据泄漏）
      4. PPI 网络
      5. 蛋白特征（维度、NaN）
      6. KEGG 通路注释
      7. 靶标覆盖（CPI 基因 vs 蛋白特征 vs PPI vs 通路）
      8. 特征维度一致性
      9. 输入文件来源追踪

    严重性分级：
      ERROR  — 阻断训练，必须修复
      WARNING — 信息性提示，训练可继续但需关注
    """
    logger.info("=" * 60)
    logger.info("开始管线自检...")
    results: Dict[str, Any] = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "severity": "UNKNOWN",
        },
        "errors": [],
        "warnings": [],
    }

    # ---- 0. 输入文件来源追踪 ----
    if input_files:
        logger.info("[自检] 输入文件来源:")
        for key, path in input_files.items():
            logger.info(f"  {key}: {path}")
        results["input_files"] = input_files

    # ---- 1. TCM 池完整性 ----
    logger.info("[自检 1/9] TCM 候选池完整性...")
    tcm_smiles = tcm_df["SMILES_std"].dropna().tolist()
    invalid_smiles = 0
    invalid_smiles_list = []
    for smi in tcm_smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            invalid_smiles += 1
            invalid_smiles_list.append(str(smi)[:80])

    tcm_check = {
        "total": len(tcm_df),
        "valid_smiles": len(tcm_smiles) - invalid_smiles,
        "invalid_smiles": invalid_smiles,
        "passed": invalid_smiles == 0,
    }
    if invalid_smiles > 0:
        results["errors"].append(
            f"TCM池含 {invalid_smiles} 个无效SMILES: {invalid_smiles_list[:5]}"
        )
    results["tcm_pool"] = tcm_check
    logger.info(f"  TCM池: {len(tcm_df)} 化合物, 无效SMILES: {invalid_smiles}")

    # 分子属性分布摘要
    mols = []
    for smi in tcm_smiles:
        m = Chem.MolFromSmiles(str(smi))
        if m:
            mols.append(m)
    if mols:
        mw_list = [Descriptors.MolWt(m) for m in mols]
        logp_list = [Descriptors.MolLogP(m) for m in mols]
        tpsa_list = [Descriptors.TPSA(m) for m in mols]
        results["tcm_molecular_props"] = {
            "n_mols": len(mols),
            "MW": {"min": min(mw_list), "max": max(mw_list), "mean": np.mean(mw_list)},
            "LogP": {"min": min(logp_list), "max": max(logp_list), "mean": np.mean(logp_list)},
            "TPSA": {"min": min(tpsa_list), "max": max(tpsa_list), "mean": np.mean(tpsa_list)},
        }
        logger.info(f"  TCM分子属性: MW=[{min(mw_list):.0f}, {max(mw_list):.0f}], "
                     f"LogP=[{min(logp_list):.1f}, {max(logp_list):.1f}], "
                     f"TPSA=[{min(tpsa_list):.0f}, {max(tpsa_list):.0f}]")

    # ---- 2. CPI 数据质量 ----
    logger.info("[自检 2/9] CPI 数据质量...")
    # 无效 SMILES
    cpi_smiles = cpi_df["canonical_smiles"].dropna().unique()
    cpi_invalid = 0
    for smi in cpi_smiles:
        if Chem.MolFromSmiles(str(smi)) is None:
            cpi_invalid += 1

    # 重复条目
    cpi_dupes = cpi_df.duplicated(subset=["gene", "canonical_smiles"]).sum()

    # 每靶标 CPI 记录数（稀疏度）
    per_target_counts = cpi_df["gene"].value_counts().to_dict()
    sparse_targets = {g: c for g, c in per_target_counts.items() if c < 10}
    low_count_targets = {g: c for g, c in per_target_counts.items() if 10 <= c < 50}

    cpi_check = {
        "total_records": len(cpi_df),
        "unique_compounds": len(cpi_smiles),
        "unique_genes": cpi_df["gene"].nunique(),
        "warm_targets": len(warm_targets),
        "invalid_smiles": cpi_invalid,
        "duplicate_entries": cpi_dupes,
        "sparse_targets": sparse_targets,
        "low_count_targets": low_count_targets,
        "per_target_counts": {g: c for g, c in sorted(per_target_counts.items(), key=lambda x: -x[1])[:20]},
        "passed": len(cpi_df) > 0 and len(warm_targets) > 0,
    }
    if cpi_invalid > 0:
        results["warnings"].append(f"CPI数据含 {cpi_invalid} 个无效SMILES（将被图构建跳过）")
    if cpi_dupes > 0:
        results["warnings"].append(f"CPI数据含 {cpi_dupes} 条重复(gene, SMILES)条目")
    if sparse_targets:
        results["warnings"].append(
            f"稀疏靶标 (<10条CPI): {len(sparse_targets)}个 — {list(sparse_targets.keys())[:10]}"
        )
    if low_count_targets:
        results["warnings"].append(
            f"低样本靶标 (10-49条): {len(low_count_targets)}个"
        )
    results["cpi_data"] = cpi_check
    logger.info(f"  CPI数据: {len(cpi_df)}条, {len(cpi_smiles)}化合物, {cpi_df['gene'].nunique()}基因, "
                f"无效SMILES:{cpi_invalid}, 重复:{cpi_dupes}")
    logger.info(f"  稀疏靶标(<10): {len(sparse_targets)}, 低样本(10-49): {len(low_count_targets)}")

    # ---- 3. TCM/训练集化合物重叠检查 ----
    logger.info("[自检 3/9] TCM/训练集化合物重叠...")
    tcm_smi_set = set(tcm_smiles)
    train_smi_set = set(cpi_smiles)
    overlap = tcm_smi_set & train_smi_set
    results["tcm_train_overlap"] = {
        "n_overlap": len(overlap),
        "overlap_ratio": len(overlap) / len(tcm_smi_set) if tcm_smi_set else 0,
        "passed": len(overlap) == 0,
    }
    if overlap:
        results["warnings"].append(
            f"TCM池与训练集有 {len(overlap)} 个重叠化合物（可能数据泄漏）"
        )
        logger.warning(f"  TCM/训练集重叠: {len(overlap)} 个化合物")
    else:
        logger.info("  TCM/训练集重叠: 0 (无数据泄漏)")

    # ---- 4. PPI 网络 ----
    logger.info("[自检 4/9] PPI 网络...")
    ppi_genes = set()
    for _, row in ppi_df.iterrows():
        ppi_genes.add(str(row["source"]).strip().upper())
        ppi_genes.add(str(row["target"]).strip().upper())
    results["ppi_network"] = {
        "edges": len(ppi_df),
        "unique_genes": len(ppi_genes),
        "passed": len(ppi_df) > 0,
    }
    logger.info(f"  PPI网络: {len(ppi_df)} 条边, {len(ppi_genes)} 个基因")

    # ---- 5. 蛋白特征 ----
    logger.info("[自检 5/9] 蛋白特征...")
    feat_dim = next(iter(prot_feat.values())).shape[0] if prot_feat else 0
    # NaN 检测
    nan_genes = []
    for gene, vec in prot_feat.items():
        if np.isnan(vec).any():
            nan_genes.append(gene)
    results["protein_features"] = {
        "genes": len(prot_feat),
        "feat_dim": feat_dim,
        "nan_genes": nan_genes,
        "passed": len(prot_feat) > 0 and len(nan_genes) == 0,
    }
    if nan_genes:
        results["errors"].append(f"蛋白特征含NaN: {len(nan_genes)}个基因 — {nan_genes}")
    logger.info(f"  蛋白特征: {len(prot_feat)} 基因, dim={feat_dim}, NaN基因: {len(nan_genes)}")

    # ---- 6. KEGG 通路注释 ----
    logger.info("[自检 6/9] KEGG 通路注释...")
    all_pathways = set(pid for paths in gene_to_pathways.values() for pid in paths)
    results["kegg_pathways"] = {
        "genes_with_pathways": len(gene_to_pathways),
        "total_pathways": len(all_pathways),
        "passed": len(gene_to_pathways) > 0,
    }
    logger.info(f"  KEGG通路: {len(gene_to_pathways)} 基因, {len(all_pathways)} 通路")

    # ---- 7. 靶标覆盖矩阵 ----
    logger.info("[自检 7/9] 靶标覆盖矩阵...")
    cpi_genes = set(cpi_df["gene"].unique())
    prot_genes = set(prot_feat.keys())
    ppi_genes_set = ppi_genes
    pathway_genes = set(gene_to_pathways.keys())
    all_ref_genes = set(ALL_FERRORAGING_GENES)

    # 温靶标在各数据源中的覆盖
    coverage_detail = {}
    for gene in warm_targets:
        coverage_detail[gene] = {
            "in_cpi": gene in cpi_genes,
            "in_prot_feat": gene in prot_genes,
            "in_ppi": gene in ppi_genes_set,
            "in_pathway": gene in pathway_genes,
        }

    matched = cpi_genes & prot_genes
    missing_feat = sorted(cpi_genes - prot_genes)
    missing_from_all = sorted(all_ref_genes - cpi_genes - prot_genes - ppi_genes_set - pathway_genes)

    results["target_coverage"] = {
        "warm_targets": len(warm_targets),
        "with_protein_features": len(matched),
        "missing_features": missing_feat,
        "with_ppi": len(cpi_genes & ppi_genes_set),
        "with_pathway": len(cpi_genes & pathway_genes),
        "fully_covered": len(cpi_genes & prot_genes & ppi_genes_set & pathway_genes),
        "missing_from_all_sources": missing_from_all,
        "coverage_detail": {g: coverage_detail[g] for g in sorted(warm_targets)},
        "passed": len(matched) > 0,
    }
    logger.info(f"  温靶标蛋白特征覆盖: {len(matched)}/{len(warm_targets)}")
    logger.info(f"  PPI覆盖: {len(cpi_genes & ppi_genes_set)}/{len(warm_targets)}")
    logger.info(f"  通路覆盖: {len(cpi_genes & pathway_genes)}/{len(warm_targets)}")
    logger.info(f"  全源覆盖: {len(cpi_genes & prot_genes & ppi_genes_set & pathway_genes)}/{len(warm_targets)}")
    if missing_feat:
        results["warnings"].append(
            f"缺少蛋白特征的温靶标: {len(missing_feat)}个 — {missing_feat[:10]}"
        )
    if missing_from_all:
        results["warnings"].append(
            f"铁衰老基因在所有数据源中均缺失: {len(missing_from_all)}个 — {missing_from_all}"
        )

    # ---- 8. 特征维度一致性 ----
    logger.info("[自检 8/9] 特征维度一致性...")
    # 检查 TCM 化合物特征维度是否与训练集一致
    dim_check = {"passed": True}
    if tcm_smiles:
        tcm_sample = tcm_smiles[:min(10, len(tcm_smiles))]
        tcm_feat, _, _, _ = build_compound_features(tcm_sample)
        dim_check["tcm_feat_dim"] = tcm_feat.shape[1]
        logger.info(f"  TCM特征维度: {tcm_feat.shape[1]}")
    if cpi_smiles is not None and len(cpi_smiles) > 0:
        cpi_sample = list(cpi_smiles)[:min(10, len(cpi_smiles))]
        cpi_feat, _, _, _ = build_compound_features(cpi_sample)
        dim_check["cpi_feat_dim"] = cpi_feat.shape[1]
        logger.info(f"  CPI特征维度: {cpi_feat.shape[1]}")
        if "tcm_feat_dim" in dim_check and dim_check["tcm_feat_dim"] != dim_check["cpi_feat_dim"]:
            dim_check["passed"] = False
            results["warnings"].append(
                f"特征维度不匹配: TCM={dim_check['tcm_feat_dim']}, CPI={dim_check['cpi_feat_dim']}"
            )
    results["feature_dimension"] = dim_check

    # ---- 9. 训练图就绪检查 ----
    logger.info("[自检 9/9] 训练图就绪检查...")
    # 过滤 CPI 到温靶标后检查
    warm_cpi = cpi_df[cpi_df["gene"].isin(warm_targets)]
    n_warm_edges = len(warm_cpi)

    # 检查是否有蛋白特征缺失导致无法训练的靶标
    trainable_targets = warm_cpi["gene"].unique()
    untrainable = [t for t in trainable_targets if t not in prot_genes]
    results["training_readiness"] = {
        "n_warm_cpi_edges": n_warm_edges,
        "n_trainable_targets": len(trainable_targets) - len(untrainable),
        "n_untrainable_targets": len(untrainable),
        "untrainable_targets": untrainable,
        "min_edges_per_target": warm_cpi["gene"].value_counts().min() if len(warm_cpi) > 0 else 0,
        "max_edges_per_target": warm_cpi["gene"].value_counts().max() if len(warm_cpi) > 0 else 0,
        "passed": n_warm_edges >= 10 and len(trainable_targets) - len(untrainable) > 0,
    }
    logger.info(f"  温靶标CPI边: {n_warm_edges}, 可训练靶标: {len(trainable_targets) - len(untrainable)}")
    if n_warm_edges < 10:
        results["errors"].append(f"温靶标CPI边不足 ({n_warm_edges} < 10)，无法训练")
    if untrainable:
        results["warnings"].append(
            f"缺少蛋白特征的靶标（无法参与训练）: {len(untrainable)}个 — {untrainable}"
        )

    # ---- 总体判定 ----
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

    # 汇总输出
    logger.info("=" * 60)
    logger.info(f"管线自检结果: {results['overall']} (严重性: {results['meta']['severity']})")
    if has_errors:
        logger.error(f"ERRORS ({len(results['errors'])}):")
        for e in results["errors"]:
            logger.error(f"  - {e}")
    if has_warnings:
        logger.warning(f"WARNINGS ({len(results['warnings'])}):")
        for w in results["warnings"]:
            logger.warning(f"  - {w}")
    logger.info("=" * 60)

    return results


# ============================================================
# 12. 主流程
# ============================================================
def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Phase 4 v6: GAT + HGT 双图神经网络集成（毒性过滤后重训练）")
    logger.info("=" * 60)

    # --- 加载数据 ---
    logger.info(">>> 加载数据")
    cpi_df = load_cpi_data()
    ppi_df = load_ppi_network()
    gene_to_pathways = load_kegg_pathways()
    prot_feat, gene_to_seq = load_protein_features()
    tcm_df = load_tcm_pool()

    # 记录输入文件来源
    input_files = {
        "cpi": str(L4_ROOT / "results" / "experimental_actives_detail.csv"),
        "ppi": str(L1_RESULTS / "ppi_network_edges.csv"),
        "kegg_pathways": str(L1_RESULTS / "string_enrichment.csv"),
        "protein_features": str(L2_RESULTS / "target_protein_features.csv"),
        "protein_pseaac": str(L2_RESULTS / "protein_pseaac.csv"),
        "tcm_pool": str(L3_RESULTS / "tcm_compound_pool_tox_filtered.csv"),
        "ferroaging_genes": str(FERRORAGING_GENES_CSV),
    }

    # --- 确定温靶标 ---
    cpi_genes = set(cpi_df["gene"].unique())
    warm_targets = sorted(cpi_genes & set(ALL_FERRORAGING_GENES))
    logger.info(f"温靶标（有 CPI 数据 + 铁衰老交集）: {len(warm_targets)} 个")

    # --- 管线自检 ---
    check_results = pipeline_self_check(
        tcm_df, cpi_df, ppi_df, prot_feat, gene_to_pathways, warm_targets,
        input_files=input_files,
    )
    # 保存自检报告
    with open(L4_RESULTS / "self_check_report_v6_tox.json", "w", encoding="utf-8") as f:
        json.dump(check_results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"自检报告: {L4_RESULTS / 'self_check_report_v6_tox.json'}")

    if check_results["overall"] == "FAILED":
        logger.error("管线自检未通过，终止训练。请检查上述错误。")
        sys.exit(1)

    if check_results["overall"] == "PASSED_WITH_WARNINGS":
        logger.warning("管线自检通过但有警告，训练将继续。请关注上述警告信息。")

    # 过滤 CPI 数据到温靶标
    cpi_df = cpi_df[cpi_df["gene"].isin(warm_targets)].copy()

    # --- 构建图 ---
    logger.info(">>> 构建同质图 (GAT)")
    # 先用少量数据估算特征维度
    sample_smiles = cpi_df["canonical_smiles"].unique()[:10].tolist()
    sample_feat, _, _, _ = build_compound_features(sample_smiles)
    compound_feat_dim = sample_feat.shape[1]

    x, edge_index, _, smi_to_idx, gene_to_idx = build_cpi_homogeneous_graph(
        cpi_df, ppi_df, prot_feat, compound_feat_dim
    )
    n_compounds = len(smi_to_idx)

    logger.info(">>> 构建异质图 (HGT)")
    hetero_data = build_heterogeneous_graph(
        cpi_df, ppi_df, gene_to_pathways, prot_feat,
        smi_to_idx, gene_to_idx, n_compounds, compound_feat_dim,
    )

    # --- 训练 GAT ---
    logger.info(">>> 训练 GAT")
    gat_model = GATLinkPredictor(
        in_dim=x.shape[1],
        hidden_dim=256,
        out_dim=128,
        num_layers=2,
        dropout=0.3,
    )
    gat_model, gat_history = train_gat(
        gat_model, x, edge_index, n_compounds,
        epochs=200, lr=1e-3, patience=20,
    )

    # --- 训练 HGT ---
    logger.info(">>> 训练 HGT")
    # 获取各节点类型的实际特征维度
    hgt_node_feat_dims = {
        node_type: hetero_data[node_type].x.shape[1]
        for node_type in hetero_data.node_types
    }
    hgt_model = HGTLinkPredictor(
        hidden_dim=256,
        out_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.3,
        metadata=hetero_data.metadata(),
        compound_feat_dim=x.shape[1],
        node_feat_dims=hgt_node_feat_dims,
    )
    hgt_model, hgt_history = train_hgt(
        hgt_model, hetero_data,
        epochs=200, lr=1e-3, patience=20,
    )

    # --- 预测 TCM ---
    logger.info(">>> 预测 TCM 化合物对铁衰老靶标")
    tcm_smiles = tcm_df["SMILES_std"].dropna().tolist()
    # 加载 TCM ECFP4
    ecfp4_path = L3_RESULTS / "ecfp4_fingerprints.npy"
    tcm_ecfp4 = None
    if ecfp4_path.exists():
        tcm_ecfp4 = np.load(ecfp4_path).astype(np.float32)
        logger.info(f"  TCM ECFP4: {tcm_ecfp4.shape}")

    # 预计算训练池化合物特征统计量
    all_train_smiles = sorted(smi_to_idx.keys())
    _, cp_mean, cp_std, cp_col_mean = build_compound_features(all_train_smiles)
    compound_stats = (cp_mean, cp_std, cp_col_mean)

    pred_df = predict_tcm(
        gat_model, hgt_model,
        x, edge_index,
        hetero_data,
        tcm_smiles, tcm_ecfp4,
        warm_targets,
        compound_stats,
        smi_to_idx, gene_to_idx, n_compounds,
        gat_weight=0.5,
    )

    # 添加 TCM 名称
    if "MOL_ID" in tcm_df.columns and "molecule_name" in tcm_df.columns:
        name_map = dict(zip(tcm_df["SMILES_std"], tcm_df["molecule_name"]))
        mol_id_map = dict(zip(tcm_df["SMILES_std"], tcm_df["MOL_ID"]))
        pred_df["molecule_name"] = pred_df["SMILES"].map(name_map).fillna("")
        pred_df["MOL_ID"] = pred_df["SMILES"].map(mol_id_map).fillna("")

    # --- 排序与输出 ---
    logger.info(">>> 综合排序")
    full_df, top_df = rank_and_export(pred_df, warm_targets, top_n=500)

    # 保存
    full_df.to_csv(L4_RESULTS / "tcm_predictions_full_v6_tox.csv", index=False)
    top_df.to_csv(L4_RESULTS / "tcm_top_candidates_v6_tox.csv", index=False)
    logger.info(f"  Top 500 候选: {L4_RESULTS / 'tcm_top_candidates_v6_tox.csv'}")

    # 保存模型性能
    perf_rows = []
    if gat_history:
        best_gat = max(gat_history, key=lambda x: x["val_auc"])
        perf_rows.append({"model": "GAT", "best_val_auc": best_gat["val_auc"]})
    if hgt_history:
        best_hgt = max(hgt_history, key=lambda x: x["val_auc"])
        perf_rows.append({"model": "HGT", "best_val_auc": best_hgt["val_auc"]})
    if perf_rows:
        pd.DataFrame(perf_rows).to_csv(L4_RESULTS / "model_performance_v6_tox.csv", index=False)

    # 保存训练指标
    metrics = {
        "gat_history": gat_history,
        "hgt_history": hgt_history,
        "n_tcm": len(tcm_smiles),
        "n_warm_targets": len(warm_targets),
        "n_cpi_edges": int(edge_index.shape[1] // 2),
        "n_compounds": len(smi_to_idx),
        "n_proteins": len(gene_to_idx),
    }
    with open(L4_RESULTS / "training_metrics_v6_tox.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    # 生成报告
    total_time = time.time() - start_time
    generate_report(
        gat_history, hgt_history, top_df, total_time,
        len(tcm_smiles), len(warm_targets),
        L4_RESULTS / "phase4_report_v6_tox.md",
        check_results=check_results,
    )

    logger.info("=" * 60)
    logger.info(f"Phase 4 v6 完成！总耗时 {total_time / 60:.1f} 分钟")
    logger.info(f"输出目录: {L4_RESULTS}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()