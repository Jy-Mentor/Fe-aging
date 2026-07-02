"""数据加载模块

从各级结果目录加载 CPI、PPI、KEGG 通路和 TCM 候选池数据。
"""

import logging
import sys

import pandas as pd

from ..utils.config import Config

logger = logging.getLogger(__name__)

# 模块级路径解析（只实例化一次）
_cfg = Config()
_paths = _cfg.get_resolved_paths()


def load_cpi_data() -> pd.DataFrame:
    """加载 CPI（化合物-蛋白互作）数据"""
    cpi_path = _paths.l4_root / "results" / "experimental_actives_detail_cleaned.csv"
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
    assert len(df) > 0, "CPI 数据加载后为空，请检查数据文件内容"
    assert "gene" in df.columns, "CPI 数据缺少 gene 列"
    assert "canonical_smiles" in df.columns, "CPI 数据缺少 canonical_smiles 列"
    assert "uniprot_id" in df.columns, "CPI 数据缺少 uniprot_id 列"
    logger.info(f"CPI 数据: {len(df)} 条记录, {df['gene'].nunique()} 个基因, "
                f"{df['canonical_smiles'].nunique()} 个唯一 SMILES")
    return df


def load_ppi_network() -> pd.DataFrame:
    """加载 PPI 网络数据（优先补充版，降级为 DEG 显著子网，再降级为扩展网络）"""
    supplemented_path = _paths.l1_results / "ppi_network_supplemented.csv"
    significant_path = _paths.l1_results / "ppi_network_extended_significant_edges.csv"
    extended_path = _paths.l1_results / "ppi_network_extended_edges.csv"

    ppi_path = None
    network_type = "未知"
    if supplemented_path.exists():
        ppi_path = supplemented_path
        network_type = "补充版 (96/96基因覆盖)"
    elif significant_path.exists():
        ppi_path = significant_path
        network_type = "DEG 显著子网"
    elif extended_path.exists():
        ppi_path = extended_path
        network_type = "扩展"

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
        logger.info(f"PPI 网络（{network_type}）: {len(df)} 条边, "
                    f"{pd.concat([df['source'], df['target']]).nunique()} 个节点")
        return df

    logger.error("PPI 网络文件不存在")
    sys.exit(1)


def load_kegg_pathways() -> dict[str, list[str]]:
    """加载 KEGG 通路数据，返回 {基因符号: [通路ID列表]}"""
    kegg_path = _paths.l2_results / "kegg_pathways" / "kegg_human_pathway_genes.tsv"
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


def load_tcm_pool() -> pd.DataFrame:
    """加载 TCM 候选池（优先去泄漏版）"""
    noleak_path = _paths.l3_results / "tcm_compound_pool_tox_filtered_noleak.csv"
    original_path = _paths.l3_results / "tcm_compound_pool_tox_filtered.csv"
    tcm_path = noleak_path if noleak_path.exists() else original_path
    if not tcm_path.exists():
        logger.error(f"TCM 候选池文件不存在: {tcm_path}")
        sys.exit(1)
    try:
        df = pd.read_csv(tcm_path, low_memory=False)
    except Exception:
        logger.error(f"TCM 候选池文件读取失败: {tcm_path}", exc_info=True)
        raise
    assert len(df) > 0, "TCM 候选池加载后为空，请检查数据文件内容"
    source_tag = "去泄漏版" if tcm_path == noleak_path else "原始版"
    logger.info(f"TCM 候选池（{source_tag}）: {len(df)} 个化合物")
    return df
