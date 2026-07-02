"""管线自检模块

在训练前验证数据完整性、去泄漏和特征质量。
"""

import logging

import numpy as np
from rdkit import Chem

logger = logging.getLogger(__name__)


def pipeline_self_check(tcm_df, cpi_df, ppi_df, prot_feat, gene_to_pathways, warm_targets):
    """训练前管线自检：验证数据完整性、去泄漏与特征质量。

    Args:
        tcm_df: TCM 化合物池 DataFrame。
        cpi_df: CPI 边 DataFrame。
        ppi_df: PPI 边 DataFrame。
        prot_feat: 蛋白特征字典。
        gene_to_pathways: 基因到通路列表的映射。
        warm_targets: 有 CPI 数据的温靶标基因集合。

    Returns:
        包含 overall、errors、warnings 的自检结果字典。
    """
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
