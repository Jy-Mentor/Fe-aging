#!/usr/bin/env python3
"""
验证 Phase 4 模型输入数据加载
==============================
在不启动模型训练的前提下，调用 phase4_gat_hgt_pipeline.py 中的数据加载函数，
验证修补后的 CPI/PPI/KEGG/蛋白特征/TCM 池路径是否正确、数据是否可正常读取。

运行：
    python L4/scripts/validate_model_inputs.py
输出：
    L4/logs/validate_model_inputs.log
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
L4_LOGS = PROJECT_ROOT / "L4" / "logs"
L4_LOGS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(L4_LOGS / "validate_model_inputs.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

# 将 L4/scripts 加入路径以导入 phase4 的数据加载函数
sys.path.insert(0, str(Path(__file__).parent))

import phase4_v10_minibatch as p4


def main():
    logger.info("=" * 60)
    logger.info("验证 Phase 4 模型输入数据加载")
    logger.info("=" * 60)

    errors = []

    # 1. CPI
    try:
        cpi_df = p4.load_cpi_data()
        logger.info("[OK] CPI: %d 行, %d 基因, %d 唯一 SMILES",
                    len(cpi_df), cpi_df["gene"].nunique(), cpi_df["canonical_smiles"].nunique())
    except Exception as e:
        logger.error("[FAIL] CPI 加载失败: %s", e)
        errors.append("CPI")

    # 2. PPI
    try:
        ppi_df = p4.load_ppi_network()
        logger.info("[OK] PPI: %d 条边, 列=%s", len(ppi_df), list(ppi_df.columns))
    except Exception as e:
        logger.error("[FAIL] PPI 加载失败: %s", e)
        errors.append("PPI")

    # 3. KEGG
    try:
        kegg = p4.load_kegg_pathways()
        logger.info("[OK] KEGG: %d 个基因有通路注释", len(kegg))
    except Exception as e:
        logger.error("[FAIL] KEGG 加载失败: %s", e)
        errors.append("KEGG")

    # 4. Protein features
    try:
        prot_feat, gene_to_seq = p4.load_protein_features()
        logger.info("[OK] 蛋白特征: %d 个基因, dim=%d", len(prot_feat),
                    next(iter(prot_feat.values())).shape[0] if prot_feat else 0)
    except Exception as e:
        logger.error("[FAIL] 蛋白特征加载失败: %s", e)
        errors.append("protein_features")

    # 5. TCM pool
    try:
        tcm_df = p4.load_tcm_pool()
        logger.info("[OK] TCM 池: %d 个化合物", len(tcm_df))
    except Exception as e:
        logger.error("[FAIL] TCM 池加载失败: %s", e)
        errors.append("TCM_pool")

    logger.info("=" * 60)
    if errors:
        logger.error("验证失败模块: %s", errors)
        sys.exit(1)
    else:
        logger.info("全部输入数据加载验证通过")


if __name__ == "__main__":
    main()
