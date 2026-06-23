"""
模块: L1/load_ferroaging_genes.py
功能: 加载并验证96个铁衰老基因集
输入: 铁衰老基因.txt
输出: L1/results/ferroaging_genes_96.csv
依赖: Python 3.9+, pandas
运行: python L1/load_ferroaging_genes.py
"""

import logging
import os
import sys
import traceback
from datetime import datetime

import pandas as pd

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "load_ferroaging_genes.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

CORE_GENES = {
    "ACSL4", "HMOX1", "TFRC", "GPX4", "HIF1A", "KEAP1",
    "SOD1", "NLRP3", "IL6", "TLR4", "MAPK1", "PTGS2",
    "HMGB1", "IL1B", "IFNG", "LCN2", "CXCL10", "MPO",
    "NOX4", "LPCAT3", "MAPK14", "ATF3", "EGR1", "IRF1",
    "SLC1A5", "TXNIP", "SNCA", "LOX", "ERN1", "WNT5A",
    "YAP1", "ZEB1", "CD74", "DPP4", "EDN1", "FOSL1",
    "IGFBP7", "IRF7", "IRF9", "KDM6B", "KLF6", "SOCS1",
    "SOCS2", "SP1", "TNFAIP3", "ACVR1B", "ALOX15",
    "DUOX1", "EPHA2", "HERPUD1", "LGMN", "MCU",
    "PDE4B", "PRKD1", "SAT1", "S100A8", "TNFAIP1",
    "BCL6", "BRD7", "DYRK1A", "E2F1", "E2F3", "FBXO31",
    "MAP3K14", "MEN1", "NR1D1", "RUNX3", "SETD7",
    "SMARCB1", "SMURF2", "TBX2", "WWTR1",
    "CD82", "COX7A1", "CTSB", "DPEP1", "EMP1", "GMFB",
    "ICA1", "LACTB", "LIFR", "NUAK2", "PADI4",
    "PPP2R2B", "PTBP1", "RBM3", "SLAMF8", "SPATA2",
    "ABCC1", "ATG3", "BAP1", "CAVIN1", "CDO1",
    "EBF3", "EPHA4", "HBP1", "NR2F2",
}


def main():
    logger.info("=" * 60)
    logger.info("Phase 1 Step 12: Load and validate 96 ferroaging genes")
    logger.info("Start time: %s", datetime.now().isoformat())

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(project_root, "铁衰老基因.txt")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "ferroaging_genes_96.csv")

    if not os.path.exists(input_file):
        logger.error("Input file not found: %s", input_file)
        sys.exit(1)

    logger.info("Input file: %s", input_file)
    logger.info("Input file size: %d bytes", os.path.getsize(input_file))

    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    logger.info("Raw line count: %d", len(lines))

    genes = list(dict.fromkeys(lines))
    logger.info("Deduplicated gene count: %d", len(genes))

    if len(genes) != len(lines):
        dupes = {g for g in lines if lines.count(g) > 1}
        logger.warning("Duplicate genes found: %s", dupes)

    if len(genes) != 96:
        logger.warning("Expected 96 genes, actual: %d", len(genes))

    found_core = CORE_GENES & set(genes)
    missing_core = CORE_GENES - set(genes)
    logger.info("Core ferroaging genes hit: %d/%d", len(found_core), len(CORE_GENES))
    if missing_core:
        logger.warning("Core genes not in the 96-set: %s", missing_core)

    df = pd.DataFrame({"gene_symbol": genes, "index": range(1, len(genes) + 1)})
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    logger.info("Output file: %s", output_file)
    logger.info("Output file size: %d bytes", os.path.getsize(output_file))

    df_check = pd.read_csv(output_file, encoding="utf-8-sig")
    logger.info("Output verification passed, gene count: %d", len(df_check))
    logger.info("First 10 genes: %s", ", ".join(df_check["gene_symbol"].head(10).tolist()))
    logger.info("Last 10 genes: %s", ", ".join(df_check["gene_symbol"].tail(10).tolist()))

    logger.info("End time: %s", datetime.now().isoformat())
    logger.info("Step 12 completed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logger.error(traceback.format_exc())
        sys.exit(1)