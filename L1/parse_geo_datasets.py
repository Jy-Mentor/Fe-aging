"""
模块: L1/parse_geo_datasets.py
功能: 解析5个GEO数据集，提取表达矩阵和样本分组信息
输入: L1 数据集/bulk/ 下的5个数据集文件
输出: L1/results/ 下各数据集的表达矩阵和元数据CSV
依赖: pandas, numpy, gzip, tarfile, scipy
运行: python L1/parse_geo_datasets.py
"""

import gzip
import logging
import os
import sys
import tarfile
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

# ============================================================
# 日志配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "parse_geo_datasets.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(PROJECT_ROOT, "L1 数据集", "bulk")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "L1", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_series_matrix(filepath):
    """
    Parse GEO series matrix file (gzipped) to extract sample metadata.
    Returns dict with keys: sample_ids, titles, platforms, characteristics
    """
    metadata = {"sample_ids": [], "titles": [], "platforms": [], "characteristics": []}

    with gzip.open(filepath, "rt", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("!Sample_geo_accession"):
            parts = line.split("\t")
            metadata["sample_ids"] = [p.strip('"') for p in parts[1:]]
        elif line.startswith("!Sample_title"):
            parts = line.split("\t")
            metadata["titles"] = [p.strip('"') for p in parts[1:]]
        elif line.startswith("!Sample_platform_id"):
            parts = line.split("\t")
            metadata["platforms"] = [p.strip('"') for p in parts[1:]]
        elif line.startswith("!Sample_characteristics_ch1"):
            parts = line.split("\t")
            metadata["characteristics"].append([p.strip('"') for p in parts[1:]])

    return metadata


def parse_gse104036():
    """
    Parse GSE104036: Mouse TC-RNAseq counts matrix.
    Platform: GPL17021 (Illumina HiSeq 2500), Species: Mouse
    """
    logger.info("=" * 50)
    logger.info("Parsing GSE104036 (Mouse TC-RNAseq)")

    count_file = os.path.join(DATA_DIR, "GSE104036（多时序）", "GSE104036_TC-RNAseq_counts.txt.gz")
    series_file = os.path.join(DATA_DIR, "GSE104036（多时序）", "GSE104036_series_matrix.txt.gz")

    if not os.path.exists(count_file):
        logger.error("Count file not found: %s", count_file)
        return None

    # Read counts matrix
    logger.info("Reading counts matrix...")
    with gzip.open(count_file, "rt", encoding="utf-8", errors="replace") as f:
        # Read header
        header_line = f.readline().strip()
        # Split by tabs (may have multiple spaces as separators)
        header_parts = header_line.split("\t")
        # Clean sample names (header has exactly 27 sample names, no empty gene column)
        sample_names = [s.strip() for s in header_parts]
        logger.info("Header sample count: %d", len(sample_names))

        # Read data
        data = {}
        gene_list = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            gene = parts[0].strip()
            values = parts[1:]
            if len(values) != len(sample_names):
                logger.warning("Gene %s has %d values, expected %d", gene, len(values), len(sample_names))
                continue
            gene_list.append(gene)
            data[gene] = [float(v) if v.strip() else 0.0 for v in values]

    logger.info("Genes: %d, Samples: %d", len(gene_list), len(sample_names))

    # Create DataFrame
    df = pd.DataFrame(data, index=sample_names).T
    df.index.name = "Gene"
    logger.info("Expression matrix shape: %s", df.shape)

    # Parse metadata
    meta = parse_series_matrix(series_file)
    logger.info("Sample IDs from series: %d", len(meta["sample_ids"]))

    # Assign groups based on sample names
    # S1-S3 = Sham, C* = Contralateral, I* = Ipsilateral
    group_info = []
    time_info = []
    tissue_info = []
    for s in sample_names:
        if s.startswith("S"):
            group_info.append("Sham")
            time_info.append("0hr")
            tissue_info.append("Sham")
        elif s.startswith("C"):
            group_info.append("Contralateral")
            tissue_info.append("Contralateral")
            # Extract time: C1_3hr -> 3hr
            parts = s.split("_")
            time_info.append(parts[1] if len(parts) > 1 else "unknown")
        elif s.startswith("I"):
            group_info.append("Ipsilateral")
            tissue_info.append("Ipsilateral")
            parts = s.split("_")
            time_info.append(parts[1] if len(parts) > 1 else "unknown")
        else:
            group_info.append("Unknown")
            time_info.append("unknown")
            tissue_info.append("Unknown")

    sample_meta = pd.DataFrame({
        "sample": sample_names,
        "group": group_info,
        "time": time_info,
        "tissue": tissue_info,
        "species": "Mouse",
        "platform": "GPL17021",
        "data_type": "RNAseq_counts",
    })

    # For CIRI analysis, select Sham vs Ipsilateral 24hr
    ciri_samples = [s for s in sample_names if s.startswith("S") or
                    (s.startswith("I") and "24hr" in s)]
    logger.info("CIRI comparison samples: %d (Sham + Ipsilateral 24hr)", len(ciri_samples))

    # Save
    df.to_csv(os.path.join(OUTPUT_DIR, "GSE104036_expression_matrix.csv"))
    sample_meta.to_csv(os.path.join(OUTPUT_DIR, "GSE104036_sample_meta.csv"), index=False)
    logger.info("GSE104036 parsing complete")

    return {"matrix": df, "meta": sample_meta, "ciri_samples": ciri_samples}


def parse_gse16561():
    """
    Parse GSE16561: Human blood Illumina microarray.
    Platform: GPL6883 (Illumina HumanWG-6 v3.0), Species: Human
    """
    logger.info("=" * 50)
    logger.info("Parsing GSE16561 (Human Illumina microarray)")

    exp_file = os.path.join(DATA_DIR, "GSE16561", "GSE16561_RAW (1).txt.gz")
    series_file = os.path.join(DATA_DIR, "GSE16561", "GSE16561_series_matrix (1).txt.gz")

    if not os.path.exists(exp_file):
        logger.error("Expression file not found: %s", exp_file)
        return None

    # Read expression data
    logger.info("Reading expression matrix...")
    with gzip.open(exp_file, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split("\t")
        # First column is ID_REF (probe ID), rest are sample columns
        probe_col = header[0]
        sample_names = [h.strip() for h in header[1:]]

        data = {}
        probe_list = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            probe = parts[0].strip()
            values = []
            for v in parts[1:]:
                try:
                    values.append(float(v))
                except ValueError:
                    values.append(np.nan)
            if len(values) == len(sample_names):
                probe_list.append(probe)
                data[probe] = values

    logger.info("Probes: %d, Samples: %d", len(probe_list), len(sample_names))

    df = pd.DataFrame(data, index=sample_names).T
    df.index.name = "Probe"
    logger.info("Expression matrix shape: %s", df.shape)

    # Parse metadata
    meta = parse_series_matrix(series_file)

    # Assign groups based on sample names
    group_info = []
    for s in sample_names:
        if "Stroke" in s:
            group_info.append("Stroke")
        elif "Control" in s:
            group_info.append("Control")
        else:
            group_info.append("Unknown")

    logger.info("Stroke: %d, Control: %d",
                group_info.count("Stroke"), group_info.count("Control"))

    sample_meta = pd.DataFrame({
        "sample": sample_names,
        "group": group_info,
        "species": "Human",
        "platform": "GPL6883",
        "data_type": "Illumina_microarray",
        "tissue": "Peripheral_Blood",
    })

    # Save
    df.to_csv(os.path.join(OUTPUT_DIR, "GSE16561_expression_matrix.csv"))
    sample_meta.to_csv(os.path.join(OUTPUT_DIR, "GSE16561_sample_meta.csv"), index=False)
    logger.info("GSE16561 parsing complete")

    return {"matrix": df, "meta": sample_meta}


def parse_gse37587():
    """
    Parse GSE37587: Human blood Illumina microarray (non-normalized).
    Platform: GPL6883 (Illumina HumanWG-6 v3.0), Species: Human
    NOTE: This file has detection p-values interleaved with expression values.
    Format: ID_REF, SAMPLE 1, Detection Pval, SAMPLE 2, Detection Pval, ...
    """
    logger.info("=" * 50)
    logger.info("Parsing GSE37587 (Human Illumina microarray, non-normalized)")

    exp_file = os.path.join(DATA_DIR, "GSE37587", "GSE37587_non-normalized (1).txt.gz")
    series_file = os.path.join(DATA_DIR, "GSE37587", "GSE37587_series_matrix (1).txt.gz")

    if not os.path.exists(exp_file):
        logger.error("Expression file not found: %s", exp_file)
        return None

    # Read expression data
    logger.info("Reading expression matrix (with detection p-values)...")
    with gzip.open(exp_file, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split("\t")
        # Parse header: ID_REF, SAMPLE 1, Detection Pval, SAMPLE 2, Detection Pval, ...
        exp_samples = []
        for i, h in enumerate(header[1:], 1):
            if "Detection" not in h:
                exp_samples.append(h.strip())

        # Read data, extracting only expression columns (skip detection pval columns)
        data = {}
        probe_list = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            probe = parts[0].strip()
            values = []
            # Expression columns are at indices 1, 3, 5, ... (skip detection pval at 2, 4, 6, ...)
            for i in range(1, len(parts), 2):
                try:
                    values.append(float(parts[i]))
                except (ValueError, IndexError):
                    values.append(np.nan)
            if len(values) == len(exp_samples):
                probe_list.append(probe)
                data[probe] = values

    logger.info("Probes: %d, Samples: %d", len(probe_list), len(exp_samples))

    df = pd.DataFrame(data, index=exp_samples).T
    df.index.name = "Probe"
    logger.info("Expression matrix shape: %s", df.shape)

    # Parse metadata
    meta = parse_series_matrix(series_file)

    # Determine group from titles
    group_info = []
    for t in meta.get("titles", []):
        if "Stroke" in t or "stroke" in t.lower():
            group_info.append("Stroke")
        elif "Control" in t or "control" in t.lower():
            group_info.append("Control")
        else:
            # Check characteristics for disease state
            group_info.append("Stroke")  # default - most are stroke patients

    # Check characteristics for disease state
    if meta.get("characteristics"):
        for char_line in meta["characteristics"]:
            for i, c in enumerate(char_line):
                if "disease state" in c.lower() and "control" in c.lower() and i < len(group_info):
                    group_info[i] = "Control"

    logger.info("Stroke: %d, Control: %d",
                group_info.count("Stroke"), group_info.count("Control"))

    sample_meta = pd.DataFrame({
        "sample": exp_samples,
        "group": group_info,
        "species": "Human",
        "platform": "GPL6883",
        "data_type": "Illumina_microarray_non_normalized",
        "tissue": "Peripheral_Blood_PBMC",
    })

    # Save
    df.to_csv(os.path.join(OUTPUT_DIR, "GSE37587_expression_matrix.csv"))
    sample_meta.to_csv(os.path.join(OUTPUT_DIR, "GSE37587_sample_meta.csv"), index=False)
    logger.info("GSE37587 parsing complete")

    return {"matrix": df, "meta": sample_meta}


def parse_gse61616():
    """
    Parse GSE61616: Rat Affymetrix microarray (7 days).
    Platform: GPL1355 (Affymetrix Rat Genome 230 2.0), Species: Rat
    Data: CEL files in TAR archive
    NOTE: CEL files require R/bioconductor for proper processing.
    This function extracts the CEL files and creates sample metadata.
    Actual RMA normalization will be done in R script (parse_affy.R).
    """
    logger.info("=" * 50)
    logger.info("Parsing GSE61616 (Rat Affymetrix, 7d)")

    tar_file = os.path.join(DATA_DIR, "GSE61616（7d）", "GSE61616_RAW.tar")
    series_file = os.path.join(DATA_DIR, "GSE61616（7d）", "GSE61616_series_matrix.txt.gz")

    if not os.path.exists(tar_file):
        logger.error("RAW tar file not found: %s", tar_file)
        return None

    # Extract CEL files
    cel_dir = os.path.join(OUTPUT_DIR, "GSE61616_CEL")
    os.makedirs(cel_dir, exist_ok=True)

    logger.info("Extracting CEL files...")
    with tarfile.open(tar_file, "r") as tar:
        cel_files = [m for m in tar.getmembers() if m.name.endswith(".CEL.gz")]
        logger.info("Found %d CEL files", len(cel_files))
        for member in cel_files:
            tar.extract(member, cel_dir)

    # Parse metadata
    meta = parse_series_matrix(series_file)
    logger.info("Sample titles: %s", meta.get("titles", [])[:5])

    # Create sample metadata
    sample_names = []
    group_info = []
    for t in meta.get("titles", []):
        sample_names.append(t)
        if "Sham" in t:
            group_info.append("Sham")
        elif "Model" in t:
            group_info.append("MCAO")
        elif "XST" in t:
            group_info.append("XST_treatment")
        else:
            group_info.append("Unknown")

    logger.info("Sham: %d, MCAO: %d, XST: %d",
                group_info.count("Sham"), group_info.count("MCAO"),
                group_info.count("XST_treatment"))

    sample_meta = pd.DataFrame({
        "sample": sample_names,
        "group": group_info,
        "species": "Rat",
        "platform": "GPL1355",
        "data_type": "Affymetrix_microarray",
        "time": "7d",
        "tissue": "Brain_right_hemisphere",
    })

    sample_meta.to_csv(os.path.join(OUTPUT_DIR, "GSE61616_sample_meta.csv"), index=False)
    logger.info("GSE61616 parsing complete (CEL files extracted, need R for RMA normalization)")
    logger.info("CEL files directory: %s", cel_dir)

    return {"meta": sample_meta, "cel_dir": cel_dir}


def parse_gse97537():
    """
    Parse GSE97537: Rat Affymetrix microarray (24 hours).
    Platform: GPL1355 (Affymetrix Rat Genome 230 2.0), Species: Rat
    Data: CEL files in TAR archive
    NOTE: CEL files require R/bioconductor for proper processing.
    """
    logger.info("=" * 50)
    logger.info("Parsing GSE97537 (Rat Affymetrix, 24h)")

    tar_file = os.path.join(DATA_DIR, "GSE97537(24H)", "GSE97537_RAW.tar")
    series_file = os.path.join(DATA_DIR, "GSE97537(24H)", "GSE97537_series_matrix (1).txt.gz")

    if not os.path.exists(tar_file):
        logger.error("RAW tar file not found: %s", tar_file)
        return None

    # Extract CEL files
    cel_dir = os.path.join(OUTPUT_DIR, "GSE97537_CEL")
    os.makedirs(cel_dir, exist_ok=True)

    logger.info("Extracting CEL files...")
    with tarfile.open(tar_file, "r") as tar:
        tar.extractall(cel_dir)
    cel_files = [f for f in os.listdir(cel_dir) if f.endswith(".CEL.gz")]
    logger.info("Found %d CEL files", len(cel_files))

    # Parse metadata
    meta = parse_series_matrix(series_file)
    logger.info("Sample titles: %s", meta.get("titles", [])[:5])

    # Create sample metadata
    sample_names = []
    group_info = []
    for t in meta.get("titles", []):
        sample_names.append(t)
        if "MCAO" in t:
            group_info.append("MCAO")
        elif "Sham" in t:
            group_info.append("Sham")
        else:
            group_info.append("Unknown")

    logger.info("MCAO: %d, Sham: %d",
                group_info.count("MCAO"), group_info.count("Sham"))

    sample_meta = pd.DataFrame({
        "sample": sample_names,
        "group": group_info,
        "species": "Rat",
        "platform": "GPL1355",
        "data_type": "Affymetrix_microarray",
        "time": "24h",
        "tissue": "Brain",
    })

    sample_meta.to_csv(os.path.join(OUTPUT_DIR, "GSE97537_sample_meta.csv"), index=False)
    logger.info("GSE97537 parsing complete (CEL files extracted, need R for RMA normalization)")
    logger.info("CEL files directory: %s", cel_dir)

    return {"meta": sample_meta, "cel_dir": cel_dir}


def generate_dataset_summary(results):
    """Generate a summary table of all parsed datasets."""
    logger.info("=" * 50)
    logger.info("Generating dataset summary")

    summary_rows = []
    for name, result in results.items():
        if result is None:
            continue
        meta = result.get("meta")
        matrix = result.get("matrix")
        if meta is not None:
            n_samples = len(meta)
            n_genes = matrix.shape[0] if matrix is not None else "N/A"
            groups = meta["group"].value_counts().to_dict() if "group" in meta.columns else {}
            summary_rows.append({
                "Dataset": name,
                "Platform": meta["platform"].iloc[0] if "platform" in meta.columns else "N/A",
                "Species": meta["species"].iloc[0] if "species" in meta.columns else "N/A",
                "Data_Type": meta["data_type"].iloc[0] if "data_type" in meta.columns else "N/A",
                "Samples": n_samples,
                "Genes/Probes": n_genes,
                "Groups": str(groups),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "dataset_summary.csv"), index=False)
    logger.info("Dataset summary:\n%s", summary_df.to_string())
    return summary_df


def main():
    logger.info("=" * 60)
    logger.info("Phase 1 Steps 14-17: Parse 5 GEO datasets")
    logger.info("Start time: %s", datetime.now().isoformat())

    results = {}

    # Step 14: GSE104036
    results["GSE104036"] = parse_gse104036()

    # Step 15: GSE16561
    results["GSE16561"] = parse_gse16561()

    # Step 16: GSE37587
    results["GSE37587"] = parse_gse37587()

    # Step 17: GSE61616
    results["GSE61616"] = parse_gse61616()

    # Step 17: GSE97537
    results["GSE97537"] = parse_gse97537()

    # Generate summary
    generate_dataset_summary(results)

    logger.info("End time: %s", datetime.now().isoformat())
    logger.info("All datasets parsed")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logger.error(traceback.format_exc())
        sys.exit(1)