#!/usr/bin/env python3
"""
补充缺失的温靶标蛋白特征
========================
在 L2/results/target_protein_features.csv 中，有 4 个温靶标（有 CPI 数据且为铁衰老基因）
缺少蛋白特征：ALOX15, HIF1A, KEAP1, NOX4。

本脚本通过 UniProt REST API 获取它们的序列与注释信息，并追加到蛋白特征表中。

运行：
    python L2/add_missing_protein_features.py
输出：
    L2/results/target_protein_features.csv  (追加后)
    L2/results/target_protein_features_added_report.csv
    logs/add_missing_protein_features.log
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(Path("logs") / "add_missing_protein_features.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent
L2_RESULTS = BASE / "L2" / "results"
PROT_FILE = L2_RESULTS / "target_protein_features.csv"
REPORT_FILE = L2_RESULTS / "target_protein_features_added_report.csv"

# 缺失的温靶标及其 UniProt ID（经 UniProt 校验）
MISSING_GENES = {
    "ALOX15": "P16050",
    "HIF1A": "Q16665",
    "KEAP1": "Q14145",
    "NOX4": "Q9NPH5",
}


def fetch_uniprot_annotations(uniprot_id: str, max_retries: int = 3):
    """Fetch UniProt annotations via REST API with retries."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            logger.warning("  UniProt annotations API returned %d for %s (attempt %d/%d)",
                           resp.status_code, uniprot_id, attempt, max_retries)
        except Exception as e:
            logger.warning("  Failed to fetch annotations for %s (attempt %d/%d): %s",
                           uniprot_id, attempt, max_retries, e)
        time.sleep(2 * attempt)
    return None


def fetch_uniprot_sequence(uniprot_id: str, max_retries: int = 3) -> str:
    """Fetch protein sequence from UniProt REST API."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                lines = resp.text.strip().split("\n")
                return "".join(lines[1:])
            logger.warning("  UniProt sequence API returned %d for %s (attempt %d/%d)",
                           resp.status_code, uniprot_id, attempt, max_retries)
        except Exception as e:
            logger.warning("  Failed to fetch sequence for %s (attempt %d/%d): %s",
                           uniprot_id, attempt, max_retries, e)
        time.sleep(2 * attempt)
    return ""


def parse_uniprot_features(uniprot_id: str, data: dict) -> dict:
    """Parse UniProt JSON to extract key features（与 protein_features.py 同步）。"""
    features = {
        "uniprot_id": uniprot_id,
        "protein_name": "",
        "gene_name": "",
        "length": 0,
        "mass": 0,
        "n_domains": 0,
        "n_ptms": 0,
        "n_phospho": 0,
        "n_ubiquitination": 0,
        "n_acetylation": 0,
        "subcellular_main": "",
        "has_signal_peptide": False,
        "has_transmembrane": False,
        "n_transmembrane": 0,
        "reviewed": False,
    }

    if data is None:
        return features

    try:
        features["protein_name"] = data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "")
        features["gene_name"] = data.get("genes", [{}])[0].get("geneName", {}).get("value", "")
        features["length"] = data.get("sequence", {}).get("length", 0)
        features["mass"] = data.get("sequence", {}).get("molWeight", 0)
        features["reviewed"] = data.get("entryType", "") == "UniProtKB reviewed (Swiss-Prot)"

        comments = data.get("comments", [])
        for comment in comments:
            if comment.get("commentType", "").upper() == "SUBCELLULAR LOCATION":
                locations = comment.get("subcellularLocations", [])
                if locations:
                    features["subcellular_main"] = locations[0].get("location", {}).get("value", "")

        feat_list = data.get("features", [])
        for feat in feat_list:
            ftype = feat.get("type", "").lower()
            if ftype in ("domain", "zinc finger", "repeat"):
                features["n_domains"] += 1
            elif ftype in ("mod_res", "crosslnk", "modified residue", "cross-link",
                           "glycosylation", "disulfide bond", "lipidation",
                           "propeptide", "initiator methionine"):
                features["n_ptms"] += 1
                desc = feat.get("description", "")
                if "phospho" in desc.lower():
                    features["n_phospho"] += 1
                elif "ubiquitin" in desc.lower():
                    features["n_ubiquitination"] += 1
                elif "acetyl" in desc.lower():
                    features["n_acetylation"] += 1
            elif ftype in ("signal", "signal peptide"):
                features["has_signal_peptide"] = True
            elif ftype in ("transmem", "transmembrane"):
                features["has_transmembrane"] = True
                features["n_transmembrane"] += 1
    except Exception as e:
        logger.warning("  Error parsing UniProt features for %s: %s", uniprot_id, e)

    return features


def main():
    logger.info("=" * 60)
    logger.info("补充缺失的温靶标蛋白特征")
    logger.info("=" * 60)

    if not PROT_FILE.exists():
        logger.error("蛋白特征文件不存在: %s", PROT_FILE)
        sys.exit(1)

    df = pd.read_csv(PROT_FILE, low_memory=False)
    existing_genes = set(df["gene_symbol"].dropna().astype(str).str.upper().unique())
    logger.info("现有蛋白特征表: %d 行", len(df))

    added_rows = []
    report = []

    for gene, uniprot_id in MISSING_GENES.items():
        if gene in existing_genes:
            logger.info("%s 已存在于蛋白特征表，跳过", gene)
            continue

        logger.info("获取 %s (%s)...", gene, uniprot_id)
        annotations = fetch_uniprot_annotations(uniprot_id)
        sequence = fetch_uniprot_sequence(uniprot_id)

        if annotations is None:
            logger.error("  无法获取 %s 的注释，跳过", uniprot_id)
            continue

        parsed = parse_uniprot_features(uniprot_id, annotations)
        parsed["gene_symbol"] = gene
        parsed["sequence"] = sequence
        parsed["sequence_length"] = len(sequence)

        # 如果 UniProt length 与序列长度不一致，以序列长度为准并记录
        if parsed["length"] == 0 and sequence:
            parsed["length"] = len(sequence)

        added_rows.append(parsed)
        report.append({
            "gene_symbol": gene,
            "uniprot_id": uniprot_id,
            "protein_name": parsed["protein_name"],
            "length": parsed["length"],
            "n_domains": parsed["n_domains"],
            "n_ptms": parsed["n_ptms"],
            "n_transmembrane": parsed["n_transmembrane"],
            "has_signal_peptide": parsed["has_signal_peptide"],
            "subcellular_main": parsed["subcellular_main"],
            "sequence_fetched": bool(sequence),
            "sequence_length": len(sequence),
        })
        logger.info("  成功: %s, length=%d, n_domains=%d, n_ptms=%d",
                    gene, parsed["length"], parsed["n_domains"], parsed["n_ptms"])
        time.sleep(0.5)

    if added_rows:
        added_df = pd.DataFrame(added_rows)
        # 确保列顺序与现有文件一致
        cols = list(df.columns)
        for c in cols:
            if c not in added_df.columns:
                added_df[c] = None
        added_df = added_df[cols]
        df = pd.concat([df, added_df], ignore_index=True)
        df.to_csv(PROT_FILE, index=False)
        logger.info("已追加 %d 个蛋白，更新后共 %d 个", len(added_rows), len(df))
    else:
        logger.info("无需追加")

    report_df = pd.DataFrame(report)
    report_df.to_csv(REPORT_FILE, index=False)
    logger.info("已生成追加报告: %s", REPORT_FILE)


if __name__ == "__main__":
    main()
