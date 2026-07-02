#!/usr/bin/env python3
"""补充查询缺失蛋白的UniProt序列。

策略:
1. ENSP 开头 -> UniProt ID mapping (Ensembl Protein -> UniProtKB)
2. *_HUMAN -> 当作 UniProt accession 直接查询
3. *-2 等剪接变体 -> 尝试主基因名
4. 其余保留原名再次尝试单基因查询
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("supplement_missing")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results_v10_minibatch"
SEQ_CSV = RESULTS_DIR / "protein_sequences_6864.csv"
OUT_CSV = RESULTS_DIR / "protein_sequences_6864_supplement.csv"
UNIPROT_SLEEP = 0.6


def fetch_by_id_mapping(ids: list[str], from_db: str = "Ensembl_Protein") -> dict[str, dict]:
    """使用 UniProt ID mapping API。"""
    if not ids:
        return {}
    url = "https://rest.uniprot.org/idmapping/run"
    data = {"from": from_db, "to": "UniProtKB-Swiss-Prot", "ids": ",".join(ids)}
    r = requests.post(url, data=data, timeout=60)
    r.raise_for_status()
    job_id = r.json()["jobId"]
    logger.info(f"ID mapping job {job_id} for {len(ids)} ids")

    # poll
    status_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
    for _ in range(30):
        r = requests.get(status_url, timeout=60)
        r.raise_for_status()
        j = r.json()
        if "results" in j or j.get("jobStatus") == "FINISHED":
            break
        time.sleep(2)

    result_url = f"https://rest.uniprot.org/idmapping/results/{job_id}"
    r = requests.get(result_url, timeout=120)
    r.raise_for_status()
    j = r.json()

    out: dict[str, dict] = {}
    for item in j.get("results", []):
        src = item.get("from", "")
        tgt = item.get("to", {})
        ac = tgt.get("primaryAccession", "")
        info = tgt.get("uniProtkbId", "")
        seq = tgt.get("sequence", {}).get("sequence", "")
        reviewed = tgt.get("entryType", "") == "UniProtKB reviewed (Swiss-Prot)"
        out[src] = {"uniprot_ac": ac, "reviewed": reviewed, "sequence": seq, "info": info}
    return out


def fetch_single_gene(gene: str) -> dict | None:
    """单个基因查询，优先reviewed。"""
    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "query": f"organism_id:9606 AND gene:{gene}",
        "format": "tsv",
        "fields": "accession,gene_names,sequence,reviewed",
    }
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            break
        except Exception as e:
            logger.warning(f"  {gene} attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    else:
        return None
    df = pd.read_csv(io.StringIO(r.text), sep="\t")
    best = None
    for _, row in df.iterrows():
        ac = str(row.get("Entry", "")).strip()
        seq = str(row.get("Sequence", "")).strip()
        rev = str(row.get("Reviewed", "")).strip().lower() == "reviewed"
        if not seq:
            continue
        if best is None or (rev and not best["reviewed"]):
            best = {"uniprot_ac": ac, "reviewed": rev, "sequence": seq}
    return best


def main() -> None:
    df = pd.read_csv(SEQ_CSV)
    missing = df[df["sequence"].isna() | (df["sequence"] == "")]["gene_symbol"].tolist()
    logger.info(f"缺失序列蛋白数: {len(missing)}")

    records: dict[str, dict] = {g: {"gene_symbol": g, "uniprot_ac": "", "reviewed": False, "sequence": ""} for g in missing}

    # 1. ENSP -> ID mapping
    ensp_ids = [g for g in missing if g.startswith("ENSP")]
    if ensp_ids:
        ensp_map = fetch_by_id_mapping(ensp_ids, from_db="Ensembl_Protein")
        for src, info in ensp_map.items():
            if info.get("sequence"):
                records[src].update(info)
        logger.info(f"ENSP ID mapping找回: {sum(1 for k in ensp_ids if records[k]['sequence'])}/{len(ensp_ids)}")
        time.sleep(UNIPROT_SLEEP)

    # 2. *_HUMAN -> 直接当accession查询
    human_like = [g for g in missing if g.endswith("_HUMAN") and not records[g]["sequence"]]
    for g in human_like:
        ac = g.replace("_HUMAN", "")
        url = f"https://rest.uniprot.org/uniprotkb/{ac}.tsv?fields=accession,gene_names,sequence,reviewed"
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            row_df = pd.read_csv(io.StringIO(r.text), sep="\t")
            if len(row_df) > 0:
                row = row_df.iloc[0]
                seq = str(row.get("Sequence", "")).strip()
                if seq:
                    records[g].update({
                        "uniprot_ac": str(row.get("Entry", "")).strip(),
                        "reviewed": str(row.get("Reviewed", "")).strip().lower() == "reviewed",
                        "sequence": seq,
                    })
        except Exception as e:
            logger.warning(f"{g} direct accession query failed: {e}")
        time.sleep(0.3)
    logger.info(f"_HUMAN直接查询找回: {sum(1 for k in human_like if records[k]['sequence'])}/{len(human_like)}")

    # 3. *-2 变体 -> 尝试主基因名
    splice_vars = [g for g in missing if "-" in g and not records[g]["sequence"]]
    for g in splice_vars:
        main = g.split("-")[0]
        info = fetch_single_gene(main)
        if info:
            records[g].update(info)
        time.sleep(0.3)
    logger.info(f"剪接变体查询找回: {sum(1 for k in splice_vars if records[k]['sequence'])}/{len(splice_vars)}")

    # 4. 其余再次单基因查询
    remaining = [g for g in missing if not records[g]["sequence"]]
    for g in remaining:
        info = fetch_single_gene(g)
        if info:
            records[g].update(info)
        time.sleep(0.3)
    logger.info(f"其余单基因查询找回: {sum(1 for k in remaining if records[k]['sequence'])}/{len(remaining)}")

    out_df = pd.DataFrame.from_records(list(records.values()))
    out_df.to_csv(OUT_CSV, index=False)
    found = out_df["sequence"].astype(bool).sum()
    logger.info(f"补充查询完成: {found}/{len(missing)} 条找回，结果保存至 {OUT_CSV}")


if __name__ == "__main__":
    main()
