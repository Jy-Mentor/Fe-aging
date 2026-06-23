#!/usr/bin/env python3
"""
P4 输入数据修复脚本
==================
1. 从 UniProt REST API 获取缺失的铁衰老核心蛋白序列，
   计算 AAC/PseAAC 并追加到 L2 蛋白特征文件。
2. （可选）尝试通过 PubChem 把 DrugBank 的 drug_name 解析为 canonical_smiles。
   默认跳过：DrugBank 数据来自 UniProt cross-reference，仅有 drug_name 是真实属性；
   无 SMILES 时应作为 name-only 参考，不强制用于基于结构的训练。

运行：
    python L4/scripts/repair_p4_inputs.py
输出：
    L2/results/target_protein_features.csv   (追加缺失基因)
    L2/results/protein_descriptors.csv       (追加缺失基因 AAC)
    L2/results/protein_pseaac.csv            (追加缺失基因 PseAAC)
    L4/results/drugbank_active_compounds.csv (添加/更新 canonical_smiles 列)
    L4/logs/repair_p4_inputs.log
"""

import sys
import logging
import time
import urllib.request
import urllib.parse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(Path("L4/logs/repair_p4_inputs.log"), encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent.parent
L2 = BASE / "L2" / "results"
L4 = BASE / "L4" / "results"

MISSING_GENES = {
    "ACSL4": {"uniprot": "O60488", "name": "Long-chain-fatty-acid--CoA ligase 4"},
    "FTH1": {"uniprot": "P02794", "name": "Ferritin heavy chain"},
    "FTL": {"uniprot": "P02792", "name": "Ferritin light chain"},
    "GPX4": {"uniprot": "P36969", "name": "Phospholipid hydroperoxide glutathione peroxidase"},
    "HMOX1": {"uniprot": "P09601", "name": "Heme oxygenase 1"},
    "NFE2L2": {"uniprot": "Q16236", "name": "Nuclear factor erythroid 2-related factor 2"},
    "SLC7A11": {"uniprot": "Q9UPY5", "name": "Cystine/glutamate transporter"},
    "STAT3": {"uniprot": "P40763", "name": "Signal transducer and activator of transcription 3"},
    "TFRC": {"uniprot": "P02786", "name": "Transferrin receptor protein 1"},
    "TP53": {"uniprot": "P04637", "name": "Cellular tumor antigen p53"},
}

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

HYDROPHOBICITY = {
    "A": 0.62, "C": 0.29, "D": -0.90, "E": -0.74, "F": 1.19,
    "G": 0.48, "H": -0.40, "I": 1.38, "K": -1.50, "L": 1.06,
    "M": 0.64, "N": -0.78, "P": 0.12, "Q": -0.85, "R": -2.53,
    "S": -0.18, "T": -0.05, "V": 1.08, "W": 0.81, "Y": 0.26,
}
HYDROPHILICITY = {
    "A": -0.5, "C": -1.0, "D": 3.0, "E": 3.0, "F": -2.5,
    "G": 0.0, "H": -0.5, "I": -1.8, "K": 3.0, "L": -1.8,
    "M": -1.3, "N": 0.2, "P": 0.0, "Q": 0.2, "R": 3.0,
    "S": 0.3, "T": -0.4, "V": -1.5, "W": -3.4, "Y": -2.3,
}
MASS = {
    "A": 89.09, "C": 121.15, "D": 133.10, "E": 147.13, "F": 165.19,
    "G": 75.07, "H": 155.16, "I": 131.17, "K": 146.19, "L": 131.17,
    "M": 149.21, "N": 132.12, "P": 115.13, "Q": 146.15, "R": 174.20,
    "S": 105.09, "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19,
}


def compute_aac(sequence: str) -> np.ndarray:
    aac = np.zeros(20)
    if not sequence:
        return aac
    for aa in sequence:
        idx = AA_ORDER.find(aa)
        if idx >= 0:
            aac[idx] += 1
    return aac / len(sequence)


def compute_pseaac(sequence: str, lambda_val: int = 30, w: float = 0.05) -> np.ndarray:
    if not sequence:
        return np.zeros(20 + lambda_val)

    aac = np.zeros(20)
    for aa in sequence:
        idx = AA_ORDER.find(aa)
        if idx >= 0:
            aac[idx] += 1
    aac = aac / len(sequence)

    h1 = np.array([HYDROPHOBICITY.get(aa, 0.0) for aa in sequence])
    h2 = np.array([HYDROPHILICITY.get(aa, 0.0) for aa in sequence])
    h3 = np.array([MASS.get(aa, 0.0) for aa in sequence])
    h1 = (h1 - h1.mean()) / (h1.std() + 1e-8)
    h2 = (h2 - h2.mean()) / (h2.std() + 1e-8)
    h3 = (h3 - h3.mean()) / (h3.std() + 1e-8)

    theta = np.zeros(lambda_val)
    for k in range(1, lambda_val + 1):
        if len(sequence) <= k:
            break
        s = 0.0
        for i in range(len(sequence) - k):
            s += (h1[i] - h1[i + k]) ** 2 + (h2[i] - h2[i + k]) ** 2 + (h3[i] - h3[i + k]) ** 2
        theta[k - 1] = s / (3 * (len(sequence) - k))

    denom = 1 + w * theta.sum()
    if denom == 0:
        return np.concatenate([aac, theta])
    pseaac = np.concatenate([aac / denom, (w * theta) / denom])

    if len(pseaac) > 50:
        return pseaac[:50]
    if len(pseaac) < 50:
        return np.pad(pseaac, (0, 50 - len(pseaac)))
    return pseaac


def fetch_uniprot_sequence(uniprot_id: str, max_retries: int = 3) -> str:
    """从 UniProt REST API 获取蛋白序列。"""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                text = resp.read().decode("utf-8").strip()
                lines = text.split("\n")
                seq = "".join(lines[1:])
                return seq
        except Exception as e:
            logger.warning(f"  获取 {uniprot_id} 序列失败 (重试 {attempt + 1}/{max_retries}): {e}")
            time.sleep(1 + attempt)
    return ""


def find_missing_sequences() -> dict:
    """从 UniProt API 获取缺失基因的序列。"""
    sequences = {}
    for gene, info in MISSING_GENES.items():
        uid = info["uniprot"]
        logger.info(f"获取 {gene} ({uid}) 序列...")
        seq = fetch_uniprot_sequence(uid)
        if not seq:
            logger.error(f"无法获取 {gene} ({uid}) 序列")
            continue
        sequences[gene] = {
            "uniprot_id": uid,
            "protein_name": info["name"],
            "sequence": seq,
            "length": len(seq),
        }
        logger.info(f"  {gene} ({uid}) 序列长度: {len(seq)}")
        time.sleep(0.5)
    return sequences


def append_protein_features(sequences: dict):
    """追加缺失蛋白到 L2 蛋白特征文件。"""
    prot_path = L2 / "target_protein_features.csv"
    aac_path = L2 / "protein_descriptors.csv"
    pseaac_path = L2 / "protein_pseaac.csv"

    prot = pd.read_csv(prot_path)
    aac = pd.read_csv(aac_path)
    pseaac = pd.read_csv(pseaac_path)

    existing_genes = set(prot["gene_symbol"].dropna().astype(str).unique())

    new_rows_prot = []
    new_rows_aac = []
    new_rows_pseaac = []

    for gene, info in sequences.items():
        if gene in existing_genes:
            logger.info(f"  {gene} 已存在于蛋白特征表中，跳过追加")
            continue
        seq = info["sequence"]
        aac_vec = compute_aac(seq)
        pseaac_vec = compute_pseaac(seq)

        new_rows_prot.append({
            "uniprot_id": info["uniprot_id"],
            "protein_name": info["protein_name"],
            "gene_name": gene,
            "length": info["length"],
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
            "reviewed": True,
            "gene_symbol": gene,
            "sequence": seq,
            "sequence_length": info["length"],
        })

        aac_dict = {"gene_symbol": gene}
        aac_dict.update({f"AAC_{aa}": v for aa, v in zip(AA_ORDER, aac_vec)})
        new_rows_aac.append(aac_dict)

        pseaac_dict = {"gene_symbol": gene}
        pseaac_dict.update({f"PseAAC_{i}": v for i, v in enumerate(pseaac_vec)})
        new_rows_pseaac.append(pseaac_dict)

    if new_rows_prot:
        prot_new = pd.concat([prot, pd.DataFrame(new_rows_prot)], ignore_index=True)
        prot_new.to_csv(prot_path, index=False)
        logger.info(f"已追加 {len(new_rows_prot)} 个蛋白到 {prot_path}")

        aac_new = pd.concat([aac, pd.DataFrame(new_rows_aac)], ignore_index=True)
        aac_new.to_csv(aac_path, index=False)
        logger.info(f"已追加 {len(new_rows_aac)} 行 AAC 到 {aac_path}")

        pseaac_new = pd.concat([pseaac, pd.DataFrame(new_rows_pseaac)], ignore_index=True)
        pseaac_new.to_csv(pseaac_path, index=False)
        logger.info(f"已追加 {len(new_rows_pseaac)} 行 PseAAC 到 {pseaac_path}")

    return len(new_rows_prot)


def pubchem_name_to_smiles(name: str, max_retries: int = 2) -> str:
    """通过 PubChem PUG-REST 用药物名称查询 canonical SMILES。"""
    if not name or pd.isna(name):
        return ""
    name = str(name).strip()
    encoded = urllib.parse.quote(name)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/IsomericSMILES/JSON"
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                props = data.get("PropertyTable", {}).get("Properties", [])
                if props:
                    smi = props[0].get("IsomericSMILES", "")
                    if smi:
                        mol = Chem.MolFromSmiles(smi)
                        if mol is not None:
                            return Chem.MolToSmiles(mol, canonical=True)
            return ""
        except Exception as e:
            logger.warning(f"  PubChem 查询 '{name}' 失败 (重试 {attempt + 1}/{max_retries}): {e}")
            time.sleep(0.5 + attempt)
    return ""


def repair_drugbank_smiles():
    """尝试为 DrugBank 文件添加 canonical_smiles 列。"""
    db_path = L4 / "drugbank_active_compounds.csv"
    if not db_path.exists():
        logger.warning(f"DrugBank 文件不存在: {db_path}")
        return 0, 0

    df = pd.read_csv(db_path)
    if "canonical_smiles" in df.columns and df["canonical_smiles"].notna().sum() > 0:
        logger.info("DrugBank 文件已包含 canonical_smiles，跳过")
        return int(df["canonical_smiles"].notna().sum()), len(df)

    smiles_list = []
    mapped = 0
    total = len(df)
    for idx, name in enumerate(df["drug_name"]):
        smi = pubchem_name_to_smiles(name)
        smiles_list.append(smi)
        if smi:
            mapped += 1
        if (idx + 1) % 50 == 0:
            logger.info(f"  DrugBank SMILES 解析进度: {idx + 1}/{total}, 成功 {mapped}")
        time.sleep(0.12)  # 控制 PubChem 请求频率

    df["canonical_smiles"] = smiles_list
    df.to_csv(db_path, index=False)
    logger.info(f"DrugBank SMILES 映射完成: {mapped}/{total}")
    return mapped, total


def main(enrich_drugbank: bool = False):
    logger.info("=" * 60)
    logger.info("P4 输入数据修复")
    logger.info("=" * 60)

    try:
        sequences = find_missing_sequences()
        if not sequences:
            logger.error("未找到任何缺失蛋白序列，终止修复")
            return 1

        n_added = append_protein_features(sequences)

        if enrich_drugbank:
            mapped, total = repair_drugbank_smiles()
        else:
            db_path = L4 / "drugbank_active_compounds.csv"
            total = len(pd.read_csv(db_path)) if db_path.exists() else 0
            mapped = 0
            logger.info("DrugBank SMILES 富集已跳过（默认）；如需要可设置 enrich_drugbank=True")

        logger.info("=" * 60)
        logger.info("修复完成")
        logger.info(f"  追加蛋白: {n_added}")
        logger.info(f"  DrugBank SMILES 映射: {mapped}/{total}")
        logger.info("=" * 60)
        return 0

    except Exception as e:
        logger.exception(f"修复脚本异常: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
