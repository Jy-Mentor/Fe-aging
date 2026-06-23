#!/usr/bin/env python3
"""
P4 输入文件强化验证 (v2)
========================
对 Phase 4 所需输入文件进行真实性、一致性、质量校验。
输出结构化 JSON 报告，ERROR 会导致验证失败，WARNING 会记录原因。

运行:
    python L4/scripts/validate_p4_inputs_v2.py
输出:
    L4/logs/input_validation_report_v2.json
    L4/logs/input_checksums.json
"""

import sys
import json
import hashlib
import logging
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from rdkit import Chem

# ============================================================
# 路径配置
# ============================================================
BASE = Path(__file__).parent.parent.parent
L2 = BASE / "L2" / "results"
L3 = BASE / "L3" / "results"
L4 = BASE / "L4" / "results"
L4_LOGS = BASE / "L4" / "logs"
L4_LOGS.mkdir(parents=True, exist_ok=True)

LOG_FILE = L4_LOGS / "validate_p4_inputs_v2.log"
REPORT_FILE = L4_LOGS / "input_validation_report_v2.json"
CHECKSUM_FILE = L4_LOGS / "input_checksums.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

CORE_GENES = [
    "EMP1", "SAT1", "TLR4", "LCN2", "EPHA4", "CXCL10", "KLF6", "SP1",
    "CD74", "PTGS2", "IRF1", "FBXO31", "LGMN", "IGFBP7", "IL1B", "MAPK1",
    "KDM6B", "PDE4B", "RUNX3", "CTSB", "LACTB", "LPCAT3", "EGR1", "BCL6",
    "GMFB", "HBP1", "SOD1", "DYRK1A",
]
PRIORITY_TARGETS = [
    "ACSL4", "GPX4", "HMOX1", "FTH1", "FTL", "SLC7A11", "TFRC",
    "TLR4", "PTGS2", "IL1B", "MAPK1", "NFE2L2", "TP53", "STAT3",
]
ALL_TARGET_GENES = sorted(set(CORE_GENES + PRIORITY_TARGETS))

# 内置文献活性集（FERROPTOSIS_ACTIVES），用于校验 SMILES 有效性
FERROPTOSIS_ACTIVES = {
    "ACSL4": {"CCCCC1=CC(=O)C(=C(C1=O)O)CCCCCCCC(O)=O", "CCCCCCCCCCCC(O)=O"},
    "GPX4": {"CC1=C(C(=O)C2=C(C1=O)C(=O)C3=CC=CC=C3C2=O)N4CCN(CC4)CCO",
             "CN(C)C1=CC=C(C=C1)C=C2C(=O)C3=C(C2=O)C(=O)C4=CC=CC=C4C3=O"},
    "FTH1": {"CC1=C(C=CC(=C1)C(C)(C)C)C(C)(C)C"},
    "FTL": {"CC1=C(C=CC(=C1)C(C)(C)C)C(C)(C)C"},
    "SLC7A11": {"CC(=O)NC(CS)C(O)=O", "C(C(C(=O)O)N)C(=O)O"},
    "TFRC": {"CN(C)CC1=CC=CC=C1"},
    "HMOX1": {"COC1=C(O)C=CC(=O)C=CC2=CC=C(O)C(OC)=C2", "CC1=C(C=C(C=C1)C(C)(C)C)C(C)(C)C"},
    "NFE2L2": {"COC1=C(O)C=CC(=O)C=CC2=CC=C(O)C(OC)=C2", "C1=CC(=CC=C1C=CC(=O)CC(=O)C2=CC=CC=C2)O"},
    "KEAP1": {"COC1=C(O)C=CC(=O)C=CC2=CC=C(O)C(OC)=C2"},
    "TP53": {"CC1(C)CC(C)(C)C2=CC=CC=C2N1O"},
    "STAT3": {"CC1=CC=C(S(=O)(=O)N2CCOCC2)C=C1", "COC1=C(O)C=C(C=C1)C2=CC(=O)C3=C(C=C(C=C3O2)O)O"},
    "TLR4": {"CC1=CC(=O)C(=C(C)C1=O)C(C)(C)CCCC(C)(C)C(O)=O"},
    "PTGS2": {"CC(=O)OC1=CC=CC=C1C(O)=O", "CC1=C(C(=O)C2=CC=CC=C2)C(=O)N(C1=O)C3=CC=CC=C3"},
    "IL1B": {"CC1=C(C(O)=O)C2=CC=CC=C2N1C(=O)C3=CC=C(Cl)C=C3"},
    "MAPK1": {"CN1C=NC2=C1C(=NC=N2)NC3=CC=CC=C3"},
    "ALOX5": {"CC(C)(C)C1=CC=C(C=C1)C2=CC(=O)C3=C(C=C(C=C3O2)O)O"},
    "NOX4": {"CC1=C(C=C(C=C1)C(C)(C)C)C(C)(C)C"},
    "NFKB1": {"COC1=C(O)C=CC(=O)C=CC2=CC=C(O)C(OC)=C2"},
    "RELA": {"COC1=C(O)C=CC(=O)C=CC2=CC=C(O)C(OC)=C2"},
    "HIF1A": {"CC1=C(C=C(C=C1)C(C)(C)C)C(C)(C)C"},
}


# ============================================================
# 辅助函数
# ============================================================
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_info(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": _sha256(path) if path.exists() else None,
        "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else None,
    }


def _check_required_files():
    required = {
        L3 / "tcm_compound_pool_filtered.csv": "TCM化合物池",
        L3 / "rdkit_descriptors.csv": "RDKit描述符",
        L3 / "ecfp4_fingerprints.npy": "ECFP4指纹",
        L3 / "maccs_fingerprints.npy": "MACCS指纹",
        L2 / "target_protein_features.csv": "蛋白特征",
        L2 / "protein_descriptors.csv": "AAC描述符",
        L2 / "protein_pseaac.csv": "PseAAC特征",
    }
    errors = []
    warnings = []
    checksums = {}
    for fpath, desc in required.items():
        info = _file_info(fpath)
        checksums[desc] = info
        if not info["exists"]:
            errors.append(f"[MISSING] {desc}: {fpath}")
        elif info["size_bytes"] == 0:
            errors.append(f"[EMPTY] {desc}: {fpath}")
        else:
            logger.info(f"[OK] {desc}: {fpath} ({info['size_bytes']} bytes)")
    return errors, warnings, checksums


def _check_compound_data():
    errors = []
    warnings = []

    cp = pd.read_csv(L3 / "tcm_compound_pool_filtered.csv")
    desc = pd.read_csv(L3 / "rdkit_descriptors.csv")
    ecfp4 = np.load(L3 / "ecfp4_fingerprints.npy")
    maccs = np.load(L3 / "maccs_fingerprints.npy")

    n_cp = len(cp)
    logger.info(f"  TCM化合物池: {n_cp} 行")

    # 列检查
    for col in ["MOL_ID", "molecule_name", "SMILES_std"]:
        if col not in cp.columns:
            errors.append(f"TCM化合物池缺失列: {col}")

    # 行数一致性
    if len(desc) != n_cp:
        errors.append(f"描述符行数({len(desc)}) != 化合物数({n_cp})")
    if len(ecfp4) != n_cp:
        errors.append(f"ECFP4行数({len(ecfp4)}) != 化合物数({n_cp})")
    if len(maccs) != n_cp:
        errors.append(f"MACCS行数({len(maccs)}) != 化合物数({n_cp})")

    # SMILES 检查
    if "SMILES_std" in cp.columns:
        null_smiles = cp["SMILES_std"].isna().sum()
        if null_smiles > 0:
            errors.append(f"TCM库空SMILES: {null_smiles}")
        dup_smiles = cp["SMILES_std"].duplicated().sum()
        if dup_smiles > 0:
            warnings.append(f"TCM库重复SMILES: {dup_smiles}")

        # 100% SMILES 解析检查
        invalid = 0
        invalid_indices = []
        for i, smi in enumerate(cp["SMILES_std"]):
            if pd.isna(smi) or Chem.MolFromSmiles(str(smi)) is None:
                invalid += 1
                invalid_indices.append(i)
                if len(invalid_indices) <= 5:
                    warnings.append(f"TCM库无效SMILES索引 {i}: {smi}")
        if invalid > 0:
            errors.append(f"TCM库无效SMILES: {invalid}/{n_cp} (>{5}%阈值)")
        else:
            logger.info(f"  TCM库SMILES解析: {n_cp}/{n_cp} 通过")

    # 指纹质量
    if np.isnan(ecfp4).any() or np.isinf(ecfp4).any():
        errors.append("ECFP4指纹含NaN或Inf")
    if np.isnan(maccs).any() or np.isinf(maccs).any():
        errors.append("MACCS指纹含NaN或Inf")
    zero_rows = (ecfp4.sum(axis=1) == 0).sum()
    if zero_rows > 0:
        warnings.append(f"ECFP4全零行: {zero_rows}")

    return errors, warnings, {
        "n_compounds": n_cp,
        "n_descriptors": len(desc.columns) - 1 if "MOL_ID" in desc.columns else len(desc.columns),
        "ecfp4_shape": list(ecfp4.shape),
        "maccs_shape": list(maccs.shape),
        "invalid_smiles": invalid if "invalid" in dir() else None,
    }


def _check_protein_data():
    errors = []
    warnings = []

    prot = pd.read_csv(L2 / "target_protein_features.csv")
    aac = pd.read_csv(L2 / "protein_descriptors.csv")
    pseaac = pd.read_csv(L2 / "protein_pseaac.csv")

    n_prot = len(prot)
    logger.info(f"  蛋白特征表: {n_prot} 行")

    if "gene_symbol" not in prot.columns:
        errors.append("蛋白特征表缺失 gene_symbol 列")
    else:
        null_genes = prot["gene_symbol"].isna().sum()
        if null_genes > 0:
            errors.append(f"蛋白表空gene_symbol: {null_genes}")
        dup_genes = prot["gene_symbol"].duplicated().sum()
        if dup_genes > 0:
            errors.append(f"蛋白表重复gene_symbol: {dup_genes}")

    if len(aac) != n_prot:
        errors.append(f"AAC行数({len(aac)}) != 蛋白数({n_prot})")
    if len(pseaac) != n_prot:
        errors.append(f"PseAAC行数({len(pseaac)}) != 蛋白数({n_prot})")

    aac_cols = [c for c in aac.columns if c.startswith("AAC_")]
    if len(aac_cols) != 20:
        warnings.append(f"AAC列数={len(aac_cols)}, 预期20")

    pseaac_cols = [c for c in pseaac.columns if c.startswith("PseAAC_")]
    if len(pseaac_cols) != 50:
        warnings.append(f"PseAAC列数={len(pseaac_cols)}, 预期50")

    # 核心基因匹配
    if "gene_symbol" in prot.columns:
        prot_genes = set(prot["gene_symbol"].dropna().unique())
        missing = [g for g in ALL_TARGET_GENES if g not in prot_genes]
        if missing:
            errors.append(f"蛋白数据缺失核心基因: {missing}")
        else:
            logger.info(f"  所有 {len(ALL_TARGET_GENES)} 个核心基因均在蛋白数据中")

    return errors, warnings, {
        "n_proteins": n_prot,
        "aac_dim": len(aac_cols),
        "pseaac_dim": len(pseaac_cols),
        "missing_core_genes": missing if "missing" in dir() else [],
    }


def _check_active_data():
    errors = []
    warnings = []
    stats = {"files": {}}

    active_files = {
        "chembl": L4 / "chembl_active_compounds.csv",
        "bindingdb": L4 / "bindingdb_active_compounds.csv",
        "drugbank": L4 / "drugbank_active_compounds.csv",
    }

    any_exists = False
    for source, path in active_files.items():
        if not path.exists():
            warnings.append(f"活性数据文件缺失: {path.name}")
            stats["files"][source] = {"exists": False}
            continue
        any_exists = True
        df = pd.read_csv(path)
        n_total = len(df)

        # 列名标准化
        gene_col = next((c for c in df.columns if "gene" in c.lower()), None)
        smiles_col = next((c for c in df.columns if "smiles" in c.lower()), None)

        if gene_col is None:
            errors.append(f"{path.name}: 未找到基因列")
            continue
        if smiles_col is None:
            errors.append(f"{path.name}: 未找到SMILES列")
            continue

        # 基因符号映射检查
        valid_genes = set()
        invalid_genes = []
        for g in df[gene_col].dropna().unique():
            if g in ALL_TARGET_GENES:
                valid_genes.add(g)
            else:
                invalid_genes.append(g)
        if invalid_genes:
            warnings.append(f"{path.name}: {len(invalid_genes)} 个非目标基因, 示例: {invalid_genes[:5]}")

        # SMILES 解析
        invalid_smiles = 0
        valid_smiles = 0
        for smi in df[smiles_col].dropna():
            if Chem.MolFromSmiles(str(smi)) is not None:
                valid_smiles += 1
            else:
                invalid_smiles += 1
        parse_rate = valid_smiles / (valid_smiles + invalid_smiles) if (valid_smiles + invalid_smiles) > 0 else 0
        if parse_rate < 0.95:
            errors.append(f"{path.name}: SMILES解析率 {parse_rate:.2%} < 95%")
        else:
            logger.info(f"  {path.name}: {n_total} 行, SMILES解析率 {parse_rate:.2%}")

        # 重复检查
        dup_pairs = df.duplicated(subset=[gene_col, smiles_col]).sum()
        if dup_pairs > 0:
            warnings.append(f"{path.name}: 重复(gene,smiles)对={dup_pairs}")

        # standard_relation 检查
        relation_col = next((c for c in df.columns if "relation" in c.lower()), None)
        if relation_col is not None:
            bad_relation = (df[relation_col].fillna("=") != "=").sum()
            if bad_relation > 0:
                warnings.append(f"{path.name}: {bad_relation} 条记录 standard_relation != '='")

        stats["files"][source] = {
            "exists": True,
            "n_rows": n_total,
            "gene_col": gene_col,
            "smiles_col": smiles_col,
            "valid_genes": len(valid_genes),
            "invalid_smiles": invalid_smiles,
            "parse_rate": parse_rate,
            "duplicate_pairs": int(dup_pairs),
        }

    if not any_exists:
        errors.append("没有任何活性数据文件存在")

    # 内置文献活性集校验
    invalid_lit = 0
    for gene, smiles_set in FERROPTOSIS_ACTIVES.items():
        for smi in smiles_set:
            if Chem.MolFromSmiles(str(smi)) is None:
                invalid_lit += 1
                warnings.append(f"内置活性集 {gene} 无效SMILES: {smi}")
    stats["literature_actives_invalid"] = invalid_lit
    if invalid_lit > 0:
        warnings.append(f"内置FERROPTOSIS_ACTIVES中无效SMILES: {invalid_lit}")

    return errors, warnings, stats


def _check_cross_consistency():
    errors = []
    warnings = []

    # 读取蛋白基因
    prot = pd.read_csv(L2 / "target_protein_features.csv")
    prot_genes = set(prot["gene_symbol"].dropna().unique())

    # 读取活性数据基因
    active_genes = set()
    for fname in ["chembl_active_compounds.csv", "bindingdb_active_compounds.csv", "drugbank_active_compounds.csv"]:
        path = L4 / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        gene_col = next((c for c in df.columns if "gene" in c.lower()), None)
        if gene_col:
            active_genes.update(df[gene_col].dropna().unique())

    # 活性数据基因与蛋白基因交集
    matched = active_genes & prot_genes
    unmatched = active_genes - prot_genes
    logger.info(f"  活性数据基因: {len(active_genes)}, 与蛋白表匹配: {len(matched)}, 未匹配: {len(unmatched)}")
    if unmatched:
        warnings.append(f"活性数据中存在蛋白表未覆盖的基因: {sorted(unmatched)[:10]}")

    # 活性数据 SMILES 与 TCM 库重叠
    cp = pd.read_csv(L3 / "tcm_compound_pool_filtered.csv")
    tcm_smiles = set(cp["SMILES_std"].dropna().astype(str))

    active_smiles = set()
    for fname in ["chembl_active_compounds.csv", "bindingdb_active_compounds.csv", "drugbank_active_compounds.csv"]:
        path = L4 / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        smiles_col = next((c for c in df.columns if "smiles" in c.lower()), None)
        if smiles_col:
            active_smiles.update(df[smiles_col].dropna().astype(str))

    overlap = len(active_smiles & tcm_smiles)
    overlap_rate = overlap / len(active_smiles) if active_smiles else 0
    logger.info(f"  活性SMILES与TCM库重叠: {overlap}/{len(active_smiles)} ({overlap_rate:.2%})")
    if overlap_rate < 0.01 and active_smiles:
        warnings.append(f"活性SMILES与TCM库重叠率仅 {overlap_rate:.2%}, 可能导致训练正样本不足")

    return errors, warnings, {
        "active_genes": len(active_genes),
        "matched_protein_genes": len(matched),
        "unmatched_genes": sorted(unmatched)[:20],
        "active_smiles": len(active_smiles),
        "tcm_smiles": len(tcm_smiles),
        "overlap_smiles": overlap,
        "overlap_rate": overlap_rate,
    }


# ============================================================
# 主流程
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("P4 输入文件强化验证 (v2)")
    logger.info("=" * 60)

    report = {
        "timestamp": datetime.now().isoformat(),
        "status": "PENDING",
        "errors": [],
        "warnings": [],
        "details": {},
    }

    try:
        logger.info("[1/4] 文件级校验")
        e, w, checksums = _check_required_files()
        report["errors"].extend(e)
        report["warnings"].extend(w)
        report["details"]["files"] = checksums

        # 如果必需文件缺失，直接失败
        if e:
            report["status"] = "FAILED"
            _write_report(report)
            return 1

        logger.info("[2/4] 化合物数据校验")
        e, w, comp_stats = _check_compound_data()
        report["errors"].extend(e)
        report["warnings"].extend(w)
        report["details"]["compound_data"] = comp_stats

        logger.info("[3/4] 蛋白质数据校验")
        e, w, prot_stats = _check_protein_data()
        report["errors"].extend(e)
        report["warnings"].extend(w)
        report["details"]["protein_data"] = prot_stats

        logger.info("[4/4] 活性数据与交叉一致性校验")
        e, w, active_stats = _check_active_data()
        report["errors"].extend(e)
        report["warnings"].extend(w)
        report["details"]["active_data"] = active_stats

        e, w, cross_stats = _check_cross_consistency()
        report["errors"].extend(e)
        report["warnings"].extend(w)
        report["details"]["cross_consistency"] = cross_stats

        report["status"] = "FAILED" if report["errors"] else "PASSED"

    except Exception as e:
        logger.error(f"验证过程发生未捕获异常: {e}")
        traceback.print_exc()
        report["status"] = "CRASH"
        report["errors"].append(f"验证脚本异常: {e}")

    _write_report(report)
    _write_checksums(report["details"]["files"])

    logger.info("=" * 60)
    logger.info(f"验证结果: {report['status']}")
    logger.info(f"ERROR: {len(report['errors'])}, WARNING: {len(report['warnings'])}")
    if report["errors"]:
        for e in report["errors"]:
            logger.error(f"  {e}")
    if report["warnings"]:
        for w in report["warnings"]:
            logger.warning(f"  {w}")
    logger.info(f"报告: {REPORT_FILE}")
    logger.info(f"校验和: {CHECKSUM_FILE}")

    return 0 if report["status"] == "PASSED" else 1


def _write_report(report):
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)


def _write_checksums(files_info):
    checksums = {k: v.get("sha256") for k, v in files_info.items() if v.get("sha256")}
    with open(CHECKSUM_FILE, "w", encoding="utf-8") as f:
        json.dump(checksums, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
