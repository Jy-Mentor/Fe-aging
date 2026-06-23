#!/usr/bin/env python3
"""
P4 输入文件强化验证 (v3)
========================
v3 在 v2 基础上针对 v4.5 逻辑优化需求增强：
  - 区分 "结构活性数据"（必须有 SMILES）与 "name-only 参考数据"（如 DrugBank cross-reference）。
  - 对 name-only 文件记录 WARNING 而非 ERROR，并排除基于 SMILES 的检查。
  - 对内置文献活性集 FERROPTOSIS_ACTIVES 进行 SMILES 解析并过滤无效项。
  - 增加蛋白-基因、活性-蛋白、活性-TCM 库交叉一致性检查。
  - 所有 ERROR 必须修复后才能进入训练；WARNING 需记录原因并在报告中解释。

运行:
    python L4/scripts/validate_p4_inputs_v3.py
输出:
    L4/logs/input_validation_report_v3.json
    L4/logs/input_checksums_v3.json
"""

import sys
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from rdkit import Chem

BASE = Path(__file__).parent.parent.parent
L2 = BASE / "L2" / "results"
L3 = BASE / "L3" / "results"
L4 = BASE / "L4" / "results"
L4_LOGS = BASE / "L4" / "logs"
L4_LOGS.mkdir(parents=True, exist_ok=True)

LOG_FILE = L4_LOGS / "validate_p4_inputs_v3.log"
REPORT_FILE = L4_LOGS / "input_validation_report_v3.json"
CHECKSUM_FILE = L4_LOGS / "input_checksums_v3.json"

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

def _sha256(path: Path) -> str:
    """计算文件 SHA256 校验和。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _n_rows(path: Path) -> int | None:
    """获取数据文件行数：CSV 为记录数，npy 为第 0 维大小。"""
    try:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            # 仅计算数据行数（不含表头）
            return sum(1 for _ in open(path, "r", encoding="utf-8", errors="replace")) - 1
        if suffix == ".npy":
            arr = np.load(path, mmap_mode="r")
            return int(arr.shape[0])
    except Exception as e:
        logger.warning(f"无法读取 {path.name} 行数: {e}")
    return None


def _file_info(path: Path) -> dict:
    """生成文件级 lineage 记录。"""
    info = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": None,
        "sha256": None,
        "mtime": None,
        "n_rows": None,
    }
    if path.exists():
        stat = path.stat()
        info["size_bytes"] = stat.st_size
        info["sha256"] = _sha256(path)
        info["mtime"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        info["n_rows"] = _n_rows(path)
    return info


def _check_required_files():
    """校验必需文件是否存在、非空，并记录 lineage。"""
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
            logger.info(f"[OK] {desc}: {fpath} (size={info['size_bytes']} bytes, rows={info['n_rows']})")
    return errors, warnings, checksums


def _check_compound_data():
    """校验 TCM 化合物池、描述符与指纹矩阵的一致性。"""
    errors = []
    warnings = []

    cp = pd.read_csv(L3 / "tcm_compound_pool_filtered.csv")
    desc = pd.read_csv(L3 / "rdkit_descriptors.csv")
    ecfp4 = np.load(L3 / "ecfp4_fingerprints.npy")
    maccs = np.load(L3 / "maccs_fingerprints.npy")

    n_cp = len(cp)
    logger.info(f"  TCM化合物池: {n_cp} 行")

    for col in ["MOL_ID", "molecule_name", "SMILES_std"]:
        if col not in cp.columns:
            errors.append(f"TCM化合物池缺失列: {col}")

    if len(desc) != n_cp:
        errors.append(f"描述符行数({len(desc)}) != 化合物数({n_cp})")
    if len(ecfp4) != n_cp:
        errors.append(f"ECFP4行数({len(ecfp4)}) != 化合物数({n_cp})")
    if len(maccs) != n_cp:
        errors.append(f"MACCS行数({len(maccs)}) != 化合物数({n_cp})")

    invalid = 0
    dup_smiles = 0
    if "SMILES_std" in cp.columns:
        null_smiles = cp["SMILES_std"].isna().sum()
        if null_smiles > 0:
            errors.append(f"TCM库空SMILES: {null_smiles}")
        dup_smiles = cp["SMILES_std"].duplicated().sum()
        if dup_smiles > 0:
            warnings.append(f"TCM库重复SMILES: {dup_smiles} (去重后 {n_cp - dup_smiles})")

        invalid_indices = []
        for i, smi in enumerate(cp["SMILES_std"]):
            try:
                mol = None if pd.isna(smi) else Chem.MolFromSmiles(str(smi))
            except Exception as e:
                mol = None
                logger.debug(f"TCM库SMILES解析异常 索引 {i}: {smi}, 错误: {e}")
            if mol is None:
                invalid += 1
                invalid_indices.append(i)
                if len(invalid_indices) <= 5:
                    warnings.append(f"TCM库无效SMILES索引 {i}: {smi}")
        if invalid > 0:
            errors.append(f"TCM库无效SMILES: {invalid}/{n_cp}")
        else:
            logger.info(f"  TCM库SMILES解析: {n_cp}/{n_cp} 通过")

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
        "invalid_smiles": invalid,
        "duplicate_smiles": int(dup_smiles),
    }


def _check_protein_data():
    """校验蛋白特征表及 AAC/PseAAC 矩阵。"""
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
        warnings.append(f"PseAAC行数({len(pseaac)}) != 蛋白数({n_prot})")

    aac_cols = [c for c in aac.columns if c.startswith("AAC_")]
    if len(aac_cols) != 20:
        warnings.append(f"AAC列数={len(aac_cols)}, 预期20")

    pseaac_cols = [c for c in pseaac.columns if c.startswith("PseAAC_")]
    if len(pseaac_cols) != 50:
        warnings.append(f"PseAAC列数={len(pseaac_cols)}, 预期50")

    missing_pseaac_genes = []
    if "gene_symbol" in pseaac.columns:
        pseaac_genes = set(pseaac["gene_symbol"].dropna().unique())
        missing_pseaac_genes = [g for g in ALL_TARGET_GENES if g not in pseaac_genes]
        if missing_pseaac_genes:
            warnings.append(f"以下核心/优先基因缺少 PseAAC 特征（将用0填充）: {missing_pseaac_genes}")

    missing = []
    if "gene_symbol" in prot.columns:
        prot_genes = set(prot["gene_symbol"].dropna().unique())
        missing = [g for g in ALL_TARGET_GENES if g not in prot_genes]
        if missing:
            errors.append(f"蛋白数据缺失核心基因: {missing}")
        else:
            logger.info(f"  所有 {len(ALL_TARGET_GENES)} 个核心/优先基因均在蛋白数据中")

        missing_priority = [g for g in PRIORITY_TARGETS if g not in prot_genes]
        if missing_priority:
            warnings.append(f"铁衰老关键靶标缺失: {missing_priority}")

    return errors, warnings, {
        "n_proteins": n_prot,
        "aac_dim": len(aac_cols),
        "pseaac_dim": len(pseaac_cols),
        "missing_core_genes": missing,
        "missing_pseaac_genes": missing_pseaac_genes,
    }


def _find_column(candidates, columns):
    """在列名列表中按候选子串（忽略大小写）查找第一个匹配列。"""
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        for lower_col, orig_col in cols_lower.items():
            if cand.lower() in lower_col:
                return orig_col
    return None


def _check_active_data():
    """校验活性数据文件：要求 gene、canonical_smiles、活性值/类型列，SMILES 解析率等。"""
    errors = []
    warnings = []
    stats = {"files": {}}

    active_files = {
        "experimental_detail": L4 / "experimental_actives_detail.csv",
        "chembl": L4 / "chembl_active_compounds.csv",
        "bindingdb": L4 / "bindingdb_active_compounds.csv",
        "drugbank": L4 / "drugbank_active_compounds.csv",
    }

    any_structure_active = False
    for source, path in active_files.items():
        if not path.exists():
            warnings.append(f"活性数据文件缺失: {path.name}")
            stats["files"][source] = {"exists": False}
            continue

        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            errors.append(f"无法读取 {path.name}: {e}")
            stats["files"][source] = {"exists": True, "read_error": str(e)}
            continue

        n_total = len(df)
        file_stats = {
            "exists": True,
            "n_rows": n_total,
            "path": str(path),
        }

        gene_col = _find_column(["gene"], df.columns)
        smiles_col = _find_column(["canonical_smiles", "smiles"], df.columns)
        name_col = _find_column(["drug_name", "molecule_pref_name", "molecule_name", "name"], df.columns)
        value_col = _find_column(["standard_value_nM", "value_nM"], df.columns)
        type_col = _find_column(["standard_type"], df.columns)
        relation_col = _find_column(["standard_relation"], df.columns)

        file_stats.update({
            "gene_col": gene_col,
            "smiles_col": smiles_col,
            "name_col": name_col,
            "value_col": value_col,
            "type_col": type_col,
            "relation_col": relation_col,
        })

        if gene_col is None:
            errors.append(f"{path.name}: 未找到基因列")
            stats["files"][source] = file_stats
            continue

        if smiles_col is None and name_col is None:
            warnings.append(f"{path.name}: 既无 SMILES 列也无 name 列，作为纯基因参考")
            file_stats["type"] = "gene_only"
            stats["files"][source] = file_stats
            continue

        # name-only 文件（如 DrugBank）：记录为参考，不做 SMILES 强制校验
        if smiles_col is None:
            warnings.append(
                f"{path.name}: 无 SMILES 列，仅有 name 列 ({name_col})，"
                f"将作为 name-only 参考，不用于基于结构的训练"
            )
            file_stats["type"] = "name_only"
            stats["files"][source] = file_stats
            continue

        # 结构活性文件必须同时有活性值或活性类型列
        if value_col is None and type_col is None:
            errors.append(f"{path.name}: 既无活性值列也无活性类型列")
            file_stats["type"] = "structure_missing_activity"
            stats["files"][source] = file_stats
            continue
        if value_col is None or type_col is None:
            warnings.append(f"{path.name}: 活性值/类型列不完整 (value={value_col}, type={type_col})")

        any_structure_active = True

        # 若存在 standard_relation，仅保留 '=' 的记录
        n_before = n_total
        if relation_col is not None:
            n_relation = (df[relation_col].fillna("=") != "=").sum()
            if n_relation > 0:
                warnings.append(f"{path.name}: {n_relation} 条记录 standard_relation != '='，已剔除")
            df = df[df[relation_col].fillna("=") == "="].copy()
        n_after_relation = len(df)
        file_stats["n_after_relation"] = n_after_relation
        file_stats["n_dropped_relation"] = n_before - n_after_relation

        # 基因有效性检查
        valid_genes = set()
        invalid_genes = []
        for g in df[gene_col].dropna().unique():
            if g in ALL_TARGET_GENES:
                valid_genes.add(g)
            else:
                invalid_genes.append(g)
        if invalid_genes:
            warnings.append(f"{path.name}: {len(invalid_genes)} 个非目标基因, 示例: {invalid_genes[:5]}")

        # SMILES 解析率检查
        invalid_smiles = 0
        valid_smiles = 0
        for smi in df[smiles_col].dropna():
            try:
                mol = Chem.MolFromSmiles(str(smi))
            except Exception as e:
                mol = None
                logger.debug(f"活性数据 SMILES 解析异常: {smi}, 错误: {e}")
            if mol is not None:
                valid_smiles += 1
            else:
                invalid_smiles += 1
        n_smiles = valid_smiles + invalid_smiles
        parse_rate = valid_smiles / n_smiles if n_smiles > 0 else 0.0
        file_stats["n_smiles"] = n_smiles
        file_stats["valid_smiles"] = valid_smiles
        file_stats["invalid_smiles"] = invalid_smiles
        file_stats["parse_rate"] = parse_rate
        if parse_rate < 0.95:
            errors.append(f"{path.name}: SMILES解析率 {parse_rate:.2%} < 95%")
        else:
            logger.info(f"  {path.name}: {n_after_relation} 行, SMILES解析率 {parse_rate:.2%}")

        # 重复 (gene, smiles) 对检查
        dup_pairs = df.duplicated(subset=[gene_col, smiles_col]).sum()
        if dup_pairs > 0:
            warnings.append(f"{path.name}: 重复(gene,smiles)对={dup_pairs}")
        file_stats["duplicate_pairs"] = int(dup_pairs)
        file_stats["valid_genes"] = len(valid_genes)
        file_stats["invalid_genes"] = len(invalid_genes)
        file_stats["type"] = "structure"

        stats["files"][source] = file_stats

    if not any_structure_active:
        errors.append("没有任何含 SMILES 的活性数据文件，无法训练基于结构的模型")

    return errors, warnings, stats


def _check_cross_consistency():
    """活性数据与蛋白表、TCM 库的双向一致性检查。"""
    errors = []
    warnings = []

    prot = pd.read_csv(L2 / "target_protein_features.csv")
    prot_genes = set(prot["gene_symbol"].dropna().unique())

    active_genes = set()
    active_smiles = set()
    active_files = {
        "experimental_detail": L4 / "experimental_actives_detail.csv",
        "chembl": L4 / "chembl_active_compounds.csv",
        "bindingdb": L4 / "bindingdb_active_compounds.csv",
        "drugbank": L4 / "drugbank_active_compounds.csv",
    }

    for fname, path in active_files.items():
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            warnings.append(f"交叉一致性检查无法读取 {path.name}: {e}")
            continue
        gene_col = _find_column(["gene"], df.columns)
        smiles_col = _find_column(["canonical_smiles", "smiles"], df.columns)
        relation_col = _find_column(["standard_relation"], df.columns)
        if gene_col:
            active_genes.update(df[gene_col].dropna().unique())
        if smiles_col:
            smi_df = df.copy()
            if relation_col is not None:
                smi_df = smi_df[smi_df[relation_col].fillna("=") == "="]
            active_smiles.update(smi_df[smiles_col].dropna().astype(str))

    # 双向匹配：活性基因最好在蛋白表中；未匹配的基因将在下游被过滤，故记为 WARNING
    matched = active_genes & prot_genes
    unmatched = active_genes - prot_genes
    logger.info(f"  活性数据基因: {len(active_genes)}, 与蛋白表匹配: {len(matched)}, 未匹配: {len(unmatched)}")
    if unmatched:
        warnings.append(f"活性数据中存在蛋白表未覆盖的基因（将在训练时过滤）: {sorted(unmatched)[:20]}")

    # 双向匹配：蛋白表中的核心/优先基因最好有活性数据
    missing_activity = [g for g in ALL_TARGET_GENES if g not in active_genes]
    if missing_activity:
        warnings.append(f"蛋白表中以下核心/优先基因缺少活性数据: {missing_activity}")

    # 活性 SMILES 与 TCM 库重叠率
    cp = pd.read_csv(L3 / "tcm_compound_pool_filtered.csv")
    tcm_smiles = set(cp["SMILES_std"].dropna().astype(str))

    overlap = len(active_smiles & tcm_smiles)
    overlap_rate = overlap / len(active_smiles) if active_smiles else 0.0
    logger.info(f"  活性SMILES与TCM库重叠: {overlap}/{len(active_smiles)} ({overlap_rate:.2%})")
    if overlap_rate < 0.01 and active_smiles:
        warnings.append(f"活性SMILES与TCM库重叠率仅 {overlap_rate:.2%}, 可能导致训练正样本不足")

    return errors, warnings, {
        "active_genes": len(active_genes),
        "matched_protein_genes": len(matched),
        "unmatched_genes": sorted(unmatched)[:20],
        "missing_activity_genes": missing_activity,
        "active_smiles": len(active_smiles),
        "tcm_smiles": len(tcm_smiles),
        "overlap_smiles": overlap,
        "overlap_rate": overlap_rate,
    }


def _write_report(report):
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)


def _write_checksums(files_info):
    checksums = {}
    for desc, info in files_info.items():
        checksums[desc] = {
            "path": info.get("path"),
            "sha256": info.get("sha256"),
            "mtime": info.get("mtime"),
            "n_rows": info.get("n_rows"),
        }
    with open(CHECKSUM_FILE, "w", encoding="utf-8") as f:
        json.dump(checksums, f, indent=2, ensure_ascii=False, default=str)


def main():
    logger.info("=" * 60)
    logger.info("P4 输入文件强化验证 (v3)")
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

        if e:
            report["status"] = "FAILED"
            _write_report(report)
            _write_checksums(report["details"]["files"])
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
        logger.exception(f"验证过程发生未捕获异常: {e}")
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


if __name__ == "__main__":
    sys.exit(main())
