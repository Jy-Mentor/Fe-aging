#!/usr/bin/env python3
"""
修复 anthocyanidin (MOL000001) 身份不明问题 + 重跑去泄漏检查
================================================================
根因分析:
  1. TCMSP MOL000001 名称 "anthocyanidin" 是化学类别名，非单一化合物
  2. MW=251.22 与任何已知花青素 (MW 270-290 Da) 均不匹配
  3. SMILES C=C1C(=O)O[C@@H]2C[C@H](C)[C@@H]3CC[C@@H](O)[C@@]3(C)C[C@H]12
     实际为二萜内酯结构（非黄酮类花青素骨架）
  4. TCMSP 官网标注 PubChem CID: N/A（权威数据库无匹配）
  5. CAS 84082-34-8 是"花青素类"混合物 CAS，非单一化合物
  6. InChIKey YLJQHJNWIHUOKW-UHFFFAOYSA-N 无 PubChem 对应条目

修复策略:
  遵循项目规则"不准捏造数据 + 缺失数据必须写日志警告"，
  对身份无法验证的条目予以移除并记录原因。
  影响范围: 1 个化合物 (1/592 = 0.17%)，对模型预测影响可忽略。

同步任务:
  2. 对壮药对齐后扩展的候选池 (592 化合物) 重跑去泄漏检查
     (因为新增 75 个壮药来源化合物可能与 CPI/pheno 训练集重叠)
"""
from __future__ import annotations

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
L3_RESULTS = PROJECT_ROOT / "L3" / "results"
L4_RESULTS = PROJECT_ROOT / "L4" / "results"
L4_V10 = PROJECT_ROOT / "L4" / "results_v10_minibatch"
L3_LOGS = PROJECT_ROOT / "L3" / "logs"
L3_LOGS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            L3_LOGS / "fix_anthocyanidin_releak.log", encoding="utf-8", mode="w"
        ),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

TCM_POOL = L3_RESULTS / "tcm_compound_pool_tox_filtered.csv"
CPI_PATH = L4_RESULTS / "experimental_actives_detail_cleaned_combined.csv"
PHENO_PATH = L4_V10 / "phenotype_ferroptosis_dataset_v25_clean.csv"
REMOVAL_LOG = L3_RESULTS / "removed_compounds_log.csv"


def canon(smi: str) -> str | None:
    if pd.isna(smi):
        return None
    mol = Chem.MolFromSmiles(str(smi).strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def backup_file(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(f".csv.backup_pre_anthocyanidin_{ts}")
    shutil.copy2(path, bak)
    logger.info(f"备份: {path} -> {bak}")
    return bak


def remove_anthocyanidin(pool_path: Path) -> tuple[int, int]:
    """移除 MOL000001 anthocyanidin 条目并记录日志"""
    logger.info("=" * 70)
    logger.info("步骤 1: 移除身份不明的 anthocyanidin (MOL000001)")
    logger.info("=" * 70)

    df = pd.read_csv(pool_path, low_memory=False)
    n_before = len(df)
    logger.info(f"读取候选池: {n_before} 个化合物")

    mask_antho = df["MOL_ID"].astype(str).str.strip() == "MOL000001"
    n_antho = mask_antho.sum()
    logger.info(f"匹配 MOL000001 条目数: {n_antho}")

    if n_antho == 0:
        logger.warning("未找到 MOL000001 条目，可能已被移除")
        return n_before, n_before

    if n_antho > 1:
        logger.error(f"发现 {n_antho} 个 MOL000001 重复条目，数据异常")
        raise RuntimeError(f"MOL000001 出现 {n_antho} 次，应为 1 次")

    removed_row = df[mask_antho].iloc[0]
    removal_record = pd.DataFrame([{
        "removal_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "MOL_ID": removed_row["MOL_ID"],
        "molecule_name": removed_row["molecule_name"],
        "SMILES_std": removed_row["SMILES_std"],
        "mw": removed_row.get("mw", ""),
        "reason": (
            "身份不可验证: TCMSP MOL000001 'anthocyanidin' 为化学类别名非单一化合物; "
            "MW=251.22 与已知花青素 (270-290 Da) 不符; "
            "SMILES 为二萜内酯结构非花青素骨架; "
            "TCMSP PubChem CID=N/A; CAS 84082-34-8 为混合物 CAS; "
            "InChIKey YLJQHJNWIHUOKW-UHFFFAOYSA-N 无 PubChem 匹配"
        ),
        "action": "removed",
        "source": "TCMSP MOL000001",
    }])

    if REMOVAL_LOG.exists():
        existing_log = pd.read_csv(REMOVAL_LOG)
        combined = pd.concat([existing_log, removal_record], ignore_index=True)
    else:
        combined = removal_record
    combined.to_csv(REMOVAL_LOG, index=False, encoding="utf-8")
    logger.info(f"移除日志已更新: {REMOVAL_LOG}")

    backup_file(pool_path)

    cleaned = df[~mask_antho].copy().reset_index(drop=True)
    cleaned.to_csv(pool_path, index=False, encoding="utf-8")
    n_after = len(cleaned)
    logger.info(f"移除完成: {n_before} -> {n_after} (移除 {n_before - n_after} 个)")
    return n_before, n_after


def check_and_fix_leaks(
    tcm_path: Path, cpi_path: Path, pheno_path: Path, inplace: bool = True
) -> tuple[int, int]:
    """对候选池执行 SMILES 泄漏检查并原地修复"""
    logger.info("=" * 70)
    logger.info("步骤 2: 重跑去泄漏检查 (扩展池 592 -> 移除 anthocyanidin 后)")
    logger.info("=" * 70)

    if not cpi_path.exists():
        raise FileNotFoundError(f"CPI 文件不存在: {cpi_path}")
    if not pheno_path.exists():
        raise FileNotFoundError(f"表型文件不存在: {pheno_path}")

    tcm = pd.read_csv(tcm_path, low_memory=False)
    cpi = pd.read_csv(cpi_path, low_memory=False)
    pheno = pd.read_csv(pheno_path, low_memory=False)

    smiles_col = "SMILES_std" if "SMILES_std" in tcm.columns else "canonical_smiles"
    if smiles_col not in tcm.columns:
        raise KeyError(f"TCM 池缺少 SMILES 列: {smiles_col}")

    train_smiles = set()
    train_smiles.update(
        cpi["canonical_smiles"].dropna().astype(str).str.strip().unique()
    )
    if "canonical_smiles" in pheno.columns:
        train_smiles.update(
            pheno["canonical_smiles"].dropna().astype(str).str.strip().unique()
        )

    logger.info(f"TCM 候选池: {len(tcm)} 个化合物")
    logger.info(f"CPI 训练集 SMILES: {len(cpi['canonical_smiles'].dropna().unique())}")
    if "canonical_smiles" in pheno.columns:
        logger.info(
            f"表型训练集 SMILES: {len(pheno['canonical_smiles'].dropna().unique())}"
        )
    logger.info(f"训练集去重 SMILES 总数: {len(train_smiles)}")

    raw = tcm[smiles_col].astype(str).str.strip()
    can = raw.apply(canon)
    can_set = {s for s in can if s is not None}
    overlap_raw = raw.isin(train_smiles)
    overlap_can = can.isin(train_smiles)
    overlap = overlap_raw | overlap_can
    n_leaked = int(overlap.sum())

    logger.info(f"泄漏化合物数: {n_leaked}")

    n_removed = 0
    if n_leaked > 0:
        for _, row in tcm[overlap].iterrows():
            logger.warning(
                f"  泄漏: MOL_ID={row['MOL_ID']}, "
                f"name={row.get('molecule_name', '')}, "
                f"SMILES={row[smiles_col]}"
            )
        if inplace:
            backup_file(tcm_path)
            cleaned = tcm[~overlap].copy().reset_index(drop=True)
            cleaned.to_csv(tcm_path, index=False, encoding="utf-8")
            n_removed = len(tcm) - len(cleaned)
            logger.info(
                f"原地修复: {len(tcm)} -> {len(cleaned)} (移除 {n_removed} 个泄漏化合物)"
            )
    else:
        logger.info("无泄漏，无需修复")

    return n_leaked, n_removed


def verify_pool_integrity(pool_path: Path) -> dict:
    """验证候选池完整性"""
    logger.info("=" * 70)
    logger.info("步骤 3: 验证候选池完整性")
    logger.info("=" * 70)

    df = pd.read_csv(pool_path, low_memory=False)
    n = len(df)
    logger.info(f"总化合物数: {n}")

    checks = {}

    checks["total"] = n
    checks["mol_id_unique"] = df["MOL_ID"].is_unique
    checks["smiles_nan"] = int(df["SMILES_std"].isna().sum())
    checks["smiles_invalid"] = 0
    for s in df["SMILES_std"].dropna():
        if Chem.MolFromSmiles(str(s)) is None:
            checks["smiles_invalid"] += 1

    if "is_zhuangyao" in df.columns:
        checks["zhuangyao_count"] = int(df["is_zhuangyao"].sum())
        checks["has_herb_source"] = int(df["herb_source"].notna().sum())
    else:
        checks["zhuangyao_count"] = -1
        checks["has_herb_source"] = -1

    checks["anthocyanidin_present"] = (
        df["MOL_ID"].astype(str).str.strip() == "MOL000001"
    ).sum()
    checks["molecule_name_anthocyanidin"] = (
        df["molecule_name"].astype(str).str.lower() == "anthocyanidin"
    ).sum()

    logger.info(f"  MOL_ID 唯一: {checks['mol_id_unique']}")
    logger.info(f"  SMILES NaN: {checks['smiles_nan']}")
    logger.info(f"  SMILES 无效: {checks['smiles_invalid']}")
    logger.info(f"  壮药来源化合物: {checks['zhuangyao_count']}")
    logger.info(f"  有 herb_source 标注: {checks['has_herb_source']}")
    logger.info(f"  MOL000001 残留: {checks['anthocyanidin_present']}")
    logger.info(f"  anthocyanidin 名称残留: {checks['molecule_name_anthocyanidin']}")

    all_pass = (
        checks["mol_id_unique"]
        and checks["smiles_nan"] == 0
        and checks["smiles_invalid"] == 0
        and checks["anthocyanidin_present"] == 0
        and checks["molecule_name_anthocyanidin"] == 0
    )
    checks["all_pass"] = all_pass
    logger.info(f"  整体完整性: {'PASS' if all_pass else 'FAIL'}")
    return checks


def main() -> None:
    if not TCM_POOL.exists():
        raise FileNotFoundError(f"TCM 候选池不存在: {TCM_POOL}")

    n_before_antho, n_after_antho = remove_anthocyanidin(TCM_POOL)

    n_leaked, n_removed_leak = check_and_fix_leaks(
        TCM_POOL, CPI_PATH, PHENO_PATH, inplace=True
    )

    integrity = verify_pool_integrity(TCM_POOL)

    logger.info("=" * 70)
    logger.info("修复总结")
    logger.info("=" * 70)
    logger.info(f"  anthocyanidin 移除: {n_before_antho} -> {n_after_antho}")
    logger.info(f"  泄漏移除: {n_removed_leak}")
    logger.info(f"  最终候选池大小: {integrity['total']}")
    logger.info(f"  完整性检查: {'PASS' if integrity['all_pass'] else 'FAIL'}")

    if not integrity["all_pass"]:
        logger.error("完整性检查失败，请人工检查")
        sys.exit(1)
    logger.info("所有修复已完成，候选池通过完整性验证")


if __name__ == "__main__":
    main()
