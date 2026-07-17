#!/usr/bin/env python3
"""
HERB 2.0 壮药化合物提取 v2 — 全量纳入，补漏关键化合物
==========================================================
策略:
  1. 成分名称精确/模糊匹配壮药名（如"一点红"匹配含"一点红"的成分名）
  2. TCMSP_id 跨库映射：HERB成分 → TCMSP MOL_ID → 已有 herb_ingredient_mapping
  3. [NEW] 全量纳入所有无TCMSP_id但有SMILES的HERB成分 → OB/DL过滤

数据源:
  - D:\下载\HERB_ingredient_info_v2.txt (44,595成分 + SMILES)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import SaltRemover

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
L3_RESULTS = PROJECT_ROOT / "L3" / "results"
L3_LOGS = PROJECT_ROOT / "L3" / "logs"
L3_LOGS.mkdir(parents=True, exist_ok=True)

HERB_ING = r"D:\下载\HERB_ingredient_info_v2.txt"
EXISTING_MAP = L3_RESULTS / "herb_ingredient_mapping.xlsx"
CM_POOL = L3_RESULTS / "zhuangyao_compound_pool.csv"
OUTPUT = L3_RESULTS / "zhuangyao_herb_augmented_pool.csv"
CPI_PATH = PROJECT_ROOT / "L4" / "results" / "experimental_actives_detail_cleaned_combined.csv"
PHENO_PATH = PROJECT_ROOT / "L4" / "results_v10_minibatch" / "phenotype_ferroptosis_dataset_v25_clean.csv"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(L3_LOGS / "herb_augment_v2.log", encoding="utf-8", mode="w"),
              logging.StreamHandler(sys.stdout)], force=True
)
logger = logging.getLogger(__name__)


def standardize_smiles(smi):
    if not smi or not isinstance(smi, str) or len(smi) < 3:
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        salt_remover = SaltRemover.SaltRemover()
        mol = salt_remover.StripMol(mol)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumAtoms())
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        normalizer = rdMolStandardize.Normalizer()
        mol = normalizer.normalize(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def main():
    logger.info("=" * 70)
    logger.info("HERB 2.0 壮药化合物增强提取 v2 (全量纳入)")
    logger.info("=" * 70)

    # 1. 加载HERB全量成分
    logger.info("加载 HERB ingredient_info_v2 (44,595 成分)...")
    ing = pd.read_csv(HERB_ING, sep="\t", low_memory=False,
                      usecols=["Ingredient_id", "Ingredient_name", "Canonical_smiles",
                               "MolWt", "OB_score", "Drug_likeness", "TCMSP_id", "PubChem_id"])
    ing["Canonical_smiles"] = ing["Canonical_smiles"].astype(str).str.strip()
    ing = ing[ing["Canonical_smiles"] != "nan"].copy()
    logger.info(f"有效SMILES: {len(ing)}")

    # 2. 加载现有壮药池
    existing_pool = pd.read_csv(CM_POOL)
    existing_smiles = set(existing_pool["SMILES_std"].dropna().unique())
    logger.info(f"现有壮药池: {len(existing_pool)} 化合物")

    # 3. 策略3（核心修复）：全量纳入所有无TCMSP_id但有SMILES的成分
    strategy3 = ing[ing["TCMSP_id"].isna() | (ing["TCMSP_id"] == "nan")]
    logger.info(f"策略3候选(无TCMSP_id): {len(strategy3)} 个")

    new_compounds = []
    seen_ids = set()
    for _, row in strategy3.iterrows():
        iid = row["Ingredient_id"]
        smi = row["Canonical_smiles"]
        if iid in seen_ids or smi in existing_smiles:
            continue
        seen_ids.add(iid)
        new_compounds.append({
            "source": "HERB_noTCMSP",
            "Ingredient_id": iid,
            "Ingredient_name": row["Ingredient_name"],
            "SMILES_raw": smi,
            "MW_HERB": row["MolWt"],
            "OB_score": row["OB_score"],
            "Drug_likeness": row["Drug_likeness"],
            "TCMSP_id": row["TCMSP_id"],
            "PubChem_id": row["PubChem_id"],
        })
    logger.info(f"策略3新化合物(去重): {len(new_compounds)}")

    # 4. 同时也纳入TCMSP_id匹配的成分（保持v1策略2）
    tcmsp_ing = ing[ing["TCMSP_id"].notna() & (ing["TCMSP_id"] != "nan")].copy()
    herb_map_df = pd.read_excel(EXISTING_MAP)
    herb_mol_ids = set(herb_map_df["MOL_ID"].astype(str).str.strip().unique())
    n_tcmsp_added = 0
    for _, row in tcmsp_ing.iterrows():
        tc_ids = str(row["TCMSP_id"]).split(";")
        tc_ids = [x.strip() for x in tc_ids if x.strip()]
        if not any(tid in herb_mol_ids for tid in tc_ids):
            continue
        iid = row["Ingredient_id"]
        if iid in seen_ids:
            continue
        smi = row["Canonical_smiles"]
        if smi in existing_smiles:
            continue
        n_tcmsp_added += 1
        seen_ids.add(iid)
        new_compounds.append({
            "source": "HERB_TCMSP",
            "Ingredient_id": iid,
            "Ingredient_name": row["Ingredient_name"],
            "SMILES_raw": smi,
            "MW_HERB": row["MolWt"],
            "OB_score": row["OB_score"],
            "Drug_likeness": row["Drug_likeness"],
            "TCMSP_id": row["TCMSP_id"],
            "PubChem_id": row["PubChem_id"],
        })
    logger.info(f"策略2(TCMSP跨库)新增: {n_tcmsp_added}")

    df_new = pd.DataFrame(new_compounds)
    logger.info(f"合并新化合物: {len(df_new)}")

    # 5. SMILES标准化
    df_new["SMILES_std"] = df_new["SMILES_raw"].apply(standardize_smiles)
    before = len(df_new)
    df_new = df_new[df_new["SMILES_std"].notna()].copy()
    logger.info(f"SMILES标准化: {before} -> {len(df_new)}")

    # 6. OB/DL过滤
    df_new["OB_score"] = pd.to_numeric(df_new["OB_score"], errors="coerce")
    df_new["Drug_likeness"] = pd.to_numeric(df_new["Drug_likeness"], errors="coerce")
    # OB>=30 或 DL>=0.18，或两者都缺失则保留（后续由RDKit判断）
    pass_ob = (df_new["OB_score"] >= 30) | (df_new["OB_score"].isna())
    pass_dl = (df_new["Drug_likeness"] >= 0.18) | (df_new["Drug_likeness"].isna())
    df_new = df_new[pass_ob | pass_dl].copy()
    logger.info(f"OB/DL过滤后: {len(df_new)}")

    # 7. 计算RDKit描述符
    pains_params = FilterCatalogParams()
    pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    pains_catalog = FilterCatalog(pains_params)

    mw_list = []; logp_list = []; tpsa_list = []; hbd_list = []; hba_list = []
    qed_list = []; lip_pass = []; lip_viol = []; pains_pass_list = []

    for _, row in df_new.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES_std"])
        if mol is None:
            mw_list.append(np.nan); logp_list.append(np.nan); tpsa_list.append(np.nan)
            hbd_list.append(np.nan); hba_list.append(np.nan); qed_list.append(np.nan)
            lip_pass.append(False); lip_viol.append(99); pains_pass_list.append(False)
            continue
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        qed = Descriptors.qed(mol)
        mw_list.append(mw); logp_list.append(logp); tpsa_list.append(tpsa)
        hbd_list.append(hbd); hba_list.append(hba); qed_list.append(qed)
        viol = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        lip_pass.append(viol <= 1); lip_viol.append(viol)
        pains_pass_list.append(len(pains_catalog.GetMatches(mol)) == 0)

    df_new["MW_calc"] = mw_list; df_new["LogP_calc"] = logp_list
    df_new["TPSA_calc"] = tpsa_list; df_new["HBD_calc"] = hbd_list
    df_new["HBA_calc"] = hba_list; df_new["QED"] = qed_list
    df_new["Lipinski_Pass"] = lip_pass; df_new["Lipinski_Violations"] = lip_viol
    df_new["PAINS_Pass"] = pains_pass_list

    logger.info(f"Lipinski通过: {sum(lip_pass)}/{len(df_new)}")
    logger.info(f"PAINS通过: {sum(pains_pass_list)}/{len(df_new)}")

    # 移除计算失败的
    df_new = df_new[df_new["MW_calc"].notna()].copy()
    logger.info(f"描述符计算后: {len(df_new)}")

    # 8. 去重
    before_dedup = len(df_new)
    df_new = df_new.drop_duplicates(subset=["SMILES_std"], keep="first").reset_index(drop=True)
    logger.info(f"SMILES去重: {before_dedup} -> {len(df_new)}")

    # 9. 名称统一 — 确保 molecule_name 列在所有来源中一致
    # 原池有 molecule_name; df_new 有 Ingredient_name; 统一到 molecule_name
    if "molecule_name" not in df_new.columns:
        df_new["molecule_name"] = None
    mask_no_name = df_new["molecule_name"].isna()
    df_new.loc[mask_no_name, "molecule_name"] = df_new.loc[mask_no_name, "Ingredient_name"]
    logger.info(f"df_new molecule_name统一: {df_new['molecule_name'].notna().sum()}/{len(df_new)}")

    # 10. 合并
    combined = pd.concat([existing_pool, df_new], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["SMILES_std"], keep="first").reset_index(drop=True)
    logger.info(f"合并后总化合物: {len(combined)}")

    # 11. 泄漏检查
    n_leaked = 0
    if CPI_PATH.exists() and PHENO_PATH.exists():
        cpi = pd.read_csv(CPI_PATH, low_memory=False)
        pheno = pd.read_csv(PHENO_PATH, low_memory=False)
        train_smiles = set()
        for col in ["canonical_smiles", "SMILES"]:
            if col in cpi.columns:
                train_smiles.update(cpi[col].dropna().astype(str).str.strip().unique())
            if col in pheno.columns:
                train_smiles.update(pheno[col].dropna().astype(str).str.strip().unique())
        mask = combined["SMILES_std"].isin(train_smiles)
        n_leaked = mask.sum()
        if n_leaked > 0:
            logger.warning(f"发现 {n_leaked} 个泄漏，正在移除")
            combined = combined[~mask].copy().reset_index(drop=True)
    else:
        logger.warning("CPI/Pheno不可用，跳过泄漏检查")

    # 12. 验证关键化合物
    df_final = combined
    caryo = df_final[df_final["Ingredient_name"].str.contains(
        "caryophyllene|Caryophyllene|石竹烯", case=False, na=False
    )]
    logger.info(f"β-石竹烯类化合物: {len(caryo)} 个在最终池中")
    for _, r in caryo.iterrows():
        logger.info(f"  {r['Ingredient_name']}: SMILES={str(r['SMILES_std'])[:60]}")

    # 13. 保存
    df_final.to_csv(OUTPUT, index=False, encoding="utf-8")
    logger.info(f"\n最终壮药候选池: {len(df_final)} 个化合物")
    logger.info(f"保存: {OUTPUT}")

    smi_nan = df_final["SMILES_std"].isna().sum()
    if smi_nan > 0:
        logger.error(f"SMILES NaN: {smi_nan}, FAIL")
        sys.exit(1)
    logger.info("完整性: PASS")


if __name__ == "__main__":
    main()