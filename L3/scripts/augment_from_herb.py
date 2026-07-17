#!/usr/bin/env python3
"""
HERB 2.0 壮药化合物提取 — 多策略交叉检索
===========================================
策略:
  1. 成分名称精确/模糊匹配壮药名（如"一点红"匹配含"一点红"的成分名）
  2. TCMSP_id 跨库映射：HERB成分 → TCMSP MOL_ID → 已有 herb_ingredient_mapping
  3. PubChem_id 关联匹配
  4. 最终整合 + OB/DL过滤 + SMILES标准化 + 去泄漏

数据源:
  - D:\下载\HERB_herb_info_v2.txt (6892味药)
  - D:\下载\HERB_ingredient_info_v2.txt (44595成分 + SMILES)
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

HERB_HERB = r"D:\下载\HERB_herb_info_v2.txt"
HERB_ING = r"D:\下载\HERB_ingredient_info_v2.txt"
EXISTING_MAP = L3_RESULTS / "herb_ingredient_mapping.xlsx"
CM_POOL = L3_RESULTS / "zhuangyao_compound_pool.csv"
OUTPUT = L3_RESULTS / "zhuangyao_herb_augmented_pool.csv"
CPI_PATH = PROJECT_ROOT / "L4" / "results" / "experimental_actives_detail_cleaned_combined.csv"
PHENO_PATH = PROJECT_ROOT / "L4" / "results_v10_minibatch" / "phenotype_ferroptosis_dataset_v25_clean.csv"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(L3_LOGS / "herb_augment.log", encoding="utf-8", mode="w"),
              logging.StreamHandler(sys.stdout)], force=True
)
logger = logging.getLogger(__name__)

# 85味在HERB中匹配的壮药（精确68 + 包含17）
HERB_MATCHED_HERBS = {
    # 精确匹配 (68)
    "一点红", "八角枫", "九里香", "九层风", "三叉苦木", "三叶青藤", "三叶香茶菜", "三加",
    "大叶桉油", "大叶蒟", "大头陈", "山风", "山芝麻", "山香", "山桔叶", "山绿茶",
    "广狼毒", "小风艾", "小驳骨", "马蹄金", "天胡荽", "无根藤", "木槿花", "五指柑",
    "牛白藤", "牛耳枫", "牛尾菜", "玉叶金花", "四方木皮", "四方藤", "白花丹", "白背叶",
    "芒果叶", "羊开口", "走马风", "走马胎", "芙蓉叶", "苍耳草", "岗松", "金果榄",
    "金线风", "山银花", "阳桃根", "红鱼眼", "红药", "牛奶木", "汉桃叶", "老鸦嘴",
    "南板蓝根", "丁茄根", "大半边莲", "广钩藤", "光石韦", "余甘子汁", "大浮萍",
    "三七叶", "苦玄参", "鸡骨草", "毛鸡骨草", "秃叶黄柏", "红杜仲", "水半夏",
    "草豆蔻", "无患子果", "毛两面针", "广金钱草",
    # 包含匹配 (17)
    "九龙藤", "九里香油", "大头陈", "小槐花", "岗松油", "山桔叶",
    "广山楂叶", "白木香", "红杜仲", "走马胎",
    "羊开口", "牛奶木", "牛白藤", "大叶蒟",
    "汉桃叶", "山银花", "阳桃根",
}
# 去重
HERB_MATCHED_HERBS = list(HERB_MATCHED_HERBS)

# 排除已在现有池中的壮药
EXISTING_SOURCES = {"三七", "山药", "黄柏", "木香", "当归", "黄连", "柴胡", "半夏",
                    "骨碎补", "郁金", "余甘子", "槐花", "杜仲", "半边莲", "石韦",
                    "浮萍", "玄参", "蛇床子", "两面针", "海风藤", "山楂叶", "豆蔻",
                    "紫苏", "无患子", "金钱草", "板蓝根", "钩藤", "鸡骨草", "茄根"}


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
    logger.info("HERB 2.0 壮药化合物增强提取")
    logger.info("=" * 70)

    # 1. 加载HERB数据
    logger.info("加载 HERB ingredient_info_v2 (44,595 成分)...")
    ing = pd.read_csv(HERB_ING, sep="\t", low_memory=False,
                      usecols=["Ingredient_id", "Ingredient_name", "Canonical_smiles",
                               "MolWt", "OB_score", "Drug_likeness", "TCMSP_id", "PubChem_id"])
    ing["Canonical_smiles"] = ing["Canonical_smiles"].astype(str).str.strip()
    ing = ing[ing["Canonical_smiles"] != "nan"].copy()
    logger.info(f"有效SMILES: {len(ing)}/{44595}")

    # 2. 加载现有壮药池
    existing_pool = pd.read_csv(CM_POOL)
    existing_smiles = set(existing_pool["SMILES_std"].dropna().unique())
    logger.info(f"现有壮药池: {len(existing_pool)} 化合物, {len(existing_smiles)} 唯一SMILES")

    # 3. 策略1: 成分名称匹配壮药名
    found_strategy1 = []
    for herb_name in HERB_MATCHED_HERBS:
        mask = ing["Ingredient_name"].str.contains(herb_name, na=False, regex=False)
        matches = ing[mask]
        if len(matches) > 0:
            found_strategy1.append((herb_name, len(matches)))
            logger.info(f"  策略1-{herb_name}: {len(matches)} 个成分")
    logger.info(f"策略1匹配: {len(found_strategy1)} 味药")

    # 4. 策略2: TCMSP_id 跨库映射
    tcmsp_ing = ing[ing["TCMSP_id"].notna() & (ing["TCMSP_id"] != "nan")].copy()
    tcmsp_ing["TCMSP_id"] = tcmsp_ing["TCMSP_id"].astype(str).str.strip()
    logger.info(f"HERB中有TCMSP_id的成分: {len(tcmsp_ing)}")

    herb_map_df = pd.read_excel(EXISTING_MAP)
    herb_mol_ids = set(herb_map_df["MOL_ID"].astype(str).str.strip().unique())
    cross_matched = tcmsp_ing[tcmsp_ing["TCMSP_id"].isin(herb_mol_ids)]
    logger.info(f"策略2-TCMSP_id跨库: {len(cross_matched)} 个成分匹配已有MOL_ID")

    # 5. 合并所有唯一成分
    all_new_compounds = []
    seen_ids = set()

    # 策略1成分
    for herb_name in HERB_MATCHED_HERBS:
        mask = ing["Ingredient_name"].str.contains(herb_name, na=False, regex=False)
        for _, row in ing[mask].iterrows():
            iid = row["Ingredient_id"]
            if iid in seen_ids:
                continue
            seen_ids.add(iid)
            smi = row["Canonical_smiles"]
            if smi in existing_smiles:
                continue
            all_new_compounds.append({
                "source": f"HERB_{herb_name}",
                "Ingredient_id": iid,
                "Ingredient_name": row["Ingredient_name"],
                "SMILES_raw": smi,
                "MW_HERB": row["MolWt"],
                "OB_score": row["OB_score"],
                "Drug_likeness": row["Drug_likeness"],
                "TCMSP_id": row["TCMSP_id"],
                "PubChem_id": row["PubChem_id"],
                "strategy": "name_match",
            })

    # 策略2成分（TCMSP跨库）
    for _, row in cross_matched.iterrows():
        iid = row["Ingredient_id"]
        if iid in seen_ids:
            continue
        seen_ids.add(iid)
        smi = row["Canonical_smiles"]
        if smi in existing_smiles:
            continue
        all_new_compounds.append({
            "source": "HERB_TCMSP",
            "Ingredient_id": iid,
            "Ingredient_name": row["Ingredient_name"],
            "SMILES_raw": smi,
            "MW_HERB": row["MolWt"],
            "OB_score": row["OB_score"],
            "Drug_likeness": row["Drug_likeness"],
            "TCMSP_id": row["TCMSP_id"],
            "PubChem_id": row["PubChem_id"],
            "strategy": "tcmsp_cross",
        })

    df_new = pd.DataFrame(all_new_compounds)
    logger.info(f"新化合物(去重,排除已有): {len(df_new)}")
    logger.info(f"  策略1来源: {(df_new['strategy']=='name_match').sum()}")
    logger.info(f"  策略2来源: {(df_new['strategy']=='tcmsp_cross').sum()}")

    if len(df_new) == 0:
        logger.info("无新增化合物，直接使用现有池")
        df_final = existing_pool
    else:
        # 6. SMILES标准化
        df_new["SMILES_std"] = df_new["SMILES_raw"].apply(standardize_smiles)
        df_new = df_new[df_new["SMILES_std"].notna()].copy()
        logger.info(f"SMILES标准化: {len(df_new)} 通过")

        # 7. OB/DL过滤
        df_new["OB_score"] = pd.to_numeric(df_new["OB_score"], errors="coerce")
        df_new["Drug_likeness"] = pd.to_numeric(df_new["Drug_likeness"], errors="coerce")
        pass_ob = (df_new["OB_score"] >= 30) | (df_new["OB_score"].isna())
        pass_dl = (df_new["Drug_likeness"] >= 0.18) | (df_new["Drug_likeness"].isna())
        df_new["pass_ob_dl"] = pass_ob | pass_dl
        n_before = len(df_new)
        df_new = df_new[df_new["pass_ob_dl"]].copy()
        logger.info(f"OB/DL过滤: {n_before} -> {len(df_new)}")

        # 8. 计算描述符
        pains_params = FilterCatalogParams()
        pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        pains_catalog = FilterCatalog(pains_params)

        mw_list = []; logp_list = []; tpsa_list = []; hbd_list = []; hba_list = []
        qed_list = []; lip_pass = []; lip_viol = []; pains_pass_list = []

        for _, row in df_new.iterrows():
            mol = Chem.MolFromSmiles(row["SMILES_std"])
            if mol is None:
                for lst in [mw_list, logp_list, tpsa_list, hbd_list, hba_list, qed_list]:
                    lst.append(np.nan)
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

        # 9. MW验证: 跳过（缺少TCMSP MW参考）
        df_new = df_new[df_new["MW_calc"].notna()].copy()

        # 10. 去重
        before_dedup = len(df_new)
        df_new = df_new.drop_duplicates(subset=["SMILES_std"], keep="first").reset_index(drop=True)
        logger.info(f"SMILES去重: {before_dedup} -> {len(df_new)}")

        # 10.5 移除MOL000001 (anthocyanidin)
        if "MOL_ID" in df_new.columns:
            df_new = df_new[df_new["MOL_ID"] != "MOL000001"]
        # Also check by name
        is_antho = df_new["Ingredient_name"].str.lower().str.contains("anthocyanidin", na=False)
        if is_antho.sum() > 0:
            logger.warning(f"移除anthocyanidin相关: {is_antho.sum()} 条")
            df_new = df_new[~is_antho].copy()

        # 11. 合并到现有池
        combined = pd.concat([existing_pool, df_new], ignore_index=True, sort=False)
        combined = combined.drop_duplicates(subset=["SMILES_std"], keep="first").reset_index(drop=True)
        logger.info(f"合并后总化合物: {len(combined)}")

        # 12. 泄漏检查
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
            logger.warning("CPI/Pheno文件不可用，跳过泄漏检查")

        df_final = combined

    # 13. 保存
    df_final.to_csv(OUTPUT, index=False, encoding="utf-8")
    logger.info(f"\n最终壮药候选池: {len(df_final)} 个化合物")
    logger.info(f"保存: {OUTPUT}")

    # 完整性
    smi_nan = df_final["SMILES_std"].isna().sum()
    if smi_nan > 0:
        logger.error(f"SMILES NaN: {smi_nan}, 完整性FAIL")
        sys.exit(1)
    logger.info("完整性检查: PASS")


if __name__ == "__main__":
    main()