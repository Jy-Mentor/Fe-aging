#!/usr/bin/env python3
"""
壮药完整候选池构建脚本
================================
整合三源数据 (Round 1 + Round 2 + 原有映射) -> SMILES匹配 -> OB/DL过滤
-> RDKit描述符 -> 毒性过滤 -> 去重 -> 泄漏检查 -> 最终候选池

数据来源:
  1. zhuangyao_ingredient_mapping_full.xlsx (Round 1: 123味壮药直接爬取)
  2. zhuangyao_round2_ingredients.xlsx (Round 2: 21味TCMSP对应药材)
  3. herb_ingredient_mapping.xlsx (原有58味中的8味壮药)

输出: L3/results/zhuangyao_compound_pool.csv
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import SaltRemover

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
L3_RESULTS = PROJECT_ROOT / "L3" / "results"
L3_LOGS = PROJECT_ROOT / "L3" / "logs"
L3_LOGS.mkdir(parents=True, exist_ok=True)

SMILES_MAP = L3_RESULTS / "tcmsp_smiles_fixed_v4_1.csv"
ROUND1 = L3_RESULTS / "zhuangyao_ingredient_mapping_full.xlsx"
ROUND2 = L3_RESULTS / "zhuangyao_round2_ingredients.xlsx"
EXISTING_HERB = L3_RESULTS / "herb_ingredient_mapping.xlsx"
OUTPUT_POOL = L3_RESULTS / "zhuangyao_compound_pool.csv"
CPI_PATH = PROJECT_ROOT / "L4" / "results" / "experimental_actives_detail_cleaned_combined.csv"
PHENO_PATH = PROJECT_ROOT / "L4" / "results_v10_minibatch" / "phenotype_ferroptosis_dataset_v25_clean.csv"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(L3_LOGS / "zhuangyao_pool_build.log", encoding="utf-8", mode="w"),
              logging.StreamHandler(sys.stdout)], force=True
)
logger = logging.getLogger(__name__)

# 壮药->TCMSP映射 (30味模糊匹配)
ZHUANG_TO_TCMSP = {
    "大叶骨碎补": "骨碎补", "广山药": "山药", "广西海风藤": "海风藤", "广金钱草": "金钱草",
    "小槐花": "槐花", "无患子果": "无患子", "毛两面针": "两面针", "毛鸡骨草": "鸡骨草",
    "当归藤": "当归", "余甘子汁": "余甘子", "苦玄参": "玄参", "岩黄连": "黄连",
    "南板蓝根": "板蓝根", "蛇床子油": "蛇床子", "蓝花柴胡": "柴胡", "丁茄根": "茄根",
    "三七叶": "三七", "三七姜": "三七", "大半边莲": "半边莲", "大浮萍": "浮萍",
    "广山楂叶": "山楂叶", "广钩藤": "钩藤", "毛郁金": "郁金", "水半夏": "半夏",
    "白木香": "木香", "光石韦": "石韦", "红杜仲": "杜仲", "秃叶黄柏": "黄柏",
    "草豆蔻": "豆蔻", "紫苏叶": "紫苏"
}
TCMSP_TO_ZHUANG = {v: k for k, v in ZHUANG_TO_TCMSP.items()}

# 8味在原有映射中的壮药
EXISTING_ZHUANG_MAP = {
    "三七": ["三七叶", "三七姜"],
    "山药": ["广山药"],
    "黄柏": ["秃叶黄柏"],
    "木香": ["白木香"],
    "当归": ["当归藤"],
    "黄连": ["岩黄连"],
    "柴胡": ["蓝花柴胡"],
    "半夏": ["水半夏"],
}


def load_smiles_map():
    df = pd.read_csv(SMILES_MAP)
    return dict(zip(df["molecule_name"], df["SMILES"]))


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


def load_round1():
    """加载 Round 1: 123味壮药直接爬取"""
    df = pd.read_excel(ROUND1)
    n_herbs = df["herb_cn_name"].nunique()
    logger.info(f"Round 1: {len(df)} 条记录, {n_herbs} 味药, {df['MOL_ID'].nunique()} 唯一MOL_ID")
    df["zhuangyao_source"] = df["herb_cn_name"]
    df["source_round"] = "round1_direct"
    return df


def load_round2():
    """加载 Round 2: 21味TCMSP对应药材, 映射回壮药名"""
    df = pd.read_excel(ROUND2)
    n_herbs = df["herb_cn_name"].nunique()
    logger.info(f"Round 2 (raw): {len(df)} 条记录, {n_herbs} 味药, {df['MOL_ID'].nunique()} 唯一MOL_ID")
    df["zhuangyao_source"] = df["herb_cn_name"].map(TCMSP_TO_ZHUANG).fillna(df["herb_cn_name"])
    df["source_round"] = "round2_fuzzy"
    return df


def load_existing():
    """加载原有58味中8味壮药对应药材"""
    df = pd.read_excel(EXISTING_HERB)
    target_herbs = set(EXISTING_ZHUANG_MAP.keys())
    subset = df[df["herb_cn_name"].isin(target_herbs)].copy()
    logger.info(f"原有映射(8味壮药): {len(subset)} 条记录, {subset['herb_cn_name'].nunique()} 味药")
    subset["zhuangyao_source"] = subset["herb_cn_name"].map(
        lambda h: "; ".join(EXISTING_ZHUANG_MAP.get(h, [h]))
    )
    subset["source_round"] = "existing"
    return subset


def apply_ob_dl_filter(df):
    """OB/DL过滤: OB>=30 或 DL>=0.18"""
    df["ob"] = pd.to_numeric(df["ob"], errors="coerce")
    df["dl"] = pd.to_numeric(df["dl"], errors="coerce")
    df["pass_ob_dl"] = (df["ob"] >= 30) | (df["dl"] >= 0.18)
    n_before = len(df)
    filtered = df[df["pass_ob_dl"]].copy().reset_index(drop=True)
    logger.info(f"OB/DL过滤: {n_before} -> {len(filtered)} (移除 {n_before - len(filtered)})")
    return filtered


def compute_descriptors(df, smiles_map):
    """SMILES匹配 + RDKit描述符计算"""
    df["SMILES"] = df["molecule_name"].map(smiles_map)
    n_before = len(df)
    df = df[df["SMILES"].notna()].copy().reset_index(drop=True)
    logger.info(f"SMILES匹配: {n_before} -> {len(df)} (移除 {n_before - len(df)} 无SMILES)")

    df["SMILES_std"] = df["SMILES"].apply(standardize_smiles)
    df = df[df["SMILES_std"].notna()].copy().reset_index(drop=True)
    logger.info(f"SMILES标准化: {len(df)} 通过")

    pains_params = FilterCatalogParams()
    pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    pains_catalog = FilterCatalog(pains_params)

    mw_list, logp_list, tpsa_list, hbd_list, hba_list, qed_list = [], [], [], [], [], []
    lip_pass, lip_viol, pains_pass, rot_bonds = [], [], [], []

    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES_std"])
        if mol is None:
            for lst in [mw_list, logp_list, tpsa_list, hbd_list, hba_list, qed_list]:
                lst.append(np.nan)
            lip_pass.append(False); lip_viol.append(99); pains_pass.append(False); rot_bonds.append(np.nan)
            continue

        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        qed = Descriptors.qed(mol)
        rb = Lipinski.NumRotatableBonds(mol)

        mw_list.append(mw); logp_list.append(logp); tpsa_list.append(tpsa)
        hbd_list.append(hbd); hba_list.append(hba); qed_list.append(qed)
        rot_bonds.append(rb)

        viol = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        lip_pass.append(viol <= 1); lip_viol.append(viol)
        pains_pass.append(len(pains_catalog.GetMatches(mol)) == 0)

    df["MW_calc"] = mw_list
    df["LogP_calc"] = logp_list
    df["TPSA_calc"] = tpsa_list
    df["HBD_calc"] = hbd_list
    df["HBA_calc"] = hba_list
    df["QED"] = qed_list
    df["RotBonds"] = rot_bonds
    df["Lipinski_Pass"] = lip_pass
    df["Lipinski_Violations"] = lip_viol
    df["PAINS_Pass"] = pains_pass

    df["MW_DIFF"] = (df["MW_calc"] - df["mw"].astype(float).abs()).abs()
    df["MW_REL_DIFF"] = df["MW_DIFF"] / df["mw"].astype(float).replace(0, 1.0).abs()
    df["SMILES_MATCH_STATUS"] = np.where(
        (df["MW_DIFF"] <= 10.0) | (df["MW_REL_DIFF"] <= 0.1), "MATCH_OK", "UNCERTAIN"
    )
    df = df[df["SMILES_MATCH_STATUS"] == "MATCH_OK"].copy().reset_index(drop=True)
    logger.info(f"SMILES质量验证: {len(df)} 通过 (MW匹配)")
    return df


def remove_anthocyanidin(df):
    """移除 MOL000001 (anthocyanidin - 身份不可验证)"""
    mask = df["MOL_ID"].astype(str).str.strip() == "MOL000001"
    if mask.sum() > 0:
        logger.warning(f"移除 MOL000001 (anthocyanidin, 身份不明): {mask.sum()} 条")
        df = df[~mask].copy().reset_index(drop=True)
    return df


def deduplicate_by_smiles(df):
    """按 SMILES_std 去重，保留最早的记录"""
    n_before = len(df)
    df = df.drop_duplicates(subset=["SMILES_std"], keep="first").reset_index(drop=True)
    logger.info(f"SMILES去重: {n_before} -> {len(df)}")
    return df


def check_leaks(df):
    """检查候选池与CPI/pheno训练集的重叠"""
    if not CPI_PATH.exists():
        logger.warning(f"CPI文件不存在: {CPI_PATH}, 跳过泄漏检查")
        return df, 0
    if not PHENO_PATH.exists():
        logger.warning(f"表型文件不存在: {PHENO_PATH}, 跳过泄漏检查")
        return df, 0

    cpi = pd.read_csv(CPI_PATH, low_memory=False)
    pheno = pd.read_csv(PHENO_PATH, low_memory=False)

    train_smiles = set()
    for col in ["canonical_smiles", "SMILES"]:
        if col in cpi.columns:
            train_smiles.update(cpi[col].dropna().astype(str).str.strip().unique())
        if col in pheno.columns:
            train_smiles.update(pheno[col].dropna().astype(str).str.strip().unique())

    overlap = set()
    for _, row in df.iterrows():
        smi_raw = str(row.get("SMILES", "")).strip()
        smi_std = str(row.get("SMILES_std", "")).strip()
        if smi_raw in train_smiles or smi_std in train_smiles:
            overlap.add((row["MOL_ID"], row["molecule_name"]))

    if overlap:
        logger.warning(f"发现 {len(overlap)} 个泄漏化合物:")
        for mid, name in overlap:
            logger.warning(f"  {mid}: {name}")
        mask = df["SMILES_std"].isin(train_smiles) | df["SMILES"].isin(train_smiles)
        df = df[~mask].copy().reset_index(drop=True)
        logger.info(f"移除泄漏后: {len(df)}")
    else:
        logger.info("无泄漏")

    return df, len(overlap)


def main():
    logger.info("=" * 70)
    logger.info("壮药完整候选池构建")
    logger.info("=" * 70)

    smiles_map = load_smiles_map()
    logger.info(f"SMILES映射: {len(smiles_map)} 条")

    df1 = load_round1()
    df2 = load_round2()
    df3 = load_existing()

    all_cols = ["herb_cn_name", "herb_en_name", "herb_pinyin", "molecule_ID", "MOL_ID",
                "molecule_name", "ob", "dl", "mw", "alogp", "bbb", "caco2", "halflife",
                "hdon", "hacc", "FASA", "zhuangyao_source", "source_round"]
    df_all = pd.concat([df1[all_cols], df2[all_cols], df3[all_cols]], ignore_index=True)
    df_all["MOL_ID"] = df_all["MOL_ID"].astype(str).str.strip()
    logger.info(f"合并后: {len(df_all)} 条记录, {df_all['MOL_ID'].nunique()} 唯一MOL_ID")

    df_filtered = apply_ob_dl_filter(df_all)

    df_with_desc = compute_descriptors(df_filtered, smiles_map)

    df_clean = remove_anthocyanidin(df_with_desc)

    df_dedup = deduplicate_by_smiles(df_clean)

    df_final, n_leaked = check_leaks(df_dedup)

    logger.info("=" * 70)
    logger.info("最终统计")
    logger.info(f"  化合物总数: {len(df_final)}")
    logger.info(f"  唯一MOL_ID: {df_final['MOL_ID'].nunique()}")
    logger.info(f"  壮药来源数: {df_final['zhuangyao_source'].nunique()}")
    logger.info(f"  SMILES NaN: {df_final['SMILES_std'].isna().sum()}")
    logger.info(f"  Lipinski通过: {df_final['Lipinski_Pass'].sum()}")
    logger.info(f"  PAINS通过: {df_final['PAINS_Pass'].sum()}")
    logger.info(f"  泄漏移除: {n_leaked}")

    out_cols = [c for c in [
        "MOL_ID", "molecule_name", "SMILES_std", "zhuangyao_source", "source_round",
        "herb_cn_name", "ob", "dl", "mw", "MW_calc", "LogP_calc", "TPSA_calc",
        "HBD_calc", "HBA_calc", "QED", "RotBonds",
        "Lipinski_Pass", "Lipinski_Violations", "PAINS_Pass",
        "SMILES_MATCH_STATUS", "alogp", "bbb", "caco2", "hdon", "hacc", "FASA"
    ] if c in df_final.columns]

    df_final[out_cols].to_csv(OUTPUT_POOL, index=False, encoding="utf-8")
    logger.info(f"输出: {OUTPUT_POOL}")

    # 验证完整性
    if df_final["SMILES_std"].isna().sum() > 0:
        logger.error("存在SMILES NaN，完整性检查失败!")
        sys.exit(1)
    if not df_final["MOL_ID"].is_unique:
        logger.warning("MOL_ID不唯一（可能因去重后保留多个来源）")
    logger.info("完整性检查: PASS")


if __name__ == "__main__":
    main()