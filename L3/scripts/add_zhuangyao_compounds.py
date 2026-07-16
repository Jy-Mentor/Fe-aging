"""将壮药来源的新化合物加入 TCM 候选池

流程：
1. 从 herb_ingredient_mapping 提取壮药来源化合物 (15味壮药)
2. OB/DL 过滤 (OB>=30, DL>=0.18)
3. 按 molecule_name 匹配 tcmsp_smiles_fixed_v4_1.csv 获取 SMILES
4. RDKit 计算分子描述符 (MW, LogP, TPSA, HBD, HBA, RotBonds, QED, etc.)
5. Lipinski Rule of Five + PAINS 过滤
6. 合格化合物加入候选池

数据真实性原则：所有数据从真实文件读取，不捏造任何数据。
"""
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

PROJECT_ROOT = Path(__file__).parent.parent.parent
L3_RESULTS = PROJECT_ROOT / "L3" / "results"
HERB_MAPPING_XLSX = L3_RESULTS / "herb_ingredient_mapping.xlsx"
SMILES_CSV = L3_RESULTS / "tcmsp_smiles_fixed_v4_1.csv"
POOL_CSV = L3_RESULTS / "tcm_compound_pool_tox_filtered.csv"
ZHUANGYAO_CSV = PROJECT_ROOT / "zhuangyao_data" / "guangxi_zhuangyao_list.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# PAINS 过滤器
_pains_params = FilterCatalogParams()
_pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
_pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
_pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
PAINS_CATALOG = FilterCatalog(_pains_params)


def compute_descriptors(smiles):
    """用 RDKit 计算分子描述符"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)
    qed = Descriptors.qed(mol)
    ring_count = Lipinski.RingCount(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    fraction_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    molar_refractivity = Crippen.MolMR(mol)
    sat_rings = rdMolDescriptors.CalcNumSaturatedRings(mol)
    aliphatic_rings = rdMolDescriptors.CalcNumAliphaticRings(mol)

    # Lipinski Rule of Five
    lipinski_violations = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10,
    ])
    lipinski_pass = lipinski_violations <= 1

    # PAINS 过滤
    pains_matches = 0
    entry = PAINS_CATALOG.GetFirstMatch(mol)
    if entry is not None:
        pains_matches = 1
    pains_pass = pains_matches == 0

    # BBB 渗透性预测 (简单规则: MW<450, TPSA<90, HBD<=3)
    bbb_prediction = "High" if (mw < 450 and tpsa < 90 and hbd <= 3) else "Low"

    return {
        "MW": round(mw, 2),
        "LogP": round(logp, 2),
        "TPSA": round(tpsa, 2),
        "HBD": hbd,
        "HBA": hba,
        "RotBonds": rot_bonds,
        "QED": round(qed, 4),
        "Lipinski_Pass": lipinski_pass,
        "Lipinski_Violations": lipinski_violations,
        "BBB_Prediction": bbb_prediction,
        "PAINS_Pass": pains_pass,
        "PAINS_Matches": pains_matches,
        "RingCount": ring_count,
        "AromaticRings": aromatic_rings,
        "HeavyAtoms": heavy_atoms,
        "FractionCsp3": round(fraction_csp3, 4),
        "MolarRefractivity": round(molar_refractivity, 2),
        "NumSaturatedRings": sat_rings,
        "NumAliphaticRings": aliphatic_rings,
    }


def main():
    logger.info("=" * 70)
    logger.info("壮药来源新化合物加入候选池")
    logger.info("=" * 70)

    # 1. 加载数据
    mapping_df = pd.read_excel(HERB_MAPPING_XLSX)
    smiles_df = pd.read_csv(SMILES_CSV)
    pool_df = pd.read_csv(POOL_CSV)
    zhuangyao_df = pd.read_csv(ZHUANGYAO_CSV)
    zhuangyao_df["cn_name_clean"] = zhuangyao_df["cn_name"].astype(str).str.split("（").str[0].str.strip()
    zhuangyao_names = set(zhuangyao_df["cn_name_clean"].tolist())

    logger.info(f"药材-化合物映射: {len(mapping_df)} 条, {mapping_df['herb_cn_name'].nunique()} 味药")
    logger.info(f"SMILES 映射: {len(smiles_df)} 条")
    logger.info(f"现有候选池: {len(pool_df)} 个化合物")

    # 2. 提取壮药来源化合物 (15味壮药 ∩ 已映射药材)
    overlap_herbs = set(mapping_df["herb_cn_name"].unique()) & zhuangyao_names
    logger.info(f"壮药 ∩ 已映射药材: {len(overlap_herbs)} 味 — {sorted(overlap_herbs)}")

    zy_compounds = mapping_df[mapping_df["herb_cn_name"].isin(overlap_herbs)].copy()
    logger.info(f"壮药来源化合物 (映射中): {len(zy_compounds)} 条, {zy_compounds['MOL_ID'].nunique()} 个唯一化合物")

    # 3. OB/DL 过滤
    zy_active = zy_compounds[(zy_compounds["ob"] >= 30.0) & (zy_compounds["dl"] >= 0.18)].copy()
    logger.info(f"OB/DL 过滤后: {len(zy_active)} 条, {zy_active['MOL_ID'].nunique()} 个唯一化合物")

    # 4. 排除已在候选池中的化合物
    existing_mols = set(pool_df["MOL_ID"].tolist())
    zy_new = zy_active[~zy_active["MOL_ID"].isin(existing_mols)].copy()
    logger.info(f"新化合物 (不在现有候选池中): {len(zy_new)} 条, {zy_new['MOL_ID'].nunique()} 个唯一化合物")

    if len(zy_new) == 0:
        logger.info("无新化合物可添加")
        return

    # 5. 聚合每个化合物的来源药材
    zy_new_grouped = zy_new.groupby("MOL_ID").agg({
        "molecule_name": "first",
        "ob": "first",
        "dl": "first",
        "mw": "first",
        "herb_cn_name": lambda x: "; ".join(sorted(set(x))),
    }).reset_index()

    # 6. 按 molecule_name 匹配 SMILES
    smiles_map = dict(zip(smiles_df["molecule_name"].astype(str).str.strip(), smiles_df["SMILES"].astype(str)))
    zy_new_grouped["SMILES_std"] = zy_new_grouped["molecule_name"].map(
        lambda x: smiles_map.get(str(x).strip(), "")
    )
    n_with_smiles = (zy_new_grouped["SMILES_std"] != "").sum()
    logger.info(f"SMILES 匹配: {n_with_smiles}/{len(zy_new_grouped)} ({n_with_smiles/len(zy_new_grouped)*100:.1f}%)")

    # 7. 对有 SMILES 的化合物计算分子描述符
    new_rows = []
    n_no_smiles = 0
    n_invalid_smiles = 0
    n_lipinski_fail = 0
    n_pains_fail = 0

    for _, row in zy_new_grouped.iterrows():
        mol_id = row["MOL_ID"]
        name = row["molecule_name"]
        smiles = row["SMILES_std"]
        herb_source = row["herb_cn_name"]

        if not smiles:
            n_no_smiles += 1
            continue

        # 验证 SMILES 有效性
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"  无效SMILES: {mol_id} {name} — {smiles[:50]}")
            n_invalid_smiles += 1
            continue

        # 计算描述符
        desc = compute_descriptors(smiles)
        if desc is None:
            n_invalid_smiles += 1
            continue

        # Lipinski 过滤
        if not desc["Lipinski_Pass"]:
            n_lipinski_fail += 1
            continue

        # PAINS 过滤
        if not desc["PAINS_Pass"]:
            n_pains_fail += 1
            logger.info(f"  PAINS过滤: {mol_id} {name}")
            continue

        # 构建行 (与现有候选池列对齐)
        new_row = {
            "MOL_ID": mol_id,
            "molecule_name": name,
            "SMILES_std": smiles,
            "mw": float(row["mw"]),
            "ob": float(row["ob"]),
            "dl": float(row["dl"]),
            "alogp": np.nan,
            "bbb": np.nan,
            "tpsa": np.nan,
            "caco2": np.nan,
            "hdon": np.nan,
            "hacc": np.nan,
            "rbn": np.nan,
            "MW": desc["MW"],
            "LogP": desc["LogP"],
            "TPSA": desc["TPSA"],
            "HBD": desc["HBD"],
            "HBA": desc["HBA"],
            "RotBonds": desc["RotBonds"],
            "QED": desc["QED"],
            "Lipinski_Pass": desc["Lipinski_Pass"],
            "Lipinski_Violations": desc["Lipinski_Violations"],
            "BBB_Prediction": desc["BBB_Prediction"],
            "PAINS_Pass": desc["PAINS_Pass"],
            "PAINS_Matches": desc["PAINS_Matches"],
            "RingCount": desc["RingCount"],
            "AromaticRings": desc["AromaticRings"],
            "HeavyAtoms": desc["HeavyAtoms"],
            "FractionCsp3": desc["FractionCsp3"],
            "MolarRefractivity": desc["MolarRefractivity"],
            "NumSaturatedRings": desc["NumSaturatedRings"],
            "NumAliphaticRings": desc["NumAliphaticRings"],
            "SMILES_MATCH_STATUS": "ZHUANGYAO_NEW",
            "MW_DIFF": abs(desc["MW"] - float(row["mw"])),
            "MW_REL_DIFF": abs(desc["MW"] - float(row["mw"])) / max(float(row["mw"]), 1),
            "herb_source": herb_source,
            "is_zhuangyao": True,
        }
        new_rows.append(new_row)

    logger.info(f"\n=== 新化合物添加统计 ===")
    logger.info(f"  总计: {len(zy_new_grouped)}")
    logger.info(f"  无SMILES: {n_no_smiles}")
    logger.info(f"  无效SMILES: {n_invalid_smiles}")
    logger.info(f"  Lipinski不过: {n_lipinski_fail}")
    logger.info(f"  PAINS不过: {n_pains_fail}")
    logger.info(f"  通过全部过滤: {len(new_rows)}")

    if len(new_rows) == 0:
        logger.info("无化合物通过所有过滤，候选池不变")
        return

    # 8. 合并到候选池
    new_df = pd.DataFrame(new_rows)
    expanded_pool = pd.concat([pool_df, new_df], ignore_index=True)
    logger.info(f"\n候选池扩展: {len(pool_df)} → {len(expanded_pool)} (新增 {len(new_df)})")

    # 9. 保存
    expanded_pool.to_csv(POOL_CSV, index=False)
    logger.info(f"扩展后候选池已保存: {POOL_CSV} ({len(expanded_pool)} 行)")

    # 10. 打印新增化合物列表
    logger.info(f"\n=== 新增壮药来源化合物 ({len(new_df)} 个) ===")
    for _, row in new_df.iterrows():
        logger.info(f"  {row['MOL_ID']} {row['molecule_name']} (MW={row['MW']:.1f}, OB={row['ob']:.1f}, DL={row['dl']:.3f}) <- {row['herb_source']}")

    # 11. 验证数据完整性
    logger.info(f"\n=== 数据完整性验证 ===")
    logger.info(f"  SMILES_std NaN: {expanded_pool['SMILES_std'].isna().sum()}")
    logger.info(f"  MOL_ID 唯一: {expanded_pool['MOL_ID'].nunique() == len(expanded_pool)}")
    logger.info(f"  壮药来源化合物: {expanded_pool['is_zhuangyao'].sum()}/{len(expanded_pool)} ({expanded_pool['is_zhuangyao'].sum()/len(expanded_pool)*100:.1f}%)")

    logger.info("=" * 70)
    logger.info("壮药来源新化合物添加完成!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
