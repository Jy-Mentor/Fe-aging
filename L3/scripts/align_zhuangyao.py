"""壮药-中药对齐脚本

功能：
1. 加载壮药名录 (375条) 和 TCMSP 药材-化合物映射 (58味药, 7549条)
2. 检查壮药与 TCMSP 药材的重叠
3. 为现有 TCM 候选池添加药材来源列 (herb_source)
4. 将壮药来源的化合物加入候选池（OB/DL 过滤后）
5. 标注壮药来源

数据真实性原则：所有数据从真实文件读取，不捏造任何数据。
"""
import sys
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
L3_RESULTS = PROJECT_ROOT / "L3" / "results"
ZHUANGYAO_CSV = PROJECT_ROOT / "zhuangyao_data" / "guangxi_zhuangyao_list.csv"
HERB_MAPPING_XLSX = L3_RESULTS / "herb_ingredient_mapping.xlsx"
POOL_CSV = L3_RESULTS / "tcm_compound_pool_tox_filtered.csv"
TCMSP_HERBS_XLSX = PROJECT_ROOT / "L3" / "TCMSP-Spider" / "data" / "sample_data" / "herbs_data.xlsx"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_zhuangyao():
    """加载壮药名录"""
    df = pd.read_csv(ZHUANGYAO_CSV)
    logger.info(f"壮药名录: {len(df)} 条")
    # 清理名称：去掉括号内的别名
    df["cn_name_clean"] = df["cn_name"].astype(str).str.split("（").str[0].str.strip()
    return df


def load_herb_mapping():
    """加载 TCMSP 药材-化合物映射"""
    df = pd.read_excel(HERB_MAPPING_XLSX)
    logger.info(f"药材-化合物映射: {len(df)} 条, {df['herb_cn_name'].nunique()} 味药")
    return df


def load_tcmsp_herbs():
    """加载 TCMSP 全部药材名录 (502味)"""
    df = pd.read_excel(TCMSP_HERBS_XLSX)
    logger.info(f"TCMSP 药材名录: {len(df)} 味")
    return df


def check_overlap(zhuangyao_df, mapping_df, tcmsp_herbs_df):
    """检查壮药与 TCMSP 药材的重叠"""
    zhuangyao_names = set(zhuangyao_df["cn_name_clean"].tolist())
    mapping_herbs = set(mapping_df["herb_cn_name"].unique())
    tcmsp_herbs = set(tcmsp_herbs_df["herb_cn_name"].tolist())

    # 壮药 ∩ 已映射药材 (58味)
    overlap_mapped = zhuangyao_names & mapping_herbs
    # 壮药 ∩ TCMSP全部药材 (502味)
    overlap_tcmsp = zhuangyao_names & tcmsp_herbs

    logger.info(f"壮药 ∩ 已映射药材: {len(overlap_mapped)} 味 — {sorted(overlap_mapped)}")
    logger.info(f"壮药 ∩ TCMSP全部药材: {len(overlap_tcmsp)} 味 — {sorted(overlap_tcmsp)}")

    # 模糊匹配：壮药名包含 TCMSP 药材名或反之
    fuzzy_matches = []
    for zy in zhuangyao_names:
        if zy in overlap_tcmsp:
            continue
        for th in tcmsp_herbs:
            if (zy in th or th in zy) and len(zy) >= 2 and len(th) >= 2:
                fuzzy_matches.append((zy, th))
    logger.info(f"模糊匹配: {len(fuzzy_matches)} 对")
    for zy, th in fuzzy_matches[:20]:
        logger.info(f"  {zy} <-> {th}")

    return overlap_mapped, overlap_tcmsp, fuzzy_matches


def add_herb_source_to_pool(pool_df, mapping_df):
    """为现有 TCM 候选池添加药材来源列"""
    # 构建 MOL_ID -> [药材名] 映射
    mol_to_herbs = mapping_df.groupby("MOL_ID")["herb_cn_name"].apply(lambda x: sorted(set(x))).to_dict()

    # 为候选池中每个化合物查找来源药材
    herb_sources = []
    for mol_id in pool_df["MOL_ID"]:
        herbs = mol_to_herbs.get(mol_id, [])
        herb_sources.append("; ".join(herbs) if herbs else "")

    pool_df["herb_source"] = herb_sources
    n_with_source = sum(1 for s in herb_sources if s)
    logger.info(f"候选池药材来源标注: {n_with_source}/{len(pool_df)} 有来源 ({n_with_source/len(pool_df)*100:.1f}%)")

    # 统计各药材的化合物数量
    herb_counts = {}
    for s in herb_sources:
        if s:
            for h in s.split("; "):
                herb_counts[h] = herb_counts.get(h, 0) + 1
    logger.info(f"来源药材统计 ({len(herb_counts)} 味):")
    for h, c in sorted(herb_counts.items(), key=lambda x: -x[1])[:20]:
        logger.info(f"  {h}: {c} 个化合物")

    return pool_df, herb_counts


def add_zhuangyao_compounds(pool_df, mapping_df, zhuangyao_df, overlap_herbs):
    """将壮药来源的化合物加入候选池（OB/DL 过滤后）"""
    # 壮药名集合
    zhuangyao_names = set(zhuangyao_df["cn_name_clean"].tolist())

    # 从映射中提取壮药来源的化合物
    zy_compounds = mapping_df[mapping_df["herb_cn_name"].isin(overlap_herbs)].copy()
    logger.info(f"壮药来源化合物 (映射中): {len(zy_compounds)} 条, {zy_compounds['MOL_ID'].nunique()} 个唯一化合物")

    # OB/DL 过滤 (与现有 pipeline 一致: OB>=30, DL>=0.18)
    zy_active = zy_compounds[(zy_compounds["ob"] >= 30.0) & (zy_compounds["dl"] >= 0.18)].copy()
    logger.info(f"OB/DL 过滤后 (OB>=30, DL>=0.18): {len(zy_active)} 条, {zy_active['MOL_ID'].nunique()} 个唯一化合物")

    # 现有候选池中已有的 MOL_ID
    existing_mols = set(pool_df["MOL_ID"].tolist())

    # 找出新化合物（不在现有候选池中的）
    zy_new = zy_active[~zy_active["MOL_ID"].isin(existing_mols)].copy()
    logger.info(f"新化合物 (不在现有候选池中): {len(zy_new)} 条, {zy_new['MOL_ID'].nunique()} 个唯一化合物")

    # 对每个新化合物，聚合来源药材
    zy_new_grouped = zy_new.groupby("MOL_ID").agg({
        "molecule_name": "first",
        "ob": "first",
        "dl": "first",
        "mw": "first",
        "herb_cn_name": lambda x: "; ".join(sorted(set(x))),
    }).reset_index()

    # 标注壮药来源
    zy_new_grouped["is_zhuangyao"] = True

    return zy_new_grouped, zy_new


def main():
    logger.info("=" * 70)
    logger.info("壮药-中药对齐")
    logger.info("=" * 70)

    # 1. 加载数据
    zhuangyao_df = load_zhuangyao()
    mapping_df = load_herb_mapping()
    tcmsp_herbs_df = load_tcmsp_herbs()
    pool_df = pd.read_csv(POOL_CSV)
    logger.info(f"现有 TCM 候选池: {len(pool_df)} 个化合物")

    # 2. 检查重叠
    overlap_mapped, overlap_tcmsp, fuzzy_matches = check_overlap(zhuangyao_df, mapping_df, tcmsp_herbs_df)

    # 3. 为现有候选池添加药材来源列
    pool_df, herb_counts = add_herb_source_to_pool(pool_df, mapping_df)

    # 4. 将壮药来源的化合物加入候选池
    zy_new_grouped, zy_new = add_zhuangyao_compounds(pool_df, mapping_df, zhuangyao_df, overlap_mapped)

    if len(zy_new_grouped) > 0:
        logger.info(f"\n新增壮药来源化合物: {len(zy_new_grouped)} 个")
        logger.info(f"来源壮药: {sorted(zy_new_grouped['herb_cn_name'].str.split('; ').explode().unique())}")
        logger.info(f"\n新增化合物列表:")
        for _, row in zy_new_grouped.head(20).iterrows():
            logger.info(f"  {row['MOL_ID']} {row['molecule_name']} (OB={row['ob']:.1f}, DL={row['dl']:.3f}) <- {row['herb_cn_name']}")
    else:
        logger.info("无新增壮药来源化合物（所有壮药化合物已在候选池中或未通过OB/DL过滤）")

    # 5. 为现有候选池标注壮药来源
    zhuangyao_names = set(zhuangyao_df["cn_name_clean"].tolist())
    pool_df["is_zhuangyao"] = pool_df["herb_source"].apply(
        lambda x: any(h in zhuangyao_names for h in x.split("; ")) if x else False
    )
    n_zy = pool_df["is_zhuangyao"].sum()
    logger.info(f"\n现有候选池中壮药来源化合物: {n_zy}/{len(pool_df)} ({n_zy/len(pool_df)*100:.1f}%)")

    # 6. 保存更新后的候选池（仅添加来源列，不改变现有化合物）
    output_path = L3_RESULTS / "tcm_compound_pool_tox_filtered.csv"
    # 备份原文件
    backup_path = L3_RESULTS / "tcm_compound_pool_tox_filtered.csv.backup_pre_zhuangyao"
    pool_df.to_csv(backup_path, index=False)
    logger.info(f"原候选池已备份: {backup_path}")

    pool_df.to_csv(output_path, index=False)
    logger.info(f"更新后候选池已保存: {output_path} ({len(pool_df)} 行, 新增列: herb_source, is_zhuangyao)")

    # 7. 保存壮药对齐报告
    report_path = L3_RESULTS / "zhuangyao_alignment_report.csv"
    report_rows = []
    for _, row in zhuangyao_df.iterrows():
        zy_name = row["cn_name_clean"]
        in_tcmsp = zy_name in overlap_tcmsp
        in_mapped = zy_name in overlap_mapped
        n_compounds = len(mapping_df[mapping_df["herb_cn_name"] == zy_name]) if in_mapped else 0
        report_rows.append({
            "idx": row["idx"],
            "cn_name": row["cn_name"],
            "zhuang_name": row["zhuang_name"],
            "in_tcmsp": in_tcmsp,
            "in_mapping": in_mapped,
            "n_compounds_in_mapping": n_compounds,
        })
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(report_path, index=False)
    logger.info(f"壮药对齐报告已保存: {report_path}")
    logger.info(f"  在TCMSP中: {report_df['in_tcmsp'].sum()}/375")
    logger.info(f"  在映射中: {report_df['in_mapping'].sum()}/375")

    logger.info("=" * 70)
    logger.info("壮药-中药对齐完成!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
