"""
对比分析9味中药成分与主模块候选化合物的重叠情况及剔除原因
"""
import pandas as pd
from pathlib import Path
from collections import Counter

# 路径定义
base = Path("d:/铁衰老 绝不重蹈覆辙")
herbs_file = base / "L3/results/9herbs_tcmsp_ingredients.xlsx"
candidates_tox_file = base / "L3/results/tcm_compound_pool_tox_filtered.csv"
candidates_raw_file = base / "L3/results/tcm_compound_pool_filtered.csv"
exclusion_log_file = base / "L3/data/toxicity_exclusion_log_20260623_231907.csv"
top_candidates_v6_file = base / "L4/results_v6/tcm_top_candidates_v6.csv"

# 读取数据
print("=" * 70)
print("9味中药成分 vs 主模块候选化合物 对比分析")
print("=" * 70)

herbs_df = pd.read_excel(herbs_file)
candidates_tox = pd.read_csv(candidates_tox_file)
candidates_raw = pd.read_csv(candidates_raw_file)
exclusion_df = pd.read_csv(exclusion_log_file)
top_v6 = pd.read_csv(top_candidates_v6_file)

print(f"\n1. 数据规模")
print(f"   9味中药成分总数: {len(herbs_df)} (来自 TCMSP 实时爬取)")
print(f"   候选池(毒性过滤后): {len(candidates_tox)} 个化合物")
print(f"   候选池(毒性过滤前): {len(candidates_raw)} 个化合物")
print(f"   毒性剔除化合物: {len(exclusion_df)} 个")
print(f"   v6 Top候选化合物: {len(top_v6)} 个")

# 统计每味药的成分数
herb_stats = herbs_df.groupby('herb_cn_name').size().sort_values(ascending=False)
print(f"\n2. 各味药成分数量 (TCMSP)")
for herb, count in herb_stats.items():
    print(f"   {herb}: {count} 个")

# --- 匹配分析 ---
herb_mol_ids = set(herbs_df['MOL_ID'].dropna().astype(str))
cand_tox_mol_ids = set(candidates_tox['MOL_ID'].dropna().astype(str))
cand_raw_mol_ids = set(candidates_raw['MOL_ID'].dropna().astype(str))
excl_mol_ids = set(exclusion_df['MOL_ID'].dropna().astype(str))
top_v6_mol_ids = set(top_v6['MOL_ID'].dropna().astype(str))

print(f"\n3. MOL_ID 集合规模")
print(f"   9味中药唯一 MOL_ID: {len(herb_mol_ids)}")
print(f"   候选池(tox_filtered)唯一 MOL_ID: {len(cand_tox_mol_ids)}")
print(f"   候选池(raw)唯一 MOL_ID: {len(cand_raw_mol_ids)}")
print(f"   毒性剔除唯一 MOL_ID: {len(excl_mol_ids)}")
print(f"   v6 Top候选唯一 MOL_ID: {len(top_v6_mol_ids)}")

# 重叠分析
overlap_tox = herb_mol_ids & cand_tox_mol_ids
overlap_raw = herb_mol_ids & cand_raw_mol_ids
overlap_excl = herb_mol_ids & excl_mol_ids
overlap_top = herb_mol_ids & top_v6_mol_ids

print(f"\n4. 重叠分析")
print(f"   9味中药成分 ∩ 候选池(tox_filtered): {len(overlap_tox)} 个")
print(f"   9味中药成分 ∩ 候选池(raw): {len(overlap_raw)} 个")
print(f"   9味中药成分 ∩ 毒性剔除: {len(overlap_excl)} 个")
print(f"   9味中药成分 ∩ v6 Top候选: {len(overlap_top)} 个")

# 重叠率（相对于9味中药成分）
print(f"\n5. 重叠率 (相对于9味中药 {len(herb_mol_ids)} 个唯一MOL_ID)")
print(f"   进入候选池(tox_filtered)的比例: {len(overlap_tox)/len(herb_mol_ids)*100:.2f}%")
print(f"   进入候选池(raw)的比例: {len(overlap_raw)/len(herb_mol_ids)*100:.2f}%")
print(f"   被毒性剔除的比例: {len(overlap_excl)/len(herb_mol_ids)*100:.2f}%")
print(f"   进入v6 Top候选的比例: {len(overlap_top)/len(herb_mol_ids)*100:.2f}%")

# --- 逐味药分析 ---
print(f"\n6. 逐味药与候选池(tox_filtered)的重叠详情")
print("-" * 70)
for herb in herb_stats.index:
    herb_subset = herbs_df[herbs_df['herb_cn_name'] == herb]
    herb_mols = set(herb_subset['MOL_ID'].dropna().astype(str))
    
    in_tox = herb_mols & cand_tox_mol_ids
    in_excl = herb_mols & excl_mol_ids
    in_top = herb_mols & top_v6_mol_ids
    
    print(f"   {herb}:")
    print(f"      总成分: {len(herb_mols)}")
    print(f"      进入候选池(tox_filtered): {len(in_tox)} ({len(in_tox)/len(herb_mols)*100:.1f}%)")
    print(f"      被毒性剔除: {len(in_excl)} ({len(in_excl)/len(herb_mols)*100:.1f}%)")
    print(f"      进入v6 Top候选: {len(in_top)} ({len(in_top)/len(herb_mols)*100:.1f}%)")
    if in_excl:
        names = exclusion_df[exclusion_df['MOL_ID'].isin(in_excl)][['molecule_name', 'reasons']]
        for _, row in names.iterrows():
            reason_short = row['reasons'].split(' —')[0] if ' —' in row['reasons'] else row['reasons'][:60]
            print(f"         [剔除] {row['molecule_name']}: {reason_short}")

# --- 被剔除的重复化合物详细分析 ---
print(f"\n7. 被毒性剔除的重复化合物详细分析")
print("-" * 70)
if overlap_excl:
    excl_details = exclusion_df[exclusion_df['MOL_ID'].isin(overlap_excl)].copy()
    # 关联所属中药
    herb_map = {}
    for _, row in herbs_df.iterrows():
        mid = str(row['MOL_ID'])
        if mid not in herb_map:
            herb_map[mid] = []
        herb_map[mid].append(row['herb_cn_name'])
    excl_details['source_herbs'] = excl_details['MOL_ID'].apply(lambda x: ', '.join(herb_map.get(str(x), ['?'])))
    
    print(f"   共 {len(excl_details)} 个化合物:\n")
    for _, row in excl_details.iterrows():
        print(f"   MOL_ID: {row['MOL_ID']}")
        print(f"   名称: {row['molecule_name']}")
        print(f"   所属中药: {row['source_herbs']}")
        print(f"   剔除原因: {row['reasons']}")
        print(f"   警报数: {row['alert_count']}")
        print()
    
    # 剔除原因统计
    reasons = []
    for r in excl_details['reasons']:
        if 'Epoxide' in r:
            reasons.append('Epoxide (环氧化物)')
        elif 'PAH' in r or 'Polycyclic aromatic' in r:
            reasons.append('PAH (多环芳烃)')
        elif 'pyrrolizidine' in r.lower():
            reasons.append('Pyrrolizidine (吡咯里西啶生物碱)')
        elif 'Alkenylbenzene' in r or 'allylbenzene' in r:
            reasons.append('Alkenylbenzene (烯基苯)')
        elif 'Aromatic amine' in r:
            reasons.append('Aromatic amine (芳香胺)')
        elif 'alkyl halide' in r.lower():
            reasons.append('Alkyl halide (烷基卤化物)')
        elif 'Nitro' in r or 'N-Nitroso' in r:
            reasons.append('Nitro/N-Nitroso (硝基/亚硝基)')
        else:
            reasons.append('Other (其他)')
    reason_counts = Counter(reasons)
    print("   剔除原因分布:")
    for reason, count in reason_counts.most_common():
        print(f"      {reason}: {count} 个")
else:
    print("   无")

# --- v6 Top候选中的重叠 ---
print(f"\n8. v6 Top候选中与9味中药重叠的化合物")
print("-" * 70)
if overlap_top:
    top_overlap = top_v6[top_v6['MOL_ID'].isin(overlap_top)].copy()
    herb_map = {}
    for _, row in herbs_df.iterrows():
        mid = str(row['MOL_ID'])
        if mid not in herb_map:
            herb_map[mid] = []
        herb_map[mid].append(row['herb_cn_name'])
    top_overlap['source_herbs'] = top_overlap['MOL_ID'].apply(lambda x: ', '.join(herb_map.get(str(x), ['?'])))
    
    print(f"   共 {len(top_overlap)} 个化合物:\n")
    for _, row in top_overlap.iterrows():
        print(f"   MOL_ID: {row['MOL_ID']}")
        print(f"   名称: {row['molecule_name']}")
        print(f"   所属中药: {row['source_herbs']}")
        print(f"   v6综合得分: {row['composite_score']:.4f}")
        print(f"   v6排名: {row['rank']}")
        print(f"   命中靶点数: {row['n_targets']}")
        print(f"   主要靶点: {row['top_targets']}")
        print()
else:
    print("   无")

# --- 汇总输出CSV ---
print(f"\n9. 生成详细对比报告...")
report_dir = base / "L3/results"
report_dir.mkdir(exist_ok=True)

# 9味中药成分 vs 候选池 全量对比表
herbs_df['in_tox_filtered'] = herbs_df['MOL_ID'].astype(str).isin(cand_tox_mol_ids)
herbs_df['in_exclusion'] = herbs_df['MOL_ID'].astype(str).isin(excl_mol_ids)
herbs_df['in_top_v6'] = herbs_df['MOL_ID'].astype(str).isin(top_v6_mol_ids)

# 关联剔除原因
excl_reason_map = dict(zip(exclusion_df['MOL_ID'].astype(str), exclusion_df['reasons']))
herbs_df['exclusion_reason'] = herbs_df['MOL_ID'].astype(str).map(excl_reason_map)

output_file = report_dir / "9herbs_vs_candidates_comparison.csv"
herbs_df.to_csv(output_file, index=False)
print(f"   全量对比表已保存: {output_file}")
print(f"   总记录: {len(herbs_df)} 行")
print(f"   进入候选池: {herbs_df['in_tox_filtered'].sum()} 行")
print(f"   被剔除: {herbs_df['in_exclusion'].sum()} 行")
print(f"   进入v6 Top: {herbs_df['in_top_v6'].sum()} 行")

print("\n" + "=" * 70)
print("分析完成")
print("=" * 70)
