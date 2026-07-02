import pandas as pd
from pathlib import Path

L3 = Path('L3/results')
herb_map = pd.read_excel(L3 / 'herb_ingredient_mapping.xlsx')
pool = pd.read_csv(L3 / 'tcm_compound_pool_filtered.csv')

print('=== 艾叶中的石竹烯类 ===')
aiye = herb_map[herb_map['herb_cn_name'] == '艾叶']
bcp_aiye = aiye[aiye['molecule_name'].str.lower().str.contains('caryophyll|石竹烯', na=False)]
print(f'艾叶成分总数: {len(aiye)}')
print(f'艾叶中石竹烯类: {len(bcp_aiye)}')
for _, r in bcp_aiye.iterrows():
    print(f"  {r['molecule_name']}: MW={r['mw']:.1f}, OB={r['ob']:.1f}%, DL={r['dl']:.3f}")

# 检查BCP在原始TCMSP总表中是否通过OB/DL
print('\n=== beta-caryophyllene 详情 ===')
raw = pd.read_excel('L3/TCMSP-Spider/data/sample_data/ingredients_data.xlsx')
bcp_raw = raw[raw['molecule_name'].str.lower() == 'beta-caryophyllene']
if len(bcp_raw) > 0:
    r = bcp_raw.iloc[0]
    print(f"  MOL_ID={r['MOL_ID']}, MW={r['mw']:.1f}, OB={r['ob']:.1f}%, DL={r['dl']:.3f}")
    print(f"  OB>=30? {r['ob'] >= 30}, DL>=0.18? {r['dl'] >= 0.18}")
else:
    print('  未找到精确匹配')

# 复方覆盖情况
print('\n=== 大柴胡汤 + 桂枝茯苓丸 覆盖 ===')
formula_herbs = {
    '大柴胡汤': ['柴胡', '黄芩', '半夏', '生姜', '大枣', '枳实', '大黄', '白芍'],
    '桂枝茯苓丸': ['桂枝', '茯苓', '牡丹皮', '桃仁', '白芍'],
}
all_formula_herbs = set()
for v in formula_herbs.values():
    all_formula_herbs.update(v)

formula_rows = herb_map[herb_map['herb_cn_name'].isin(all_formula_herbs)]
print(f'两复方药味总数: {len(all_formula_herbs)}')
print(f'覆盖药味: {sorted(all_formula_herbs & set(herb_map["herb_cn_name"].unique()))}')
print(f'未覆盖: {sorted(all_formula_herbs - set(herb_map["herb_cn_name"].unique()))}')

# 复方中通过OB/DL的成分
formula_obdl = formula_rows[(formula_rows['ob'] >= 30) & (formula_rows['dl'] >= 0.18)]
print(f'\n复方中通过OB/DL的成分(MOL_ID唯一): {formula_obdl["MOL_ID"].nunique()}')

# 在化合物池中的有多少
pool_mol_ids = set(pool['MOL_ID'])
formula_in_pool = formula_obdl[formula_obdl['MOL_ID'].isin(pool_mol_ids)]
print(f'复方中同时在化合物池中的(MOL_ID唯一): {formula_in_pool["MOL_ID"].nunique()}')

# 每味药的核心单体（OB最高的前5个）
print('\n=== 各药味 Top5 高OB成分 ===')
for herb in sorted(all_formula_herbs):
    if herb not in herb_map['herb_cn_name'].values:
        print(f'\n  [缺失] {herb}')
        continue
    h = herb_map[herb_map['herb_cn_name'] == herb]
    h_obdl = h[(h['ob'] >= 30) & (h['dl'] >= 0.18)].nlargest(5, 'ob')
    print(f'\n  {herb} (总成分{len(h)}, OB/DL通过{len(h[ (h["ob"]>=30) & (h["dl"]>=0.18) ])}):')
    for _, r in h_obdl.iterrows():
        in_pool = '✓' if r['MOL_ID'] in pool_mol_ids else '✗'
        print(f"    {r['molecule_name']}: OB={r['ob']:.1f}%, DL={r['dl']:.3f} [池内:{in_pool}]")
