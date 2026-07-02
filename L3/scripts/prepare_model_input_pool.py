"""
从综合评分版化合物池中提取A级以上化合物，生成模型输入格式文件

输出：L3/results/tcm_compound_pool_v21_Alevel.csv
- 包含所有模型需要的字段
- A+和A级化合物
"""
import pandas as pd
import numpy as np
from pathlib import Path

L3 = Path('L3')
RESULTS = L3 / 'results'

# 读取综合评分版
comprehensive = pd.read_csv(RESULTS / 'tcm_compound_pool_comprehensive.csv')
print(f'综合评分版总数: {len(comprehensive)}')

# 提取A级以上（A+ 和 A）
a_level = comprehensive[comprehensive['tier'].str.startswith('A')].copy()
print(f'A级以上: {len(a_level)} (A+: {(a_level["tier"] == "A+（高优先级）").sum()}, A: {(a_level["tier"] == "A（推荐）").sum()})')

# 有中药来源的数量
with_herb = (a_level['n_herbs'] > 0).sum()
print(f'有中药来源: {with_herb}/{len(a_level)} ({with_herb/len(a_level)*100:.1f}%)')
print(f'白名单化合物: {a_level["is_whitelist"].sum()}')

# 检查字段
print(f'\n字段列表:')
print(list(a_level.columns))

# 保存为模型可用的版本
out_path = RESULTS / 'tcm_compound_pool_v21_Alevel.csv'
a_level.to_csv(out_path, index=False, float_format='%.4f')
print(f'\n保存到: {out_path}')
print(f'文件大小: {out_path.stat().st_size:,} bytes')

# 快速验证：复方覆盖
herb_map_df = pd.read_excel(RESULTS / 'herb_ingredient_mapping.xlsx')
herb_map = {}
for _, row in herb_map_df.iterrows():
    mol_id = str(row.get('MOL_ID', '')).strip()
    herb = str(row.get('herb_cn_name', '')).strip()
    if not mol_id or not herb or pd.isna(row.get('MOL_ID')):
        continue
    if mol_id not in herb_map:
        herb_map[mol_id] = []
    if herb not in herb_map[mol_id]:
        herb_map[mol_id].append(herb)

formulas = {
    '大柴胡汤': ['柴胡', '黄芩', '半夏', '生姜', '大枣', '枳实', '大黄', '白芍'],
    '桂枝茯苓丸': ['桂枝', '茯苓', '牡丹皮', '桃仁', '白芍'],
}

print(f'\n复方药味覆盖（A级以上）:')
for formula_name, herbs in formulas.items():
    print(f'\n  [{formula_name}]')
    for herb in herbs:
        herb_mol_ids = set(herb_map_df[herb_map_df['herb_cn_name'] == herb]['MOL_ID'])
        in_pool = a_level[a_level['MOL_ID'].isin(herb_mol_ids)]
        print(f'    {herb}: {len(in_pool)} 个')
        if len(in_pool) > 0:
            top3 = in_pool.nlargest(3, 'comprehensive_score')
            for _, r in top3.iterrows():
                wl = '★' if r['is_whitelist'] else ' '
                print(f'      {wl} {r["molecule_name"][:30]:30s} {r["comprehensive_score"]:.1f}分')

print('\n✅ 完成！')
