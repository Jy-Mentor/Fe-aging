import pandas as pd

# 检查BCP
pool = pd.read_csv('L3/results/tcm_compound_pool_filtered.csv')
bcp_pool = pool[pool['molecule_name'].str.lower().str.contains('caryophyll|石竹烯', na=False)]
print('=== 石竹烯类（化合物池中）===')
if len(bcp_pool) > 0:
    for _, r in bcp_pool.iterrows():
        print(f"  {r['molecule_name']}: MW={r['MW']:.1f}, OB={r['ob']:.1f}%, BBB={r['BBB_Prediction']}")
else:
    print('  NOT FOUND in pool')

# 原始TCMSP
raw = pd.read_excel('L3/TCMSP-Spider/data/sample_data/ingredients_data.xlsx')
bcp_raw = raw[raw['molecule_name'].str.lower().str.contains('caryophyll|石竹烯', na=False)]
print(f'\n=== 原始TCMSP中石竹烯类（{len(bcp_raw)}个）===')
for _, r in bcp_raw.iterrows():
    print(f"  {r['molecule_name']}: MW={r['mw']:.1f}, OB={r['ob']:.1f}%, DL={r['dl']:.3f}")

# 检查艾草相关
print('\n=== 草药数据概览 ===')
herbs = pd.read_excel('L3/TCMSP-Spider/data/sample_data/herbs_data.xlsx')
print(f'草药总数: {len(herbs)}')
print(f'列名: {herbs.columns.tolist()}')

# 搜索艾草、柴胡等
herb_names = ['艾草', '艾', '柴胡', '黄芩', '半夏', '生姜', '大枣', '枳实', '大黄', 
              '芍药', '白芍', '桂枝', '茯苓', '丹皮', '牡丹皮', '桃仁', '白术', '甘草']
print('\n=== 关键中药检索 ===')
for h in herb_names:
    matches = herbs[herbs.apply(lambda row: row.astype(str).str.contains(h, case=False, na=False).any(), axis=1)]
    if len(matches) > 0:
        for _, r in matches.iterrows():
            cn = r.get('cn_name', r.get('herb_name', ''))
            en = r.get('en_name', '')
            pinyin = r.get('pinyin_name', '')
            print(f"  [FOUND] {h}: cn={cn}, en={en}, pinyin={pinyin}")
    else:
        print(f"  [MISSING] {h}")
