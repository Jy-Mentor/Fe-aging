import pandas as pd

# 检查ingredients数据结构
raw = pd.read_excel('L3/TCMSP-Spider/data/sample_data/ingredients_data.xlsx')
print('成分数据列名:', raw.columns.tolist())
print(f'\n行数: {len(raw)}')
print('\n前5行:')
print(raw.head(5).to_string())

# 检查是否有herb相关列
print('\n=== 是否有草药关联列? ===')
herb_cols = [c for c in raw.columns if 'herb' in c.lower() or 'cn' in c.lower() or 'pinyin' in c.lower()]
print(f'可能的草药列: {herb_cols}')

# 检查MOL_ID是否重复
print(f'\nMOL_ID 唯一值: {raw["MOL_ID"].nunique()} / {len(raw)}')
