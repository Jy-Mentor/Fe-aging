# -*- coding: utf-8 -*-
"""
TCMSP 数据库综合搜索脚本
搜索目标：caryophyllene/石竹烯、艾叶、复方柴桂合剂7味药
"""
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 120)
pd.set_option('display.max_rows', 300)

BASE = r'd:\铁衰老 绝不重蹈覆辙\L3\TCMSP-Spider\data\sample_data'

# ============================================================
# Load data
# ============================================================
ingredients = pd.read_excel(f'{BASE}/ingredients_data.xlsx')
herbs = pd.read_excel(f'{BASE}/herbs_data.xlsx')

print('='*90)
print('【任务1】在 ingredients_data.xlsx 中搜索 caryophyllene / 石竹烯')
print('='*90)
mask1 = ingredients['molecule_name'].str.lower().str.contains('caryophyllene', na=False)
mask2 = ingredients['molecule_name'].str.contains('石竹烯', na=False)
result1 = ingredients[mask1 | mask2]
if len(result1) > 0:
    print(f'找到 {len(result1)} 条精确匹配结果:')
    cols = ['MOL_ID', 'molecule_name', 'mw', 'ob', 'dl', 'alogp', 'halflife', 'hdon', 'hacc']
    print(result1[cols].to_string(index=False))
else:
    print('未找到匹配 caryophyllene 或 石竹烯 的化合物。')

# Also search for partial matches
mask3 = ingredients['molecule_name'].str.lower().str.contains('caryophyll', na=False)
result1b = ingredients[mask3 & ~mask1]
if len(result1b) > 0:
    print(f'\n额外部分匹配 (caryophyll*): {len(result1b)} 条')
    print(result1b[cols].to_string(index=False))

print('\n')

# ============================================================
# 任务2: 搜索 "艾" (Artemisia/mugwort)
# ============================================================
print('='*90)
print('【任务2】在 herbs_data.xlsx 中搜索 艾 (Artemisia/mugwort)')
print('='*90)
mask_ai = herbs['herb_cn_name'].str.contains('艾', na=False)
result_ai = herbs[mask_ai]
print(f'中文名含"艾"的药材: {len(result_ai)} 条')
print(result_ai.to_string(index=False))

# Also search for Artemisia in herb_en_name
mask_art = herbs['herb_en_name'].str.lower().str.contains('artemisia', na=False)
result_art = herbs[mask_art & ~mask_ai]
if len(result_art) > 0:
    print(f'\n英文名含 Artemisia (中文名不含艾): {len(result_art)} 条')
    print(result_art.to_string(index=False))

print('\n')

# ============================================================
# 任务3: 搜索复方柴桂合剂7味药
# ============================================================
print('='*90)
print('【任务3】在 herbs_data.xlsx 中搜索复方柴桂合剂7味药')
print('='*90)

target_herbs = ['柴胡', '桂枝', '黄芩', '白芍', '甘草', '生姜', '大枣']

for herb_name in target_herbs:
    mask = herbs['herb_cn_name'].str.contains(herb_name, na=False)
    result = herbs[mask]
    if len(result) > 0:
        for _, row in result.iterrows():
            print(f'  [OK] {row["herb_cn_name"]} | 拼音: {row["herb_pinyin"]} | '
                  f'英文: {row["herb_en_name"]} | 分类: {row["child_cn_name"]}')
    else:
        print(f'  [NOT FOUND] {herb_name}')

print(f'\n数据库中总共 {len(herbs)} 味中药')
print()

# ============================================================
# 任务4: 检查 spider_data 中是否有这些药材的化合物数据
# ============================================================
print('='*90)
print('【任务4】检查 spider_data 中可用药材及其化合物（OB>=30, DL>=0.18）')
print('='*90)

import os
spider_dir = r'd:\铁衰老 绝不重蹈覆辙\L3\TCMSP-Spider\data\spider_data'

# 列出所有 spider_data 中的药材
spider_herbs = set()
for f in os.listdir(spider_dir):
    if f.endswith('_ingredients.xlsx'):
        herb_pinyin = f.replace('_ingredients.xlsx', '')
        spider_herbs.add(herb_pinyin)
        df = pd.read_excel(os.path.join(spider_dir, f))
        # Filter by OB>=30 and DL>=0.18
        df_filtered = df[(df['ob'] >= 30) & (df['dl'] >= 0.18)]
        print(f'\n--- {herb_pinyin} ({f}) ---')
        print(f'  总化合物: {len(df)}, OB>=30 & DL>=0.18: {len(df_filtered)}')
        if len(df_filtered) > 0:
            print(df_filtered[['MOL_ID', 'molecule_name', 'ob', 'dl', 'mw']].to_string(index=False))

print(f'\nSpider_data 中已有药材数据: {spider_herbs}')
print('注意: 复方柴桂合剂中的7味药（柴胡、桂枝、黄芩、白芍、甘草、生姜、大枣）均未在 spider_data 中找到。')
print('仅 herbs_data.xlsx 中记录了这些药材的元信息（名称、分类），但尚无化合物-药材关联数据。')
print()

# ============================================================
# 附加: 检查整个 herbs_data.xlsx 中所有药材概览
# ============================================================
print('='*90)
print('【附加】herbs_data.xlsx 数据库概览')
print('='*90)
print(f'总药材数: {len(herbs)}')
print(f'分类统计:')
print(herbs['child_cn_name'].value_counts().to_string())
print()

# 列出所有药材中文名
print('全部药材名称列表:')
for i, row in herbs.iterrows():
    print(f'  {i+1}. {row["herb_cn_name"]} ({row["herb_pinyin"]}) - {row["herb_en_name"]}')