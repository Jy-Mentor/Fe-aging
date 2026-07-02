import pandas as pd
from pathlib import Path
import os

SPIDER_DIR = Path('L3/TCMSP-Spider/data/spider_data')
SAMPLE_DIR = Path('L3/TCMSP-Spider/data/sample_data')

# 1. 从spider_data中的单味药成分文件构建草药-成分映射
print('=== 构建草药-成分映射 ===')
herb_ing_files = list(SPIDER_DIR.glob('*_ingredients.xlsx'))
print(f'已有的草药成分文件: {len(herb_ing_files)} 个')
for f in herb_ing_files:
    print(f'  {f.name}')

# 2. 检查TCMSP官网的502种草药中，每味药的成分在哪里
# 看起来spider_data只有4味药（白术、陈皮、麻黄根、麻黄）的详细数据
# 而ingredients_data.xlsx是所有13729个成分总表，但没有草药关联

# 3. 查看TCMSP官网browse接口，看是否有herb-ingredient关系
# 先看TCMSP-Spider的结构
import sys
sys.path.insert(0, 'L3/TCMSP-Spider/src')
print('\n=== TCMSP-Spider爬虫功能 ===')

# 检查03_Molecules_CAS_Relationships.xlsx
rel_file = SAMPLE_DIR / '03_Molecules_CAS_Relationships.xlsx'
if rel_file.exists():
    rel = pd.read_excel(rel_file)
    print(f'\nMolecules_CAS_Relationships: {len(rel)} 行')
    print(f'列名: {rel.columns.tolist()}')
    print(rel.head(5).to_string())

# 检查是否有herb-ingredient关系的其他文件
print('\n=== sample_data所有文件 ===')
for f in SAMPLE_DIR.iterdir():
    print(f'  {f.name}: {f.stat().st_size/1024:.1f} KB')
