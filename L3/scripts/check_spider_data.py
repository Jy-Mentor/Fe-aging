import pandas as pd
import os

# 检查spider_data目录下是否有按草药分类的成分数据
spider_dir = 'L3/TCMSP-Spider/data/spider_data'
if os.path.exists(spider_dir):
    files = os.listdir(spider_dir)
    print(f'spider_data 中有 {len(files)} 个文件')
    print('前20个:', files[:20])
    
    # 统计有多少个ingredients文件
    ing_files = [f for f in files if 'ingredient' in f.lower()]
    print(f'\ningredients文件数: {len(ing_files)}')
    
    # 加载一个样例
    if ing_files:
        sample = pd.read_excel(os.path.join(spider_dir, ing_files[0]))
        print(f'\n样例: {ing_files[0]}')
        print(f'  列名: {sample.columns.tolist()}')
        print(f'  行数: {len(sample)}')
        print(f'  前3行:')
        print(sample.head(3).to_string())
else:
    print('spider_dir 不存在')
    
    # 看sample_data中的targets和diseases
    targets = pd.read_excel('L3/TCMSP-Spider/data/sample_data/targets_data.xlsx')
    print(f'\ntargets_data 列名: {targets.columns.tolist()}')
    print(f'行数: {len(targets)}')
    print(targets.head(3).to_string())
