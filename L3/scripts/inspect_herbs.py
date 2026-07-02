import pandas as pd

herbs = pd.read_excel('L3/TCMSP-Spider/data/sample_data/herbs_data.xlsx')
print('列名:', herbs.columns.tolist())
print('\n前10行:')
print(herbs.head(10).to_string())

print(f'\n总数: {len(herbs)}')
print(f'\n草药拼音名样本:')
print(herbs['herb_pinyin'].head(20).tolist())
