import pandas as pd
from pathlib import Path

file_path = Path("d:/铁衰老 绝不重蹈覆辙/L3/results/9herbs_tcmsp_ingredients.xlsx")
df = pd.read_excel(file_path)

print("=" * 60)
print("TCMSP 9味中药单体成分提取结果验证")
print("=" * 60)
print(f"\n文件路径: {file_path}")
print(f"总记录数: {len(df)}")
print(f"列名: {df.columns.tolist()}")
print(f"\n前5条记录:")
print(df.head().to_string(index=False))

print("\n" + "=" * 60)
print("各味药成分数量统计:")
print("=" * 60)
stats = df.groupby('herb_cn_name').size().sort_values(ascending=False)
for herb, count in stats.items():
    print(f"  {herb}: {count} 个")

print("\n" + "=" * 60)
print("共有成分分析（按MOL_ID）:")
print("=" * 60)
# 统计跨药味出现的成分
mol_counts = df.groupby('MOL_ID')['herb_cn_name'].nunique().sort_values(ascending=False)
shared = mol_counts[mol_counts > 1]
print(f"  跨药味出现的成分数: {len(shared)} 个")
print(f"  仅出现在1味药中的成分数: {len(mol_counts[mol_counts == 1])} 个")
if len(shared) > 0:
    print(f"\n  出现频率最高的前10个共有成分:")
    for mol_id, herb_count in shared.head(10).items():
        names = df[df['MOL_ID'] == mol_id]['molecule_name'].unique()
        herbs = df[df['MOL_ID'] == mol_id]['herb_cn_name'].unique()
        print(f"    {mol_id} | {names[0]} | 出现在 {herb_count} 味药中: {', '.join(herbs)}")
