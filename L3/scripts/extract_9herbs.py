import pandas as pd
from pathlib import Path

# 读取herbs_data，查找9味药
herbs_df = pd.read_excel(Path("d:/铁衰老 绝不重蹈覆辙/L3/TCMSP-Spider/data/sample_data") / "herbs_data.xlsx")

target_herbs = ["柴胡", "桂枝", "黄芩", "人参", "甘草", "半夏", "白芍", "大枣", "生姜"]

print("=== 在herbs_data中查找9味药 ===")
for herb in target_herbs:
    match = herbs_df[herbs_df['herb_cn_name'] == herb]
    if not match.empty:
        print(f"{herb}: {match.iloc[0].to_dict()}")
    else:
        # 尝试模糊匹配
        fuzzy = herbs_df[herbs_df['herb_cn_name'].str.contains(herb, na=False)]
        if not fuzzy.empty:
            print(f"{herb}: 模糊匹配 -> {fuzzy.iloc[0].to_dict()}")
        else:
            print(f"{herb}: 未找到")

print("\n=== herbs_data中所有包含'参'的药 ===")
print(herbs_df[herbs_df['herb_cn_name'].str.contains('参', na=False)]['herb_cn_name'].tolist())

print("\n=== herbs_data中所有包含'柴'的药 ===")
print(herbs_df[herbs_df['herb_cn_name'].str.contains('柴', na=False)]['herb_cn_name'].tolist())
