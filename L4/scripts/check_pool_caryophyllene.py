"""检查壮药池中石竹烯和艾叶化合物"""
import csv
from pathlib import Path

pool = Path(r"d:\铁衰老 绝不重蹈覆辙\L3\results\zhuangyao_herb_augmented_pool.csv")
with open(pool, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print("=== 壮药池中石竹烯相关化合物 ===")
for row in rows:
    name = (row.get("molecule_name", "") or "").lower()
    if "caryophyllene" in name:
        print(f"  MOL_ID={row.get('MOL_ID','')}, name={row.get('molecule_name','')}")

print()
print("=== 壮药池中 coumarin/quercetin/sitosterol 数量 ===")
counts = {"coumarin": 0, "quercetin": 0, "sitosterol": 0, "cycloartenol": 0, "dammaradienyl": 0}
for row in rows:
    name = (row.get("molecule_name", "") or "").lower()
    for key in counts:
        if key in name:
            counts[key] += 1
            break
for k, v in counts.items():
    print(f"  {k}: {v}")

print()
print("=== 壮药列表中的艾叶条目 ===")
herb_path = Path(r"d:\铁衰老 绝不重蹈覆辙\zhuangyao_data\guangxi_zhuangyao_list.csv")
with open(herb_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if "艾叶" in str(row.get("herb_cn_name", "")):
            print(f"  herb_cn_name={row.get('herb_cn_name','')}, idx={row.get('idx','')}, zhuang_name={row.get('zhuang_name','')}")