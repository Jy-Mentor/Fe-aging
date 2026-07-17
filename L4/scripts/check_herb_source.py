"""检查壮药池中caryophyllene oxide的来源和艾叶关联"""
import csv
from pathlib import Path

pool = Path(r"d:\铁衰老 绝不重蹈覆辙\L3\results\zhuangyao_herb_augmented_pool.csv")
with open(pool, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames
    print(f"壮药池列名: {cols}")

    found_caryo = False
    aiye_count = 0
    for row in reader:
        name = (row.get("molecule_name", "") or "").lower()
        if "caryophyllene" in name and not found_caryo:
            found_caryo = True
            print(f"\ncaryophyllene oxide:")
            for k, v in row.items():
                if v and str(v).strip():
                    print(f"  {k}: {v}")

        herb = str(row.get("herb_cn_name", "") or "")
        if "艾叶" in herb or "艾" in herb:
            aiye_count += 1
            if aiye_count <= 5:
                print(f"\n艾叶关联: {row.get('molecule_name','')} (MOL_ID={row.get('MOL_ID','')}, herb_cn_name={herb})")

    print(f"\n总艾叶关联化合物: {aiye_count}")

# 搜索Bata-caryophyllene是否在壮药池中
print("\n=== 搜索Bata-caryophyllene ===")
with open(pool, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = (row.get("molecule_name", "") or "").lower()
        if "bata-caryophyllene" in name:
            print(f"找到: {row.get('molecule_name','')} (MOL_ID={row.get('MOL_ID','')})")
            break
    else:
        print("未在壮药池中找到Bata-caryophyllene")