"""检查壮药列表中的艾叶条目"""
import csv
from pathlib import Path

herb_path = Path(r"d:\铁衰老 绝不重蹈覆辙\zhuangyao_data\guangxi_zhuangyao_list.csv")
with open(herb_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"壮药列表总数: {len(rows)}")
print(f"列名: {list(rows[0].keys()) if rows else 'N/A'}")
print()

# 搜索"艾"字
for row in rows:
    all_text = " ".join(str(v) for v in row.values())
    if "艾" in all_text:
        print(f"  找到: {dict(row)}")

# 也搜索 Artemisia
print()
print("=== 搜索 Artemisia ===")
for row in rows:
    all_text = " ".join(str(v) for v in row.values())
    if "artemisia" in all_text.lower():
        print(f"  找到: {dict(row)}")

# 搜索 argyi
print()
print("=== 搜索 argyi ===")
for row in rows:
    all_text = " ".join(str(v) for v in row.values())
    if "argyi" in all_text.lower():
        print(f"  找到: {dict(row)}")