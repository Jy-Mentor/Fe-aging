"""读取Bata-caryophyllene和caryophyllene oxide的完整靶标得分"""
import csv
from pathlib import Path

pred_path = Path(r"d:\铁衰老 绝不重蹈覆辙\L4\results_v10_minibatch\tcm_predictions_full_v70_fixed.csv")
with open(pred_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

KEY_GENES = ["NFE2L2", "HMOX1", "GPX4", "KEAP1", "TFRC", "SLC7A11", "PTGS2", "HIF1A", "ACSL4", "LPCAT3"]

for row in rows:
    name = (row.get("molecule_name", "") or "").lower()
    # 精确匹配
    if name == "bata-caryophyllene":
        print(f"=== {row.get('molecule_name','')} (MOL_ID={row.get('MOL_ID','')}) ===")
        print(f"  composite_score: {row.get('composite_score','')}")
        for g in KEY_GENES:
            print(f"  {g}: {row.get(g, 'N/A')}")
        break
else:
    print("未找到 Bata-caryophyllene")
    # 模糊搜索
    for row in rows:
        name = (row.get("molecule_name", "") or "").lower()
        if "bata-caryophyllene" in name:
            print(f"模糊匹配: {row.get('molecule_name','')} (MOL_ID={row.get('MOL_ID','')})")
            print(f"  composite_score: {row.get('composite_score','')}")