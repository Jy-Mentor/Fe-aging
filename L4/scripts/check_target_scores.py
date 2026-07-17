"""读取石竹烯类化合物的靶标得分"""
import csv
from pathlib import Path

# 读取原始预测（未调整的）
pred_path = Path(r"d:\铁衰老 绝不重蹈覆辙\L4\results_v10_minibatch\tcm_predictions_full_v70_fixed.csv")
with open(pred_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# 关键基因
KEY_GENES = ["NFE2L2", "HMOX1", "GPX4", "KEAP1", "TFRC", "SLC7A11", "PTGS2", "HIF1A", "ACSL4", "LPCAT3"]

targets = ["caryophyllene oxide", "Bata-caryophyllene", "beta-caryophyllene"]

for row in rows:
    name = (row.get("molecule_name", "") or "").lower()
    for t in targets:
        if t in name:
            print(f"\n=== {row.get('molecule_name','')} (MOL_ID={row.get('MOL_ID','')}) ===")
            print(f"  composite_score: {row.get('composite_score','')}")
            for g in KEY_GENES:
                val = row.get(g, "N/A")
                print(f"  {g}: {val}")
            break