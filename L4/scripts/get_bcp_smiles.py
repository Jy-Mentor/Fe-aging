"""获取Bata-caryophyllene的SMILES"""
import csv
from pathlib import Path

pred_path = Path(r"d:\铁衰老 绝不重蹈覆辙\L4\results_v10_minibatch\tcm_predictions_full_v70_fixed.csv")
with open(pred_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = (row.get("molecule_name", "") or "").lower()
        if "bata-caryophyllene" in name:
            print(f"name: {row.get('molecule_name','')}")
            print(f"MOL_ID: {row.get('MOL_ID','')}")
            print(f"SMILES: {row.get('SMILES','')}")
            print(f"composite_score: {row.get('composite_score','')}")
            break
    else:
        print("未找到Bata-caryophyllene")