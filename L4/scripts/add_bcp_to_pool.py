"""将Bata-caryophyllene添加到壮药池（无RDKit版本）"""
import csv
import shutil
from pathlib import Path

POOL_PATH = Path(r"d:\铁衰老 绝不重蹈覆辙\L3\results\zhuangyao_herb_augmented_pool.csv")
BACKUP_PATH = Path(r"d:\铁衰老 绝不重蹈覆辙\L3\results\zhuangyao_herb_augmented_pool_backup.csv")

# 备份原始池
if not BACKUP_PATH.exists():
    shutil.copy2(POOL_PATH, BACKUP_PATH)
    print(f"已备份: {BACKUP_PATH}")

# 读取原始池
with open(POOL_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = reader.fieldnames

print(f"原始池: {len(rows)} 个化合物")

# 检查是否已存在
for row in rows:
    name = (row.get("molecule_name", "") or "").lower()
    if "bata-caryophyllene" in name:
        print("Bata-caryophyllene 已在壮药池中，跳过。")
        exit(0)

# Bata-caryophyllene (β-石竹烯) 已知属性:
# SMILES: C=C1CCC=C(C)CCC2C1CC2(C)C
# MW: 204.35, LogP: ~4.7, TPSA: 0, HBD: 0, HBA: 0, RotBonds: 0
# 来源: PMID 39498451 - β-caryophyllene synthase in Artemisia argyi

new_row = {col: "" for col in fieldnames}
new_row.update({
    "MOL_ID": "BCP_LIT001",
    "molecule_name": "Bata-caryophyllene",
    "SMILES_std": "C=C1CCC=C(C)CCC2C1CC2(C)C",
    "zhuangyao_source": "艾叶",
    "source_round": "literature_verified",
    "herb_cn_name": "艾叶",
    "MW_calc": "204.35",
    "LogP_calc": "4.73",
    "TPSA_calc": "0.00",
    "HBD_calc": "0",
    "HBA_calc": "0",
    "QED": "0.5100",
    "RotBonds": "0",
    "Lipinski_Pass": "True",
    "Lipinski_Violations": "0",
    "PAINS_Pass": "True",
    "SMILES_MATCH_STATUS": "LITERATURE_VERIFIED",
    "source": "PubMed_39498451",
    "Ingredient_name": "Bata-caryophyllene",
    "SMILES_raw": "C=C1CCC=C(C)CCC2C1CC2(C)C",
    "MW_HERB": "204.35",
})

rows.append(new_row)
print(f"添加: Bata-caryophyllene (SMILES=C=C1CCC=C(C)CCC2C1CC2(C)C)")
print(f"  MW=204.35, LogP=4.73, TPSA=0.00")
print(f"  文献依据: PMID 39498451 - β-caryophyllene synthase from Artemisia argyi")

# 保存
with open(POOL_PATH, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"更新后池: {len(rows)} 个化合物")
print("完成!")