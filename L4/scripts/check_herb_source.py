"""检查HERB源数据(Tab分隔格式)中beta-caryophyllene"""
import csv
import os

HERB_PATH = r"D:\下载\HERB_ingredient_info_v2.txt"

print("=" * 60)
print("1. HERB源数据(TSV格式)")
print("=" * 60)

with open(HERB_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    cols = reader.fieldnames
    print(f"  列名: {cols}")
    
    all_rows = list(reader)
    print(f"  总行数: {len(all_rows)}")

# 2. beta-caryophyllene in HERB
print("\n" + "=" * 60)
print("2. HERB中beta-caryophyllene相关条目")
print("=" * 60)

beta = []
for r in all_rows:
    name = r.get('Ingredient_name', '')
    if name and ('caryophyllene' in name.lower() or '石竹烯' in name):
        beta.append(r)

print(f"  名称匹配找到 {len(beta)} 条:")

# Also search by known SMILES
known_smiles = 'C=C1CCC=C(C)CCC2C1CC2(C)C'
for r in all_rows:
    smi = r.get('Canonical_smiles', '').strip()
    if smi == known_smiles and r not in beta:
        beta.append(r)

print(f"  名称+SMILES匹配共 {len(beta)} 条:")
for rec in beta:
    print(f"    Ingredient_name={rec.get('Ingredient_name','?')}")
    print(f"    Ingredient_id={rec.get('Ingredient_id','?')}")
    print(f"    Canonical_smiles={rec.get('Canonical_smiles','?')[:80]}")
    print(f"    Cas_id={rec.get('CAS_id','?')}")
    print(f"    PubChem_id={rec.get('PubChem_id','?')}")
    print(f"    TCMSP_id={rec.get('TCMSP_id','?')}")
    print(f"    Drug_likeness={rec.get('Drug_likeness','?')}")
    print(f"    OB_score={rec.get('OB_score','?')}")
    print(f"    Molecular_formula={rec.get('Molecular_formula','?')}")
    print("    ---")

# 3. 检查空名比例
print("\n" + "=" * 60)
print("3. 缺失数据统计")
print("=" * 60)
empty_name = sum(1 for r in all_rows if not r.get('Ingredient_name', '').strip())
empty_smiles = sum(1 for r in all_rows if not r.get('Canonical_smiles', '').strip())
empty_tcmsp = sum(1 for r in all_rows if not r.get('TCMSP_id', '').strip())
print(f"  Ingredient_name为空: {empty_name}/{len(all_rows)} ({100*empty_name/len(all_rows):.1f}%)")
print(f"  Canonical_smiles为空: {empty_smiles}/{len(all_rows)} ({100*empty_smiles/len(all_rows):.1f}%)")
print(f"  TCMSP_id为空: {empty_tcmsp}/{len(all_rows)} ({100*empty_tcmsp/len(all_rows):.1f}%)")

print("完成")
