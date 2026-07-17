"""深度审查v70预测数据完整性"""
import csv
import os

RESULTS_DIR = r"d:\铁衰老 绝不重蹈覆辙\L4\results_v10_minibatch"
POOL_PATH = r"d:\铁衰老 绝不重蹈覆辙\L3\results\zhuangyao_herb_augmented_pool.csv"

# ====== 1. 检查候选池中beta-caryophyllene的所有条目 ======
print("=" * 60)
print("1. 候选池全部beta-caryophyllene相关条目")
print("=" * 60)

with open(POOL_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    pool_cols = reader.fieldnames
    print(f"  候选池列名: {pool_cols}")
    
    all_beta = []
    all_rows = []
    for row in reader:
        all_rows.append(row)
        name = row.get('molecule_name', row.get('Ingredient_name', ''))
        if '石竹烯' in name or 'caryophyllene' in name.lower() or 'bata' in name.lower():
            all_beta.append(row)
    
    print(f"\n  找到 {len(all_beta)} 条相关条目:")
    for rec in all_beta:
        name = rec.get('molecule_name', rec.get('Ingredient_name', '?'))
        smiles = rec.get('SMILES_std', rec.get('SMILES', '?'))
        source = rec.get('zhuangyao_source', '?')
        herb = rec.get('herb_cn_name', '?')
        mol_id = rec.get('MOL_ID', '?')
        print(f"    - {name} | MOL_ID={mol_id} | source={source} | herb={herb}")
        print(f"      SMILES={smiles[:80]}...")
    
    # Check for Bata-caryophyllene specifically
    print("\n  Bata-caryophyllene 精确搜索:")
    for rec in all_rows:
        name = rec.get('molecule_name', rec.get('Ingredient_name', ''))
        if 'bata' in name.lower():
            print(f"    - {name} | SMILES={rec.get('SMILES_std','?')[:60]}")
    
    print(f"\n  候选池总数: {len(all_rows)}")

# ====== 2. 检查预测文件中molecule_name空白问题 ======
print("\n" + "=" * 60)
print("2. 预测文件molecule_name空白统计")
print("=" * 60)

full_path = os.path.join(RESULTS_DIR, "tcm_predictions_full_v70.csv")
with open(full_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    pred_rows = list(reader)
    
    empty_name = 0
    empty_mol = 0
    empty_smiles = 0
    
    for row in pred_rows:
        if not row.get('molecule_name', '').strip():
            empty_name += 1
        if not row.get('MOL_ID', '').strip():
            empty_mol += 1
        if not row.get('SMILES', '').strip():
            empty_smiles += 1
    
    print(f"  总行数: {len(pred_rows)}")
    print(f"  molecule_name为空: {empty_name} ({100*empty_name/len(pred_rows):.1f}%)")
    print(f"  MOL_ID为空: {empty_mol} ({100*empty_mol/len(pred_rows):.1f}%)")
    print(f"  SMILES为空: {empty_smiles} ({100*empty_smiles/len(pred_rows):.1f}%)")
    
    # 显示一些有空名称的样本
    print("\n  有空名称的样本 (前10条):")
    count = 0
    for row in pred_rows:
        if not row.get('molecule_name', '').strip() and count < 10:
            print(f"    MOL_ID={row.get('MOL_ID','?')} | molecule_name='{row.get('molecule_name','')}' | SMILES={row.get('SMILES','?')[:60]}... | composite_score={row.get('composite_score','?')}")
            count += 1
    
    # 显示一些有名称的样本
    print("\n  有名称的样本 (前10条):")
    count = 0
    for row in pred_rows:
        if row.get('molecule_name', '').strip() and count < 10:
            print(f"    MOL_ID={row.get('MOL_ID','?')} | molecule_name='{row.get('molecule_name','')}' | composite_score={row.get('composite_score','?')}")
            count += 1

# ====== 3. 候选池空白统计 ======
print("\n" + "=" * 60)
print("3. 候选池molecule_name空白统计")
print("=" * 60)

pool_empty = sum(1 for r in all_rows if not r.get('molecule_name', r.get('Ingredient_name', '')).strip())
print(f"  候选池总数: {len(all_rows)}")
print(f"  molecule_name为空: {pool_empty} ({100*pool_empty/len(all_rows):.1f}%)")

if pool_empty > 0:
    print("\n  候选池空名称样本 (前5条):")
    count = 0
    for r in all_rows:
        name = r.get('molecule_name', r.get('Ingredient_name', ''))
        if not name.strip() and count < 5:
            mol_id = r.get('MOL_ID', '?')
            smiles = r.get('SMILES_std', r.get('SMILES', '?'))
            source = r.get('zhuangyao_source', '?')
            print(f"    MOL_ID={mol_id} | name='{name}' | source={source}")
            print(f"    SMILES={smiles[:80]}...")
            count += 1

# ====== 4. 检查prediction文件中beta-caryophyllene的所有SMILES ======
print("\n" + "=" * 60)
print("4. 预测文件中caryophyllene相关条目(按SMILES匹配)")
print("=" * 60)

# beta-caryophyllene known SMILES: C=C1CCC=C(C)CCC2C1CC2(C)C
known_smiles = [
    ('C=C1CCC=C(C)CCC2C1CC2(C)C', 'beta-caryophyllene (expected)'),
    ('C=C1CCC2OC2(C)CCC2C1CC2(C)C', 'caryophyllene oxide'),
]

for smi, desc in known_smiles:
    found = []
    for i, row in enumerate(pred_rows):
        pred_smi = row.get('SMILES', '').strip()
        if pred_smi == smi:
            found.append({
                'row_idx': i,
                'name': row.get('molecule_name', '?'),
                'score': row.get('composite_score', '?'),
                'mol_id': row.get('MOL_ID', '?')
            })
    print(f"  {desc}:")
    print(f"    SMILES匹配数: {len(found)}")
    for f_rec in found:
        print(f"    - row={f_rec['row_idx']} name={f_rec['name']} MOL_ID={f_rec['mol_id']} score={f_rec['score']}")

# ====== 5. 候选池中查找beta-caryophyllene by SMILES ======
print("\n" + "=" * 60)
print("5. 候选池中按SMILES查找beta-caryophyllene")
print("=" * 60)

for smi, desc in known_smiles:
    found = []
    for i, row in enumerate(all_rows):
        pool_smi = row.get('SMILES_std', row.get('SMILES', '')).strip()
        if pool_smi == smi:
            found.append({
                'row_idx': i,
                'name': row.get('molecule_name', row.get('Ingredient_name', '?')),
                'mol_id': row.get('MOL_ID', '?'),
                'source': row.get('zhuangyao_source', '?'),
                'herb': row.get('herb_cn_name', '?')
            })
    print(f"  {desc}:")
    print(f"    候选池SMILES匹配数: {len(found)}")
    for f_rec in found:
        print(f"    - row={f_rec['row_idx']} name='{f_rec['name']}' MOL_ID={f_rec['mol_id']} source={f_rec['source']} herb={f_rec['herb']}")

print("\n" + "=" * 60)
print("审计完成")
print("=" * 60)
