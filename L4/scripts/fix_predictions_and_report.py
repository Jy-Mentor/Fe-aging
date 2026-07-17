"""
修复v70预测文件名称丢失 + 生成壮药排名报告
=============================================
1. 从增强池(Ingredient_name)补充预测文件中缺失的molecule_name/MOL_ID
2. 生成Top 500正确排名报告
3. 生成beta-caryophyllene专项分析
"""
import csv
import os
import statistics

RESULTS_DIR = r"d:\铁衰老 绝不重蹈覆辙\L4\results_v10_minibatch"
POOL_PATH = r"d:\铁衰老 绝不重蹈覆辙\L3\results\zhuangyao_herb_augmented_pool.csv"
PRED_PATH = os.path.join(RESULTS_DIR, "tcm_predictions_full_v70.csv")

print("=" * 60)
print("阶段1: 加载增强池名称映射")
print("=" * 60)

# 构建 SMILES -> (molecule_name, Ingredient_name, MOL_ID) 映射
smiles_map = {}
with open(POOL_PATH, 'r', encoding='utf-8') as f:
    pool_reader = csv.DictReader(f)
    pool_rows = list(pool_reader)
    for row in pool_rows:
        smi = row.get('SMILES_std', '').strip()
        if smi:
            mol_name = row.get('molecule_name', '').strip()
            ing_name = row.get('Ingredient_name', '').strip()
            mol_id = row.get('MOL_ID', '').strip()
            # 优先用molecule_name，否则用Ingredient_name
            display_name = mol_name if mol_name else ing_name
            
            if smi not in smiles_map or not smiles_map[smi][1]:
                smiles_map[smi] = (display_name, mol_id, ing_name, mol_name)

print(f"  增强池映射: {len(smiles_map)} 唯一SMILES")
has_name = sum(1 for v in smiles_map.values() if v[0])
print(f"  有名称: {has_name}/{len(smiles_map)}")

# ====== 阶段2: 修复预测文件 ======
print("\n" + "=" * 60)
print("阶段2: 修复预测文件名称")
print("=" * 60)

FIXED_PATH = os.path.join(RESULTS_DIR, "tcm_predictions_full_v70_fixed.csv")

with open(PRED_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames
    
    fixed_rows = []
    name_fixed_count = 0
    mol_fixed_count = 0
    no_match_count = 0
    
    for i, row in enumerate(reader):
        smi = row.get('SMILES', '').strip()
        
        if smi in smiles_map:
            display_name, mol_id, ing_name, orig_mol_name = smiles_map[smi]
            
            # 修复molecule_name
            if not row.get('molecule_name', '').strip() and display_name:
                row['molecule_name'] = display_name
                name_fixed_count += 1
            
            # 修复MOL_ID
            if not row.get('MOL_ID', '').strip() and mol_id:
                row['MOL_ID'] = mol_id
                mol_fixed_count += 1
        else:
            no_match_count += 1
        
        fixed_rows.append(row)
    
    print(f"  总行数: {len(fixed_rows)}")
    print(f"  molecule_name修复: {name_fixed_count}")
    print(f"  MOL_ID修复: {mol_fixed_count}")
    print(f"  无匹配: {no_match_count}")

# 保存修复后文件
with open(FIXED_PATH, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    writer.writerows(fixed_rows)

fix_size_mb = os.path.getsize(FIXED_PATH) / (1024 * 1024)
print(f"  保存: {FIXED_PATH} ({fix_size_mb:.1f} MB)")

# ====== 阶段3: 验证修复 ======
print("\n" + "=" * 60)
print("阶段3: 验证修复结果")
print("=" * 60)

# 搜索beta-caryophyllene
beta_found = []
for row in fixed_rows:
    name = row.get('molecule_name', '')
    if 'caryophyllene' in name.lower() or '石竹烯' in name:
        beta_found.append(row)

print(f"  修复后beta-caryophyllene条目: {len(beta_found)}")
for rec in beta_found[:15]:
    print(f"    - {rec['molecule_name']} | score={rec.get('composite_score','?')} | MOL_ID={rec.get('MOL_ID','?')}")

# 统计名称覆盖率
has_name_after = sum(1 for r in fixed_rows if r.get('molecule_name', '').strip())
print(f"\n  修复后molecule_name覆盖率: {has_name_after}/{len(fixed_rows)} ({100*has_name_after/len(fixed_rows):.1f}%)")

# ====== 阶段4: 生成Top 500排名报告 ======
print("\n" + "=" * 60)
print("阶段4: 生成壮药排名报告")
print("=" * 60)

# 按composite_score降序排列
def get_score(row):
    try:
        return float(row.get('composite_score', 0))
    except:
        return 0.0

sorted_rows = sorted(fixed_rows, key=get_score, reverse=True)

REPORT_PATH = os.path.join(RESULTS_DIR, "zhuangyao_top500_ranked_report.csv")
TOP_REPORT_FIELDS = ['rank', 'MOL_ID', 'molecule_name', 'SMILES', 'composite_score',
                     'ABCC1', 'GPX4', 'ACSL4', 'TFRC', 'SLC7A11', 'FTH1', 'NFE2L2']

top500 = []
for i, row in enumerate(sorted_rows[:500]):
    entry = {'rank': i + 1}
    for col in TOP_REPORT_FIELDS[1:]:
        entry[col] = row.get(col, '')
    top500.append(entry)

with open(REPORT_PATH, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=TOP_REPORT_FIELDS)
    writer.writeheader()
    writer.writerows(top500)

print(f"  保存: {REPORT_PATH}")

print("\n  Top 30 壮药候选化合物:")
for entry in top500[:30]:
    print(f"    Rank {entry['rank']:3d}: {entry['molecule_name'][:50]:50s} | score={entry['composite_score']}")

# ====== 阶段5: Top 500中beta-caryophyllene ======
print("\n" + "=" * 60)
print("阶段5: Top 500中beta-caryophyllene")
print("=" * 60)

beta_in_top = [e for e in top500 if 'caryophyllene' in e['molecule_name'].lower() or '石竹烯' in e['molecule_name']]
print(f"  Top 500中caryophyllene相关: {len(beta_in_top)} 条")
for e in beta_in_top:
    print(f"    Rank {e['rank']}: {e['molecule_name']} | score={e['composite_score']}")

# 搜索beta-caryophyllene在所有排名中的位置
print("\n  beta-caryophyllene在所有化合物中的排名:")
for i, row in enumerate(sorted_rows):
    name = row.get('molecule_name', '')
    if 'bata' in name.lower() or 'beta-caryophyllene' in name.lower():
        score = row.get('composite_score', '?')
        print(f"    Rank {i+1}: {name} | score={score}")

print("\n" + "=" * 60)
print("修复完成!")
print("=" * 60)
