"""检查数据池列结构差异"""
import csv
import os

POOL_PATH = r"d:\铁衰老 绝不重蹈覆辙\L3\results\zhuangyao_compound_pool.csv"
AUG_PATH = r"d:\铁衰老 绝不重蹈覆辙\L3\results\zhuangyao_herb_augmented_pool.csv"

print("=" * 60)
print("1. 原始壮药池 (zhuangyao_compound_pool.csv)")
print("=" * 60)

if os.path.exists(POOL_PATH):
    with open(POOL_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        print(f"  列名: {cols}")
        rows = list(reader)
        print(f"  行数: {len(rows)}")
        print(f"  前5行 molecule_name/Ingredient_name:")
        for i, r in enumerate(rows[:5]):
            mn = r.get('molecule_name', '?')
            in_name = r.get('Ingredient_name', '?')
            mol_id = r.get('MOL_ID', '?')
            print(f"    [{i}] MOL_ID={mol_id} | molecule_name='{mn}' | Ingredient_name='{in_name}'")
else:
    print("  文件不存在!")

print("\n" + "=" * 60)
print("2. 增强后壮药池 (zhuangyao_herb_augmented_pool.csv)")
print("=" * 60)

if os.path.exists(AUG_PATH):
    with open(AUG_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        print(f"  列名: {cols}")
        rows = list(reader)
        print(f"  行数: {len(rows)}")
        # 检查有名称 vs 无名称
        has_name = sum(1 for r in rows if r.get('molecule_name', '').strip())
        has_ing = sum(1 for r in rows if r.get('Ingredient_name', '').strip())
        has_both = sum(1 for r in rows if r.get('molecule_name', '').strip() and r.get('Ingredient_name', '').strip())
        has_none = sum(1 for r in rows if not r.get('molecule_name', '').strip() and not r.get('Ingredient_name', '').strip())
        print(f"  molecule_name有值: {has_name}/{len(rows)}")
        print(f"  Ingredient_name有值: {has_ing}/{len(rows)}")
        print(f"  两者都有: {has_both}")
        print(f"  两者都无: {has_none}")
        
        # 检查哪些source有Ingredient_name但无molecule_name
        print("\n  按source分类的名称完整性:")
        sources = {}
        for r in rows:
            src = r.get('source', '?')
            if src not in sources:
                sources[src] = {'total': 0, 'has_mol_name': 0, 'has_ing_name': 0}
            sources[src]['total'] += 1
            if r.get('molecule_name', '').strip():
                sources[src]['has_mol_name'] += 1
            if r.get('Ingredient_name', '').strip():
                sources[src]['has_ing_name'] += 1
        
        for src, stats in sorted(sources.items()):
            print(f"    {src}: total={stats['total']}, molecule_name={stats['has_mol_name']}, Ingredient_name={stats['has_ing_name']}")
        
        # 显示 HERB_noTCMSP 来源前5条（这些应该是名称丢失的主要来源）
        print("\n  HERB_noTCMSP来源样本(前5):")
        count = 0
        for r in rows:
            if r.get('source', '') == 'HERB_noTCMSP' and count < 5:
                print(f"    MOL_ID={r.get('MOL_ID','?')} | molecule_name='{r.get('molecule_name','?')}' | Ingredient_name='{r.get('Ingredient_name','?')}' | Ingredient_id={r.get('Ingredient_id','?')}")
                count += 1

print("\n完成")
