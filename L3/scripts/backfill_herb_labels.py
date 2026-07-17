"""回填候选池herb_cn_name — 从原始池构建MOL_ID→herb映射"""
import csv, os, shutil

BASE = r"d:\铁衰老 绝不重蹈覆辙"
POOL_PATH = os.path.join(BASE, "L3", "results", "zhuangyao_herb_augmented_pool.csv")
ORIG_PATH = os.path.join(BASE, "L3", "results", "zhuangyao_compound_pool.csv")
POOL_BACKUP = os.path.join(BASE, "L3", "results", "zhuangyao_herb_augmented_pool_backup_v2.csv")

# 1. 从原始池构建 MOL_ID -> herb_cn_name 映射
print("1. 从原始池构建MOL_ID->herb映射...")
mol_to_herbs = {}
with open(ORIG_PATH, "r", encoding="utf-8") as f:
    orig = list(csv.DictReader(f))
print(f"  原始池: {len(orig)} 条")

for x in orig:
    mol_id = x.get("MOL_ID", "").strip()
    herb = x.get("herb_cn_name", "").strip()
    if mol_id and herb:
        mol_to_herbs.setdefault(mol_id, set()).add(herb)

print(f"  MOL_ID->herb映射: {len(mol_to_herbs)} 个MOL_ID")
print(f"  涉及壮药: {len(set().union(*mol_to_herbs.values()))} 种")

# 2. 加载增强池
print("\n2. 加载增强池...")
with open(POOL_PATH, "r", encoding="utf-8") as f:
    pool = list(csv.DictReader(f))
print(f"  池中总数: {len(pool)}")

before_herbs = set()
for x in pool:
    h = x.get("herb_cn_name", "").strip()
    if h:
        before_herbs.add(h)
print(f"  回填前: {len(before_herbs)} 种壮药有标注")

# 3. 回填
print("\n3. 回填中...")
filled = 0
filled_herbs = set()

for x in pool:
    if x.get("herb_cn_name", "").strip():
        continue  # 已有标注
    
    # 通过MOL_ID
    mol_id = x.get("MOL_ID", "").strip()
    if mol_id and mol_id in mol_to_herbs:
        herbs = ";".join(sorted(mol_to_herbs[mol_id]))
        x["herb_cn_name"] = herbs
        filled += 1
        filled_herbs.update(mol_to_herbs[mol_id])
        continue
    
    # 通过TCMSP_id (HERB策略的TCMSP_id是MOL_ID格式)
    tc_id = x.get("TCMSP_id", "").strip()
    if tc_id:
        for tid in tc_id.split(";"):
            tid = tid.strip()
            if tid in mol_to_herbs:
                herbs = ";".join(sorted(mol_to_herbs[tid]))
                x["herb_cn_name"] = herbs
                filled += 1
                filled_herbs.update(mol_to_herbs[tid])
                break

print(f"  回填: {filled} 条")
print(f"  新增壮药标注: {len(filled_herbs)} 种")

# 4. 统计
after_herbs = set()
after_has = 0
for x in pool:
    h = x.get("herb_cn_name", "").strip()
    if h:
        after_herbs.add(h)
        after_has += 1

print(f"\n4. 回填后统计:")
print(f"  有herb_cn_name的条数: {after_has}/{len(pool)} ({100*after_has/len(pool):.1f}%)")
print(f"  标注壮药数: {len(after_herbs)} 种")

# 5. 艾叶
print(f"\n5. 艾叶回填后:")
aiye = [x for x in pool if "艾叶" in x.get("herb_cn_name", "")]
print(f"  艾叶关联: {len(aiye)} 条")
for x in aiye[:15]:
    print(f"    {x.get('molecule_name','?')[:50]} | MOL_ID={x.get('MOL_ID','?')}")

# 6. 保存
print(f"\n6. 保存...")
shutil.copy2(POOL_PATH, POOL_BACKUP)
print(f"  备份: {POOL_BACKUP}")

with open(POOL_PATH, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=pool[0].keys())
    writer.writeheader()
    writer.writerows(pool)
print(f"  保存: {POOL_PATH}")

print("\n完成!")