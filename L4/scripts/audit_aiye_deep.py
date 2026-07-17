"""深度审查艾叶化合物关联 — 检查HERB关联表、已知活性成分归属"""
import csv, os

BASE = r"d:\铁衰老 绝不重蹈覆辙"

# 1. 候选池中艾叶3个化合物的预测分数
print("=" * 60)
print("1. 艾叶3个候选化合物预测分数")
print("=" * 60)
pool_path = os.path.join(BASE, "L3", "results", "zhuangyao_herb_augmented_pool.csv")
with open(pool_path, "r", encoding="utf-8") as f:
    pool = list(csv.DictReader(f))

aiye_pool = [x for x in pool if "艾叶" in x.get("herb_cn_name", "")]
print(f"  艾叶关联化合物: {len(aiye_pool)} 条")
for x in aiye_pool:
    mn = x.get("molecule_name", "?")
    smi = x.get("SMILES_std", "?")[:50]
    src = x.get("source", "?")
    print(f"    {mn} | SMILES={smi} | source={src}")

# 2. 在预测文件中查找这3个化合物
print("\n" + "=" * 60)
print("2. 艾叶3个化合物在预测中的分数")
print("=" * 60)
pred_path = os.path.join(BASE, "L4", "results_v10_minibatch", "tcm_predictions_full_v70_fixed.csv")
with open(pred_path, "r", encoding="utf-8") as f:
    pred = list(csv.DictReader(f))

# 按SMILES匹配
aiye_smiles = set(x.get("SMILES_std", "").strip() for x in aiye_pool)
for r in pred:
    smi = r.get("SMILES", "").strip()
    if smi in aiye_smiles:
        name = r.get("molecule_name", "?")
        score = r.get("composite_score", "?")
        print(f"    {name} | SMILES={smi[:50]} | score={score}")

# 3. 检查HERB herb-ingredient关联文件
print("\n" + "=" * 60)
print("3. HERB关联文件检查")
print("=" * 60)
herb_dir = r"D:\下载"
for fname in os.listdir(herb_dir):
    if "HERB" in fname and fname.endswith(".txt"):
        fpath = os.path.join(herb_dir, fname)
        size_mb = os.path.getsize(fpath) / (1024*1024)
        print(f"  {fname} ({size_mb:.1f}MB)")

# 4. 查找HERB herb_ingredient_relation
print("\n" + "=" * 60)
print("4. HERB中艾叶(Herb_id=HERB000066)关联的成分")
print("=" * 60)
# 如果有relation文件，读取关联
rel_path = os.path.join(herb_dir, "HERB_herb_ingredient_relation.txt")
if os.path.exists(rel_path):
    with open(rel_path, "r", encoding="utf-8") as f:
        rel = list(csv.DictReader(f, delimiter="\t"))
    print(f"  Relation文件列名: {list(rel[0].keys()) if rel else 'N/A'}")
    aiye_rel = [x for x in rel if x.get("Herb_id", "") == "HERB000066"]
    print(f"  艾叶关联成分数: {len(aiye_rel)}")
    
    # 获取关联的Ingredient_id
    aiye_ing_ids = set(x.get("Ingredient_id", "").strip() for x in aiye_rel)
    print(f"  唯一Ingredient_id: {len(aiye_ing_ids)}")
    
    # 从HERB成分文件中查找这些成分
    ing_path = r"D:\下载\HERB_ingredient_info_v2.txt"
    with open(ing_path, "r", encoding="utf-8") as f:
        ing = list(csv.DictReader(f, delimiter="\t"))
    
    aiye_ings = [x for x in ing if x.get("Ingredient_id", "").strip() in aiye_ing_ids]
    print(f"  艾叶关联成分详情: {len(aiye_ings)} 条")
    for x in aiye_ings[:20]:
        print(f"    {x.get('Ingredient_name','?')} | Ingredient_id={x.get('Ingredient_id','?')} | SMILES={x.get('Canonical_smiles','?')[:60]}")
else:
    print("  [WARN] HERB_herb_ingredient_relation.txt 不存在")
    
    # 尝试从augment脚本中查看HERB关联逻辑
    aug_path = os.path.join(BASE, "L3", "scripts", "augment_from_herb_v2.py")
    if os.path.exists(aug_path):
        with open(aug_path, "r", encoding="utf-8") as f:
            content = f.read()
        if "relation" in content.lower():
            print("  augment脚本中使用了relation逻辑")
        else:
            print("  augment脚本中未使用relation文件")

# 5. 检查已知艾叶活性成分在池中的归属
print("\n" + "=" * 60)
print("5. 已知艾叶活性成分在池中归属(PMID:37169131)")
print("=" * 60)
known_compounds = [
    "caryophyllene oxide", "石竹烯氧化物", "石竹烯",
    "alpha-bisabolol", "α-bisabolol", "bisabolol",
    "dihydro-beta-ionone", "dihydro-β-ionone",
]
for kw in known_compounds:
    found = [x for x in pool if kw.lower() in x.get("molecule_name", "").lower() or kw.lower() in x.get("Ingredient_name", "").lower()]
    if found:
        for x in found[:3]:
            mn = x.get("molecule_name", "?")
            herb = x.get("herb_cn_name", "?")
            src = x.get("source", "?")
            print(f"  '{kw}' → {mn} | herb={herb} | source={src}")
    else:
        print(f"  '{kw}' → 未在池中找到")

print("\n" + "=" * 60)
print("艾叶深度审查完成")
print("=" * 60)