"""审查无患子(此芒苍)壮药化合物覆盖"""
import csv
import os

BASE = r"d:\铁衰老 绝不重蹈覆辙"

# 1. 壮药列表
print("=" * 60)
print("1. 壮药列表: 无患子")
print("=" * 60)
with open(os.path.join(BASE, "zhuangyao_data", "guangxi_zhuangyao_list.csv"), "r", encoding="utf-8-sig") as f:
    zy_rows = list(csv.DictReader(f))
wu = [x for x in zy_rows if "无患子" in x.get("cn_name", "")]
for x in wu:
    print(f"  idx={x['idx']}, cn_name={x['cn_name']}, zhuang_name={x['zhuang_name']}, vol={x['volume']}, year={x['year']}")

# 2. 候选池中无患子
print("\n" + "=" * 60)
print("2. 候选池中无患子来源")
print("=" * 60)
pool_path = os.path.join(BASE, "L3", "results", "zhuangyao_herb_augmented_pool.csv")
with open(pool_path, "r", encoding="utf-8") as f:
    pool_rows = list(csv.DictReader(f))
pool_cols = pool_rows[0].keys() if pool_rows else []
print(f"  候选池列名: {list(pool_cols)[:15]}...")

# 按 herb_cn_name 搜索
pool_wu = [x for x in pool_rows if "无患子" in x.get("herb_cn_name", "")]
print(f"  herb_cn_name匹配: {len(pool_wu)} 条")
for x in pool_wu[:10]:
    mn = x.get("molecule_name", "?")
    ing = x.get("Ingredient_name", "?")
    src = x.get("source", "?")
    smi = x.get("SMILES_std", "?")[:50]
    print(f"    {mn} | Ingredient={ing} | source={src} | SMILES={smi}")

# 3. HERB 源数据
print("\n" + "=" * 60)
print("3. HERB源数据中无患子")
print("=" * 60)
herb_path = r"D:\下载\HERB_herb_info_v2.txt"
with open(herb_path, "r", encoding="utf-8") as f:
    herb_rows = list(csv.DictReader(f, delimiter="\t"))
wu_herb = [x for x in herb_rows if "无患子" in x.get("Herb_cn_name", "")]
print(f"  HERB中无患子药材: {len(wu_herb)} 条")
for x in wu_herb:
    print(f"    Herb_id={x.get('Herb_id','?')} | cn_name={x.get('Herb_cn_name','?')} | pinyin={x.get('Herb_pinyin_name','?')}")

# 4. HERB 成分中无患子关联
print("\n" + "=" * 60)
print("4. HERB成分数据中无患子关联")
print("=" * 60)
ing_path = r"D:\下载\HERB_ingredient_info_v2.txt"
with open(ing_path, "r", encoding="utf-8") as f:
    ing_rows = list(csv.DictReader(f, delimiter="\t"))
print(f"  HERB成分总行数: {len(ing_rows)}")
print(f"  HERB成分列名: {list(ing_rows[0].keys())[:15]}...")

# 5. 按 herb_ingredient_relation 查找 (无患子 herb_id 关联的成分)
# 先找无患子的 herb_id
wu_herb_ids = set()
for x in wu_herb:
    hid = x.get("Herb_id", "").strip()
    if hid:
        wu_herb_ids.add(hid)

print(f"  无患子 Herb_id: {wu_herb_ids}")

# HERB成分没有直接关联herb_id, 需要查 relation 表
# 直接通过 Ingredient_name 含 "无患子" 或 "Sapindus" 搜索
wu_ing = [x for x in ing_rows if "无患子" in x.get("Ingredient_name", "") or "Sapindus" in x.get("Ingredient_name", "")]
print(f"  Ingredient_name含无患子/Sapindus: {len(wu_ing)} 条")
for x in wu_ing[:10]:
    print(f"    {x.get('Ingredient_name','?')} | SMILES={x.get('Canonical_smiles','?')[:50]}")

# 6. 预测文件中无患子相关
print("\n" + "=" * 60)
print("5. 预测文件(v70 fixed)中无患子相关")
print("=" * 60)
pred_path = os.path.join(BASE, "L4", "results_v10_minibatch", "tcm_predictions_full_v70_fixed.csv")
if os.path.exists(pred_path):
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_rows = list(csv.DictReader(f))
    pred_wu = [x for x in pred_rows if "无患子" in x.get("molecule_name", "")]
    print(f"  molecule_name含无患子: {len(pred_wu)} 条")
    pred_wu_sorted = sorted(pred_wu, key=lambda x: float(x.get("composite_score", 0)), reverse=True)
    for x in pred_wu_sorted[:10]:
        print(f"    {x['molecule_name']} | score={x.get('composite_score','?')} | MOL_ID={x.get('MOL_ID','?')}")
else:
    print("  [ERROR] 修复版预测文件不存在!")

# 7. TCMSP 爬取数据中无患子
print("\n" + "=" * 60)
print("6. TCMSP爬取数据中无患子")
print("=" * 60)
tcmsp_path = os.path.join(BASE, "L3", "results", "zhuangyao_ingredient_mapping_full.xlsx")
if os.path.exists(tcmsp_path):
    print(f"  TCMSP映射文件存在: {tcmsp_path}")
else:
    print(f"  [WARN] TCMSP映射文件不存在: {tcmsp_path}")

failed_path = os.path.join(BASE, "L3", "results", "zhuangyao_scrape_failed.csv")
if os.path.exists(failed_path):
    with open(failed_path, "r", encoding="utf-8") as f:
        failed_rows = list(csv.DictReader(f))
    wu_failed = [x for x in failed_rows if "无患子" in x.get("cn_name", "")]
    print(f"  TCMSP爬取失败列表中含无患子: {len(wu_failed)} 条")
    for x in wu_failed:
        print(f"    {x.get('cn_name','?')} | reason={x.get('reason','?')}")
else:
    print(f"  [WARN] 失败列表不存在: {failed_path}")

print("\n" + "=" * 60)
print("无患子审查完成")
print("=" * 60)