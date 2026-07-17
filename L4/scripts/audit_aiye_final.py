"""检查艾叶在原始池和HERB增强池中的详细情况"""
import csv, os

BASE = r"d:\铁衰老 绝不重蹈覆辙"

# 1. 原始池
print("=" * 60)
print("1. 原始壮药池中艾叶 (TCMSP爬取)")
print("=" * 60)
orig_path = os.path.join(BASE, "L3", "results", "zhuangyao_compound_pool.csv")
if os.path.exists(orig_path):
    with open(orig_path, "r", encoding="utf-8") as f:
        orig = list(csv.DictReader(f))
    print(f"  原始池列名: {list(orig[0].keys())}")
    aiye_orig = [x for x in orig if "艾叶" in x.get("herb_cn_name", "")]
    print(f"  艾叶关联: {len(aiye_orig)} 条")
    for x in aiye_orig:
        print(f"    {x.get('molecule_name','?')} | MOL_ID={x.get('MOL_ID','?')} | source={x.get('source','?')}")

# 2. 增强池列名
print("\n" + "=" * 60)
print("2. 增强池列名 + 艾叶详细信息")
print("=" * 60)
pool_path = os.path.join(BASE, "L3", "results", "zhuangyao_herb_augmented_pool.csv")
with open(pool_path, "r", encoding="utf-8") as f:
    pool = list(csv.DictReader(f))
print(f"  增强池列名: {list(pool[0].keys())}")

# 艾叶
aiye = [x for x in pool if "艾叶" in x.get("herb_cn_name", "")]
print(f"  艾叶: {len(aiye)} 条")
for x in aiye:
    print(f"    molecule_name={x.get('molecule_name','?')}")
    print(f"    MOL_ID={x.get('MOL_ID','?')}")
    print(f"    SMILES_std={x.get('SMILES_std','?')[:60]}...")
    print(f"    source={x.get('source','?')}")
    print(f"    herb_cn_name={x.get('herb_cn_name','?')}")
    print(f"    Ingredient_name={x.get('Ingredient_name','?')}")
    print(f"    Ingredient_id={x.get('Ingredient_id','?')}")
    print("    ---")

# 3. HERB成分中与艾叶TCMSP_id=2关联的成分
print("\n" + "=" * 60)
print("3. HERB成分中TCMSP_id含'2'的成分")
print("=" * 60)
ing_path = r"D:\下载\HERB_ingredient_info_v2.txt"
with open(ing_path, "r", encoding="utf-8") as f:
    ing_all = list(csv.DictReader(f, delimiter="\t"))

# HERB成分中TCMSP_id=2的是艾叶相关的
tcmsp2_ings = []
for x in ing_all:
    tc = x.get("TCMSP_id", "").strip()
    if tc and "2" in tc.split(";"):
        tcmsp2_ings.append(x)

print(f"  TCMSP_id含'2'的HERB成分: {len(tcmsp2_ings)} 条")
print(f"  前10条:")
for x in tcmsp2_ings[:10]:
    print(f"    {x.get('Ingredient_name','?')} | TCMSP_id={x.get('TCMSP_id','?')} | SMILES={x.get('Canonical_smiles','?')[:50]}")

# 4. 检查池中source=HERB_TCMSP且TCMSP_id=2的
print("\n" + "=" * 60)
print("4. 增强池中HERB_TCMSP来源 + TCMSP_id=2")
print("=" * 60)
herb_tcmsp2 = [x for x in pool if x.get("source", "") == "HERB_TCMSP" and "2" in x.get("TCMSP_id", "")]
print(f"  匹配: {len(herb_tcmsp2)} 条")
for x in herb_tcmsp2[:10]:
    print(f"    {x.get('molecule_name','?')} | TCMSP_id={x.get('TCMSP_id','?')} | herb_cn_name={x.get('herb_cn_name','?')}")

# 5. 检查池中source=HERB_noTCMSP且名称含Artemisia/艾/argyi
print("\n" + "=" * 60)
print("5. 池中HERB_noTCMSP来源含Artemisia/argyi/艾")
print("=" * 60)
artemisia_pool = []
for x in pool:
    mn = x.get("molecule_name", "").lower()
    ing = x.get("Ingredient_name", "").lower()
    if "artemisia" in mn or "artemisia" in ing or "argyi" in mn or "argyi" in ing:
        artemisia_pool.append(x)
print(f"  匹配: {len(artemisia_pool)} 条")
for x in artemisia_pool[:10]:
    print(f"    {x.get('molecule_name','?')} | herb_cn_name={x.get('herb_cn_name','?')} | source={x.get('source','?')}")

# 6. 汇总: 艾叶已知活性成分在预测中的排名
print("\n" + "=" * 60)
print("6. 艾叶已知活性成分预测排名 (PMID:37169131)")
print("=" * 60)
pred_path = os.path.join(BASE, "L4", "results_v10_minibatch", "tcm_predictions_full_v70_fixed.csv")
with open(pred_path, "r", encoding="utf-8") as f:
    pred = list(csv.DictReader(f))

# 按SMILES查找
targets = {
    "caryophyllene oxide": "CC1=CCCC2(CO2)C(C)(C)C2CCC(C)(C)C2C1",
    "alpha-bisabolol": None,
    "dihydro-beta-ionone": None,
}
# 用名称搜索
for r in pred:
    name = r.get("molecule_name", "").lower()
    score = float(r.get("composite_score", 0))
    if "caryophyllene oxide" in name:
        all_sorted = sorted(pred, key=lambda x: float(x.get("composite_score", 0)), reverse=True)
        for i, sr in enumerate(all_sorted):
            if sr.get("molecule_name", "") == r.get("molecule_name", ""):
                print(f"  caryophyllene oxide: rank={i+1}/{len(pred)}, score={score:.4f}")
                break
    if "bisabolol" in name:
        all_sorted = sorted(pred, key=lambda x: float(x.get("composite_score", 0)), reverse=True)
        for i, sr in enumerate(all_sorted):
            if sr.get("molecule_name", "") == r.get("molecule_name", ""):
                print(f"  {r.get('molecule_name','?')}: rank={i+1}/{len(pred)}, score={score:.4f}")
                break
    if "dihydro" in name and "ionone" in name:
        all_sorted = sorted(pred, key=lambda x: float(x.get("composite_score", 0)), reverse=True)
        for i, sr in enumerate(all_sorted):
            if sr.get("molecule_name", "") == r.get("molecule_name", ""):
                print(f"  {r.get('molecule_name','?')}: rank={i+1}/{len(pred)}, score={score:.4f}")
                break

print("\n" + "=" * 60)
print("审查完成")
print("=" * 60)