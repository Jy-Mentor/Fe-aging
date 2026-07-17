"""审查艾叶(Artemisia argyi)壮药化合物覆盖"""
import csv, os

BASE = r"d:\铁衰老 绝不重蹈覆辙"

# 1. 壮药列表
print("=" * 60)
print("1. 壮药列表: 艾叶")
print("=" * 60)
with open(os.path.join(BASE, "zhuangyao_data", "guangxi_zhuangyao_list.csv"), "r", encoding="utf-8-sig") as f:
    zy = list(csv.DictReader(f))
aiye = [x for x in zy if "艾叶" in x.get("cn_name", "")]
for x in aiye:
    print(f"  idx={x['idx']}, cn_name={x['cn_name']}, zhuang_name={x['zhuang_name']}, vol={x['volume']}, year={x['year']}")

# 2. 候选池中艾叶
print("\n" + "=" * 60)
print("2. 候选池中艾叶来源")
print("=" * 60)
pool_path = os.path.join(BASE, "L3", "results", "zhuangyao_herb_augmented_pool.csv")
with open(pool_path, "r", encoding="utf-8") as f:
    pool = list(csv.DictReader(f))

# 按 herb_cn_name 搜索
pool_aiye = [x for x in pool if "艾叶" in x.get("herb_cn_name", "")]
print(f"  herb_cn_name匹配: {len(pool_aiye)} 条")
for x in pool_aiye[:15]:
    mn = x.get("molecule_name", "?")
    src = x.get("source", "?")
    print(f"    {mn} | source={src}")

# 3. TCMSP爬取失败列表中艾叶
print("\n" + "=" * 60)
print("3. TCMSP爬取失败列表")
print("=" * 60)
failed_path = os.path.join(BASE, "L3", "results", "zhuangyao_scrape_failed.csv")
if os.path.exists(failed_path):
    with open(failed_path, "r", encoding="utf-8-sig") as f:
        failed = list(csv.DictReader(f))
    aiye_failed = [x for x in failed if "艾叶" in x.get("cn_name", "")]
    if aiye_failed:
        print(f"  艾叶在TCMSP爬取失败列表中: {len(aiye_failed)} 条")
        for x in aiye_failed:
            print(f"    {x.get('cn_name','?')} | reason={x.get('reason','?')}")
    else:
        print("  艾叶不在TCMSP爬取失败列表中")
else:
    # 检查TCMSP爬取结果
    print("  失败列表不存在，检查TCMSP爬取Excel")
    import importlib
    if importlib.util.find_spec("openpyxl"):
        import openpyxl
        xlsx_path = os.path.join(BASE, "L3", "results", "zhuangyao_ingredient_mapping_full.xlsx")
        if os.path.exists(xlsx_path):
            wb = openpyxl.load_workbook(xlsx_path, read_only=True)
            ws = wb.active
            aiye_found = False
            for row in ws.iter_rows(min_row=2, values_only=True):
                if row[0] and "艾叶" in str(row[0]):
                    aiye_found = True
                    break
            print(f"  TCMSP中有艾叶: {aiye_found}")
            wb.close()

# 4. HERB源数据中艾叶
print("\n" + "=" * 60)
print("4. HERB源数据中艾叶")
print("=" * 60)
herb_path = r"D:\下载\HERB_herb_info_v2.txt"
with open(herb_path, "r", encoding="utf-8") as f:
    herb = list(csv.DictReader(f, delimiter="\t"))
aiye_herb = [x for x in herb if "艾叶" in x.get("Herb_cn_name", "")]
print(f"  HERB中艾叶药材: {len(aiye_herb)} 条")
for x in aiye_herb:
    print(f"    Herb_id={x.get('Herb_id','?')} | cn_name={x.get('Herb_cn_name','?')} | pinyin={x.get('Herb_pinyin_name','?')}")

# 5. HERB成分中艾叶相关
print("\n" + "=" * 60)
print("5. HERB成分中艾叶相关")
print("=" * 60)
ing_path = r"D:\下载\HERB_ingredient_info_v2.txt"
with open(ing_path, "r", encoding="utf-8") as f:
    ing = list(csv.DictReader(f, delimiter="\t"))
aiye_ing = [x for x in ing if "艾叶" in x.get("Ingredient_name", "")]
print(f"  Ingredient_name含艾叶: {len(aiye_ing)} 条")
for x in aiye_ing[:10]:
    print(f"    {x.get('Ingredient_name','?')} | SMILES={x.get('Canonical_smiles','?')[:50]}")

# 6. 预测文件中艾叶
print("\n" + "=" * 60)
print("6. 预测文件(v70 fixed)中艾叶相关")
print("=" * 60)
pred_path = os.path.join(BASE, "L4", "results_v10_minibatch", "tcm_predictions_full_v70_fixed.csv")
if os.path.exists(pred_path):
    with open(pred_path, "r", encoding="utf-8") as f:
        pred = list(csv.DictReader(f))
    pred_aiye = [x for x in pred if "艾叶" in x.get("molecule_name", "")]
    print(f"  molecule_name含艾叶: {len(pred_aiye)} 条")
    pred_aiye.sort(key=lambda x: float(x.get("composite_score", 0)), reverse=True)
    for x in pred_aiye[:15]:
        print(f"    {x['molecule_name']} | score={x.get('composite_score','?')} | MOL_ID={x.get('MOL_ID','?')}")
else:
    print("  [ERROR] 修复版预测文件不存在!")

# 7. 艾叶在候选池中的排名
print("\n" + "=" * 60)
print("7. 艾叶在预测中的排名统计")
print("=" * 60)
if os.path.exists(pred_path):
    # 按分数排序全部
    all_sorted = sorted(pred, key=lambda x: float(x.get("composite_score", 0)), reverse=True)
    for i, row in enumerate(all_sorted):
        if "艾叶" in row.get("molecule_name", ""):
            rank = i + 1
            pct = 100 * rank / len(all_sorted)
            if pct <= 10:  # Top 10%
                print(f"  Rank {rank}/{len(all_sorted)} (top {pct:.1f}%): {row['molecule_name']} | score={row.get('composite_score','?')}")

print("\n" + "=" * 60)
print("艾叶审查完成")
print("=" * 60)