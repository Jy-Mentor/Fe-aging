"""分析当前候选池中哪些化合物可追溯至壮药"""
import csv, os

BASE = r"d:\铁衰老 绝不重蹈覆辙"
POOL_PATH = os.path.join(BASE, "L3", "results", "zhuangyao_herb_augmented_pool.csv")

# 1. 加载壮药列表
with open(os.path.join(BASE, "zhuangyao_data", "guangxi_zhuangyao_list.csv"), "r", encoding="utf-8-sig") as f:
    zy_list = list(csv.DictReader(f))
zy_names = set(x.get("cn_name", "").strip() for x in zy_list)
print(f"壮药列表: {len(zy_names)} 种")

# 2. 加载候选池
with open(POOL_PATH, "r", encoding="utf-8") as f:
    pool = list(csv.DictReader(f))
print(f"候选池总数: {len(pool)}")

# 3. 从原始池构建 MOL_ID -> herb 映射
orig_path = os.path.join(BASE, "L3", "results", "zhuangyao_compound_pool.csv")
mol_to_herbs = {}
with open(orig_path, "r", encoding="utf-8") as f:
    for x in csv.DictReader(f):
        mid = x.get("MOL_ID", "").strip()
        herb = x.get("herb_cn_name", "").strip()
        if mid and herb:
            mol_to_herbs.setdefault(mid, set()).add(herb)
print(f"MOL_ID->herb映射: {len(mol_to_herbs)} 个MOL_ID")

# 4. 分类
zhuangyao_rows = []      # 可追溯至壮药
orphan_rows = []          # 无法追溯
herb_matched = set()      # 壮药名匹配
tcmsp_matched = set()     # TCMSP_id匹配
total_matched = set()

for x in pool:
    mol_id = x.get("MOL_ID", "").strip()
    tc_id = x.get("TCMSP_id", "").strip()
    herb_cn = x.get("herb_cn_name", "").strip()
    source = x.get("source", "").strip()
    
    is_zhuangyao = False
    match_method = None
    
    # 方式1: herb_cn_name直接匹配壮药列表
    if herb_cn:
        for h in herb_cn.split(";"):
            h = h.strip()
            if h in zy_names:
                is_zhuangyao = True
                match_method = "herb_cn_name"
                herb_matched.add(h)
                break
    
    # 方式2: MOL_ID匹配
    if not is_zhuangyao and mol_id and mol_id in mol_to_herbs:
        herbs = mol_to_herbs[mol_id]
        for h in herbs:
            if h in zy_names:
                is_zhuangyao = True
                match_method = "MOL_ID_map"
                herb_matched.add(h)
                break
    
    # 方式3: TCMSP_id匹配
    if not is_zhuangyao and tc_id:
        for tid in tc_id.split(";"):
            tid = tid.strip()
            if tid in mol_to_herbs:
                for h in mol_to_herbs[tid]:
                    if h in zy_names:
                        is_zhuangyao = True
                        match_method = "TCMSP_id_map"
                        herb_matched.add(h)
                        break
            if is_zhuangyao:
                break
    
    if is_zhuangyao:
        zhuangyao_rows.append(x)
    else:
        orphan_rows.append(x)

# 5. 按来源分类统计
from collections import Counter
source_dist = Counter()
for x in zhuangyao_rows:
    src = x.get("source", "").strip()
    if not src:
        src = "TCMSP_original"
    source_dist[src] += 1

orphan_source_dist = Counter()
for x in orphan_rows:
    src = x.get("source", "").strip()
    if not src:
        src = "unknown"
    orphan_source_dist[src] += 1

print(f"\n{'='*60}")
print(f"壮药可追溯: {len(zhuangyao_rows)}/{len(pool)} ({100*len(zhuangyao_rows)/len(pool):.1f}%)")
print(f"无法追溯:   {len(orphan_rows)}/{len(pool)} ({100*len(orphan_rows)/len(pool):.1f}%)")
print(f"覆盖壮药: {len(herb_matched)}/{len(zy_names)}")

print(f"\n壮药来源分布:")
for src, cnt in source_dist.most_common():
    print(f"  {src}: {cnt}")

print(f"\n孤儿来源分布:")
for src, cnt in orphan_source_dist.most_common():
    print(f"  {src}: {cnt}")

# 6. 艾叶专项
print(f"\n{'='*60}")
print("艾叶化合物归属验证")
print(f"{'='*60}")
aiye_in_zy = [x for x in zhuangyao_rows if "艾叶" in (x.get("herb_cn_name","") + ";" + ";".join(mol_to_herbs.get(x.get("MOL_ID",""), set())) + ";" + ";".join(mol_to_herbs.get(x.get("TCMSP_id",""), set())))]
print(f"可追溯至艾叶: {len(aiye_in_zy)} 条")
for x in aiye_in_zy[:15]:
    print(f"  {x.get('molecule_name','?')[:50]} | MOL_ID={x.get('MOL_ID','?')} | source={x.get('source','?')}")

# 7. 缺失壮药
missing_herbs = zy_names - herb_matched
print(f"\n无任何化合物的壮药: {len(missing_herbs)} 种")
if missing_herbs:
    print("  前20:")
    for h in sorted(missing_herbs)[:20]:
        print(f"    {h}")

print(f"\n{'='*60}")
print("分析完成")
print(f"{'='*60}")