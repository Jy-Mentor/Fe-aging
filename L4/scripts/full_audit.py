"""全量数据完整性二次审查"""
import csv, os

BASE = r"d:\铁衰老 绝不重蹈覆辙"

# 1. 壮药列表覆盖率
print("=" * 60)
print("1. 壮药列表覆盖率")
print("=" * 60)
with open(os.path.join(BASE, "zhuangyao_data", "guangxi_zhuangyao_list.csv"), "r", encoding="utf-8-sig") as f:
    zy_list = list(csv.DictReader(f))
print(f"  壮药总数: {len(zy_list)}")

# 候选池
pool_path = os.path.join(BASE, "L3", "results", "zhuangyao_herb_augmented_pool.csv")
with open(pool_path, "r", encoding="utf-8") as f:
    pool = list(csv.DictReader(f))

# 统计每个壮药的化合物数
herb_compounds = {}
for x in pool:
    herb = x.get("herb_cn_name", "").strip()
    if herb:
        herb_compounds.setdefault(herb, []).append(x)

# 有化合物的壮药
has_compounds = set(herb_compounds.keys())
# 从壮药列表匹配
matched = set()
for h in zy_list:
    cn = h.get("cn_name", "").strip()
    if cn in has_compounds:
        matched.add(cn)

print(f"  候选池中有herb_cn_name的壮药: {len(has_compounds)}")
print(f"  壮药列表中匹配到的: {len(matched)}/{len(zy_list)}")

# 缺少化合物的壮药
missing = []
for h in zy_list:
    cn = h.get("cn_name", "").strip()
    if cn not in has_compounds:
        missing.append((cn, h.get("zhuang_name", "?")))

print(f"  缺少标注化合物的壮药: {len(missing)}")
print(f"  前20个缺失:")
for cn, zname in missing[:20]:
    print(f"    {cn} ({zname})")

# 2. 预测文件统计
print("\n" + "=" * 60)
print("2. 预测文件完整性")
print("=" * 60)
pred_path = os.path.join(BASE, "L4", "results_v10_minibatch", "tcm_predictions_full_v70_fixed.csv")
with open(pred_path, "r", encoding="utf-8") as f:
    pred = list(csv.DictReader(f))

print(f"  预测总数: {len(pred)}")
has_name = sum(1 for r in pred if r.get("molecule_name", "").strip())
has_score = 0
for r in pred:
    try:
        float(r.get("composite_score", 0))
        has_score += 1
    except:
        pass
print(f"  有名称: {has_name}/{len(pred)}")
print(f"  有分数: {has_score}/{len(pred)}")

# 3. 关键模型文件
print("\n" + "=" * 60)
print("3. 模型文件完整性")
print("=" * 60)
results_dir = os.path.join(BASE, "L4", "results_v10_minibatch")
checkpoints = ["sage_best_v70.pt", "hgt_best_v70.pt", "simplehgn_best_v70.pt"]
for cp in checkpoints:
    cp_path = os.path.join(results_dir, cp)
    if os.path.exists(cp_path):
        print(f"  [OK] {cp} ({os.path.getsize(cp_path)/1024/1024:.1f}MB)")
    else:
        print(f"  [MISSING] {cp}")

# 4. 训练日志
print("\n" + "=" * 60)
print("4. 训练日志")
print("=" * 60)
log_dir = os.path.join(BASE, "L4", "logs")
if os.path.exists(log_dir):
    logs = [f for f in os.listdir(log_dir) if "v70" in f]
    print(f"  v70相关日志: {len(logs)} 个")
    for l in logs[:5]:
        print(f"    {l}")
else:
    print("  logs目录不存在")

# 5. 壮药化合物分布统计
print("\n" + "=" * 60)
print("5. 壮药化合物分布 (Top 20)")
print("=" * 60)
sorted_herbs = sorted(herb_compounds.items(), key=lambda x: len(x[1]), reverse=True)
print("  Top 20 化合物最多的壮药:")
for herb, compounds in sorted_herbs[:20]:
    print(f"    {herb}: {len(compounds)} 个")

# 化合物数分布
from collections import Counter
dist = Counter(len(v) for v in herb_compounds.values())
print("\n  化合物数分布:")
for n in sorted(dist.keys()):
    print(f"    {n}个化合物: {dist[n]}种壮药")

print("=" * 60)
print("全量审查完成")
print("=" * 60)