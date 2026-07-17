"""检查文献先验调整后的β-石竹烯和艾叶化合物排名"""
import csv
from pathlib import Path

RESULTS_DIR = Path(r"d:\铁衰老 绝不重蹈覆辙\L4\results_v10_minibatch")

# 读取调整后的全量预测
with open(RESULTS_DIR / "tcm_predictions_v70_literature_adjusted.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"全量调整后预测: {len(rows)} 个化合物")
print()

# 查找目标化合物
targets = [
    "beta-caryophyllene", "bata-caryophyllene", "caryophyllene oxide",
    "石竹烯", "coumarin", "quercetin", "beta-sitosterol",
    "dammaradienyl", "cycloartenol",
]

found = []
for i, row in enumerate(rows):
    name = (row.get("molecule_name", "") or "").lower()
    mol_id = row.get("MOL_ID", "")
    for t in targets:
        if t in name:
            found.append((
                i + 1, row.get("molecule_name", ""), mol_id,
                row.get("composite_score", ""), row.get("composite_score_adjusted", ""),
            ))
            break

print("目标化合物调整后排名 (全量):")
print(f"  {'排名':<8} {'化合物名':<45} {'MOL_ID':<12} {'原始得分':<14} {'调整后得分':<14}")
for rank, name, mol_id, orig, adj in sorted(found, key=lambda x: x[0]):
    print(f"  {rank:<8} {name:<45} {mol_id:<12} {orig:<14} {adj:<14}")

# 壮药池
print()
with open(RESULTS_DIR / "zhuangyao_top500_literature_adjusted.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    zhuang_rows = list(reader)

print(f"壮药Top500调整后排名: {len(zhuang_rows)} 个化合物")
found_z = []
for i, row in enumerate(zhuang_rows):
    name = (row.get("molecule_name", "") or "").lower()
    for t in targets:
        if t in name:
            found_z.append((
                i + 1, row.get("molecule_name", ""), row.get("MOL_ID", ""),
                row.get("composite_score", ""), row.get("composite_score_adjusted", ""),
            ))
            break

if found_z:
    print(f"  {'排名':<8} {'化合物名':<45} {'MOL_ID':<12} {'原始得分':<14} {'调整后得分':<14}")
    for rank, name, mol_id, orig, adj in sorted(found_z, key=lambda x: x[0]):
        print(f"  {rank:<8} {name:<45} {mol_id:<12} {orig:<14} {adj:<14}")
else:
    print("  壮药Top500中未找到目标化合物")

# 显示调整后的Top 20
print()
print("调整后Top 20:")
for i, row in enumerate(rows[:20]):
    name = row.get("molecule_name", "")
    orig = row.get("composite_score", "")
    adj = row.get("composite_score_adjusted", "")
    print(f"  {i+1:<4} {name:<50} orig={orig:<12} adj={adj:<12}")