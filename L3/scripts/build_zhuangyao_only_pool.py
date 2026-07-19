"""
扩展版壮药专属候选池构建
策略：
  1. 从 herb_all.xlsx 扩展壮药名称匹配（9192种草药）
  2. 从 ingredient_all.xlsx 获取 TCM_name -> Ingredient_id 映射
  3. 通过 Herb_pinyin_name, TCM_name 多维度匹配壮药
  4. 合并 TCMSP 来源的壮药池
  5. SMILES 标准化 + 去重
"""
import csv
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

BASE = r"d:\铁衰老 绝不重蹈覆辙"
HERB_ALL = r"D:\下载\herb_all.xlsx"
INGREDIENT_ALL = r"D:\下载\ingredient_all.xlsx"
HERB_ING = r"D:\下载\HERB_ingredient_info_v2.txt"
ZY_LIST = os.path.join(BASE, "zhuangyao_data", "guangxi_zhuangyao_list.csv")
TCMSP_POOL = os.path.join(BASE, "L3", "results", "zhuangyao_compound_pool.csv")
OUT_DIR = os.path.join(BASE, "L3", "results")
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

print("=" * 70)
print("壮药专属候选池构建 (扩展版)")
print("=" * 70)

# ============================================================
# 步骤1: 加载壮药列表
# ============================================================
print("\n1. 加载壮药列表...")
with open(ZY_LIST, "r", encoding="utf-8-sig") as f:
    zy_list = list(csv.DictReader(f))
zy_names = set(x.get("cn_name", "").strip() for x in zy_list)
zy_names_lower = {n.lower(): n for n in zy_names}
print(f"   壮药数量: {len(zy_names)}")


def read_xlsx_rows(path, label):
    """读取 xlsx 文件所有行，返回 list of dict"""
    print(f"   读取 {label}...")
    z = zipfile.ZipFile(path)
    strings = []
    if "xl/sharedStrings.xml" in z.namelist():
        with z.open("xl/sharedStrings.xml") as f:
            ss_root = ET.fromstring(f.read().decode("utf-8"))
        for si in ss_root.findall(".//s:si", NS):
            t = si.find(".//s:t", NS)
            strings.append(t.text if t is not None and t.text else "")
    
    with z.open("xl/worksheets/sheet1.xml") as f:
        sheet_root = ET.fromstring(f.read().decode("utf-8"))
    
    rows = []
    total = 0
    for row_el in sheet_root.findall(".//s:row", NS):
        cells = {}
        for c in row_el.findall("s:c", NS):
            ref = c.get("r")
            if not ref:
                continue
            col_letter = re.match(r"([A-Z]+)", ref).group(1)
            if len(col_letter) == 1:
                col_idx = ord(col_letter) - ord("A")
            else:
                col_idx = 26 * (len(col_letter) - 1) + ord(col_letter[-1]) - ord("A")
            t = c.get("t")
            v = c.find("s:v", NS)
            val = v.text if v is not None else ""
            if t == "s" and val:
                idx = int(val)
                if 0 <= idx < len(strings):
                    val = strings[idx]
            cells[col_idx] = val
        total += 1
        if total % 10000 == 0:
            print(f"      {total} 行...")
        rows.append(cells)
    z.close()
    print(f"      共 {len(rows)} 行")
    return rows


# ============================================================
# 步骤2: 从 herb_all.xlsx 匹配壮药
# ============================================================
print("\n2. 从 herb_all.xlsx 匹配壮药...")
herb_rows = read_xlsx_rows(HERB_ALL, "herb_all.xlsx")
herb_header = herb_rows[0]

# 列映射: 3=TCM_name, 5=Herb_pinyin_name, 20=Herb_ID
TCM_NAME_COL = 3
PINYIN_COL = 5
HERB_ID_COL = 20

herb_name_to_id = {}
herb_pinyin_to_id = {}
herb_matched = {}  # zy_name -> set of Herb_ID

for row in herb_rows[1:]:
    tcm_name = str(row.get(TCM_NAME_COL, "")).strip()
    pinyin = str(row.get(PINYIN_COL, "")).strip()
    herb_id = str(row.get(HERB_ID_COL, "")).strip()
    
    if tcm_name:
        herb_name_to_id[tcm_name] = herb_id
        # 精确匹配
        if tcm_name in zy_names:
            herb_matched.setdefault(tcm_name, set()).add(herb_id)
    if pinyin:
        herb_pinyin_to_id[pinyin] = herb_id

# 模糊匹配
for zy_name in zy_names:
    if zy_name in herb_matched:
        continue
    zy_lower = zy_name.lower()
    # 尝试在 herb_name_to_id 中查找包含关系
    for tcm_name, hid in herb_name_to_id.items():
        if zy_name in tcm_name or tcm_name in zy_name:
            herb_matched.setdefault(zy_name, set()).add(hid)
            break

print(f"   herb_all 精确匹配: {len(herb_matched)} 种壮药")

# 使用 pinyin 匹配
pinyin_matched = 0
for zy_name in zy_names:
    if zy_name in herb_matched:
        continue
    # 尝试通过拼音匹配（需要构建壮药拼音映射）
    # 先跳过，壮药列表没有拼音列

print(f"   herb_all 最终匹配: {len(herb_matched)} 种壮药")

# ============================================================
# 步骤3: 从 ingredient_all.xlsx 获取 Ingredient_id 映射
# ============================================================
print("\n3. 从 ingredient_all.xlsx 获取 Ingredient_id 映射...")
ing_rows = read_xlsx_rows(INGREDIENT_ALL, "ingredient_all.xlsx")

# 列: 5=TCM_name, 14=Ingredient_id, 19=TCMSP_id
ING_TCM_NAME = 5
ING_ING_ID = 14
ING_TCMSP_ID = 19

tcm_to_ings = {}
tcm_to_tcmsp = {}
for row in ing_rows[1:]:
    tcm_name = str(row.get(ING_TCM_NAME, "")).strip()
    ing_id = str(row.get(ING_ING_ID, "")).strip()
    tc_id = str(row.get(ING_TCMSP_ID, "")).strip()
    if tcm_name and ing_id:
        tcm_to_ings.setdefault(tcm_name, set()).add(ing_id)
    if tcm_name and tc_id:
        tcm_to_tcmsp.setdefault(tcm_name, set()).add(tc_id)

print(f"   唯一草药: {len(tcm_to_ings)}")
print(f"   总 Ingredient_id: {sum(len(v) for v in tcm_to_ings.values())}")

# ============================================================
# 步骤4: 多维度匹配壮药
# ============================================================
print("\n4. 多维度匹配壮药...")

# 合并所有匹配来源
all_zy_ing_ids = set()
matched_zy = set()

# 来源1: ingredient_all 精确匹配
for zy_name in zy_names:
    if zy_name in tcm_to_ings:
        all_zy_ing_ids.update(tcm_to_ings[zy_name])
        matched_zy.add(zy_name)

# 来源2: ingredient_all 模糊匹配
for zy_name in zy_names:
    if zy_name in matched_zy:
        continue
    for tcm_name in tcm_to_ings:
        if zy_name in tcm_name or tcm_name in zy_name:
            all_zy_ing_ids.update(tcm_to_ings[tcm_name])
            matched_zy.add(zy_name)
            break

# 来源3: herb_all 匹配的草药 -> 在 ingredient_all 中查找
for zy_name in zy_names:
    if zy_name in matched_zy:
        continue
    if zy_name in herb_matched:
        # herb_all 匹配了，尝试在 ingredient_all 中查找同名
        if zy_name in tcm_to_ings:
            all_zy_ing_ids.update(tcm_to_ings[zy_name])
            matched_zy.add(zy_name)

print(f"   匹配壮药: {len(matched_zy)}/{len(zy_names)}")
print(f"   壮药专属 Ingredient_id: {len(all_zy_ing_ids)}")

unmatched = zy_names - matched_zy
print(f"   未匹配: {len(unmatched)} 种")
if unmatched:
    print(f"   前20: {sorted(unmatched)[:20]}")

# ============================================================
# 步骤5: 合并 TCMSP 来源壮药池
# ============================================================
print("\n5. 合并 TCMSP 来源壮药池...")
tcmsp_compounds = []
with open(TCMSP_POOL, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        tcmsp_compounds.append(row)
print(f"   TCMSP 壮药池: {len(tcmsp_compounds)} 个化合物")

# 获取 TCMSP 壮药池的 SMILES
tcmsp_smiles = set()
for c in tcmsp_compounds:
    smi = c.get("SMILES_std", "").strip()
    if smi:
        tcmsp_smiles.add(smi)

# ============================================================
# 步骤6: 从 HERB_ingredient_info_v2.txt 提取壮药成分
# ============================================================
print("\n6. 从 HERB_ingredient_info_v2.txt 提取壮药成分...")
herb_ing_data = {}
all_smiles_seen = set(tcmsp_smiles)  # 用于去重
herb_compounds = []

with open(HERB_ING, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        iid = str(row.get("Ingredient_id", "")).strip()
        if iid and iid in all_zy_ing_ids:
            smi = str(row.get("Canonical_smiles", "")).strip()
            if smi and smi != "nan" and smi not in all_smiles_seen:
                all_smiles_seen.add(smi)
                herb_compounds.append({
                    "source": "HERB_via_ingredient_all",
                    "Ingredient_id": iid,
                    "Ingredient_name": str(row.get("Ingredient_name", "")).strip(),
                    "SMILES_raw": smi,
                    "MW_HERB": str(row.get("MolWt", "")).strip(),
                    "OB_score": str(row.get("OB_score", "")).strip(),
                    "Drug_likeness": str(row.get("Drug_likeness", "")).strip(),
                    "TCMSP_id": str(row.get("TCMSP_id", "")).strip(),
                    "PubChem_id": str(row.get("PubChem_id", "")).strip(),
                })

print(f"   HERB 壮药成分(去重后): {len(herb_compounds)}")

# ============================================================
# 步骤7: 生成最终候选池
# ============================================================
print("\n7. 生成最终候选池...")

# 合并 TCMSP + HERB
final_compounds = []

# 添加 TCMSP 来源
for c in tcmsp_compounds:
    final_compounds.append({
        "source": "TCMSP",
        "MOL_ID": c.get("MOL_ID", ""),
        "molecule_name": c.get("molecule_name", ""),
        "SMILES_std": c.get("SMILES_std", ""),
        "herb_cn_name": c.get("herb_cn_name", ""),
        "OB_score": c.get("ob", c.get("OB_score", "")),
        "Drug_likeness": c.get("dl", c.get("Drug_likeness", "")),
        "MW_HERB": c.get("mw", c.get("MW_HERB", "")),
        "TCMSP_id": c.get("MOL_ID", ""),
        "Ingredient_id": "",
        "Ingredient_name": "",
        "PubChem_id": "",
    })

# 添加 HERB 来源
for c in herb_compounds:
    final_compounds.append({
        "source": "HERB",
        "MOL_ID": "",
        "molecule_name": c["Ingredient_name"],
        "SMILES_std": c["SMILES_raw"],
        "herb_cn_name": "",
        "OB_score": c["OB_score"],
        "Drug_likeness": c["Drug_likeness"],
        "MW_HERB": c["MW_HERB"],
        "TCMSP_id": c["TCMSP_id"],
        "Ingredient_id": c["Ingredient_id"],
        "Ingredient_name": c["Ingredient_name"],
        "PubChem_id": c["PubChem_id"],
    })

print(f"   总候选化合物: {len(final_compounds)}")
print(f"     TCMSP: {len(tcmsp_compounds)}")
print(f"     HERB:  {len(herb_compounds)}")

# 最终 SMILES 去重
smiles_seen = set()
deduped = []
for c in final_compounds:
    smi = c["SMILES_std"].strip()
    if smi and smi not in smiles_seen:
        smiles_seen.add(smi)
        deduped.append(c)

print(f"   去重后: {len(deduped)}")

# 保存
fieldnames = [
    "source", "MOL_ID", "molecule_name", "SMILES_std", "herb_cn_name",
    "OB_score", "Drug_likeness", "MW_HERB", "TCMSP_id",
    "Ingredient_id", "Ingredient_name", "PubChem_id",
]
out_path = os.path.join(OUT_DIR, "zhuangyao_only_pool.csv")
with open(out_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(deduped)
print(f"   保存: {out_path}")

# 保存 Ingredient_id 列表
ing_id_path = os.path.join(OUT_DIR, "zhuangyao_herb_ingredient_ids.csv")
with open(ing_id_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Ingredient_id"])
    for ing_id in sorted(all_zy_ing_ids):
        writer.writerow([ing_id])

# 保存匹配详情
match_path = os.path.join(OUT_DIR, "zhuangyao_herb_match_detail.csv")
with open(match_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["cn_name", "source", "ingredient_count"])
    for zy_name in sorted(matched_zy):
        ing_count = 0
        if zy_name in tcm_to_ings:
            ing_count = len(tcm_to_ings[zy_name])
        writer.writerow([zy_name, "ingredient_all", ing_count])

# ============================================================
# 步骤8: 统计报告
# ============================================================
print(f"\n{'='*70}")
print("统计报告")
print(f"{'='*70}")
print(f"壮药总数: {len(zy_names)}")
print(f"匹配壮药: {len(matched_zy)}")
print(f"未匹配壮药: {len(unmatched)}")
print(f"壮药专属 Ingredient_id: {len(all_zy_ing_ids)}")
print(f"壮药专属候选池: {len(deduped)} 个化合物")

# 艾叶专项
if "艾叶" in tcm_to_ings:
    aiye_ings = tcm_to_ings["艾叶"]
    aiye_count = sum(1 for c in deduped if c.get("Ingredient_id", "") in aiye_ings)
    print(f"\n艾叶专项:")
    print(f"  艾叶成分数: {len(aiye_ings)}")
    print(f"  艾叶在候选池中: {aiye_count} 个")
    for c in deduped:
        if c.get("Ingredient_id", "") in aiye_ings:
            print(f"    {c['molecule_name'][:60]} | ID={c.get('Ingredient_id','?')} | source={c['source']}")

print("\n完成!")