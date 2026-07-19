"""解析HERB herb_ingredient关联表，提取壮药专属成分
处理内联字符串（inlineStr）格式的 xlsx 文件。
"""
import csv
import os
import re
import zipfile
import xml.etree.ElementTree as ET

BASE = r"d:\铁衰老 绝不重蹈覆辙"
HERB_REL_PATH = r"D:\下载\herb_ingredient_2026_1_17.xlsx"
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

print("1. 解析HERB herb_ingredient关联表...")

z = zipfile.ZipFile(HERB_REL_PATH)

# 检查 sharedStrings 是否存在
has_shared = "xl/sharedStrings.xml" in z.namelist()
print(f"  有 sharedStrings: {has_shared}")

strings = []
if has_shared:
    with z.open("xl/sharedStrings.xml") as f:
        ss_xml = f.read().decode("utf-8")
    ss_root = ET.fromstring(ss_xml)
    for si in ss_root.findall(".//s:si", NS):
        t = si.find(".//s:t", NS)
        strings.append(t.text if t is not None and t.text else "")
    print(f"  Shared strings: {len(strings)}")


def get_cell_value(c, strings):
    """获取单元格值，支持 shared string 和 inline string"""
    t = c.get("t")
    # 先尝试 v 元素
    v = c.find("s:v", NS)
    if v is not None and v.text:
        val = v.text
        if t == "s":
            idx = int(val)
            if 0 <= idx < len(strings):
                return strings[idx]
            return val
        return val
    # 尝试 inlineStr
    inline = c.find("s:is", NS)
    if inline is not None:
        t_el = inline.find("s:t", NS)
        if t_el is not None and t_el.text:
            return t_el.text
    return ""


# 读取 sheet 数据
with z.open("xl/worksheets/sheet1.xml") as f:
    sheet_xml = f.read().decode("utf-8")
sheet_root = ET.fromstring(sheet_xml)

rows = []
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
        cells[col_idx] = get_cell_value(c, strings)
    if cells:
        rows.append(cells)

print(f"  数据行数: {len(rows)}")

# 2. 检查列结构
if rows:
    header = rows[0]
    print(f"  列: {dict(sorted(header.items()))}")
    print(f"  前5行:")
    for i, row in enumerate(rows[1:6]):
        vals = {k: str(row.get(k, ""))[:60] for k in sorted(row.keys())[:5]}
        print(f"    [{i+1}] {vals}")

# 3. 加载壮药列表
with open(os.path.join(BASE, "zhuangyao_data", "guangxi_zhuangyao_list.csv"), "r", encoding="utf-8-sig") as f:
    zy_list = list(csv.DictReader(f))
zy_names = set(x.get("cn_name", "").strip() for x in zy_list)
print(f"\n  壮药列表: {len(zy_names)} 种")

# 4. 自动识别列
herb_col = None
ing_col = None
for k, v in header.items():
    v_lower = str(v).lower().strip()
    if "herb" in v_lower and ("name" in v_lower or "cn" in v_lower):
        herb_col = k
    if "ingredient" in v_lower and "id" in v_lower:
        ing_col = k

# 回退猜测
if herb_col is None:
    for k in sorted(header.keys()):
        v_lower = str(header[k]).lower().strip()
        if "herb" in v_lower or "name" in v_lower:
            herb_col = k
            break
if ing_col is None:
    for k in sorted(header.keys()):
        v_lower = str(header[k]).lower().strip()
        if "ingredient" in v_lower or "ing" in v_lower:
            ing_col = k
            break

print(f"  herb_col={herb_col} ({header.get(herb_col, 'N/A')})")
print(f"  ing_col={ing_col} ({header.get(ing_col, 'N/A')})")

if herb_col is None or ing_col is None:
    print("ERROR: 无法识别列，请检查表头")
    print(f"  可用列: {dict(sorted(header.items()))}")
    z.close()
    exit(1)

# 5. 匹配壮药
zhuangyao_ingredients = set()
matched_herbs = set()
for row in rows[1:]:
    herb_name = str(row.get(herb_col, "")).strip()
    ing_id = str(row.get(ing_col, "")).strip()
    if herb_name in zy_names and ing_id:
        zhuangyao_ingredients.add(ing_id)
        matched_herbs.add(herb_name)

print(f"\n  壮药匹配成分: {len(zhuangyao_ingredients)} 个Ingredient_id")
print(f"  匹配壮药: {len(matched_herbs)}/{len(zy_names)} 种")

# 6. 缺失壮药
missing = zy_names - matched_herbs
print(f"  缺失壮药: {len(missing)} 种")
if missing:
    print(f"  前20: {sorted(missing)[:20]}")

# 7. 保存结果
out_path = os.path.join(BASE, "L3", "results", "zhuangyao_herb_ingredient_ids.csv")
with open(out_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Ingredient_id"])
    for ing_id in sorted(zhuangyao_ingredients):
        writer.writerow([ing_id])
print(f"\n  保存: {out_path}")

z.close()
print("\n完成!")