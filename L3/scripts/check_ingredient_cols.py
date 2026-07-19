"""检查 ingredient_all.xlsx 完整列结构"""
import zipfile, re, xml.etree.ElementTree as ET

ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
z = zipfile.ZipFile(r"D:\下载\ingredient_all.xlsx")

strings = []
with z.open("xl/sharedStrings.xml") as f:
    ss_root = ET.fromstring(f.read().decode("utf-8"))
for si in ss_root.findall(".//s:si", ns):
    t = si.find(".//s:t", ns)
    strings.append(t.text if t is not None and t.text else "")

with z.open("xl/worksheets/sheet1.xml") as f:
    sheet_root = ET.fromstring(f.read().decode("utf-8"))

rows = list(sheet_root.findall(".//s:row", ns))
print(f"Total rows: {len(rows)}")

# 解析第一行(header)
cells = {}
for c in rows[0].findall("s:c", ns):
    ref = c.get("r")
    if not ref:
        continue
    col_letter = re.match(r"([A-Z]+)", ref).group(1)
    if len(col_letter) == 1:
        col_idx = ord(col_letter) - ord("A")
    else:
        col_idx = 26 * (len(col_letter) - 1) + ord(col_letter[-1]) - ord("A")
    t = c.get("t")
    v = c.find("s:v", ns)
    val = v.text if v is not None else ""
    if t == "s" and val:
        idx = int(val)
        if 0 <= idx < len(strings):
            val = strings[idx]
    cells[col_idx] = val

print(f"Header columns:")
for k, v in sorted(cells.items()):
    print(f"  [{k}] {v}")

# 检查是否有 Herb_id 相关列
has_herb_id = any("herb" in str(v).lower() for v in cells.values())
print(f"\nHas Herb_id column: {has_herb_id}")

# 检查几行数据中 Herb_id 相关列的值
herb_id_cols = [k for k, v in cells.items() if "herb" in str(v).lower() or "herb_id" in str(v).lower()]
print(f"Herb-related cols: {herb_id_cols}")

z.close()