"""检查各xlsx文件结构"""
import zipfile, re, xml.etree.ElementTree as ET

ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

def peek_xlsx(path, label):
    print(f"\n{'='*60}")
    print(f"文件: {label}")
    print(f"{'='*60}")
    z = zipfile.ZipFile(path)
    print(f"内部文件: {z.namelist()}")
    has_ss = "xl/sharedStrings.xml" in z.namelist()
    print(f"sharedStrings: {has_ss}")
    
    strings = []
    if has_ss:
        with z.open("xl/sharedStrings.xml") as f:
            ss_root = ET.fromstring(f.read().decode("utf-8"))
        for si in ss_root.findall(".//s:si", ns):
            t = si.find(".//s:t", ns)
            strings.append(t.text if t is not None and t.text else "")
    
    with z.open("xl/worksheets/sheet1.xml") as f:
        sheet_root = ET.fromstring(f.read().decode("utf-8"))
    
    rows = []
    for row_el in sheet_root.findall(".//s:row", ns):
        cells = {}
        for c in row_el.findall("s:c", ns):
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
            if t == "s" and val and int(val) < len(strings):
                val = strings[int(val)]
            cells[col_idx] = val
        if cells:
            rows.append(cells)
    
    print(f"行数: {len(rows)}")
    if rows:
        h = {k: str(v)[:80] for k, v in sorted(rows[0].items())}
        print(f"表头: {h}")
        for i in range(1, min(4, len(rows))):
            r = {k: str(v)[:80] for k, v in sorted(rows[i].items())}
            print(f"  [{i}] {r}")
    z.close()

peek_xlsx(r"D:\下载\herb_all.xlsx", "herb_all")
peek_xlsx(r"D:\下载\herb_ingredient_2026_1_17.xlsx", "herb_ingredient_2026_1_17")
peek_xlsx(r"D:\下载\herb_target_2026_1_17.xlsx", "herb_target_2026_1_17")
peek_xlsx(r"D:\下载\ingredient_all.xlsx", "ingredient_all")