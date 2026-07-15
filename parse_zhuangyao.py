"""解析 all_entries.md, 输出壮药名录 CSV/JSON。支持 markdown 和纯文本两种格式。"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

MD_RE = re.compile(
    r'\[\s*([^/\]]+?)(?:/([^\]]+?))?\s*\]'
    r'\(https://db2\.ouryao\.com/dfbz/view\.php\?bookid=gxzyzlbz&idx=(\d+)\)'
    r'第([一二三])卷[（(](\d{4})年版[)）]'
)
TXT_RE = re.compile(
    r'^([^\n/]+?)(?:/([^\n]+?))?\s*\n\s*第([一二三])卷[（(](\d{4})年版[)）]',
    re.MULTILINE,
)
VOL_MAP = {"一": 1, "二": 2, "三": 3}


def parse_text(text):
    entries = []
    md_spans = []
    for m in MD_RE.finditer(text):
        md_spans.append((m.start(), m.end()))
        entries.append({
            "_pos": m.start(),
            "idx": int(m.group(3)),
            "cn_name": m.group(1).strip(),
            "zhuang_name": (m.group(2) or "").strip(),
            "volume": VOL_MAP[m.group(4)],
            "year": int(m.group(5)),
        })
    for m in TXT_RE.finditer(text):
        if any(s <= m.start() < e for s, e in md_spans):
            continue
        entries.append({
            "_pos": m.start(),
            "idx": None,
            "cn_name": m.group(1).strip(),
            "zhuang_name": (m.group(2) or "").strip(),
            "volume": VOL_MAP[m.group(3)],
            "year": int(m.group(4)),
        })
    entries.sort(key=lambda e: e["_pos"])
    for e in entries:
        del e["_pos"]
    return entries


def main():
    data_file = Path(__file__).resolve().parent / "zhuangyao_data" / "all_entries.md"
    if not data_file.exists():
        print(f"[错误] 文件不存在: {data_file}")
        return
    text = data_file.read_text(encoding="utf-8", errors="replace")
    entries = parse_text(text)
    print(f"[解析] 共匹配 {len(entries)} 条")

    no_idx_count = sum(1 for e in entries if e["idx"] is None)
    if no_idx_count:
        print(f"[推断] {no_idx_count} 条无idx, 按文件位置顺序填补")
        next_idx = 1
        for e in entries:
            if e["idx"] is None:
                e["idx"] = next_idx
                next_idx += 1
            else:
                next_idx = e["idx"] + 1

    seen = {}
    for e in entries:
        seen[e["idx"]] = e
    ordered = [seen[k] for k in sorted(seen.keys())]

    vol_c = Counter(e["volume"] for e in ordered)
    yr_c = Counter(e["year"] for e in ordered)
    print("\n===== 统计 =====")
    print(f"总条目: {len(ordered)}")
    for v in sorted(vol_c):
        print(f"  第{v}卷: {vol_c[v]} 条")
    for y in sorted(yr_c):
        print(f"  {y}年版: {yr_c[y]} 条")
    print(f"官方总数: 490 (含第三卷2018)")
    print(f"差异: {len(ordered)-490:+d}")

    out_dir = data_file.parent
    csv_path = out_dir / "guangxi_zhuangyao_list.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx","cn_name","zhuang_name","volume","year"])
        w.writeheader()
        w.writerows(ordered)
    print(f"\nCSV -> {csv_path}")

    json_path = out_dir / "guangxi_zhuangyao_list.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)
    print(f"JSON -> {json_path}")

    print("\n===== 前5条 =====")
    for e in ordered[:5]:
        print(f"  idx={e['idx']:3d} | 第{e['volume']}卷({e['year']}) | {e['cn_name']} / {e['zhuang_name']}")
    print("===== 末5条 =====")
    for e in ordered[-5:]:
        print(f"  idx={e['idx']:3d} | 第{e['volume']}卷({e['year']}) | {e['cn_name']} / {e['zhuang_name']}")

    max_idx = max(e["idx"] for e in ordered)
    missing = set(range(1, max_idx+1)) - set(e["idx"] for e in ordered)
    if missing:
        print(f"\n[警告] idx缺失: {sorted(missing)}")
    else:
        print(f"\n[OK] idx 1-{max_idx} 连续")

    name_c = Counter(e["cn_name"] for e in ordered)
    dups = {k:v for k,v in name_c.items() if v > 1}
    if dups:
        print(f"[警告] 重复药名: {dups}")


if __name__ == "__main__":
    main()
