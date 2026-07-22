#!/usr/bin/env python3
"""
将 标书_unpacked/ 重新打包成 .docx 文件
docx = ZIP archive with specific structure
"""
import zipfile
import os
from pathlib import Path

src_dir = Path(r"D:\铁衰老 绝不重蹈覆辙\标书_unpacked")
out_docx = Path(r"D:\铁衰老 绝不重蹈覆辙\标书_终版_v14_含拟时序方法学.docx")

# docx 文件结构: [Content_Types].xml 在最前, 然后是其他文件
# 按 _rels, docProps, word 的顺序, 但 [Content_Types].xml 必须第一个
files_to_zip = []

# 1. [Content_Types].xml 必须是第一个
ct_file = src_dir / "[Content_Types].xml"
if ct_file.exists():
    files_to_zip.append(("[Content_Types].xml", ct_file))

# 2. 递归添加所有其他文件
for root, dirs, files in os.walk(src_dir):
    for f in sorted(files):
        full = Path(root) / f
        rel = full.relative_to(src_dir).as_posix()
        if rel == "[Content_Types].xml":
            continue  # 已添加
        files_to_zip.append((rel, full))

# 写入 docx (使用 ZIP_DEFLATED 压缩)
with zipfile.ZipFile(out_docx, "w", zipfile.ZIP_DEFLATED) as zf:
    for arcname, filepath in files_to_zip:
        zf.write(filepath, arcname)

print(f"[DONE] 已创建: {out_docx}")
print(f"  文件数: {len(files_to_zip)}")
print(f"  大小: {out_docx.stat().st_size / 1024:.1f} KB")
