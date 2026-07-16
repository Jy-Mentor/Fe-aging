"""验证 nature-skill 和 medical-mechanism-diagram 输出图像规格"""
from PIL import Image
import os

OUTDIR = r"d:/铁衰老 绝不重蹈覆辙/figures/skill_test"

specs = [
    ("nature_test_umap_violin.png", "nature-skill R", 183, 120, 300),
    ("BCP_mechanism_test.png", "medical-mechanism-diagram Python", None, None, 300),
]

print("=" * 70)
print("输出质量验证报告")
print("=" * 70)
all_ok = True
for fname, skill, w_mm, h_mm, min_dpi in specs:
    p = os.path.join(OUTDIR, fname)
    if not os.path.exists(p):
        print(f"[FAIL] {fname}: 文件不存在")
        all_ok = False
        continue
    img = Image.open(p)
    dpi = img.info.get("dpi", (None, None))
    width_px, height_px = img.size
    actual_dpi = dpi[0] if dpi and dpi[0] else None

    print(f"\n[{skill}] {fname}")
    print(f"  尺寸 (px): {width_px} x {height_px}")
    print(f"  DPI: {actual_dpi}")
    print(f"  文件大小: {os.path.getsize(p) / 1024:.1f} KB")

    # Verify DPI (allow 0.5 DPI floating-point tolerance)
    if actual_dpi and actual_dpi >= min_dpi - 0.5:
        print(f"  [OK] DPI ~ {min_dpi} (actual: {actual_dpi:.4f})")
    else:
        print(f"  [FAIL] DPI < {min_dpi} (actual: {actual_dpi})")
        all_ok = False

    # Verify expected dimensions if specified
    if w_mm and h_mm and actual_dpi:
        expected_w_px = int(w_mm / 25.4 * actual_dpi)
        expected_h_px = int(h_mm / 25.4 * actual_dpi)
        if abs(width_px - expected_w_px) <= 5 and abs(height_px - expected_h_px) <= 5:
            print(f"  [OK] 尺寸匹配 {w_mm}x{h_mm}mm @ {actual_dpi}DPI "
                  f"(预期 {expected_w_px}x{expected_h_px}px)")
        else:
            print(f"  [WARN] 尺寸偏差: 预期 {expected_w_px}x{expected_h_px}px, "
                  f"实际 {width_px}x{height_px}px")

# Check PDF files exist
print("\n" + "-" * 70)
print("PDF 矢量输出检查:")
for fname in ["nature_test_umap_violin.pdf", "BCP_mechanism_test.pdf"]:
    p = os.path.join(OUTDIR, fname)
    if os.path.exists(p):
        print(f"  [OK] {fname} ({os.path.getsize(p)/1024:.1f} KB)")
    else:
        print(f"  [FAIL] {fname}: 文件不存在")
        all_ok = False

print("\n" + "=" * 70)
print(f"总体验证: {'全部通过' if all_ok else '存在失败项'}")
print("=" * 70)
