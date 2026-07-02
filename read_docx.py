import zipfile
import xml.etree.ElementTree as ET
import re

files = [
    (r"D:\微信聊天记录\xwechat_files\wxid_1gtfoi7l8op622_8a67\msg\attach\416af4d18df61956691bee3bfd3e0e6a\2026-06\Rec\f92ecdaec578883c\F\1\第一章单选题.docx", "第一章"),
    (r"D:\微信聊天记录\xwechat_files\wxid_1gtfoi7l8op622_8a67\msg\attach\416af4d18df61956691bee3bfd3e0e6a\2026-06\Rec\f92ecdaec578883c\F\3\第二章单选题.docx", "第二章"),
    (r"D:\微信聊天记录\xwechat_files\wxid_1gtfoi7l8op622_8a67\msg\attach\416af4d18df61956691bee3bfd3e0e6a\2026-06\Rec\f92ecdaec578883c\F\2\第三章单选题.docx", "第三章"),
]

ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

for filepath, chapter in files:
    print(f"\n{'='*60}")
    print(f"{chapter}: {filepath}")
    print(f"{'='*60}")
    try:
        with zipfile.ZipFile(filepath, 'r') as z:
            # print file list
            print("文件列表:")
            for f in z.namelist()[:5]:
                print(f"  {f}")
            
            # Read document.xml
            content = z.read('word/document.xml')
            root = ET.fromstring(content)
            
            paragraphs = []
            for p in root.findall('.//w:p', ns):
                texts = []
                for t in p.findall('.//w:t', ns):
                    if t.text:
                        texts.append(t.text)
                text = ''.join(texts).strip()
                if text:
                    paragraphs.append(text)
            
            print(f"总段落数: {len(paragraphs)}")
            for i, p in enumerate(paragraphs[:30]):
                print(f"  [{i}] {repr(p[:120])}")
            if len(paragraphs) > 30:
                print("  ...")
                for i, p in enumerate(paragraphs[-10:]):
                    print(f"  [{len(paragraphs)-10+i}] {repr(p[:120])}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()