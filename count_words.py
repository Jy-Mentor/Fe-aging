import zipfile
import xml.etree.ElementTree as ET
import re

file_path = r'D:\铁衰老 绝不重蹈覆辙\标书_最终版_广西道地壮药桂艾BCP靶向Nrf2抑制铁依赖性SIPS改善CIRI.docx'

with zipfile.ZipFile(file_path, 'r') as z:
    with z.open('word/document.xml') as f:
        content = f.read().decode('utf-8')

# 提取所有文本
text = re.sub(r'<[^>]+>', '', content)
text = re.sub(r'\s+', '', text)

# 统计中文字符
chinese_chars = re.findall(r'[\u4e00-\u9fa5]', text)
total_chinese = len(chinese_chars)

# 找到各部分位置
sec1_start = text.find('一、立项依据与研究内容')
sec2_start = text.find('二、研究目标')
ref_start = text.find('参考文献')

section1_full = text[sec1_start:sec2_start]
section1_no_ref = text[sec1_start:ref_start]

# 立项依据中文字数（不含参考文献）
sec1_chinese = len(re.findall(r'[\u4e00-\u9fa5]', section1_no_ref))
sec1_full_chinese = len(re.findall(r'[\u4e00-\u9fa5]', section1_full))

# 全文不含参考文献
before_ref = text[:ref_start]
after_sec1 = text[sec2_start:]
total_no_ref = len(re.findall(r'[\u4e00-\u9fa5]', before_ref)) + len(re.findall(r'[\u4e00-\u9fa5]', after_sec1))

ref_text = text[ref_start:sec2_start]
ref_chinese = len(re.findall(r'[\u4e00-\u9fa5]', ref_text))

print('=== 字数统计（中文字符）===')
print(f'立项依据板块（一）中文字数（不含参考文献）: {sec1_chinese}')
print(f'立项依据板块（一）中文字数（含参考文献）: {sec1_full_chinese}')
print(f'参考文献部分中文字数: {ref_chinese}')
print(f'全文总中文字数（含参考文献）: {total_chinese}')
print(f'全文总中文字数（不含参考文献）: {total_no_ref}')
print()
print(f'立项依据≤8000字限制: {"符合" if sec1_chinese <= 8000 else "超出 " + str(sec1_chinese - 8000) + " 字"}')
print(f'总字数≤12000字限制: {"符合" if total_no_ref <= 12000 else "超出 " + str(total_no_ref - 12000) + " 字"}')
