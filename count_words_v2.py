import zipfile
import xml.etree.ElementTree as ET
import re

file_path = r'D:\铁衰老 绝不重蹈覆辙\标书_最终版_广西道地壮药桂艾BCP靶向Nrf2抑制铁依赖性SIPS改善CIRI.docx'

with zipfile.ZipFile(file_path, 'r') as z:
    with z.open('word/document.xml') as f:
        content = f.read().decode('utf-8')

text = re.sub(r'<[^>]+>', '', content)
text = re.sub(r'\s+', '', text)

chinese_chars = re.findall(r'[\u4e00-\u9fa5]', text)
total_chinese = len(chinese_chars)

sec1_start = text.find('一、立项依据与研究内容')
sec2_start = text.find('二、研究目标')
ref_start = text.find('参考文献')

print('=== 关键位置 ===')
print(f'一、立项依据: {sec1_start}')
print(f'参考文献: {ref_start}')
print(f'二、研究目标: {sec2_start}')
print()

section1_all = text[sec1_start:sec2_start]
section1_no_ref = text[sec1_start:ref_start]
ref_section = text[ref_start:sec2_start]
after_sec1 = text[sec2_start:]

sec1_all_chinese = len(re.findall(r'[\u4e00-\u9fa5]', section1_all))
sec1_no_ref_chinese = len(re.findall(r'[\u4e00-\u9fa5]', section1_no_ref))
ref_chinese = len(re.findall(r'[\u4e00-\u9fa5]', ref_section))
after_sec1_chinese = len(re.findall(r'[\u4e00-\u9fa5]', after_sec1))

total_no_ref = sec1_no_ref_chinese + after_sec1_chinese

print('=== 字数统计（中文字符）===')
print(f'立项依据板块（含参考文献）: {sec1_all_chinese}')
print(f'立项依据板块（不含参考文献）: {sec1_no_ref_chinese}')
print(f'参考文献部分中文字数: {ref_chinese}')
print(f'立项之后部分（二~九）: {after_sec1_chinese}')
print(f'全文总中文字数（含参考文献）: {total_chinese}')
print(f'全文总中文字数（不含参考文献）: {total_no_ref}')
print()
print(f'立项依据≤8000字限制: {"符合" if sec1_no_ref_chinese <= 8000 else "超出 " + str(sec1_no_ref_chinese - 8000) + " 字"}')
print(f'总字数≤12000字限制: {"符合" if total_no_ref <= 12000 else "超出 " + str(total_no_ref - 12000) + " 字"}')
