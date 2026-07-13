import re
import zipfile
from xml.etree import ElementTree

doc_path = r'D:\铁衰老 绝不重蹈覆辙\标书_国自然标准_final_v7_桂艾BCP靶向Nrf2抑制铁依赖性SIPS改善CIRI.docx'

ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

def extract_text_from_docx(path):
    texts = []
    with zipfile.ZipFile(path) as z:
        xml_content = z.read('word/document.xml').decode('utf-8')
        root = ElementTree.fromstring(xml_content)
        for p in root.findall('.//w:p', ns):
            para_texts = []
            for t in p.findall('.//w:t', ns):
                if t.text:
                    para_texts.append(t.text)
            texts.append(''.join(para_texts))
    return '\n'.join(texts)

all_text = extract_text_from_docx(doc_path)

sec1_start = all_text.find('一、立项依据与研究内容')
sec2_start = all_text.find('二、研究目标')
ref_start = all_text.find('参考文献')

section1_full = all_text[sec1_start:sec2_start]
section1_no_ref = all_text[sec1_start:ref_start]
rest_section = all_text[sec2_start:]

chinese_chars = re.findall(r'[\u4e00-\u9fa5]', all_text)
total_chinese = len(chinese_chars)

sec1_chinese = len(re.findall(r'[\u4e00-\u9fa5]', section1_full))
sec1_no_ref_chinese = len(re.findall(r'[\u4e00-\u9fa5]', section1_no_ref))
rest_chinese = len(re.findall(r'[\u4e00-\u9fa5]', rest_section))
ref_chinese = sec1_chinese - sec1_no_ref_chinese

total_no_ref = sec1_no_ref_chinese + rest_chinese

print('=== 关键位置 ===')
print(f'一、立项依据: {sec1_start}')
print(f'参考文献: {ref_start}')
print(f'二、研究目标: {sec2_start}')

print('\n=== 字数统计（中文字符）===')
print(f'立项依据板块（含参考文献）: {sec1_chinese}')
print(f'立项依据板块（不含参考文献）: {sec1_no_ref_chinese}')
print(f'参考文献部分中文字数: {ref_chinese}')
print(f'立项之后部分（二~九）: {rest_chinese}')
print(f'全文总中文字数（含参考文献）: {total_chinese}')
print(f'全文总中文字数（不含参考文献）: {total_no_ref}')

print(f'\n立项依据<=8000字限制: {"符合" if sec1_no_ref_chinese <= 8000 else "超出 " + str(sec1_no_ref_chinese - 8000) + " 字"}')
print(f'总字数<=12000字限制: {"符合" if total_no_ref <= 12000 else "超出 " + str(total_no_ref - 12000) + " 字"}')
