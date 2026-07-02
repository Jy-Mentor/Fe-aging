import zipfile
import xml.etree.ElementTree as ET
import re
import json
from collections import Counter

# ====== 第一部分：从PPT提取70道原题 ======
ppt_path = r'C:\Users\Jy-Mentor-7\Desktop\马原理复习.pptx'
ns_a = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main', 'p': 'http://schemas.openxmlformats.org/presentationml/2006/main'}

all_questions = []

with zipfile.ZipFile(ppt_path, 'r') as z:
    slide_files = [f for f in z.namelist() if re.match(r'ppt/slides/slide\d+\.xml$', f)]
    slide_files.sort(key=lambda x: int(re.search(r'slide(\d+)\.xml', x).group(1)))
    
    for slide_file in slide_files:
        slide_num = int(re.search(r'slide(\d+)\.xml', slide_file).group(1))
        if slide_num == 1 or slide_num == 16:
            continue
        
        content = z.read(slide_file)
        root = ET.fromstring(content)
        
        lines = []
        shapes = root.findall('.//p:sp', ns_a)
        for shape in shapes:
            tx_bodies = shape.findall('.//p:txBody', ns_a)
            for tb in tx_bodies:
                paras = tb.findall('a:p', ns_a)
                for para in paras:
                    runs = para.findall('a:r', ns_a)
                    text = ''.join(
                        r.find('a:t', ns_a).text 
                        for r in runs 
                        if r.find('a:t', ns_a) is not None and r.find('a:t', ns_a).text
                    )
                    lines.append(text.strip())
        
        non_empty_lines = [l for l in lines if l.strip()]
        if len(non_empty_lines) < 2:
            continue
        
        answer_line = non_empty_lines[-1]
        if not re.match(r'^[A-D]{5}$', answer_line):
            continue
        
        answers = answer_line
        content_lines = non_empty_lines[:-1]
        
        questions = []
        current_question = None
        current_options = {}
        
        for line in content_lines:
            has_option = re.search(r'[A-D][．.]\s*', line)
            if has_option:
                opt_pattern = r'([A-D])[．.]\s*'
                opt_positions = [(m.start(), m.group(1)) for m in re.finditer(opt_pattern, line)]
                for i, (pos, letter) in enumerate(opt_positions):
                    start = pos + len(letter) + 1
                    if i + 1 < len(opt_positions):
                        end = opt_positions[i+1][0]
                    else:
                        end = len(line)
                    opt_text = line[start:end].strip()
                    if opt_text:
                        current_options[letter] = opt_text
            else:
                if current_question is not None and current_options:
                    questions.append({
                        'question': current_question,
                        'options': current_options,
                        'chapter': '综合',
                        'source': '复习PPT'
                    })
                    current_options = {}
                current_question = line
        
        if current_question is not None and current_options:
            questions.append({
                'question': current_question,
                'options': current_options,
                'chapter': '综合',
                'source': '复习PPT'
            })
        
        for i, q in enumerate(questions):
            if i < len(answers):
                q['answer'] = answers[i]
                all_questions.append(q)

print(f"PPT提取: {len(all_questions)} 题")

# ====== 第二部分：从docx提取章节题 ======
docx_files = [
    (r"D:\微信聊天记录\xwechat_files\wxid_1gtfoi7l8op622_8a67\msg\attach\416af4d18df61956691bee3bfd3e0e6a\2026-06\Rec\f92ecdaec578883c\F\1\第一章单选题.docx", "第一章"),
    (r"D:\微信聊天记录\xwechat_files\wxid_1gtfoi7l8op622_8a67\msg\attach\416af4d18df61956691bee3bfd3e0e6a\2026-06\Rec\f92ecdaec578883c\F\3\第二章单选题.docx", "第二章"),
    (r"D:\微信聊天记录\xwechat_files\wxid_1gtfoi7l8op622_8a67\msg\attach\416af4d18df61956691bee3bfd3e0e6a\2026-06\Rec\f92ecdaec578883c\F\2\第三章单选题.docx", "第三章"),
]

ns_w = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

for filepath, chapter in docx_files:
    try:
        with zipfile.ZipFile(filepath, 'r') as z:
            content = z.read('word/document.xml')
            root = ET.fromstring(content)
            
            paragraphs = []
            for p in root.findall('.//w:p', ns_w):
                texts = []
                for t in p.findall('.//w:t', ns_w):
                    if t.text:
                        texts.append(t.text)
                text = ''.join(texts).strip()
                if text:
                    paragraphs.append(text)
            
            q_count = 0
            i = 0
            while i < len(paragraphs):
                line = paragraphs[i]
                q_match = re.match(r'^(\d+)\.\s*(.+)', line)
                if q_match:
                    q_text = q_match.group(2).strip()
                    
                    options = {}
                    for j in range(4):
                        opt_idx = i + 1 + j
                        if opt_idx >= len(paragraphs):
                            break
                        opt_line = paragraphs[opt_idx]
                        opt_match = re.match(r'^([A-D])\s*(.+)', opt_line)
                        if opt_match:
                            options[opt_match.group(1)] = opt_match.group(2).strip()
                    
                    ans_idx = i + 5
                    if ans_idx < len(paragraphs):
                        ans_line = paragraphs[ans_idx]
                        ans_match = re.match(r'^答案\s*([A-D])', ans_line)
                        if ans_match and len(options) == 4:
                            all_questions.append({
                                'question': q_text,
                                'options': options,
                                'answer': ans_match.group(1),
                                'chapter': chapter,
                                'source': '章节题库'
                            })
                            q_count += 1
                    i += 6
                else:
                    i += 1
            print(f"{chapter}: {q_count} 题")
    except Exception as e:
        print(f"解析{chapter}出错: {e}")

print(f"\n=== 总计: {len(all_questions)} 题 ===")
chapter_counts = Counter(q['chapter'] for q in all_questions)
for ch, cnt in sorted(chapter_counts.items()):
    print(f"  {ch}: {cnt} 题")

with open(r'd:\铁衰老 绝不重蹈覆辙\questions.json', 'w', encoding='utf-8') as f:
    json.dump(all_questions, f, ensure_ascii=False, indent=2)

print("\n已保存到 questions.json")