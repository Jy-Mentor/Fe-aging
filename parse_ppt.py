import zipfile
import xml.etree.ElementTree as ET
import re
import json

ppt_path = r'C:\Users\Jy-Mentor-7\Desktop\马原理复习.pptx'
ns = {
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'p': 'http://schemas.openxmlformats.org/presentationml/2006/main'
}

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
        shapes = root.findall('.//p:sp', ns)
        for shape in shapes:
            tx_bodies = shape.findall('.//p:txBody', ns)
            for tb in tx_bodies:
                paras = tb.findall('a:p', ns)
                for para in paras:
                    runs = para.findall('a:r', ns)
                    text = ''.join(
                        r.find('a:t', ns).text 
                        for r in runs 
                        if r.find('a:t', ns) is not None and r.find('a:t', ns).text
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
                        'options': current_options
                    })
                    current_options = {}
                current_question = line
        
        if current_question is not None and current_options:
            questions.append({
                'question': current_question,
                'options': current_options
            })
        
        for i, q in enumerate(questions):
            if i < len(answers):
                q['answer'] = answers[i]
                all_questions.append(q)

print(f'解析出 {len(all_questions)} 道题\n')

errors = []
for i, q in enumerate(all_questions):
    if not q.get('question'):
        errors.append(f'第{i+1}题: 题干为空')
    if len(q.get('options', {})) != 4:
        errors.append(f'第{i+1}题: 选项数量={len(q.get("options", {}))}')
    if q.get('answer', '') not in ['A', 'B', 'C', 'D']:
        errors.append(f'第{i+1}题: 答案异常={q.get("answer")}')

if errors:
    print(f'发现 {len(errors)} 个问题:')
    for e in errors:
        print(f'  {e}')
else:
    print('所有题目解析成功！')

print('\n=== 第1-5题 ===')
for i, q in enumerate(all_questions[:5]):
    print(f'\n第{i+1}题: {q["question"]}')
    for k in sorted(q['options'].keys()):
        print(f'  {k}. {q["options"][k]}')
    print(f'  答案: {q["answer"]}')

print('\n=== 第21-23题 ===')
for i, q in enumerate(all_questions[20:23]):
    print(f'\n第{i+21}题: {q["question"]}')
    for k in sorted(q['options'].keys()):
        print(f'  {k}. {q["options"][k]}')
    print(f'  答案: {q["answer"]}')

print('\n=== 第67-70题 ===')
for i, q in enumerate(all_questions[-4:]):
    print(f'\n第{len(all_questions)-4+i+1}题: {q["question"]}')
    for k in sorted(q['options'].keys()):
        print(f'  {k}. {q["options"][k]}')
    print(f'  答案: {q["answer"]}')

with open(r'd:\铁衰老 绝不重蹈覆辙\questions.json', 'w', encoding='utf-8') as f:
    json.dump(all_questions, f, ensure_ascii=False, indent=2)

print(f'\n已保存到 questions.json')
