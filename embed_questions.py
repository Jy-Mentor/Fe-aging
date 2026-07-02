import json
import re

with open(r'd:\铁衰老 绝不重蹈覆辙\questions.json', 'r', encoding='utf-8') as f:
    questions = json.load(f)

with open(r'd:\铁衰老 绝不重蹈覆辙\quiz.html', 'r', encoding='utf-8') as f:
    html = f.read()

questions_json = json.dumps(questions, ensure_ascii=False)

# Replace the old questions array with the new one
old_pattern = r'const questions = \[.*?\];'
new_replacement = f'const questions = {questions_json};'
html = re.sub(old_pattern, new_replacement, html, count=1, flags=re.DOTALL)

# Update the array size
html = html.replace('new Array(306).fill(null)', f'new Array({len(questions)}).fill(null)')

with open(r'd:\铁衰老 绝不重蹈覆辙\quiz.html', 'w', encoding='utf-8') as f:
    f.write(html)

print(f'题目数据已嵌入HTML文件，共 {len(questions)} 题')
print(f'验证: const questions = [{html[html.index("const questions = [")+20:html.index("const questions = [")+50]}...')