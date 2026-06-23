"""提取所有参考论文PDF的文本内容，保存到单独文件"""
import pdfplumber, os, glob

pdf_dir = r'd:\铁衰老 绝不重蹈覆辙\参考论文'
out_dir = r'd:\铁衰老 绝不重蹈覆辙\paper_texts'
os.makedirs(out_dir, exist_ok=True)

pdfs = sorted(glob.glob(os.path.join(pdf_dir, '*.pdf')))

for pdf in pdfs:
    fname = os.path.basename(pdf)
    short_name = fname.replace('.pdf', '')[:40]
    out_path = os.path.join(out_dir, f'{short_name}.txt')
    
    print(f'Processing: {fname}')
    try:
        with pdfplumber.open(pdf) as p:
            full_text = ''
            for i, page in enumerate(p.pages):
                t = page.extract_text()
                if t:
                    full_text += f'\n--- Page {i+1} ---\n' + t
            
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            print(f'  Saved: {out_path} ({len(full_text)} chars, {len(p.pages)} pages)')
    except Exception as e:
        print(f'  ERROR: {e}')
        import traceback
        traceback.print_exc()

print('\nDone!')