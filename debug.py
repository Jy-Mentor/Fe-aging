import zipfile
import xml.etree.ElementTree as ET
import re

ppt_path = r'C:\Users\Jy-Mentor-7\Desktop\马原理复习.pptx'
ns = {
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'p': 'http://schemas.openxmlformats.org/presentationml/2006/main'
}

with zipfile.ZipFile(ppt_path, 'r') as z:
    slide_file = 'ppt/slides/slide2.xml'
    content = z.read(slide_file)
    root = ET.fromstring(content)
    
    print('=== 查找所有文本 ===')
    all_ts = root.findall('.//a:t', ns)
    print(f'找到 {len(all_ts)} 个 a:t 元素')
    for t in all_ts:
        print(repr(t.text))
    
    print('\n=== 按shape分组 ===')
    shapes = root.findall('.//p:sp', ns)
    print(f'找到 {len(shapes)} 个 shape')
    for si, shape in enumerate(shapes):
        print(f'\nShape {si}:')
        tx_bodies = shape.findall('.//p:txBody', ns)
        print(f'  txBody数量: {len(tx_bodies)}')
        for ti, tb in enumerate(tx_bodies):
            paras = tb.findall('a:p', ns)
            print(f'  txBody {ti}: {len(paras)} 个段落')
            for pi, para in enumerate(paras):
                runs = para.findall('a:r', ns)
                text = ''.join(r.find('a:t', ns).text for r in runs if r.find('a:t', ns) is not None and r.find('a:t', ns).text)
                print(f'    段落{pi}: {repr(text)}')
