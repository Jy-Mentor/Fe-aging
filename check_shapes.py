import zipfile
import xml.etree.ElementTree as ET
import re
import json

ppt_path = r'C:\Users\Jy-Mentor-7\Desktop\马原理复习.pptx'
ns = {
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'p': 'http://schemas.openxmlformats.org/presentationml/2006/main'
}

with zipfile.ZipFile(ppt_path, 'r') as z:
    slide_files = [f for f in z.namelist() if re.match(r'ppt/slides/slide\d+\.xml$', f)]
    slide_files.sort(key=lambda x: int(re.search(r'slide(\d+)\.xml', x).group(1)))
    
    for slide_file in slide_files[:3]:
        slide_num = int(re.search(r'slide(\d+)\.xml', slide_file).group(1))
        content = z.read(slide_file)
        root = ET.fromstring(content)
        
        print(f'=== Slide {slide_num} ===')
        
        shapes = root.findall('.//p:sp', ns)
        for si, shape in enumerate(shapes):
            text_frames = shape.findall('.//p:txBody', ns)
            for tf in text_frames:
                paragraphs = tf.findall('a:p', ns)
                shape_texts = []
                for para in paragraphs:
                    runs = para.findall('a:r', ns)
                    para_text = ''.join(r.find('a:t', ns).text for r in runs if r.find('a:t', ns) is not None and r.find('a:t', ns).text)
                    if para_text.strip():
                        shape_texts.append(para_text.strip())
                if shape_texts:
                    print(f'  Shape {si}:')
                    for t in shape_texts:
                        print(f'    {repr(t)}')
        print()
