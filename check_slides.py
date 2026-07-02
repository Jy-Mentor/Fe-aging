import zipfile
import xml.etree.ElementTree as ET
import re

ppt_path = r'C:\Users\Jy-Mentor-7\Desktop\马原理复习.pptx'
ns = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}

with zipfile.ZipFile(ppt_path, 'r') as z:
    slide_files = [f for f in z.namelist() if re.match(r'ppt/slides/slide\d+\.xml$', f)]
    slide_files.sort(key=lambda x: int(re.search(r'slide(\d+)\.xml', x).group(1)))
    
    for slide_file in slide_files:
        slide_num = int(re.search(r'slide(\d+)\.xml', slide_file).group(1))
        if slide_num < 5 or slide_num > 7:
            continue
        content = z.read(slide_file)
        root = ET.fromstring(content)
        print(f'=== Slide {slide_num} ===')
        for t_elem in root.findall('.//a:t', ns):
            if t_elem.text is not None:
                print(repr(t_elem.text))
        print()
