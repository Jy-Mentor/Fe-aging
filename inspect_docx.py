import zipfile, re, sys

z = zipfile.ZipFile(r'D:\铁衰老 绝不重蹈覆辙\标书_终版_v13_含图表_fixed.docx')
doc = z.read('word/document.xml').decode('utf-8')

img_refs = re.findall(r'r:embed="(rId\d+)"', doc)
print(f'Image embed references in document.xml: {len(img_refs)}')
print(img_refs)

idx = doc.find('<w:drawing')
print('\nFirst drawing XML snippet:')
print(doc[idx:idx+2000])

# Check for each image relationship exists
rels = z.read('word/_rels/document.xml.rels').decode('utf-8')
for rid in img_refs:
    exists = rid in rels
    print(f'{rid} in rels: {exists}')
