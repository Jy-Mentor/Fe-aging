"""测试从TCMSP化合物详情页获取PubChem CID"""
import sys, os, re, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TCMSP-Spider', 'src'))
from tcmsp import TcmspSpider

t = TcmspSpider()
token = t.get_token()

# 测试几个化合物
test_mols = ["MOL000001", "MOL000422", "MOL000173", "MOL002288", "MOL001001"]

for mol_id in test_mols:
    url = f"https://www.tcmsp-e.com/tcmspsearch.php?qr={mol_id}&qsr=mol_id&token={token}"
    html = t.get_response(url)
    
    if not html:
        print(f"{mol_id}: FAILED to get response")
        continue
    
    # 找PubChem CID
    cids = re.findall(r'pubchem\.ncbi\.nlm\.nih\.gov/compound/(\d+)', html)
    
    # 找mol2下载链接
    mol2_links = re.findall(r'href="([^"]*\.mol2[^"]*)"', html)
    
    # 提取分子名称
    name_match = re.search(r'<h3[^>]*>([^<]+)</h3>', html)
    mol_name = name_match.group(1).strip() if name_match else "N/A"
    
    print(f"\n{mol_id} ({mol_name}):")
    print(f"  PubChem CID: {cids[0] if cids else 'NOT FOUND'}")
    print(f"  MOL2 links: {len(mol2_links)} found")
    if mol2_links:
        print(f"  First MOL2: ...{mol2_links[0][-50:]}")
    
    time.sleep(0.5)  # 礼貌爬取