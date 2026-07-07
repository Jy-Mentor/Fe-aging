import logging
logger = logging.getLogger(__name__)

import requests
import json

url = "https://bindingdb.org/axis2/services/BDBService/getLigandsByUniprotID"
params = {"uniprot": "P27487", "affinity_type": "IC50", "cutoff": 10000, "response": "json"}
resp = requests.get(url, params=params, timeout=30)
print(f"Status: {resp.status_code}")
print(f"Content-Type: {resp.headers.get('content-type','')}")
print(f"Length: {len(resp.text)}")

# 打印前3000字符
text = resp.text[:3000]
print(text)

# 尝试解析
try:
    data = resp.json()
    print(f"\nData type: {type(data).__name__}")
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        for k in list(data.keys())[:5]:
            v = data[k]
            if isinstance(v, str) and len(v) > 200:
                print(f"  {k}: {v[:200]}...")
            elif isinstance(v, list):
                print(f"  {k}: list of {len(v)}")
            elif isinstance(v, dict):
                print(f"  {k}: dict with keys {list(v.keys())[:5]}")
            else:
                print(f"  {k}: {v}")
    elif isinstance(data, list):
        print(f"List length: {len(data)}")
        if data:
            print(f"First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else type(data[0])}")
except Exception as e:
    print(f"Parse error: {e}")