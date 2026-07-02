import requests
import json

# Test with www subdomain
url = "https://www.bindingdb.org/axis2/services/BDBService/getLigandsByUniprotID"
params = {"uniprot_id": "P27487", "affinity_type": "IC50", "affinity_cutoff": 10000, "response": "json"}
resp = requests.get(url, params=params, timeout=30)
print(f"Status: {resp.status_code}")
print(f"Content-Type: {resp.headers.get('content-type','')}")
print(f"Length: {len(resp.text)}")
print(resp.text[:3000])