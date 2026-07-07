import logging
logger = logging.getLogger(__name__)

import requests

# The ByUniProtids.jsp page is 11MB - let's look at the structure
# First, let's try to get a specific UniProt entry
url = "https://bindingdb.org/rwd/bind/ByUniProtids.jsp?uniprot=P27487"
resp = requests.get(url, timeout=30)
print(f"ByUniProtids with uniprot param: {resp.status_code}, Length: {len(resp.text)}")
print(resp.text[:3000])

# Also try the TSV download with range header to see format
url2 = "https://bindingdb.org/rwd/bind/downloads/BindingDB_All.tsv"
headers = {"Range": "bytes=0-5000"}
resp2 = requests.get(url2, headers=headers, timeout=30)
print(f"\nTSV Range: {resp2.status_code}")
if resp2.status_code in [200, 206]:
    print(resp2.text[:2000])