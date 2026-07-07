import logging
logger = logging.getLogger(__name__)

import requests

# The main page redirects to /rwd/bind/index.jsp
# Try the API with the full path
url = "https://bindingdb.org/rwd/bind/axis2/services/BDBService?wsdl"
try:
    resp = requests.get(url, timeout=30)
    print(f"rwd/bind WSDL: {resp.status_code}, Length: {len(resp.text)}")
except Exception as e:
    print(f"rwd/bind WSDL error: {e}")

# Try the API on bdb99.ucsd.edu
url2 = "https://bdb99.ucsd.edu/axis2/services/BDBService?wsdl"
try:
    resp2 = requests.get(url2, timeout=30, verify=False)
    print(f"bdb99 WSDL: {resp2.status_code}, Length: {len(resp2.text)}")
except Exception as e:
    print(f"bdb99 WSDL error: {e}")

# Try the ByUniProtids download page
url3 = "https://bindingdb.org/rwd/bind/ByUniProtids.jsp"
try:
    resp3 = requests.get(url3, timeout=30)
    print(f"ByUniProtids: {resp3.status_code}, Length: {len(resp3.text)}")
    print(resp3.text[:1000])
except Exception as e:
    print(f"ByUniProtids error: {e}")

# Try TSV download
url4 = "https://bindingdb.org/rwd/bind/downloads/BindingDB_All.tsv"
try:
    resp4 = requests.head(url4, timeout=30)
    print(f"TSV download: {resp4.status_code}")
except Exception as e:
    print(f"TSV download error: {e}")