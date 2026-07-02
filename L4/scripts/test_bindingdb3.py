import requests

# Try WSDL
print("=== Testing WSDL ===")
url = "https://bindingdb.org/axis2/services/BDBService?wsdl"
resp = requests.get(url, timeout=30)
print(f"WSDL Status: {resp.status_code}, Length: {len(resp.text)}")

# Try www subdomain
url2 = "https://www.bindingdb.org/axis2/services/BDBService?wsdl"
resp2 = requests.get(url2, timeout=30)
print(f"www WSDL Status: {resp2.status_code}, Length: {len(resp2.text)}")

# Try the ByUniProtids.jsp page
print("\n=== Testing ByUniProtids.jsp ===")
url3 = "https://bindingdb.org/bind/ByUniProtids.jsp"
resp3 = requests.get(url3, timeout=30)
print(f"Status: {resp3.status_code}, Length: {len(resp3.text)}")

# Try main page
print("\n=== Testing main page ===")
url4 = "https://bindingdb.org/"
resp4 = requests.get(url4, timeout=30)
print(f"Status: {resp4.status_code}, Length: {len(resp4.text)}")

# Try to get TSV data
print("\n=== Testing TSV download ===")
url5 = "https://bindingdb.org/bind/downloads/BindingDB_All.tsv"
# Just check headers
resp5 = requests.head(url5, timeout=30)
print(f"Status: {resp5.status_code}, Headers: {dict(resp5.headers)}")