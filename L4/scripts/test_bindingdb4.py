import requests

# Check main page content
url = "https://bindingdb.org/"
resp = requests.get(url, timeout=30)
print(f"Main page ({resp.status_code}):")
print(resp.text[:2000])

# Try bdb subdomain
url2 = "https://bdb.bindingdb.org/"
resp2 = requests.get(url2, timeout=30)
print(f"\nbdb subdomain ({resp2.status_code}):")
print(resp2.text[:2000])

# Try the bdb99.ucsd.edu as seen in search results
url3 = "https://bdb99.ucsd.edu/"
resp3 = requests.get(url3, timeout=30)
print(f"\nbdb99.ucsd.edu ({resp3.status_code}):")
print(resp3.text[:2000])

# Try the axis2 services on bdb99
url4 = "https://bdb99.ucsd.edu/axis2/services/BDBService?wsdl"
resp4 = requests.get(url4, timeout=30)
print(f"\nbdb99 WSDL ({resp4.status_code}), Length: {len(resp4.text)}")