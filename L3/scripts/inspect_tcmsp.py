import requests
import re
import json
from bs4 import BeautifulSoup as bs

root_url = "https://www.tcmsp-e.com/tcmspsearch.php"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
}

session = requests.Session()
session.headers.update(headers)

# 1. 获取 token
html = session.get(root_url, timeout=30).text
soup = bs(html, "html.parser")
token_input = soup.find("input", {"name": "token"})
token = token_input["value"] if token_input else None
print(f"Token: {token}")

# 2. 获取柴胡的英文名称（从搜索结果）
search_url = f"{root_url}?qs=herb_all_name&q=柴胡&token={token}"
html = session.get(search_url, timeout=30).text
# 提取 herb_en_name
match = re.search(r'data:\s*(\[.*?\]),', html)
if match:
    herbs = json.loads(match.group(1))
    print(f"搜索结果: {herbs}")
    target_herb = herbs[0]  # 选择第一个（柴胡）
    en_name = target_herb['herb_en_name']
    cn_name = target_herb['herb_cn_name']
    print(f"选择: {cn_name} / {en_name}")
else:
    print("未找到搜索结果")
    exit()

# 3. 访问柴胡的成分页面
en_name_encoded = en_name.replace(" ", "%20")
herb_url = f"{root_url}?qr={en_name_encoded}&qsr=herb_en_name&token={token}"
print(f"\n访问: {herb_url}")
html = session.get(herb_url, timeout=30).text

# 查找成分数据 (grid)
soup = bs(html, "html.parser")
scripts = soup.find_all("script")
print(f"Script tags count: {len(scripts)}")

for i, script in enumerate(scripts):
    text = script.string if script.string else ""
    if "kendoGrid" in text and "data:" in text:
        print(f"\n--- Script {i} contains kendoGrid ---")
        # 提取 data 部分
        m = re.search(r'data:\s*(\[.*?\]),\s*pageSize', text, re.DOTALL)
        if m:
            data_str = m.group(1)
            print(f"数据长度: {len(data_str)}")
            print(f"数据前500字符: {data_str[:500]}")
            try:
                data = json.loads(data_str)
                print(f"成功解析JSON，记录数: {len(data)}")
                if len(data) > 0:
                    print(f"第一条记录: {data[0]}")
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}")
        break
