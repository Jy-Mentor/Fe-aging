import requests
from pathlib import Path

base_url = "https://tcmsp-e.com/attachment/tcmspDB/"
files = [
    "23_Herbs_Molecules_Relationships.xlsx",
    "03_Info_Molecules.xlsx",
]

save_dir = Path("d:/铁衰老 绝不重蹈覆辙/L3/data/tcmsp_official")
save_dir.mkdir(exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://tcmsp-e.com/load_intro.php?id=31",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

session = requests.Session()
session.headers.update(headers)

# 先访问 referer 页面
print("访问 referer 页面...")
resp = session.get("https://tcmsp-e.com/load_intro.php?id=31", timeout=30)
print(f"referer 状态: {resp.status_code}")
print(f"Content-Type: {resp.headers.get('Content-Type')}")
print(f"Cookies: {dict(session.cookies)}")

for fname in files:
    url = base_url + fname
    save_path = save_dir / fname
    if save_path.exists():
        save_path.unlink()
    print(f"\n正在下载: {url} ...")
    try:
        r = session.get(url, timeout=60, allow_redirects=True)
        r.raise_for_status()
        save_path.write_bytes(r.content)
        print(f"状态码: {r.status_code}")
        print(f"Content-Type: {r.headers.get('Content-Type')}")
        print(f"大小: {len(r.content)} bytes")
        print(f"URL: {r.url}")
        print(f"前50字节: {r.content[:50]}")
    except Exception as e:
        print(f"错误: {e}")
