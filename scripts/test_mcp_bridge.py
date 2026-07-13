import subprocess
import json
import time
import threading
import os

bridge_cmd = [r'C:\Users\Jy-Mentor-7\anaconda3\python.exe', r'D:\铁衰老 绝不重蹈覆辙\scripts\paper_search_mcp_stdio_bridge.py']

env = {
    **os.environ,
    "PAPER_SEARCH_MCP_UNPAYWALL_EMAIL": "",
    "PAPER_SEARCH_MCP_CORE_API_KEY": "",
    "PAPER_SEARCH_MCP_SEMANTIC_SCHOLAR_API_KEY": "",
    "PAPER_SEARCH_MCP_ZENODO_ACCESS_TOKEN": "",
    "PAPER_SEARCH_MCP_GOOGLE_SCHOLAR_PROXY_URL": "",
    "PAPER_SEARCH_MCP_IEEE_API_KEY": "",
    "PAPER_SEARCH_MCP_ACM_API_KEY": "",
}

proc = subprocess.Popen(
    bridge_cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    bufsize=0,
    env=env,
)

stderr_lines = []

def read_stderr():
    try:
        for line in proc.stderr:
            stderr_lines.append(line.decode('utf-8', errors='replace'))
    except Exception:
        pass

threading.Thread(target=read_stderr, daemon=True).start()

time.sleep(5)
print('--- stderr after startup ---')
print(''.join(stderr_lines)[:2000] if stderr_lines else '(empty)')

def send_content_length(obj):
    body = json.dumps(obj).encode('utf-8')
    msg = f"Content-Length: {len(body)}\r\n\r\n".encode('utf-8') + body
    print('--- sending ---')
    print(msg.decode('utf-8', errors='replace'))
    proc.stdin.write(msg)
    proc.stdin.flush()

init = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1.0"}
    }
}
send_content_length(init)

time.sleep(3)
print('--- stderr after initialize ---')
print(''.join(stderr_lines)[:4000] if stderr_lines else '(empty)')

print('--- stdout response (initialize) ---')
try:
    out = proc.stdout.read(4096)
    if out:
        print(out.decode('utf-8', errors='replace'))
    else:
        print('(no stdout response)')
except Exception as e:
    print('read error:', e)

tools_list = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list",
    "params": {}
}
send_content_length(tools_list)

time.sleep(3)
print('--- stdout response (tools/list) ---')
try:
    out = proc.stdout.read(8192)
    if out:
        print(out.decode('utf-8', errors='replace')[:4000])
    else:
        print('(no stdout response)')
except Exception as e:
    print('read error:', e)

proc.kill()
