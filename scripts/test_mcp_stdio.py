import subprocess
import json
import sys
import time
import threading
import os

python_exe = r'C:\Users\Jy-Mentor-7\anaconda3\python.exe'

proc = subprocess.Popen(
    [python_exe, '-m', 'paper_search_mcp.server'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=False,
    bufsize=0,
)

stderr_lines = []

def read_stderr():
    try:
        for line in proc.stderr:
            stderr_lines.append(line.decode('utf-8', errors='replace'))
    except Exception:
        pass

stderr_thread = threading.Thread(target=read_stderr, daemon=True)
stderr_thread.start()

time.sleep(5)
print('--- stderr after startup ---')
print(''.join(stderr_lines)[:2000] if stderr_lines else '(empty)')

def send_json_line(obj):
    body = (json.dumps(obj) + '\n').encode('utf-8')
    print('--- sending raw json line ---')
    print(body.decode('utf-8', errors='replace'))
    proc.stdin.write(body)
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
send_json_line(init)

time.sleep(3)
print('--- stderr after initialize ---')
print(''.join(stderr_lines)[:4000] if stderr_lines else '(empty)')

print('--- stdout response ---')
try:
    out = proc.stdout.read(8192)
    if out:
        print(out.decode('utf-8', errors='replace'))
    else:
        print('(no stdout response)')
except Exception as e:
    print('read error:', e)

print('--- return code ---')
print(proc.poll())
proc.kill()
