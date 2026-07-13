#!/usr/bin/env python3
import logging
import sys
import os
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
L4_SCRIPTS = PROJECT_ROOT / 'L4' / 'scripts'
L4_ROOT = PROJECT_ROOT / 'L4'
RUFF_CONFIG = L4_ROOT / 'pyproject.toml'

def run_check(name, cmd, cwd=None):
    sep = '=' * 60
    print(f"\n{sep}\n  [{name}]\n{sep}")
    start = time.time()
    try:
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(cmd, shell=True, cwd=cwd,
                                capture_output=True, text=True, timeout=300,
                                encoding='utf-8', errors='replace', env=env)
        elapsed = time.time() - start
        if result.returncode == 0:
            print(f'  PASS ({elapsed:.1f}s)')
            if result.stdout.strip():
                for line in result.stdout.strip().split(chr(10))[-5:]:
                    print(f'  | {line}')
        else:
            print(f'  FAIL ({elapsed:.1f}s)')
            if result.stderr.strip():
                for line in result.stderr.strip().split(chr(10))[-10:]:
                    print(f'  ! {line}')
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print('  TIMEOUT (>300s)')
        return False
    except Exception as e:
        print(f'  ERROR: {e}')
        return False

def main():
    print('=' * 60)
    print('  Iron Aging GNN - Quality Gate')
    print('=' * 60)
    python_exe = sys.executable
    checks = [
        ('Ruff Lint', f'{python_exe} -m ruff check scripts/ src/ --config pyproject.toml', str(L4_ROOT)),
        ('Smoke Test', f'{python_exe} smoke_test.py', str(L4_SCRIPTS)),
    ]
    results = {}
    for name, cmd, cwd in checks:
        results[name] = run_check(name, cmd, cwd)
    sep = '=' * 60
    print(f'\n{sep}\n  Summary\n{sep}')
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {name}')
    print(f'\n  {passed}/{total} passed')
    return 0 if passed == total else 1

if __name__ == '__main__':
    sys.exit(main())
