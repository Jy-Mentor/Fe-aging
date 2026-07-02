#!/usr/bin/env python3
import sys, os, subprocess, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
L4_SCRIPTS = PROJECT_ROOT / 'L4' / 'scripts'

def run_check(name, cmd, cwd=None):
    sep = '=' * 60
    print(f"\n{sep}\n  [{name}]\n{sep}")
    start = time.time()
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd or str(L4_SCRIPTS),
                                capture_output=True, text=True, timeout=300)
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
    checks = [
        ('Ruff Lint', 'ruff check .'),
        ('Smoke Test', 'python smoke_test.py'),
        ('Model Input', 'python validate_model_inputs.py'),
    ]
    results = {}
    for name, cmd in checks:
        results[name] = run_check(name, cmd)
    sep = '=' * 60
    print(f'
{sep}
  Summary
{sep}')
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {name}')
    print(f'
  {passed}/{total} passed')
    return 0 if passed == total else 1

if __name__ == '__main__':
    sys.exit(main())
