"""PLIP批量分析编排器 - 逐个启动子进程隔离openbabel segfault"""
import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

PDB_DIR = Path(r'C:\Users\Jy-Mentor-7\Desktop\23')
OUTPUT_DIR = Path(r'D:\铁衰老 绝不重蹈覆辙\plip_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SINGLE_SCRIPT = Path(r'D:\铁衰老 绝不重蹈覆辙\plip_single_analysis.py')

pdb_files = sorted(PDB_DIR.glob('*.pdb'))
print(f"找到 {len(pdb_files)} 个PDB文件")
print("=" * 80)

all_results = {}
summary = defaultdict(lambda: {'total': 0, 'hydrophobic': 0, 'hbond': 0,
                                'water_bridge': 0, 'salt_bridge': 0,
                                'pi_stacking': 0, 'pi_cation': 0,
                                'halogen': 0, 'metal': 0})

for i, pdb_path in enumerate(pdb_files, 1):
    print(f"[{i}/{len(pdb_files)}] 分析: {pdb_path.name} ...", end=' ', flush=True)

    try:
        proc = subprocess.run(
            [sys.executable, str(SINGLE_SCRIPT), str(pdb_path), str(OUTPUT_DIR)],
            capture_output=True, text=True, timeout=120,
            env={**__import__('os').environ, 'PYTHONPATH': ''}
        )

        if proc.returncode != 0 and proc.returncode != -1073741819:
            result = {'error': f'returncode={proc.returncode}', 'total_interactions': 0,
                      'ligands': [], 'interactions': {}, 'num_ligands': 0}
            print(f"ERROR (rc={proc.returncode})")
        else:
            try:
                result = json.loads(proc.stdout.strip().split('\n')[-1])
            except json.JSONDecodeError:
                result = {'error': 'json_decode_failed', 'total_interactions': 0,
                          'ligands': [], 'interactions': {}, 'num_ligands': 0}
                print("JSON_PARSE_ERROR")
                all_results[pdb_path.name] = result
                continue

            target = pdb_path.stem.split('_')[0]
            for site_key, site_data in result['interactions'].items():
                for int_type in ['hydrophobic', 'hbond', 'water_bridge', 'salt_bridge',
                                 'pi_stacking', 'pi_cation', 'halogen', 'metal']:
                    count = site_data.get(int_type, 0)
                    summary[target][int_type] += count
                    summary[target]['total'] += count

            status = "OK" if not result.get('error') else f"ERROR: {result['error']}"
            print(f"{status}, 配体: {result.get('num_ligands', 0)}, 相互作用: {result['total_interactions']}")

    except subprocess.TimeoutExpired:
        result = {'error': 'timeout', 'total_interactions': 0, 'ligands': [], 'interactions': {}, 'num_ligands': 0}
        print("TIMEOUT")

    all_results[pdb_path.name] = result

    if i % 5 == 0:
        _tmp = {
            'summary': {k: dict(v) for k, v in summary.items()},
            'grand_total': sum(s['total'] for s in summary.values()),
            'results': all_results,
            'pdb_count': len(pdb_files),
            'targets': list(summary.keys()),
        }
        (OUTPUT_DIR / 'plip_summary.json').write_text(
            json.dumps(_tmp, indent=2, ensure_ascii=False, default=str), encoding='utf-8')

print("\n" + "=" * 80)
print("相互作用汇总（按靶点）")
print("-" * 80)
header = f"{'靶点':<10} {'总计':>6} {'疏水':>6} {'氢键':>6} {'水桥':>6} {'盐桥':>6} {'π堆叠':>6} {'π阳离子':>6} {'卤键':>6} {'金属':>6}"
print(header)
print("-" * 80)

for target in sorted(summary.keys()):
    s = summary[target]
    pdb_count = len([r for r in all_results.values() if r.get('pdb_file', '').startswith(target)])
    row = f"{target:<10} {s['total']:>6} {s['hydrophobic']:>6} {s['hbond']:>6} {s['water_bridge']:>6} {s['salt_bridge']:>6} {s['pi_stacking']:>6} {s['pi_cation']:>6} {s['halogen']:>6} {s['metal']:>6}  ({pdb_count} PDBs)"
    print(row)

grand_total = sum(s['total'] for s in summary.values())
print("-" * 80)
print(f"{'总计':<10} {grand_total:>6}")

output_json = {
    'summary': {k: dict(v) for k, v in summary.items()},
    'grand_total': grand_total,
    'results': all_results,
    'pdb_count': len(pdb_files),
    'targets': list(summary.keys()),
}

json_path = OUTPUT_DIR / 'plip_summary.json'
json_path.write_text(json.dumps(output_json, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
print(f"\n详细结果已保存到: {json_path}")