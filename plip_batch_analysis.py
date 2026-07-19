"""PLIP批量分析脚本 - 对23目录下所有PDB文件进行蛋白质-配体相互作用分析"""
import sys
import json
import os
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, r'C:\Users\Jy-Mentor-7\AppData\Roaming\Python\Python310\site-packages')
sys.path.insert(0, r'D:\铁衰老 绝不重蹈覆辙\plip_repo')

from plip.structure.preparation import PDBComplex
from plip.basic import config

PDB_DIR = Path(r'C:\Users\Jy-Mentor-7\Desktop\23')
OUTPUT_DIR = Path(r'D:\铁衰老 绝不重蹈覆辙\plip_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

config.TXT = True
config.XML = True
config.PYMOL = False
config.PICS = False
config.QUIET = True
config.MAXTHREADS = 1


def _write_txt_report(mol, output_path):
    """手动生成TXT格式报告，避免lxml DLL依赖"""
    lines = []
    lines.append("=" * 80)
    lines.append("PLIP - Protein-Ligand Interaction Profiler")
    lines.append(f"PDB File: {mol.pymol_name if hasattr(mol, 'pymol_name') else 'unknown'}")
    lines.append("=" * 80)

    for site_key in sorted(mol.interaction_sets.keys()):
        site = mol.interaction_sets[site_key]
        if not site.interacting_res:
            continue
        lines.append(f"\n{'='*80}")
        lines.append(f"Binding Site: {site_key}")
        lines.append(f"{'='*80}")

        if site.hydrophobic_contacts:
            lines.append(f"\nHydrophobic Interactions ({len(site.hydrophobic_contacts)}):")
            lines.append(f"{'Residue':<15} {'LigandAtom':<15} {'Distance':<10}")
            for h in site.hydrophobic_contacts:
                lines.append(f"{h.restype}{h.resnr}{h.reschain:<13} {h.ligatom:<15} {h.distance:.2f}")

        if site.hbonds_pdon or site.hbonds_ldon:
            lines.append(f"\nHydrogen Bonds ({len(site.hbonds_pdon)+len(site.hbonds_ldon)}):")
            lines.append(f"{'Residue':<15} {'Type':<8} {'D-A Dist':<10} {'H-A Dist':<10} {'Angle':<8}")
            for h in site.hbonds_pdon:
                lines.append(f"{h.restype}{h.resnr}{h.reschain:<13} {'pdon':<8} {h.distance_ad:.2f}{'':>4} {h.distance_ah:.2f}{'':>4} {h.angle:.1f}")
            for h in site.hbonds_ldon:
                lines.append(f"{h.restype}{h.resnr}{h.reschain:<13} {'ldon':<8} {h.distance_ad:.2f}{'':>4} {h.distance_ah:.2f}{'':>4} {h.angle:.1f}")

        if site.saltbridge_lneg or site.saltbridge_pneg:
            lines.append(f"\nSalt Bridges ({len(site.saltbridge_lneg)+len(site.saltbridge_pneg)}):")
            lines.append(f"{'Residue':<15} {'ProtIsPos':<10} {'Distance':<10}")
            for sb in site.saltbridge_lneg + site.saltbridge_pneg:
                lines.append(f"{sb.restype}{sb.resnr}{sb.reschain:<13} {str(sb.protispos):<10} {sb.distance:.2f}")

        if site.pistacking:
            lines.append(f"\nPi-Stacking ({len(site.pistacking)}):")
            lines.append(f"{'Residue':<15} {'Type':<8} {'Distance':<10} {'Angle':<8} {'Offset':<8}")
            for ps in site.pistacking:
                lines.append(f"{ps.restype}{ps.resnr}{ps.reschain:<13} {ps.type:<8} {ps.distance:.2f}{'':>4} {ps.angle:.1f}{'':>4} {ps.offset:.2f}")

        if site.pication_laro or site.pication_paro:
            lines.append(f"\nPi-Cation ({len(site.pication_laro)+len(site.pication_paro)}):")
            lines.append(f"{'Residue':<15} {'Type':<8} {'Distance':<10} {'ProtCharged':<12}")
            for pc in site.pication_laro + site.pication_paro:
                lines.append(f"{pc.restype}{pc.resnr}{pc.reschain:<13} {pc.type:<8} {pc.distance:.2f}{'':>4} {str(pc.protcharged):<12}")

        if site.water_bridges:
            lines.append(f"\nWater Bridges ({len(site.water_bridges)}):")
            lines.append(f"{'Residue':<15} {'Water':<10} {'D-AW':<8} {'DW-A':<8} {'D-Ang':<8} {'W-Ang':<8}")
            for wb in site.water_bridges:
                lines.append(f"{wb.restype}{wb.resnr}{wb.reschain:<13} {wb.water:<10} {wb.distance_aw:.2f}{'':>2} {wb.distance_dw:.2f}{'':>2} {wb.d_angle:.1f}{'':>2} {wb.w_angle:.1f}")

        if site.halogen_bonds:
            lines.append(f"\nHalogen Bonds ({len(site.halogen_bonds)}):")
            lines.append(f"{'Residue':<15} {'Distance':<10} {'DonAngle':<10} {'AccAngle':<10}")
            for hb in site.halogen_bonds:
                lines.append(f"{hb.restype}{hb.resnr}{hb.reschain:<13} {hb.distance:.2f}{'':>4} {hb.don_angle:.1f}{'':>4} {hb.acc_angle:.1f}")

        if site.metal_complexes:
            lines.append(f"\nMetal Complexes ({len(site.metal_complexes)}):")
            lines.append(f"{'Metal':<10} {'Residue':<15} {'Distance':<10} {'Geometry':<12}")
            for mc in site.metal_complexes:
                geom = mc.geometry if hasattr(mc, 'geometry') else 'unknown'
                lines.append(f"{mc.metal_type:<10} {mc.restype}{mc.resnr}{mc.reschain:<13} {mc.distance:.2f}{'':>4} {geom:<12}")

    output_path.write_text('\n'.join(lines), encoding='utf-8')


def analyze_pdb(pdb_path):
    """分析单个PDB文件，返回相互作用数据"""
    result = {
        'pdb_file': pdb_path.name,
        'pdb_path': str(pdb_path),
        'ligands': [],
        'interactions': defaultdict(list),
        'total_interactions': 0,
        'error': None
    }

    try:
        mol = PDBComplex()
        mol.output_path = str(OUTPUT_DIR)
        mol.load_pdb(str(pdb_path))

        result['num_ligands'] = len(mol.ligands)

        for ligand in mol.ligands:
            ligand_info = {
                'hetid': ligand.hetid if hasattr(ligand, 'hetid') else 'unknown',
                'chain': ligand.chain if hasattr(ligand, 'chain') else 'unknown',
                'position': ligand.position if hasattr(ligand, 'position') else 'unknown',
            }
            try:
                mol.characterize_complex(ligand)
            except Exception as e:
                ligand_info['error'] = str(e)
            result['ligands'].append(ligand_info)

        for site_key, site in mol.interaction_sets.items():
            if not site.interacting_res:
                continue

            site_data = {
                'site': site_key,
                'hydrophobic': [],
                'hbond': [],
                'water_bridge': [],
                'salt_bridge': [],
                'pi_stacking': [],
                'pi_cation': [],
                'halogen': [],
                'metal': [],
            }

            for h in site.hydrophobic_contacts:
                site_data['hydrophobic'].append({
                    'residue': f"{h.restype}{h.resnr}{h.reschain}",
                    'ligand_atom': h.ligatom,
                    'distance': round(h.distance, 2),
                })

            for h in site.hbonds_pdon + site.hbonds_ldon:
                site_data['hbond'].append({
                    'residue': f"{h.restype}{h.resnr}{h.reschain}",
                    'ligand_atom': getattr(h, 'a_orig_idx', '?'),
                    'distance_h_a': round(getattr(h, 'distance_ah', 0), 2),
                    'distance_d_a': round(getattr(h, 'distance_ad', 0), 2),
                    'angle': round(getattr(h, 'angle', 0), 1),
                    'type': 'pdon' if h in site.hbonds_pdon else 'ldon',
                })

            for wb in site.water_bridges:
                site_data['water_bridge'].append({
                    'residue': f"{wb.restype}{wb.resnr}{wb.reschain}",
                    'water': wb.water,
                    'distance_aw': round(wb.distance_aw, 2),
                    'distance_dw': round(wb.distance_dw, 2),
                    'd_angle': round(wb.d_angle, 1),
                    'w_angle': round(wb.w_angle, 1),
                    'type': wb.protisdon,
                })

            for sb in site.saltbridge_lneg + site.saltbridge_pneg:
                site_data['salt_bridge'].append({
                    'residue': f"{sb.restype}{sb.resnr}{sb.reschain}",
                    'ligand_atom': getattr(sb, 'lig_idx', '?'),
                    'distance': round(sb.distance, 2),
                    'prot_is_pos': sb.protispos,
                })

            for ps in site.pistacking:
                site_data['pi_stacking'].append({
                    'residue': f"{ps.restype}{ps.resnr}{ps.reschain}",
                    'ligand_atom': getattr(ps, 'lig_idx', '?'),
                    'distance': round(ps.distance, 2),
                    'angle': round(ps.angle, 1),
                    'offset': round(ps.offset, 2),
                    'type': ps.type,
                })

            for pc in site.pication_laro + site.pication_paro:
                site_data['pi_cation'].append({
                    'residue': f"{pc.restype}{pc.resnr}{pc.reschain}",
                    'ligand_atom': getattr(pc, 'lig_idx', '?'),
                    'distance': round(pc.distance, 2),
                    'type': pc.type,
                    'prot_is_cation': pc.protcharged,
                })

            for hb in site.halogen_bonds:
                site_data['halogen'].append({
                    'residue': f"{hb.restype}{hb.resnr}{hb.reschain}",
                    'ligand_atom': getattr(hb, 'lig_idx', '?'),
                    'distance': round(hb.distance, 2),
                    'angle_don': round(hb.don_angle, 1),
                    'angle_acc': round(hb.acc_angle, 1),
                })

            for mc in site.metal_complexes:
                site_data['metal'].append({
                    'metal': mc.metal_type,
                    'metal_residue': f"{mc.restype}{mc.resnr}{mc.reschain}",
                    'ligand_atom': getattr(mc, 'lig_idx', '?'),
                    'distance': round(mc.distance, 2),
                    'geometry': mc.geometry if hasattr(mc, 'geometry') else 'unknown',
                })

            site_total = sum(len(v) for v in site_data.values())
            result['total_interactions'] += site_total
            site_data['_total'] = site_total
            result['interactions'][site_key] = site_data

        # 生成TXT报告（手动写入，绕过lxml）
        try:
            txt_path = OUTPUT_DIR / f"plip_{pdb_path.stem}.txt"
            _write_txt_report(mol, txt_path)
        except Exception as e:
            result['report_error'] = str(e)

    except Exception as e:
        result['error'] = str(e)

    return result


def main():
    pdb_files = sorted(PDB_DIR.glob('*.pdb'))
    print(f"找到 {len(pdb_files)} 个PDB文件")
    print("=" * 80)

    all_results = {}
    summary = defaultdict(lambda: {'total': 0, 'hydrophobic': 0, 'hbond': 0,
                                    'water_bridge': 0, 'salt_bridge': 0,
                                    'pi_stacking': 0, 'pi_cation': 0,
                                    'halogen': 0, 'metal': 0})

    for i, pdb_path in enumerate(pdb_files, 1):
        print(f"[{i}/{len(pdb_files)}] 分析: {pdb_path.name} ...")
        result = analyze_pdb(pdb_path)
        all_results[pdb_path.name] = result

        target = pdb_path.stem.split('_')[0]
        for site_key, site_data in result['interactions'].items():
            for int_type in ['hydrophobic', 'hbond', 'water_bridge', 'salt_bridge',
                             'pi_stacking', 'pi_cation', 'halogen', 'metal']:
                count = len(site_data[int_type])
                summary[target][int_type] += count
                summary[target]['total'] += count

        status = "OK" if not result['error'] else f"ERROR: {result['error']}"
        print(f"  -> {status}, 配体: {result.get('num_ligands', 0)}, 总相互作用: {result['total_interactions']}")

    print("\n" + "=" * 80)
    print("相互作用汇总（按靶点）")
    print("-" * 80)
    header = f"{'靶点':<10} {'总计':>6} {'疏水':>6} {'氢键':>6} {'水桥':>6} {'盐桥':>6} {'π堆叠':>6} {'π阳离子':>6} {'卤键':>6} {'金属':>6}"
    print(header)
    print("-" * 80)

    for target in sorted(summary.keys()):
        s = summary[target]
        pdb_count = len([r for r in all_results.values() if r['pdb_file'].startswith(target)])
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
    print(f"报告文件已保存到: {OUTPUT_DIR}")

    return output_json


if __name__ == '__main__':
    main()