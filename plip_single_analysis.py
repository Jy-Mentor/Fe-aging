"""PLIP批量分析 - 单文件分析子进程（隔离openbabel segfault）"""
import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, r'C:\Users\Jy-Mentor-7\AppData\Roaming\Python\Python310\site-packages')
sys.path.insert(0, r'D:\铁衰老 绝不重蹈覆辙\plip_repo')

from plip.structure.preparation import PDBComplex
from plip.basic import config

config.QUIET = True
config.MAXTHREADS = 1


def analyze_single_pdb(pdb_path, output_dir):
    """分析单个PDB文件，返回JSON字符串"""
    result = {
        'pdb_file': pdb_path.name,
        'ligands': [],
        'interactions': {},
        'total_interactions': 0,
        'error': None,
        'num_ligands': 0,
    }

    try:
        mol = PDBComplex()
        mol.output_path = str(output_dir)
        mol.load_pdb(str(pdb_path))
        result['num_ligands'] = len(mol.ligands)

        for ligand in mol.ligands:
            ligand_info = {
                'hetid': str(ligand.hetid),
                'chain': str(ligand.chain),
                'position': str(ligand.position),
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
                'hydrophobic': len(site.hydrophobic_contacts),
                'hbond': len(site.hbonds_pdon) + len(site.hbonds_ldon),
                'water_bridge': len(site.water_bridges),
                'salt_bridge': len(site.saltbridge_lneg) + len(site.saltbridge_pneg),
                'pi_stacking': len(site.pistacking),
                'pi_cation': len(site.pication_laro) + len(site.pication_paro),
                'halogen': len(site.halogen_bonds),
                'metal': len(site.metal_complexes),
            }
            site_total = sum(site_data.values())
            result['total_interactions'] += site_total
            site_data['_total'] = site_total
            result['interactions'][site_key] = site_data

    except Exception as e:
        result['error'] = str(e)

    return result


if __name__ == '__main__':
    pdb_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    result = analyze_single_pdb(pdb_path, output_dir)
    print(json.dumps(result, ensure_ascii=False, default=str))