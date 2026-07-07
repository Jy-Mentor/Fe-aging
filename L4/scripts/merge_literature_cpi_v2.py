import logging
logger = logging.getLogger(__name__)

"""
整理文献挖掘的CPI数据 — v2: 手动整理SMILES，不依赖PubChem API
"""
import pandas as pd
import numpy as np
from rdkit import Chem

# ============================================================
# 文献挖掘的抑制剂数据（基因 -> 化合物列表，含手动整理的SMILES）
# 格式: gene, uniprot_id, compound_name, canonical_smiles, activity_value_nM, activity_type, source_ref
# ============================================================
LITERATURE_CPI = [
    # === EDN1 (Endothelin-1) - ETA/ETB receptor antagonists ===
    ("EDN1", "P05305", "Bosentan",
     "CC(C)(C)C1=CC=C(C=C1)S(=O)(=O)NC2=C(C(=NC(=N2)C3=NC=CC=N3)OCCO)OC4=CC=CC=C4OC",
     80.0, "Ki", "Guide to Pharmacology"),
    ("EDN1", "P05305", "Ambrisentan",
     "CC1=CC(=NC(=N1)OC(C(=O)O)C(C2=CC=CC=C2)(C3=CC=CC=C3)OC)C",
     1.0, "Ki", "Guide to Pharmacology"),
    ("EDN1", "P05305", "Macitentan",
     "CCCNS(=O)(=O)Nc1ncnc(c1c1ccc(cc1)Br)OCCOc1ncc(cn1)Br",
     0.5, "IC50", "Guide to Pharmacology"),
    ("EDN1", "P05305", "Atrasentan",
     "CCCCN(C(=O)CN1CC(C(C1c1ccc(cc1)OC)C(=O)O)c1ccc2c(c1)OCO2)CCCC",
     0.034, "Ki", "BenchChem"),
    ("EDN1", "P05305", "Clazosentan",
     "CC1=CN=C(C=C1)S(=O)(=O)NC2=C(C(=NC(=N2)C3=CC(=NC=C3)C4=NN=NN4)OCCO)OC5=CC=CC=C5OC",
     0.3, "IC50", "Guide to Pharmacology"),
    # Zibotentan, Sitaxsentan, Darusentan, Aprocitentan, Sparsentan, Avosentan, SC0062 — SMILES unavailable from PubChem
    # These are known EDN1 antagonists but SMILES not retrievable

    # === WNT5A ===
    # Box5 is a peptide (t-Boc-Gly-Met-Arg-Arg-NH-CH(CH3)-(CH2)3-NH-Boc), no SMILES

    # === IRF1 ===
    # IRF1-IN-1, ALEKSIN — SMILES unavailable

    # === E2F1 ===
    # Bigelovin, HLM006474, HR488B — SMILES unavailable

    # === E2F3 ===
    # HLM006474, M606 — SMILES unavailable

    # === TXNIP ===
    ("TXNIP", "Q9H3M7", "SRI-37330",
     "CS(=O)(=O)NCC1CCCN(C1)c1ncnc2ccc(cc12)C(F)(F)F",
     640.0, "IC50", "Selleckchem"),
    ("TXNIP", "Q9H3M7", "Verapamil",
     "COc1ccc(cc1OC)CCN(CCCC(c1ccc(c(c1)OC)OC)(C(C)C)C#N)C",
     100000.0, "IC50", "Front Endocrinol 2024"),
    # TXNIP-IN-1, TIX100 — SMILES unavailable
]

def validate_smiles(smiles):
    """验证SMILES有效性"""
    if not smiles or pd.isna(smiles):
        return False
    mol = Chem.MolFromSmiles(str(smiles))
    return mol is not None

def main():
    print("=" * 60)
    print("文献挖掘CPI数据整理 v2 (手动SMILES)")
    print("=" * 60)

    # 读取现有combined
    combined = pd.read_csv(
        r'd:\铁衰老 绝不重蹈覆辙\L4\results\experimental_actives_detail_cleaned_combined.csv',
        low_memory=False
    )
    print(f"现有数据: {len(combined)} 条, {combined.gene.nunique()} 基因")

    new_rows = []
    valid_count = 0
    invalid_count = 0

    for gene, uniprot, name, smiles, value, atype, source in LITERATURE_CPI:
        if validate_smiles(smiles):
            valid_count += 1
            print(f"  [OK] {gene} / {name}: SMILES valid")
            new_rows.append({
                'source': 'Literature_' + source.replace(' ', '_'),
                'gene': gene,
                'uniprot_id': uniprot,
                'target_chembl_id': '',
                'target_pref_name': gene,
                'molecule_chembl_id': '',
                'molecule_pref_name': name,
                'canonical_smiles': smiles,
                'standard_type': atype,
                'standard_value_nM': value,
                'pchembl_value': 9 - np.log10(value) if value > 0 else np.nan,
                'confidence_score': 4,
                'assay_description': f'Literature mined: {name} ({source})',
                'molecule_name': name,
                'bindingdb_monomer_id': '',
                'target_name': gene,
                'pmid': '',
                'doi': '',
                'drugbank_id': '',
                'drug_name': '',
                'note': f'Literature supplement v32: {source}'
            })
        else:
            invalid_count += 1
            print(f"  [SKIP] {gene} / {name}: SMILES invalid or unavailable")

    print(f"\nSMILES验证: {valid_count} 有效, {invalid_count} 跳过")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        before = len(new_df)
        new_df = new_df.drop_duplicates(subset=['gene', 'canonical_smiles'])
        print(f"新记录: {len(new_df)} 条 (去重前 {before})")

        # 合并
        combined_new = pd.concat([combined, new_df], ignore_index=True)
        combined_new = combined_new.drop_duplicates(subset=['gene', 'canonical_smiles'])

        # 检查铁衰老基因
        from pathlib import Path
        iron_gene_file = Path(r'd:\铁衰老 绝不重蹈覆辙\铁衰老基因.txt')
        if iron_gene_file.exists():
            with open(iron_gene_file, 'r', encoding='utf-8') as f:
                iron_genes = {line.strip() for line in f if line.strip()}
        else:
            iron_genes = set()

        iron_in = combined_new[combined_new['gene'].isin(iron_genes)]
        new_iron_genes = {row['gene'] for row in new_rows} & iron_genes
        print(f"\n合并后: {len(combined_new)} 条, {combined_new.gene.nunique()} 基因")
        if iron_genes:
            print(f"铁衰老基因: {iron_in.gene.nunique()}/{len(iron_genes)}")
        print(f"新增铁衰老基因: {sorted(new_iron_genes)}")

        # 保存
        combined_new.to_csv(
            r'd:\铁衰老 绝不重蹈覆辙\L4\results\experimental_actives_detail_cleaned_combined.csv',
            index=False
        )
        print("\n已保存!")

        # 统计缺失基因
        if iron_genes:
            missing = sorted(iron_genes - set(combined_new['gene'].dropna().unique()))
            print(f"\n仍缺失的铁衰老基因 ({len(missing)}个):")
            for i, g in enumerate(missing):
                print(f"  {i+1}. {g}")
    else:
        print("\n无新数据可添加")

if __name__ == "__main__":
    main()