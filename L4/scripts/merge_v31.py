import pandas as pd
import numpy as np
from rdkit import Chem

# 1. existing
combined = pd.read_csv(r'd:\铁衰老 绝不重蹈覆辙\L4\results\experimental_actives_detail_cleaned_combined.csv', low_memory=False)
print(f'Existing: {len(combined)} rows, {combined.gene.nunique()} genes')

# 2. v31
v31 = pd.read_csv(r'd:\铁衰老 绝不重蹈覆辙\L4\results_v10_minibatch\cpi_supplement_v31_bindingdb.csv')
print(f'v31: {len(v31)} rows, {v31.gene.nunique()} genes')
print(f'New genes: {sorted(v31.gene.unique())}')

# 3. Validate SMILES
valid = [Chem.MolFromSmiles(s) is not None for s in v31['canonical_smiles']]
print(f'Valid SMILES: {sum(valid)}/{len(v31)}')
v31 = v31[valid].reset_index(drop=True)

# 4. Convert format
new_rows = []
for _, row in v31.iterrows():
    val = row['activity_value']
    pchembl = 9 - np.log10(val) if val > 0 else np.nan
    new_rows.append({
        'source': row['source'],
        'gene': row['gene'],
        'uniprot_id': row['uniprot_id'],
        'target_chembl_id': '',
        'target_pref_name': row['gene'],
        'molecule_chembl_id': '',
        'molecule_pref_name': '',
        'canonical_smiles': row['canonical_smiles'],
        'standard_type': row['activity_type'],
        'standard_value_nM': val,
        'pchembl_value': pchembl,
        'confidence_score': 6,
        'assay_description': 'ChEMBL multi-type activity (v31 supplement)',
        'molecule_name': '',
        'bindingdb_monomer_id': '',
        'target_name': row['gene'],
        'pmid': str(row['pubmed_id']),
        'doi': '',
        'drugbank_id': '',
        'drug_name': '',
        'note': 'supplemental v31 (multi-type activity search)'
    })
new_df = pd.DataFrame(new_rows)

# 5. Merge
combined_new = pd.concat([combined, new_df], ignore_index=True)
combined_new = combined_new.drop_duplicates(subset=['gene', 'canonical_smiles'])

# 6. Stats
with open(r'd:\铁衰老 绝不重蹈覆辙\铁衰老基因.txt','r',encoding='utf-8') as f:
    iron_genes = set(line.strip() for line in f if line.strip())
iron_in = combined_new[combined_new['gene'].isin(iron_genes)]
print(f'\nMerged: {len(combined_new)} rows, {combined_new.gene.nunique()} genes')
print(f'Iron-aging genes: {iron_in.gene.nunique()}/{len(iron_genes)}')
new_iron = sorted(set(v31.gene.unique()) & iron_genes)
print(f'New iron-aging genes: {new_iron}')

# 7. Save
combined_new.to_csv(r'd:\铁衰老 绝不重蹈覆辙\L4\results\experimental_actives_detail_cleaned_combined.csv', index=False)
print('\nSaved!')