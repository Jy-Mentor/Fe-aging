import pandas as pd

# Check if ChEMBL data has SMILES for the DrugBank drugs by name
chembl = pd.read_csv(r'd:\铁衰老 绝不重蹈覆辙\L4\results\chembl_active_compounds.csv', low_memory=False)
main_cpi = pd.read_csv(r'd:\铁衰老 绝不重蹈覆辙\L4\results\experimental_actives_detail_cleaned.csv', low_memory=False)

# Target drug names from DrugBank
target_drugs = {
    'DB00159': 'Icosapent',
    'DB00412': 'Rosiglitazone', 
    'DB00197': 'Troglitazone',
    'DB00030': 'Insulin human',
    'DB01629': '5-fluorouridine',
    'DB14001': 'alpha-Tocopherol succinate',
    'DB09096': 'Benzoyl peroxide',
    'DB09061': 'Cannabidiol',
    'DB00958': 'Carboplatin',
    'DB00515': 'Cisplatin',
    'DB09130': 'Copper',
    'DB14002': 'D-alpha-Tocopherol acetate',
    'DB00988': 'Dopamine',
    'DB01064': 'Isoprenaline',
    'DB14009': 'Medical Cannabis',
    'DB14011': 'Nabiximols',
    'DB00526': 'Oxaliplatin',
    'DB09221': 'Polaprezinc',
    'DB03382': 'S-oxy-L-cysteine',
    'DB05088': 'Tetrathiomolybdate',
    'DB00163': 'Vitamin E',
    'DB01593': 'Zinc',
    'DB14487': 'Zinc acetate',
    'DB14533': 'Zinc chloride',
    'DB14548': 'Zinc sulfate, unspecified form',
    'DB12449': 'Tempol',
    'DB14782': 'Tofersen',
    'DB05874': 'AEOL-10150',
    'DB14511': 'Acetate',
    'DB17641': 'Ammonium tetrathiomolybdate',
}

# Search in ChEMBL data for these drug names (case-insensitive)
print("Searching in ChEMBL data for DrugBank drug names...")
for db_id, drug_name in target_drugs.items():
    # Search in molecule_pref_name and molecule_name
    matches = chembl[chembl['molecule_pref_name'].str.contains(drug_name, case=False, na=False)]
    if len(matches) == 0:
        matches = chembl[chembl['molecule_name'].str.contains(drug_name, case=False, na=False)] if 'molecule_name' in chembl.columns else pd.DataFrame()
    
    if len(matches) > 0:
        smiles = matches['canonical_smiles'].dropna().values[0] if len(matches['canonical_smiles'].dropna()) > 0 else 'N/A'
        print(f"  {db_id} ({drug_name}): FOUND {len(matches)} records, SMILES: {str(smiles)[:80]}")
    else:
        print(f"  {db_id} ({drug_name}): NOT FOUND in ChEMBL")

# Also search in main CPI
print("\nSearching in main CPI for DrugBank drug names...")
for db_id, drug_name in target_drugs.items():
    matches = main_cpi[main_cpi['molecule_pref_name'].str.contains(drug_name, case=False, na=False)]
    if len(matches) == 0:
        matches = main_cpi[main_cpi['molecule_name'].str.contains(drug_name, case=False, na=False)] if 'molecule_name' in main_cpi.columns else pd.DataFrame()
    if len(matches) == 0:
        matches = main_cpi[main_cpi['drug_name'].str.contains(drug_name, case=False, na=False)] if 'drug_name' in main_cpi.columns else pd.DataFrame()
    
    if len(matches) > 0:
        smiles = matches['canonical_smiles'].dropna().values[0] if len(matches['canonical_smiles'].dropna()) > 0 else 'N/A'
        gene = matches['gene'].values[0] if 'gene' in matches.columns else 'N/A'
        print(f"  {db_id} ({drug_name}): FOUND {len(matches)} records, gene={gene}, SMILES: {str(smiles)[:80]}")
    else:
        print(f"  {db_id} ({drug_name}): NOT FOUND in main CPI")