import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

df = pd.read_csv(r'd:\铁衰老 绝不重蹈覆辙\L4\results\cpi_supplement_v27.csv')
print(f"Total records: {len(df)}")
print(f"Genes: {sorted(df['gene'].unique())}")
print()

for gene in sorted(df['gene'].unique()):
    gdf = df[df['gene'] == gene]
    print(f"{gene}: {len(gdf)} compounds")
    for _, row in gdf.iterrows():
        smi = row['smiles']
        mol = Chem.MolFromSmiles(smi)
        mw = round(Descriptors.MolWt(mol), 1) if mol else 'N/A'
        name = row['compound_name']
        print(f"  - {name} (MW={mw})")

print()
main = pd.read_csv(r'd:\铁衰老 绝不重蹈覆辙\L4\results\experimental_actives_detail_cleaned.csv', low_memory=False)
main['gene'] = main['gene'].str.strip().str.upper()
for gene in sorted(df['gene'].unique()):
    main_gene = main[main['gene'] == gene]
    if len(main_gene) > 0:
        print(f"WARNING: {gene} already in main CPI!")
    else:
        print(f"OK: {gene} NOT in main CPI")

dups = df.duplicated(subset=['gene', 'smiles'])
print(f"Internal duplicates: {dups.sum()}")