import logging
logger = logging.getLogger(__name__)

"""
整理文献挖掘的CPI数据，通过PubChem获取SMILES并合并到combined数据集
"""
import pandas as pd
import numpy as np
import requests
import time
from rdkit import Chem

# 文献挖掘的抑制剂数据（基因 -> 化合物列表）
# 格式: gene, uniprot_id, compound_name, cas_no, activity_value_nM, activity_type, source_ref
LITERATURE_CPI = [
    # === EDN1 (Endothelin-1) - ETA/ETB receptor antagonists ===
    # 来自 Guide to Pharmacology + Selleckchem + BenchChem
    ("EDN1", "P05305", "Bosentan", "147536-97-8", 80.0, "Ki", "Guide to Pharmacology"),
    ("EDN1", "P05305", "Ambrisentan", "177036-94-1", 1.0, "Ki", "Guide to Pharmacology"),
    ("EDN1", "P05305", "Zibotentan", "186497-07-4", 21.0, "IC50", "Selleckchem"),
    ("EDN1", "P05305", "Atrasentan", "195704-72-4", 0.034, "Ki", "BenchChem"),
    ("EDN1", "P05305", "Macitentan", "441798-33-0", 0.5, "IC50", "Guide to Pharmacology"),
    ("EDN1", "P05305", "Clazosentan", "180384-57-0", 0.3, "IC50", "Guide to Pharmacology"),
    ("EDN1", "P05305", "Sitaxsentan", "184036-34-8", 10.0, "IC50", "Guide to Pharmacology"),
    ("EDN1", "P05305", "Darusentan", "171714-84-4", 1.0, "IC50", "Guide to Pharmacology"),
    ("EDN1", "P05305", "Aprocitentan", "1103522-45-7", 200.0, "IC50", "Guide to Pharmacology"),
    ("EDN1", "P05305", "Sparsentan", "936727-05-8", 0.8, "IC50", "Guide to Pharmacology"),
    ("EDN1", "P05305", "Avosentan", "290815-26-8", 3.0, "IC50", "Guide to Pharmacology"),
    ("EDN1", "P05305", "SC0062", "NA", 2.3, "IC50", "Can J Physiol Pharmacol 2026"),
    
    # === WNT5A ===
    # Box5 is a peptide, but we can try to find it
    ("WNT5A", "P41221", "Box5", "NA", 1000.0, "IC50", "MCE"),
    
    # === IRF1 ===
    ("IRF1", "P10914", "IRF1-IN-1", "701225-07-2", 20000.0, "IC50", "MCE"),
    ("IRF1", "P10914", "ALEKSIN", "NA", 5000.0, "IC50", "Front Pharmacol 2025"),
    
    # === E2F1 ===
    ("E2F1", "Q01094", "Bigelovin", "NA", 5000.0, "IC50", "Cancer Sci 2013"),
    ("E2F1", "Q01094", "HLM006474", "NA", 30000.0, "IC50", "PLoS ONE 2014"),
    ("E2F1", "Q01094", "HR488B", "NA", 380.0, "IC50", "Cell Death Dis 2023"),
    
    # === E2F3 ===
    ("E2F3", "O00716", "HLM006474", "NA", 30000.0, "IC50", "PLoS ONE 2014"),
    ("E2F3", "O00716", "M606", "NA", 1000.0, "IC50", "PNAS 2025"),
    
    # === TXNIP ===
    ("TXNIP", "Q9H3M7", "SRI-37330", "2322245-49-6", 640.0, "IC50", "Selleckchem"),
    ("TXNIP", "Q9H3M7", "TXNIP-IN-1", "1268955-50-5", 500.0, "IC50", "MCE"),
    ("TXNIP", "Q9H3M7", "Verapamil", "52-53-9", 100000.0, "IC50", "Front Endocrinol 2024"),
    ("TXNIP", "Q9H3M7", "TIX100", "NA", 1000.0, "IC50", "Front Endocrinol 2024"),
]

def get_smiles_from_pubchem(compound_name, cas_no=None):
    """从PubChem获取SMILES"""
    try:
        # Method 1: search by name
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_name}/property/CanonicalSMILES/JSON"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props and "CanonicalSMILES" in props[0]:
                smi = props[0]["CanonicalSMILES"]
                if Chem.MolFromSmiles(smi):
                    return smi
    except Exception:
        logger.exception("捕获到异常并继续执行（原 except '' 静默吞掉）")

    # Method 2: search by CAS
    if cas_no and cas_no != "NA":
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas_no}/property/CanonicalSMILES/JSON"
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                props = data.get("PropertyTable", {}).get("Properties", [])
                if props and "CanonicalSMILES" in props[0]:
                    smi = props[0]["CanonicalSMILES"]
                    if Chem.MolFromSmiles(smi):
                        return smi
        except Exception:
            logger.exception("捕获到异常并继续执行（原 except '' 静默吞掉）")

    return None

def main():
    print("=" * 60)
    print("文献挖掘CPI数据整理")
    print("=" * 60)
    
    # 读取现有combined
    combined = pd.read_csv(
        r'd:\铁衰老 绝不重蹈覆辙\L4\results\experimental_actives_detail_cleaned_combined.csv',
        low_memory=False
    )
    print(f"现有数据: {len(combined)} 条, {combined.gene.nunique()} 基因")
    
    # 获取SMILES
    smi_cache = {}
    new_rows = []
    found = 0
    not_found = 0
    
    for gene, uniprot, name, cas, value, atype, source in LITERATURE_CPI:
        if name not in smi_cache:
            smi = get_smiles_from_pubchem(name, cas)
            smi_cache[name] = smi
            if smi:
                found += 1
                print(f"  ✓ {gene} / {name}: SMILES found")
            else:
                not_found += 1
                print(f"  ✗ {gene} / {name}: SMILES NOT found in PubChem")
            time.sleep(0.3)
        
        smi = smi_cache[name]
        if smi:
            new_rows.append({
                'source': 'Literature_' + source.replace(' ', '_'),
                'gene': gene,
                'uniprot_id': uniprot,
                'target_chembl_id': '',
                'target_pref_name': gene,
                'molecule_chembl_id': '',
                'molecule_pref_name': name,
                'canonical_smiles': smi,
                'standard_type': atype,
                'standard_value_nM': value,
                'pchembl_value': 9 - np.log10(value) if value > 0 else np.nan,
                'confidence_score': 4,  # 文献来源，置信度低于数据库
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
    
    print(f"\nSMILES获取: {found} 成功, {not_found} 失败")
    
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # 去重
        before = len(new_df)
        new_df = new_df.drop_duplicates(subset=['gene', 'canonical_smiles'])
        print(f"新记录: {len(new_df)} 条 (去重前 {before})")
        
        # 合并
        combined_new = pd.concat([combined, new_df], ignore_index=True)
        combined_new = combined_new.drop_duplicates(subset=['gene', 'canonical_smiles'])
        
        # 统计
        with open(r'd:\铁衰老 绝不重蹈覆辙\铁衰老基因.txt','r',encoding='utf-8') as f:
            iron_genes = {line.strip() for line in f if line.strip()}
        iron_in = combined_new[combined_new['gene'].isin(iron_genes)]
        new_iron_genes = {row['gene'] for row in new_rows} & iron_genes
        print(f"\n合并后: {len(combined_new)} 条, {combined_new.gene.nunique()} 基因")
        print(f"铁衰老基因: {iron_in.gene.nunique()}/{len(iron_genes)}")
        print(f"新增铁衰老基因: {sorted(new_iron_genes)}")
        
        # 保存
        combined_new.to_csv(
            r'd:\铁衰老 绝不重蹈覆辙\L4\results\experimental_actives_detail_cleaned_combined.csv',
            index=False
        )
        print("\n已保存!")
    else:
        print("\n无新数据可添加")

if __name__ == "__main__":
    main()