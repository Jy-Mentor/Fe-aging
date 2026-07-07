import logging
logger = logging.getLogger(__name__)

"""
交叉引用 CPIExtract 数据与铁衰老缺失基因
"""
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
CPIEXTRACT_DIR = PROJECT_ROOT / "L4" / "data" / "github_sources" / "CPIExtract"

# 铁衰老全部核心基因
CORE_GENES = [
    "ABCC1", "ACVR1B", "ACSL4", "ALOX15", "ATF3", "ATG3",
    "BAP1", "BCL6", "BRD7",
    "CAVIN1", "CD74", "CD82", "CDO1", "COX7A1", "CTSB", "CXCL10",
    "DPEP1", "DPP4", "DUOX1", "DYRK1A",
    "E2F1", "E2F3", "EBF3", "EDN1", "EGR1", "EMP1", "EPHA2", "EPHA4", "ERN1",
    "FBXO31", "FOSL1",
    "GMFB",
    "HBP1", "HERPUD1", "HIF1A", "HMGB1", "HMOX1",
    "ICA1", "IFNG", "IGFBP7", "IL1B", "IL6", "IRF1", "IRF7", "IRF9",
    "KDM6B", "KEAP1", "KLF6",
    "LACTB", "LCN2", "LGMN", "LIFR", "LOX", "LPCAT3",
    "MAP3K14", "MAPK1", "MAPK14", "MCU", "MEN1", "MPO",
    "NLRP3", "NOX4", "NR1D1", "NR2F2", "NUAK2",
    "PADI4", "PDE4B", "PPP2R2B", "PRKD1", "PTBP1", "PTGS2",
    "RBM3", "RUNX3",
    "S100A8", "SAT1", "SETD7", "SLAMF8", "SLC1A5", "SMARCB1", "SMURF2", "SNCA",
    "SOCS1", "SOCS2", "SOD1", "SP1", "SPATA2",
    "TBX2", "TFRC", "TLR4", "TNFAIP1", "TNFAIP3", "TXNIP",
    "WNT5A", "WWTR1",
    "YAP1",
    "ZEB1",
]

# 读取 C2P (Compound to Protein)
c2p = pd.read_csv(CPIEXTRACT_DIR / "data" / "output" / "C2P.csv")
print(f"C2P rows: {len(c2p)}")
print(f"C2P columns: {list(c2p.columns)}")
print(f"C2P unique genes: {c2p['hgnc_symbol'].nunique()}")

# 读取 P2C (Protein to Compound)
p2c = pd.read_csv(CPIEXTRACT_DIR / "data" / "output" / "P2C.csv")
print(f"\nP2C rows: {len(p2c)}")
print(f"P2C columns: {list(p2c.columns)}")
print(f"P2C unique genes: {p2c['hgnc_symbol'].nunique()}")

# 合并基因集
all_genes_c2p = set(c2p['hgnc_symbol'].dropna().str.upper())
all_genes_p2c = set(p2c['hgnc_symbol'].dropna().str.upper())
all_genes = all_genes_c2p | all_genes_p2c

core_upper = {g.upper() for g in CORE_GENES}
hits = core_upper & all_genes
missing = core_upper - all_genes

print(f"\n=== 铁衰老基因命中情况 ===")
print(f"总基因数: {len(CORE_GENES)}")
print(f"CPIExtract 命中: {len(hits)}")
print(f"仍缺失: {len(missing)}")
print(f"命中基因: {sorted(hits)}")
print(f"缺失基因: {sorted(missing)}")

# 详细查看命中基因的数据
print(f"\n=== 命中基因详细数据 ===")
for gene in sorted(hits):
    in_c2p = gene in all_genes_c2p
    in_p2c = gene in all_genes_p2c
    
    c2p_count = 0
    p2c_count = 0
    c2p_avg = "N/A"
    
    if in_c2p:
        sub = c2p[c2p['hgnc_symbol'].str.upper() == gene]
        c2p_count = len(sub)
        # 过滤有 pchembl 值的
        with_pchembl = sub[sub['ave_pchembl'].apply(lambda x: isinstance(x, (int, float)) and x > 0)]
        if len(with_pchembl) > 0:
            c2p_avg = f"{with_pchembl['ave_pchembl'].mean():.2f}"
    
    if in_p2c:
        sub = p2c[p2c['hgnc_symbol'].str.upper() == gene]
        p2c_count = len(sub)
    
    print(f"  {gene}: C2P={c2p_count}, P2C={p2c_count}, avg_pChEMBL={c2p_avg}")

# 提取命中基因的详细CPI数据
print(f"\n=== 提取命中基因CPI数据 ===")
all_hits_data = []
for gene in sorted(hits):
    if gene in all_genes_c2p:
        sub = c2p[c2p['hgnc_symbol'].str.upper() == gene].copy()
        sub['gene'] = gene
        sub['source'] = 'CPIExtract_C2P'
        all_hits_data.append(sub)
    if gene in all_genes_p2c:
        sub = p2c[p2c['hgnc_symbol'].str.upper() == gene].copy()
        sub['gene'] = gene
        sub['source'] = 'CPIExtract_P2C'
        all_hits_data.append(sub)

if all_hits_data:
    combined = pd.concat(all_hits_data, ignore_index=True)
    # 去重
    combined = combined.drop_duplicates(subset=['gene', 'isomeric_smiles'])
    print(f"合并后总记录数: {len(combined)}")
    print(f"唯一基因数: {combined['gene'].nunique()}")
    print(f"唯一化合物数: {combined['isomeric_smiles'].nunique()}")
    
    # 保存
    out_path = PROJECT_ROOT / "L4" / "data" / "github_sources" / "cpiextract_hits.csv"
    combined.to_csv(out_path, index=False)
    print(f"已保存到: {out_path}")
else:
    print("无命中基因数据")