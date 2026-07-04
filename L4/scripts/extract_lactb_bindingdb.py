"""从dhimmel/bindingdb提取LACTB数据并合并到combined文件"""
import gzip
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

L4_ROOT = Path(r"d:\铁衰老 绝不重蹈覆辙\L4")
COMBINED_FILE = L4_ROOT / "results" / "experimental_actives_detail_cleaned_combined.csv"
BINDINGDB_FILE = L4_ROOT / "data" / "github_sources" / "dhimmel_bindingdb" / "data" / "binding.tsv.gz"

# 读取BindingDB数据
logger.info(f"Loading BindingDB data: {BINDINGDB_FILE}")
with gzip.open(BINDINGDB_FILE, 'rt') as f:
    bdb = pd.read_csv(f, sep='\t', low_memory=False)

# 提取LACTB (P83111)的数据
lactb = bdb[bdb['uniprot'].str.upper() == 'P83111'].copy()
logger.info(f"LACTB records: {len(lactb)}")

# 读取combined文件
df = pd.read_csv(COMBINED_FILE, low_memory=False)
logger.info(f"Existing rows: {len(df)}")

# 需要获取SMILES - 从bindingdb的compound表中获取
# BindingDB的binding表有bindingdb_id，可以关联到compound表
# 但当前只有binding.tsv.gz，没有SMILES信息
# 我们需要从ChEMBL获取SMILES（因为source是ChEMBL）

# 读取ChEMBL数据（如果存在）
chembl_file = L4_ROOT / "results" / "chembl_cpi_combined.csv"
if chembl_file.exists():
    chembl = pd.read_csv(chembl_file, low_memory=False)
    logger.info(f"ChEMBL data: {len(chembl)} rows")
else:
    logger.warning("ChEMBL file not found, trying alternative source")
    chembl = None

# 从bindingdb数据中提取能用的字段
# 注意：binding.tsv.gz没有SMILES，只有bindingdb_id和affinity
# 我们需要通过bindingdb_id从其他来源获取SMILES
print("\nLACTB BindingDB records (no SMILES in this file):")
print(lactb[['bindingdb_id', 'measure', 'affinity_nM', 'source', 'pubmed']].to_string())

# 尝试从现有combined文件中查找是否有LACTB的BindingDB数据
existing_lactb = df[df['gene'] == 'LACTB']
print(f"\nExisting LACTB records in combined file: {len(existing_lactb)}")

# 由于binding.tsv.gz没有SMILES，我们需要通过BindingDB API或PubChem获取
# 这里先记录需要获取SMILES的化合物列表
print("\n需要获取SMILES的BindingDB IDs:")
print(lactb['bindingdb_id'].unique().tolist())