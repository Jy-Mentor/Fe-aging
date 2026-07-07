import logging
logger = logging.getLogger(__name__)

"""检查当前combined CPI文件的基因覆盖情况"""
import pandas as pd

df = pd.read_csv(r'd:\铁衰老 绝不重蹈覆辙\L4\results\experimental_actives_detail_cleaned_combined.csv', low_memory=False)
print(f'Total rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
if 'Gene' in df.columns:
    print(f'Unique genes: {df["Gene"].nunique()}')
    print(f'Genes: {sorted(df["Gene"].unique())}')
elif 'gene' in df.columns:
    print(f'Unique genes: {df["gene"].nunique()}')
    print(f'Genes: {sorted(df["gene"].unique())}')
elif 'symbol' in df.columns:
    print(f'Unique genes: {df["symbol"].nunique()}')
    print(f'Genes: {sorted(df["symbol"].unique())}')
else:
    print(f'First 3 rows:')
    print(df.head(3).to_string())