import logging
logger = logging.getLogger(__name__)

import pandas as pd
nodes = pd.read_csv(r'd:\铁衰老 绝不重蹈覆辙\L4\data\primekg\nodes.tab', sep='\t', low_memory=False)
print('Node types:', nodes['node_type'].unique())
print('Node sources:', nodes['node_source'].unique())

# Check for drug nodes
drug_nodes = nodes[nodes['node_source'] == 'DrugBank']
print(f'DrugBank nodes: {len(drug_nodes)}')
print('Sample:', drug_nodes[['node_id','node_name']].head(10).to_string())

# Check for our target drugbank IDs
target_ids = ['DB00159','DB00412','DB00197','DB00030','DB01629','DB14511','DB05874','DB14001','DB17641','DB09096','DB09061','DB00958','DB00515','DB09130','DB14002','DB00988','DB01064','DB14009','DB14011','DB00526','DB09221','DB03382','DB12449','DB05088','DB14782','DB00163','DB01593','DB14487','DB14533','DB14548']
for tid in target_ids:
    match = drug_nodes[drug_nodes['node_id'] == tid]
    if len(match) > 0:
        print(f"  {tid}: {match['node_name'].values[0]}")
    else:
        print(f"  {tid}: NOT in nodes")