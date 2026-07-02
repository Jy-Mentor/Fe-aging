import os
files = [
    r'd:\铁衰老 绝不重蹈覆辙\L4\results_v10_minibatch\cpi_supplement_v25_cleaned.csv',
    r'd:\铁衰老 绝不重蹈覆辙\L4\results\bindingdb_active_compounds_cleaned.csv',
    r'd:\铁衰老 绝不重蹈覆辙\L1\results\ppi_network_extended_significant_edges_dedup.csv',
    r'd:\铁衰老 绝不重蹈覆辙\L3\results\tcm_compound_pool_v21_Alevel_leakage_checked.csv',
]
for f in files:
    if os.path.exists(f):
        n = sum(1 for _ in open(f, 'r', encoding='utf-8'))
        print(f'OK: {f} ({n} lines)')
    else:
        print(f'MISSING: {f}')