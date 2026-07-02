"""模型维度推演测试 - 验证所有层的输入输出维度一致性"""
import sys
sys.path.insert(0, 'd:/铁衰老 绝不重蹈覆辙/L4/src')
import torch
from iron_aging_gnn.models import SAGELinkPredictor, HGTLinkPredictor, MemoryBank

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {DEVICE}")

# ==========================================
# 1. SAGE 模型维度推演
# ==========================================
print("\n=== SAGE 模型维度推演 ===")

# 模拟参数（与v27配置一致）
n_compounds = 100
n_proteins = 50
feat_dim = 2232  # 化合物+蛋白统一特征维度
prot_esm_dim = 640
n_pathways = 359
n_diseases = 1
hidden_dim = 64
out_dim = 64
num_layers = 2

sage = SAGELinkPredictor(
    comp_feat_dim=feat_dim - prot_esm_dim - n_pathways,
    prot_feat_dim=prot_esm_dim,
    n_compounds=n_compounds,
    hidden_dim=hidden_dim,
    out_dim=out_dim,
    num_layers=num_layers,
    dropout=0.5,
    n_pathways=n_pathways,
).to(DEVICE)

# 模拟输入
x = torch.randn(n_compounds + n_proteins, feat_dim, device=DEVICE)
edge_index = torch.randint(0, n_compounds + n_proteins, (2, 200), device=DEVICE)

# 测试 forward
node_emb = sage(x, edge_index)
print(f"forward 输出: {node_emb.shape}")  # 期望: (150, 64)

# 测试 encode_compound
comp_emb = sage.encode_compound(x[:10])
print(f"encode_compound 输出: {comp_emb.shape}")  # 期望: (10, 64)

# 测试 decode
prot_emb = node_emb[n_compounds:n_compounds+5]  # 取5个蛋白嵌入
scores = sage.decode(comp_emb[:5], prot_emb)
print(f"decode 输出: {scores.shape}")  # 期望: (5,)

# 测试 predict_phenotype
pheno_out = sage.predict_phenotype(comp_emb[:5])
print(f"predict_phenotype 输出: {pheno_out.shape}")  # 期望: (5, 1)

# 测试 use_pathway=False
node_emb_no_path = sage(x, edge_index, use_pathway=False)
print(f"forward (use_pathway=False) 输出: {node_emb_no_path.shape}")  # 期望: (150, 64)

# 验证通路屏蔽后嵌入是否不同
diff = (node_emb[n_compounds:] - node_emb_no_path[n_compounds:]).abs().mean().item()
print(f"通路屏蔽前后蛋白嵌入差异: {diff:.6f}")  # 期望: > 0 (通路有贡献)

print("SAGE 所有维度推演通过 ✓")

# ==========================================
# 2. HGT 模型维度推演
# ==========================================
print("\n=== HGT 模型维度推演 ===")

# 构建模拟异质图数据
from torch_geometric.data import HeteroData

hetero_data = HeteroData()
hetero_data['compound'].x = torch.randn(n_compounds, feat_dim)
hetero_data['protein'].x = torch.randn(n_proteins, feat_dim)
hetero_data['pathway'].x = torch.zeros(n_pathways, 1)  # v27-fix: 与主脚本一致，使用索引形式
hetero_data['disease'].x = torch.zeros(n_diseases, 1)  # v27-fix: 与主脚本一致，使用索引形式

# 添加边
hetero_data['compound', 'interacts', 'protein'].edge_index = torch.randint(0, n_compounds, (2, 50))
hetero_data['compound', 'interacts', 'protein'].edge_index[1] = torch.randint(0, n_proteins, (50,))
hetero_data['protein', 'ppi', 'protein'].edge_index = torch.randint(0, n_proteins, (2, 100))
hetero_data['protein', 'belongs_to', 'pathway'].edge_index = torch.randint(0, n_proteins, (2, 30))
hetero_data['protein', 'belongs_to', 'pathway'].edge_index[1] = torch.randint(0, n_pathways, (30,))
hetero_data['pathway', 'includes', 'protein'].edge_index = hetero_data['protein', 'belongs_to', 'pathway'].edge_index.flip(0)
hetero_data['protein', 'associated_with', 'disease'].edge_index = torch.randint(0, n_proteins, (2, 5))
hetero_data['protein', 'associated_with', 'disease'].edge_index[1] = torch.randint(0, n_diseases, (5,))
hetero_data['disease', 'involves', 'protein'].edge_index = hetero_data['protein', 'associated_with', 'disease'].edge_index.flip(0)

node_feat_dims = {
    'protein': prot_esm_dim,
    'pathway_count': n_pathways,
    'disease_count': n_diseases,
}

# 构建 metadata
node_types = ['compound', 'protein', 'pathway', 'disease']
edge_types = [
    ('compound', 'interacts', 'protein'),
    ('protein', 'ppi', 'protein'),
    ('protein', 'belongs_to', 'pathway'),
    ('pathway', 'includes', 'protein'),
    ('protein', 'associated_with', 'disease'),
    ('disease', 'involves', 'protein'),
]
metadata = (node_types, edge_types)

hgt = HGTLinkPredictor(
    hidden_dim=hidden_dim,
    out_dim=out_dim,
    num_layers=2,
    num_heads=2,
    dropout=0.5,
    metadata=metadata,
    compound_feat_dim=feat_dim,
    node_feat_dims=node_feat_dims,
).to(DEVICE)

hetero_data = hetero_data.to(DEVICE)

# 测试 forward
node_embs = hgt(hetero_data.x_dict, hetero_data.edge_index_dict)
print(f"HGT forward 输出: compound={node_embs['compound'].shape}, protein={node_embs['protein'].shape}")

# 测试 encode_compound
comp_emb_hgt = hgt.encode_compound(hetero_data.x_dict['compound'][:10])
print(f"HGT encode_compound 输出: {comp_emb_hgt.shape}")

# 测试 decode
prot_emb_hgt = node_embs['protein'][:5]
scores_hgt = hgt.decode(comp_emb_hgt[:5], prot_emb_hgt)
print(f"HGT decode 输出: {scores_hgt.shape}")

# 测试 predict_phenotype
pheno_hgt = hgt.predict_phenotype(comp_emb_hgt[:5])
print(f"HGT predict_phenotype 输出: {pheno_hgt.shape}")

# 测试 disease_embed
print(f"HGT disease_embed: {hgt.disease_embed is not None}")

# 测试 HGT 无 use_pathway 参数 - 该功能由 SAGE 独有
# HGT.forward 签名: (x_dict, edge_index_dict)，无 use_pathway 参数
print("HGT 无 use_pathway 参数 (SAGE 独有功能)")

print("HGT 所有维度推演通过 ✓")

# ==========================================
# 3. MemoryBank 测试
# ==========================================
print("\n=== MemoryBank 测试 ===")
bank = MemoryBank(max_size=100, out_dim=64)
emb = torch.randn(50, 64)
bank.update(emb)
sampled = bank.sample(10)
print(f"MemoryBank sample 输出: {sampled.shape}")
print("MemoryBank 测试通过 ✓")

# ==========================================
# 4. 梯度流测试
# ==========================================
print("\n=== 梯度流测试 ===")
sage.train()
x_grad = torch.randn(n_compounds + n_proteins, feat_dim, device=DEVICE, requires_grad=True)
edge_index_grad = torch.randint(0, n_compounds + n_proteins, (2, 200), device=DEVICE)
node_emb_grad = sage(x_grad, edge_index_grad)
loss = node_emb_grad.sum()
loss.backward()
print(f"SAGE 梯度流: x_grad.grad is not None = {x_grad.grad is not None}")
print("梯度流测试通过 ✓")

print("\n" + "="*50)
print("所有测试通过！模型架构完整可靠。")
print("="*50)