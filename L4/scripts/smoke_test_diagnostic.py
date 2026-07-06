#!/usr/bin/env python3
"""HGT 模型烟雾测试 — 系统性诊断核心前向/反向传播

验证内容:
  1. HGT 前向传播 + 解码器输出
  2. 损失函数正/负样本区分度
  3. 梯度方向正确性（正样本得分应上升，负样本应下降）
  4. prot_map/compound_to_prot_locals 索引一致性
  5. 验证管线 AUC/AUPR 计算正确性
"""

import logging
import sys
import os
import math

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"设备: {DEVICE}")

# ============================================================================
# 1. 模型创建
# ============================================================================
logger.info("=" * 60)
logger.info("1. 创建 HGT 模型 (residue_bilinear)")
logger.info("=" * 60)

from iron_aging_gnn.models.hgt import HGTLinkPredictor
from iron_aging_gnn.models.decoders import ResidueAwareBilinearDecoder

metadata = (["compound", "protein", "pathway", "disease"],
            [("compound", "interacts", "protein"), ("protein", "ppi", "protein"),
             ("protein", "belongs_to", "pathway"), ("compound", "similar_to", "compound")])

node_feat_dims = {"protein": 640, "pathway_count": 359, "disease_count": 1}
model = HGTLinkPredictor(
    hidden_dim=64, out_dim=64, num_heads=2, num_layers=2,
    metadata=metadata, compound_feat_dim=200, node_feat_dims=node_feat_dims,
    decoder_type="residue_bilinear",
).to(DEVICE)

# 注册残基特征（模拟数据）
n_mock_proteins = 20
n_mock_residues = 100
mock_embeddings = torch.randn(n_mock_residues, 640)
mock_offsets = torch.zeros(n_mock_proteins + 1, dtype=torch.long)
mock_lengths = torch.full((n_mock_proteins,), 5, dtype=torch.long)
for i in range(n_mock_proteins):
    mock_offsets[i + 1] = mock_offsets[i] + mock_lengths[i].item()
prot_to_residue_idx = torch.arange(n_mock_proteins, dtype=torch.long)
model.set_residue_features(mock_embeddings, mock_offsets, mock_lengths,
                           prot_to_residue_idx, max_len=10, residue_device="cpu")

# ============================================================================
# 2. 测试前向传播
# ============================================================================
logger.info("-" * 60)
logger.info("2. 测试前向传播 + decode")
logger.info("-" * 60)

n_compounds, n_proteins = 8, 20
x_dict = {
    "compound": torch.randn(n_compounds, 200, device=DEVICE),
    "protein": torch.randn(n_proteins, 640, device=DEVICE),
    "pathway": torch.zeros(0, 1, device=DEVICE),
    "disease": torch.zeros(0, 1, device=DEVICE),
}
edge_index_dict = {
    ("compound", "interacts", "protein"): torch.tensor([[0, 1, 2], [0, 1, 2]], device=DEVICE),
    ("protein", "ppi", "protein"): torch.tensor([[0, 1], [1, 2]], device=DEVICE),
    ("protein", "belongs_to", "pathway"): torch.zeros(2, 0, dtype=torch.long, device=DEVICE),
}

out = model(x_dict, edge_index_dict)
assert "compound" in out and "protein" in out, f"HGT 输出缺少节点类型: {list(out.keys())}"
assert out["compound"].shape == (n_compounds, 64), f"化合物输出维度: {out['compound'].shape}"
assert out["protein"].shape == (n_proteins, 64), f"蛋白输出维度: {out['protein'].shape}"
logger.info(f"  HGT 前向 OK: comp={out['compound'].shape}, prot={out['protein'].shape}")

# 测试 decode
comp_emb = out["compound"]
prot_emb = out["protein"]
prot_indices = torch.arange(n_proteins, device=DEVICE)

scores = model.decode(comp_emb[:1].expand(n_proteins, -1), prot_emb, prot_residue_indices=prot_indices)
logger.info(f"  decode(1 compound, {n_proteins} proteins): {scores.shape}, "
            f"mean={scores.mean().item():.4f}, std={scores.std().item():.4f}")

# 快速双线性（无残基索引）
scores_fast = model.decode(comp_emb[:1].expand(n_proteins, -1), prot_emb, prot_residue_indices=None)
logger.info(f"  decode fast(1 compound, {n_proteins} proteins): {scores_fast.shape}, "
            f"mean={scores_fast.mean().item():.4f}, std={scores_fast.std().item():.4f}")

# ============================================================================
# 3. 测试正/负样本区分 — 随机初始化时的区分度
# ============================================================================
logger.info("-" * 60)
logger.info("3. 随机初始化时的正/负样本区分度")
logger.info("-" * 60)

T = model.temperature
# 构建 5 个正样本对 + 所有其他蛋白作为负样本
pos_compounds = torch.tensor([0, 1, 2, 3, 4], device=DEVICE)
pos_proteins = torch.tensor([0, 1, 2, 3, 4], device=DEVICE)
pos_scores = model.decode(comp_emb[pos_compounds], prot_emb[pos_proteins],
                           prot_residue_indices=pos_proteins) / T

# 对每个正样本化合物，随机挑一个不同的蛋白作为负样本
neg_proteins = torch.tensor([10, 11, 12, 13, 14], device=DEVICE)
neg_scores = model.decode(comp_emb[pos_compounds], prot_emb[neg_proteins],
                           prot_residue_indices=neg_proteins) / T

pos_mean, neg_mean = pos_scores.detach().mean().item(), neg_scores.detach().mean().item()
pos_std, neg_std = pos_scores.detach().std().item(), neg_scores.detach().std().item()
logger.info(f"  初始 pos_scores: mean={pos_mean:.4f}±{pos_std:.4f}")
logger.info(f"  初始 neg_scores: mean={neg_mean:.4f}±{neg_std:.4f}")
logger.info(f"  初始 gap: {pos_mean - neg_mean:.4f} {'OK' if pos_mean > neg_mean else 'WARNING: pos < neg'}")

# ============================================================================
# 4. 测试训练一步 — 梯度方向
# ============================================================================
logger.info("-" * 60)
logger.info("4. 训练一步 — 梯度方向验证")
logger.info("-" * 60)

# 重新初始化模型，确保从头开始
model2 = HGTLinkPredictor(
    hidden_dim=64, out_dim=64, num_heads=2, num_layers=2,
    metadata=metadata, compound_feat_dim=200, node_feat_dims=node_feat_dims,
    decoder_type="residue_bilinear",
).to(DEVICE)
model2.set_residue_features(mock_embeddings, mock_offsets, mock_lengths,
                            prot_to_residue_idx, max_len=10, residue_device="cpu")
for p in model2.parameters():
    if p.dim() >= 2:
        nn.init.xavier_uniform_(p)

# 构建一批数据
n_batch_comps, n_batch_prots = 5, 20
x_dict2 = {
    "compound": torch.randn(n_batch_comps, 200, device=DEVICE),
    "protein": torch.randn(n_batch_prots, 640, device=DEVICE),
    "pathway": torch.zeros(0, 1, device=DEVICE),
    "disease": torch.zeros(0, 1, device=DEVICE),
}
# 正样本对：0-0, 1-1, 2-2, 3-3, 4-4
ei_dict2 = {
    ("compound", "interacts", "protein"): torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], device=DEVICE),
    ("protein", "ppi", "protein"): torch.zeros(2, 0, dtype=torch.long, device=DEVICE),
    ("protein", "belongs_to", "pathway"): torch.zeros(2, 0, dtype=torch.long, device=DEVICE),
}
out2 = model2(x_dict2, ei_dict2)

comp_emb2 = out2["compound"]
prot_emb2 = out2["protein"]

pos_src = torch.tensor([0, 1, 2, 3, 4], device=DEVICE)
pos_dst = torch.tensor([0, 1, 2, 3, 4], device=DEVICE)

# 构建 prot_map (0-based protein index -> local batch index)
prot_map = {i: i for i in range(n_batch_prots)}
comp_sorted = list(range(n_batch_comps))

# 构建 compound_to_prot_locals (global compound -> list of 0-based protein indices)
compound_to_prot_locals = {i: [i] for i in range(n_batch_comps)}  # each compound has 1 positive protein

# 调用实际的损失函数
import importlib.util
spec = importlib.util.spec_from_file_location("phase4_v10_minibatch",
    os.path.join(os.path.dirname(__file__), "phase4_v10_minibatch.py"))
phase4_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase4_mod)
_compute_cpi_loss = phase4_mod._compute_cpi_loss
_validate_hgt = phase4_mod._validate_hgt

opt = torch.optim.AdamW(model2.parameters(), lr=1e-3)

# 前一步的得分
with torch.no_grad():
    scores_before_pos = model2.decode(comp_emb2[pos_src], prot_emb2[pos_dst],
                                       prot_residue_indices=pos_dst) / model2.temperature
    scores_before_neg = model2.decode(comp_emb2[pos_src], prot_emb2[torch.tensor([10,11,12,13,14], device=DEVICE)],
                                       prot_residue_indices=torch.tensor([10,11,12,13,14], device=DEVICE)) / model2.temperature
    logger.info(f"  训练前 pos_scores: {scores_before_pos.detach().mean().item():.4f}")
    logger.info(f"  训练前 neg_scores: {scores_before_neg.detach().mean().item():.4f}")

# 训练一步
loss = _compute_cpi_loss(
    model=model2,
    comp_emb=comp_emb2,
    prot_emb=prot_emb2,
    pos_src=pos_src,
    pos_dst=pos_dst,
    comp_sorted=comp_sorted,
    prot_map=prot_map,
    precomputed_pos={i: {i} for i in range(n_batch_comps)},
    n_compounds=n_batch_comps,
    prot_to_path_neighbors=None,
    epoch=5,
    stage_epochs=10,
    memory_bank=None,
    compound_to_prot_locals=compound_to_prot_locals,
    use_infonce=False,
    bpr_weight=0.4,
    use_curriculum=True,
    use_topology_neg=False,
    focal_gamma=2.0,
    focal_alpha=0.75,
)
logger.info(f"  loss = {loss.item():.6f}")

opt.zero_grad()
loss.backward()

# 检查梯度方向
grad_norm = 0.0
for p in model2.parameters():
    if p.grad is not None:
        grad_norm += p.grad.data.norm(2).item() ** 2
grad_norm = grad_norm ** 0.5
logger.info(f"  梯度范数 = {grad_norm:.4f}")
assert grad_norm > 0, "梯度为 0，训练可能卡住"

opt.step()

# 训练后的得分
with torch.no_grad():
    out2_post = model2(x_dict2, ei_dict2)
    comp_emb2_post = out2_post["compound"]
    prot_emb2_post = out2_post["protein"]
    scores_after_pos = model2.decode(comp_emb2_post[pos_src], prot_emb2_post[pos_dst],
                                      prot_residue_indices=pos_dst) / model2.temperature
    scores_after_neg = model2.decode(comp_emb2_post[pos_src], prot_emb2_post[torch.tensor([10,11,12,13,14], device=DEVICE)],
                                      prot_residue_indices=torch.tensor([10,11,12,13,14], device=DEVICE)) / model2.temperature

pos_change = scores_after_pos.detach().mean().item() - scores_before_pos.detach().mean().item()
neg_change = scores_after_neg.detach().mean().item() - scores_before_neg.detach().mean().item()
logger.info(f"  训练后 pos_scores: {scores_after_pos.detach().mean().item():.4f} (变化: {pos_change:+.4f})")
logger.info(f"  训练后 neg_scores: {scores_after_neg.detach().mean().item():.4f} (变化: {neg_change:+.4f})")

if pos_change > 0 and neg_change < 0:
    logger.info("  ✅ 梯度方向正确: 正样本↑ 负样本↓")
elif pos_change > neg_change:
    logger.info("  ⚠️ 部分正确: 正样本变化 > 负样本变化, 但负样本未下降")
else:
    logger.warning("  ❌ 梯度方向异常: 正样本变化 < 负样本变化, 模型可能学到反向")

# ============================================================================
# 5. 测试验证逻辑
# ============================================================================
logger.info("-" * 60)
logger.info("5. 验证逻辑 — 构建验证集，检查 AUC/AUPR 计算")
logger.info("-" * 60)

# 模拟验证集：10个化合物，每个有1~3个正样本蛋白
n_val_comps = 10
n_val_prots = 50
val_compounds = list(range(n_val_comps))
all_compound_to_pos = {i: {i, i + n_val_comps, i + 2 * n_val_comps} for i in range(n_val_comps)}

# 模拟 hetero_adj
hetero_adj_val = {
    ("compound", "interacts", "protein"): {i: [i, i + n_val_comps, i + 2 * n_val_comps]
                                            for i in range(n_val_comps)},
    ("protein", "ppi", "protein"): {},
    ("protein", "belongs_to", "pathway"): {},
    ("protein", "associated_with", "disease"): {},
    ("disease", "involves", "protein"): {},
}

# 创建一个有意义的模型：如果化合物索引与蛋白索引匹配，则为正样本
class MockHGT(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_dim = 64
        self.temperature = nn.Parameter(torch.tensor(5.0))
        self.pathway_embed = nn.Embedding(1, 64)
        self.comp_proj = nn.Linear(200, 64)
        self.prot_proj = nn.Linear(640, 64)
        self.out_proj = nn.Linear(64, 64)
        self.pheno_head = nn.Linear(64, 1)
        
    def forward(self, x_dict, edge_index_dict):
        return {
            "compound": self.out_proj(torch.randn(len(x_dict.get("compound", torch.zeros(0))), 64, device=DEVICE)),
            "protein": self.out_proj(torch.randn(len(x_dict.get("protein", torch.zeros(0))), 64, device=DEVICE)),
            "pathway": torch.zeros(0, 64, device=DEVICE),
            "disease": torch.zeros(0, 64, device=DEVICE),
        }
    
    def decode(self, comp_emb, prot_emb, prot_residue_indices=None):
        # 故意给正样本（索引匹配）更高分，验证能否正确评估
        if prot_residue_indices is not None and comp_emb.shape[0] == 1:
            c_idx = 0  # 只有一个化合物
            scores = torch.zeros(prot_emb.shape[0], device=DEVICE)
            # 给索引匹配的蛋白高分
            match = prot_residue_indices < n_val_comps
            scores[match] = 2.0
            scores[~match] = -1.0
            return scores
        return (comp_emb * prot_emb).sum(dim=-1) * 0.1

    def predict_phenotype(self, x):
        return self.pheno_head(x)

    def free_residue_features(self):
        pass

    def set_residue_features(self, *args, **kwargs):
        pass

mock_model = MockHGT().to(DEVICE)

# 构造 hetero_data
from torch_geometric.data import HeteroData
hetero_data_val = HeteroData()
hetero_data_val["compound"].x = torch.randn(n_val_comps + 40, 200)
hetero_data_val["protein"].x = torch.randn(n_val_prots, 640)

val_result = _validate_hgt(
    mock_model, hetero_data_val, val_compounds,
    all_compound_to_pos, n_val_comps, n_val_prots,
    hetero_adj=hetero_adj_val,
)
logger.info(f"  Mock 模型验证结果: AUC={val_result.get('auc', 'N/A'):.4f}, "
            f"AUPR={val_result.get('aupr', 'N/A'):.4f}")

# 理想模型应该 AUC ≈ 1.0, AUPR ≈ 1.0（因为正样本分数 > 负样本）
auc = val_result.get("auc", 0.5)
if auc > 0.8:
    logger.info(f"  ✅ 验证管线正确 (AUC={auc:.4f} > 0.8)")
elif auc > 0.5:
    logger.info(f"  ⚠️ 验证管线较弱 (AUC={auc:.4f})")
else:
    logger.warning(f"  ❌ 验证管线可能异常 (AUC={auc:.4f} <= 0.5)")

# ============================================================================
# 6. 检查复合损失中的 mask 构建
# ============================================================================
logger.info("-" * 60)
logger.info("6. 检查损失函数中的 mask 构建正确性")
logger.info("-" * 60)

# 创建已知的正/负样本数据
n_c, n_p = 3, 10
test_comp_emb = torch.randn(n_c, 64, device=DEVICE)
test_prot_emb = torch.randn(n_p, 64, device=DEVICE)
test_pos_src = torch.tensor([0, 1, 2], device=DEVICE)
test_pos_dst = torch.tensor([0, 1, 2], device=DEVICE)
test_comp_sorted = list(range(n_c))
# 正确的 prot_map: 0-based protein index -> local batch index
test_prot_map = {i: i for i in range(n_p)}
# 正确的 compound_to_prot_locals: global compound -> [0-based protein indices]
test_c2p = {0: [0], 1: [1], 2: [2]}

try:
    test_loss = _compute_cpi_loss(
        model=model2,
        comp_emb=test_comp_emb,
        prot_emb=test_prot_emb,
        pos_src=test_pos_src,
        pos_dst=test_pos_dst,
        comp_sorted=test_comp_sorted,
        prot_map=test_prot_map,
        precomputed_pos={0: {0}, 1: {1}, 2: {2}},
        n_compounds=n_c,
        prot_to_path_neighbors=None,
        epoch=5,
        stage_epochs=10,
        memory_bank=None,
        compound_to_prot_locals=test_c2p,
        use_infonce=False,
        bpr_weight=0.4,
        use_curriculum=True,
        use_topology_neg=False,
        focal_gamma=2.0,
        focal_alpha=0.75,
    )
    logger.info(f"  训练 loss = {test_loss.item():.6f}")
    
    # 使用错误索引验证 mask 是否为空
    wrong_c2p = {}  # 空的 compound_to_prot_locals
    test_loss_wrong = _compute_cpi_loss(
        model=model2,
        comp_emb=test_comp_emb,
        prot_emb=test_prot_emb,
        pos_src=test_pos_src,
        pos_dst=test_pos_dst,
        comp_sorted=test_comp_sorted,
        prot_map=test_prot_map,
        precomputed_pos={0: {0}, 1: {1}, 2: {2}},
        n_compounds=n_c,
        prot_to_path_neighbors=None,
        epoch=5,
        stage_epochs=10,
        memory_bank=None,
        compound_to_prot_locals=wrong_c2p,  # 空 → mask 全 0
        use_infonce=False,
        bpr_weight=0.4,
        use_curriculum=True,
        use_topology_neg=False,
        focal_gamma=2.0,
        focal_alpha=0.75,
    )
    logger.info(f"  空 mask 时的 loss = {test_loss_wrong.item():.6f} "
                f"(应高于正确 mask 的 {test_loss.item():.6f})")
    
    if test_loss_wrong.item() > test_loss.item() * 0.9:
        logger.warning("  ⚠️ 空 mask 与正确 mask 的 loss 接近，mask 可能不生效")
    else:
        logger.info("  ✅ mask 构建正确，空 mask 显著改变 loss")
except Exception as e:
    logger.error(f"  ❌ 损失函数执行异常: {e}", exc_info=True)

# ============================================================================
# 总结
# ============================================================================
logger.info("=" * 60)
logger.info("烟雾测试完成")
logger.info("=" * 60)
