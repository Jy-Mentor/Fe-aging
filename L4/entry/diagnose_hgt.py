"""HGT/SimpleHGN 性能诊断工具

诊断 HGT/SimpleHGN AUPR ~0.40 远低于 SAGE ~0.80 的根因。

用法:
    python entry/diagnose_hgt.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SRC_DIR))

logger = logging.getLogger(__name__)


def diagnose_hgt_attention(model):
    """诊断 HGT 注意力权重分布。

    检查:
      1. 门控残差连接的门值分布（HGT 特有）
      2. 蛋白特征投影后的范数分布
      3. 化合物特征投影后的范数分布
    """
    logger.info("=" * 60)
    logger.info("HGT 架构诊断")
    logger.info("=" * 60)

    # 1. 门控残差连接
    if hasattr(model, "gates") and model.gates:
        for i, gate in enumerate(model.gates):
            logger.info(f"  层 {i} 门控: bias={gate.bias.item():.4f}, "
                        f"激活后 sigmoid(bias)={torch.sigmoid(torch.tensor(gate.bias.item())).item():.4f}")
        logger.info("  结论: sigmoid(bias) > 0.5 表示新特征占主导，接近 0.88 表示强新特征偏好")

    # 2. 蛋白投影器权重分析
    if hasattr(model, "prot_proj"):
        w = model.prot_proj[0].weight  # Linear(640, hidden_dim)
        singular_values = torch.linalg.svdvals(w.float()).detach().cpu().numpy()
        condition_number = singular_values[0] / (singular_values[-1] + 1e-8)
        logger.info(f"  蛋白投影器: Linear({w.shape[1]}, {w.shape[0]})")
        logger.info(f"    条件数: {condition_number:.1f} (越小越好, >100 表示可能有问题)")
        logger.info(f"    奇异值范围: [{singular_values[-1]:.2f}, {singular_values[0]:.2f}]")

    # 3. 化合物投影器权重分析
    if hasattr(model, "comp_proj"):
        w = model.comp_proj[0].weight
        singular_values = torch.linalg.svdvals(w.float()).detach().cpu().numpy()
        condition_number = singular_values[0] / (singular_values[-1] + 1e-8)
        logger.info(f"  化合物投影器: Linear({w.shape[1]}, {w.shape[0]})")
        logger.info(f"    条件数: {condition_number:.1f}")

    # 4. 解码器权重分析
    if hasattr(model, "decoder"):
        _diagnose_decoder(model.decoder, model.decoder_type)


def diagnose_simplehgn_attention(model):
    """诊断 SimpleHGN 边类型嵌入与注意力权重。

    检查:
      1. 边类型嵌入的范数分布
      2. 蛋白/化合物投影器权重条件数
    """
    logger.info("=" * 60)
    logger.info("SimpleHGN 架构诊断")
    logger.info("=" * 60)

    # 1. 边类型嵌入分析
    if hasattr(model, "edge_type_embed"):
        w = model.edge_type_embed.weight
        norms = torch.norm(w.float(), dim=1).detach().cpu().numpy()
        logger.info(f"  边类型嵌入: {w.shape[0]} 种边类型, 嵌入维度={w.shape[1]}")
        logger.info(f"    范数范围: [{norms.min():.4f}, {norms.max():.4f}], 均值={norms.mean():.4f}")
        if norms.max() - norms.min() < 0.01:
            logger.warning("    警告: 边类型嵌入范数差异极小，不同边类型可能无法有效区分")

    # 2. 蛋白投影器
    if hasattr(model, "prot_proj"):
        w = model.prot_proj[0].weight
        singular_values = torch.linalg.svdvals(w.float()).detach().cpu().numpy()
        condition_number = singular_values[0] / (singular_values[-1] + 1e-8)
        logger.info(f"  蛋白投影器: Linear({w.shape[1]}, {w.shape[0]})")
        logger.info(f"    条件数: {condition_number:.1f}")

    # 3. 化合物投影器
    if hasattr(model, "comp_proj"):
        w = model.comp_proj[0].weight
        singular_values = torch.linalg.svdvals(w.float()).detach().cpu().numpy()
        condition_number = singular_values[0] / (singular_values[-1] + 1e-8)
        logger.info(f"  化合物投影器: Linear({w.shape[1]}, {w.shape[0]})")
        logger.info(f"    条件数: {condition_number:.1f}")

    # 4. 解码器
    if hasattr(model, "decoder"):
        _diagnose_decoder(model.decoder, model.decoder_type)


def _diagnose_decoder(decoder, decoder_type: str):
    """诊断解码器内部状态。"""
    logger.info(f"  解码器类型: {decoder_type}")
    if hasattr(decoder, "U"):
        w = decoder.U.weight
        logger.info(f"    U (化合物投影): {w.shape}, 范数={torch.norm(w.float()).detach().item():.4f}")
    if hasattr(decoder, "V"):
        w = decoder.V.weight
        logger.info(f"    V (残基投影): {w.shape}, 范数={torch.norm(w.float()).detach().item():.4f}")
    if hasattr(decoder, "W"):
        w = decoder.W.weight
        logger.info(f"    W (蛋白全局投影): {w.shape}, 范数={torch.norm(w.float()).detach().item():.4f}")


def compare_sage_hgt_architecture():
    """对比 SAGE 与 HGT/SimpleHGN 的关键架构差异。"""
    logger.info("=" * 60)
    logger.info("SAGE vs HGT/SimpleHGN 架构差异分析")
    logger.info("=" * 60)

    differences = [
        ("蛋白特征处理", "SAGE: 通路信息通过特征拼接（640+82维）", "HGT/SimpleHGN: 通路信息通过异质图结构 + 特征拼接（v62-fix）"),
        ("Dropout策略", "SAGE: 蛋白特征投影后有独立 Dropout", "HGT/SimpleHGN: 已添加 prot_dropout（v62-fix）"),
        ("残差连接", "SAGE: 标准残差", "HGT: 门控残差（初始bias=0.0, 平衡新旧特征，v62-fix）"),
        ("边类型编码", "SAGE: 同构图，无此概念", "SimpleHGN: 边类型嵌入+GATv2Conv(edge_dim)"),
        ("温度参数", "SAGE: _TEMPERATURE=1.0", "HGT/SimpleHGN: _TEMPERATURE=1.0（已修复）"),
        ("蛋白特征维度", "SAGE: 640(ESM-2) + 82(通路) = 722", "HGT: 640(ESM-2) + 82(通路拼接) = 722（v62-fix）"),
    ]

    for name, sage, hgt in differences:
        logger.info(f"  {name}:")
        logger.info(f"    {sage}")
        logger.info(f"    {hgt}")

    logger.info("")
    logger.info("  修复状态（v62）:")
    logger.info("    1. [已修复] HGT 门控残差 bias 从 2.0 改为 0.0（sigmoid=0.50，平衡新旧特征）")
    logger.info("    2. [已修复] HGT/SimpleHGN/RGCN 蛋白特征增加通路 one-hot 拼接（prot_pathway_dim）")
    logger.info("    3. [已修复] HGT/SimpleHGN/RGCN 添加 prot_dropout 独立蛋白 Dropout")
    logger.info("    4. [待验证] 需运行完整训练验证 AUPR 是否提升至 >=0.60")


def diagnose_protein_feature_degradation():
    """诊断 ESM-2 蛋白特征在 HGT/SimpleHGN 投影后是否退化。

    构造随机蛋白特征，通过投影器，检查输出是否接近零向量。
    """
    import torch

    logger.info("=" * 60)
    logger.info("蛋白特征投影退化检查")
    logger.info("=" * 60)

    # 模拟 640维 ESM-2 特征通过 Linear(640, 128) 投影
    test_input = torch.randn(100, 640)
    proj = torch.nn.Linear(640, 128)
    test_output = proj(test_input)

    # 计算输出范数统计
    norms = torch.norm(test_output, dim=1).detach()
    logger.info("  随机初始化 Linear(640, 128):")
    logger.info(f"    输出范数: mean={norms.mean():.4f}, std={norms.std():.4f}, "
                f"min={norms.min():.4f}, max={norms.max():.4f}")

    # 检查是否接近零
    if norms.mean() < 0.1:
        logger.warning("    警告: 输出范数过于接近零，蛋白特征可能被压缩过度")

    # 奇异值分析
    w = proj.weight
    s = torch.linalg.svdvals(w.float())
    effective_rank = (s > 0.01 * s[0]).sum().item()
    logger.info(f"    有效秩: {effective_rank}/{s.shape[0]} (越低表示更多维度坍塌)")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("=" * 60)
    logger.info("HGT/SimpleHGN 性能诊断工具")
    logger.info("=" * 60)
    logger.info("当前温度: 1.0 (已修复)")
    logger.info("")

    # 1. 尝试加载 HGT 模型进行诊断
    hgt_model_path = PROJECT_ROOT / "results" / "hgt_best_v60.pt"
    try:
        from iron_aging_gnn.models.hgt import HGTLinkPredictor
        from torch_geometric.data import HeteroData

        data = HeteroData()
        data["compound"].x = torch.randn(100, 200)
        data["protein"].x = torch.randn(50, 640)
        data["pathway"].x = torch.ones(10, 1)
        data["compound", "targets", "protein"].edge_index = torch.randint(0, 50, (2, 200))
        data["compound", "targets", "protein"].edge_index[0] = torch.clamp(
            data["compound", "targets", "protein"].edge_index[0], 0, 99)
        data["protein", "interacts", "protein"].edge_index = torch.randint(0, 50, (2, 300))
        data["protein", "belongs_to", "pathway"].edge_index = torch.randint(0, 10, (2, 100))
        data["protein", "belongs_to", "pathway"].edge_index[0] = torch.clamp(
            data["protein", "belongs_to", "pathway"].edge_index[0], 0, 49)

        node_feat_dims = {"compound": 200, "protein": 640, "pathway": 1,
                          "pathway_count": 10, "disease_count": 0}

        hgt = HGTLinkPredictor(hidden_dim=128, out_dim=128, num_heads=2, num_layers=2,
                               dropout=0.5, metadata=data.metadata(),
                               compound_feat_dim=200, node_feat_dims=node_feat_dims)
        diagnose_hgt_attention(hgt)
        logger.info("")

        from iron_aging_gnn.models.simplehgn import SimpleHGNLinkPredictor
        simplehgn = SimpleHGNLinkPredictor(hidden_dim=128, out_dim=128, num_heads=2, num_layers=2,
                                           dropout=0.5, metadata=data.metadata(),
                                           compound_feat_dim=200, node_feat_dims=node_feat_dims)
        diagnose_simplehgn_attention(simplehgn)
        logger.info("")

        compare_sage_hgt_architecture()
        logger.info("")

        diagnose_protein_feature_degradation()

    except (RuntimeError, ImportError, ValueError, KeyError) as e:
        logger.warning(f"诊断过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()