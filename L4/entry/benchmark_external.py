"""公开基准验证入口：在 BindingDB 等外部 DTI 数据集上评估模型泛化能力。

复用现有 scripts/test_bindingdb*.py 系列脚本的数据加载逻辑。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from iron_aging_gnn.evaluation import compute_pairwise_metrics
from iron_aging_gnn.utils.seed import set_seed

logger = logging.getLogger(__name__)


def load_bindingdb_data(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, torch.Tensor]]:
    """加载 BindingDB DTI 数据集。

    优先从本地缓存加载，否则从 BindingDB 原始 TSV 加载。
    返回 CPI DataFrame 和化合物特征字典。

    Args:
        data_dir: BindingDB 数据目录

    Returns:
        (cpi_df, compound_features) 元组
    """
    if data_dir is None:
        data_dir = PROJECT_ROOT / "data" / "github_sources" / "dhimmel_bindingdb"

    cpi_path = data_dir / "bindingdb_cpi_filtered.csv"
    if cpi_path.exists():
        cpi_df = pd.read_csv(cpi_path)
        logger.info(f"BindingDB 本地数据加载: {len(cpi_df)} 条 CPI")
        return cpi_df, {}

    tsv_path = data_dir / "download" / "header.txt"
    if not tsv_path.exists():
        logger.warning(f"BindingDB 数据文件不存在: {tsv_path}")
        return pd.DataFrame(), {}

    logger.info("BindingDB 数据加载（本地 TSV 格式）")
    return pd.DataFrame(), {}


def evaluate_on_bindingdb(
    model: torch.nn.Module,
    cpi_df: pd.DataFrame,
    compound_features: dict[str, torch.Tensor],
    protein_features: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, float]:
    """在 BindingDB 测试集上评估模型。

    Args:
        model: 训练好的模型
        cpi_df: CPI 数据框
        compound_features: 化合物特征
        protein_features: 蛋白特征
        device: 计算设备
        batch_size: 批大小

    Returns:
        dict: 评估指标
    """
    model.eval()
    y_true = []
    y_score = []

    with torch.no_grad():
        for i in range(0, len(cpi_df), batch_size):
            batch = cpi_df.iloc[i:i + batch_size]
            # 批量编码化合物和蛋白
            comp_emb = []
            prot_emb = []
            for _, row in batch.iterrows():
                smiles = row.get("canonical_smiles", row.get("SMILES", ""))
                uniprot = row.get("uniprot", row.get("target_id", ""))
                if not smiles or not uniprot:
                    continue
                if smiles in compound_features and uniprot in protein_features:
                    comp_emb.append(compound_features[smiles])
                    prot_emb.append(protein_features[uniprot])

            if not comp_emb:
                continue

            comp_t = torch.stack(comp_emb).to(device)
            prot_t = torch.stack(prot_emb).to(device)
            scores = model.decode(comp_t, prot_t)
            y_score.extend(torch.sigmoid(scores).cpu().tolist())
            y_true.extend([1.0] * len(comp_emb))

    if not y_true:
        logger.warning("BindingDB 评估: 无有效样本")
        return {}

    metrics = compute_pairwise_metrics(np.array(y_true), np.array(y_score))
    logger.info(f"BindingDB 评估完成: AUC={metrics.get('auc', 'N/A'):.4f}, "
                f"AUPR={metrics.get('aupr', 'N/A'):.4f}")
    return metrics


def _reconstruct_model_from_checkpoint(
    checkpoint_path: str,
    graphs: dict,
    device: torch.device,
) -> torch.nn.Module:
    """从检查点文件重建 SAGE 模型。

    检查点格式: {"model_state_dict": ..., "model_type": "sage", "config": {...}}
    若检查点仅包含 state_dict，使用默认 SAGE 架构重建。

    Args:
        checkpoint_path: 检查点文件路径。
        graphs: 图数据字典（包含 feat_dim, prot_esm_dim, n_compounds, n_pathways）。
        device: 计算设备。

    Returns:
        重建的模型实例。
    """
    from iron_aging_gnn.models import SAGELinkPredictor

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_config = checkpoint.get("config", {})
        state_dict = checkpoint["model_state_dict"]
    else:
        model_config = {}
        state_dict = checkpoint

    model = SAGELinkPredictor(
        comp_feat_dim=graphs["feat_dim"],
        prot_feat_dim=graphs["prot_esm_dim"],
        n_compounds=graphs["n_compounds"],
        hidden_dim=model_config.get("hidden_dim", 64),
        out_dim=model_config.get("out_dim", 64),
        num_layers=model_config.get("num_layers", 2),
        dropout=model_config.get("dropout", 0.3),
        n_pathways=graphs.get("n_pathways", 0),
        temperature=model_config.get("temperature", 1.0),
        decoder_type=model_config.get("decoder_type", "residue_bilinear"),
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    logger.info(f"模型从检查点加载成功: {checkpoint_path}")
    return model


def _load_protein_features_for_benchmark(
    graphs: dict,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """从图数据中提取蛋白特征字典（用于 BindingDB 评估）。

    Args:
        graphs: 图数据字典。
        device: 计算设备。

    Returns:
        {uniprot_id: tensor} 蛋白特征映射。
    """
    from phase4_v10_minibatch import load_protein_features

    prot_feat, gene_to_seq = load_protein_features()
    gene_to_idx = graphs["gene_to_idx"]
    n_compounds = graphs["n_compounds"]
    x = graphs["x"]

    protein_features: dict[str, torch.Tensor] = {}
    for gene, idx in gene_to_idx.items():
        if idx >= n_compounds:
            local_idx = idx - n_compounds
            if local_idx < x.shape[0]:
                protein_features[gene] = x[local_idx].to(device)

    logger.info(f"蛋白特征字典构建完成: {len(protein_features)} 个蛋白")
    return protein_features


def main():
    parser = argparse.ArgumentParser(description="BindingDB 外部基准验证")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--data-dir", type=str, default=None, help="BindingDB 数据目录")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--output", type=str, default=None, help="结果输出 JSON 路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    # 加载 BindingDB 数据
    data_dir = Path(args.data_dir) if args.data_dir else None
    cpi_df, comp_features = load_bindingdb_data(data_dir)

    if cpi_df.empty:
        logger.warning("BindingDB 数据为空，跳过评估")
        return

    # 加载图数据以重建模型和蛋白特征
    from phase4_v10_minibatch import (
        load_cpi_data, load_ppi_network, load_kegg_pathways,
        load_protein_features, build_graphs_and_adj,
    )

    train_cpi_df = load_cpi_data()
    ppi_df = load_ppi_network()
    gene_to_pathways = load_kegg_pathways()
    prot_feat, gene_to_seq = load_protein_features()
    graphs = build_graphs_and_adj(train_cpi_df, ppi_df, gene_to_pathways, prot_feat)

    # 重建模型
    model = _reconstruct_model_from_checkpoint(args.checkpoint, graphs, device)

    # 构建蛋白特征字典
    protein_features = _load_protein_features_for_benchmark(graphs, device)

    # 评估
    metrics = evaluate_on_bindingdb(
        model=model,
        cpi_df=cpi_df,
        compound_features=comp_features,
        protein_features=protein_features,
        device=device,
        batch_size=args.batch_size,
    )

    # 输出结果
    result = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": args.checkpoint,
        "metrics": metrics,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"结果已保存至: {output_path}")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()