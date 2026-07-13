"""入口脚本: TCM 候选化合物预测

加载已训练模型，对 TCM 化合物池进行靶标预测。

用法:
    python entry/predict.py --checkpoint results_v10_minibatch/sage_best.pt
    python entry/predict.py --checkpoint results_v10_minibatch/sage_best.pt --mc-samples 30
    python entry/predict.py --checkpoint results_v10_minibatch/sage_best.pt --top-k 100
    python entry/predict.py --checkpoint results_v10_minibatch/sage_best.pt --output results/predictions.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from phase4_v10_minibatch import (
    predict_tcm,
    load_tcm_pool,
    load_cpi_data,
    load_ppi_network,
    load_kegg_pathways,
    load_protein_features,
    build_graphs_and_adj,
    build_compound_features,
    _log_step_time,
)

logger = logging.getLogger(__name__)


def _reconstruct_models_from_checkpoint(
    checkpoint_path: str,
    graphs: dict,
    device: torch.device,
) -> tuple:
    """从检查点文件重建 SAGE 和 HGT/SimpleHGN 模型。"""
    from iron_aging_gnn.models import SAGELinkPredictor

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = {}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_config = checkpoint.get("config", {})
        sage_state = checkpoint["model_state_dict"]
    else:
        sage_state = checkpoint

    sage_model = SAGELinkPredictor(
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
    sage_model.load_state_dict(sage_state, strict=False)
    sage_model.to(device)
    sage_model.eval()
    logger.info(f"SAGE 模型从检查点加载成功: {checkpoint_path}")

    hgt_model = None
    simplehgn_model = None

    return sage_model, hgt_model, simplehgn_model


def main():
    parser = argparse.ArgumentParser(
        description="铁衰老 GNN TCM 候选化合物预测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python entry/predict.py --checkpoint results_v10_minibatch/sage_best.pt
  python entry/predict.py --checkpoint results_v10_minibatch/sage_best.pt --mc-samples 30
  python entry/predict.py --checkpoint results_v10_minibatch/sage_best.pt --top-k 100
        """,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型检查点路径")
    parser.add_argument("--mc-samples", type=int, default=30,
                        help="MC Dropout 采样次数（默认: 30）")
    parser.add_argument("--top-k", type=int, default=500,
                        help="输出 top-k 候选（默认: 500）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 CSV 文件路径（默认: results_v10_minibatch/tcm_predictions.csv）")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("=" * 60)
    logger.info("TCM 候选化合物预测")
    logger.info("=" * 60)

    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    logger.info(">>> 加载数据")
    t0 = time.time()
    cpi_df = load_cpi_data()
    ppi_df = load_ppi_network()
    gene_to_pathways = load_kegg_pathways()
    prot_feat, gene_to_seq = load_protein_features()
    tcm_df = load_tcm_pool()
    t0 = _log_step_time(t0, "数据加载完成")

    logger.info(">>> 构建图结构")
    t0 = time.time()
    graphs = build_graphs_and_adj(cpi_df, ppi_df, gene_to_pathways, prot_feat)
    t0 = _log_step_time(t0, "图构建完成")

    logger.info(">>> 加载模型")
    t0 = time.time()
    sage_model, hgt_model, simplehgn_model = _reconstruct_models_from_checkpoint(
        args.checkpoint, graphs, device)
    t0 = _log_step_time(t0, "模型加载完成")

    tcm_smiles_col = "SMILES_std" if "SMILES_std" in tcm_df.columns else (
        "SMILES" if "SMILES" in tcm_df.columns else "canonical_smiles")
    tcm_smiles = tcm_df[tcm_smiles_col].dropna().tolist()
    logger.info(f"TCM 化合物: {len(tcm_smiles)} 个 SMILES")

    all_train_smiles = list(graphs["smi_to_idx"].keys())
    _, cp_mean, cp_std, cp_col_mean = build_compound_features(all_train_smiles)
    compound_stats = (cp_mean, cp_std, cp_col_mean)

    all_target_genes = sorted(graphs["gene_to_idx"].keys())
    logger.info(f"预测靶标: {len(all_target_genes)} 个基因")

    if args.output is None:
        output_path = graphs.get("results_dir", PROJECT_ROOT / "results_v10_minibatch") / "tcm_predictions.csv"
    else:
        output_path = Path(args.output)

    logger.info(">>> 执行 TCM 预测")
    t0 = time.time()
    predictions = predict_tcm(
        sage_model= sage_model,
        hgt_model=hgt_model,
        simplehgn_model=simplehgn_model,
        graphs=graphs,
        tcm_smiles=tcm_smiles,
        target_genes=all_target_genes,
        compound_stats=compound_stats,
        mc_samples=args.mc_samples,
    )
    t0 = _log_step_time(t0, "TCM 预测完成")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    logger.info(f"预测结果已保存: {output_path} ({len(predictions)} 行)")

    total_time = time.time() - start_time
    logger.info(f"预测总耗时: {total_time:.1f}s")
    logger.info("TCM 预测完成。")


if __name__ == "__main__":
    main()