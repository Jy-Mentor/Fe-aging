"""入口脚本: 构建异质图数据

从原始数据（CPI/PPI/KEGG/蛋白特征）构建图数据并缓存。

用法:
    python entry/build_graph.py                   # 使用缓存构建
    python entry/build_graph.py --force-rebuild   # 强制重建
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from phase4_v10_minibatch import (
    load_cpi_data,
    load_ppi_network,
    load_kegg_pathways,
    load_protein_features,
    build_graphs_and_adj,
    _log_step_time,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="构建异质图数据")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="强制重建图（忽略缓存）")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    import time

    logger.info("=" * 60)
    logger.info("构建异质图数据")
    logger.info("=" * 60)

    start_time = time.time()

    logger.info(">>> 加载数据")
    t0 = time.time()
    cpi_df = load_cpi_data()
    ppi_df = load_ppi_network()
    gene_to_pathways = load_kegg_pathways()
    prot_feat, gene_to_seq = load_protein_features()
    t0 = _log_step_time(t0, "数据加载完成")

    logger.info(">>> 构建图结构")
    t0 = time.time()
    graphs = build_graphs_and_adj(cpi_df, ppi_df, gene_to_pathways, prot_feat,
                                  force_rebuild=args.force_rebuild)
    t0 = _log_step_time(t0, "图构建完成")

    n_nodes = graphs["n_compounds"] + graphs["n_proteins"]
    n_edges_homo = graphs["homo_edge_index"].shape[1]
    logger.info(f"图结构完整性: {n_nodes} 节点 ({graphs['n_compounds']}c + {graphs['n_proteins']}p), "
                f"{n_edges_homo} 边, feat_dim={graphs['feat_dim']}")

    total_time = time.time() - start_time
    logger.info(f"图构建总耗时: {total_time:.1f}s")
    logger.info("图构建完成。")


if __name__ == "__main__":
    main()