"""入口脚本: 一键流水线

按顺序执行: 图构建 → 训练 → 评估 → 预测

用法:
    python entry/run_pipeline.py                                    # 完整流水线
    python entry/run_pipeline.py --model sage                       # 仅 SAGE
    python entry/run_pipeline.py --decoder residue_bilinear         # 残基感知解码器
    python entry/run_pipeline.py --skip-build                       # 跳过图构建
    python entry/run_pipeline.py --sage-epochs 5 --hgt-epochs 3     # 快速测试
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from phase4_v10_minibatch import main as _phase4_main

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="铁衰老 GNN 一键流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python entry/run_pipeline.py                                    # 完整流水线
  python entry/run_pipeline.py --model sage                       # 仅 SAGE
  python entry/run_pipeline.py --decoder residue_bilinear         # 残基感知解码器
  python entry/run_pipeline.py --skip-build                       # 跳过图构建
  python entry/run_pipeline.py --sage-epochs 5 --hgt-epochs 3     # 快速测试
        """,
    )
    parser.add_argument("--skip-build", action="store_true",
                        help="跳过图构建（使用缓存）")
    parser.add_argument("--model", type=str, default=None,
                        choices=["sage", "hgt", "simplehgn", "all"],
                        help="仅训练指定模型")
    parser.add_argument("--decoder", type=str, default=None,
                        choices=["mlp", "dot", "bilinear", "residue_bilinear"],
                        help="解码器类型")
    parser.add_argument("--sage-epochs", type=int, default=None,
                        help="SAGE epoch 数")
    parser.add_argument("--hgt-epochs", type=int, default=None,
                        help="HGT epoch 数")
    parser.add_argument("--simplehgn-epochs", type=int, default=None,
                        help="SimpleHGN epoch 数")
    parser.add_argument("--pretrain-epochs", type=int, default=None,
                        help="预训练 epoch 数")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if args.skip_build:
        logger.info("跳过图构建（使用缓存）")
    else:
        logger.info(">>> 构建异质图")
        import time
        t0 = time.time()
        from phase4_v10_minibatch import (
            load_cpi_data, load_ppi_network, load_kegg_pathways,
            load_protein_features, build_graphs_and_adj, _log_step_time,
        )
        cpi_df = load_cpi_data()
        ppi_df = load_ppi_network()
        gene_to_pathways = load_kegg_pathways()
        prot_feat, gene_to_seq = load_protein_features()
        build_graphs_and_adj(cpi_df, ppi_df, gene_to_pathways, prot_feat)
        _log_step_time(t0, "图构建完成")

    skip_sage = args.model is not None and args.model != "sage" and args.model != "all"
    skip_hgt = args.model is not None and args.model != "hgt" and args.model != "all"
    skip_simplehgn = args.model is not None and args.model != "simplehgn" and args.model != "all"

    global_overrides = {}
    if args.sage_epochs is not None:
        global_overrides["EPOCHS"] = args.sage_epochs
    if args.hgt_epochs is not None:
        global_overrides["EPOCHS_HGT"] = args.hgt_epochs
    if args.simplehgn_epochs is not None:
        global_overrides["EPOCHS_SIMPLEHGN"] = args.simplehgn_epochs
    if args.pretrain_epochs is not None:
        global_overrides["PRETRAIN_EPOCHS"] = args.pretrain_epochs
    if args.seed is not None:
        global_overrides["RANDOM_SEED"] = args.seed

    _phase4_main(
        decoder_type=args.decoder,
        skip_sage=skip_sage,
        skip_hgt=skip_hgt,
        skip_simplehgn=skip_simplehgn,
        global_overrides=global_overrides if global_overrides else None,
    )


if __name__ == "__main__":
    main()