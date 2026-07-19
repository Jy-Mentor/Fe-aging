"""入口脚本: 模型训练

调用 phase4_v10_modular.main() 进行 SAGE/HGT/SimpleHGN 多分支训练。

用法:
    python entry/train.py                                    # 完整训练（三分支）
    python entry/train.py --model sage                       # 仅 SAGE
    python entry/train.py --model hgt                        # 仅 HGT
    python entry/train.py --model simplehgn                  # 仅 SimpleHGN
    python entry/train.py --decoder residue_bilinear         # 残基感知解码器
    python entry/train.py --sage-epochs 5 --hgt-epochs 3     # 快速测试
    python entry/train.py --seed 123                         # 指定随机种子
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from phase4_v10_modular import main as _phase4_main

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="铁衰老 GNN 模型训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python entry/train.py                                    # 完整训练（三分支）
  python entry/train.py --model sage                       # 仅 SAGE
  python entry/train.py --model hgt                        # 仅 HGT
  python entry/train.py --model simplehgn                  # 仅 SimpleHGN
  python entry/train.py --decoder residue_bilinear         # 残基感知解码器
  python entry/train.py --sage-epochs 5 --hgt-epochs 3     # 快速测试
  python entry/train.py --seed 123                         # 指定随机种子
        """,
    )
    parser.add_argument("--model", type=str, default=None,
                        choices=["sage", "hgt", "simplehgn", "all"],
                        help="仅训练指定模型（默认: all）")
    parser.add_argument("--decoder", type=str, default=None,
                        choices=["mlp", "dot", "bilinear", "residue_bilinear"],
                        help="解码器类型")
    parser.add_argument("--sage-epochs", type=int, default=None,
                        help="SAGE 训练 epoch 数")
    parser.add_argument("--hgt-epochs", type=int, default=None,
                        help="HGT 训练 epoch 数")
    parser.add_argument("--simplehgn-epochs", type=int, default=None,
                        help="SimpleHGN 训练 epoch 数")
    parser.add_argument("--pretrain-epochs", type=int, default=None,
                        help="预训练 epoch 数")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

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