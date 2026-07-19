"""入口脚本: 模型评估

加载已训练模型并重新计算验证指标。

用法:
    python entry/evaluate.py --checkpoint results_v10_minibatch/sage_best.pt
    python entry/evaluate.py --checkpoint results_v10_minibatch/sage_best.pt --full
    python entry/evaluate.py --reevaluate
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
    parser = argparse.ArgumentParser(description="铁衰老 GNN 模型评估")
    parser.add_argument("--reevaluate", action="store_true", default=False,
                        help="重新评估模式（跳过训练）")
    parser.add_argument("--model", type=str, default="all",
                        choices=["sage", "hgt", "simplehgn", "all"],
                        help="评估指定模型")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    _phase4_main(
        skip_sage=(args.model != "sage" and args.model != "all"),
        skip_hgt=(args.model != "hgt" and args.model != "all"),
        skip_simplehgn=(args.model != "simplehgn" and args.model != "all"),
        reevaluate=args.reevaluate,
    )


if __name__ == "__main__":
    main()