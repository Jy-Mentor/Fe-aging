"""入口脚本: 构建异质图数据

调用 phase4_v10_modular.main(build_graph_only=True) 从原始数据构建图数据并缓存。

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

from phase4_v10_modular import main as _phase4_main

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

    logger.info("=" * 60)
    logger.info("构建异质图数据")
    logger.info("=" * 60)

    if args.force_rebuild:
        import os
        cache_path = SCRIPTS_DIR.parent / "results_v10_minibatch" / "graph_cache_v70.pkl"
        if cache_path.exists():
            try:
                os.remove(cache_path)
                logger.info(f"已删除旧图缓存: {cache_path}")
            except Exception as _e:
                logger.warning(f"删除旧图缓存失败: {_e}")

    _phase4_main(build_graph_only=True)

    logger.info("图构建完成。")


if __name__ == "__main__":
    main()
