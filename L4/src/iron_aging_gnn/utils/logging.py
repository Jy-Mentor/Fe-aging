"""日志配置工具
============
创建同时输出到文件和控制台的 logger 实例。
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str,
    log_file: Path,
    level: int = logging.INFO,
) -> logging.Logger:
    """创建并配置 logger 实例。

    日志同时输出到：
    - 指定文件（UTF-8 编码，每次运行覆盖）
    - 标准输出（控制台）

    Args:
        name: logger 名称。
        log_file: 日志文件路径。
        level: 日志级别，默认 INFO。

    Returns:
        配置好的 logging.Logger 实例。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # 文件 handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台 handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
