import logging
logger = logging.getLogger(__name__)

"""设备选择工具
============
自动检测可用 GPU，支持手动指定设备。
"""


import torch


def get_device(device_str: str | None = None) -> torch.device:
    """获取 torch 设备。

    优先使用用户指定的设备字符串；否则自动检测 CUDA 可用性。

    Args:
        device_str: 设备字符串（如 "cuda:0", "cpu", "mps"）。
                    为 None 时自动选择。

    Returns:
        torch.device: 设备对象。
    """
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
