"""Pipeline 模块 — 训练/验证/预测编排功能

本模块从主脚本 phase4_v10_minibatch.py 提取，通过依赖注入（函数参数）
接收全局配置常量，避免循环导入和模块级全局状态。
"""
from .validation import validate_sage, validate_hgt, validate_hgt_minibatch, validate_simplehgn
from .prediction import predict_hgt_scores, predict_hgt_target_proteins_minibatch, predict_simplehgn_scores, predict_tcm
from .utils import check_gpu_memory, log_gpu_memory, log_step_time, check_gradient_norm, get_prot_feat_dim

__all__ = [
    "validate_sage", "validate_hgt", "validate_hgt_minibatch", "validate_simplehgn",
    "predict_hgt_scores", "predict_hgt_target_proteins_minibatch", "predict_simplehgn_scores", "predict_tcm",
    "check_gpu_memory", "log_gpu_memory", "log_step_time", "check_gradient_norm", "get_prot_feat_dim",
]