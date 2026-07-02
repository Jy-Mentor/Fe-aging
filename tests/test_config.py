"""测试配置系统。"""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from L4.src.iron_aging_gnn.utils.config import Config, load_config


def test_default_config_loads():
    """默认配置应能正常实例化并包含关键字段。"""
    cfg = Config()
    assert cfg.random_seed == 42
    assert cfg.model.hidden_dim == 64
    assert cfg.sage.lr == 5e-4
    assert cfg.hgt.lr == 1e-3
    assert cfg.loss.focal_alpha == 0.75


def test_load_config_from_yaml(tmp_path):
    """从 YAML 加载配置时应正确合并默认值。"""
    cfg_path = tmp_path / "custom.yaml"
    custom = {
        "random_seed": 2024,
        "model": {"hidden_dim": 128},
        "sage": {"lr": 1e-4},
    }
    cfg_path.write_text(yaml.safe_dump(custom), encoding="utf-8")

    cfg = load_config(str(cfg_path))
    assert cfg.random_seed == 2024
    assert cfg.model.hidden_dim == 128
    assert cfg.model.num_layers == 2  # 默认值保留
    assert cfg.sage.lr == 1e-4
    assert cfg.hgt.lr == 1e-3  # 未被覆盖


def test_load_config_missing_file_raises():
    """加载不存在的配置文件应抛出 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        load_config("/path/that/does/not/exist.yaml")


def test_resolved_paths_are_absolute():
    """路径配置解析后应为绝对路径。"""
    cfg = Config()
    resolved = cfg.get_resolved_paths()
    assert resolved.project_root.is_absolute()
    assert resolved.l4_results.is_absolute()


def test_validation_config_bounds():
    """配置验证应拒绝越界值。"""
    with pytest.raises(ValidationError):
        Config(model={"hidden_dim": 1024})  # 超过上限 512
