
"""utils 模块和 data.constants 模块单元测试

测试覆盖:
  - utils/config.py: Config, PathConfig, ModelConfig, SageConfig, HgtConfig, LossConfig, load_config
  - utils/device.py: get_device
  - utils/seed.py: set_seed 可复现性
  - utils/logging.py: 模块可导入
  - data/constants.py: 常量类型和内容校验
"""

import os
import random
import tempfile
from pathlib import Path

import pytest
import torch
import yaml


# ============================================================================
# 1. config.py 测试
# ============================================================================

class TestConfigDefaults:
    """验证默认 Config 可实例化且字段默认值正确。"""

    def test_config_instantiate_default(self):
        """默认 Config 可以正常实例化。"""
        from iron_aging_gnn.utils.config import Config
        cfg = Config()
        assert cfg is not None
        assert isinstance(cfg.random_seed, int)
        assert cfg.random_seed == 42

    def test_path_config_defaults(self):
        """PathConfig 默认值正确且 resolve 可解析相对路径。"""
        from iron_aging_gnn.utils.config import PathConfig
        pc = PathConfig()
        resolved = pc.resolve()
        assert resolved.project_root == pc.project_root
        assert resolved.l4_root.is_absolute()
        assert resolved.l4_results.is_absolute()
        assert resolved.l4_logs.is_absolute()
        assert "L4" in str(resolved.l4_root)

    def test_model_config_defaults(self):
        """ModelConfig 各字段默认值正确。"""
        from iron_aging_gnn.utils.config import ModelConfig
        mc = ModelConfig()
        assert mc.hidden_dim == 64
        assert mc.out_dim == 64
        assert mc.num_layers == 2
        assert mc.dropout == 0.5
        assert mc.num_heads == 2
        assert mc.decoder_type == "mlp"
        assert mc.temperature == 1.0
        assert mc.prot_proj_dropout == 0.4
        assert mc.prot_proj_inner_dropout == 0.3
        assert mc.pathway_proj_dropout == 0.3
        assert mc.pheno_head_dropout == 0.3
        assert mc.score_clamp == 10.0

    def test_sage_config_defaults(self):
        """SageConfig 各字段默认值正确。"""
        from iron_aging_gnn.utils.config import SageConfig
        sc = SageConfig()
        assert sc.epochs == 15
        assert sc.lr == 5e-4
        assert sc.patience == 5
        assert sc.batch_size == 256
        assert sc.num_neighbors == [32, 16]
        assert sc.two_stage is True
        assert sc.pretrain_epochs == 10
        assert sc.pretrain_lr == 7.5e-4
        assert sc.finetune_lr_multiplier == 0.5
        assert sc.use_plateau_scheduler is True
        assert sc.plateau_patience == 2
        assert sc.plateau_factor == 0.5

    def test_hgt_config_defaults(self):
        """HgtConfig 各字段默认值正确。"""
        from iron_aging_gnn.utils.config import HgtConfig
        hc = HgtConfig()
        assert hc.epochs == 15
        assert hc.lr == 1e-3
        assert hc.patience == 5
        assert hc.batch_size == 128
        assert hc.num_neighbors == [32, 16]
        assert hc.two_stage is True
        assert hc.pretrain_epochs == 10
        assert hc.pretrain_lr == 1.5e-3
        assert hc.finetune_lr_multiplier == 0.5
        assert hc.use_plateau_scheduler is True
        assert hc.plateau_patience == 2
        assert hc.plateau_factor == 0.5

    def test_loss_config_defaults(self):
        """LossConfig 各字段默认值正确。"""
        from iron_aging_gnn.utils.config import LossConfig
        lc = LossConfig()
        assert lc.focal_alpha == 0.75
        assert lc.focal_gamma == 2.0
        assert lc.label_smoothing == 0.0
        assert lc.label_smoothing_pos == 0.9
        assert lc.label_smoothing_neg == 0.1
        assert lc.bce_weight == 0.6
        assert lc.bpr_weight == 0.4
        assert lc.infonce_weight == 0.1
        assert lc.temperature == 5.0
        assert lc.infonce_temperature == 0.07

    def test_config_ferrogenesis_genes(self):
        """Config 中 ferrogenesis_genes 为字符串列表且非空。"""
        from iron_aging_gnn.utils.config import Config
        cfg = Config()
        assert isinstance(cfg.ferrogenesis_genes, list)
        assert len(cfg.ferrogenesis_genes) > 0
        for gene in cfg.ferrogenesis_genes:
            assert isinstance(gene, str)

    def test_config_get_resolved_paths(self):
        """get_resolved_paths() 返回绝对路径。"""
        from iron_aging_gnn.utils.config import Config
        cfg = Config()
        resolved = cfg.get_resolved_paths()
        assert resolved.project_root.is_absolute()
        assert resolved.l4_results.is_absolute()

    def test_config_get_l4_results_dir(self):
        """get_l4_results_dir() 返回的目录存在。"""
        from iron_aging_gnn.utils.config import Config
        cfg = Config()
        d = cfg.get_l4_results_dir()
        assert d.exists()

    def test_config_get_l4_logs_dir(self):
        """get_l4_logs_dir() 返回的目录存在。"""
        from iron_aging_gnn.utils.config import Config
        cfg = Config()
        d = cfg.get_l4_logs_dir()
        assert d.exists()


class TestPathResolution:
    """验证 PathConfig.resolve() 路径解析逻辑。"""

    def test_resolve_relative_paths(self):
        """相对路径解析为 project_root 下的绝对路径。"""
        from iron_aging_gnn.utils.config import PathConfig
        fake_root = Path("C:/fake_project")
        pc = PathConfig(
            project_root=fake_root,
            l4_root=Path("L4"),
            l4_results=Path("L4/results"),
            l4_logs=Path("L4/logs"),
        )
        resolved = pc.resolve()
        assert resolved.project_root == fake_root
        assert resolved.l4_root == fake_root / "L4"
        assert resolved.l4_results == fake_root / "L4" / "results"
        assert resolved.l4_logs == fake_root / "L4" / "logs"

    def test_resolve_absolute_paths_unchanged(self):
        """绝对路径在 resolve 时不改变。"""
        from iron_aging_gnn.utils.config import PathConfig
        abs_path = Path("C:/absolute/path")
        pc = PathConfig(
            project_root=Path("C:/fake_project"),
            l4_results=abs_path,
        )
        resolved = pc.resolve()
        assert resolved.l4_results == abs_path


class TestLoadConfig:
    """验证 load_config() 从 YAML 加载配置。"""

    def test_load_config_returns_config(self):
        """load_config() 返回 Config 实例。"""
        from iron_aging_gnn.utils.config import Config, load_config
        cfg = load_config(None)
        assert isinstance(cfg, Config)

    def test_load_config_file_not_found(self):
        """不存在的文件抛出 FileNotFoundError。"""
        from iron_aging_gnn.utils.config import load_config
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_path.yaml")

    def test_load_config_from_default_yaml(self):
        """从项目 default.yaml 加载配置成功。"""
        from iron_aging_gnn.utils.config import Config, load_config
        yaml_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
        cfg = load_config(str(yaml_path))
        assert isinstance(cfg, Config)
        assert cfg.model.hidden_dim == 128
        assert cfg.model.out_dim == 128
        assert cfg.model.num_layers == 3
        assert cfg.model.dropout == 0.3
        assert cfg.model.decoder_type == "residue_bilinear"

    def test_load_config_overrides_sage(self):
        """YAML 中 sage 配置覆盖了默认值。"""
        from iron_aging_gnn.utils.config import Config, load_config
        yaml_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
        cfg = load_config(str(yaml_path))
        assert cfg.sage.epochs == 30
        assert cfg.sage.lr == 0.0005
        assert cfg.sage.patience == 10
        assert cfg.sage.num_neighbors == [16, 8]

    def test_load_config_overrides_hgt(self):
        """YAML 中 hgt 配置覆盖了默认值。"""
        from iron_aging_gnn.utils.config import Config, load_config
        yaml_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
        cfg = load_config(str(yaml_path))
        assert cfg.hgt.epochs == 20
        assert cfg.hgt.lr == 0.0005
        assert cfg.hgt.patience == 7
        assert cfg.hgt.num_neighbors == [16, 8]

    def test_load_config_overrides_loss(self):
        """YAML 中 loss 配置覆盖了默认值。"""
        from iron_aging_gnn.utils.config import Config, load_config
        yaml_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
        cfg = load_config(str(yaml_path))
        assert cfg.loss.bce_weight == 1.0
        assert cfg.loss.bpr_weight == 0.3

    def test_load_config_minimal_yaml(self):
        """最小 YAML 文件（仅覆盖一个字段）加载成功，其余使用默认值。"""
        from iron_aging_gnn.utils.config import Config, load_config
        yaml_content = {"random_seed": 123, "model": {"hidden_dim": 256}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump(yaml_content, f)
            tmp_path = f.name
        try:
            cfg = load_config(tmp_path)
            assert isinstance(cfg, Config)
            assert cfg.random_seed == 123
            assert cfg.model.hidden_dim == 256
            assert cfg.model.out_dim == 64
            assert cfg.sage.epochs == 15
        finally:
            os.unlink(tmp_path)

    def test_load_config_empty_yaml(self):
        """空 YAML 文件返回全默认 Config。"""
        from iron_aging_gnn.utils.config import Config, load_config
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            tmp_path = f.name
        try:
            cfg = load_config(tmp_path)
            assert isinstance(cfg, Config)
            assert cfg.random_seed == 42
        finally:
            os.unlink(tmp_path)


class TestConfigSerialization:
    """验证 Config 的基本序列化/反序列化行为。"""

    def test_config_model_dump(self):
        """Config 可以 model_dump 为字典。"""
        from iron_aging_gnn.utils.config import Config
        cfg = Config()
        d = cfg.model_dump()
        assert isinstance(d, dict)
        assert "random_seed" in d
        assert "paths" in d
        assert "model" in d
        assert d["random_seed"] == 42

    def test_config_roundtrip(self):
        """Config -> dict -> 新 Config 往返一致。"""
        from iron_aging_gnn.utils.config import Config
        cfg1 = Config()
        d = cfg1.model_dump()
        cfg2 = Config(**d)
        assert cfg2.random_seed == cfg1.random_seed
        assert cfg2.model.hidden_dim == cfg1.model.hidden_dim


class TestConfigFieldValidation:
    """验证 pydantic 字段约束。"""

    def test_model_config_hidden_dim_bounds(self):
        """hidden_dim 超出范围应触发 ValidationError。"""
        from iron_aging_gnn.utils.config import ModelConfig
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ModelConfig(hidden_dim=0)
        with pytest.raises(ValidationError):
            ModelConfig(hidden_dim=1024)

    def test_model_config_dropout_bounds(self):
        """dropout 超出 [0, 0.9] 应触发 ValidationError。"""
        from iron_aging_gnn.utils.config import ModelConfig
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ModelConfig(dropout=1.5)

    def test_loss_config_focal_alpha_bounds(self):
        """focal_alpha 超出 [0, 1] 应触发 ValidationError。"""
        from iron_aging_gnn.utils.config import LossConfig
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LossConfig(focal_alpha=1.5)


# ============================================================================
# 2. device.py 测试
# ============================================================================

class TestGetDevice:
    """验证 get_device() 返回 torch.device。"""

    def test_get_device_default(self):
        """默认调用返回 torch.device 对象。"""
        from iron_aging_gnn.utils.device import get_device
        dev = get_device()
        assert isinstance(dev, torch.device)

    def test_get_device_explicit_cpu(self):
        """显式指定 'cpu' 返回 cpu 设备。"""
        from iron_aging_gnn.utils.device import get_device
        dev = get_device("cpu")
        assert isinstance(dev, torch.device)
        assert dev.type == "cpu"

    def test_get_device_string(self):
        """传入任意设备字符串应返回对应的 torch.device。"""
        from iron_aging_gnn.utils.device import get_device
        dev = get_device("cpu")
        assert dev == torch.device("cpu")


# ============================================================================
# 3. seed.py 测试
# ============================================================================

class TestSetSeed:
    """验证 set_seed() 产生可复现的随机数。"""

    def test_set_seed_reproducible_python(self):
        """set_seed 后 Python random 可复现。"""
        from iron_aging_gnn.utils.seed import set_seed
        set_seed(42)
        a = random.random()
        set_seed(42)
        b = random.random()
        assert a == b

    def test_set_seed_reproducible_torch(self):
        """set_seed 后 torch 随机数可复现。"""
        from iron_aging_gnn.utils.seed import set_seed
        set_seed(42)
        x1 = torch.randn(5)
        set_seed(42)
        x2 = torch.randn(5)
        assert torch.equal(x1, x2)

    def test_set_seed_different_seeds_different(self):
        """不同种子产生不同随机数。"""
        from iron_aging_gnn.utils.seed import set_seed
        set_seed(42)
        x1 = torch.randn(10)
        set_seed(123)
        x2 = torch.randn(10)
        assert not torch.equal(x1, x2)

    def test_set_seed_deterministic_false(self):
        """deterministic=False 时 set_seed 不报错。"""
        from iron_aging_gnn.utils.seed import set_seed
        set_seed(42, deterministic=False)
        x = torch.randn(5)
        assert x.numel() == 5

    def test_seed_worker(self):
        """seed_worker 函数可调用且不报错。"""
        from iron_aging_gnn.utils.seed import seed_worker
        seed_worker(0, base_seed=42)
        seed_worker(1, base_seed=42)


# ============================================================================
# 4. logging.py 测试
# ============================================================================

class TestLogging:
    """验证 logging 模块可导入且 setup_logger 可正常使用。"""

    def test_module_import(self):
        """模块可正常导入。"""
        import iron_aging_gnn.utils.logging as log_mod
        assert hasattr(log_mod, "setup_logger")

    def test_setup_logger_creates_logger(self):
        """setup_logger 返回 logging.Logger 实例。"""
        import logging
        from iron_aging_gnn.utils.logging import setup_logger
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = setup_logger("test_logger", log_path)
            assert isinstance(logger, logging.Logger)
            assert logger.name == "test_logger"
            for h in logger.handlers:
                h.close()
            for h in logger.handlers:
                h.close()

    def test_setup_logger_writes_to_file(self):
        """setup_logger 创建的 logger 可写入文件。"""
        from iron_aging_gnn.utils.logging import setup_logger
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_write.log"
            logger = setup_logger("test_write", log_path)
            logger.info("hello world")
            for h in logger.handlers:
                h.flush()
                h.close()
            content = log_path.read_text(encoding="utf-8")
            assert "hello world" in content


# ============================================================================
# 5. data/constants.py 测试
# ============================================================================

class TestConstants:
    """验证 data/constants.py 中常量类型和内容。"""

    def test_all_ferroaging_genes_is_list_of_str(self):
        """ALL_FERRORAGING_GENES 是字符串列表。"""
        from iron_aging_gnn.data.constants import ALL_FERRORAGING_GENES
        assert isinstance(ALL_FERRORAGING_GENES, list)
        assert len(ALL_FERRORAGING_GENES) > 0
        for gene in ALL_FERRORAGING_GENES:
            assert isinstance(gene, str)

    def test_all_ferroaging_genes_contains_known(self):
        """ALL_FERRORAGING_GENES 包含已知铁死亡基因。"""
        from iron_aging_gnn.data.constants import ALL_FERRORAGING_GENES
        known_genes = {"ACSL4", "TFRC", "HMOX1", "GPX4", "SLC7A11", "FTH1"}
        found = set(ALL_FERRORAGING_GENES)
        overlap = found & known_genes
        assert len(overlap) >= 3, (
            f"期望至少包含 3 个已知铁死亡基因，但只找到 {overlap}"
        )

    def test_all_ferroaging_genes_sorted(self):
        """ALL_FERRORAGING_GENES 已排序。"""
        from iron_aging_gnn.data.constants import ALL_FERRORAGING_GENES
        assert ALL_FERRORAGING_GENES == sorted(ALL_FERRORAGING_GENES)

    def test_rdkit_descriptor_names_is_list(self):
        """RDKIT_DESCRIPTOR_NAMES 是字符串列表。"""
        from iron_aging_gnn.data.constants import RDKIT_DESCRIPTOR_NAMES
        assert isinstance(RDKIT_DESCRIPTOR_NAMES, list)
        assert len(RDKIT_DESCRIPTOR_NAMES) > 0
        for name in RDKIT_DESCRIPTOR_NAMES:
            assert isinstance(name, str)

    def test_rdkit_descriptor_names_content(self):
        """RDKIT_DESCRIPTOR_NAMES 包含常见描述符。"""
        from iron_aging_gnn.data.constants import RDKIT_DESCRIPTOR_NAMES
        expected = {"MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors"}
        found = set(RDKIT_DESCRIPTOR_NAMES)
        assert expected.issubset(found), (
            f"RDKIT_DESCRIPTOR_NAMES 缺少: {expected - found}"
        )

    def test_ecfp4_nbits_is_int(self):
        """ECFP4_NBITS 是 int 且为正数。"""
        from iron_aging_gnn.data.constants import ECFP4_NBITS
        assert isinstance(ECFP4_NBITS, int)
        assert ECFP4_NBITS == 2048

    def test_random_seed_is_int(self):
        """RANDOM_SEED 是 int 且为 42。"""
        from iron_aging_gnn.data.constants import RANDOM_SEED
        assert isinstance(RANDOM_SEED, int)
        assert RANDOM_SEED == 42

    def test_constants_consistency_with_config(self):
        """constants.py 中的常量与 config.py 默认值一致。"""
        from iron_aging_gnn.data.constants import (
            ALL_FERRORAGING_GENES, ECFP4_NBITS, RANDOM_SEED, RDKIT_DESCRIPTOR_NAMES,
        )
        from iron_aging_gnn.utils.config import Config
        cfg = Config()
        assert RANDOM_SEED == cfg.random_seed
        assert ECFP4_NBITS == cfg.compound_feature.ecfp4_nbits
        assert set(ALL_FERRORAGING_GENES) == set(cfg.ferrogenesis_genes)
        assert set(RDKIT_DESCRIPTOR_NAMES) == set(cfg.compound_feature.rdkit_descriptor_names)
