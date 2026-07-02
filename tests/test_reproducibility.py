"""测试可复现性清单生成。"""

from __future__ import annotations

from L4.src.iron_aging_gnn.utils.reproducibility import (
    generate_reproducibility_manifest,
    save_reproducibility_manifest,
)


def test_manifest_contains_required_keys(tmp_path):
    """复现清单应包含时间戳、Python 环境、依赖等关键字段。"""
    manifest = generate_reproducibility_manifest(
        project_root=tmp_path,
        config_path=None,
        data_files=None,
        seed=42,
    )
    assert "timestamp" in manifest
    assert "python" in manifest
    assert "git" in manifest
    assert "dependencies" in manifest
    assert manifest["seed"] == 42


def test_manifest_records_config_and_data(tmp_path):
    """复现清单应正确记录配置文件和数据文件校验和。"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("random_seed: 42\n", encoding="utf-8")

    data_path = tmp_path / "data.csv"
    data_path.write_text("id,value\n1,10\n", encoding="utf-8")

    manifest = generate_reproducibility_manifest(
        project_root=tmp_path,
        config_path=str(cfg_path),
        data_files=[str(data_path)],
        seed=42,
    )

    assert manifest["config"]["path"] == "config.yaml"
    assert "sha256" in manifest["config"]
    assert len(manifest["data_files"]) == 1
    assert manifest["data_files"][0]["path"] == "data.csv"
    assert "sha256" in manifest["data_files"][0]


def test_save_manifest(tmp_path):
    """复现清单应能保存为 JSON 文件。"""
    manifest = generate_reproducibility_manifest(project_root=tmp_path, seed=42)
    output = tmp_path / "manifest.json"
    saved = save_reproducibility_manifest(manifest, output)
    assert saved == output
    assert saved.exists()
    assert saved.stat().st_size > 0
