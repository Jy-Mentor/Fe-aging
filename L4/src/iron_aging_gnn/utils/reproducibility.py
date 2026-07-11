"""可复现性管理工具
================
生成实验复现清单（Reproducibility Manifest），记录代码版本、依赖版本、
数据文件校验和、配置快照及随机种子，以满足学术规范对可复现性的要求。
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    """计算文件的 SHA-256 校验和。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info(project_root: Path) -> dict[str, Any]:
    """获取当前 Git 仓库信息；若不在 git 仓库中则返回空字典。"""
    info: dict[str, Any] = {}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            != ""
        )
        info = {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception as e:
        logger.warning(f"无法获取 Git 信息: {e}")
    return info


def _dependency_versions(packages: list[str]) -> dict[str, str | None]:
    """获取指定 Python 包的版本号。"""
    versions: dict[str, str | None] = {}
    for pkg in packages:
        try:
            mod = __import__(pkg.split(".")[0])
            versions[pkg] = getattr(mod, "__version__", None)
        except Exception as e:
            logger.warning(f"无法获取 {pkg} 版本: {e}")
            versions[pkg] = None
    return versions


def _load_yaml_safe(path: Path) -> Any | None:
    """安全加载 YAML 文件；失败时返回 None。"""
    try:
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"无法读取配置文件 {path}: {e}")
        return None


def generate_reproducibility_manifest(
    project_root: str | Path,
    config_path: str | Path | None = None,
    data_files: list[str | Path] | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """生成实验复现清单。

    清单内容包括：
      - 时间戳与运行环境（Python / OS / CPU 架构）
      - Git 提交哈希、分支、工作区是否 dirty
      - 关键依赖版本（torch, torch-geometric, numpy, pandas, rdkit, pydantic 等）
      - 配置文件内容快照与校验和
      - 关键数据文件校验和与行数
      - 随机种子
      - 用户传入的额外字段

    Args:
        project_root: 项目根目录。
        config_path: 配置文件路径（相对或绝对）。
        data_files: 需要记录校验和的数据文件路径列表（相对或绝对）。
        seed: 实验随机种子。
        extra: 用户自定义额外字段。

    Returns:
        可序列化为 JSON 的复现清单字典。
    """
    project_root = Path(project_root).resolve()
    manifest: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "git": _git_info(project_root),
        "dependencies": _dependency_versions(
            [
                "torch",
                "torch_geometric",
                "numpy",
                "pandas",
                "sklearn",
                "scipy",
                "rdkit",
                "pydantic",
                "yaml",
                "transformers",
            ]
        ),
    }

    if config_path is not None:
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = project_root / cfg_path
        if cfg_path.exists():
            config_content = _load_yaml_safe(cfg_path)
            manifest["config"] = {
                "path": str(cfg_path.relative_to(project_root)),
                "sha256": _sha256_file(cfg_path),
                "content": config_content if config_content is not None else {},
            }
        else:
            manifest["config"] = {"path": str(cfg_path), "error": "file not found"}

    if data_files:
        manifest["data_files"] = []
        for f in data_files:
            fpath = Path(f)
            if not fpath.is_absolute():
                fpath = project_root / fpath
            entry: dict[str, Any] = {"path": str(fpath.relative_to(project_root))}
            if fpath.exists():
                entry["sha256"] = _sha256_file(fpath)
                entry["size_bytes"] = fpath.stat().st_size
                entry["mtime"] = datetime.fromtimestamp(
                    fpath.stat().st_mtime, tz=timezone.utc
                ).isoformat()
            else:
                entry["error"] = "file not found"
            manifest["data_files"].append(entry)

    if seed is not None:
        manifest["seed"] = seed

    if extra is not None:
        manifest["extra"] = extra

    return manifest


def export_environment_fingerprint(output_dir: str | Path) -> Path:
    """导出环境指纹（pip freeze + 系统信息）到指定目录。

    生成文件:
      - {output_dir}/environment_fingerprint.txt: pip freeze 完整输出
      - {output_dir}/environment_fingerprint.json: 结构化系统信息

    Args:
        output_dir: 输出目录路径。

    Returns:
        生成的 fingerprint 文件路径。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # pip freeze 文本输出
    pip_path = output_dir / "environment_fingerprint.txt"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True,
        )
        pip_content = result.stdout
    except Exception as e:
        logger.warning(f"pip freeze 失败: {e}")
        pip_content = f"# pip freeze failed: {e}\n"
    pip_path.write_text(pip_content, encoding="utf-8")
    logger.info(f"环境指纹 (pip freeze) 已保存: {pip_path}")

    # 结构化系统信息
    json_path = output_dir / "environment_fingerprint.json"
    sys_info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cuda_available": False,
    }
    try:
        import torch
        sys_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            sys_info["cuda_version"] = torch.version.cuda
            sys_info["gpu_count"] = torch.cuda.device_count()
            sys_info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    json_path.write_text(json.dumps(sys_info, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"系统信息指纹已保存: {json_path}")

    return pip_path


def save_reproducibility_manifest(
    manifest: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """将复现清单保存为 JSON 文件。

    Args:
        manifest: generate_reproducibility_manifest 生成的字典。
        output_path: 输出 JSON 文件路径。

    Returns:
        保存后的文件路径。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info(f"复现清单已保存: {output_path}")
    return output_path
