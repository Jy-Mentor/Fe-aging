"""MLflow 实验追踪与模型版本管理设置

提供统一的 MLflow 集成接口，用于：
- 实验追踪（超参数、指标、日志）
- 模型版本管理（注册、阶段转换）
- Artifact 管理（检查点、配置文件、评估结果）
- 与 Optuna 超参数搜索集成

用法:
    python mlflow_setup.py --setup                  # 初始化 MLflow 追踪服务器
    python mlflow_setup.py --register-model <run_id> # 注册模型到 MLflow Registry
    python mlflow_setup.py --list-experiments       # 列出所有实验
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
MLFLOW_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_ARTIFACT_DIR = PROJECT_ROOT / "mlartifacts"


def setup_mlflow_tracking(
    tracking_uri: str | None = None,
    experiment_name: str = "iron_aging_gnn",
    artifact_location: str | None = None,
) -> str:
    """初始化 MLflow 追踪环境。

    Args:
        tracking_uri: MLflow 追踪 URI。为 None 时使用默认本地路径。
        experiment_name: 实验名称。
        artifact_location: Artifact 存储位置。为 None 时使用默认路径。

    Returns:
        MLflow 追踪 URI。
    """
    import mlflow

    if tracking_uri is None:
        tracking_uri = f"file:///{MLFLOW_DIR.resolve().as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow 追踪 URI: {tracking_uri}")

    if artifact_location is None:
        artifact_location = MLFLOW_ARTIFACT_DIR.resolve().as_posix()

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
        )
        logger.info(f"创建实验 '{experiment_name}' (id={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"实验 '{experiment_name}' 已存在 (id={experiment_id})")

    mlflow.set_experiment(experiment_name)
    return tracking_uri


def log_model_checkpoint(
    model: Any,
    checkpoint_path: Path,
    run_id: str | None = None,
    model_name: str = "iron_aging_gnn_model",
    metrics: dict[str, float] | None = None,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """将模型检查点记录到 MLflow。

    Args:
        model: PyTorch 模型实例。
        checkpoint_path: 模型检查点文件路径。
        run_id: MLflow 运行 ID。为 None 时创建新运行。
        model_name: MLflow 模型名称。
        metrics: 评估指标字典。
        params: 超参数字典。
        tags: 标签字典。

    Returns:
        MLflow 运行 ID。
    """
    import mlflow
    import torch

    if run_id is None:
        run = mlflow.start_run(run_name=f"model_registry_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        run_id = run.info.run_id
    else:
        mlflow.start_run(run_id=run_id)

    if params:
        mlflow.log_params(params)
    if metrics:
        mlflow.log_metrics(metrics)
    if tags:
        mlflow.set_tags(tags)

    mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")

    # 使用 PyTorch 风格保存模型
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model_state")

    mlflow.set_tag("model_type", type(model).__name__)
    mlflow.set_tag("checkpoint_path", str(checkpoint_path))

    mlflow.end_run()
    logger.info(f"模型已记录到 MLflow: run_id={run_id}, model_name={model_name}")
    return run_id


def register_model(
    run_id: str,
    model_name: str = "iron_aging_gnn_model",
    stage: str = "Staging",
    artifact_path: str = "model_state",
) -> str:
    """将 MLflow 运行中的模型注册到 Model Registry。

    Args:
        run_id: MLflow 运行 ID。
        model_name: 注册的模型名称。
        stage: 模型阶段: "None" | "Staging" | "Production" | "Archived"。
        artifact_path: 模型 artifact 路径。

    Returns:
        模型版本号。
    """
    import mlflow

    model_uri = f"runs:/{run_id}/{artifact_path}"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = result.version

    if stage and stage != "None":
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
        )
        logger.info(f"模型 '{model_name}' v{version} 已注册并转换到 {stage} 阶段")
    else:
        logger.info(f"模型 '{model_name}' v{version} 已注册")

    return version


def list_experiments() -> list[dict]:
    """列出所有 MLflow 实验。"""
    import mlflow

    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    result = []
    for exp in experiments:
        result.append({
            "experiment_id": exp.experiment_id,
            "name": exp.name,
            "artifact_location": exp.artifact_location,
            "lifecycle_stage": exp.lifecycle_stage,
        })
    return result


def list_registered_models() -> list[dict]:
    """列出所有已注册的 MLflow 模型。"""
    import mlflow

    client = mlflow.tracking.MlflowClient()
    models = client.search_registered_models()
    result = []
    for m in models:
        latest_versions = [v for v in m.latest_versions]
        result.append({
            "name": m.name,
            "latest_versions": [
                {"version": v.version, "stage": v.current_stage}
                for v in latest_versions
            ],
        })
    return result


def get_best_run(
    experiment_name: str = "iron_aging_gnn",
    metric: str = "best_aupr",
    mode: str = "max",
) -> dict | None:
    """从实验中获取最佳运行。

    Args:
        experiment_name: 实验名称。
        metric: 排序指标。
        mode: "max" 或 "min"。

    Returns:
        最佳运行的参数字典，或 None。
    """
    import mlflow

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning(f"实验 '{experiment_name}' 不存在")
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {'DESC' if mode == 'max' else 'ASC'}"],
        max_results=1,
    )

    if not runs:
        logger.warning(f"实验 '{experiment_name}' 中无运行记录")
        return None

    best_run = runs[0]
    return {
        "run_id": best_run.info.run_id,
        "run_name": best_run.info.run_name,
        "params": best_run.data.params,
        "metrics": best_run.data.metrics,
        "tags": best_run.data.tags,
    }


def main():
    parser = argparse.ArgumentParser(
        description="MLflow 实验追踪与模型版本管理",
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="初始化 MLflow 追踪环境",
    )
    parser.add_argument(
        "--register-model", type=str, default=None,
        help="注册模型到 MLflow Registry（需提供 run_id）",
    )
    parser.add_argument(
        "--list-experiments", action="store_true",
        help="列出所有实验",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="列出所有已注册模型",
    )
    parser.add_argument(
        "--get-best", action="store_true",
        help="获取最佳运行",
    )
    parser.add_argument(
        "--experiment-name", type=str, default="iron_aging_gnn",
        help="实验名称",
    )
    parser.add_argument(
        "--model-name", type=str, default="iron_aging_gnn_model",
        help="模型名称",
    )
    parser.add_argument(
        "--stage", type=str, default="Staging",
        choices=["None", "Staging", "Production", "Archived"],
        help="模型阶段",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        import mlflow
    except ImportError:
        logger.error("MLflow 未安装。pip install mlflow")
        sys.exit(1)

    if args.setup:
        uri = setup_mlflow_tracking(experiment_name=args.experiment_name)
        print(f"MLflow 追踪已初始化: {uri}")

    if args.list_experiments:
        experiments = list_experiments()
        for exp in experiments:
            print(f"  [{exp['experiment_id']}] {exp['name']} ({exp['lifecycle_stage']})")

    if args.list_models:
        models = list_registered_models()
        for m in models:
            versions_str = ", ".join(
                f"v{v['version']}({v['stage']})" for v in m["latest_versions"]
            )
            print(f"  {m['name']}: {versions_str}")

    if args.get_best:
        best = get_best_run(experiment_name=args.experiment_name)
        if best:
            print(f"最佳运行: {best['run_name']} ({best['run_id']})")
            print(f"参数: {best['params']}")
            print(f"指标: {best['metrics']}")
        else:
            print("未找到最佳运行")

    if args.register_model:
        version = register_model(
            run_id=args.register_model,
            model_name=args.model_name,
            stage=args.stage,
        )
        print(f"模型已注册: {args.model_name} v{version} ({args.stage})")


if __name__ == "__main__":
    main()