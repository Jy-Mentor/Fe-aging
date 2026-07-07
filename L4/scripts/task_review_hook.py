import logging
logger = logging.getLogger(__name__)

"""
task_review_hook.py — 任务完成审查钩子模块

本模块为AI助手提供任务完成后的审查触发机制。
各阶段脚本可在主流程结束时调用此模块，作为双重保险。

用法示例（在脚本末尾）：
    from task_review_hook import trigger_task_review
    trigger_task_review(task_name="phase4_model_training", outputs=["results/model.csv"])

注意：本模块仅为辅助提醒，真正的审查流程由AI助手通过 AskUserQuestion 工具执行。
"""

import sys
from pathlib import Path
from datetime import datetime


def trigger_task_review(
    task_name: str = "unknown_task",
    outputs: list[str] | None = None,
    raise_on_missing: bool = True,
) -> dict:
    """
    触发任务完成审查钩子。

    参数:
        task_name: 任务标识名称
        outputs: 预期输出文件路径列表
        raise_on_missing: 若输出文件缺失是否报错

    返回:
        审查状态字典
    """
    review_status = {
        "task_name": task_name,
        "timestamp": datetime.now().isoformat(),
        "outputs_checked": [],
        "missing_outputs": [],
        "review_triggered": True,
    }

    # 检查输出文件是否存在
    if outputs:
        for out_path in outputs:
            p = Path(out_path)
            if p.exists():
                review_status["outputs_checked"].append(str(p.resolve()))
            else:
                review_status["missing_outputs"].append(str(p))

    # 若要求严格检查且文件缺失，抛出异常（不静默吞掉）
    if raise_on_missing and review_status["missing_outputs"]:
        missing = ", ".join(review_status["missing_outputs"])
        raise FileNotFoundError(
            f"[{task_name}] 审查钩子发现缺失输出: {missing}. "
            "任务可能未完成，禁止跳过审查。"
        )

    # 打印醒目的审查提醒（供AI助手识别）
    banner = "=" * 60
    print(f"\n{banner}")
    print("[TASK_REVIEW_HOOK] 任务执行完毕，等待AI审查...")
    print(f"任务: {task_name}")
    print(f"输出文件: {review_status['outputs_checked']}")
    print("AI助手必须调用 AskUserQuestion 发送结构化审查提示。")
    print(f"{banner}\n")

    return review_status


def check_global_rules_exist() -> bool:
    """
    检查项目根目录的 GLOBAL_RULES.md 是否存在。
    若不存在，打印警告（不静默跳过）。
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    rules_file = project_root / "GLOBAL_RULES.md"

    if rules_file.exists():
        return True

    print(
        f"\n[WARNING] GLOBAL_RULES.md 未找到于 {project_root}. "
        "全局规则系统可能未激活。\n"
    )
    return False


if __name__ == "__main__":
    # 自检：确认规则文件存在
    if not check_global_rules_exist():
        sys.exit(1)

    print("task_review_hook.py 自检通过。全局规则系统就绪。")
