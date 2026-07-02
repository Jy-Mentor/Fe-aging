#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v24 smoke test 实时监控脚本"""
import os
import re
import sys
import time
import subprocess
from datetime import datetime, timedelta

LOG_PATH = r"d:\铁衰老 绝不重蹈覆辙\L4\logs\v24_smoke_test.log"
TARGET_PID = 41084  # 训练进程PID，若失效则自动检测
POLL_INTERVAL = 30  # 秒
PROGRESS_INTERVAL = 120  # 秒
STUCK_THRESHOLD = 180  # 秒


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{now_str()}] [MONITOR] {msg}", flush=True)


def find_training_process():
    """通过命令行匹配 phase4_v10_minibatch.py 返回 PID"""
    try:
        result = subprocess.run(
            [
                "powershell",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "Where-Object { $_.CommandLine -like '*phase4_v10_minibatch*' } | "
                "Select-Object -ExpandProperty ProcessId",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        pids = [int(x.strip()) for x in result.stdout.strip().splitlines() if x.strip().isdigit()]
        if pids:
            return pids[0]
    except Exception as e:
        log(f"自动检测PID失败: {e}")
    return None


def process_alive(pid):
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in result.stdout and "python" in result.stdout.lower()
    except Exception as e:
        log(f"检测进程存活失败: {e}")
        return False


def stop_process(pid):
    log(f"正在停止训练进程 PID={pid}")
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"], check=False)
    except Exception as e:
        log(f"taskkill 失败: {e}")


def read_log_tail(path, chars=2000):
    if not os.path.exists(path):
        return "", None
    try:
        size = os.path.getsize(path)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            if size > chars:
                f.seek(size - chars)
                # 跳到下一行开头，避免读到半截行
                f.readline()
            return f.read(), os.path.getmtime(path)
    except Exception as e:
        log(f"读取日志失败: {e}")
        return "", None


def extract_floats(pattern, text):
    return [float(m) for m in re.findall(pattern, text)]


def check_anomalies(tail_text, idle_sec, loss_history, auc_history, aupr_history):
    anomalies = []

    # 1. 崩溃/OOM/CUDA 错误
    if "RuntimeError" in tail_text:
        anomalies.append(("RuntimeError", "日志中出现 RuntimeError"))
    if re.search(r"CUDA out of memory", tail_text, re.IGNORECASE):
        anomalies.append(("OOM", "CUDA out of memory"))
    if re.search(r"CUDA error", tail_text, re.IGNORECASE):
        anomalies.append(("CUDA_ERROR", "CUDA error"))

    # 2. 数值异常
    if re.search(r"(?<![a-zA-Z])nan(?![a-zA-Z])", tail_text, re.IGNORECASE):
        anomalies.append(("NaN", "检测到 NaN"))
    if re.search(r"(?<![a-zA-Z])inf(?![a-zA-Z])", tail_text, re.IGNORECASE):
        anomalies.append(("Inf", "检测到 inf"))

    # 3. loss 异常
    losses = extract_floats(r"loss=([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)", tail_text)
    for lv in losses:
        loss_history.append(lv)
    if losses:
        latest = losses[-1]
        if latest > 10:
            anomalies.append(("LOSS_HIGH", f"loss={latest:.4f} > 10"))
        # 连续增长判定：最近至少5个loss单调递增
        if len(loss_history) >= 5:
            recent = list(loss_history)[-5:]
            if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
                anomalies.append(("LOSS_INCREASING", f"最近5个loss持续上升: {recent}"))

    # 4. AUC 接近随机
    val_aucs = extract_floats(r"val_auc=([0-9]+\.?[0-9]*)", tail_text)
    prot_aucs = extract_floats(r"prot_auc=([0-9]+\.?[0-9]*)", tail_text)
    for v in val_aucs:
        auc_history.append(("val_auc", v))
    for v in prot_aucs:
        auc_history.append(("prot_auc", v))
    for v in val_aucs + prot_aucs:
        if v < 0.52:
            anomalies.append(("AUC_RANDOM", f"AUC={v:.4f} < 0.52"))

    # 5. prot_aupr 持续低于0.05
    auprs = extract_floats(r"prot_aupr=([0-9]+\.?[0-9]*)", tail_text)
    for v in auprs:
        aupr_history.append(v)
    if auprs and auprs[-1] < 0.05:
        # 连续3次低于阈值视为持续
        if len(aupr_history) >= 3 and all(x < 0.05 for x in list(aupr_history)[-3:]):
            anomalies.append(("AUPR_LOW", f"prot_aupr 最近3次均 < 0.05: {list(aupr_history)[-3:]}"))

    # 6. 训练卡死
    if idle_sec > STUCK_THRESHOLD:
        anomalies.append(("STUCK", f"日志已 {idle_sec} 秒未更新 (>180s)"))

    return anomalies


def summarize_metrics(tail_text):
    """从日志末尾提取关键指标用于进度报告"""
    info = {}
    for key in ["loss", "val_auc", "prot_auc", "prot_aupr", "epoch"]:
        pattern = rf"{key}=([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)"
        matches = extract_floats(pattern, tail_text)
        if matches:
            info[key] = matches[-1]
    return info


def main():
    pid = TARGET_PID
    if not process_alive(pid):
        detected = find_training_process()
        if detected:
            pid = detected
            log(f"目标PID {TARGET_PID} 已失效，自动检测到新PID={pid}")
        else:
            log(f"未找到训练进程，将使用目标PID={pid}并等待其出现")

    log(f"开始监控日志: {LOG_PATH}")
    log(f"训练进程PID: {pid}")

    loss_history = []
    auc_history = []
    aupr_history = []
    last_progress = time.time()
    last_tail = ""
    prev_mod = None
    prev_mod_time = time.time()

    while True:
        alive = process_alive(pid)

        tail_text, last_mod = read_log_tail(LOG_PATH, chars=2000)
        if tail_text:
            last_tail = tail_text

        now = time.time()
        # 卡死判定：日志最后修改时间持续未变超过阈值
        if last_mod is not None:
            if prev_mod is None or last_mod != prev_mod:
                prev_mod = last_mod
                prev_mod_time = now
            idle_sec = int(now - prev_mod_time)
        else:
            idle_sec = 0

        anomalies = check_anomalies(tail_text, idle_sec, loss_history, auc_history, aupr_history)
        if anomalies:
            log("=" * 60)
            log("检测到异常，准备生成报告并停止训练")
            for atype, desc in anomalies:
                log(f"  异常类型: {atype} | {desc}")
            log("=" * 60)
            log("相关日志片段（最近2000字符）：")
            print(tail_text[-2000:], flush=True)
            print("\n" + "=" * 60, flush=True)
            if alive:
                stop_process(pid)
            log("监控结束")
            sys.exit(1)

        if not alive:
            log(f"训练进程 PID={pid} 已结束，停止监控")
            break

        if now - last_progress >= PROGRESS_INTERVAL:
            metrics = summarize_metrics(last_tail)
            log(f"进度正常 | 关键指标: {metrics}")
            last_progress = now

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
