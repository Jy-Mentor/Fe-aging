# 项目全局规则系统 — 快速指针

> **规则权威来源**：`project_memory.md`（Trae 记忆系统）
> 本文件仅作为快速参考指针，完整规则请查阅记忆系统。

---

## 规则索引

| 规则 | 内容 | 位置 |
|------|------|------|
| 规则0 | 会话启动自检 | project_memory.md |
| 规则1 | 任务终结前强制审查（AskUserQuestion） | project_memory.md |
| 规则2 | GitHub 提交前审查 | project_memory.md |
| 规则3 | MCP 工具链与效能优化 | project_memory.md |
| 规则4 | 代码库健康标准 | project_memory.md |

## 快速验证命令

```bash
ruff check L4/scripts/
python L4/scripts/smoke_test.py
python L4/scripts/validate_model_inputs.py
```

## MCP 工具速查

| 工具 | 用途 |
|------|------|
| mcp_Excel | Excel 读写 |
| mcp_GitHub | GitHub 全操作 |
| mcp_biotools | 序列分析（10工具） |

## 反造假铁律

1. 数据来源真实 — 不生成/模拟数据
2. 路径真实 — 引用文件/API/论文均真实存在
3. 异常传播 — 禁止 `try-except: pass`
4. 缺失告警 — 缺失数据写日志
5. 预处理完整 — QC/标准化/校正不跳过
6. 结果真实 — 不报 fake success