# 铁衰老项目依赖安装报告

## 1. 目标环境

- **主环境**: `d:\铁衰老 绝不重蹈覆辙\.venv`
- **Python 版本**: 3.12.3
- **pip 版本**: 24.0
- **GPU**: NVIDIA GeForce RTX 5060 Laptop GPU
- **驱动 CUDA 版本**: 12.8（`nvidia-smi` 确认）

> 注意：系统 Python 3.10.0a5 的 pip 因 `typing.TypeGuard` 缺失已损坏，所有操作均通过 `.venv` 完成。

## 2. 本次安装的依赖

### 2.1 核心建模依赖（首次安装）

| 包名 | 版本 | 用途 |
|------|------|------|
| transformers | 5.14.1 | ESM-2 蛋白特征提取 |
| biopython | 1.87 | 蛋白序列处理 |
| optuna | 4.9.0 | 超参数搜索 |
| mlflow | 3.14.0 | 实验追踪 |

### 2.2 开发/质量工具（首次安装）

| 包名 | 版本 | 用途 |
|------|------|------|
| pytest | 9.1.1 | 测试 |
| ruff | 0.15.22 | 静态检查 |
| mypy | 2.3.0 | 类型检查 |
| pytest-cov | 7.1.0 | 测试覆盖率 |
| types-pyyaml | 6.0.12.20260518 | PyYAML 类型存根 |

### 2.3 项目包

- 以 editable 模式安装 `iron_aging_gnn==0.1.0`（来源：`L4/pyproject.toml`）

## 3. 兼容性修复：PyTorch 升级

### 3.1 问题

初始环境为 `torch==2.5.1+cu121`，运行时触发警告：

```text
NVIDIA GeForce RTX 5060 Laptop GPU with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90.
```

RTX 5060（sm_120）需要 CUDA 12.8 构建的 PyTorch。

### 3.2 升级操作

1. 卸载旧包：`torch`, `torchvision`, `torchaudio`, `torch-geometric`, `torch-scatter`, `torch-sparse`
2. 安装 CUDA 12.8 版本：
   ```powershell
   .venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   .venv\Scripts\python.exe -m pip install torch-scatter torch-sparse torch-geometric --find-links https://data.pyg.org/whl/torch-2.11.0+cu128.html
   ```

### 3.3 升级后版本

| 包名 | 升级前 | 升级后 |
|------|--------|--------|
| torch | 2.5.1+cu121 | **2.11.0+cu128** |
| torchvision | 0.20.1+cu121 | **0.26.0+cu128** |
| torchaudio | 2.5.1+cu121 | **2.11.0+cu128** |
| torch-geometric | 2.8.0 | 2.8.0 |
| torch-scatter | 2.1.2+pt25cu121 | **2.1.2+pt211cu128** |
| torch-sparse | 0.6.18+pt25cu121 | **0.6.18+pt211cu128** |
| sympy | 1.13.1 | 1.14.0 |
| pandas | 3.0.3 | 2.3.3（mlflow 约束 pandas<3） |
| matplotlib | 3.11.0 | 3.10.3（scanpy 约束 !=3.11） |

## 4. 依赖冲突修复

安装过程中 `pip check` 发现以下历史遗留冲突并已修复：

| 冲突 | 修复方式 |
|------|----------|
| `anndata` 缺失 `zarr`, `array-api-compat`, `legacy-api-wrap` | 安装 zarr 3.2.1、array-api-compat 1.15.0、legacy-api-wrap 1.5 |
| `scanpy` 缺失 `fast-array-utils`, `pynndescent`, `seaborn`, `umap-learn` | 安装 fast-array-utils 1.5、pynndescent 0.6.0、seaborn 0.13.2、umap-learn 0.5.12 |
| `scanpy` 要求 `matplotlib !=3.11` | 降级 matplotlib 至 3.10.3 |
| `leidenalg` 缺失 `igraph` | 安装 igraph 1.0.0 |

## 5. 验证结果

### 5.1 依赖完整性

```powershell
.venv\Scripts\python.exe -m pip check
# No broken requirements found.
```

### 5.2 CUDA / GPU

```text
torch: 2.11.0+cu128
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
```

### 5.3 冒烟测试

```powershell
cd L4\scripts
..\..\.venv\Scripts\python.exe smoke_test.py
```

结果：**全部通过**，无 sm_120 兼容性警告。

### 5.4 静态检查

```powershell
.venv\Scripts\python.exe -m ruff check L4\src\iron_aging_gnn --output-format concise
# All checks passed!
```

## 6. 入口点修复

### 6.1 问题

`L4/pyproject.toml` 中注册的 console_scripts 入口点无法运行：

- `entry/` 目录缺少 `__init__.py`，安装后无法作为 Python 包导入
- `entry/train.py`、`entry/evaluate.py`、`entry/build_graph.py` 导入的是已废弃的 `phase4_v10_minibatch.py`

### 6.2 修复内容

| 文件 | 修改 |
|------|------|
| `L4/entry/__init__.py` | 新建，使 `entry` 成为合法包 |
| `L4/entry/train.py` | 改从 `phase4_v10_modular` 导入 `main` |
| `L4/entry/evaluate.py` | 改从 `phase4_v10_modular` 导入 `main` |
| `L4/entry/build_graph.py` | 重写为调用 `phase4_v10_modular.main(build_graph_only=True)`，支持 `--force-rebuild` |
| `L4/entry/diagnose_hgt.py` | 删除未使用的 `hgt_model_path` 变量，消除 ruff F841 错误 |
| `L4/scripts/phase4_v10_modular.py` | `main()` 新增 `build_graph_only` 参数，图构建完成后提前退出 |
| `L4/pyproject.toml` | `setuptools.packages.find` 同时搜索 `.` 和 `src`，包含 `entry*` 与 `iron_aging_gnn*` |

### 6.3 验证

四个入口点均已可执行：

```powershell
.venv\Scripts\iron-aging-train.exe --help
.venv\Scripts\iron-aging-evaluate.exe --help
.venv\Scripts\iron-aging-build-graph.exe --help
.venv\Scripts\iron-aging-predict.exe --help
```

运行示例：

```powershell
# 完整训练（三分支）
.venv\Scripts\iron-aging-train.exe

# 仅训练 SAGE
.venv\Scripts\iron-aging-train.exe --model sage

# 仅构建图缓存
.venv\Scripts\iron-aging-build-graph.exe

# 强制重建图缓存
.venv\Scripts\iron-aging-build-graph.exe --force-rebuild

# 重新评估
.venv\Scripts\iron-aging-evaluate.exe --reevaluate

# TCM 预测
.venv\Scripts\iron-aging-predict.exe --checkpoint results_v10_minibatch/sage_best.pt
```

## 7. 关键命令汇总

```powershell
# 激活/使用环境（无需 conda）
.venv\Scripts\python.exe --version

# 项目包 editable 安装（不强制升级依赖）
.venv\Scripts\python.exe -m pip install -e L4/ --no-deps

# 安装缺失依赖
.venv\Scripts\python.exe -m pip install transformers biopython optuna mlflow pytest ruff mypy pytest-cov types-pyyaml

# 运行质量门禁（直接调用 ruff + smoke_test）
.venv\Scripts\python.exe -m ruff check L4\src\iron_aging_gnn --output-format concise
.venv\Scripts\python.exe L4\scripts\smoke_test.py
```

## 8. 环境变量

无需额外设置环境变量。`KMP_DUPLICATE_LIB_OK=TRUE` 与 `PYTHONIOENCODING=utf-8` 由项目脚本内部设置。
