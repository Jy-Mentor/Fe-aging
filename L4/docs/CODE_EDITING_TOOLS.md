# 代码修改工具完整指南

> 本文档汇总了项目中最强、最快、最稳定的代码修改工具和方法。

---

## 一、工具安装状态

| 工具 | 版本 | 状态 | 安装位置 |
|---|---|---|---|
| **ast-grep (sg)** | 0.44.0 | ✅ 已安装 | WinGet 包管理器 |
| **LibCST** | 1.8.6 | ✅ 已安装 | pip (conda 环境) |
| **Bowler** | 0.9.0 | ✅ 已安装 | pip (conda 环境) |
| **serpl** | - | ❌ 未安装 | 需手动从 GitHub releases 下载 |

---

## 二、现有项目工具（L4/scripts/）

### 1. `_patch_file.py` — 流式字符串替换
**适用场景**: 大文件 (>10KB) 的精确字符串替换，不加载全文件到内存

```bash
python L4/scripts/_patch_file.py <目标文件> <旧字符串文件> <新字符串文件>
```

**特点**:
- 使用缓冲区流式处理，内存占用极低
- 支持跨行匹配
- 原子性操作（失败时不修改原文件）

### 2. `_edit_lines.py` — 行范围替换
**适用场景**: 修改特定行区间的内容

```bash
python L4/scripts/_edit_lines.py <目标文件> <起始行> <结束行> <替换内容文件>
```

**示例**: 替换第 50-100 行的内容
```bash
python L4/scripts/_edit_lines.py L4/src/models.py 50 100 new_code.txt
```

### 3. `_concat_files.py` — 多文件合并
**适用场景**: 将多个小文件合并为一个大文件

```bash
python L4/scripts/_concat_files.py <输出文件> <输入文件1> <输入文件2> ...
```

### 4. `_write_file.py` — 文件复制
**适用场景**: 备份文件或覆盖文件

```bash
python L4/scripts/_write_file.py <源文件> <目标文件>
```

### 5. `_base64_append.py` — Base64 追加
**适用场景**: 向二进制文件追加内容

---

## 三、ast-grep (sg) — AST 级重构工具 ⭐ 推荐

### 安装验证
```bash
sg --version
# 输出: ast-grep 0.44.0
```

### 基本用法

#### 1. 简单搜索
```bash
# 搜索所有 print 语句
sg -p "print($X)"

# 搜索特定函数调用
sg -p "torch.optim.AdamW($$$ARGS)"
```

#### 2. 搜索并替换
```bash
# 将 print(x) 替换为 logging.info(x)
sg -p "print($X)" -r "logging.info($X)" -l python --write

# 将 AdamW(lr=xxx) 替换为 AdamW(lr=yyy, weight_decay=0.01)
sg -p "AdamW($$$ARGS, lr=$LR)" -r "AdamW($$$ARGS, lr=$LR, weight_decay=0.01)" -l python --write
```

#### 3. 多文件批量替换
```bash
# 在整个 L4/src 目录替换
sg -p "old_pattern" -r "new_pattern" -l python L4/src/ --write
```

#### 4. 使用 YAML 规则文件
创建 `rules/my_rule.yaml`:
```yaml
id: my-transform
language: python
severity: warning
rule:
  pattern: torch.nn.Linear($IN, $OUT)
fix: nn.Linear($IN, $OUT)
```

应用规则:
```bash
sg scan -r rules/my_rule.yaml --fix
```

### 模式语法

| 元变量 | 含义 | 示例 |
|---|---|---|
| `$X` | 匹配单个 AST 节点 | `print($X)` 匹配 `print("hello")` |
| `$$$ARGS` | 匹配多个参数 | `func($$$ARGS)` 匹配 `func(a, b, c)` |
| `$_` | 匹配任意内容但不捕获 | `func($_)` |
| `$X($$$)` | 匹配函数调用 | |

### 常用场景

#### 场景 1: 重命名函数/变量
```bash
sg -p "old_function($$$ARGS)" -r "new_function($$$ARGS)" -l python --write
```

#### 场景 2: 添加参数
```bash
sg -p "train_sage($$$ARGS)" -r "train_sage($$$ARGS, use_amp=True)" -l python --write
```

#### 场景 3: 替换导入
```bash
sg -p "from torch import nn" -r "import torch.nn as nn" -l python --write
```

---

## 四、LibCST — Python 保留格式 AST 工具

### 安装验证
```bash
python -c "import libcst; print('LibCST installed')"
```

### 基本用法

#### 1. 编写 CST Transformer
创建 `transformers/add_type_hints.py`:
```python
import libcst as cst
from libcst.codemod import CodemodCommand, VisitorBasedCodemodCommand

class AddTypeHints(VisitorBasedCodemodVisitor):
    """为函数添加类型提示"""
    
    def leave_FunctionDef(self, original_node, updated_node):
        # 如果函数没有返回类型注解，添加 -> None
        if updated_node.returns is None:
            return updated_node.with_changes(
                returns=cst.Annotation(annotation=cst.Name("None"))
            )
        return updated_node

class AddTypeHintsCommand(CodemodCommand):
    DESCRIPTION = "Add type hints to functions"
    
    def get_transformer(self, context):
        return AddTypeHints(context)
```

#### 2. 运行 Codemod
```bash
python -m libcst.codemod L4/src/ --command AddTypeHintsCommand
```

#### 3. 使用内置 Codemod
```bash
# 列出所有可用 codemod
python -m libcst.codemod --list

# 运行特定 codemod
python -m libcst.codemod L4/src/ --command ConvertFormatStringCommand
```

### 优势
- ✅ 保留所有原始格式（缩进、空行、注释）
- ✅ 理解 Python 语法，不会破坏代码
- ✅ 支持复杂的 AST 转换

### 劣势
- ❌ 学习曲线较陡
- ❌ 仅限 Python

---

## 五、Bowler — Facebook Python 重构框架

### 安装验证
```bash
python -c "import bowler; print(f'Bowler {bowler.__version__}')"
# 输出: Bowler 0.9.0
```

### 基本用法

#### 1. 创建重构脚本
创建 `refactors/rename_train.py`:
```python
from bowler import Query

# 重命名函数
Query().select_function("old_train").rename("new_train").execute()

# 重命名方法
Query().select_method("Model", "forward").rename("predict").execute()

# 删除参数
Query().select_function("train_sage").remove_argument("use_memory_bank").execute()
```

#### 2. 运行重构
```bash
# 预览变更（不写入）
python -m bowler refactors/rename_train.py L4/src/

# 应用变更
python -m bowler refactors/rename_train.py L4/src/ --write
```

#### 3. 交互式审查
```bash
# 交互式模式，逐个审查变更
python -m bowler refactors/rename_train.py L4/src/ --interactive
```

### 常用操作

| 操作 | API | 示例 |
|---|---|---|
| 重命名 | `.rename(new_name)` | `.select_function("old").rename("new")` |
| 删除参数 | `.remove_argument(name)` | `.select_function("f").remove_argument("x")` |
| 添加参数 | `.add_argument(arg)` | `.select_function("f").add_argument("y=1")` |
| 重命名类 | `.select_class(old).rename(new)` | |

### 优势
- ✅ 流式 API，易于编写
- ✅ 交互式审查模式
- ✅ Facebook 内部大规模使用验证

### 劣势
- ❌ 基于 lib2to3，对 Python 3.12+ 新语法支持有限
- ❌ 复杂转换能力不如 LibCST

---

## 六、经典命令行工具

### 1. sed — 简单文本替换
```bash
# 替换所有文件中的字符串
Get-ChildItem -Recurse -Filter "*.py" | ForEach-Object {
    (Get-Content $_.FullName -Raw) -replace 'old_text', 'new_text' |
    Set-Content $_.FullName -NoNewline
}

# 单文件替换
$content = Get-Content file.py -Raw
$content -replace 'old', 'new' | Set-Content file.py
```

### 2. perl — 复杂正则替换
```bash
# 多行替换（需要安装 perl）
perl -i -0pe 's/old_pattern.*?new_pattern/replacement/gs' *.py
```

---

## 七、工具选型决策树

```
需要修改代码？
├── 简单文本替换（不关心语法结构）
│   ├── 单文件 → _patch_file.py 或 sed
│   └── 多文件 → PowerShell -replace 或 ast-grep
│
├── 语法级替换（需要理解代码结构）
│   ├── Python 专用
│   │   ├── 保留格式 → LibCST
│   │   └── 流式 API → Bowler
│   └── 跨语言 → ast-grep (sg)
│
├── 语义级重构（需要理解作用域/类型）
│   └── Python → rope
│
└── 大文件 (>10KB) 编辑
    ├── 精确替换 → _patch_file.py
    └── 行范围 → _edit_lines.py
```

---

## 八、最佳实践

### 1. 修改前必做
- [ ] 备份原文件（`git commit` 或复制）
- [ ] 使用 `--dry-run` 或预览模式查看变更
- [ ] 确认替换模式不会误匹配

### 2. 修改后必做
- [ ] 运行语法检查（`python -m py_compile file.py`）
- [ ] 运行测试验证功能
- [ ] 检查 git diff 确认变更符合预期

### 3. 大文件编辑规则
- 文件 >10KB 时，**禁止**使用 Trae IDE 的 Edit/Write 工具
- 必须使用 `_patch_file.py` 或 ast-grep
- 修改后验证文件完整性

---

## 九、快速参考卡片

### ast-grep 速查
```bash
# 搜索
sg -p "pattern" -l python

# 替换（预览）
sg -p "old" -r "new" -l python

# 替换（写入）
sg -p "old" -r "new" -l python --write

# 扫描规则
sg scan -r rules/

# 扫描并修复
sg scan -r rules/ --fix
```

### Bowler 速查
```python
from bowler import Query

# 重命名
Query().select_function("old").rename("new").execute()

# 删除参数
Query().select_function("f").remove_argument("x").execute()

# 添加装饰器
Query().select_function("f").add_decorator("@timer").execute()
```

### LibCST 速查
```python
import libcst as cst

# 解析代码
tree = cst.parse_module(code)

# 访问节点
class MyVisitor(cst.CSTVisitor):
    def visit_FunctionDef(self, node):
        print(f"Found function: {node.name.value}")

# 转换代码
class MyTransformer(cst.CSTTransformer):
    def leave_FunctionDef(self, original, updated):
        return updated.with_changes(...)

# 应用转换
tree.visit(MyVisitor())
new_code = tree.visit(MyTransformer()).code
```

---

## 十、serpl 安装（可选）

serpl 是终端 UI 批量替换工具，类似 VS Code 的查找替换界面。

### 手动安装
1. 访问 [GitHub Releases](https://github.com/yassinebridi/serpl/releases)
2. 下载 Windows 版本（`serpl-x86_64-pc-windows-msvc.zip`）
3. 解压到 PATH 中的目录（如 `C:\Windows\System32` 或用户 PATH）
4. 运行 `serpl` 启动

### 使用
```bash
# 在当前目录搜索替换
serpl

# 指定目录
serpl L4/src/
```
