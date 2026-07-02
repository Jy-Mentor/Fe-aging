# v24 项目静态质量审查报告

**审查对象**：`L4/scripts/phase4_v10_minibatch.py` 及 v24 相关输入/输出文件  
**审查日期**：2026-06-28  
**审查方式**：真实文件检查 + 静态代码分析 + 最小化可复现 smoke test（未运行完整训练）  
**输出路径**：`d:\铁衰老 绝不重蹈覆辙\L4\docs\v24_quality_audit.md`

---

## 总体结论

v24 数据层基本完整：实验活性数据、ESM-2 蛋白嵌入、疾病边、铁衰老 96 基因覆盖、TCM 化合物池 SMILES 质量均符合预期。但**模型架构层存在一处可导致 HGT mini-batch 训练/验证直接崩溃的缺陷**（疾病节点子图缺少 `disease.x`），且**蛋白冷启动的 mini-batch 邻接表隔离不完整**（疾病边未对验证蛋白过滤）。代码工业化方面，存在配置系统弃用、大量魔法数字、`try-except: pass` 静默失败、ruff 189 条告警等问题。学术规范方面，评估指标缺少 P@K/EF/ROCE，蛋白冷启动协议在 mini-batch 降级路径上存在泄露风险。

| 类别 | 通过项 | 关键问题数 |
|------|--------|------------|
| A. 数据质量 | 5/5 | 0 |
| B. 模型架构 | 2/4 | 2（1 个 CRITICAL） |
| C. 代码工业化 | 1/4 | 3 |
| D. 学术规范 | 1/4 | 3 |

---

## A. 数据质量

### A1. `experimental_actives_detail_cleaned.csv` 完整性 ✅

| 检查项 | 结果 |
|--------|------|
| 文件存在 | ✅ |
| 行数 | 43,238 |
| 必需列 `gene`/`canonical_smiles`/`uniprot_id` | ✅ 全部存在 |
| 空 SMILES | 0 |
| 唯一基因数 | 42 |
| 唯一 SMILES 数 | 41,283 |

**结论**：数据完整，列齐全，SMILES 无缺失。

### A2. `esm2_protein_embeddings.npz` 蛋白数量 ✅

| 检查项 | 结果 |
|--------|------|
| 文件存在 | ✅ |
| 蛋白数 | **103** |
| 嵌入维度 | 640（一致） |

**结论**：符合“103 个蛋白含 ACSL4 等核心基因”的 v24 描述。

### A3. `disease_gene_edges.csv` 存在性与边数 ✅

| 检查项 | 结果 |
|--------|------|
| 文件存在 | ✅ |
| 行数 | **58** |
| 列 | `disease_name`, `disease_type`, `gene_symbol`, `evidence`, `source`, `padj`, `nes` |
| 唯一疾病 | 1（GSE61616 Ferroaging） |
| 唯一基因 | 58 |

**结论**：58 条疾病-蛋白边存在，结构与预期一致。

### A4. 铁衰老 96 基因 ESM-2 特征覆盖 ✅

| 检查项 | 结果 |
|--------|------|
| 铁衰老基因数 | 96 |
| 在 ESM-2 npz 中缺失 | **0** |

**结论**：全部 96 个铁衰老基因均有 ESM-2 嵌入。

### A5. TCM 化合物池 `tcm_compound_pool_v21_Alevel.csv` SMILES 质量 ✅

| 检查项 | 结果 |
|--------|------|
| 文件存在 | ✅ |
| 行数 | 628 |
| SMILES 列 | `SMILES_std` |
| RDKit 可解析 | **628/628 (100%)** |

**结论**：SMILES 质量 100% 合格。

---

## B. 模型架构

### B1. v24 改动检查（疾病节点、蛋白扩展、版本号、输出文件名） ⚠️

**状态**：部分符合，但存在关键实现缺陷。

- ✅ 文档字符串与日志中明确标注 `v24`，输出文件名统一为 `v24`：
  - `L4/results_v10_minibatch/sage_best_v24.pt`
  - `L4/results_v10_minibatch/hgt_best_v24.pt`
  - `L4/results_v10_minibatch/tcm_predictions_full_v24.csv`
  - `L4/results_v10_minibatch/tcm_top_candidates_v24.csv`
  - `L4/results_v10_minibatch/model_performance_v24.csv`
- ✅ 铁衰老 96 基因被强制加入蛋白节点（`build_graphs_and_adj`，line 679-688）。
- ✅ 疾病节点通过 `disease_gene_edges.csv` 被读入并构建异质边（line 784-885）。
- ❌ **疾病节点加入后，mini-batch 子图训练/验证路径未正确填充 `disease.x`，会导致 HGT 前向传播 KeyError**（详见 B3，CRITICAL）。

### B2. HGT 模型是否正确处理 disease 节点嵌入 ⚠️

**相关代码**：`L4/scripts/phase4_v10_minibatch.py` line 1363-1423

- `HGTLinkPredictor` 根据 `node_feat_dims["disease_count"]` 创建 `disease_embed`。
- `forward` 中通过 `disease_embed(x_dict["disease"].squeeze(-1).long())` 将 disease 初始索引映射为隐藏向量。
- **问题**：全图 `hetero_data["disease"].x` 被初始化为 `torch.zeros(n_diseases, 1)`（line 884）。当 `n_diseases > 1` 时，所有 disease 节点共享索引 0，初始嵌入完全相同，仅能通过后续消息传递区分。此设计虽可运行，但削弱了多疾病节点的可辨识性。

| 严重度 | MEDIUM |
|--------|--------|
| 修复建议 | 使用 `torch.arange(n_diseases).unsqueeze(-1)` 作为 disease 初始索引，使每个 disease 节点拥有独立的可学习嵌入。 |

### B3. `sample_hetero_subgraph` 是否正确采样 disease 节点 ❌

**相关代码**：`L4/scripts/phase4_v10_minibatch.py` line 1003-1121

**问题**：
1. `sample_hetero_subgraph` 在 2-hop 采样时会把与蛋白相连的疾病节点加入 `diseases` 集合，并构建 `protein ↔ disease` 边（line 1090-1119）。
2. 函数返回了 `disease_sorted` 和 `disease_map`，但**从未为子图设置 `sg["disease"].x`**。
3. `train_hgt`（line 3271 起）、`_validate_hgt_minibatch`（line 2862 起）、`_validate_hgt_protein_cold_minibatch`（line 2974 起）均只使用 `comp_sorted/prot_sorted/path_sorted`，忽略 `disease_sorted`。
4. 当子图边索引中存在 `protein→disease` 或 `disease→protein` 边而 `x_dict` 缺少 `disease` 时，`HGTConv` 会抛出 `KeyError: 'disease'`。

**可复现验证**：`L4/docs/_audit_hgt_disease.py`（运行后输出如下）

```text
EXPECTED ERROR: KeyError: 'disease'
EXPECTED: forward succeeded with disease.x
```

由于 `disease_gene_edges.csv` 真实存在（58 条边），HGT mini-batch 训练与 OOM 降级验证在当前代码下**无法运行**。

| 严重度 | **CRITICAL** |
|--------|--------------|
| 修复建议 | 在 `sample_hetero_subgraph` 返回前，仿照 pathway 处理方式为 disease 节点填充 `x`：<br>`sg["disease"].x = torch.zeros(len(disease_sorted), 1, dtype=torch.float32)`（全图 `hetero_data["disease"].x` 的设计即为 `zeros(n_diseases, 1)`，子图只需保持同样语义）。同时确保调用方使用 `disease_map` 构建反向边。 |

### B4. 训练/验证图隔离是否正确保留 disease 边 ❌

**相关代码**：
- `_build_val_safe_hetero_data`（line 2366-2414）：对全图 HeteroData 的隔离是正确的，会按 `src_type == "protein"` 或 `dst_type == "protein"` 移除涉及验证蛋白的 disease 边。
- `_build_train_safe_hetero_adj`（line 2493-2527）：对 mini-batch 邻接表的隔离**不完整**。其 `else` 分支对 `("protein", "associated_with", "disease")` 和 `("disease", "involves", "protein")` 直接原样复制，未按 `val_prot_set` 过滤。
- `_build_val_safe_hetero_adj`（line 2530-2543）直接复用 `_build_train_safe_hetero_adj`，问题相同。

**后果**：在 HGT 蛋白冷启动的 OOM 降级 mini-batch 验证中，`hetero_adj_prot_cold` 仍保留验证蛋白的疾病边。当 `sample_hetero_subgraph(..., seed_proteins=val_prot_list)` 被调用时，验证蛋白可通过 disease 节点与训练蛋白建立消息传递路径，造成**冷启动信息泄露**。

**可复现验证**：`L4/docs/_audit_adj_leak.py`（运行后输出如下）

```text
protein->disease edges kept: {10: [0], 99: [0]}
val protein 99 still has disease edge? True
disease->protein edges kept: {0: [10, 99]}
val protein 99 still reached from disease? True
```

| 严重度 | **HIGH** |
|--------|----------|
| 修复建议 | 在 `_build_train_safe_hetero_adj` 中显式处理 disease 边类型：<br>1. `("protein", "associated_with", "disease")`：跳过 `src in val_prot_set` 的条目，并过滤 `dst`（若未来存在验证 disease 集合）。<br>2. `("disease", "involves", "protein")`：跳过 `dst in val_prot_set` 的条目。<br>使 mini-batch 邻接表与全图 HeteroData 的隔离语义严格一致。 |

---

## C. 代码工业化

### C1. 硬编码路径 / 魔法数字 ❌

**状态**：配置系统存在但未使用；主流程充满魔法数字。

- `L4/configs/default.yaml` 与 `L4/src/iron_aging_gnn/utils/config.py` 已提供完整的 pydantic 配置类（SageConfig/HgtConfig/ModelConfig 等）。
- 但 `phase4_v10_minibatch.py` 的 `main()` **完全没有调用 `load_config()`**，所有超参数直接硬编码：
  - `epochs=3, pretrain_epochs=3`（line 3999, 4027），而 `default.yaml` 中默认值为 **15/10**。
  - `hidden_dim=64, out_dim=64, num_layers=2, dropout=0.5, num_heads=2`（line 3991-4023）。
  - `batch_size=256/128`, `lr=5e-4/1e-3`, `num_neighbors=[32,16]` 等。
  - 拆分比例 `0.85`、`0.20`，损失权重 `0.6/0.4`、`focal_alpha=0.75` 等散落在代码中。
- 文件名（如 `experimental_actives_detail_cleaned.csv`、`tcm_compound_pool_v21_Alevel.csv`）亦未集中管理。

| 严重度 | **HIGH** |
|--------|----------|
| 修复建议 | 在 `main()` 开头加载 `load_config("L4/configs/default.yaml")`，用配置对象替换所有硬编码超参数与路径常量；保留命令行覆盖接口。 |

### C2. 错误处理 ⚠️

**状态**：关键路径有保护，但存在静默吞错。

- ✅ 关键文件缺失时调用 `logger.error` 并 `sys.exit(1)`（line 437-439, 475-476）。
- ✅ ESM-2 加载/计算失败时有 warning 并降级（line 540-553）。
- ❌ `_compute_ecfp4` 中存在 `try...except Exception: pass`（line 240-241），非法 SMILES 的指纹失败被静默忽略，仅保留全零向量。
- ❌ 同一函数中 `mol is None` 时直接 `continue`，未记录日志（line 234-235），违反项目“缺失数据必须写日志警告”的铁律。

| 严重度 | **MEDIUM** |
|--------|------------|
| 修复建议 | 将 `except Exception: pass` 改为 `logger.warning(...)` 并继续；对无法解析的 SMILES 记录行号/原始字符串。 |

### C3. 日志 ✅

- 日志格式统一：`%(asctime)s [%(levelname)s] %(message)s`。
- 训练、验证、图构建、采样各阶段均有 INFO 级日志。
- 建议：将 C2 中的静默失败转为 WARNING，增强可审计性。

### C4. 可通过 `py_compile` / lint ✅ ❌

| 检查项 | 结果 |
|--------|------|
| `python -m py_compile`（phase4 及 src 关键模块） | ✅ 通过 |
| `ruff check` | ❌ **189 条告警** |

**ruff 统计（前 10 类）**：

| 数量 | 规则 | 说明 |
|------|------|------|
| 86 | UP006 | 使用 `list` 替代 `List` 等 typing 泛型 |
| 18 | E501 | 行过长 |
| 16 | UP045 | 使用 `X \| None` 替代 `Optional[X]` |
| 12 | B905 | `zip()` 缺少 `strict=` |
| 9 | N806 | 函数内变量未小写 |
| 8 | D103 | 公共函数缺少 docstring |
| 6 | B006 | 可变默认参数 |
| 6 | F841 | 未使用变量 |
| 5 | D102 | 公共方法缺少 docstring |
| 3 | D107 | `__init__` 缺少 docstring |

| 严重度 | **MEDIUM** |
|--------|------------|
| 修复建议 | 运行 `ruff check L4/scripts/phase4_v10_minibatch.py L4/src/iron_aging_gnn --fix` 修复安全项；对 B006/B905 等手动修正；将 lint 纳入 CI/提交前钩子。 |

---

## D. 学术规范

### D1. 版本号与输出文件名一致性 ✅

- 代码注释、日志、模型 checkpoint、预测结果、性能文件均使用 `v24`。
- 未发现 `v23` 残留到新输出文件名中（`results_v10_minibatch` 目录中的 `*_v23.*` 为历史文件，非本次生成）。

### D2. 数据泄漏风险 ⚠️

- **全图隔离**：`_build_val_safe_homo_edge_index` 与 `_build_val_safe_hetero_data` 正确移除了验证集化合物/验证集蛋白的所有 CPI/PPI/通路边，包括 disease 边。**全图评估路径无泄露**。
- **mini-batch 隔离**：`_build_train_safe_hetero_adj` / `_build_val_safe_hetero_adj` 未对 disease 边按验证蛋白过滤（详见 B4）。在 HGT OOM 降级验证中，验证蛋白可能通过 disease 节点接收训练侧信息，导致蛋白冷启动 AUPR 虚高。

| 严重度 | **HIGH** |
|--------|----------|

### D3. 评估指标完整性 ❌

**现状**：代码中仅计算 AUC 与 AUPR：
- `_validate_sage` / `_validate_sage_protein_cold`（line 2624-2625, 2723-2724）
- `_validate_hgt` / `_validate_hgt_minibatch` / `_validate_hgt_protein_cold`（line 2814-2815, 2938-2939, 3056-3057, 3166-3167）
- `model_performance_v24.csv` 仅记录 `best_auc` 与 `best_aupr`（line 4249-4259）。

**缺失指标**：P@K（Precision@K）、EF（Enrichment Factor）、ROCE（Receiver Operating Characteristic Enrichment）。这些在药物-靶点预测论文中属于标准报告指标。

| 严重度 | **HIGH** |
|--------|----------|
| 修复建议 | 在验证函数中增加 `compute_enrichment_metrics(y_true, y_score, k_values=[10, 50, 100])`，计算 P@K、EF、ROCE，并写入 `model_performance_v24.csv` 与日志。 |

### D4. 蛋白冷启动协议严格性 ⚠️

**已做到**：
- 将蛋白分为 CPI 蛋白与非 CPI 蛋白，各自按 20% 拆分，保证验证集既有交互蛋白也有 zero-shot 蛋白（line 3859-3876）。
- 全图训练/验证/蛋白冷启动图严格分离（line 3957-3985）。
- 最终重新评估使用 `homo_edge_index_prot_cold` / `hetero_data_prot_cold`（line 4044-4058）。

**未做到**：
- mini-batch 邻接表未隔离 disease 边（B4），蛋白冷启动 OOM 降级验证不严格。
- 评估指标未单独记录蛋白冷启动 AUC/AUPR 到性能文件，仅打印在日志中。

| 严重度 | **HIGH** |
|--------|----------|
| 修复建议 | 修复 B4 后，在 `model_performance_v24.csv` 中增加 `prot_cold_auc`、`prot_cold_aupr`、`compound_cold_auc`、`compound_cold_aupr` 四列。 |

---

## 附录：审查用脚本输出摘要

### 数据检查（`_audit_data_check.py`）

```text
A1 rows=43238, missing_cols=[], null_smiles=0
A2 proteins=103, dims=[640]
A3 rows=58, cols=[...]
A4 missing=[], n_missing=0
A5 rows=628, valid=628, invalid_count=0
```

### HGT disease 子图 smoke test（`_audit_hgt_disease.py`）

```text
EXPECTED ERROR: KeyError: 'disease'
EXPECTED: forward succeeded with disease.x
```

### 邻接表泄露 smoke test（`_audit_adj_leak.py`）

```text
val protein 99 still has disease edge? True
val protein 99 still reached from disease? True
```

### lint

```text
ruff check ... --statistics
Found 189 errors.
```

### py_compile

```text
PY_COMPILE_OK
```

---

## 修复优先级建议

1. **CRITICAL**：修复 `sample_hetero_subgraph` 中 `disease.x` 缺失问题（B3），否则 v24 HGT 分支无法训练。
2. **HIGH**：修复 `_build_train_safe_hetero_adj` / `_build_val_safe_hetero_adj` 对 disease 边的验证蛋白过滤（B4），堵住蛋白冷启动 mini-batch 泄露。
3. **HIGH**：将 `main()` 中的硬编码超参数改为读取 `default.yaml` 配置（C1），避免生产训练使用 `epochs=3` 这种调试参数。
4. **HIGH**：补充 P@K/EF/ROCE 评估指标并写入性能文件（D3）。
5. **MEDIUM**：替换 `_compute_ecfp4` 中的 `except: pass`（C2），增加缺失/无效 SMILES 日志。
6. **MEDIUM**：运行 `ruff --fix` 并解决 B006/B905 等手动项（C4）。
7. **LOW**：改进 disease 节点初始索引，避免所有 disease 节点共享同一嵌入（B2）。
