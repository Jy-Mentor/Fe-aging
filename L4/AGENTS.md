# AGENTS.md — 铁衰老 GNN 项目 AI 导航

## 1. 项目定位

基于异构图神经网络（Heterogeneous GNN）的铁衰老-中药单体-脑缺血再灌注（CIRI）药物重定位系统。

核心任务：预测中药单体活性成分对 CIRI 相关铁衰老靶标的潜在治疗作用。

## 2. 架构概览

```
L4/
├── entry/                  # 入口脚本
│   ├── train.py            # 训练入口
│   ├── evaluate.py         # 评估入口
│   ├── predict.py          # TCM 预测入口
│   ├── build_graph.py      # 图构建入口
│   └── run_pipeline.py     # 端到端流水线
├── src/iron_aging_gnn/     # 核心模块
│   ├── data/               # 数据加载、常量、特征
│   ├── evaluation/         # AUC/AUPR/ROCE/BEDROC/EF/NDCG
│   ├── graph/              # 图构建、采样、负采样、分割
│   ├── models/             # SAGE / HGT / SimpleHGN / RGCN / 解码器
│   ├── prediction/         # TCM 预测推理
│   ├── training/           # 训练器、配置、组件
│   └── utils/              # 配置、设备、日志、种子
├── configs/default.yaml    # 主配置文件
├── scripts/                # 数据处理与实验脚本
└── tests/                  # 单元测试
```

## 3. 不可违背的硬约束

- 蛋白特征维度必须与模型初始化参数匹配。
- HGT/SimpleHGN 验证必须使用全图（禁止 minibatch），避免化合物节点孤立。
- SimpleHGN 必须通过 `edge_attr` 向 `GATv2Conv` 传入边类型嵌入。
- SAGE 验证必须使用全图前向传播。
- CPI 边（监督信号）禁止 DropEdge；仅 PPI/pathway 等结构边可 DropEdge。
- SAGE 验证后需立即释放张量并调用 `torch.cuda.empty_cache()`。
- 学习率调度器 `step()` 必须在每个 epoch 结束时调用。
- HGT/SimpleHGN 蛋白输入严格为 640D ESM-2；通路信息通过异构图结构传递。
- HGT/SimpleHGN 温度参数固定为 1.0。
- TCM 预测对 top-k 候选必须使用 residue-aware 解码路径。
- SAGE 配置独立：`hidden_dim:64, out_dim:64, num_layers:2`；不可与 HGT/SimpleHGN（128/3）混淆。

## 4. 关键文件映射

| 主题 | 必读文件 |
|---|---|
| 主配置 | `configs/default.yaml` |
| SAGE 模型 | `src/iron_aging_gnn/models/sage.py` |
| HGT 模型 | `src/iron_aging_gnn/models/hgt.py` |
| SimpleHGN 模型 | `src/iron_aging_gnn/models/simplehgn.py` |
| 解码器 | `src/iron_aging_gnn/models/decoders.py` |
| 训练逻辑 | `src/iron_aging_gnn/training/trainer.py` |
| 图构建 | `src/iron_aging_gnn/graph/build.py` |
| 评估 | `src/iron_aging_gnn/evaluation/metrics.py` |
| 常量 | `src/iron_aging_gnn/data/constants.py` |

## 5. 修改任何代码前必须执行的操作

1. 读取 `configs/default.yaml` 中对应模型分支的配置。
2. 读取目标模型文件的前 80 行（类定义与 `__init__` 签名）。
3. 确认蛋白特征维度：`640 + n_pathways`。
4. 运行相关单元测试：`pytest tests/test_models.py tests/test_evaluation.py`。
5. 修改后重新运行 smoke test 或最小复现脚本。

## 6. 常见陷阱

- 维度不匹配：HGT/SimpleHGN 用 128/3，SAGE 用 64/2。
- 验证时 minibatch 导致 HGT/SimpleHGN 化合物节点孤立。
- `empty_cache()` 每 batch 调用会严重降低吞吐量，应每 10 batch 一次。
- SimpleHGN 未传 `edge_attr` 会退化为普通 GAT。
- CPI 边被 DropEdge 会污染监督信号。
