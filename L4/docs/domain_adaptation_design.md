# v23 双分支 GNN 蛋白域自适应对抗训练设计

> 目标：在现有 SAGE + HGT 双分支 CPI 模型上引入对标 DrugBAN/CDAN 的域自适应对抗训练，缓解蛋白冷启动场景下的分布偏移，提升 `prot_aupr`。
>
> 参考：Bai et al. (2023) *Interpretable bilinear attention network with domain adaptation improves drug-target prediction*, Nat. Mach. Intell.; Long et al. (2018) *Conditional Adversarial Domain Adaptation* (CDAN), NeurIPS.

---

## 1. 当前代码基线分析

已阅读 `d:\铁衰老 绝不重蹈覆辙\L4\scripts\phase4_v10_minibatch.py` 中的相关模块：

| 模块 | 关键信息 |
|------|----------|
| `SAGELinkPredictor` | 同构图编码器，`out_dim=64`，`forward` 输出 `(N, 64)` 节点嵌入；蛋白节点从 ESM-2 640 维经 `prot_feat_proj` 投影到 `hidden_dim` 再图卷积。 |
| `HGTLinkPredictor` | 异构图编码器，`out_dim=64`，`forward` 输出 `dict{"compound":..., "protein":...}`；蛋白投影 `prot_proj` + HGTConv。 |
| `sample_homo_subgraph` | SAGE mini-batch 采样，以化合物为种子做邻居采样，返回 `node_list, node_to_local, edge_index`。 |
| `sample_hetero_subgraph` | HGT mini-batch 采样，已有 `seed_proteins` 参数可把指定蛋白作为孤立节点加入子图（v18 用于 OOM 降级验证）。 |
| `train_sage` / `train_hgt` | 内嵌 `_train_one_epoch`；每 batch 得到 `comp_emb` 与 `prot_emb` 后调用 `_compute_cpi_loss`；存在 `val_proteins: set` 形参。 |
| `_compute_cpi_loss` | 返回 `loss = 0.6*(pos_loss+neg_loss) + bpr_weight*bpr_loss`（InfoNCE 默认关闭）。 |
| 训练/验证图隔离 | `homo_adj_train` / `hetero_adj_train` 已移除所有与验证蛋白相连的边，避免信息泄露。 |

设计原则：**域自适应只能在训练阶段引入；验证蛋白必须以“孤立节点”形式进入训练 batch，不能带入任何 CPI/PPI/通路邻居边，否则破坏蛋白冷启动评估的隔离性。**

---

## 2. ProteinDomainDiscriminator 模块与 GRL 设计

### 2.1 梯度反转层（GRL）

直接复用 DANN 经典实现，符号与 DrugBAN 一致：

```python
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """GRL：前向恒等，反向传播时梯度乘以 -alpha。"""
    return GradientReversalFunction.apply(x, alpha)
```

### 2.2 蛋白域判别器

输入为蛋白节点最终嵌入（`out_dim=64`），输出二分类 logit（domain 0 = 训练蛋白，domain 1 = 验证/冷启动蛋白）。

```python
class ProteinDomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int = 64, hidden_dims: tuple = (128, 64), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, prot_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prot_emb: (N_prot, out_dim)
        Returns:
            logits: (N_prot,)   domain=0（训练）或 domain=1（验证）
        """
        return self.net(prot_emb).squeeze(-1)
```

> 说明：隐藏层宽度 `(128, 64)` 可根据过拟合情况缩放到 `(64, 32)`；`LayerNorm + Dropout` 与现有 v23 风格保持一致。

---

## 3. 域标签分配与 Mini-Batch 实现策略

### 3.1 域定义

| 节点集合 | domain 标签 | 是否参与 CPI 损失 | 是否参与域损失 | 是否可带图边 |
|----------|-------------|-------------------|----------------|--------------|
| 训练蛋白（train proteins） | `0` | 是 | 是 | 是 |
| 验证/冷启动蛋白（val proteins） | `1` | 否（训练图中无 CPI 边） | 是 | **否，必须为孤立节点** |

### 3.2 SAGE 分支的 mini-batch 实现

当前 `sample_homo_subgraph` 只接受化合物种子，需要增加 `seed_proteins` 参数，把这些蛋白作为孤立节点加入子图：

```python
def sample_homo_subgraph(
    seed_compounds: List[int],
    homo_adj: Dict[int, List[int]],
    num_neighbors: List[int] = [32, 16],
    seed: Optional[int] = None,
    seed_proteins: Optional[List[int]] = None,   # <- 新增
):
    if seed is not None:
        random.seed(seed)
    nodes = set(seed_compounds)
    if seed_proteins:
        nodes.update(seed_proteins)
    frontier = set(seed_compounds)
    # ... 邻居采样逻辑不变 ...
    node_list = sorted(nodes)
    node_to_local = {n: i for i, n in enumerate(node_list)}
    # ... 边构建逻辑不变，seed_proteins 自然不会有出/入边 ...
    return node_list, node_to_local, edge_index
```

在 `train_sage._train_one_epoch` 中：

```python
# 每 batch 随机抽取 k 个验证蛋白作为域自适应目标样本
k_val_prot = min(len(val_proteins), max(16, batch_size // 4))
val_prot_batch = random.sample(sorted(val_proteins), k_val_prot) if val_proteins else []

node_list, node_to_local, edge_index = sample_homo_subgraph(
    batch_seeds, homo_adj, num_neighbors,
    seed=epoch * 10000 + batch_start,
    seed_proteins=val_prot_batch)

# ... 前向 ...
n_compounds_in_sub = sum(1 for n in node_list if n < n_compounds)
node_emb = model(sub_x, edge_index, n_compounds=n_compounds_in_sub)

# 蛋白局部索引（含训练蛋白 + 孤立验证蛋白）
prot_local_indices = [i for i, n in enumerate(node_list) if n >= n_compounds]
if not prot_local_indices:
    continue
prot_emb = node_emb[torch.tensor(prot_local_indices, device=DEVICE)]

# 构建 domain 标签
domain_labels = torch.zeros(len(prot_local_indices), device=DEVICE)
for i, local_pos in enumerate(prot_local_indices):
    global_prot_idx = node_list[local_pos] - n_compounds
    if global_prot_idx in val_proteins:
        domain_labels[i] = 1.0

# 域损失
if use_domain_adapt and len(val_proteins) > 0:
    domain_logits = discriminator(grad_reverse(prot_emb, alpha=1.0))
    domain_loss = F.binary_cross_entropy_with_logits(domain_logits, domain_labels)
    loss = loss + lambda_adv * domain_loss
```

**关键点**：`val_prot_batch` 只进入 `node_list` 作为孤立节点，因此：
- 不通过 `homo_adj_train` 拉取任何 CPI/PPI 边；
- 不会出现在 `_compute_cpi_loss` 的正/负样本中（因为训练正样本只存在于 `precomputed_pos` 的训练化合物-训练蛋白对中）；
- Memory Bank 仍只更新训练蛋白子集。

### 3.3 HGT 分支的 mini-batch 实现

`sample_hetero_subgraph` 已支持 `seed_proteins`，直接复用：

```python
k_val_prot = min(len(val_proteins), max(16, batch_size // 4))
val_prot_batch = random.sample(sorted(val_proteins), k_val_prot) if val_proteins else []

sg, comp_sorted, prot_sorted, path_sorted, comp_map, prot_map = sample_hetero_subgraph(
    batch_seeds, hetero_adj, num_neighbors,
    seed=epoch * 10000 + batch_start,
    seed_proteins=val_prot_batch)

# ... 前向 ...
hgt_out = model(sg.x_dict, sg.edge_index_dict)
prot_emb = hgt_out["protein"]          # 含训练蛋白 + 孤立验证蛋白

# domain 标签：prot_sorted 中属于 val_proteins 的为 1
domain_labels = torch.tensor(
    [0.0 if p not in val_proteins else 1.0 for p in prot_sorted],
    device=DEVICE, dtype=torch.float32)

if use_domain_adapt and len(val_proteins) > 0:
    domain_logits = discriminator(grad_reverse(prot_emb, alpha=1.0))
    domain_loss = F.binary_cross_entropy_with_logits(domain_logits, domain_labels)
    loss = loss + lambda_adv * domain_loss
```

HGT 通路嵌入仍只来自 `path_sorted` 中训练蛋白关联的通路；验证蛋白作为孤立节点不带 `belongs_to` 边，因此不会引入新的通路节点。

### 3.4 两类分支的统一封装（可选）

为减少重复，可在 `_compute_cpi_loss` 外新增工具函数：

```python
def compute_domain_adversarial_loss(
    prot_emb: torch.Tensor,
    prot_global_indices: List[int],
    val_proteins: set,
    discriminator: ProteinDomainDiscriminator,
    lambda_adv: float,
) -> torch.Tensor:
    if lambda_adv <= 0.0 or not val_proteins:
        return torch.tensor(0.0, device=prot_emb.device)
    domain_labels = torch.tensor(
        [0.0 if p not in val_proteins else 1.0 for p in prot_global_indices],
        device=prot_emb.device, dtype=torch.float32)
    domain_logits = discriminator(grad_reverse(prot_emb, alpha=1.0))
    return F.binary_cross_entropy_with_logits(domain_logits, domain_labels)
```

---

## 4. 总损失函数与 `lambda_adv` Warm-Up 策略

### 4.1 总损失

```
L_total = L_CPI + lambda_adv * L_domain
```

其中：
- `L_CPI` 保持不变：`0.6*(pos_loss + neg_loss) + bpr_weight*bpr_loss`；
- `L_domain = BCEWithLogitsLoss(discriminator(GRL(prot_emb)), domain_labels)`；
- `lambda_adv` 控制对抗强度。

### 4.2 Warm-Up 调度

建议从 `lambda_min=0.01` 平滑增长至 `lambda_max=0.1`，避免早期训练被域损失主导而破坏 CPI 主任务学习。使用 Sigmoid 型曲线：

```python
def get_lambda_adv(epoch: int, init_epoch: int = 5, total_epochs: int = 100,
                   lambda_min: float = 0.01, lambda_max: float = 0.1) -> float:
    if epoch <= init_epoch:
        return 0.0
    progress = (epoch - init_epoch) / max(1, total_epochs - init_epoch)
    # 从 0 平滑过渡到 1 的 Sigmoid 曲线
    factor = 2.0 / (1.0 + math.exp(-5.0 * progress)) - 1.0
    return lambda_min + (lambda_max - lambda_min) * factor
```

若需更保守，可改用线性 warm-up：

```python
def get_lambda_adv_linear(epoch, init_epoch=5, total_epochs=100,
                          lambda_min=0.01, lambda_max=0.1):
    if epoch <= init_epoch:
        return 0.0
    progress = (epoch - init_epoch) / max(1, total_epochs - init_epoch)
    return lambda_min + (lambda_max - lambda_min) * progress
```

> 推荐先使用 Sigmoid 版本，因为对抗训练在中后期（模型已学到初步 CPI 表示）引入更稳定。

### 4.3 在训练函数中的调用位置

以 `train_sage` 为例，在主训练循环外初始化：

```python
discriminator = ProteinDomainDiscriminator(in_dim=model.out_dim).to(DEVICE)
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(discriminator.parameters()),
    lr=lr, weight_decay=1e-4)
```

每个 batch 内：

```python
lambda_adv = get_lambda_adv(epoch, init_epoch=da_init_epoch, total_epochs=epochs,
                            lambda_min=da_lambda_min, lambda_max=da_lambda_max)

# ... 计算 cpi_loss ...
loss = cpi_loss

if use_domain_adapt and lambda_adv > 0.0 and len(val_proteins) > 0:
    domain_loss = compute_domain_adversarial_loss(
        prot_emb, [node_list[i] - n_compounds for i in prot_local_indices],
        val_proteins, discriminator, lambda_adv)
    loss = loss + lambda_adv * domain_loss
```

> 注意：GRL 的 `alpha` 固定为 `1.0`，由 `lambda_adv` 统一缩放梯度；避免同时调节两个超参。

---

## 5. 对蛋白冷启动 AUPR 的预期机制

蛋白冷启动的核心难点：验证/冷启动蛋白在训练图中被完全隔离，模型只能基于其自身特征（ESM-2 + 通路）生成嵌入，而无法像训练蛋白那样聚合 CPI/PPI 邻居信息。这导致训练分布与冷启动分布存在本质偏移：

- **训练蛋白**：`ESM-2 → proj → GNN 聚合邻居 → out_dim`；
- **冷启动蛋白**：`ESM-2 → proj → out_dim`（无聚合）。

引入蛋白域自适应后：

1. **分布对齐**：判别器迫使蛋白投影层 + GNN 输出的特征分布对训练/验证蛋白不可区分。模型将减少对“训练特有邻居模式”的依赖，转而强化从 ESM-2 特征和通路特征中提取的、在双域间共有的不变信号。

2. **解码器泛化**：当验证蛋白嵌入与训练蛋白嵌入分布对齐后，MLP/Bilinear 解码器对冷启动蛋白的打分更可靠，降低随机性，从而提升 `prot_aupr`。

3. **防止过拟合到训练蛋白子图结构**：现有模型可能过度记忆训练蛋白的局部拓扑；对抗训练相当于对蛋白嵌入施加一种“域不变正则”，抑制这种过拟合。

4. **与现有 v23 机制的协同**：
   - 与 `use_pathway=False` 的蛋白冷启动验证兼容（验证时屏蔽通路，域自适应仍对齐 ESM-2 主导的特征）；
   - 与 `Memory Bank` 不冲突，只需保证 bank 只入队训练蛋白；
   - 与两阶段迁移学习、表型辅助任务互不干扰，可共存。

**预期效果**：
- 若当前 `prot_aupr` 明显低于 `val_aupr`（例如差距 > 0.05），域自适应有望缩小差距；
- 保守预期 `prot_aupr` 提升 **5%–15% 相对值**（如从 0.20 → 0.22~0.23），具体取决于验证蛋白比例和分布差异；
- 若训练蛋白与验证蛋白本身特征分布高度重叠，提升可能不显著。

---

## 6. 代码改动点与集成难度估算

### 6.1 改动文件与行号

| 文件 | 改动项 | 难度 |
|------|--------|------|
| `phase4_v10_minibatch.py` | 新增 `GradientReversalFunction`、`ProteinDomainDiscriminator`、`compute_domain_adversarial_loss` | 低 |
| `phase4_v10_minibatch.py` | 修改 `sample_homo_subgraph` 增加 `seed_proteins` | 低 |
| `phase4_v10_minibatch.py` | 修改 `train_sage`：接收 DA 参数、初始化判别器、更新 optimizer、在 `_train_one_epoch` 中加入域损失 | 中 |
| `phase4_v10_minibatch.py` | 修改 `train_hgt`：类似 SAGE，复用 `sample_hetero_subgraph(seed_proteins=...)` | 中 |
| `phase4_v10_minibatch.py` | 日志输出增加 `domain_loss` 与 `lambda_adv` | 低 |
| `main()` / 调用方 | 传入 `use_domain_adapt=True`、`da_init_epoch`、`da_lambda_min`、`da_lambda_max` 等参数 | 低 |

### 6.2 关键集成细节

1. **优化器参数聚合**  
   判别器必须与编码器一起优化：
   ```python
   optimizer = torch.optim.AdamW(
       list(model.parameters()) + list(discriminator.parameters()),
       lr=lr, weight_decay=1e-4)
   ```

2. **梯度裁剪范围**  
   `clip_grad_norm_(model.parameters() + discriminator.parameters(), 1.0)`；若分开则两者都裁剪。

3. **验证蛋白隔离**  
   严禁在 `homo_adj_train` / `hetero_adj_train` 中保留任何验证蛋白边；`seed_proteins` 只能作为孤立节点传入。

4. **Memory Bank 过滤**  
   保持现有逻辑：bank 只入队训练蛋白。域自适应蛋白可额外缓存，但非必需。

5. **早停指标**  
   仍使用 `prot_aupr` 作为早停指标；域损失仅用于训练，不参与验证指标计算。

6. **HGT 与 SAGE 是否共享判别器**  
   建议每个分支独立一个 `ProteinDomainDiscriminator`，因为两个分支的蛋白嵌入空间不同（SAGE 用 MLP decoder，HGT 用 Bilinear decoder）。

### 6.3 集成难度

- **代码量**：约 80–120 行新增/修改代码；
- **逻辑复杂度**：中；最大风险是验证蛋白边泄露，必须严格保持“孤立节点”策略；
- **调参成本**：中；主要调 `lambda_max`、`da_init_epoch` 和判别器隐藏层大小；
- **训练开销**：每个 batch 增加一次判别器前向/反向，GPU 开销增加约 **3%–8%**。

---

## 7. 风险与缓解

| 风险 | 原因 | 缓解措施 |
|------|------|----------|
| 验证蛋白信息泄露 | 把验证蛋白当作正常节点采样会引入 PPI/通路边，破坏冷启动 | 强制 `seed_proteins` 孤立化；采样前检查 `hetero_adj_train` 不含 val 蛋白边 |
| 域损失淹没 CPI 损失 | `lambda_max` 过大 | 从 0.01 warm-up 到 0.1；监控 `domain_loss` 与 `cpi_loss` 比例 |
| 判别器过强导致特征退化 | 判别器把两类蛋白直接推平 | 使用较窄隐藏层、Dropout；必要时对域损失做梯度惩罚 |
| 蛋白冷启动提升有限 | 训练/验证蛋白特征分布本身接近 | 作为消融实验，与 `use_domain_adapt=False` 基线对比 |

---

## 8. 下一步建议

1. 先实现 **HGT 分支** 的域自适应（`sample_hetero_subgraph` 已支持 `seed_proteins`，改动最小），跑一组消融验证 `prot_aupr` 是否提升；
2. 再迁移到 **SAGE 分支**；
3. 若效果稳定，可进一步尝试 CDAN-E：将 CPI 预测分数（softmax/predicted probability）与蛋白特征做外积后输入判别器，但实现复杂度更高，建议在基础 DANN 蛋白判别器验证有效后再升级。
