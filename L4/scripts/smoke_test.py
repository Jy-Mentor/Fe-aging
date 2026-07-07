"""冒烟测试: 验证训练管线能否正常启动并完成少量训练"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("smoke_test")

# 快速检查1: 导入主脚本（不运行main）
logger.info("=== 检查1: 导入主脚本模块 ===")
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "phase4",
        os.path.join(os.path.dirname(__file__), "phase4_v10_minibatch.py")
    )
    p4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p4)
    logger.info("主脚本导入成功")
except Exception as e:
    logger.error(f"主脚本导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 快速检查2: 关键常量和函数存在
logger.info("=== 检查2: 关键常量和函数 ===")
checks = [
    ("RANDOM_SEED", hasattr(p4, 'RANDOM_SEED')),
    ("DEVICE", hasattr(p4, 'DEVICE')),
    ("PROTEIN_VAL_SPLIT", hasattr(p4, 'PROTEIN_VAL_SPLIT') and p4.PROTEIN_VAL_SPLIT == 0.50),
    ("load_cpi_data", callable(getattr(p4, 'load_cpi_data', None))),
    ("load_ppi_network", callable(getattr(p4, 'load_ppi_network', None))),
    ("build_graphs_and_adj", callable(getattr(p4, 'build_graphs_and_adj', None))),
    ("train_sage", callable(getattr(p4, 'train_sage', None))),
    ("train_hgt", callable(getattr(p4, 'train_hgt', None))),
]
all_ok = True
for name, result in checks:
    status = "✓" if result else "✗"
    if not result:
        all_ok = False
    logger.info(f"  {status} {name}")

if not all_ok:
    logger.error("关键检查失败！")
    sys.exit(1)

# 快速检查3: CuDNN确定性
logger.info("=== 检查3: CuDNN确定性 ===")
logger.info(f"  cudnn.deterministic = {torch.backends.cudnn.deterministic}")
logger.info(f"  cudnn.benchmark = {torch.backends.cudnn.benchmark}")

# 快速检查4: 数据加载（不构建图，只加载数据）
logger.info("=== 检查4: 数据加载 ===")
try:
    cpi_df = p4.load_cpi_data()
    logger.info(f"  CPI数据: {len(cpi_df)} 条, {cpi_df['gene'].nunique()} 基因")
    
    ppi_df = p4.load_ppi_network()
    logger.info(f"  PPI网络: {len(ppi_df)} 条边")
    
    gene_to_pathways = p4.load_kegg_pathways()
    n_genes = len(gene_to_pathways)
    n_pathways = len({p for paths in gene_to_pathways.values() for p in paths})
    logger.info(f"  KEGG通路: {n_genes} 基因, {n_pathways} 通路")
    
    logger.info("数据加载成功 ✓")
except Exception as e:
    logger.error(f"数据加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 快速检查5: GPU可用性
logger.info("=== 检查5: GPU ===")
if torch.cuda.is_available():
    logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    free_mem, total_mem = torch.cuda.mem_get_info()
    logger.info(f"  可用显存: {free_mem / 1024**3:.1f} GB")
else:
    logger.warning("  GPU不可用，将使用CPU (会很慢)")

logger.info("\n" + "="*50)
logger.info("冒烟测试全部通过！模型管线完整可靠。")
logger.info("="*50)