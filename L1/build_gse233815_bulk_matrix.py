"""
模块: L1/build_gse233815_bulk_matrix.py
功能: 从 GSE233815_RAW.tar 解压的 48 个 featureCounts 文件构建:
  1. L1/results/GSE233815_bulk_expression_matrix.csv  (gene x sample 整数 count 矩阵)
  2. data/external/GSE233815/bulk/sample_metadata.csv  (sample_id, gsm, condition, timepoint, replicate)

数据来源 (真实文件, 不模拟):
  data/external/GSE233815/bulk/GSM7437165-7437212_*_union_name.count.txt.gz
  每个文件格式 (无表头, tab 分隔):
    ENSMUSG00000000001.4    15
    ENSMUSG00000000003.15   0

文件名命名规则:
  GSM<id>_<condition>_<timepoint>[_SH]_<replicate>_union_name.count.txt.gz
  其中 condition ∈ {MCAO, Sham(隐含在 SH 标记)}, timepoint ∈ {3h, 12h, 24h, 3D, 7D, Ctrl}
  SH 标记代表 Sham 对照组 (单独手术不阻塞血管), 无 SH 标记代表 MCAO 模型组

运行:
  python L1/build_gse233815_bulk_matrix.py
"""

import gzip
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BULK_DIR = PROJECT_ROOT / "data" / "external" / "GSE233815" / "bulk"
OUTPUT_MATRIX = PROJECT_ROOT / "L1" / "results" / "GSE233815_bulk_expression_matrix.csv"
OUTPUT_META = BULK_DIR / "sample_metadata.csv"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"build_gse233815_bulk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# 文件名解析正则
# 示例: GSM7437165_MCAO_12h_A_union_name.count.txt.gz
#       GSM7437170_MCAO_12h_SH_A_union_name.count.txt.gz
#       GSM7437209_MCAO_Ctrl_A_union_name.count.txt.gz
FILENAME_RE = re.compile(
    r"^(GSM\d+)_"
    r"(?P<condition>MCAO|Sham)_"
    r"(?P<timepoint>3h|12h|24h|3D|7D|Ctrl)_"
    r"(?P<sham_marker>SH_)?"  # 可选的 SH_ 标记
    r"(?P<replicate>[A-E])_"
    r"union_name\.count\.txt\.gz$"
)


def parse_filename(filename: str) -> dict:
    """解析 bulk 文件名, 返回样本元数据."""
    m = FILENAME_RE.match(filename)
    if m is None:
        raise ValueError(f"无法解析文件名: {filename}")
    gsm = m.group(1)
    condition = m.group("condition")
    timepoint = m.group("timepoint")
    has_sham_marker = m.group("sham_marker") is not None
    replicate = m.group("replicate")

    # SH 标记代表 Sham 手术对照 (单独手术不阻塞), 无 SH 标记为 MCAO
    # 但是文件名已经显式标注了 condition=MCAO, 所以 SH 是 MCAO 组的 Sham 处理亚组
    # 实际上根据 GSE233815 原文, MCAO 组是 tMCAO (transient MCAO), SH 是 sham-operated 对照
    # 所以样本的实际分组是:
    #   - condition="MCAO", timepoint=... 且无 SH 标记 → tMCAO 模型组
    #   - condition="MCAO", timepoint=... 且有 SH 标记 → Sham 手术对照组
    # 但 timepoint=Ctrl 时也是对照组 (无 SH 标记, 因为就是 baseline control)
    if has_sham_marker:
        surgery = "Sham"
    elif timepoint == "Ctrl":
        surgery = "Ctrl"
    else:
        surgery = "MCAO"

    sample_id = f"{timepoint}_{surgery}_{replicate}"
    return {
        "sample_id": sample_id,
        "gsm": gsm,
        "surgery": surgery,
        "timepoint": timepoint,
        "replicate": replicate,
        "filename": filename,
    }


def read_count_file(filepath: Path) -> pd.Series:
    """读取单个 featureCounts 文件, 返回 gene_id → count 的 Series."""
    counts = {}
    with gzip.open(filepath, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"文件 {filepath.name} 行格式错误 (期望 2 列): {line[:80]}"
                )
            gene_id, count_str = parts
            try:
                count = int(count_str)
            except ValueError as e:
                raise ValueError(
                    f"文件 {filepath.name} 基因 {gene_id} 的 count 值非整数: {count_str}"
                ) from e
            counts[gene_id] = count
    return pd.Series(counts, name=filepath.stem)


def main() -> None:
    logger.info("=" * 60)
    logger.info("构建 GSE233815 bulk 表达矩阵与样本元数据")
    logger.info("=" * 60)
    logger.info("数据目录: %s", BULK_DIR)
    logger.info("输出矩阵: %s", OUTPUT_MATRIX)
    logger.info("输出元数据: %s", OUTPUT_META)

    if not BULK_DIR.exists():
        raise FileNotFoundError(
            f"Bulk 数据目录不存在: {BULK_DIR}. 请先运行 scripts/extract_gse233815.py 解压 RAW.tar"
        )

    # 1. 列出所有 count 文件
    count_files = sorted(BULK_DIR.glob("*_union_name.count.txt.gz"))
    logger.info("找到 %d 个 featureCounts 文件", len(count_files))
    if len(count_files) == 0:
        raise FileNotFoundError(
            f"未在 {BULK_DIR} 找到任何 *_union_name.count.txt.gz 文件"
        )
    if len(count_files) != 48:
        logger.warning(
            "预期 48 个 bulk 文件, 实际 %d 个. 仍继续处理.", len(count_files)
        )

    # 2. 解析文件名 → 元数据
    metadata_records = []
    for fp in count_files:
        meta = parse_filename(fp.name)
        meta["filepath"] = str(fp)
        metadata_records.append(meta)
    meta_df = pd.DataFrame(metadata_records)
    logger.info("样本元数据预览:\n%s", meta_df.head(8).to_string(index=False))

    # 3. 读取所有 count 文件, 合并为基因 × 样本矩阵
    logger.info("读取 %d 个 count 文件并合并矩阵...", len(count_files))
    all_series = []
    sample_columns = []
    for fp in count_files:
        meta = next(m for m in metadata_records if m["filename"] == fp.name)
        sample_id = meta["sample_id"]
        logger.info("  读取 %s → %s", fp.name, sample_id)
        series = read_count_file(fp)
        series.name = sample_id
        all_series.append(series)
        sample_columns.append(sample_id)

    # 检查样本 ID 唯一性
    if len(set(sample_columns)) != len(sample_columns):
        duplicates = [s for s in sample_columns if sample_columns.count(s) > 1]
        raise ValueError(f"样本 ID 重复: {set(duplicates)}")

    logger.info("合并 %d 个样本矩阵...", len(all_series))
    expr_matrix = pd.concat(all_series, axis=1)
    expr_matrix.index.name = "gene_id"
    logger.info(
        "合并完成: %d 基因 x %d 样本", expr_matrix.shape[0], expr_matrix.shape[1]
    )

    # 4. 检查缺失值 (featureCounts 不同文件可能基因集略有差异)
    n_missing = expr_matrix.isna().sum().sum()
    if n_missing > 0:
        logger.warning(
            "矩阵中有 %d 个缺失值 (基因未在某些样本中出现), 填充为 0", n_missing
        )
        expr_matrix = expr_matrix.fillna(0).astype(int)
    else:
        logger.info("无缺失值")

    # 5. 基本统计验证
    logger.info("矩阵统计:")
    logger.info("  总 count: %s", f"{int(expr_matrix.values.sum()):,}")
    logger.info("  每样本 count 范围: [%d, %d]",
                int(expr_matrix.sum(axis=0).min()),
                int(expr_matrix.sum(axis=0).max()))
    logger.info("  每基因 count 范围: [%d, %d]",
                int(expr_matrix.sum(axis=1).min()),
                int(expr_matrix.sum(axis=1).max()))

    # 6. 写入表达矩阵
    OUTPUT_MATRIX.parent.mkdir(parents=True, exist_ok=True)
    logger.info("写入表达矩阵到 %s", OUTPUT_MATRIX)
    expr_matrix.to_csv(OUTPUT_MATRIX)
    logger.info("矩阵文件大小: %.2f MB", OUTPUT_MATRIX.stat().st_size / 1024 / 1024)

    # 7. 写入样本元数据
    meta_out = meta_df[["sample_id", "gsm", "surgery", "timepoint", "replicate"]].copy()
    # 添加 DESeq2 需要的列名: timepoint (时间点因子), condition (手术分组)
    meta_out = meta_out.rename(columns={"surgery": "condition"})
    # 按 timepoint + condition 排序
    timepoint_order = ["Ctrl", "3h", "12h", "24h", "3D", "7D"]
    meta_out["timepoint"] = pd.Categorical(
        meta_out["timepoint"], categories=timepoint_order, ordered=True
    )
    meta_out = meta_out.sort_values(["timepoint", "condition", "replicate"]).reset_index(
        drop=True
    )
    logger.info("写入元数据到 %s", OUTPUT_META)
    meta_out.to_csv(OUTPUT_META, index=False)
    logger.info("样本分组统计:\n%s",
                meta_out.groupby(["timepoint", "condition"], observed=False).size().to_string())

    logger.info("=" * 60)
    logger.info("完成. 输出文件:")
    logger.info("  矩阵: %s (%d 基因 x %d 样本)",
                OUTPUT_MATRIX, expr_matrix.shape[0], expr_matrix.shape[1])
    logger.info("  元数据: %s (%d 样本)", OUTPUT_META, len(meta_out))
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("构建失败: %s", e, exc_info=True)
        sys.exit(1)
