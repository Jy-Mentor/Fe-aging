"""
下载两个基因集数据集:
1. FerrDb V3 铁死亡驱动基因 (FerrDb V3 driver genes)
2. CellAge 细胞衰老标志基因 (CellAge cellular senescence genes)
3. GenAge 人类衰老基因 (GenAge human aging genes) - 补充

数据来源:
- FerrDb V3: https://www.zhounan.org/ferrdb/ (Zhou N, et al. Nucleic Acids Res, 2026)
- CellAge: https://hagr.ageing-map.org/cells/cellAge.zip (Avelar RA, et al. Genome Biol, 2020)
- GenAge: https://hagr.ageing-map.org/genes/human_genes.zip (de Magalhães JP, et al. Nucleic Acids Res, 2024)
"""

import logging
import os
import sys
import traceback
from datetime import datetime

import requests
import zipfile
import io
import csv

# ============================================================
# 日志配置
# ============================================================
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "download_gene_datasets.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "L1", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. FerrDb V3 铁死亡驱动基因
# ============================================================
def download_ferrdb_drivers():
    """
    从 FerrDb V3 下载铁死亡驱动基因。
    FerrDb V3 扩展至 22 种 RCD 模式，包含铁死亡驱动基因。
    """
    logger.info("=" * 60)
    logger.info("下载 FerrDb V3 铁死亡驱动基因")
    logger.info("=" * 60)

    v3_url = "https://www.zhounan.org/ferrdb/current/extdownload/ferroptosis_early_preview_upto20231231.zip"
    output_file = os.path.join(OUTPUT_DIR, "ferrdb_drivers.csv")
    
    try:
        logger.info("正在下载 FerrDb V3: %s", v3_url)
        resp = requests.get(v3_url, timeout=120, verify=False)
        resp.raise_for_status()
        logger.info("下载成功, 大小: %.2f MB", len(resp.content) / 1024 / 1024)
        
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            file_list = [f for f in z.namelist() if not f.startswith('__MACOSX')]
            logger.info("文件列表: %s", file_list)
            
            # 读取 driver.csv
            driver_files = [f for f in file_list if 'driver' in f.lower() and f.endswith('.csv')]
            if not driver_files:
                logger.error("未找到 driver CSV 文件")
                return None
            
            logger.info("读取驱动基因文件: %s", driver_files[0])
            with z.open(driver_files[0]) as f:
                raw = f.read().decode('utf-8-sig', errors='replace')
            
            # 检查分隔符和列名
            logger.info("前 3 行原始内容:")
            for i, line in enumerate(raw.strip().split('\n')[:3]):
                logger.info("  行 %d: %s", i + 1, line)
            
            # 尝试 ',' 分隔
            reader = csv.DictReader(io.StringIO(raw))
            rows = list(reader)
            
            if not rows or len(rows[0].keys()) <= 1:
                # 尝试 ';' 分隔
                reader = csv.DictReader(io.StringIO(raw), delimiter=';')
                rows = list(reader)
            
            logger.info("总行数: %d, 列: %s", len(rows), list(rows[0].keys()) if rows else [])
            
            if not rows:
                logger.error("无法解析 CSV")
                return None
            
            # 查找基因符号列
            first_row = rows[0]
            gene_col = None
            for col in first_row.keys():
                col_clean = col.strip().strip('"').lower().replace('_', '')
                if col_clean in ['symbol', 'gene', 'genesymbol', 'genename', 'hgncsymbol',
                                'symbolorreportedabbr']:
                    gene_col = col
                    break
            
            if not gene_col:
                gene_col = list(first_row.keys())[0]
                logger.warning("使用第一列作为基因列: %s", gene_col)
            
            logger.info("基因列: %s", gene_col)
            
            # 提取基因 (只提取 RCD 为 Ferroptosis 的条目)
            genes = set()
            for row in rows:
                rcd = row.get('RCD', '').strip()
                if rcd != 'Ferroptosis':
                    continue
                val = row.get(gene_col, '').strip().strip('"').strip()
                if val and val == val.upper() and len(val) > 1:
                    # 标准基因符号全大写，且长度 > 1
                    genes.add(val)
            
            genes = sorted(genes)
            logger.info("FerrDb V3 驱动基因 (去重后): %d 个", len(genes))
            logger.info("前 20 个: %s", ", ".join(genes[:20]))
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("gene_symbol,source,category\n")
                for g in genes:
                    f.write(f"{g},FerrDb_V3,driver\n")
            
            logger.info("已保存: %s (%d 个基因)", output_file, len(genes))
            
            # 同时保存完整原始数据
            raw_file = os.path.join(OUTPUT_DIR, "ferrdb_driver_raw.csv")
            with open(raw_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            logger.info("原始数据已保存: %s (%d 行)", raw_file, len(rows))
            
            return genes
            
    except Exception as e:
        logger.error("FerrDb V3 下载失败: %s", str(e))
        logger.error(traceback.format_exc())
        return None


# ============================================================
# 2. CellAge 细胞衰老标志基因
# ============================================================
def download_cellage_genes():
    """
    从 HAGR 下载 CellAge 数据库（细胞衰老基因）。
    来源: Avelar RA, et al. Genome Biol, 2020
    下载: https://hagr.ageing-map.org/cells/cellAge.zip
    """
    logger.info("=" * 60)
    logger.info("下载 CellAge 细胞衰老标志基因")
    logger.info("=" * 60)
    
    cellage_url = "https://hagr.ageing-map.org/cells/cellAge.zip"
    output_file = os.path.join(OUTPUT_DIR, "cellage_senescence_genes.csv")
    
    try:
        logger.info("正在下载 CellAge: %s", cellage_url)
        resp = requests.get(cellage_url, timeout=120, verify=False)
        resp.raise_for_status()
        logger.info("下载成功, 大小: %.2f KB", len(resp.content) / 1024)
        
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            file_list = z.namelist()
            logger.info("ZIP 文件: %s", file_list)
            
            csv_files = [f for f in file_list if f.endswith('.csv')]
            if not csv_files:
                logger.error("未找到 CSV 文件")
                return None
            
            logger.info("读取: %s", csv_files[0])
            with z.open(csv_files[0]) as f:
                raw = f.read().decode('utf-8-sig', errors='replace')
            
            # CellAge 使用 ';' 分隔符，列名带引号
            logger.info("前 3 行原始内容:")
            for i, line in enumerate(raw.strip().split('\n')[:3]):
                logger.info("  行 %d: %s", i + 1, line)
            
            # 手动解析：分隔符是 ';'，列名带引号
            lines = raw.strip().split('\n')
            header = [h.strip().strip('"') for h in lines[0].split(';')]
            logger.info("列名: %s", header)
            
            # 查找基因名列
            gene_col_idx = None
            for i, h in enumerate(header):
                h_lower = h.strip().lower()
                if h_lower in ['gene_name', 'symbol', 'gene', 'gene_symbol', 'hgnc_symbol']:
                    gene_col_idx = i
                    break
            
            if gene_col_idx is None:
                logger.error("无法找到基因名列, 可用列: %s", header)
                return None
            
            logger.info("基因列 (索引 %d): %s", gene_col_idx, header[gene_col_idx])
            
            # 解析数据行
            all_genes = set()
            parsed_rows = []
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                # 分割，注意引号内的内容
                parts = []
                current = []
                in_quotes = False
                for ch in line:
                    if ch == '"':
                        in_quotes = not in_quotes
                    elif ch == ';' and not in_quotes:
                        parts.append(''.join(current).strip().strip('"'))
                        current = []
                    else:
                        current.append(ch)
                parts.append(''.join(current).strip().strip('"'))
                
                if len(parts) >= gene_col_idx + 1:
                    gene = parts[gene_col_idx].strip().upper()
                    if gene:
                        all_genes.add(gene)
                        parsed_rows.append(parts)
            
            genes = sorted(all_genes)
            logger.info("CellAge 唯一基因数: %d", len(genes))
            logger.info("前 20 个: %s", ", ".join(genes[:20]))
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("gene_symbol,source\n")
                for g in genes:
                    f.write(f"{g},CellAge\n")
            
            logger.info("已保存: %s (%d 个基因)", output_file, len(genes))
            
            # 保存详细数据
            detail_file = os.path.join(OUTPUT_DIR, "cellage_detailed.csv")
            with open(detail_file, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(parsed_rows)
            logger.info("详细数据已保存: %s (%d 行)", detail_file, len(parsed_rows))
            
            return genes
            
    except Exception as e:
        logger.error("CellAge 下载失败: %s", str(e))
        logger.error(traceback.format_exc())
        return None


# ============================================================
# 3. GenAge 人类衰老相关基因
# ============================================================
def download_genage_genes():
    """
    从 HAGR 下载 GenAge 人类衰老基因。
    来源: de Magalhães JP, et al. Nucleic Acids Res, 2024
    下载: https://hagr.ageing-map.org/genes/human_genes.zip
    """
    logger.info("=" * 60)
    logger.info("下载 GenAge 人类衰老基因")
    logger.info("=" * 60)
    
    genage_url = "https://hagr.ageing-map.org/genes/human_genes.zip"
    output_file = os.path.join(OUTPUT_DIR, "genage_aging_genes.csv")
    
    try:
        logger.info("正在下载 GenAge: %s", genage_url)
        resp = requests.get(genage_url, timeout=120, verify=False)
        resp.raise_for_status()
        logger.info("下载成功, 大小: %.2f KB", len(resp.content) / 1024)
        
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            file_list = z.namelist()
            logger.info("ZIP 文件: %s", file_list)
            
            # GenAge 的人类基因文件
            csv_files = [f for f in file_list if f.endswith('.csv') and 'human' in f.lower()]
            if not csv_files:
                csv_files = [f for f in file_list if f.endswith('.csv')]
            
            if not csv_files:
                logger.error("未找到 CSV 文件")
                return None
            
            logger.info("读取: %s", csv_files[0])
            with z.open(csv_files[0]) as f:
                raw = f.read().decode('utf-8-sig', errors='replace')
            
            logger.info("前 3 行:")
            for i, line in enumerate(raw.strip().split('\n')[:3]):
                logger.info("  行 %d: %s", i + 1, line)
            
            # 尝试 ',' 分隔
            reader = csv.DictReader(io.StringIO(raw))
            rows = list(reader)
            
            if not rows:
                # 尝试 ';' 分隔
                reader = csv.DictReader(io.StringIO(raw), delimiter=';')
                rows = list(reader)
            
            if not rows or len(rows) < 5:
                # 尝试 tab 分隔
                reader = csv.DictReader(io.StringIO(raw), delimiter='\t')
                rows = list(reader)
            
            logger.info("GenAge 条目: %d, 列: %s", len(rows), list(rows[0].keys()) if rows else [])
            
            if rows:
                # 查找基因列
                gene_col = None
                for col in rows[0].keys():
                    col_clean = col.strip().strip('"').lower()
                    if col_clean in ['symbol', 'gene', 'gene_symbol', 'gene_name', 'hgnc_symbol', 'genesymbol']:
                        gene_col = col
                        break
                
                if not gene_col:
                    gene_col = list(rows[0].keys())[0]
                    logger.warning("使用第一列作为基因列: %s", gene_col)
                
                genes = sorted(set(
                    row[gene_col].strip().strip('"').upper() for row in rows
                    if row[gene_col].strip().strip('"')
                ))
                logger.info("GenAge 人类衰老基因: %d 个", len(genes))
                logger.info("前 20 个: %s", ", ".join(genes[:20]))
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write("gene_symbol,source\n")
                    for g in genes:
                        f.write(f"{g},GenAge\n")
                
                logger.info("已保存: %s (%d 个基因)", output_file, len(genes))
                return genes
            
    except Exception as e:
        logger.warning("GenAge 下载失败 (非关键): %s", str(e))
    
    return None


# ============================================================
# 主函数
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("下载基因集数据集")
    logger.info("开始时间: %s", datetime.now().isoformat())
    logger.info("=" * 60)
    
    results = {}
    
    # 1. FerrDb 驱动基因
    logger.info("\n")
    ferrdb_genes = download_ferrdb_drivers()
    if ferrdb_genes:
        results['ferrdb_drivers (FerrDb V3)'] = len(ferrdb_genes)
    
    # 2. CellAge 衰老基因
    logger.info("\n")
    cellage_genes = download_cellage_genes()
    if cellage_genes:
        results['cellage_senescence (CellAge)'] = len(cellage_genes)
    
    # 3. GenAge 衰老基因
    logger.info("\n")
    genage_genes = download_genage_genes()
    if genage_genes:
        results['genage_aging (GenAge)'] = len(genage_genes)
    
    # 汇总
    logger.info("\n" + "=" * 60)
    logger.info("下载完成汇总")
    logger.info("=" * 60)
    for name, count in results.items():
        logger.info("  %s: %d 个基因", name, count)
    
    if cellage_genes and ferrdb_genes:
        overlap = set(ferrdb_genes) & set(cellage_genes)
        logger.info("FerrDb 驱动基因 ∩ CellAge 衰老基因: %d 个重叠", len(overlap))
        if overlap:
            logger.info("  重叠基因: %s", ", ".join(sorted(overlap)))
    
    if cellage_genes and results.get('genage_aging (GenAge)'):
        overlap2 = set(cellage_genes) & set(genage_genes)
        logger.info("CellAge ∩ GenAge: %d 个重叠", len(overlap2))
    
    # 验证输出文件
    logger.info("\n输出文件验证:")
    for fname in ['ferrdb_drivers.csv', 'ferrdb_driver_raw.csv',
                  'cellage_senescence_genes.csv', 'cellage_detailed.csv',
                  'genage_aging_genes.csv']:
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            logger.info("  ✓ %s (%.2f KB)", fname, size / 1024)
        else:
            logger.info("  - %s (未生成)", fname)
    
    logger.info("结束时间: %s", datetime.now().isoformat())
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logger.error(traceback.format_exc())
        sys.exit(1)
