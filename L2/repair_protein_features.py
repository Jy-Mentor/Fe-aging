#!/usr/bin/env python3
"""
修复 L2 蛋白特征表中 domain / PTM / transmembrane / signal peptide 计数。

问题：原 parse_uniprot_features 使用全大写 feature type 与 UniProt REST API
返回的 Title Case 不匹配，导致上述字段全为 0。

本脚本读取现有 target_protein_features.csv，重新获取每个 UniProt ID 的 JSON，
使用修复后的 parse_uniprot_features 解析，并更新相关字段。

运行：
    python L2/repair_protein_features.py
输出：
    L2/results/target_protein_features.csv  (覆盖更新)
    L2/results/target_protein_features_repair_report.csv
    logs/repair_protein_features.log
"""

import sys
import time
import logging
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(Path('logs') / 'repair_protein_features.log', mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent
L2_RESULTS = BASE / 'L2' / 'results'
PROT_FILE = L2_RESULTS / 'target_protein_features.csv'
REPORT_FILE = L2_RESULTS / 'target_protein_features_repair_report.csv'


def fetch_uniprot_annotations(uniprot_id, max_retries=3):
    """Fetch UniProt annotations via REST API with retries."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.warning("  UniProt annotations API returned %d for %s (attempt %d/%d)",
                               resp.status_code, uniprot_id, attempt, max_retries)
        except Exception as e:
            logger.warning("  Failed to fetch annotations for %s (attempt %d/%d): %s",
                           uniprot_id, attempt, max_retries, e)
        time.sleep(2 * attempt)
    return None


def parse_uniprot_features(uniprot_id, data):
    """解析 UniProt JSON 提取关键特征（与 L2/protein_features.py 同步逻辑）。"""
    features = {
        'uniprot_id': uniprot_id,
        'protein_name': '',
        'gene_name': '',
        'length': 0,
        'mass': 0,
        'n_domains': 0,
        'n_ptms': 0,
        'n_phospho': 0,
        'n_ubiquitination': 0,
        'n_acetylation': 0,
        'subcellular_main': '',
        'has_signal_peptide': False,
        'has_transmembrane': False,
        'n_transmembrane': 0,
        'reviewed': False,
    }

    if data is None:
        return features

    try:
        features['protein_name'] = data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', '')
        features['gene_name'] = data.get('genes', [{}])[0].get('geneName', {}).get('value', '')
        features['length'] = data.get('sequence', {}).get('length', 0)
        features['mass'] = data.get('sequence', {}).get('molWeight', 0)
        features['reviewed'] = data.get('entryType', '') == 'UniProtKB reviewed (Swiss-Prot)'

        comments = data.get('comments', [])
        for comment in comments:
            if comment.get('commentType', '').upper() == 'SUBCELLULAR LOCATION':
                locations = comment.get('subcellularLocations', [])
                if locations:
                    features['subcellular_main'] = locations[0].get('location', {}).get('value', '')

        feat_list = data.get('features', [])
        for feat in feat_list:
            ftype = feat.get('type', '').lower()
            if ftype in ('domain', 'zinc finger', 'repeat'):
                features['n_domains'] += 1
            elif ftype in ('mod_res', 'crosslnk', 'modified residue', 'cross-link',
                           'glycosylation', 'disulfide bond', 'lipidation',
                           'propeptide', 'initiator methionine'):
                features['n_ptms'] += 1
                desc = feat.get('description', '')
                if 'phospho' in desc.lower():
                    features['n_phospho'] += 1
                elif 'ubiquitin' in desc.lower():
                    features['n_ubiquitination'] += 1
                elif 'acetyl' in desc.lower():
                    features['n_acetylation'] += 1
            elif ftype in ('signal', 'signal peptide'):
                features['has_signal_peptide'] = True
            elif ftype in ('transmem', 'transmembrane'):
                features['has_transmembrane'] = True
                features['n_transmembrane'] += 1

    except Exception as e:
        logger.warning("  Error parsing UniProt features for %s: %s", uniprot_id, e)

    return features


def main():
    logger.info("=" * 60)
    logger.info("修复蛋白特征表")
    logger.info("=" * 60)

    if not PROT_FILE.exists():
        logger.error("蛋白特征文件不存在: %s", PROT_FILE)
        sys.exit(1)

    df = pd.read_csv(PROT_FILE)
    logger.info("读取蛋白特征表: %d 行", len(df))

    update_cols = ['n_domains', 'n_ptms', 'n_phospho', 'n_ubiquitination',
                   'n_acetylation', 'has_signal_peptide', 'has_transmembrane',
                   'n_transmembrane', 'subcellular_main', 'protein_name',
                   'gene_name', 'length', 'mass', 'reviewed']

    report = []

    for idx, row in df.iterrows():
        uniprot_id = str(row.get('uniprot_id', '')).strip()
        gene = str(row.get('gene_symbol', '')).strip()
        if not uniprot_id or uniprot_id.lower() == 'nan':
            logger.warning("第 %d 行缺少 uniprot_id，跳过", idx)
            continue

        logger.info("[%d/%d] 处理 %s (%s)...", idx + 1, len(df), gene, uniprot_id)
        old = {c: row.get(c) for c in update_cols}

        data = fetch_uniprot_annotations(uniprot_id)
        if data is None:
            logger.warning("  无法获取 %s 的 annotations，跳过", uniprot_id)
            continue

        parsed = parse_uniprot_features(uniprot_id, data)
        for c in update_cols:
            if c in parsed:
                df.at[idx, c] = parsed[c]

        new = {c: df.at[idx, c] for c in update_cols}
        report.append({
            'gene_symbol': gene,
            'uniprot_id': uniprot_id,
            'old_n_domains': old['n_domains'],
            'new_n_domains': new['n_domains'],
            'old_n_ptms': old['n_ptms'],
            'new_n_ptms': new['n_ptms'],
            'old_n_transmembrane': old['n_transmembrane'],
            'new_n_transmembrane': new['n_transmembrane'],
            'old_has_signal_peptide': old['has_signal_peptide'],
            'new_has_signal_peptide': new['has_signal_peptide'],
        })

        time.sleep(0.5)

    df.to_csv(PROT_FILE, index=False)
    logger.info("已更新: %s", PROT_FILE)

    report_df = pd.DataFrame(report)
    report_df.to_csv(REPORT_FILE, index=False)
    logger.info("已生成修复报告: %s", REPORT_FILE)

    # 汇总
    logger.info("修复汇总:")
    logger.info("  总蛋白数: %d", len(df))
    logger.info("  平均 n_domains: %.2f", df['n_domains'].mean())
    logger.info("  平均 n_ptms: %.2f", df['n_ptms'].mean())
    logger.info("  平均 n_transmembrane: %.2f", df['n_transmembrane'].mean())
    logger.info("  有 signal peptide: %d", df['has_signal_peptide'].sum())
    logger.info("  有 transmembrane: %d", df['has_transmembrane'].sum())


if __name__ == '__main__':
    main()
