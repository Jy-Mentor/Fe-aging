#!/usr/bin/env python3
"""
在标书 document.xml 中, 在 "1.6单细胞图谱与细胞通讯分析" 段落之后
插入 "1.7 单细胞拟时序与细胞类型优先级分析" 方法学段落
"""
import re
import sys
import shutil
from pathlib import Path

doc_xml_path = Path(r"D:\铁衰老 绝不重蹈覆辙\标书_unpacked\word\document.xml")
content = doc_xml_path.read_text(encoding="utf-8")

# 定位 1.7实验技术平台 段落标题 (在它之前插入新段落)
# 查找模式: 1.7实验技术平台与预实验结果
target_pattern = r'1\.7实验技术平台与预实验结果'
match = re.search(target_pattern, content)
if not match:
    print("ERROR: 找不到 '1.7实验技术平台与预实验结果' 标记", file=sys.stderr)
    sys.exit(1)

# 找到包含 1.7 的 <w:p> 段落的起始位置
# 向前查找最近的 <w:p>
pos = match.start()
# 向前搜索 <w:p
p_start = content.rfind('<w:p>', 0, pos)
if p_start == -1:
    p_start = content.rfind('<w:p ', 0, pos)
if p_start == -1:
    print("ERROR: 找不到包含 1.7 的段落起始 <w:p>", file=sys.stderr)
    sys.exit(1)

print(f"[INFO] 找到 1.7 段落位置: char {p_start}")
print(f"[INFO] 在此位置插入新的 1.7 拟时序方法学段落, 原 1.7 变为 1.8")

# 构造新的段落 XML - 拟时序方法学
# 使用与文档一致的格式 (从上下文推断)
new_section_title = (
    '<w:p>'
    '<w:pPr><w:pStyle w:val="3"/><w:spacing w:before="240" w:after="120"/>'
    '<w:ind w:firstLineChars="0" w:firstLine="0"/>'
    '<w:jc w:val="left"/></w:pPr>'
    '<w:r><w:rPr><w:rFonts w:ascii="宋体" w:eastAsia="宋体" w:hAnsi="宋体" w:hint="eastAsia"/>'
    '<w:b/><w:sz w:val="24"/></w:rPr>'
    '<w:t>1.7单细胞拟时序分析与细胞类型优先级排序</w:t></w:r>'
    '</w:p>'
)

# 方法正文段落 (参照文档已有正文风格)
method_text_1 = (
    '为解析铁衰老特征在神经元亚群中的动态演化轨迹, '
    '申请人采用monocle3 (Qiu et al., 2017, Nat Methods, PMID:28825705) '
    '对GSE233815 snRNA-seq中的神经元亚群 (NeuronsGABA与NeuronsGLUT, n=4,787) '
    '进行拟时序分析。具体流程: (1)通过SeuratWrappers::as.cell_data_set将Seurat v5对象转换为cell_data_set (CDS); '
    '(2)使用cluster_cells进行UMAP空间聚类, learn_graph学习主图轨迹; '
    '(3)以对照组 (Ctrl/sham) 神经元占多数的principal graph node为根节点 (root_pr_nodes), '
    '通过order_cells计算所有细胞沿轨迹的pseudotime; '
    '(4)提取4个铁衰老核心基因集 (Ferroptosis/Senescence/Ferrosenescence/Ferroaging) '
    '的UCell评分, 沿pseudotime绘制LOESS平滑曲线 (span=0.4) 展示动态变化趋势。'
)

new_para_1 = (
    '<w:p>'
    '<w:pPr><w:pStyle w:val="2"/><w:spacing w:line="360" w:lineRule="auto" w:before="60" w:after="60"/>'
    '<w:ind w:firstLineChars="200" w:firstLine="480"/>'
    '<w:jc w:val="both"/></w:pPr>'
    '<w:r><w:rPr><w:rFonts w:ascii="宋体" w:eastAsia="宋体" w:hAnsi="宋体" w:hint="eastAsia"/>'
    '<w:sz w:val="24"/></w:rPr>'
    f'<w:t>{method_text_1}</w:t></w:r>'
    '</w:p>'
)

method_text_2 = (
    '同时, 为定量评估各细胞类型在缺血再灌注不同时间点的转录组扰动强度, '
    '采用Augur (Skinnider et al., 2021, Nat Biotechnol, PMID:32690972; '
    'Squair et al., 2021, Nat Protoc, PMID:34172974) 进行细胞类型优先级排序。'
    'Augur核心思想: 若某细胞类型在扰动条件下转录组可分类性越高 (AUC接近1.0), '
    '则该类型受扰动越强。参数设置: 随机森林分类器 (rf), n_subsamples=50 (官方默认), '
    'subsample_size=20 (每次每条件抽样细胞数), folds=3, feature_perc=0.5; '
    '为保留时间点特异差异, 按时间点分层分析 (1DPI/3DPI/7DPI vs Ctrl分别独立运行), '
    '而非将所有非Ctrl条件合并为单一stim组; 每细胞类型抽样上限500细胞以控制运行时长。'
    '结果显示1DPI急性期Microglia扰动最强 (AUC=0.811), 其次为NeuronsGLUT (0.768) '
    '和NeuronsGABA (0.720), 7DPI恢复期各细胞类型AUC回落至基线附近 (0.50-0.56), '
    '符合缺血再灌注急性损伤-慢性修复的时序规律。'
)

new_para_2 = (
    '<w:p>'
    '<w:pPr><w:pStyle w:val="2"/><w:spacing w:line="360" w:lineRule="auto" w:before="60" w:after="60"/>'
    '<w:ind w:firstLineChars="200" w:firstLine="480"/>'
    '<w:jc w:val="both"/></w:pPr>'
    '<w:r><w:rPr><w:rFonts w:ascii="宋体" w:eastAsia="宋体" w:hAnsi="宋体" w:hint="eastAsia"/>'
    '<w:sz w:val="24"/></w:rPr>'
    f'<w:t>{method_text_2}</w:t></w:r>'
    '</w:p>'
)

# 更新后续编号: 1.7实验技术平台 -> 1.8, 1.8与已有文献 -> 1.9, 1.9壮瑶医药 -> 1.10
# 注意只替换作为标题的编号, 避免误伤正文中的数字
renumber_replacements = [
    ('1.7实验技术平台与预实验结果', '1.8实验技术平台与预实验结果'),
    ('1.8与已有文献的一致性与互补性', '1.9与已有文献的一致性与互补性'),
    ('1.9壮瑶医药理论研究积累', '1.10壮瑶医药理论研究积累'),
]

# 在 p_start 位置插入新段落
new_content = (
    content[:p_start] +
    new_section_title +
    new_para_1 +
    new_para_2 +
    content[p_start:]
)

# 执行编号更新
for old, new in renumber_replacements:
    if old in new_content:
        new_content = new_content.replace(old, new)
        print(f"[INFO] 编号更新: {old[:20]}... -> {new[:20]}...")
    else:
        print(f"[WARN] 未找到: {old}", file=sys.stderr)

# 写回
doc_xml_path.write_text(new_content, encoding="utf-8")
print(f"[DONE] document.xml 已更新, 新长度: {len(new_content)} (原 {len(content)})")
