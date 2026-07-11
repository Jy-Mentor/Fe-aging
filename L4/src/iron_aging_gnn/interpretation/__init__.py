"""生物学解释模块：提供通路富集、铁衰老交集分析和可解释链条输出。"""

from .explanation_pipeline import explain_predictions
from .iron_aging_overlap import (
    compute_iron_aging_overlap,
    load_ferroptosis_gene_sets,
    load_iron_aging_genes,
)
from .pathway_analysis import (
    load_kegg_pathway_genes,
    load_pathway_annotations,
    pathway_enrichment,
)

__all__ = [
    "load_iron_aging_genes",
    "load_ferroptosis_gene_sets",
    "compute_iron_aging_overlap",
    "load_kegg_pathway_genes",
    "load_pathway_annotations",
    "pathway_enrichment",
    "explain_predictions",
]