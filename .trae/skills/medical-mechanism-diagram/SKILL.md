---
name: "medical-mechanism-diagram"
description: "Generate publication-quality medical mechanism diagrams, signaling pathway schematics, cell-cell communication maps, and disease mechanism illustrations. Invoke when user asks for mechanism diagrams, pathway figures, signaling cascades (e.g. Nrf2/Keap1, ferroptosis, iron metabolism, CIRI cascade), drug-target schematics, or BioRender-style scientific illustrations programmatically via R (ggplot2 + geom_curve + annotation_custom) and Python (matplotlib patches + networkx)."
---

# Medical Mechanism Diagram Skill

## 1. When to Invoke

Invoke this skill IMMEDIATELY when the user asks for any of the following:
- "机制图" / "通路图" / "信号级联" / "pathway diagram" / "mechanism figure"
- Disease mechanism schematics (CIRI cascade, ferroptosis regulation, iron aging axis)
- Signaling pathway illustrations (Nrf2/Keap1, ACSL4/LPCAT3, GPX4/SLC7A11-xCT, p53/SLC7A11)
- Cell-cell communication maps (microglia-neuron-astrocyte crosstalk, ligand-receptor)
- Drug-target interaction schematics (compound → target → downstream → phenotype)
- Multi-omics integration flowcharts (GWAS → eQTL → TWAS → MR → colocalization)
- BioRender-style scientific illustrations (cells, organelles, membranes, receptors)
- Experimental workflow diagrams (in vivo MCAO → snRNA-seq → SCISSOR → validation)
- Schematic overview figures (graphical abstracts for Nature/Cell/Science)

Do NOT invoke for: data-driven statistical plots (use `nature-skill` instead), pure network analysis without visual schematic, or 3D molecular rendering (use PyMOL/ChimeraX).

## 2. Environment Setup (MANDATORY preflight)

### Python engine (primary — for schematic diagrams)
```powershell
# Use the dedicated venv (NOT conda base — has pyparsing conflict):
& "D:\铁衰老 绝不重蹈覆辙\.plotenv\Scripts\python.exe" -c "import matplotlib, numpy, networkx; print('OK')"
```

Required packages (install if missing): `matplotlib`, `numpy`, `networkx`, `scipy`, `shapely` (for curved membranes), `matplotlib-venn` (for set overlaps in mechanisms).

### R engine (for pathway overlays on real data)
```r
stopifnot(requireNamespace("ggplot2", quietly=TRUE))
stopifnot(requireNamespace("ggrepel", quietly=TRUE))   # node labels
stopifnot(requireNamespace("igraph",   quietly=TRUE))   # pathway graph layout
stopifnot(requireNamespace("ggraph",   quietly=TRUE))   # pathway visualization
```

If `igraph`/`ggraph` missing: `install.packages(c('igraph','ggraph'), repos='https://mirrors.tuna.tsinghua.edu.cn/CRAN/')`.

## 3. Diagram Taxonomy & Standards

### 3.1 Diagram types this skill produces

| Type | Best tool | Example use case |
|---|---|---|
| Signaling cascade (linear) | matplotlib (FancyArrowPatch) | Nrf2 → HO-1 → ferritin → iron sequestration |
| Pathway network (graph) | igraph + ggraph or networkx | Compound → multiple targets → multiple pathways |
| Cellular schematic | matplotlib patches (Ellipse, FancyBboxPatch, Wedge) | Membrane receptor → cytosol kinase → nucleus TF |
| Multi-compartment | matplotlib + custom boundaries | Blood-BBB-Brain compartments with cell types |
| Cell-cell communication | networkx bipartite or circlize chord | Microglia → neuron ligand-receptor pairs |
| Workflow / flowchart | matplotlib FancyBboxPatch + arrows | MCAO model → snRNA-seq → SCISSOR → candidates |
| Graphical abstract | composite of all above | Nature/Cell graphical abstract 183×100 mm |

### 3.2 Sizing & Resolution
- **Single-panel mechanism**: 89 mm (single column) or 120 mm (1.5 column) wide
- **Multi-compartment / graphical abstract**: 183 mm (double column) wide
- **Height**: 60-120 mm typical; maintain aspect ratio for readability
- **Resolution**: 300 DPI minimum for PNG/TIFF; PDF/SVG vector preferred
- **Font**: Arial 7-9 pt for labels, 6 pt for secondary annotations

### 3.3 Color Conventions (Biomedical Standard)

| Biological element | Recommended color (hex) | Rationale |
|---|---|---|
| Cell membrane | #FFD7B5 (light tan) | Conventional phospholipid |
| Cytoplasm | #E8F4FD (light blue) | Standard cell interior |
| Nucleus | #D7BDE2 (light purple) | Conventional nuclear |
| Mitochondria | #F5B7B1 (pink-red) | Conventional mitochondria |
| Endoplasmic reticulum | #A9DFBF (light green) | Conventional ER |
| Reactive oxygen species (ROS) | #E74C3C (red) | Danger signal |
| Iron ion (Fe2+/Fe3+) | #7D6608 (dark gold) or #AAB7B8 (grey iron) | Convention |
| Lipid peroxidation products | #E67E22 (orange) | Oxidation |
| Anti-oxidant defense | #27AE60 (green) | Protection |
| Drug / compound | #3498DB (blue) | Intervention |
| Target protein | #8E44AD (purple) | Specificity |
| Disease phenotype | #C0392B (dark red) | Pathology |
| Healthy phenotype | #16A085 (teal) | Normal |

### 3.4 Arrow Conventions

| Arrow type | Meaning | matplotlib |
|---|---|---|
| Solid arrowhead → | Activation / conversion | `FancyArrowPatch(arrowstyle='-|>')` |
| Blunt line ⊣ | Inhibition | `FancyArrowPatch(arrowstyle='-|')` |
| Dashed arrow ⇢ | Indirect / unknown mechanism | `linestyle='--'` |
| Double arrow ↔ | Bidirectional / binding | `arrowstyle='<->'` |
| Thick arrow ➤ | Translocation / trafficking | `arrowstyle='-|>', mutation_scale=25, linewidth=2.5` |
| Curved arrow | Regulation at distance | `connectionstyle='arc3,rad=0.3'` |

## 4. Real Mechanism Context (Project-Specific)

This project studies **铁衰老 (ferroptosis-related SIPS / iron-driven aging)** in **CIRI (cerebral ischemia-reperfusion injury)**. Real mechanism axes to draw:

### 4.1 Core ferroaging axis (verified from literature + project data)
```
CIRI (ischemia-reperfusion)
  ↓
Iron overload (Fe2+ liberation from lysosome)
  ↓
ROS surge (Fenton reaction: Fe2+ + H2O2 → Fe3+ + OH• + OH-)
  ↓
Lipid peroxidation (ACSL4 → LPCAT3 → PUFA-PL)
  ↓
Membrane damage → ferroptosis / SIPS
  ↓
Microglia polarization (M1 pro-inflammatory)
  ↓
Neuroinflammation → neuronal death → CIRI exacerbation
```

### 4.2 BCP (β-caryophyllene) protective mechanism
```
BCP (compound, from 桂艾 Guìngài / Zhuang medicine)
  ↓ binds
CB2 receptor (membrane)
  ↓ activates
PI3K/Akt pathway
  ↓ phosphorylates
Nrf2 (releases from Keap1)
  ↓ translocates to nucleus
Nrf2 → ARE → HO-1 / NQO1 / FTH1 (ferritin heavy chain)
  ↓
Iron sequestration (FTH1 traps Fe2+ as Fe3+)
ROS scavenging (HO-1, NQO1)
GSH restoration (SLC7A11 → GPX4)
  ↓
Lipid peroxidation ↓
Ferroptosis ↓
Microglia M1 → M2 shift
  ↓
Neuroprotection in CIRI
```

### 4.3 Verified key targets (from L1/L2 analysis)
- **Ferroaging signature genes (LASSO-selected)**: SAT1, CD74, KLF6, LIFR, EBF3
- **Iron metabolism**: TFRC, FTH1, FTL, SLC40A1 (ferroportin), NFS1
- **Lipid peroxidation**: ACSL4, LPCAT3, ALOX15
- **Antioxidant defense**: GPX4, SLC7A11, NQO1, HMOX1
- **Microglia markers**: ITGAM (CD11b), P2RY12, CX3CR1, TMEM119

## 5. Implementation Templates

### 5.1 Template A: Signaling Cascade (matplotlib)
```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(7.2, 4.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

def node(ax, x, y, text, color, w=1.8, h=0.7):
    box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                         boxstyle="round,pad=0.08",
                         linewidth=1.2, edgecolor='black', facecolor=color)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

def arrow(ax, x1, y1, x2, y2, style='-|>', color='black', lw=1.5, rad=0):
    cs = f'arc3,rad={rad}' if rad else 'arc3'
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, mutation_scale=18,
                        color=color, linewidth=lw, connectionstyle=cs)
    ax.add_patch(a)

# Example: BCP → CB2 → Nrf2 → HO-1/FTH1 → neuroprotection
node(ax, 1.5, 8, 'BCP', '#3498DB')
node(ax, 3.5, 8, 'CB2', '#8E44AD')
node(ax, 5.5, 8, 'PI3K/Akt', '#F39C12')
node(ax, 7.5, 8, 'Nrf2', '#16A085')
node(ax, 5.5, 4, 'HO-1\nFTH1\nGPX4', '#27AE60', w=2.2, h=1.2)
node(ax, 1.5, 1, 'Iron↓', '#7D6608')
node(ax, 4.0, 1, 'ROS↓', '#E74C3C')
node(ax, 6.5, 1, 'LPO↓', '#E67E22')
node(ax, 9.0, 1, 'Neuroprotection', '#16A085', w=2.4, h=0.9)

arrow(ax, 2.4, 8, 3.1, 8)                    # BCP→CB2
arrow(ax, 4.4, 8, 5.1, 8)                    # CB2→PI3K
arrow(ax, 6.4, 8, 7.1, 8)                    # PI3K→Nrf2
arrow(ax, 7.5, 7.6, 6.0, 4.6, rad=0.2)       # Nrf2→HO-1 etc
arrow(ax, 5.0, 3.4, 1.7, 1.5, rad=-0.2)      # HO-1→Iron↓
arrow(ax, 5.3, 3.4, 4.0, 1.5, rad=-0.1)      # HO-1→ROS↓
arrow(ax, 5.6, 3.4, 6.3, 1.5, rad=0.1)       # HO-1→LPO↓
arrow(ax, 6.9, 1.0, 8.1, 1.0)                # LPO↓→Neuroprotection

plt.tight_layout()
plt.savefig('BCP_mechanism.pdf', bbox_inches='tight')
plt.savefig('BCP_mechanism.png', dpi=300, bbox_inches='tight')
```

### 5.2 Template B: Cellular Schematic (multi-compartment)
For membrane receptor → cytosol → nucleus signaling, draw:
- Outer cell boundary: `Ellipse(xy=(cx, cy), width=8, height=6, fill=False, lw=1.5)`
- Membrane band: `Ellipse(..., width=8.2, height=6.2, fill=True, facecolor='#FFD7B5', alpha=0.3)`
- Nucleus: `Ellipse(xy=(cx, cy), width=3, height=2.5, facecolor='#D7BDE2', alpha=0.6)`
- Receptor: `Rectangle` straddling membrane (transmembrane helix schematic)
- Translocation arrow: `FancyArrowPatch` with `arrowstyle='-|>', mutation_scale=25, lw=2.5` (thick for trafficking)

### 5.3 Template C: Cell-Cell Communication (networkx bipartite)
```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
# Source cells (microglia) | ligand-receptor pairs | target cells (neurons)
sources = ['Microglia', 'Astrocyte', 'OLs']
targets = ['Neuron', 'Endothelial']
edges = [('Microglia','Neuron','TNF-TNFR1'), ('Astrocyte','Neuron','GDNF-GFRa1')]
# Position sources left, targets right
pos = {n:(0,i) for i,n in enumerate(sources)}
pos.update({n:(2,i) for i,n in enumerate(targets)})
nx.draw(G, pos, with_labels=True, node_color=['#F5B7B1']*len(sources)+['#AED6F1']*len(targets),
        node_size=2000, font_size=9)
```

### 5.4 Template D: Workflow / Flowchart
For experimental workflow (MCAO → snRNA-seq → SCISSOR → BCP validation):
- Use `FancyBboxPatch` with `boxstyle="round,pad=0.1"` for each step
- Connect with `FancyArrowPatch` `arrowstyle='-|>'`
- Group parallel branches with a vertical bracket `)`
- Add icons via `ax.imshow()` only if real PNG/SVG assets exist in `figures/icons/`

## 6. Drawing Rules (HARD constraints)

### 6.1 Layout
- **Flow direction**: top-to-bottom (cascade) or left-to-right (causal chain); pick one and stay consistent
- **Spacing**: minimum 1.5× node height between consecutive nodes; avoid crossing arrows
- **Grouping**: cluster related nodes within a translucent `Rectangle` background (`alpha=0.1`)
- **Compartment labels**: italicized grey text at top-left of each compartment (e.g., *"Blood"*, *"BBB"*, *"Brain parenchyma"*)

### 6.2 Typography
- **Node labels**: 8-9 pt bold, centered
- **Edge labels** (when needed): 6-7 pt italic, placed at 0.5 along the arrow with `ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', pad=1)`
- **Title**: 11 pt bold, left-aligned at top
- **Legend**: 7 pt, bottom-right or as separate `ax_legend`

### 6.3 Annotations
- **Statistical overlays** (e.g., "P < 0.001", "FC = 2.3"): place near the corresponding node with a small arrow
- **Quantitative data**: NEVER fabricate — only annotate with values read from real CSV/TSV files
- **References**: if drawing a literature-based mechanism, cite source (PMID) in figure legend

## 7. Export Checklist

- [ ] PDF saved (vector, preferred for journal submission)
- [ ] PNG at ≥300 DPI saved for preview
- [ ] Figure width matches column (89 / 120 / 183 mm)
- [ ] All nodes have labels with readable font (≥7 pt)
- [ ] Arrows clearly indicate direction (activation vs inhibition)
- [ ] Color conventions followed (Section 3.3)
- [ ] No fabricated data values (all annotations from real sources)
- [ ] Mechanism legend written to `Figure_Legends.txt`
- [ ] File named `Mechanism_{description}.pdf` in `figures/mechanisms/`

## 8. Integration with Project Data

When drawing project-specific mechanisms, integrate with real analysis outputs:

| Mechanism element | Real data source | How to overlay |
|---|---|---|
| Ferroaging score per cell type | `figures/meta_with_umap.csv` (AddModuleScore_FA96) | Color-code nodes by mean score |
| DE genes in CIRI vs sham | `L2/results/...` | Highlight nodes whose genes are DE (FC>1, P<0.05) |
| BCP predicted targets | `L4/results_v10_minibatch/tcm_top_candidates_v*.csv` | Bold outline around confirmed targets |
| CellChat ligand-receptor pairs | `L2/results/cellchat/` | Edge thickness proportional to communication probability |
| SCISSOR-selected cells | `figures/scissor_umap_metadata.csv` | Annotate which cell types are SCISSOR+/- |

## 9. Common Failure Modes & Fixes

| Symptom | Cause | Fix |
|---|---|---|
| Arrows overlap nodes | connectionstyle default | Use `connectionstyle='arc3,rad=0.3'` to curve around |
| Text clipped at figure edge | tight_layout not used | `plt.savefig(..., bbox_inches='tight')` |
| Inconsistent box sizes | Manual width/height per node | Define `node()` helper with default w/h (see Template A) |
| Compartment boundaries invisible | alpha too low | Use `alpha=0.2-0.3` for fill, `lw=1.2` for edge |
| Chinese characters as boxes | matplotlib font not set | `plt.rcParams['font.sans-serif']=['SimHei','Microsoft YaHei']` + `plt.rcParams['axes.unicode_minus']=False` |
| Networkx layout overlaps | Default spring_layout | Use `pos = nx.bipartite_layout(G, sources)` or manual `pos` dict |

## 10. Reference Materials

- **Pathway databases** (verify via real URLs, do NOT fabricate):
  - KEGG: https://www.kegg.jp/kegg/pathway.html
  - Reactome: https://reactome.org/PathwayBrowser/
  - WikiPathways: https://www.wikipathways.org/
- **Project-specific verified mechanisms** (from `标书_终版_v13_含图表.docx` and L1-L4 analysis):
  - BCP → CB2 → Nrf2 axis (project central hypothesis)
  - Ferroaging 96-gene signature (L2 snRNA-seq result)
  - SAT1/CD74/KLF6/LIFR/EBF3 LASSO signature (L2 cross-cohort)
- **R templates** for pathway overlays: `D:/R语言绘图模板/25相关性网络图/` (corNetwork), `D:/R语言绘图模板/51解剖图/` (gganatogram — if installed)

## 11. Workflow

1. **Identify mechanism type** (cascade / network / cellular / workflow)
2. **List nodes & edges** from real literature or project data — do NOT invent
3. **Choose layout direction** (top-down or left-right)
4. **Draft positions** on paper or via `networkx.spring_layout` for initial placement
5. **Refine manually** to minimize crossings and balance spacing
6. **Apply color conventions** (Section 3.3)
7. **Add statistical / data overlays** from real CSV files
8. **Export PDF + PNG** to `figures/mechanisms/`
9. **Write figure legend** with PMID citations and data sources
