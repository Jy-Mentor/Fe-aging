const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        AlignmentType, HeadingLevel, LevelFormat, WidthType, BorderStyle,
        ShadingType, Header, Footer, PageNumber } = require('docx');
const fs = require('fs');

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

function cell(text, width, fill = "FFFFFF") {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill, type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun(text)] })]
  });
}

function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 240, after: 120 },
    children: [new TextRun(text)]
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 200, after: 100 },
    children: [new TextRun(text)]
  });
}

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120, line: 360 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, bold: opts.bold || false, italics: opts.italics || false })]
  });
}

function figTitle(text) {
  return new Paragraph({
    spacing: { before: 160, after: 80 },
    alignment: AlignmentType.LEFT,
    children: [new TextRun({ text, bold: true, size: 22 })]
  });
}

function figLegend(text) {
  return new Paragraph({
    spacing: { after: 160, line: 320 },
    alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, size: 20, italics: true })]
  });
}

const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: "Arial", size: 24 }
      }
    },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } }
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    headers: {
      default: new Header({ children: [new Paragraph({ children: [new TextRun({ text: "CIRI-ferroaging signature and beta-caryophyllene target overlap", italics: true, size: 18 })] })] })
    },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })] })] })
    },
    children: [
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 120 },
        children: [new TextRun({ text: "Identification of a CIRI-ferroaging transcriptional signature and its overlap with beta-caryophyllene targets: a multi-dataset exploratory study", bold: true, size: 32 })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 240 },
        children: [new TextRun({ text: "[Author names and affiliations to be added]", italics: true, size: 22 })]
      }),

      heading1("Abstract"),
      p("Background: Cerebral ischemia-reperfusion injury (CIRI) involves ferroptosis and cellular senescence, yet a unified \"ferroaging\" signature bridging the two processes remains undefined. Methods: We performed single-sample gene set enrichment analysis (ssGSEA) across four brain ischemia time-course datasets using FerrDb driver genes, cellular senescence markers, and a project-specific ferroaging gene set. A high-ferroaging activity state was defined by the median ferroaging score in GSE104036. LASSO logistic regression with 50 repetitions of 6-fold cross-validation was used to identify CIRI-ferroaging feature genes, followed by Spearman correlation validation in three independent datasets. Network proximity between beta-caryophyllene targets and the signature was assessed using official STRING v12.0 protein-protein interactions. Results: The ferroaging score showed the strongest disease-vs-control effect size across all datasets. In GSE104036, ferroaging peaked at 6 h post-MCAO, coinciding with ferroptosis but diverging from canonical senescence trajectories. Five genes (SAT1, CD74, KLF6, LIFR, EBF3) were stably selected. Predicted ferroaging probabilities correlated significantly with ferroaging scores in GSE16561 (rho=0.56), GSE61616 (rho=0.75), and GSE97537 (rho=0.88). SAT1 was a direct beta-caryophyllene target; CD74, KLF6, and LIFR were first-order PPI neighbors. Ferroptosis driver genes were strongly enriched among beta-caryophyllene target neighbors (hypergeometric p=2.41e-43). Conclusions: The CIRI-ferroaging signature captures an acute ferroptosis-associated transcriptional state rather than a late-onset senescence transition. Beta-caryophyllene may modulate this state via SAT1 and a broader ferroptosis-regulatory network."),

      heading1("Introduction"),
      p("Cerebral ischemia-reperfusion injury (CIRI) triggers multiple interconnected cell death and stress responses. Among these, ferroptosis—an iron-dependent lipid peroxidation-driven form of regulated cell death—has been strongly implicated in acute ischemic brain damage. Concurrently, cellular senescence, characterized by stable cell-cycle arrest and a pro-inflammatory secretome, is increasingly recognized in chronic post-stroke pathology."),
      p("The concept of \"ferroaging\" posits that iron dysregulation and lipid peroxidation may converge with senescence pathways, creating a transition window during which acute ferroptotic stress evolves into persistent senescent phenotypes. Identifying the genes that define this transition, and determining whether natural compounds such as beta-caryophyllene can modulate them, may provide new therapeutic angles for CIRI."),
      p("Here, we analyzed four publicly available brain ischemia datasets spanning mouse and rat models as well as human stroke patients. Using ssGSEA, we computed ferroptosis, senescence, and ferroaging scores per sample. Because the classical temporal pattern of \"ferroptosis rises then senescence rises\" was not supported by the data, we reframed the analytical question: we sought genes distinguishing a high-ferroaging activity state from baseline/low-activity states. We then tested whether these genes intersect with the predicted human target profile of beta-caryophyllene."),

      heading1("Methods"),

      heading2("Datasets and preprocessing"),
      p("Four datasets were analyzed: GSE104036 (mouse MCAO, RNA-seq, 27 samples across 0, 3, 6, 12, 24, 48, and 72 h), GSE16561 (human ischemic stroke vs. control, Illumina microarray, 63 samples), GSE61616 (rat MCAO with or without XST treatment, Affymetrix microarray, 15 samples), and GSE97537 (rat MCAO vs. sham, Affymetrix microarray, 12 samples). RNA-seq counts were converted to counts per million and log2-transformed. Microarray probes were collapsed to gene symbols using platform-specific annotations; cross-species symbols were harmonized by capitalization-based fallback mapping where needed."),

      heading2("Gene sets"),
      p("Three gene sets were used: (1) FerrDb v2 driver genes for ferroptosis activity; (2) a curated cellular senescence marker set; and (3) a project-specific ferroaging gene set derived from the intersection of iron metabolism, lipid peroxidation, and aging-related pathways."),

      heading2("ssGSEA scoring"),
      p("Single-sample GSEA scores were computed for each gene set and sample using a rank-based enrichment statistic implemented in Python (numpy), following the method described by Barbie et al. (Nature Protocols, 2009). Scores were standardized within each dataset by z-score transformation for cross-sample comparison."),

      heading2("Definition of high-ferroaging activity state"),
      p("In GSE104036, ipsilateral samples with ferroaging scores greater than or equal to the dataset median were labeled as high-ferroaging activity (Y=1); sham samples and ipsilateral samples below the median were labeled as low-activity/baseline (Y=0). This definition reflects the observed 6 h ferroaging peak and explicitly does not assume a sequential ferroptosis-to-senescence transition."),

      heading2("LASSO feature selection"),
      p("Feature selection was performed using L1-regularized logistic regression (coordinate descent). The expression matrix of the ferroaging gene set served as predictor variables. To ensure stability, we repeated 50 random subsamplings of 6-fold cross-validation and retained genes selected in more than 50% of repetitions. The final model was refit on all training samples using the selected gene set, and permutation testing (n=500) was used to assess model significance."),

      heading2("External validation"),
      p("The trained model was applied to independent datasets without retraining. In each validation set, predicted probabilities were correlated with the independently computed ferroaging score using Spearman's rank correlation. Additionally, samples were split by the within-dataset median ferroaging score into high and low groups, and the area under the ROC curve (AUC) was computed."),

      heading2("Beta-caryophyllene target overlap and network analysis"),
      p("Predicted human targets of beta-caryophyllene were intersected with the CIRI-ferroaging signature and with the FerrDb ferroptosis driver gene set. Protein-protein interactions were obtained from STRING v12.0 (human, combined score > 700). First-order neighbors of beta-caryophyllene targets were identified and hypergeometric tests were used to test enrichment of ferroptosis driver genes and CIRI-ferroaging signature genes among these neighbors."),

      heading1("Results"),

      heading2("Ferroaging score shows robust disease association across datasets"),
      p("In all four datasets, the ferroaging score demonstrated the largest disease-vs-control effect size (Cohen's d) compared with the ferroptosis and senescence scores. This consistency suggests that the ferroaging gene set captures a transcriptional signal that is closely linked to ischemic brain injury across species and platforms."),

      figTitle("Figure 1. Temporal dynamics of ferroptosis, senescence, and ferroaging scores in brain ischemia."),
      figLegend("(A) Mean ssGSEA trajectories in GSE104036 mouse MCAO samples. (B-D) Disease-vs-control effect sizes (Cohen's d) across the four datasets for ferroptosis, senescence, and ferroaging scores. Error bars represent 95% confidence intervals."),

      heading2("Temporal pattern in GSE104036 does not support a classical transition window"),
      p("In GSE104036, both ferroptosis and ferroaging scores peaked at 6 h post-MCAO and declined thereafter. Unexpectedly, the canonical cellular senescence score did not rise at later time points; instead, it decreased over the acute-to-subacute window. Consequently, the original hypothesis of a sequential \"ferroptosis peak → senescence rise → ferroaging transition\" was not supported by the data. We therefore redefined the analytical target as a high-ferroaging activity state rather than a transition phase."),

      heading2("LASSO identifies five stable CIRI-ferroaging feature genes"),
      p("Using the median-split ferroaging activity label, LASSO stability selection identified five genes with selection frequency greater than 50% across 50 repeated cross-validations: SAT1 (96%), EBF3 (88%), KLF6 (88%), LIFR (72%), and CD74 (70%). Refitting the model on the full training set yielded non-zero coefficients for SAT1, CD74, and KLF6, while LIFR and EBF3 were regularized to zero in the final sparse model. Internal cross-validation AUC was 0.73 ± 0.09, and permutation testing yielded p = 0.002."),

      figTitle("Figure 2. Stable CIRI-ferroaging feature genes selected by LASSO."),
      figLegend("(A) Selection frequency across 50 repeated 6-fold cross-validations. (B) Temporal expression of the five candidate genes in GSE104036 sham and ipsilateral samples. (C) Permutation distribution of cross-validation AUC; the observed AUC is indicated by the red dashed line."),

      heading2("Signature predicts ferroaging scores in independent datasets"),
      p("When applied to external datasets, the model's predicted probabilities correlated positively with the independently computed ferroaging score in GSE16561 (rho = 0.56, p < 0.0001), GSE61616 (rho = 0.75, p < 0.0001), and GSE97537 (rho = 0.88, p < 0.0001). High/low ferroaging groups showed AUCs of 0.74, 1.00, and 1.00 respectively. However, GSE61616 and GSE97537 have only 12–15 samples; these perfect AUCs should be interpreted with caution and require replication in larger cohorts."),

      figTitle("Figure 3. External validation of the CIRI-ferroaging signature."),
      figLegend("(A-C) Scatter plots of model-predicted probability versus ferroaging score in GSE16561, GSE61616, and GSE97537. (D) Model score distributions across disease/control or treatment groups. Spearman correlation coefficients and p-values are shown in each panel."),

      heading2("Beta-caryophyllene targets converge on the ferroaging-ferroptosis axis"),
      p("Among the five CIRI-ferroaging signature genes, SAT1 was a direct predicted human target of beta-caryophyllene. CD74, KLF6, and LIFR were first-order STRING PPI neighbors of beta-caryophyllene targets. EBF3 showed no direct or neighbor connection. More broadly, 44 ferroptosis driver genes were direct beta-caryophyllene targets, and 336 ferroptosis driver genes were first-order neighbors. This neighbor enrichment was highly significant (hypergeometric p = 2.41e-43)."),

      figTitle("Figure 4. Network convergence between beta-caryophyllene targets and ferroptosis/ferroaging genes."),
      figLegend("(A) Overlap between beta-caryophyllene targets and ferroptosis driver genes. (B) First-order PPI neighbors linking beta-caryophyllene targets to the CIRI-ferroaging signature. (C) Hypergeometric enrichment of ferroptosis driver genes among beta-caryophyllene target neighbors in STRING v12.0."),

      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [1600, 1600, 6160],
        rows: [
          new TableRow({ children: [cell("Gene", 1600, "D5E8F0"), cell("Frequency", 1600, "D5E8F0"), cell("Functional relevance", 6160, "D5E8F0")] }),
          new TableRow({ children: [cell("SAT1", 1600), cell("96%", 1600), cell("Spermidine/spermine N1-acetyltransferase 1; core mediator of p53-driven ferroptosis (PMID: 27698118). Direct beta-caryophyllene target.", 6160)] }),
          new TableRow({ children: [cell("KLF6", 1600), cell("88%", 1600), cell("Kruppel-like factor 6; linked to ferroptosis regulation after MCAO via Nrf2/HO-1 axis.", 6160)] }),
          new TableRow({ children: [cell("CD74", 1600), cell("70%", 1600), cell("MHC class II invariant chain; MIF receptor involved in microglial activation and neuroinflammation after stroke.", 6160)] }),
          new TableRow({ children: [cell("LIFR", 1600), cell("72%", 1600), cell("Leukemia inhibitory factor receptor; neuroprotective signaling after cerebral ischemia.", 6160)] }),
          new TableRow({ children: [cell("EBF3", 1600), cell("88%", 1600), cell("Early B-cell factor 3; neuronal differentiation factor. No direct PPI link to beta-caryophyllene targets.", 6160)] })
        ]
      }),

      heading1("Discussion"),
      p("Our analysis yields two main findings. First, the ferroaging gene set provides a robust cross-disease, cross-species transcriptional signal for brain ischemia, but its dynamics do not match the originally hypothesized sequential transition from ferroptosis to senescence. Instead, ferroaging activity peaks synchronously with ferroptosis in the acute window. Second, beta-caryophyllene's predicted target profile is strongly enriched for ferroptosis regulators and directly includes SAT1, a key ferroptosis enzyme."),
      p("The five CIRI-ferroaging feature genes can be interpreted as markers of an acute iron-dependent stress state. SAT1 and KLF6 have the strongest mechanistic links to ferroptosis and brain ischemia. CD74 and LIFR relate to inflammation and neuroprotection, respectively. EBF3, although stably selected, lacks direct literature support for ferroptosis or senescence and may reflect neuronal injury rather than a specific ferroaging mechanism."),
      p("The convergence of beta-caryophyllene targets with the ferroptosis network is the most robust network-level finding. The hypergeometric enrichment of ferroptosis drivers among beta-caryophyllene target neighbors (p = 2.41e-43) supports a model in which beta-caryophyllene does not necessarily bind every ferroptosis regulator directly, but acts through a densely connected subnetwork that modulates ferroptotic cell death."),

      heading1("Limitations"),
      p("This study is exploratory and has important limitations. The training sample size is small (n = 15), limiting the stability of LASSO selection and the generalizability of the model. The external validation datasets have even fewer samples, and AUCs of 1.00 in GSE61616 and GSE97537 likely reflect small-sample overfitting rather than perfect discrimination. The PPI analysis is based on predicted protein interactions and predicted compound targets; experimental validation is required. Finally, the finding that canonical senescence scores decreased over time contradicts the original transition-window hypothesis and implies that the ferroaging signature captures an acute ferroptosis-associated state rather than a late senescent transition."),

      heading1("Conclusion"),
      p("We identified a five-gene CIRI-ferroaging transcriptional signature that robustly tracks ferroaging activity across brain ischemia datasets. The signature peaks in the acute post-ischemic window alongside ferroptosis. Beta-caryophyllene's predicted targets significantly overlap with the ferroptosis regulatory network, with SAT1 as a direct target. These findings support beta-caryophyllene as a candidate modulator of acute ferroptosis-ferroaging responses in CIRI, pending experimental validation."),

      heading1("Data availability"),
      p("All data were obtained from publicly available GEO datasets GSE104036, GSE16561, GSE61616, and GSE97537. Analysis code and result tables are available upon reasonable request."),

      heading1("Competing interests"),
      p("The authors declare no competing interests.")
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("D:/铁衰老 绝不重蹈覆辙/L2/results/CIRI_ferroaging_paper_draft.docx", buffer);
  console.log("Paper draft saved to: D:/铁衰老 绝不重蹈覆辙/L2/results/CIRI_ferroaging_paper_draft.docx");
});
