const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        AlignmentType, HeadingLevel, LevelFormat, WidthType, BorderStyle,
        ShadingType, Header, Footer, PageNumber } = require('docx');
const fs = require('fs');

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

function cell(text, width, fill = "FFFFFF", opts = {}) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill, type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text, bold: opts.bold || false, italics: opts.italics || false, size: opts.size || 20 })] })]
  });
}

function headerCell(text, width) {
  return cell(text, width, "D5E8F0", { bold: true, size: 20 });
}

function tableTitle(text) {
  return new Paragraph({
    spacing: { before: 160, after: 80 },
    alignment: AlignmentType.LEFT,
    children: [new TextRun({ text, bold: true, size: 22 })]
  });
}

function tableFootnote(text) {
  return new Paragraph({
    spacing: { after: 160, line: 300 },
    alignment: AlignmentType.LEFT,
    children: [new TextRun({ text, size: 18, italics: true })]
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

// 三线表边框（标准学术格式：顶线粗、表头底线细、底线粗）
const threeLineTop = { style: BorderStyle.SINGLE, size: 3, color: "000000" };
const threeLineMiddle = { style: BorderStyle.SINGLE, size: 1, color: "000000" };
const threeLineBottom = { style: BorderStyle.SINGLE, size: 3, color: "000000" };
const threeLineNone = { style: BorderStyle.NONE, size: 0 };

function threeLineCell(text, width, bold = false, size = 20) {
  return new TableCell({
    borders: { top: threeLineNone, bottom: threeLineNone, left: threeLineNone, right: threeLineNone },
    width: { size: width, type: WidthType.DXA },
    margins: { top: 40, bottom: 40, left: 60, right: 60 },
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, bold, size, font: "Times New Roman" })]
    })]
  });
}

function threeLineHeaderRow(cells) {
  return new TableRow({
    tableHeader: true,
    children: cells,
    borders: { top: threeLineTop, bottom: threeLineMiddle, left: threeLineNone, right: threeLineNone }
  });
}

function threeLineBodyRow(cells) {
  return new TableRow({
    children: cells,
    borders: { top: threeLineNone, bottom: threeLineNone, left: threeLineNone, right: threeLineNone }
  });
}

function threeLineBottomRow(cells) {
  return new TableRow({
    children: cells,
    borders: { top: threeLineNone, bottom: threeLineBottom, left: threeLineNone, right: threeLineNone }
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
    children: [
      heading1("CIRI-铁衰老转录特征的生物信息学鉴定及其对体外模型的指导意义"),

      p("为筛选可用于体外干预实验的分子靶标与细胞模型，我们整合四个脑缺血转录组数据集开展了系统的生物信息学分析。方法上，利用FerrDb v2铁死亡驱动基因、细胞衰老标志物和课题组前期构建的铁衰老基因集（96个基因），在GSE104036（小鼠MCAO）、GSE16561（人脑卒中）、GSE61616（大鼠MCAO）和GSE97537（大鼠MCAO）四个数据集中计算ssGSEA评分。以GSE104036中铁衰老评分中位数定义高铁衰老活性状态，通过50次重复6折交叉验证的LASSO逻辑回归筛选特征基因，并在三个外部数据集中验证。利用STRING v12.0评估\u03B2-石竹烯（BCP）靶标与特征基因的网络关系。此外，对GSE233815 snRNA-seq数据（7,414个细胞核）计算SenePy通用衰老评分和铁衰老AddModuleScore，构建双轴铁衰老复合评分。"),

      figTitle("表1. 分析所用数据集"),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [1400, 1400, 1200, 1000, 1000, 1200, 2160],
        rows: [
          new TableRow({ children: [
            headerCell("数据集", 1400), headerCell("物种", 1400), headerCell("平台", 1200),
            headerCell("样本量", 1000), headerCell("分组", 1000), headerCell("时间点", 1200),
            headerCell("用途", 2160)
          ]}),
          new TableRow({ children: [
            cell("GSE104036", 1400), cell("小鼠", 1400), cell("RNA-seq", 1200),
            cell("27", 1000), cell("Sham/Ipsi/Contra", 1000), cell("0\u201372 h", 1200),
            cell("LASSO训练；时间动态", 2160)
          ]}),
          new TableRow({ children: [
            cell("GSE16561", 1400), cell("人", 1400), cell("芯片", 1200),
            cell("63", 1000), cell("Stroke/Ctrl", 1000), cell("N/A", 1200),
            cell("外部验证集1", 2160)
          ]}),
          new TableRow({ children: [
            cell("GSE61616", 1400), cell("大鼠", 1400), cell("芯片", 1200),
            cell("15", 1000), cell("MCAO/XST", 1000), cell("24 h", 1200),
            cell("外部验证集2", 2160)
          ]}),
          new TableRow({ children: [
            cell("GSE97537", 1400), cell("大鼠", 1400), cell("芯片", 1200),
            cell("12", 1000), cell("MCAO/Sham", 1000), cell("24 h", 1200),
            cell("外部验证集3", 2160)
          ]}),
          new TableRow({ children: [
            cell("GSE233815", 1400), cell("小鼠", 1400), cell("snRNA-seq", 1200),
            cell("7,414", 1000), cell("Ctrl/1-7DPI", 1000), cell("0\u20137 d", 1200),
            cell("单核衰老评分", 2160)
          ]})
        ]
      }),
      tableFootnote("MCAO：大脑中动脉闭塞；Ipsi：患侧；Contra：对侧；XST：血栓通；DPI：损伤后天数。"),

      p("结果一：铁衰老评分在四个数据集中均显示最强的疾病vs对照效应量（Cohen\u2019s d），且铁死亡与铁衰老评分均在MCAO后6小时同步达峰，而经典衰老评分在急性至亚急性期持续下降，不支持铁死亡向衰老顺序过渡的假设。"),

      p("结果二：LASSO稳定性选择鉴定出五个高频特征基因（表2），内部交叉验证AUC=0.73\u00B10.09（置换p=0.002）。SAT1（直接靶标）和KLF6、CD74、LIFR（一阶PPI邻居）纳入BCP调控网络；EBF3无已知铁死亡关联但选择频率高（88%），其生物学意义待进一步体外验证。"),

      tableTitle("表2. CIRI-铁衰老特征基因与BCP网络连接"),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [900, 800, 800, 800, 800, 1260, 3200],
        rows: [
          new TableRow({ children: [
            headerCell("基因", 900), headerCell("选择\n频率", 800), headerCell("最终\n系数", 800),
            headerCell("Cohen's\nd", 800), headerCell("置换\np值", 800),
            headerCell("BCP连接", 1260), headerCell("功能与体外检测建议", 3200)
          ]}),
          new TableRow({ children: [
            cell("SAT1", 900), cell("96%", 800), cell("0.42", 800),
            cell("1.11", 800), cell("0.002", 800),
            cell("直接靶标", 1260),
            cell("p53\u2192SAT1\u2192铁死亡轴核心酶；BCP直接抑制；qPCR/WB最优先验证", 3200)
          ]}),
          new TableRow({ children: [
            cell("KLF6", 900), cell("88%", 800), cell("0.31", 800),
            cell("0.93", 800), cell("\u2014", 800),
            cell("一阶PPI邻居", 1260),
            cell("Nrf2/HO-1通路上游；OGD/R模型中检测表达变化", 3200)
          ]}),
          new TableRow({ children: [
            cell("CD74", 900), cell("70%", 800), cell("0.25", 800),
            cell("0.86", 800), cell("\u2014", 800),
            cell("一阶PPI邻居", 1260),
            cell("MIF受体，小胶质细胞M1极化标志；BV2/原代小胶质细胞检测", 3200)
          ]}),
          new TableRow({ children: [
            cell("LIFR", 900), cell("72%", 800), cell("\u2014", 800),
            cell("0.78", 800), cell("\u2014", 800),
            cell("一阶PPI邻居", 1260),
            cell("LIF/LIFR\u2192STAT3神经保护通路；SH-SY5Y缺氧模型验证", 3200)
          ]}),
          new TableRow({ children: [
            cell("EBF3", 900), cell("88%", 800), cell("\u2014", 800),
            cell("0.71", 800), cell("\u2014", 800),
            cell("无连接", 1260),
            cell("神经元分化因子；选择频率高但功能不明，机制待探索", 3200)
          ]})
        ]
      }),
      tableFootnote("BCP：\u03B2-石竹烯；PPI：蛋白互作（STRING v12.0, combined score>700）；\u2014表示该参数不适用。"),

      p("结果三：模型预测概率与独立铁衰老评分在三个外部验证集中均显著相关（GSE16561: rho=0.56; GSE61616: rho=0.75; GSE97537: rho=0.88, 均p<0.0001），表明该特征具有跨物种、跨平台的稳健性。"),

      p("结果四（通路支撑）：44个铁死亡驱动基因是BCP直接靶标，336个为一阶邻居，超几何检验p=2.41e-43（图4）。这提示BCP并非通过单一靶点，而是通过密集连接的铁死亡调控子网络发挥作用。SAT1作为BCP直接靶标，成为体外验证BCP抗铁死亡活性的首选基因。"),

      // PLIP + P2Rank整合三线表
      figTitle("表4. 铁衰老靶点蛋白-配体结合特征分析（PLIP + P2Rank/PRANKWeb v4）"),
      new Table({
        width: { size: 9400, type: WidthType.DXA },
        columnWidths: [1400, 1000, 1000, 2800, 3200],
        rows: [
          threeLineHeaderRow([
            threeLineCell("蛋白(结构)", 1400, true),
            threeLineCell("疏水\n作用数", 1000, true),
            threeLineCell("结合能\n(kcal/mol)", 1000, true),
            threeLineCell("主要残基", 2800, true),
            threeLineCell("意义", 3200, true),
          ]),
          threeLineBodyRow([
            threeLineCell("SAT1\n(2B4B)", 1400),
            threeLineCell("10", 1000),
            threeLineCell("\u22127.405", 1000),
            threeLineCell("Trp84(3.78\u00C5)\nLeu148(3.76\u00C5)\nTyr27(3.81\u00C5)\nPhe94(3.64\u00C5)\nPhe139(3.85\u00C5)", 2800),
            threeLineCell("P2Rank评分49.51，为铁死亡靶点最强可药化信号；BCP占据COA活性位，抑制多胺氧化促铁死亡通路(PMID:33545065)", 3200),
          ]),
          threeLineBodyRow([
            threeLineCell("KLF6\n(7QPG)", 1400),
            threeLineCell("0", 1000),
            threeLineCell("\u22127.232", 1000),
            threeLineCell("Asp279 Val286\nTrp287 Val309\n(PRANKWeb\npocket1, score=8.63)", 2800),
            threeLineCell("apo无配体；P2Rank最高8.63远低SAT1(49.51)；BCP直接结合ZnF证据不足，对铁死亡的调控以间接为主", 3200),
          ]),
          threeLineBodyRow([
            threeLineCell("CD74\n(1L3H)", 1400),
            threeLineCell("7", 1000),
            threeLineCell("\u22125.113", 1000),
            threeLineCell("Phe19 Tyr29\nTyr35 Trp42\n(PRANKWeb\npocket1, score=8.20)", 2800),
            threeLineCell("P2Rank评分8.20中等；BCP结合能-5.113较弱，且CD74与铁死亡关联有限，直接靶向意义存疑", 3200),
          ]),
          threeLineBottomRow([
            threeLineCell("LIFR\n(2Q7H)", 1400),
            threeLineCell("1", 1000),
            threeLineCell("\u22126.700", 1000),
            threeLineCell("Leu284 Tyr306\nPhe342 Trp417\n(PRANKWeb\npocket1, score=22.47)", 2800),
            threeLineCell("P2Rank评分22.47高可信；BCP结合能-6.700；裂隙位于D2-D3域，但PLIP仅1个接触，需共晶验证", 3200),
          ]),
        ]
      }),
      tableFootnote("PLIP v2.4(openbabel 3.2.1)分析23个PDB结构中结晶配体与蛋白的疏水接触数；结合能来自PRANKWeb v4 AutoDock Vina模块(BCP分子对接)；P2Rank(2.5)评分范围为0~100(>10为强可药化信号)。"),

      p("结果五（细胞模型支撑）：GSE233815单核数据分析显示，SenePy与铁衰老评分正交（rho=0.185，仅8/635共享基因）。铁衰老评分在少突胶质细胞（均值0.178）和神经元（0.137）中最高，提示这两类细胞是铁依赖性应激的优选体外模型；SenePy评分在小胶质细胞（0.158）和内皮/周细胞（0.149）中最高，且小胶质细胞在3DPI、内皮/周细胞在7DPI显著升高，表明二者是研究缺血后衰老转变的关键细胞类型。双轴铁衰老复合评分在星形胶质细胞中检测到单独评分未能捕捉的显著变化（3DPI, p_adj=0.047），建议体外模型联合使用铁死亡和衰老双指标。"),

      figTitle("表5. 各类细胞的评分特征与体外模型建议"),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [1400, 1200, 1200, 1200, 1800, 2560],
        rows: [
          new TableRow({ children: [
            headerCell("细胞类别", 1400), headerCell("SenePy\n(均值)", 1200), headerCell("FA-96\n(均值)", 1200),
            headerCell("显著\n条件", 1200), headerCell("优选检测指标", 1800), headerCell("体外模型建议", 2560)
          ]}),
          new TableRow({ children: [
            cell("神经元", 1400), cell("0.082", 1200), cell("0.137", 1200),
            cell("ns", 1200), cell("FA-96、SAT1、KLF6", 1800),
            cell("SH-SY5Y/原代神经元OGD/R + BCP干预；检测铁死亡标志物", 2560)
          ]}),
          new TableRow({ children: [
            cell("星形胶质细胞", 1400), cell("0.094", 1200), cell("0.089", 1200),
            cell("3DPI铁衰老复合", 1200), cell("铁衰老复合评分", 1800),
            cell("原代星形胶质细胞OGD；同步检测铁死亡+衰老双通路", 2560)
          ]}),
          new TableRow({ children: [
            cell("小胶质细胞", 1400), cell("0.158", 1200), cell("0.071", 1200),
            cell("3DPI SenePy", 1200), cell("SenePy、CD74、SASP因子", 1800),
            cell("BV2/原代小胶质细胞LPS/OGD；BCP抗炎抗衰老验证", 2560)
          ]}),
          new TableRow({ children: [
            cell("少突胶质细胞", 1400), cell("0.073", 1200), cell("0.178", 1200),
            cell("ns", 1200), cell("FA-96、铁代谢基因", 1800),
            cell("少突前体细胞(OPC)缺氧模型；铁螯合干预", 2560)
          ]}),
          new TableRow({ children: [
            cell("内皮/\n周细胞", 1400), cell("0.149", 1200), cell("0.061", 1200),
            cell("7DPI SenePy", 1200), cell("SenePy、LIFR", 1800),
            cell("bEnd.3/原代脑内皮细胞OGD；检测衰老相关分泌表型", 2560)
          ]})
        ]
      }),
      tableFootnote("ns：不显著（Wilcoxon test vs Ctrl, BH校正）。OGD/R：氧糖剥夺/复糖复氧。SASP：衰老相关分泌表型。"),

      p("综上，本分析为体外实验提供了以下可操作的支撑：（1）**分子靶标**\u2014\u2014SAT1为首选BCP验证靶点，联合KLF6、CD74、LIFR构成CIRI-铁衰老转录特征检测面板；（2）**通路框架**\u2014\u2014BCP通过SAT1\u2192铁死亡调控网络发挥作用，建议体外检测脂质过氧化（MDA/4-HNE）、GPX4和SLC7A11水平作为铁死亡通路验证；（3）**细胞模型**\u2014\u2014神经元（SH-SY5Y OGD/R）用于铁死亡机制验证，小胶质细胞（BV2 OGD/LPS）用于衰老-炎症表型评估，原代星形胶质细胞用于双轴铁衰老复合变化监测。五基因特征面板可作为体外干预效果的核心评价指标。")
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  const outPath = "D:/铁衰老 绝不重蹈覆辙/L2/results/CIRI_ferroaging_IVD_section.docx";
  fs.writeFileSync(outPath, buffer);
  console.log("体外模型指导小节已保存至: " + outPath);
});
