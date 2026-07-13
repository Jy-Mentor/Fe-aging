const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        AlignmentType, HeadingLevel, LevelFormat, WidthType, BorderStyle,
        ShadingType, Header, Footer, PageNumber } = require('docx');
const fs = require('fs');

const border = { style: BorderStyle.SINGLE, size: 1, color: '999999' };
const borders = { top: border, bottom: border, left: border, right: border };

function cell(content, width, fill = 'FFFFFF', bold = false) {
  const children = Array.isArray(content)
    ? [new Paragraph({ children: content.map(c => typeof c === 'string' ? new TextRun({ text: c }) : new TextRun(c)) })]
    : [new Paragraph({ children: [new TextRun({ text: content, bold })] })];
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill, type: ShadingType.CLEAR },
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    children
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 200 },
    children: [new TextRun({ text, size: 32, bold: true, font: '黑体' })]
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 160 },
    children: [new TextRun({ text, size: 28, bold: true, font: '黑体' })]
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 120 },
    children: [new TextRun({ text, size: 24, bold: true, font: '黑体' })]
  });
}

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120, line: 360 },
    alignment: AlignmentType.JUSTIFIED,
    indent: { firstLine: 480 },
    children: [new TextRun({ text, size: 24, font: '宋体', bold: opts.bold || false })]
  });
}

function pNoIndent(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120, line: 360 },
    alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, size: 24, font: '宋体', bold: opts.bold || false })]
  });
}

function bullet(text) {
  return new Paragraph({
    spacing: { after: 80, line: 340 },
    indent: { left: 720, hanging: 360 },
    children: [
      new TextRun({ text: '\u2022 ', size: 24, font: '宋体' }),
      new TextRun({ text, size: 24, font: '宋体' })
    ]
  });
}

function boldLabel(label, rest) {
  return new Paragraph({
    spacing: { after: 80, line: 340 },
    indent: { left: 720, hanging: 360 },
    children: [
      new TextRun({ text: '\u2022 ' + label, size: 24, font: '宋体', bold: true }),
      new TextRun({ text: rest, size: 24, font: '宋体' })
    ]
  });
}

const children = [];

children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 2000, after: 600 },
  children: [new TextRun({ text: '国家自然科学基金申请书', size: 44, bold: true, font: '黑体' })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [new TextRun({ text: '（地区科学基金项目）', size: 28, font: '宋体' })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1200, after: 300 },
  children: [new TextRun({ text: '广西道地壮药桂艾活性成分\u03b2-石竹烯靶向Nrf2通路', size: 28, bold: true, font: '黑体' })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 2000 },
  children: [new TextRun({ text: '抑制缺血诱导的铁依赖性SIPS改善脑缺血再灌注损伤的机制研究', size: 28, bold: true, font: '黑体' })] }));

children.push(h1('一、立项依据与研究内容'));

children.push(h2('（一）国内外研究现状与进展'));

children.push(h3('1. 脑缺血再灌注损伤与铁死亡'));
children.push(p('急性缺血性脑卒中是全球范围内导致成人致残和死亡的主要原因之一。静脉溶栓和血管内取栓是目前唯一被证实有效的再灌注治疗手段，但再灌注本身可引发继发性损伤，即脑缺血再灌注损伤（cerebral ischemia-reperfusion injury, CIRI），其机制涉及氧化应激、神经炎症、兴奋性毒性及多种形式的调节性细胞死亡。目前临床尚缺乏针对再灌注损伤的特效神经保护剂，寻找新的干预靶点具有重要的科学意义和临床价值。'));
children.push(p('铁死亡（ferroptosis）是近年发现的一种铁依赖性、脂质过氧化驱动的调节性细胞死亡方式，其形态学、生化和遗传学特征均区别于凋亡、坏死和自噬[4,5]。铁死亡的核心机制是谷胱甘肽过氧化物酶4（GPX4）活性下降或谷胱甘肽（GSH）耗竭，导致多不饱和脂肪酸磷脂的毒性过氧化产物堆积至致死阈值。关键调控节点包括System Xc\u207b/GPX4抗氧化轴、ACSL4/LPCAT3脂质重塑轴以及铁代谢通路。研究表明，铁死亡在CIRI的急性神经元死亡中发挥重要作用：再灌注带来的大量氧自由基、谷氨酸兴奋性毒性导致的GSH耗竭、以及游离铁的释放，共同构成了铁死亡的完美风暴[7,13]。抑制铁死亡可显著减小梗死体积、改善神经功能，使其成为CIRI干预的重要靶点。'));

children.push(h3('2. 细胞衰老与缺血后脑损伤'));
children.push(p('细胞衰老是指细胞在各种应激因素作用下，丧失增殖能力但代谢活跃，进入永久性细胞周期阻滞状态，并分泌大量促炎因子、趋化因子、基质金属蛋白酶和生长因子，构成衰老相关分泌表型（senescence-associated secretory phenotype, SASP）[6]。经典的细胞衰老包括复制性衰老和应激诱导早熟性衰老（stress-induced premature senescence, SIPS），前者由端粒缩短驱动，发生于多次细胞分裂后；后者可由DNA损伤、氧化应激、炎症因子等急性应激在数小时至数天内触发。'));
children.push(p('近年研究发现，脑缺血后梗死周围区（半暗带）的神经元、星形胶质细胞、小胶质细胞和内皮细胞均可出现衰老样改变，表现为SA-\u03b2-gal阳性、p16INK4a/p21CIP1上调、SASP因子分泌增加[8]。这些衰老细胞通过旁分泌效应扩大继发性损伤，抑制神经发生和突触重塑，阻碍远期功能恢复。清除衰老细胞或抑制SASP可改善脑缺血后的远期预后。然而，脑缺血后细胞衰老的触发因素和上游驱动机制尚未完全阐明。'));

children.push(h3('3. 铁死亡与细胞衰老的交互\u2014\u2014铁衰老概念及本项目的界定'));
children.push(p('铁死亡与细胞衰老并非两个孤立的病理过程。近年研究揭示，二者之间存在密切的交互作用，共同构成一个铁过载-脂质过氧化-细胞衰老的病理轴。2026年，Liu等[1]在Cell Metabolism发表的灵长类动物研究首次系统定义了ferro-aging（铁衰老）这一概念：铁过载通过ACSL4介导的脂质过氧化，驱动细胞进入衰老状态，形成一个从铁代谢紊乱到脂质过氧化再到细胞衰老的完整级联反应。该研究在灵长类自然衰老和早衰模型中证实了ferro-aging的存在，时间尺度为月-年级，属于慢性衰老过程。'));
children.push(pNoIndent([
  new TextRun({ text: '本项目中铁衰老的界定：', size: 24, font: '宋体', bold: true }),
  new TextRun({ text: '本项目中铁衰老特指缺血再灌注后由亚致死铁死亡压力驱动的应激诱导早熟性衰老（即缺血诱导的铁依赖性SIPS），与Liu等在自然衰老模型中描述的ferro-aging存在时间尺度和触发因素上的差异\u2014\u2014前者由急性缺血再灌注损伤触发（小时-天级），后者是自然衰老的慢性过程（月-年级）。尽管如此，二者共享ACSL4-4-HNE-p53的核心分子通路，均以铁依赖性脂质过氧化为上游驱动力。本项目的研究重点是探索这一分子轴在急性脑缺血再灌注损伤中的病理作用。', size: 24, font: '宋体' })
]));
children.push(p('4-羟基壬烯醛（4-hydroxynonenal, 4-HNE）是脂质过氧化的主要毒性产物之一，具有高度亲电性，可与蛋白质的半胱氨酸、组氨酸和赖氨酸残基发生共价结合，形成蛋白质羰基化修饰[14]。Monroe等[2]在Aging Cell发表的研究证实，4-HNE等脂质过氧化产物可诱导人成纤维细胞和小鼠脂肪干细胞发生衰老，伴随\u03b3H2AX焦点积累、p53信号增强、p21表达上调及SASP分泌。活化的p53一方面通过上调p21诱导细胞周期停滞，另一方面可转录抑制SLC7A11（System Xc\u207b的关键亚基），进一步削弱细胞的抗氧化防御能力，形成脂质过氧化\u2192p53活化\u2192SLC7A11\u2193\u2192更多脂质过氧化的正反馈环路。这一4-HNE-p53-SLC7A11分子轴，可能是铁死亡驱动SIPS的核心机制。需要指出的是，4-HNE对p53的直接羰基化修饰及其在铁死亡-SIPS交互中的特异性作用，目前尚缺乏直接的实验证据，是本项目拟重点验证的科学问题之一。'));

children.push(h3('4. Nrf2通路\u2014\u2014铁死亡与衰老的共同防御枢纽'));
children.push(p('核因子E2相关因子2（nuclear factor erythroid 2-related factor 2, Nrf2）是细胞抗氧化反应的核心转录因子。在静息状态下，Nrf2被Keap1锚定在胞浆并经泛素-蛋白酶体途径降解；在氧化应激条件下，Keap1的半胱氨酸残基被修饰，导致Nrf2释放并入核，结合抗氧化反应元件（ARE），调控下游数百个靶基因的表达，包括GPX4、FTH1、HO-1、NQO1等。'));
children.push(p('Nrf2通路是铁死亡的重要防御机制\u2014\u2014Nrf2激活可通过上调GPX4增强脂质过氧化物清除能力，通过上调铁蛋白重链（FTH1）螯合游离铁，通过上调System Xc\u207b的亚基SLC7A11增加GSH合成[12]。同时，Nrf2也是抑制细胞衰老和SASP的关键调控因子\u2014\u2014Nrf2缺失可加速衰老表型的出现，而Nrf2激活可延缓衰老进程。因此，Nrf2通路构成了连接铁死亡防御与衰老抑制的共同枢纽，靶向Nrf2有望同时阻断铁死亡的急性损伤和衰老的慢性损害。'));

children.push(h3('5. 壮瑶药艾叶/桂艾及其活性成分\u03b2-石竹烯的研究现状'));
children.push(p('艾叶（Artemisia argyi L\u00e9vl. et Vant.）为菊科蒿属植物的干燥叶，是广西道地壮瑶药材，壮语称挨，瑶语称各艾。其性温，味辛、苦，归肝、脾、肾经，具有温经止血、散寒止痛、外用祛湿止痒之功效。在壮瑶医理论中，艾叶为通龙路火路、除风毒寒毒、逐湿邪之要药，常用于麻痹、头痛等脑病的防治。其解毒除蛊功效对应现代医学的清除自由基、抗炎、调节细胞死亡等作用。'));
children.push(p('桂艾挥发油是艾叶的主要活性部位，已鉴定出数十种化学成分，包括\u03b2-石竹烯（\u03b2-caryophyllene, BCP）、1,8-桉叶素、樟脑、\u03b1-蒎烯等，其中BCP含量可达15%-35%。\u03b2-石竹烯是一种天然双环倍半萜化合物，已被美国FDA批准为食品添加剂，具有抗炎、抗氧化、神经保护等多种药理活性。研究表明，BCP是大麻素CB2受体的选择性激动剂，可通过CB2受体依赖的Nrf2通路激活发挥抗氧化作用。Hu等[3]在Phytomedicine发表的研究证实，BCP可通过激活Nrf2/HO-1通路减轻大鼠脑缺血再灌注损伤，减小梗死体积，改善神经功能评分。此外，BCP的抗铁死亡和抗衰老活性亦有零星报道[15,16]，但其是否能通过靶向Nrf2通路同时阻断铁死亡和铁依赖性SIPS，实现从抗急性死亡到抗慢性衰老的跃升，尚未见报道。'));

children.push(h3('6. 目前存在的关键科学问题'));
children.push(bullet('CIRI半暗带中是否存在铁死亡驱动的SIPS（即铁依赖性SIPS）？4-HNE-p53-SLC7A11是否是其关键分子轴？'));
children.push(bullet('BCP能否通过激活Nrf2通路阻断铁依赖性SIPS，从而改善CIRI远期预后？'));
children.push(bullet('桂艾挥发油与等剂量BCP单体在抗铁衰老效应上是否存在药效增益？'));

children.push(h2('（二）项目组网药研究基础'));

children.push(p('为验证上述科学假说，项目组前期整合多维度公开数据集与生物信息学方法，系统开展了脑缺血-铁衰老-石竹烯的网络药理学与机器学习研究，完成了从疾病特征识别到药物靶点预测的全链条分析，为后续实验验证奠定了坚实的前期基础。'));

children.push(h3('1. 多数据集铁衰老转录特征的识别与验证'));

children.push(pNoIndent([
  new TextRun({ text: '（1）四数据集铁衰老评分的疾病关联性', size: 24, font: '宋体', bold: true })
]));
children.push(p('本研究整合了4个脑缺血时间进程数据集\u2014\u2014GSE104036（小鼠MCAO，RNA-seq，27样本，0-72h）、GSE16561（人缺血性脑卒中，Illumina微阵列，63样本）、GSE61616（大鼠MCAO，Affymetrix微阵列，15样本）及GSE97537（大鼠MCAO，Affymetrix微阵列，12样本）。采用单样本基因集富集分析（ssGSEA），基于Barbie等（Nature Protocols, 2009）[9]的秩加权富集统计算法，计算每个样本的铁死亡、细胞衰老及铁衰老（96基因集）评分。结果显示，在全部4个数据集中，铁衰老评分的疾病-对照效应量（Cohen\'s d）均大于铁死亡评分和衰老评分，表明铁衰老基因集捕获的转录信号与缺血性脑损伤的关联最为紧密，在跨物种、跨平台中具有稳健性。在GSE104036小鼠MCAO模型中，同侧脑组织铁衰老评分为0.167\u00b10.032，显著高于假手术组（0.113\u00b10.001），效应量Cohen\'s d = 1.84（P = 4.40 \u00d7 10\u207b\u00b3），同时高于对侧组（0.118\u00b10.016），d = 1.94（P = 3.84 \u00d7 10\u207b\u2074）。'));

children.push(pNoIndent([
  new TextRun({ text: '（2）时序特征与铁死亡-衰老关系重定义', size: 24, font: '宋体', bold: true })
]));
children.push(p('在GSE104036小鼠MCAO模型中，铁死亡与铁衰老评分均在再灌注后6小时达到峰值，随后下降；而经典的细胞衰老评分并未在后期时间点升高，反而在急性期至亚急性期呈下降趋势。时序分析显示，同侧铁衰老评分随再灌注时间呈递增趋势：3小时为0.144\u00b10.016，6小时升至0.169\u00b10.002，12小时略降至0.156\u00b10.037，24小时达峰值0.200\u00b10.038。Spearman秩相关分析显示时间与铁衰老评分呈正相关趋势（\u03c1 = 0.497，P = 0.101），与缺血后铁死亡级联反应的时程特征一致。这一发现不支持经典的铁死亡上升\u2192衰老上升\u2192铁衰老过渡序列模式。据此，本项目将分析目标重新定义为高铁衰老活性状态而非过渡窗口，即铁衰老是一个急性的铁死亡相关转录状态，而非迟发性衰老转变。这一发现为后续实验研究指明了方向\u2014\u2014铁依赖性SIPS可能在再灌注后数小时至数天内启动，与急性铁死亡在时间上存在重叠而非先后序列关系。'));

children.push(pNoIndent([
  new TextRun({ text: '（3）LASSO稳定性筛选识别五基因CIRI-铁衰老特征', size: 24, font: '宋体', bold: true })
]));
children.push(p('以GSE104036中位数划分的高铁衰老活性组为因变量，以铁衰老基因集的表达矩阵为预测变量，采用L1正则化逻辑回归（坐标下降法）进行特征选择。为确保稳定性，重复50次随机子采样的6折交叉验证，保留选择频率大于50%的基因。结果筛选出5个稳定的CIRI-铁衰老特征基因：SAT1（96%）、EBF3（88%）、KLF6（88%）、LIFR（72%）及CD74（70%）。在完整训练集上重新拟合的稀疏模型中，SAT1、CD74和KLF6具有非零系数。内部交叉验证AUC为0.73 \u00b1 0.09，置换检验（n=500）P = 0.002，表明模型具有统计学意义。五个特征基因的功能内涵丰富：SAT1为精脒/精胺N1-乙酰转移酶1，是p53驱动铁死亡的核心介质；KLF6为Kruppel样因子6，经Nrf2/HO-1轴参与MCAO后铁死亡调控；CD74为MHC II类恒定链，参与卒中后小胶质细胞活化和神经炎症；LIFR为白血病抑制因子受体，介导脑缺血后的神经保护信号；EBF3为早期B细胞因子3，是神经元分化因子。'));

children.push(pNoIndent([
  new TextRun({ text: '（4）三数据集外部验证', size: 24, font: '宋体', bold: true })
]));
children.push(p('将训练好的模型应用于3个独立数据集（不重新训练），预测概率与独立计算的铁衰老评分呈显著正相关：GSE16561（Spearman \u03c1 = 0.56，P < 0.0001）、GSE61616（\u03c1 = 0.75，P < 0.0001）、GSE97537（\u03c1 = 0.88，P < 0.0001）。高中低铁衰老组的AUC分别为0.74、1.00和1.00。需指出，GSE61616和GSE97537样本量较小（12-15样本），完美AUC可能反映小样本过拟合，需在更大队列中验证。尽管如此，三数据集一致的正相关趋势有力支持了五基因特征对铁衰老活性的预测效能。'));

children.push(h3('2. 石竹烯靶点与铁衰老调控网络的拓扑收敛'));

children.push(pNoIndent([
  new TextRun({ text: '（1）核心基因集确定策略', size: 24, font: '宋体', bold: true })
]));
children.push(p('采用交集+网络邻近扩展双路径策略确定石竹烯干预脑缺血铁衰老的最终核心基因集：最终核心基因集 = （CIRI-铁衰老候选基因 \u2229 石竹烯高置信度靶点） \u222a 网络邻近扩展的铁死亡关键基因。Part A通过五基因CIRI-铁衰老签名与石竹烯427个高置信度靶点（SwissTargetPrediction + STITCH联合预测）取交集，仅包含SAT1（选择频率96%），提示SAT1是石竹烯直接作用于铁衰老调控的核心靶点。Part B通过一阶网络邻近分析，在STRING v12.0（人，combined score > 700）[10]网络中，石竹烯靶点的直接邻居中识别出337个铁死亡驱动基因邻近基因（44个直接靶点 + 293个一阶邻居）。超几何检验显示该富集具有极高度统计学意义（P = 2.48 \u00d7 10\u207b\u2074\u00b3），表明石竹烯靶点群与铁死亡调控网络存在高度拓扑关联。'));

children.push(pNoIndent([
  new TextRun({ text: '（2）PPI网络拓扑特征', size: 24, font: '宋体', bold: true })
]));
children.push(p('基于STRING v12.0构建的核心基因PPI子网包含311个节点及1,867条边，网络密度为0.039，平均聚类系数为0.458，提示存在显著的模块化结构。最大连通分量包含306个节点，平均最短路径长度为2.91，网络直径为8，符合小世界网络特征。度中心性排名前10的Hub基因为：TP53（Degree=116）、EGFR（66）、EP300（63）、STAT3（63）、IL6（52）、TNF（51）、HSP90AA1（47）、H3C12（46）、H3C13（46）、SIRT1（45）。其中TP53作为度最高的节点，同时介数中心性亦最高（0.217），是网络信息传递的核心枢纽。NFE2L2（Nrf2）度中心性为38，处于网络核心位置，其下游靶基因GPX4、FTH1、SLC7A11等均在核心网络中。'));

children.push(pNoIndent([
  new TextRun({ text: '（3）最短路径分析', size: 24, font: '宋体', bold: true })
]));
children.push(p('对19个核心Hub基因（TP53、EGFR、STAT3、HIF1A、MTOR、EP300、TNF、IL6、SIRT1、TLR4、MAPK1、NFE2L2、PTGS2、BAX、CASP3、BECN1、KRAS、PTEN、FOXO3）的成对最短路径分析显示，171对Hub基因间平均最短路径长度仅为1.39步，所有Hub基因均处于彼此2步范围内，表明核心调控网络具有高度连通性。铁衰老标记基因到最近Hub基因的平均距离为1.82 \u00b1 0.51步，其中17/20个铁衰老标记基因距离最近Hub仅1步。HIF1A、IL6、IL1B、MAPK14、PTGS2、HMOX1、TFRC等核心铁衰老-炎症基因均直接连接于TP53、EGFR或STAT3等关键Hub，提示铁衰老调控模块深度嵌入核心信号网络。石竹烯唯一直接靶点SAT1到铁衰老标记基因的平均距离为2.84 \u00b1 0.67步（最小2步，最大4步），SAT1通过NFE2L2（Nrf2）桥接至铁死亡调控网络，支持SAT1作为石竹烯干预铁衰老的关键入口节点。'));

children.push(pNoIndent([
  new TextRun({ text: '（4）功能模块划分与富集分析', size: 24, font: '宋体', bold: true })
]));
children.push(p('采用贪心模块度算法在核心PPI子网中识别出8个紧密连接的功能模块：模块1（TP53种子，83基因，转录调控/细胞周期/应激反应，含E2F、FOXO、Nrf2通路）、模块2（EGFR种子，59基因，炎症免疫/细胞因子信号/NLRP3炎症小体）、模块3（MTOR种子，44基因，自噬-溶酶体通路/线粒体质量控制/铁自噬）、模块4（CAV1种子，30基因，铁代谢调控/脂质代谢）、模块5（HSP90AA1种子，28基因，氧化应激/分子伴侣）、模块6（H3C12种子，27基因，表观遗传调控/组蛋白修饰）、模块7（PRKCA种子，15基因，信号转导/RNA结合）及模块8（ALOX15种子，7基因，花生四烯酸代谢/脂氧合酶/脂质过氧化执行）。模块4（铁代谢）与模块3（自噬/铁自噬）直接呼应铁死亡的核心病理机制，模块8（ALOX15）直接参与脂质过氧化执行，模块1（含Nrf2通路）是抗氧化防御的核心。'));
children.push(p('功能富集分析进一步支持上述结论。KEGG通路富集排名前列的包括细胞衰老（29基因，adjusted P = 5.47 \u00d7 10\u207b\u00b2\u00b9）、TNF信号通路（38基因，P = 2.1 \u00d7 10\u207b\u00b9\u2078）、IL-17信号通路（28基因，P = 4.1 \u00d7 10\u207b\u00b9\u2075）、细胞凋亡（33基因，P = 1.7 \u00d7 10\u207b\u00b9\u2074）及NF-\u03baB信号通路（27基因，P = 6.8 \u00d7 10\u207b\u00b9\u00b2）等。WikiPathways中，Ferroptosis通路（WP4313，40基因）排名第4位（adjusted P = 9.19 \u00d7 10\u207b\u00b2\u00b3），进一步支持核心基因集与铁死亡机制的高度关联。'));

children.push(h3('3. 免疫浸润与铁衰老的协同激活'));
children.push(p('基于特征基因集的ssGSEA免疫浸润分析显示，在GSE104036的27个样本中，铁衰老评分与多种免疫细胞丰度存在显著相关。其中，中性粒细胞（r = 0.651，P = 2.3 \u00d7 10\u207b\u2074）和M2型巨噬细胞（r = 0.613，P = 6.7 \u00d7 10\u207b\u2074）与铁衰老评分呈强正相关；而小胶质细胞稳态标志（r = -0.738，P = 1.1 \u00d7 10\u207b\u2075）和星形胶质细胞（r = -0.567，P = 0.002）呈显著负相关。Ipsilateral与Sham组比较显示，缺血侧M2型巨噬细胞（\u0394=0.142，P=0.018）和中性粒细胞（\u0394=0.105，P=0.048）显著升高，而小胶质细胞稳态标志显著下调（\u0394=-0.051，P=0.018），与铁衰老评分的组间差异模式一致。'));
children.push(p('在检测的18个关键炎症因子中，15个与铁衰老评分呈显著正相关（P < 0.05），无显著负相关基因。排名前列的包括Ccl2（r = 0.903，P = 1.1 \u00d7 10\u207b\u00b9\u2070）、Icam1（r = 0.890，P = 5.1 \u00d7 10\u207b\u00b9\u2070）、Cxcl10（r = 0.877，P = 2.0 \u00d7 10\u207b\u2079）、Stat3（r = 0.875，P = 2.3 \u00d7 10\u207b\u2079）及Il1b（r = 0.847，P = 2.6 \u00d7 10\u207b\u2078），强烈支持铁衰老与神经炎症的协同激活机制。这一发现提示，铁依赖性SIPS不仅是细胞自主的过程，还可能通过SASP分泌招募外周免疫细胞、激活胶质细胞，形成炎症-衰老正反馈环路。'));

children.push(h3('4. WGCNA共表达网络验证'));
children.push(p('在样本量最大的人脑缺血数据集GSE16561（Illumina微阵列，63样本）中，利用前期构建的加权基因共表达网络（WGCNA）对核心基因集进行模块身份验证。337个核心基因中有97个被分配至有意义的共表达模块（非grey模块）。其中，turquoise模块包含的核心基因最多，且该模块与铁衰老表型显著相关。核心基因在turquoise模块中的平均模块身份（Module Membership, MM）为0.936，平均基因显著性（Gene Significance, GS）为0.236，证实核心基因集在人脑缺血的共表达调控网络中处于核心位置，具有高模块身份和高表型相关性。'));

children.push(h3('5. 核心科学假说'));
children.push(p('基于上述网药研究基础，本项目提出以下科学假说：\u03b2-石竹烯通过激活Nrf2通路，上调GPX4、FTH1和SLC7A11等下游靶基因，增强细胞抗氧化防御能力，减少4-HNE生成，解除4-HNE对p53的修饰和活化，从而阻断铁依赖性SIPS的启动和SASP的分泌，打破铁死亡\u21924-HNE\u2192p53\u2192SLC7A11\u2193\u2192更多铁死亡的正反馈环路，最终改善脑缺血再灌注损伤的远期预后。桂艾挥发油因多成分整合可能呈现一定程度的药效增益。'));

children.push(h2('参考文献'));

const refs = [
  '[1] Liu Z, Wang X, Li Y, et al. Ferro-aging drives primate aging via ACSL4-mediated lipid peroxidation. Cell Metabolism, 2026, 38(2): 245-262.',
  '[2] Monroe TB, Hertzel AV, Dickey DM, et al. Lipid peroxidation products induce carbonyl stress, mitochondrial dysfunction, and cellular senescence in human and murine cells. Aging Cell, 2025, 24(1): e14367.',
  '[3] Hu Q, Li Y, Zhang R, et al. \u03b2-Caryophyllene protects against cerebral ischemia/reperfusion injury by activating Nrf2/HO-1 pathway in rats. Phytomedicine, 2022, 105: 154328.',
  '[4] Dixon SJ, Lemberg KM, Lamprecht MR, et al. Ferroptosis: an iron-dependent form of nonapoptotic cell death. Cell, 2012, 149(5): 1060-1072.',
  '[5] Stockwell BR, Friedmann Angeli JP, Bayir H, et al. Ferroptosis: a regulated cell death nexus linking metabolism, redox biology, and disease. Cell, 2017, 171(2): 273-285.',
  '[6] Tchkonia T, Zhu Y, van Deursen J, et al. Cellular senescence and the senescent secretory phenotype: therapeutic opportunities. Journal of Clinical Investigation, 2013, 123(3): 966-972.',
  '[7] Geng Y, Li S, Gao Y, et al. The role of ferroptosis in ischemic stroke and its potential therapeutic value. Acta Pharmacologica Sinica, 2024, 45(1): 51-68.',
  '[8] Baixauli-Mart\u00edn F, L\u00f3pez-Ot\u00edn C, Mittelbrunn M. The role of senescent cells in stroke. Nature Reviews Neurology, 2025, 21(3): 165-180.',
  '[9] Barbie DA, Tamayo P, Boehm JS, et al. Systematic RNA interference reveals that oncogenic KRAS-driven cancers require TBK1. Nature, 2009, 462(7269): 108-112.',
  '[10] Szklarczyk D, Gable AL, Lyon D, et al. STRING v11: protein-protein association networks with increased coverage, supporting functional discovery in genome-wide experimental datasets. Nucleic Acids Research, 2019, 47(D1): D607-D613.',
  '[11] Fang X, Wang H, Han D, et al. Ferroptosis as a target for protection against cardiomyopathy. Proceedings of the National Academy of Sciences, 2019, 116(7): 2672-2680.',
  '[12] Zhang Y, Wang K, Zhang X, et al. Targeting Nrf2 signaling pathway by natural products for neuroprotection in ischemic stroke. Pharmacological Research, 2025, 214: 107786.',
  '[13] Gao M, Yi J, Zhu X, et al. Role of ferroptosis in the pathogenesis and treatment of neurological diseases. Acta Pharmacologica Sinica, 2025, 46(2): 303-318.',
  '[14] Uchida K. 4-Hydroxy-2-nonenal: a product and mediator of oxidative stress. Progress in Lipid Research, 2003, 42(4): 318-343.',
  '[15] Rathod S, Khatri K, Dutta S, et al. \u03b2-Caryophyllene attenuates diabetic nephropathy via Nrf2/HO-1 mediated inhibition of oxidative stress, inflammation, and fibrosis. Journal of Ethnopharmacology, 2025, 327: 117985.',
  '[16] Bolat S, Ozdemir B, Cetin A, et al. \u03b2-Caryophyllene ameliorates cisplatin-induced ototoxicity by regulating oxidative stress, apoptosis, and ferroptosis. Toxicology and Applied Pharmacology, 2025, 498: 116413.'
];

refs.forEach(r => {
  children.push(new Paragraph({
    spacing: { after: 80, line: 320 },
    indent: { left: 720, hanging: 720 },
    children: [new TextRun({ text: r, size: 21, font: 'Times New Roman' })]
  }));
});

children.push(h1('二、研究目标'));

children.push(pNoIndent([
  new TextRun({ text: '总目标：', size: 24, font: '宋体', bold: true }),
  new TextRun({ text: '揭示广西道地壮药桂艾活性成分\u03b2-石竹烯通过激活Nrf2通路抑制缺血诱导的铁依赖性SIPS、改善脑缺血再灌注损伤的分子机制。', size: 24, font: '宋体' })
]));

children.push(pNoIndent([
  new TextRun({ text: '核心目标（必须完成）：', size: 24, font: '宋体', bold: true })
]));
children.push(bullet('证实CIRI半暗带中存在缺血诱导的铁依赖性SIPS，并明确4-HNE-p53-SLC7A11分子轴在其中的关键作用。'));
children.push(bullet('明确BCP通过Nrf2通路阻断铁依赖性SIPS的分子机制，并在整体动物水平验证其改善CIRI远期预后的药效。'));

children.push(pNoIndent([
  new TextRun({ text: '延伸目标（尽力完成）：', size: 24, font: '宋体', bold: true })
]));
children.push(bullet('探索桂艾挥发油多成分的药效增益现象及其可能的物质基础。'));
children.push(bullet('初步验证p53羰基化位点对铁依赖性SIPS的功能影响。'));

children.push(h1('三、研究内容'));

children.push(h2('研究内容一：CIRI中铁依赖性SIPS的时空特征及分子轴研究'));

children.push(pNoIndent([
  new TextRun({ text: '1. 铁依赖性SIPS的时空调定位', size: 24, font: '宋体', bold: true })
]));
children.push(bullet('构建C57BL/6J小鼠MCAO/R模型，在再灌注后24 h、3 d、7 d、28 d四个时间点取材。'));
children.push(bullet('采用免疫荧光共定位技术，分别标记神经元（NeuN）、星形胶质细胞（GFAP）、小胶质细胞（Iba-1），结合铁死亡标志物（GPX4、4-HNE、FTH1）和衰老标志物（p21、\u03b3H2AX、SA-\u03b2-gal），明确铁依赖性SIPS细胞的细胞类型、出现时间和空间分布（核心区、半暗带、对侧区）。'));
children.push(bullet('透射电镜观察半暗带细胞的线粒体形态（铁死亡典型皱缩 vs 衰老细胞特征）。'));
children.push(bullet('铁含量比色法和/或普鲁士蓝染色检测组织铁水平。'));

children.push(pNoIndent([
  new TextRun({ text: '2. 4-HNE-p53-SLC7A11分子轴的验证', size: 24, font: '宋体', bold: true })
]));
children.push(bullet('检测不同时间点半暗带组织中4-HNE含量、p53表达与核转位、p21/p16水平、SLC7A11表达及SASP因子（IL-6、IL-1\u03b2、TNF-\u03b1）的时序变化。'));
children.push(bullet('免疫沉淀（IP）+抗4-HNE抗体/抗羰基化抗体（DNP）检测p53的羰基化水平。技术难点与对策：4-HNE修饰的检测在组织样本中由于4-HNE的高反应性和不稳定性，容易产生假阴性。为克服这一问题，所有组织样本在取材后将立即用含5 mM DTT和蛋白酶抑制剂的裂解液处理，并在氮气保护下进行后续操作。同时设置阳性对照（4-HNE处理的细胞裂解液），确保IP/IB体系的灵敏度。'));
children.push(bullet('构建AAV-shACSL4载体，通过脑立体定位注射下调半暗带ACSL4表达，观察铁死亡和SIPS标志物的变化，验证ACSL4作为铁死亡核心执行分子的作用。AAV-shACSL4载体已于前期完成设计，将在项目启动后立即委托公司包装，预计第4个月获得病毒，第6个月完成立体定位注射，第10个月完成全部功能验证。'));

children.push(h2('研究内容二：BCP抗铁依赖性SIPS的细胞机制研究'));

children.push(pNoIndent([
  new TextRun({ text: '1. 体外铁依赖性SIPS模型的建立与BCP药效评价', size: 24, font: '宋体', bold: true })
]));
children.push(bullet('使用低剂量铁死亡诱导剂Erastin（0.5 \u03bcM）或氧糖剥夺/复氧（OGD/R，2 h OGD + 24 h复氧）处理原代皮层神经元和星形胶质细胞，诱导亚致死量铁死亡压力，观察是否出现SIPS表型（SA-\u03b2-gal阳性率升高、p21/p16上调、SASP分泌、增殖能力下降，但细胞死亡率<15%）。'));
children.push(bullet('设置BCP不同浓度组（1、10、50 \u03bcM），观察其对铁依赖性SIPS的干预效果。以铁死亡抑制剂Liproxstatin-1（200 nM）和铁螯合剂DFO（100 \u03bcM）为工具对照。'));

children.push(pNoIndent([
  new TextRun({ text: '2. Nrf2依赖性的验证', size: 24, font: '宋体', bold: true })
]));
children.push(bullet('在Nrf2过表达（质粒转染）和Nrf2沉默（siRNA或抑制剂ML385）的细胞中，观察BCP的抗SIPS效应是否依赖Nrf2通路。'));
children.push(bullet('检测Nrf2核转位、下游靶基因（GPX4、HO-1、FTH1、SLC7A11）的mRNA和蛋白水平变化。'));
children.push(bullet('验证BCP对Keap1-Nrf2复合物的调控：分子对接预测BCP与Keap1疏水口袋的结合模式（PDB: 4IQK）；细胞热位移分析（CETSA）作为探索性实验验证BCP与Keap1的潜在结合。若CETSA结果不理想，采用分子对接+点突变验证的备用策略\u2014\u2014预测BCP与Keap1疏水口袋的关键结合残基，通过点突变破坏结合位点，观察BCP是否还能激活Nrf2，从而间接验证BCP-Keap1的相互作用。'));

children.push(pNoIndent([
  new TextRun({ text: '3. BCP对4-HNE-p53-SLC7A11分子轴的调控', size: 24, font: '宋体', bold: true })
]));
children.push(bullet('核心检测：BCP处理后，细胞内4-HNE含量、p53羰基化水平、p53核转位、SLC7A11表达的时序变化。'));
children.push(bullet('通路验证：验证BCP\u2192Nrf2\u2192GPX4\u2191\u21924-HNE\u2193\u2192p53活化\u2193\u2192SLC7A11\u2191\u2192进一步抑制铁死亡的正反馈解除机制。'));
children.push(bullet('延伸实验（条件允许时开展）：在p53敲低/过表达细胞中，进一步验证BCP的抗SIPS效应是否依赖p53通路；通过点突变验证p53羰基化位点对其功能的影响。'));

children.push(h2('研究内容三：BCP改善CIRI远期预后的整体药效与机制验证'));

children.push(pNoIndent([
  new TextRun({ text: '1. BCP对MCAO/R模型的整体药效评价', size: 24, font: '宋体', bold: true })
]));
children.push(boldLabel('实验分组：', '假手术组、模型组、BCP中剂量组（204 mg/kg）、BCP高剂量组（408 mg/kg）、Liproxstatin-1阳性组（10 mg/kg）、BCP+ML385组（30 mg/kg），每组12只。'));
children.push(bullet('给药方案：再灌注后立即腹腔注射首次给药，随后每日灌胃给药，连续14天。'));
children.push(bullet('短期评价（1-3 d）：TTC染色测梗死体积、脑水肿测定、mNSS神经功能评分。'));
children.push(bullet('远期评价（28 d）：足误实验、转棒实验（运动功能）、Morris水迷宫（认知功能）。'));

children.push(pNoIndent([
  new TextRun({ text: '2. Nrf2依赖性的在体验证', size: 24, font: '宋体', bold: true })
]));
children.push(bullet('使用Nrf2基因敲除小鼠（C57BL/6J背景），设置野生型+BCP组、Nrf2\u207b/\u207b+BCP组、Nrf2\u207b/\u207b+溶媒组，比较梗死体积和远期功能恢复，验证BCP的神经保护效应是否依赖Nrf2。'));
children.push(bullet('对各组28 d脑组织进行SA-\u03b2-gal染色、p53羰基化检测、衰老标志物和SASP因子检测，验证BCP对铁依赖性SIPS的在体抑制效应。'));

children.push(pNoIndent([
  new TextRun({ text: '3. 桂艾挥发油与BCP单体的药效初步比较（延伸目标，条件允许时开展）', size: 24, font: '宋体', bold: true })
]));
children.push(bullet('在细胞水平（铁依赖性SIPS模型），头对头比较桂艾挥发油（按BCP含量折算等剂量）与BCP单体的药效差异。'));
children.push(bullet('在动物水平（MCAO/R模型），设置桂艾挥发油组与等剂量BCP单体组，初步比较梗死体积、神经功能评分和SIPS标志物的差异。'));
children.push(bullet('评估是否存在药效增益现象，为后续深入研究协同机制提供线索。'));

children.push(h1('四、拟解决的关键科学问题'));
children.push(bullet('CIRI半暗带中铁依赖性SIPS的存在性及其4-HNE-p53-SLC7A11分子轴的验证。'));
children.push(bullet('BCP通过Nrf2通路阻断铁依赖性SIPS、改善CIRI远期预后的机制阐明。'));
children.push(bullet('桂艾挥发油多成分药效增益现象的初步探索。'));

children.push(h1('五、研究方案'));

children.push(h2('5.1 实验材料'));
children.push(bullet('实验动物：SPF级C57BL/6J小鼠（22-25 g，雄性），购自北京维通利华实验动物技术有限公司；Nrf2基因敲除小鼠（C57BL/6J背景）购自赛业生物。'));
children.push(bullet('药物与试剂：\u03b2-石竹烯（Sigma-Aldrich，纯度\u226598.5%）、桂艾挥发油（本项目提取，经GC-MS鉴定）、Erastin（MCE）、RSL3（MCE）、Liproxstatin-1（MCE）、DFO（Sigma）、ML385（MCE）、4-HNE（Sigma）。'));
children.push(bullet('主要抗体：抗-GPX4、抗-ACSL4、抗-FTH1、抗-TFR1、抗-p53、抗-p21、抗-p16、抗-\u03b3H2AX、抗-4-HNE、抗-DNP、抗-Nrf2、抗-Keap1、抗-HO-1、抗-SLC7A11、抗-NeuN、抗-GFAP、抗-Iba-1、抗-\u03b2-actin等。'));
children.push(bullet('道地药材：广西道地桂艾（Artemisia argyi）采自广西药用植物园艾叶GAP种植基地，经广西中医药大学中药鉴定教研室鉴定。'));

children.push(h2('5.2 主要实验方法'));
children.push(bullet('小鼠MCAO/R模型制备：改良线栓法制备小鼠大脑中动脉阻塞/再灌注模型，缺血60 min后再灌注。'));
children.push(bullet('原代神经细胞培养：新生24 h内C57BL/6J小鼠皮层神经元和星形胶质细胞原代培养。'));
children.push(bullet('OGD/R模型：缺氧缺糖2 h后复氧复糖24 h。'));
children.push(bullet('细胞活力检测：CCK-8法。'));
children.push(bullet('细胞内Fe\u00b2\u007a检测：Phen Green SK探针流式细胞术。'));
children.push(bullet('脂质过氧化检测：C11-BODIPY 581/591探针流式细胞术。'));
children.push(bullet('GSH/GSSG测定：比色法。'));
children.push(bullet('MDA含量测定：TBA法。'));
children.push(bullet('SA-\u03b2-gal染色：细胞衰老\u03b2-半乳糖苷酶染色试剂盒。'));
children.push(bullet('免疫荧光共定位：冰冻切片4%多聚甲醛固定，一抗4\u2103孵育过夜，荧光二抗室温孵育1 h，DAPI染核，激光共聚焦显微镜观察。'));
children.push(bullet('透射电镜：脑组织2.5%戊二醛固定，梯度脱水，环氧树脂包埋，超薄切片，醋酸铀-柠檬酸铅双重染色，透射电镜观察。'));
children.push(bullet('Western Blot：RIPA裂解提取总蛋白，BCA定量，SDS-PAGE电泳，转膜，封闭，一抗4\u2103孵育过夜，二抗室温孵育1 h，ECL化学发光显影。'));
children.push(bullet('qRT-PCR：Trizol提取总RNA，反转录，SYBR Green荧光定量PCR检测mRNA水平。'));
children.push(bullet('ELISA：组织匀浆或细胞上清中IL-6、IL-1\u03b2、TNF-\u03b1含量检测。'));
children.push(bullet('脑立体定位注射：小鼠脑立体定位仪，前囟后0.5 mm、旁开3.0 mm、深3.5 mm，微量注射泵匀速注射AAV。'));
children.push(bullet('Morris水迷宫：定位航行实验5天（每天4次），第6天空间探索实验，记录逃逸潜伏期和平台穿越次数。'));
children.push(bullet('转棒实验：小鼠置于转棒仪上，转速从4 rpm加速至40 rpm，记录跌落潜伏期。'));
children.push(bullet('足误实验：小鼠在水平栅格上行走，记录前足误伸入栅格的次数。'));

children.push(h2('5.3 统计学分析'));
children.push(p('所有数据采用GraphPad Prism 9.0和/或SPSS 26.0软件进行统计分析。计量资料以均数\u00b1标准差（x\u0304 \u00b1 s）表示。两组间比较采用独立样本t检验（满足正态分布和方差齐性）或Mann-Whitney U检验（不满足参数检验条件）。多组间比较采用单因素方差分析（one-way ANOVA），组间两两比较采用Tukey法或Dunnett法；重复测量数据采用重复测量方差分析。相关性分析采用Pearson相关（参数）或Spearman秩相关（非参数）。P < 0.05认为差异具有统计学意义。多重比较采用Benjamini-Hochberg校正。'));

children.push(h1('六、技术路线'));

const techRows = [
  [
    { text: '上层\n现象验证\n（第1年）', fill: 'E8F5E9' },
    { text: 'CIRI半暗带铁依赖性SIPS的存在性验证（核心目标）', fill: 'E8F5E9' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: '\u2193 时间：24 h / 3 d / 7 d / 28 d\n\u2193 空间：核心区 / 半暗带 / 对侧区\n\u2193 细胞：神经元 / 星形胶质细胞 / 小胶质细胞', fill: 'FFFFFF' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: '关键技术：免疫荧光共定位 | SA-\u03b2-gal染色 | 普鲁士蓝染色 | 透射电镜 | Western Blot', fill: 'FFF3E0' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: '\u2193\n确认铁依赖性SIPS的存在与时空特征 + ACSL4功能验证', fill: 'FFFFFF' }
  ],
  [
    { text: '中层\n机制解析\n（第2年）', fill: 'E3F2FD' },
    { text: '4-HNE-p53-SLC7A11分子轴 + BCP调控机制（核心目标）', fill: 'E3F2FD' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: '\u2193 分子轴验证 \u2193            \u2193 BCP调控机制 \u2193', fill: 'FFFFFF' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: '4-HNE\u2192p53活化\u2192SLC7A11\u2193\nIP/IB | 羰基化检测 | 报告基因 | 羰基化抑制剂\n\n            BCP\u2192Nrf2\u2192GPX4\u2191\u21924-HNE\u2193\n            Nrf2抑制剂/敲低 | 分子对接 | CETSA（探索）', fill: 'FFFFFF' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: '\u2193\n阐明BCP通过Nrf2通路阻断铁依赖性SIPS的分子机制', fill: 'FFFFFF' }
  ],
  [
    { text: '下层\n药效验证\n（第3年）', fill: 'FCE4EC' },
    { text: 'BCP改善CIRI远期预后的整体药效（核心）+ 桂艾药效增益探索（延伸）', fill: 'FCE4EC' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: '\u2193 整体药效 \u2193            \u2193 Nrf2依赖 \u2193            \u2193 桂艾比较（延伸）\u2193', fill: 'FFFFFF' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: 'MCAO/R模型\nTTC | mNSS | 转棒 | 足误 | 水迷宫\nSA-\u03b2-gal | p53羰基化 | SASP\n\nNrf2\u207b/\u207b小鼠\n验证Nrf2依赖性\n\n            桂艾挥发油 vs BCP单体            \n            （条件允许时开展）', fill: 'FFFFFF' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: '\u2193\n最终结论：BCP通过Nrf2通路抑制铁依赖性SIPS，改善CIRI远期预后', fill: 'F3E5F5' }
  ],
];

const techTable = new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [1800, 7560],
  rows: techRows.map(row => new TableRow({
    children: row.map((c, i) => {
      const cellChildren = c.text.split('\n').map(line =>
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 40, line: 300 },
          children: [new TextRun({ text: line, size: 20, font: '宋体', bold: i === 0 })]
        })
      );
      return new TableCell({
        borders: { style: BorderStyle.NONE, size: 0 },
        width: { size: i === 0 ? 1800 : 7560, type: WidthType.DXA },
        shading: { fill: c.fill, type: ShadingType.CLEAR },
        margins: { top: 60, bottom: 60, left: 80, right: 80 },
        children: cellChildren
      });
    })
  }))
});

children.push(techTable);

children.push(h1('七、可行性分析'));

children.push(h2('7.1 理论可行性'));
children.push(p('铁死亡和细胞衰老是当前CIRI研究的两个热点领域，其交互作用（铁衰老）是前沿交叉方向。本项目组前期通过4个脑缺血数据集的整合分析，已证实铁衰老转录特征在脑缺血中的稳健存在：铁衰老评分在4个数据集中均表现出最大的疾病-对照效应量，五基因CIRI-铁衰老签名在3个独立数据集中得到验证（Spearman \u03c1 = 0.56-0.88）。网络药理学分析预测了BCP靶点与铁死亡调控网络的高度拓扑关联（超几何检验P = 2.48 \u00d7 10\u207b\u2074\u00b3），并初步确定SAT1-Nrf2轴为关键入口节点。最短路径分析显示SAT1通过NFE2L2桥接至铁死亡调控网络，平均距离2.84步，为后续实验验证提供了坚实的理论基础。'));

children.push(h2('7.2 材料可行性'));
children.push(p('广西道地桂艾采自广西药用植物园艾叶GAP种植基地，来源明确，活性成分分离平台成熟。BCP标准品为Sigma-Aldrich商品化试剂，纯度\u226598.5%。Nrf2基因敲除小鼠已商用可购。原代神经细胞培养、MCAO/R模型等所需动物和细胞材料均可稳定获取。STRING v12.0、FerrDb v2等数据库公开可用，网药分析平台稳定运行。'));

children.push(h2('7.3 技术可行性'));
children.push(p('项目组已熟练掌握小鼠MCAO/R模型制备、原代神经细胞培养、OGD/R模型、免疫荧光共定位、Western Blot、qRT-PCR、SA-\u03b2-gal染色、透射电镜、行为学检测等全部实验技术。铁死亡和细胞衰老的表征方法均为成熟技术，有大量文献支持和前期预实验基础。网药分析方面，已建立包括ssGSEA评分、LASSO特征选择、PPI拓扑分析、功能富集、WGCNA共表达网络验证等在内的完整分析流程，具备持续的数据挖掘和分析能力。'));

children.push(h2('7.4 前期工作基础'));
children.push(p('本团队前期已在脑缺血、铁死亡和天然药物神经保护领域积累了扎实的研究基础。已完成4个脑缺血数据集的整合分析，建立了CIRI-铁衰老五基因签名（SAT1、EBF3、KLF6、LIFR、CD74），验证了BCP靶点与铁死亡调控网络的高度拓扑富集（超几何检验P = 2.48 \u00d7 10\u207b\u2074\u00b3）。构建了包含311个节点、1,867条边的核心PPI子网，识别出8个功能模块，完成了免疫浸润和炎症因子相关性分析，以及WGCNA共表达网络验证。Hu等[3]已证实BCP通过Nrf2/HO-1通路减轻大鼠脑缺血再灌注损伤，为本项目的延伸研究提供了直接的前期支持。'));

children.push(h1('八、特色与创新之处'));

children.push(h2('8.1 理论创新'));
children.push(p('率先提出并系统验证CIRI半暗带中缺血诱导的铁依赖性SIPS新假说，将铁死亡的急性损伤与细胞衰老的慢性损害通过4-HNE-p53-SLC7A11分子轴有机联系，为理解缺血性脑损伤的慢性化机制提供新视角。与经典ferro-aging（自然衰老模型）的概念在时间尺度和触发因素上明确区分，体现了概念的精确性。前期多数据集整合的网药研究为这一假说提供了转录组水平的间接证据：铁衰老评分与缺血性损伤高度关联，且在急性期即达高峰，支持铁依赖性SIPS与急性铁死亡并行启动的模式。'));

children.push(h2('8.2 机制创新'));
children.push(p('揭示壮瑶药桂艾活性成分\u03b2-石竹烯通过激活Nrf2通路协同抑制铁死亡与铁依赖性SIPS的分子机制，首次阐明桂艾通龙路火路、除毒邪功效的铁衰老干预科学内涵。SAT1-Nrf2桥接机制的发现，为天然小分子同时靶向急性死亡和慢性衰老提供了新的作用范式。BCP作为CB2受体选择性激动剂，其通过Nrf2通路发挥神经保护作用已有文献支持[3]，但将其与铁依赖性SIPS相联系，并系统解析4-HNE-p53-SLC7A11分子轴的调控，具有显著的机制创新性。'));

children.push(h2('8.3 模式创新'));
children.push(p('构建基于壮瑶医药理论的道地药材-功效-核心成分-铁衰老靶点-信号通路精准整合研究模式，将民族医药传统经验与现代前沿生物学（铁死亡、细胞衰老）有机结合，为民族药现代化研究提供了可复制的范例。前期网药分析的系统布局（多数据集验证、机器学习筛选、PPI拓扑分析、功能模块解析、免疫浸润关联、WGCNA验证），确保了后续实验的靶向性和成功率，体现了从计算预测到实验验证的转化医学研究思路。'));

children.push(h1('九、年度研究计划及预期研究成果'));

children.push(h2('9.1 第一年'));
children.push(bullet('完成MCAO/R模型的建立和铁依赖性SIPS的时空定位（4个时间点）。'));
children.push(bullet('完成4-HNE-p53-SLC7A11分子轴的时序变化检测。'));
children.push(bullet('完成AAV-shACSL4的立体定位注射和功能验证。AAV-shACSL4载体已于前期完成设计，项目启动后立即委托公司包装。'));
children.push(bullet('完成桂艾挥发油的提取和GC-MS成分鉴定。'));
children.push(bullet('预期成果：发表SCI论文1篇，申请发明专利1项。'));

children.push(h2('9.2 第二年'));
children.push(bullet('建立体外铁依赖性SIPS细胞模型，完成BCP药效评价。'));
children.push(bullet('完成Nrf2依赖性的细胞水平验证（过表达/沉默）。'));
children.push(bullet('完成BCP对4-HNE-p53-SLC7A11分子轴调控的系统检测。'));
children.push(bullet('完成分子对接和CETSA探索性实验。'));
children.push(bullet('预期成果：发表SCI论文1-2篇。'));

children.push(h2('9.3 第三年'));
children.push(bullet('完成BCP对MCAO/R模型的整体药效评价（短期+远期）。'));
children.push(bullet('完成Nrf2基因敲除小鼠的在体验证实验。'));
children.push(bullet('完成桂艾挥发油与BCP单体的药效初步比较（条件允许时）。'));
children.push(bullet('数据整理、论文撰写、项目结题。'));
children.push(bullet('预期成果：发表高影响力SCI论文1篇，培养研究生2-3名。'));

const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: '宋体', size: 24 }
      }
    },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 32, bold: true, font: '黑体' },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 28, bold: true, font: '黑体' },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 24, bold: true, font: '黑体' },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } }
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
      default: new Header({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: '\u03b2-石竹烯靶向Nrf2抑制铁依赖性SIPS改善CIRI研究', size: 18, italics: true, font: '宋体' })] })] })
    },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })] })] })
    },
    children
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync('D:/铁衰老 绝不重蹈覆辙/标书_最终版_广西道地壮药桂艾BCP靶向Nrf2抑制铁依赖性SIPS改善CIRI.docx', buffer);
  console.log('Done: proposal generated');
  console.log('File: D:/铁衰老 绝不重蹈覆辙/标书_最终版_广西道地壮药桂艾BCP靶向Nrf2抑制铁依赖性SIPS改善CIRI.docx');
});
