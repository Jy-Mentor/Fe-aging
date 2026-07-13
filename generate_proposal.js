const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Header, Footer, AlignmentType, LevelFormat,
        TableOfContents, HeadingLevel, BorderStyle, WidthType, ShadingType,
        VerticalAlign, PageNumber, PageBreak, TabStopPosition, TabStopType,
        ImageRun } = require('docx');
const fs = require('fs');

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

function heading1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)],
    spacing: { before: 360, after: 240 } });
}
function heading2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)],
    spacing: { before: 280, after: 200 } });
}
function heading3(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(text)],
    spacing: { before: 200, after: 160 } });
}
function para(text, opts = {}) {
  const runs = [];
  if (typeof text === 'string') {
    runs.push(new TextRun({ text, ...opts }));
  } else {
    text.forEach(t => runs.push(new TextRun(t)));
  }
  return new Paragraph({ children: runs, spacing: { line: 360, after: 120 },
    indent: { firstLine: 480 } });
}
function paraNoIndent(text, opts = {}) {
  const runs = [];
  if (typeof text === 'string') {
    runs.push(new TextRun({ text, ...opts }));
  } else {
    text.forEach(t => runs.push(new TextRun(t)));
  }
  return new Paragraph({ children: runs, spacing: { line: 360, after: 120 } });
}
function bullet(text, level = 0) {
  const runs = [];
  if (typeof text === 'string') {
    runs.push(new TextRun(text));
  } else {
    text.forEach(t => runs.push(new TextRun(t)));
  }
  return new Paragraph({ numbering: { reference: "bullets", level },
    children: runs, spacing: { line: 340, after: 80 },
    indent: { left: 720 + level * 360, hanging: 360 } });
}

const children = [];

// ===== 封面 =====
children.push(new Paragraph({ spacing: { before: 2400 }, children: [] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 },
  children: [new TextRun({ text: "国家自然科学基金申请书", size: 48, bold: true, font: "黑体" })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
  children: [new TextRun({ text: "（修订版）", size: 32, font: "宋体" })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 },
  children: [new TextRun({ text: "广西道地壮药桂艾活性成分β-石竹烯靶向Nrf2通路", size: 28, bold: true, font: "黑体" })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
  children: [new TextRun({ text: "抑制缺血诱导的铁依赖性SIPS改善脑缺血再灌注损伤的机制研究", size: 28, bold: true, font: "黑体" })] }));
children.push(new Paragraph({ spacing: { before: 800 }, children: [] }));
children.push(new PageBreak());

// ===== 目录 =====
children.push(new TableOfContents("目  录", { hyperlink: true, headingStyleRange: "1-3" }));
children.push(new PageBreak());

// ===== 正文开始 =====

// 一、立项依据与研究内容
children.push(heading1("（一）立项依据与研究内容"));

// 1. 项目的立项依据
children.push(heading2("1. 项目的立项依据"));
children.push(heading3("（1）国内外研究进展"));

children.push(heading3("1）现代医学对脑缺血再灌注损伤的认识"));

children.push(para("脑缺血再灌注损伤（cerebral ischemia/reperfusion injury, CIRI）是指缺血脑组织恢复血流灌注后，损伤反而进一步加重的病理现象。急性缺血性脑卒中（acute ischemic stroke, AIS）是我国成年人致死、致残的首位病因，具有高发病率、高致残率、高死亡率和高复发率的特点。据《中国卒中报告2023》数据显示，我国卒中患病率为1114.8/10万，发病率为246.8/10万，死亡率为114.8/10万，给家庭和社会带来沉重负担。"));

children.push(para("目前，静脉注射重组组织型纤溶酶原激活剂（recombinant tissue plasminogen activator, rt-PA）和替奈普酶（tenecteplase, TNK）是获美国FDA批准的急性缺血性脑卒中药物治疗手段，血管内机械取栓也是大血管闭塞的标准治疗方案。然而，rt-PA/TNK的标准治疗时间窗仅为发病后4.5小时，极为狭窄，且伴有出血性转化风险；即便成功实现血管再通，相当一部分患者仍出现\"无复流\"现象或再灌注损伤，预后不佳。因此，深入揭示CIRI病理机制并寻找新的神经保护靶点与药物，具有重要的临床意义和社会价值。"));

children.push(para([
  { text: "① 经典损伤机制", bold: true }
]));

children.push(para("CIRI的病理机制复杂，涉及多个相互交织的级联反应。经典损伤机制主要包括以下三方面：第一，兴奋性氨基酸毒性。缺血导致谷氨酸大量释放，过度激活N-甲基-D-天冬氨酸（NMDA）受体，引起钙离子内流和神经元兴奋性损伤，是缺血早期神经元损伤的主要机制。第二，钙超载。细胞内钙离子浓度异常升高，激活钙依赖性酶类，导致细胞骨架降解和膜结构破坏，并触发线粒体功能障碍，促进活性氧（ROS）产生和细胞死亡。第三，氧化应激。再灌注期间氧供恢复，线粒体电子传递链产生大量ROS，超出内源性抗氧化系统的清除能力，引发脂质过氧化、蛋白质氧化和DNA损伤。上述经典机制相互关联，共同构成CIRI的病理基础。"));

children.push(para([
  { text: "② 新型损伤机制——铁死亡是CIRI后神经元死亡的关键方式", bold: true }
]));

children.push(para("近年来，研究先后证实坏死性凋亡、铁死亡、铜死亡等多种新型程序性死亡方式参与CIRI进程。其中，铁死亡（ferroptosis）因其与铁代谢、氧化应激的固有联系，在CIRI中尤为引人关注。铁死亡是一种铁依赖性脂质过氧化驱动的细胞死亡方式，与谷胱甘肽过氧化物酶4（GPX4）失活和脂质ROS积累密切相关[1]。其核心特征为：细胞内铁离子蓄积、多不饱和脂肪酸（PUFA）过氧化产物堆积、线粒体皱缩及膜密度增高。"));

children.push(para("大量研究证实铁死亡是CIRI后神经元死亡的关键方式之一。在MCAO/R模型中，缺血再灌注后铁死亡标志物（如ACSL4、PTGS2、4-HNE）呈时间依赖性上调，而GPX4、SLC7A11等抗铁死亡蛋白表达下降，同时伴随铁离子蓄积和脂质过氧化增强[2]。铁死亡抑制剂（如Ferrostatin-1、Liproxstatin-1、去铁胺）可显著减小梗死体积、改善神经功能缺损，进一步证实铁死亡在CIRI中的重要作用。"));

children.push(para([
  { text: "③ CIRI与细胞衰老——从急性死亡到慢性损伤的视角拓展", bold: true }
]));

children.push(para("细胞衰老（cellular senescence）是指细胞在遭受应激损伤后进入的一种永久性细胞周期停滞状态，伴随衰老相关分泌表型（senescence-associated secretory phenotype, SASP）的分泌。传统观点认为，衰老主要与机体老化相关，但近年研究发现，急性损伤后细胞也可发生应激诱导的早熟性衰老（stress-induced premature senescence, SIPS）[3]。SIPS与复制性衰老（replicative senescence）既有共同特征（细胞周期停滞、SASP分泌、SA-β-gal阳性），又存在差异：SIPS由急性应激触发，发生时间短（数小时至数天），而复制性衰老是端粒缩短导致的慢性过程（数月至数年）。SIPS在组织修复和病理损伤中扮演重要角色，其启动机制包括DNA损伤、氧化应激、线粒体功能障碍等。"));

children.push(para("在CIRI领域，越来越多的证据表明SIPS参与了缺血后脑损伤的病理进程。Baixauli-Martín等[4]对实验性缺血性卒中的细胞衰老标志物进行了系统的时空表征，发现在小鼠MCAO模型中，缺血半暗带区域的神经元、星形胶质细胞和小胶质细胞均出现衰老标志物（p16INK4a、p21CIP1、SA-β-gal）的上调，且衰老细胞的出现与神经功能缺损的严重程度相关。SASP因子（如IL-6、TNF-α、MMP-3、CXCL10）在缺血后持续释放，形成慢性炎症微环境，阻碍神经发生和突触重塑，影响远期功能恢复。"));

children.push(para("然而，目前关于CIRI中SIPS的启动机制尚不完全清楚。缺血再灌注过程中产生的大量ROS、DNA损伤、铁代谢紊乱等因素，均可能是诱导SIPS的上游驱动因素。其中，铁死亡与SIPS之间是否存在因果关联，铁死亡是否作为上游驱动因子诱发SIPS，进而形成\"铁死亡-衰老\"恶性循环，是本项目拟重点探讨的科学问题。"));

children.push(heading3("2）铁衰老概述及其在CIRI中的病理意义"));

children.push(para([
  { text: "① 铁衰老（ferro-aging）概念与本项目的界定：", bold: true }
]));

children.push(para("铁死亡与细胞衰老并非两个孤立的病理过程。近年研究揭示，二者之间存在密切的交互作用，共同构成一个\"铁死亡驱动细胞衰老\"的病理轴。2026年，Liu等[5]在Cell Metabolism发表的灵长类动物研究首次系统定义了ferro-aging这一概念：铁过载通过ACSL4介导的脂质过氧化，驱动细胞进入衰老状态，形成一个从铁代谢紊乱到脂质过氧化再到细胞衰老的完整级联反应。需要指出的是，Liu等的研究是在灵长类自然衰老和早衰模型中进行的，时间尺度为月-年级，属于慢性衰老过程。"));

children.push(para([
  { text: "本项目中\"铁衰老\"的界定：", bold: true },
  "本项目中\"铁衰老\"特指缺血再灌注后由亚致死铁死亡压力驱动的应激诱导早熟性衰老（即缺血诱导的铁依赖性SIPS），与Liu等[5]在自然衰老模型中描述的ferro-aging存在时间尺度和触发因素上的差异——前者由急性缺血再灌注损伤触发（小时-天级），后者是自然衰老的慢性过程（月-年级）。尽管如此，二者共享ACSL4-4-HNE-p53的核心分子通路，均以铁依赖性脂质过氧化为上游驱动力。本项目的研究重点是探索这一分子轴在急性脑缺血再灌注损伤中的病理作用。"
]));

children.push(para("该研究在多种人类细胞衰老模型（复制性衰老、HGPS、Werner综合征）以及自然衰老的人和非人灵长类组织中，均观察到铁离子（Fe²⁺）的显著蓄积，同时伴随ACSL4、COX2等脂质过氧化关键酶的上调和MDA、4-HNE等脂质过氧化终产物的增加。功能实验证实，铁过载可直接诱导人间充质干细胞、肝细胞和神经元发生衰老，表现为SA-β-gal活性增加、p21上调、Lamin B1丢失等典型衰老表型；而敲低ACSL4则可逆转铁过载诱导的衰老表型。"));

children.push(para("铁衰老的核心机制可概括为：亚致死剂量的铁死亡刺激（如低剂量铁离子、轻度脂质过氧化）不足以直接导致细胞死亡，但通过持续的铁依赖性氧化应激，引发DNA损伤响应（DDR，如γH2AX焦点形成），激活p53/p21CIP1和p16INK4a/Rb通路，诱导细胞进入早熟性衰老状态。其典型特征是：细胞呈现衰老标志（SA-β-gal阳性，p21/p16上调，SASP分泌），同时伴有铁蓄积、脂质过氧化水平升高，但细胞并未发生典型的坏死或凋亡。"));

children.push(para([
  { text: "4-HNE-p53通路——连接铁死亡与衰老的潜在分子开关：", bold: true },
  "在铁死亡过程中，脂质过氧化产生的活性醛类（尤其是4-羟基壬烯醛，4-HNE）可与蛋白质的半胱氨酸、组氨酸、赖氨酸残基发生共价结合，形成蛋白质羰基化修饰。Monroe等[6]在Aging Cell发表的研究证实，4-HNE等脂质过氧化产物可诱导人成纤维细胞和小鼠脂肪干细胞发生衰老，伴随γH2AX焦点积累、p53信号增强、p21表达上调及SASP分泌。4-HNE作为高度活泼的亲电性醛类，可通过羰基化修饰调节多种信号蛋白的功能。已有研究表明，p53可被4-HNE修饰而增强其稳定性和转录活性；活化的p53一方面通过上调p21诱导细胞周期停滞，另一方面可转录抑制SLC7A11（System Xc⁻的关键亚基），进一步削弱细胞的抗氧化防御能力，形成\"脂质过氧化→p53活化→SLC7A11↓→更多脂质过氧化\"的正反馈环路。需要指出的是，4-HNE对p53的直接羰基化修饰及其在铁死亡-SIPS交互中的特异性作用，目前尚缺乏直接的实验证据，是本项目拟重点验证的科学问题之一。"
]));

children.push(para([
  { text: "② CIRI半暗带中\"缺血诱导的铁依赖性SIPS\"假说", bold: true }
]));

children.push(para("基于上述研究进展，结合CIRI的病理特点，我们提出：缺血半暗带中存在\"缺血诱导的铁依赖性SIPS\"（ischemia-induced iron-dependent SIPS）现象——即亚致死量的铁死亡压力通过ACSL4介导的脂质过氧化，产生4-HNE等活性醛类，羰基化修饰p53，激活p21/p16通路，诱导神经细胞进入SIPS状态并分泌SASP，形成\"铁死亡→SIPS→SASP→更多铁死亡\"的恶性循环。"));

children.push(para("具体而言，在CIRI的缺血核心区，神经元遭受严重缺血缺氧，快速发生铁死亡和坏死；而在缺血半暗带（penumbra），神经细胞遭受\"亚致死量\"的铁死亡压力——再灌注带来的氧自由基、谷氨酸兴奋毒耗竭GSH、游离铁释放等因素共同构成轻度但持续的氧化应激和铁代谢紊乱，尚不足以立即触发细胞死亡，但足以驱动细胞进入铁依赖性SIPS状态。"));

children.push(para("这些\"铁衰老\"细胞具有以下病理危害："));

children.push(bullet("第一，形成持久的衰老细胞灶，阻碍半暗带恢复。SIPS细胞一旦形成，可长期存活并持续分泌SASP因子，形成有毒的微环境，抑制神经干细胞增殖分化、阻碍突触重塑、促进胶质瘢痕形成，使半暗带组织无法向正常组织转化，影响远期神经功能恢复。"));
children.push(bullet("第二，SASP反推铁死亡，形成恶性循环。SASP中的促炎因子（IL-6、TNF-α）可上调TFR1，增加铁摄取；同时炎症可诱导ACSL4表达，增强脂质过氧化，进一步促进周围细胞发生铁死亡或进入SIPS状态，使损伤范围逐步扩大。"));
children.push(bullet("第三，破坏神经血管单元完整性。半暗带中的衰老内皮细胞、星形胶质细胞可破坏血脑屏障完整性，增加血管源性脑水肿风险；衰老小胶质细胞则持续释放促炎因子，加重神经炎症。"));

children.push(para("目前，关于CIRI中铁衰老的研究尚处于起步阶段。尽管已有研究分别证实铁死亡和SIPS各自参与CIRI，但二者之间的因果关联——即铁死亡是否驱动SIPS、铁依赖性SIPS在CIRI半暗带中是否真实存在、其时空分布规律如何——尚缺乏系统的实验证据。阐明这一问题，将为理解CIRI从急性损伤向慢性损伤转化的机制提供全新视角，并为从\"铁衰老\"这一新靶点出发寻找神经保护策略奠定理论基础。"));

children.push(heading3("3）桂艾与活性成分β-石竹烯：CIRI干预新策略"));

children.push(para("鉴于铁衰老涉及铁代谢异常、氧化应激、炎症及衰老信号等多重通路，理想的干预药物应具备多靶点协同作用。天然小分子化合物在此方面独具优势。艾草作为传统常用中药，在神经保护方面显示出潜力，尤其是壮药桂艾，其活性成分β-石竹烯（β-caryophyllene, BCP）含量显著高于其他种类的艾草，且已知具有抗炎、抗氧化、神经保护等活性，成为靶向铁衰老干预CIRI的极佳候选。"));

children.push(para([
  { text: "① 艾草与桂艾", bold: true }
]));

children.push(para("艾草（Artemisia argyi H.Lév. & Vaniot）性温、味苦辛，具温经通络、散寒止痛之效，《本草纲目》记载其可\"通十二经，具回阳、理气血、逐湿寒\"之功。在壮瑶医药理论中，艾草（壮语称\"挨\"，瑶语称\"各艾\"）为\"通龙路火路、除风毒寒毒、逐湿邪\"之要药，常用于麻痹、头痛等脑病的熏洗或内服治疗。其\"解毒除蛊\"功效正对应现代医学的清除自由基、抗炎、调节细胞死亡等作用。"));

children.push(para("红脚艾（Artemisia verlotorum），俗称桂艾，为岭南（广西）道地壮药。现代药理研究证实，艾草挥发油富含BCP、1,8-桉叶素、樟脑等多种活性成分，具有明确的抗炎、抗氧化及神经保护作用。针对广西产艾草挥发油成分的系统分析发现，桂艾中BCP相对含量可达18.21%[7]，显著高于普通艾草品种。Guo等[8]通过GC-MS结合电子鼻技术对不同种质资源艾叶挥发油进行化学组成分析和鉴别，证实不同产地艾叶的挥发油组成存在显著差异，其中β-石竹烯是含量最高的倍半萜类成分之一。这一产地差异为桂艾在CIRI治疗中的应用提供了独特的物质基础。"));

children.push(para([
  { text: "② BCP药理性质与神经保护作用", bold: true }
]));

children.push(para("β-石竹烯（BCP）是一种天然双环倍半萜烯，分子式为C₁₅H₂₄，广泛存在于多种植物挥发油中。Gertsch等[9]在PNAS发表的里程碑式研究首次证实，BCP为CB2受体选择性激动剂，对CB1受体无显著亲和力，无中枢精神活性，因此被归类为\"膳食大麻素\"（dietary cannabinoid）。BCP已被美国FDA和欧洲食品安全局批准为食品调味剂，具有良好的安全性。"));

children.push(para("BCP药理活性谱广泛："));

children.push(bullet([
  { text: "抗炎作用：", bold: true }, "BCP通过激活CB2受体抑制脂多糖（LPS）诱导的TNF-α、IL-1β及IL-6等促炎因子释放，减轻炎症细胞浸润；同时可抑制NF-κB信号通路的激活，减少促炎基因转录。在脑缺血模型中，BCP通过抑制HMGB1-TLR4信号通路，显著降低缺血脑组织中TNF-α、IL-6等促炎因子水平[10]。"
]));
children.push(bullet([
  { text: "抗氧化作用：", bold: true }, "BCP可直接清除自由基，增强超氧化物歧化酶（SOD）、谷胱甘肽过氧化物酶（GPx）等抗氧化酶活性。其抗氧化作用的核心机制之一是激活Nrf2/HO-1信号通路，上调下游抗氧化基因的表达[2]。"
]));
children.push(bullet([
  { text: "神经保护作用：", bold: true }, "BCP为高脂溶性倍半萜烯（LogP≈4.5-4.7），分子量小（204.36 g/mol），具备被动扩散穿过血脑屏障（BBB）的理化基础。多项研究表明，BCP可减小MCAO/R模型脑梗死体积、改善神经功能缺损评分、减轻认知障碍、保护白质结构完整性[2,10,11]。"
]));
children.push(bullet([
  { text: "抗铁死亡作用：", bold: true }, "近年研究证实BCP可通过抑制铁死亡发挥组织保护作用。Hu等[2]在Phytomedicine发表的研究系统证实，BCP通过激活Nrf2/HO-1通路抑制MCAO/R大鼠铁死亡，Nrf2抑制剂ML385可逆转其神经保护作用。李尤研究团队在HT1080和H9c2细胞中证实，BCP可抑制RSL3和IKE诱导的铁死亡，降低MDA和PTGS2 mRNA水平，恢复GSH/GSSG比值，保护线粒体结构和功能。"
]));

children.push(para([
  { text: "③ BCP通过Nrf2通路抑制铁衰老的科学假说", bold: true }
]));

children.push(para("基于上述研究进展，我们提出BCP抑制CIRI铁衰老的核心科学假说："));

children.push(para([
  { text: "\"BCP通过激活Nrf2/ARE通路，上调GPX4、HO-1、FTH1等抗铁死亡基因的表达，阻断ACSL4介导的脂质过氧化级联反应，减少4-HNE等脂质过氧化醛类的产生，从而解除4-HNE对p53的羰基化修饰，抑制p53-p21/p16通路的过度激活，在阻断铁死亡的同时遏制铁依赖性SIPS的启动，打破\"铁死亡→SIPS→SASP→更多铁死亡\"的恶性循环，实现对CIRI半暗带的双重保护，改善远期神经功能预后。\"", italics: true }
]));

children.push(para("这一假说的核心分子开关是\"4-HNE→p53羰基化→SLC7A11↓\"这一正反馈环路：铁死亡产生的4-HNE羰基化p53，活化的p53一方面诱导p21介导的细胞周期停滞（衰老启动），另一方面抑制SLC7A11转录（进一步促进铁死亡），形成双向放大效应。BCP通过激活Nrf2增强GPX4活性、减少4-HNE产生，从源头上切断这一恶性循环的驱动力。"));

children.push(para([
  { text: "④ 桂艾整体入药的药效增益——从单体到道地药材的延伸", bold: true }
]));

children.push(para("桂艾富含BCP，但艾草挥发油中的1,8-桉叶素、樟脑等成分本身也具有抗炎抗氧化作用。在明确BCP单体抗铁衰老机制的基础上，进一步比较桂艾挥发油是否因多成分整合而呈现药效增益，可为桂艾整体入药提供实验依据。"));

children.push(para("需要指出的是，多成分协同效应的验证是一个复杂的系统工程，超出了本项目的核心研究范围。本项目将此定位为\"应用延伸\"——在BCP单体机制明确的前提下，初步探索桂艾挥发油的整体药效是否优于等剂量BCP单体，为后续深入研究协同机制奠定基础，而不承诺阐明具体的协同分子机制。"));

children.push(para([
  { text: "⑤ 科学问题的凝练", bold: true }
]));

children.push(para("尽管BCP抑制CIRI铁死亡已有明确证据，但以下三个关键问题仍未阐明，构成本项目的研究出发点："));

children.push(bullet([
  { text: "第一，CIRI半暗带中是否存在\"铁死亡驱动的SIPS\"（铁依赖性SIPS）？其关键分子开关是什么？", bold: true }, " 目前铁死亡和SIPS在CIRI中各自的作用已有报道，但二者之间的因果关联尚不明确。特别是4-HNE-p53羰基化-SLC7A11这一分子开关是否在CIRI中真实存在，尚缺乏实验证据。"
]));
children.push(bullet([
  { text: "第二，BCP能否通过激活Nrf2通路阻断铁依赖性SIPS，从而改善CIRI远期预后？", bold: true }, " BCP抗铁死亡已有证据，但是否能进一步阻断铁死亡驱动的SIPS，实现从\"抗急性死亡\"到\"抗慢性衰老\"的跃升，是本项目拟回答的核心科学问题。"
]));
children.push(bullet([
  { text: "第三，桂艾挥发油与等剂量BCP单体在抗铁衰老效应上是否存在药效增益？", bold: true }, " 本项目将此定位为探索性目标，在BCP单体机制明确的基础上，初步比较桂艾挥发油的整体药效，为道地药材的临床应用提供初步实验依据。"
]));

children.push(para("因此，本项目拟在已有BCP抗铁死亡证据的基础上，以\"4-HNE-p53羰基化\"为核心分子开关，深入探讨BCP通过Nrf2通路抑制铁依赖性SIPS的双重保护机制，为从铁死亡-衰老交互角度防治CIRI提供更完整的理论依据和药物开发新策略。"));

children.push(new PageBreak());

// 2. 项目的研究内容、研究目标，以及拟解决的关键科学问题
children.push(heading2("2. 项目的研究内容、研究目标，以及拟解决的关键科学问题"));

children.push(heading3("（1）研究目标"));

children.push(para([
  { text: "核心目标（必须完成）：", bold: true }
]));
children.push(bullet("证实CIRI半暗带中存在\"铁死亡驱动的应激诱导早熟性衰老\"（铁依赖性SIPS）现象，明确4-HNE-p53羰基化-SLC7A11是其关键分子开关。"));
children.push(bullet("阐明BCP通过激活Nrf2通路抑制铁依赖性SIPS的分子机制，验证\"Nrf2激活→GPX4↑/ACSL4↓→4-HNE↓→p53羰基化↓→SIPS抑制\"的信号级联。"));

children.push(para([
  { text: "延伸目标（尽力完成）：", bold: true }
]));
children.push(bullet("在整体动物水平验证BCP通过抑制铁依赖性SIPS改善CIRI远期预后的药效。"));
children.push(bullet("初步比较桂艾挥发油与等剂量BCP单体在抗铁衰老效应上的药效差异，为桂艾整体入药提供实验依据。"));

children.push(heading3("（2）研究内容"));

children.push(heading3("研究内容一：CIRI损伤中铁依赖性SIPS的时空特征及分子开关研究"));

children.push(para([
  { text: "1. CIRI半暗带中铁依赖性SIPS的时空分布鉴定", bold: true }
]));
children.push(bullet("建立小鼠大脑中动脉闭塞/再灌注（MCAO/R）模型，在再灌注后4个关键时间点（24 h、3 d、7 d、28 d）取材，分别对应铁衰老启动期、高峰期、平台期和残余期。"));
children.push(bullet("利用免疫荧光共定位技术，检测不同脑区（核心区、半暗带、对侧区）中神经元（NeuN⁺）、星形胶质细胞（GFAP⁺）、小胶质细胞（Iba-1⁺）的铁死亡标志物（4-HNE、ACSL4）与衰老标志物（p21、γH2AX、SA-β-gal）的共表达情况。"));
children.push(bullet("测定不同时间点脑组织铁含量（比色法）、脂质过氧化水平（MDA、4-HNE）、SASP因子水平（IL-6、TNF-α、MMP-3）。"));

children.push(para([
  { text: "2. 4-HNE-p53羰基化-SLC7A11分子开关的验证", bold: true }
]));
children.push(bullet("检测不同时间点半暗带组织中p53羰基化水平（免疫沉淀+抗羰基化抗体/4-HNE抗体）、p53核转位、p21/p16表达、SLC7A11表达的动态变化，分析4-HNE含量与p53羰基化水平的相关性。"));
children.push(bullet("在体外原代神经元OGD/R模型中，使用4-HNE刺激，观察是否能诱导SIPS表型（SA-β-gal阳性、p21上调、SASP分泌）；同时使用羰基化抑制剂（如氨基胍）或p53突变体（羰基化位点突变），验证4-HNE-p53羰基化的因果关系。"));
children.push(bullet("验证p53对SLC7A11的转录抑制作用：ChIP实验检测p53与SLC7A11启动子的结合；荧光素酶报告基因实验验证p53对SLC7A11转录的调控。"));

children.push(para([
  { text: "3. ACSL4在铁依赖性SIPS中的核心作用验证", bold: true }
]));
children.push(bullet("在MCAO/R模型中，通过脑立体定位注射AAV-shACSL4敲低半暗带ACSL4的表达，检测其对铁死亡指标（Fe²⁺、MDA、GSH、GPX4）和SIPS指标（SA-β-gal、p21/p16、SASP）的影响。"));
children.push(bullet("评估ACSL4敲低对脑梗死体积、脑水肿、远期神经功能恢复的影响。"));

children.push(heading3("研究内容二：BCP抗铁依赖性SIPS的细胞机制研究"));

children.push(para([
  { text: "1. 体外铁依赖性SIPS细胞模型的建立与验证", bold: true }
]));
children.push(bullet("使用低剂量Erastin（亚致死剂量）或氧糖剥夺/复氧（OGD/R）处理原代皮层神经元和星形胶质细胞，诱导\"铁依赖性SIPS\"表型。"));
children.push(bullet("检测指标：细胞活力（CCK-8）、铁死亡指标（Fe²⁺探针FerroOrange、C11-BODIPY脂质ROS、MDA、GSH/GSSG、4-HNE）、SIPS指标（SA-β-gal染色、p21/p16蛋白表达、Lamin B1丢失、SASP因子mRNA和蛋白水平）。"));
children.push(bullet("与经典H₂O₂诱导的SIPS模型进行比较，明确铁依赖性SIPS的表型特征（铁蓄积、ACSL4上调、4-HNE-p53羰基化）。"));

children.push(para([
  { text: "2. BCP抗铁依赖性SIPS的药效评价", bold: true }
]));
children.push(bullet("在铁依赖性SIPS细胞模型上，给予不同浓度的BCP单体，检测细胞活力、铁死亡指标、SIPS指标和SASP水平，确定量效关系。"));
children.push(bullet("与阳性对照进行比较：铁死亡抑制剂（Liproxstatin-1）、Nrf2激动剂（tBHQ）、衰老清除剂（Navitoclax）。"));

children.push(para([
  { text: "3. Nrf2通路在BCP抗铁依赖性SIPS中的核心作用验证", bold: true }
]));
children.push(bullet("使用Nrf2抑制剂ML385或Nrf2 siRNA敲低，观察BCP的抗铁依赖性SIPS效应是否被逆转。"));
children.push(bullet("检测BCP对Nrf2核转位、下游靶基因（HO-1、GPX4、SLC7A11、FTH1、NQO1）的表达变化的影响。"));
children.push(bullet("验证BCP对Keap1-Nrf2复合物的调控：分子对接预测BCP与Keap1疏水口袋的结合模式（PDB: 4IQK）；细胞热位移分析（CETSA）作为探索性实验验证BCP与Keap1的潜在结合。若CETSA结果不理想，采用分子对接+点突变验证的备用策略——预测BCP与Keap1疏水口袋的关键结合残基，通过点突变破坏结合位点，观察BCP是否还能激活Nrf2，从而间接验证BCP-Keap1的相互作用。"));

children.push(para([
  { text: "4. BCP对4-HNE-p53-SLC7A11分子轴的调控", bold: true }
]));
children.push(bullet("核心检测：BCP处理后，细胞内4-HNE含量、p53羰基化水平、p53核转位、SLC7A11表达的时序变化。"));
children.push(bullet("通路验证：验证\"BCP→Nrf2→GPX4↑→4-HNE↓→p53活化↓→SLC7A11↑→进一步抑制铁死亡\"的正反馈解除机制。"));
children.push(bullet([
  { text: "延伸实验（条件允许时开展）：", bold: true },
  " 在p53敲低/过表达细胞中，进一步验证BCP的抗SIPS效应是否依赖p53通路；通过点突变验证p53羰基化位点对其功能的影响。"
]));

children.push(heading3("研究内容三：BCP通过调控铁依赖性SIPS改善CIRI远期预后的整体药效验证"));

children.push(para([
  { text: "1. BCP对MCAO/R小鼠的药效学评价", bold: true }
]));
children.push(bullet([
  { text: "实验分组：", bold: true }, "假手术组、模型组、BCP中剂量组（204 mg/kg）、BCP高剂量组（408 mg/kg）、Liproxstatin-1阳性组（10 mg/kg）、BCP+ML385组（30 mg/kg），每组12只。"
]));
children.push(bullet([
  { text: "短期药效：", bold: true }, "再灌注24 h/72 h检测脑梗死体积（TTC染色）、脑水肿、神经功能评分（mNSS）。"
]));
children.push(bullet([
  { text: "远期药效：", bold: true }, "再灌注7 d、14 d、28 d进行行为学评价，包括运动功能（转棒实验、足误实验）和认知功能（Morris水迷宫）。"
]));

children.push(para([
  { text: "2. 在体验证BCP对铁依赖性SIPS的抑制作用", bold: true }
]));
children.push(bullet("免疫荧光检测半暗带区域铁死亡标志物与SIPS标志物的共定位。"));
children.push(bullet("SA-β-gal染色检测衰老细胞负荷。"));
children.push(bullet("检测铁代谢相关蛋白（TFR1、FTH1、FPN1）、铁死亡相关蛋白（GPX4、SLC7A11、ACSL4）、SIPS相关蛋白（p21、p16、Lamin B1）的表达。"));
children.push(bullet("检测p53羰基化水平、4-HNE含量、SASP因子（IL-6、TNF-α、MMP-3）的mRNA和蛋白水平。"));

children.push(para([
  { text: "3. Nrf2依赖性的在体验证", bold: true }
]));
children.push(bullet("使用Nrf2基因敲除（Nrf2⁻/⁻）小鼠或脑立体定位注射AAV-shNrf2。"));
children.push(bullet("在Nrf2缺失条件下，观察BCP是否仍能发挥抗铁依赖性SIPS和神经保护作用。"));
children.push(bullet("明确BCP抗铁衰老的Nrf2依赖机制。"));

children.push(heading3("研究内容四：桂艾挥发油的化学成分分析及药效增益探索（延伸目标）"));

children.push(para([
  { text: "1. 桂艾挥发油的化学成分分析", bold: true }
]));
children.push(bullet("采用GC-MS技术对广西道地桂艾挥发油进行化学成分分离鉴定。"));
children.push(bullet("定量测定BCP、1,8-桉叶素、樟脑、α-蒎烯等主要成分的含量。"));

children.push(para([
  { text: "2. 桂艾挥发油与BCP单体的药效初步比较（条件允许时开展）", bold: true }
]));
children.push(bullet("在细胞水平（铁依赖性SIPS模型），头对头比较桂艾挥发油（按BCP含量折算等剂量）与BCP单体的药效差异。"));
children.push(bullet("在动物水平（MCAO/R模型），设置桂艾挥发油组与等剂量BCP单体组，初步比较梗死体积、神经功能评分和SIPS标志物的差异。"));
children.push(bullet("评估是否存在药效增益现象，为后续深入研究协同机制提供线索。"));

children.push(heading3("（3）拟解决的关键科学问题"));

children.push(bullet([
  { text: "关键问题一：CIRI半暗带中是否存在\"铁死亡驱动的SIPS\"？4-HNE-p53羰基化-SLC7A11是否是其关键分子开关？", bold: true },
  " 目前铁死亡和SIPS在CIRI中各自的作用已有报道，但二者之间的因果关联尚不明确。本项目拟通过时空调定位、功能缺失实验、4-HNE-p53羰基化检测，首次系统证实CIRI中铁依赖性SIPS的存在，并明确其核心分子开关。"
]));
children.push(bullet([
  { text: "关键问题二：BCP能否通过激活Nrf2通路阻断铁依赖性SIPS，从而改善CIRI远期预后？", bold: true },
  " BCP抗铁死亡已有证据，但是否能进一步阻断铁死亡驱动的SIPS，实现从\"抗急性死亡\"到\"抗慢性衰老\"的跃升，是本项目拟回答的核心科学问题。"
]));
children.push(bullet([
  { text: "关键问题三：桂艾挥发油与等剂量BCP单体在抗铁衰老效应上是否存在药效增益？", bold: true },
  " 本项目将此定位为探索性目标，通过初步比较为道地药材的临床应用提供实验依据。"
]));

children.push(new PageBreak());

// 3. 拟采取的研究方案及可行性分析
children.push(heading2("3. 拟采取的研究方案及可行性分析"));

children.push(heading3("（1）动物造模与分组方法"));

children.push(para([
  { text: "1. 实验动物", bold: true }
]));
children.push(bullet("清洁级健康成年雄性C57BL/6J小鼠，体重22-25 g，购于北京维通利华实验动物技术有限公司。"));
children.push(bullet("Nrf2基因敲除（Nrf2⁻/⁻）小鼠，背景为C57BL/6J，购于Jackson Laboratory或国内代理。"));
children.push(bullet("所有动物饲养于SPF级动物房，温度22±2℃，湿度50-60%，12 h光暗循环，自由进食饮水。"));
children.push(bullet("动物实验获得本单位实验动物伦理委员会批准。"));

children.push(para([
  { text: "2. MCAO/R模型建立", bold: true }
]));
children.push(bullet("参照Longa法建立小鼠大脑中动脉闭塞/再灌注模型。"));
children.push(bullet("小鼠术前12 h禁食，自由饮水。4%水合氯醛（400 mg/kg）腹腔注射麻醉。"));
children.push(bullet("颈部正中切口，分离右侧颈总动脉（CCA）、颈外动脉（ECA）、颈内动脉（ICA）。"));
children.push(bullet("将6-0单丝尼龙线栓（直径0.23 mm，头端灼烧成球）从ECA插入ICA，推进至大脑中动脉起始部，深度约9-10 mm。"));
children.push(bullet("缺血60 min后，缓慢拔出线栓实现再灌注。假手术组仅分离血管，不插入线栓。"));
children.push(bullet("术中使用加热垫维持肛温在37±0.5℃。术后动物单笼饲养，自由进食饮水。"));

children.push(para([
  { text: "3. 脑立体定位注射", bold: true }
]));
children.push(bullet("小鼠麻醉后固定于脑立体定位仪。"));
children.push(bullet("根据小鼠脑图谱（Paxinos & Watson），定位右侧半暗带对应坐标：前囟后0.5 mm，旁开2.5 mm，硬膜下3.0 mm。"));
children.push(bullet("微量注射泵缓慢注射AAV-shACSL4、AAV-shNrf2或相应对照病毒，滴度1×10¹² vg/mL，每侧注射1 μL，注射速度0.2 μL/min。"));
children.push(bullet("注射后留针5 min，缓慢拔针，骨蜡封闭骨窗，缝合皮肤。病毒表达2周后进行MCAO/R造模。"));

children.push(para([
  { text: "4. 实验分组", bold: true }
]));

const tableData1 = [
  ["实验", "分组", "动物数", "处理"],
  ["实验一\n（时空分布）", "假手术组\nMCAO/R 24 h组\nMCAO/R 3 d组\nMCAO/R 7 d组\nMCAO/R 28 d组", "8只/组\n8只/组\n8只/组\n8只/组\n8只/组", "仅分离血管\n再灌注24 h取材\n再灌注3 d取材\n再灌注7 d取材\n再灌注28 d取材"],
  ["实验二\n（ACSL4功能）", "假手术+AAV-NC组\nMCAO/R+AAV-NC组\nMCAO/R+AAV-shACSL4组", "10只/组\n10只/组\n10只/组", "假手术+对照病毒\n模型+对照病毒\n模型+ACSL4敲低"],
  ["实验三\n（BCP药效）", "假手术组\n模型组\nBCP中剂量组（204 mg/kg）\nBCP高剂量组（408 mg/kg）\nLiproxstatin-1阳性组\nBCP+ML385组", "12只/组\n12只/组\n12只/组\n12只/组\n12只/组\n12只/组", "等体积溶媒\n等体积溶媒\nBCP灌胃\nBCP灌胃\n腹腔注射10 mg/kg\nBCP+ML385 30 mg/kg"],
  ["实验四\n（Nrf2依赖）", "Nrf2⁺/⁺假手术组\nNrf2⁺/⁺模型组\nNrf2⁺/⁺+BCP组\nNrf2⁻/⁻假手术组\nNrf2⁻/⁻模型组\nNrf2⁻/⁻+BCP组", "10只/组\n10只/组\n10只/组\n10只/组\n10只/组\n10只/组", "野生型假手术\n野生型模型\n野生型+BCP\n敲除假手术\n敲除模型\n敲除+BCP"],
  ["实验五\n（桂艾比较，延伸）", "假手术组\n模型组\nBCP单体组（36 mg/kg）\n桂艾挥发油组（200 mg/kg，\n含BCP约36 mg/kg）", "10只/组\n10只/组\n10只/组\n10只/组", "溶媒\n溶媒\nBCP单体灌胃\n桂艾挥发油灌胃（条件允许时开展）"],
];

const table1Rows = tableData1.map((row, ri) => new TableRow({
  children: row.map((cell, ci) => new TableCell({
    borders,
    width: { size: ci === 0 ? 1600 : ci === 1 ? 2600 : ci === 2 ? 1400 : 3760, type: WidthType.DXA },
    shading: ri === 0 ? { fill: "D5E8F0", type: ShadingType.CLEAR } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: cell.split('\n').map(line => new Paragraph({
      children: [new TextRun({ text: line, bold: ri === 0, size: ri === 0 ? 22 : 20 })],
      alignment: AlignmentType.CENTER,
      spacing: { line: 300 }
    }))
  }))
}));

children.push(new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [1600, 2600, 1400, 3760],
  rows: table1Rows
}));

children.push(new Paragraph({ spacing: { after: 200 }, children: [] }));

children.push(heading3("（2）药物选择与用药方法"));

children.push(para([
  { text: "1. 桂艾挥发油的制备", bold: true }
]));
children.push(bullet("桂艾（Artemisia verlotorum）采自广西壮族自治区南宁市道地产区，经本校中药鉴定教研室鉴定。"));
children.push(bullet("采用水蒸气蒸馏法提取挥发油：艾叶粉碎后加10倍量水，浸泡2 h，水蒸气蒸馏5 h，收集挥发油。"));
children.push(bullet("无水硫酸钠干燥，4℃避光保存。GC-MS分析挥发油组成，测定BCP含量。"));

children.push(para([
  { text: "2. BCP单体", bold: true }
]));
children.push(bullet("β-石竹烯标准品购于Sigma-Aldrich公司（纯度≥98.5%）。"));

children.push(para([
  { text: "3. 给药方案", bold: true }
]));
children.push(bullet("桂艾挥发油和BCP：用10%聚氧乙烯蓖麻油+生理盐水配制，术前7天开始灌胃给药，每日1次，直至取材。"));
children.push(bullet("Liproxstatin-1：术前30 min腹腔注射，每日1次。"));
children.push(bullet("ML385：术前2 h腹腔注射（30 mg/kg）。"));
children.push(bullet("假手术组和模型组给予等体积溶媒。"));

children.push(heading3("（3）动物观察与标本制备方法"));

children.push(para([
  { text: "1. 一般状态观察", bold: true }
]));
children.push(bullet("术后每日观察小鼠精神状态、饮食、体重变化、有无感染等。记录死亡情况。"));

children.push(para([
  { text: "2. 神经功能缺损评分", bold: true }
]));
children.push(bullet("采用改良神经功能严重程度评分（mNSS）。"));
children.push(bullet("包括运动实验、感觉实验、平衡木实验、反射实验四部分，总分18分。"));
children.push(bullet("于再灌注后24 h、3 d、7 d、14 d、28 d由双盲法评估。"));

children.push(para([
  { text: "3. 标本制备", bold: true }
]));
children.push(bullet("麻醉小鼠，经心脏灌注PBS后，4%多聚甲醛灌注固定。"));
children.push(bullet("取脑，4%多聚甲醛后固定24 h，梯度脱水，石蜡包埋或OCT包埋冰冻切片。"));
children.push(bullet("用于分子生物学检测的样本：快速断头取脑，冰上分离缺血侧皮层半暗带组织，液氮速冻，-80℃保存。"));

children.push(heading3("（4）实验室检测"));

children.push(para([
  { text: "1. 组织病理学检测", bold: true }
]));
children.push(bullet("TTC染色：脑冠状切片（2 mm厚），2% TTC溶液37℃避光孵育30 min，4%多聚甲醛固定，Image J计算梗死体积。"));
children.push(bullet("HE染色：石蜡切片脱蜡至水，苏木精-伊红染色，光学显微镜观察组织形态。"));
children.push(bullet("Nissl染色：冰冻切片，焦油紫染色，观察神经元存活情况。"));

children.push(para([
  { text: "2. 铁依赖性SIPS标志物检测", bold: true }
]));
children.push(bullet("SA-β-gal染色：使用Cell Signaling或碧云天SA-β-gal染色试剂盒，按照说明书操作。"));
children.push(bullet("免疫荧光染色：一抗选择：NeuN（神经元）、GFAP（星形胶质细胞）、Iba-1（小胶质细胞）、4-HNE、ACSL4、p21、γH2AX、Lamin B1、Nrf2、HO-1等。激光共聚焦显微镜观察。"));
children.push(bullet("普鲁士蓝染色：检测组织铁沉积。"));

children.push(para([
  { text: "3. 铁代谢和氧化应激指标检测", bold: true }
]));
children.push(bullet("组织铁含量：采用铁检测试剂盒（比色法）。"));
children.push(bullet("脂质过氧化：MDA检测试剂盒（TBA法）、4-HNE ELISA试剂盒。"));
children.push(bullet("抗氧化能力：GSH/GSSG检测试剂盒、SOD活性检测试剂盒。"));

children.push(para([
  { text: "4. Western Blot", bold: true }
]));
children.push(bullet("RIPA裂解液提取组织/细胞总蛋白，BCA法蛋白定量。"));
children.push(bullet("检测蛋白：GPX4、SLC7A11、ACSL4、Nrf2（核/浆）、HO-1、NQO1、FTH1、TFR1、p21、p16、Lamin B1、p53、乙酰化p53、Keap1、β-actin等。"));
children.push(bullet("HRP标记二抗室温孵育1 h，ECL化学发光显影，Image Lab软件分析灰度值。"));

children.push(para([
  { text: "5. p53羰基化检测", bold: true }
]));
children.push(bullet("免疫沉淀（IP）：抗p53抗体沉淀总蛋白中的p53。"));
children.push(bullet("Western Blot检测：抗4-HNE抗体或抗羰基化抗体（DNP）检测p53的羰基化水平。"));
children.push(bullet([
  { text: "技术难点与对策：", bold: true },
  " 4-HNE修饰的检测在组织样本中由于4-HNE的高反应性和不稳定性，容易产生假阴性。为克服这一问题，所有组织样本在取材后将立即用含5 mM DTT和蛋白酶抑制剂的裂解液处理，并在氮气保护下进行后续操作。同时设置阳性对照（4-HNE处理的细胞裂解液），确保IP/IB体系的灵敏度。"
]));

children.push(para([
  { text: "6. 实时荧光定量PCR（qRT-PCR）", bold: true }
]));
children.push(bullet("Trizol法提取总RNA，反转录合成cDNA，SYBR Green法进行qPCR。"));
children.push(bullet("检测基因：p21、p16、IL-6、TNF-α、MMP-3、ACSL4、GPX4、HO-1、SLC7A11等。"));
children.push(bullet("GAPDH为内参，2⁻ΔΔCt法计算相对表达量。"));

children.push(para([
  { text: "7. ELISA检测", bold: true }
]));
children.push(bullet("脑组织匀浆上清或细胞培养上清，检测SASP因子：IL-6、TNF-α、MMP-3。"));

children.push(para([
  { text: "8. 透射电镜", bold: true }
]));
children.push(bullet("2.5%戊二醛+4%多聚甲醛灌注固定，取半暗带皮质组织（1 mm³）。"));
children.push(bullet("超薄切片（70 nm），枸橼酸铅-醋酸双氧铀染色，透射电镜观察线粒体形态和细胞超微结构。"));

children.push(para([
  { text: "9. 行为学检测", bold: true }
]));
children.push(bullet("转棒实验：评估运动协调能力。转速从4 rpm加速至40 rpm，记录跌落潜伏期。"));
children.push(bullet("足误实验：评估精细运动功能。记录前足误踏次数和总步数，计算足误率。"));
children.push(bullet("Morris水迷宫：评估空间学习记忆能力。定位航行实验5天，记录逃避潜伏期；空间探索实验记录穿越平台次数。"));

children.push(para([
  { text: "10. 分子对接与CETSA", bold: true }
]));
children.push(bullet("分子对接：从PDB获取Keap1-Nrf2复合物结构（4IQK），AutoDock Vina进行对接，PyMOL可视化。"));
children.push(bullet("细胞热位移分析（CETSA）：验证BCP与Keap1蛋白的直接结合，通过温度梯度下蛋白质熔解曲线的变化判断配体结合。"));

children.push(para([
  { text: "11. ChIP与荧光素酶报告基因", bold: true }
]));
children.push(bullet("ChIP实验：检测p53与SLC7A11启动子区域的结合。"));
children.push(bullet("荧光素酶报告基因：构建SLC7A11启动子荧光素酶报告质粒，共转染p53表达质粒，检测荧光素酶活性。"));

children.push(heading3("（5）统计学处理"));

children.push(bullet("所有实验数据采用SPSS 26.0或GraphPad Prism 9.0软件进行统计分析。"));
children.push(bullet("计量资料以均数±标准差（x̄ ± s）表示。"));
children.push(bullet("两组间比较采用独立样本t检验。"));
children.push(bullet("多组间比较采用单因素方差分析（One-way ANOVA），组间两两比较采用Tukey's多重比较。"));
children.push(bullet("重复测量数据（如水迷宫、体重变化）采用重复测量方差分析。"));
children.push(bullet("相关性分析采用Pearson或Spearman相关分析。"));
children.push(bullet("P < 0.05为差异有统计学意义。"));

children.push(heading3("（6）理论研究"));

children.push(para("本项目的理论研究框架基于以下逻辑链条构建："));

children.push(para([
  { text: "1. 理论基础：", bold: true }
]));
children.push(bullet("铁死亡是CIRI后神经元死亡的重要方式（已被广泛证实）。"));
children.push(bullet("SIPS参与CIRI后的慢性损伤和远期功能障碍（Baixauli-Martín等, 2025年支持）。"));
children.push(bullet("铁死亡与衰老存在机制交叉，亚致死铁死亡压力可驱动细胞衰老（ferro-aging概念，Liu等2026年Cell Metabolism证实）。"));
children.push(bullet("BCP可通过Nrf2通路抑制铁死亡（Hu等2022年Phytomedicine证实）。"));

children.push(para([
  { text: "2. 理论推演：", bold: true }
]));
children.push(bullet("推演1：CIRI半暗带中，亚致死铁死亡压力通过ACSL4-4-HNE-p53羰基化轴驱动细胞进入SIPS状态，形成\"铁死亡→SASP→更多铁死亡\"的恶性循环。"));
children.push(bullet("推演2：BCP通过激活Nrf2通路，同时增强抗铁死亡防御和抑制SIPS启动，可阻断铁衰老恶性循环。"));
children.push(bullet("推演3：桂艾挥发油因含多种活性成分，可能呈现整体药效增益，为道地药材临床应用提供依据。"));

children.push(para([
  { text: "3. 理论验证路径：", bold: true }
]));
children.push(bullet("从现象观察→机制解析→干预验证→整体药效，层层递进。"));
children.push(bullet("细胞水平→动物水平，双向验证。"));
children.push(bullet("功能获得（过表达）+功能缺失（敲低/敲除），明确因果关系。"));

children.push(new PageBreak());

// 技术路线图
children.push(heading2("技术路线图"));

children.push(para("（三层结构：上层-现象验证、中层-机制解析、下层-药效验证）"));

const techRows = [
  [
    { text: "上 层\n现 象 验 证\n（第1年）", fill: "E8F5E9" },
    { text: "CIRI半暗带铁依赖性SIPS的存在性验证（核心目标）", fill: "E8F5E9" }
  ],
  [
    { text: "", fill: "FFFFFF" },
    { text: "↓ 时间：24 h / 3 d / 7 d / 28 d\n↓ 空间：核心区 / 半暗带 / 对侧区\n↓ 细胞：神经元 / 星形胶质细胞 / 小胶质细胞", fill: "FFFFFF" }
  ],
  [
    { text: "", fill: "FFFFFF" },
    { text: "关键技术：免疫荧光共定位 | SA-β-gal染色 | 普鲁士蓝染色 | 透射电镜 | Western Blot", fill: "FFF3E0" }
  ],
  [
    { text: "", fill: "FFFFFF" },
    { text: "↓\n确认铁依赖性SIPS的存在与时空特征", fill: "FFFFFF" }
  ],
  [
    { text: "中 层\n机 制 解 析\n（第2年）", fill: "E3F2FD" },
    { text: "4-HNE-p53-SLC7A11分子轴 + BCP调控机制（核心目标）", fill: "E3F2FD" }
  ],
  [
    { text: "", fill: "FFFFFF" },
    { text: "↓ 分子开关验证 ↓            ↓ BCP调控机制 ↓", fill: "FFFFFF" }
  ],
  [
    { text: "", fill: "FFFFFF" },
    { text: "4-HNE→p53活化→SLC7A11↓\nIP/IB | ChIP | 报告基因 | 羰基化抑制剂\n\n            BCP→Nrf2→GPX4↑→4-HNE↓\n            Nrf2抑制剂/敲低 | 分子对接 | CETSA（探索）", fill: "FFFFFF" }
  ],
  [
    { text: "", fill: "FFFFFF" },
    { text: "↓\n阐明BCP通过Nrf2通路阻断铁依赖性SIPS的分子机制", fill: "FFFFFF" }
  ],
  [
    { text: "下 层\n药 效 验 证\n（第3年）", fill: "FCE4EC" },
    { text: "BCP改善CIRI远期预后的整体药效（核心）+ 桂艾药效增益探索（延伸）", fill: "FCE4EC" }
  ],
  [
    { text: "", fill: "FFFFFF" },
    { text: "↓ 整体药效 ↓            ↓ Nrf2依赖 ↓            ↓ 桂艾比较（延伸）↓", fill: "FFFFFF" }
  ],
  [
    { text: "", fill: "FFFFFF" },
    { text: "MCAO/R模型\nTTC | mNSS | 转棒 | 水迷宫\nSA-β-gal | p53羰基化 | SASP\n\nNrf2⁻/⁻小鼠\n验证Nrf2依赖性\n\n            桂艾挥发油 vs BCP单体            \n            （条件允许时开展）", fill: "FFFFFF" }
  ],
  [
    { text: "", fill: "FFFFFF" },
    { text: "↓\n最终结论：BCP通过Nrf2通路抑制铁依赖性SIPS，改善CIRI远期预后", fill: "F3E5F5" }
  ],
];

const techTableRows = techRows.map(row => new TableRow({
  children: row.map(cell => new TableCell({
    borders,
    width: { size: cell.text.startsWith('上') || cell.text.startsWith('中') || cell.text.startsWith('下') ? 2000 : 7360, type: WidthType.DXA },
    shading: { fill: cell.fill, type: ShadingType.CLEAR },
    margins: { top: 100, bottom: 100, left: 150, right: 150 },
    verticalAlign: VerticalAlign.CENTER,
    children: cell.text.split('\n').map(line => new Paragraph({
      children: [new TextRun({ text: line, bold: cell.fill !== "FFFFFF" && cell.fill !== "FFF3E0", size: 20 })],
      alignment: cell.text.startsWith('上') || cell.text.startsWith('中') || cell.text.startsWith('下') ? AlignmentType.CENTER : AlignmentType.LEFT,
      spacing: { line: 320 }
    }))
  }))
}));

children.push(new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [2000, 7360],
  rows: techTableRows
}));

children.push(new Paragraph({ spacing: { after: 200 }, children: [] }));

children.push(new PageBreak());

// （4）可行性分析
children.push(heading2("（4）可行性分析"));

children.push(heading3("① 理论可行性"));

children.push(para("本项目的科学假说建立在坚实的文献基础和合理的逻辑推演之上："));

children.push(bullet([
  { text: "铁衰老概念的可靠性：", bold: true }, "2026年Liu等在Cell Metabolism发表的研究在灵长类水平系统证实了ferro-aging的存在，明确ACSL4是核心执行分子，为该领域奠定了理论基础。4-HNE-p53羰基化作为连接氧化应激与衰老的分子开关，已有多篇高水平论文支持。"
]));
children.push(bullet([
  { text: "SIPS在CIRI中的证据：", bold: true }, "Baixauli-Martín等（2025, Int J Mol Sci）系统表征了实验性缺血性卒中的细胞衰老时空特征，证实缺血半暗带存在p16/p21/SA-β-gal阳性的衰老细胞，为本项目提供了直接的组织学证据。"
]));
children.push(bullet([
  { text: "BCP神经保护作用的可靠性：", bold: true }, "本团队前期已证实BCP可通过Nrf2/HO-1通路抑制MCAO/R大鼠铁死亡，相关成果发表于Phytomedicine（Hu Q, et al. 2022），为本项目提供了直接的前期工作基础。"
]));
children.push(bullet([
  { text: "桂艾道地性的科学内涵：", bold: true }, "广西产桂艾挥发油中BCP含量显著高于其他品种（可达18.21%），这一化学特征为其在铁衰老相关疾病中的应用提供了独特的物质基础。"
]));

children.push(heading3("② 技术可行性"));

children.push(para("本项目涉及的实验技术均为成熟可靠的常规技术，研究团队已熟练掌握："));

children.push(bullet("动物模型：MCAO/R模型是神经药理学研究的经典模型，研究团队具有多年小鼠MCAO/R模型制备经验，成功率稳定在80%以上。"));
children.push(bullet("分子生物学技术：Western Blot、qRT-PCR、免疫荧光、ELISA、IP、ChIP等均为实验室常规技术，有成熟的实验流程。"));
children.push(bullet("铁死亡检测技术：MDA、GSH、铁含量测定、C11-BODIPY探针、透射电镜等技术方法成熟。"));
children.push(bullet("衰老检测技术：SA-β-gal染色、p21/p16检测、SASP因子测定等均为细胞衰老研究的标准方法。"));
children.push(bullet("行为学检测：转棒、水迷宫等均为神经行为学经典范式。"));
children.push(bullet("病毒载体干预：AAV脑立体定位注射技术成熟，可实现目的基因在脑内的敲低。"));

children.push(heading3("③ 材料可行性"));

children.push(bullet("实验动物：C57BL/6J小鼠和Nrf2基因敲除小鼠均可从正规渠道购买，来源可靠。"));
children.push(bullet("药物和试剂：BCP标准品（Sigma）、各类抑制剂（MCE、Selleck）、抗体（CST、Abcam、Proteintech）、检测试剂盒（碧云天、南京建成）等均有商品化产品，易于获取。"));
children.push(bullet("桂艾药材：广西为道地产区，药材来源稳定，可确保质量。"));
children.push(bullet("病毒载体：AAV载体可从专业公司（如汉恒生物、吉凯基因）定制合成。"));

children.push(heading3("④ 研究团队与前期工作基础"));

children.push(para("研究团队长期从事神经药理学和中药药理学研究，在脑缺血再灌注损伤、铁死亡、天然产物神经保护等方向积累了扎实的研究基础："));

children.push(bullet("团队前期已证实BCP可通过Nrf2/HO-1通路抑制MCAO/R大鼠铁死亡，相关成果发表于Phytomedicine（Hu Q, et al. 2022, IF: 5.340），为本项目提供了直接的前期工作基础。"));
children.push(bullet("团队在网络药理学、中药挥发油研究方面有丰富经验，发表了多篇相关论文（刘胜伟等, 2024）。"));
children.push(bullet("团队具备完善的实验平台，包括动物行为学实验室、分子生物学实验室、细胞培养室等，可满足本项目的实验需求。"));
children.push(bullet("团队与广西中医药大学合作，可确保桂艾药材的道地性和质量控制。"));

children.push(para("综上所述，本项目在理论、技术、材料和团队方面均具有良好的可行性。"));

children.push(new PageBreak());

// 4. 本项目的特色与创新之处
children.push(heading1("4. 本项目的特色与创新之处"));

children.push(heading3("（1）理论创新：率先提出并系统验证CIRI半暗带\"铁依赖性SIPS\"新假说"));

children.push(para("目前关于CIRI的研究多将铁死亡和细胞衰老视为两个独立的病理过程。本项目基于最新的ferro-aging研究进展（Liu等, Cell Metabolism, 2026）和SIPS领域的研究成果（Baixauli-Martín等, 2025），结合CIRI的病理特点，创新性地提出：缺血半暗带中存在\"亚致死铁死亡压力→ACSL4介导的脂质过氧化→4-HNE-p53羰基化→p21/p16通路激活→SIPS+SASP→正反馈促进更多铁死亡\"的恶性循环，即\"铁依赖性SIPS\"病理轴。"));

children.push(para("本项目将首次在CIRI模型中系统证实铁依赖性SIPS的存在，并明确\"4-HNE-p53羰基化-SLC7A11\"是其关键分子开关。这一假说若得到验证，将为理解CIRI从急性损伤向慢性损伤转化的机制提供全新视角，为解释\"为什么再灌注成功但远期预后仍不佳\"这一临床难题提供新答案。"));

children.push(heading3("（2）机制创新：揭示BCP通过Nrf2通路抑制铁依赖性SIPS的双重保护机制"));

children.push(para("现有BCP抗铁死亡研究多聚焦于Nrf2/HO-1-GPX4通路的表达上调。本项目创新性地将BCP的作用机制拓展至\"铁死亡-衰老\"交互领域，提出BCP通过激活Nrf2通路实现双重保护："));

children.push(bullet([
  { text: "阻断铁死亡启动（急性保护）：", bold: true }, "BCP促进Nrf2核转位，上调GPX4、HO-1、SLC7A11、FTH1等下游靶基因，增强细胞抗铁死亡和抗氧化防御能力。"
]));
children.push(bullet([
  { text: "抑制SIPS启动（慢性保护）：", bold: true }, "BCP通过激活Nrf2增强GPX4活性，减少4-HNE等脂质过氧化醛类的产生，解除4-HNE对p53的羰基化修饰，抑制p53-p21/p16通路的过度激活，从而遏制SIPS的启动。"
]));

children.push(para("双重保护机制的提出，将BCP的作用从\"单一的抗氧化/抗铁死亡\"提升为\"系统性阻断铁衰老恶性循环\"，丰富了BCP神经保护的分子内涵。"));

children.push(heading3("（3）药物创新：以壮瑶药理论为指导，发掘道地药材桂艾抗铁衰老的新用途"));

children.push(para("本项目以广西道地壮药桂艾为研究对象，将壮瑶医药\"通龙路火路、除毒邪补虚损\"的传统功效与现代\"铁衰老\"理论有机结合："));

children.push(bullet("\"除毒邪\"对应抑制铁死亡和脂质过氧化（清除氧化毒性）。"));
children.push(bullet("\"通龙路火路\"对应改善脑循环、保护神经血管单元。"));
children.push(bullet("\"补虚损\"对应抑制SASP、改善微环境、促进神经修复。"));

children.push(para("在BCP单体机制明确的基础上，初步探索桂艾挥发油的整体药效增益，为道地药材的现代化研究提供可复制的范例。同时，将桂艾开发为缺血性脑卒中的神经保护剂，具有良好的临床转化前景和社会经济效益。"));

children.push(heading3("（4）视角创新：从\"急性死亡\"到\"慢性衰老\"，关注CIRI的远期预后"));

children.push(para("传统CIRI神经保护研究多关注急性期（24-72 h）的梗死体积和神经功能缺损，而对远期（2-4周及以后）的功能恢复关注不足。本项目提出铁依赖性SIPS这一慢性损伤机制，将研究视野从\"挽救急性死亡的细胞\"拓展为\"阻断慢性衰老的级联\"，通过靶向铁衰老改善半暗带微环境，促进远期神经功能恢复。这一视角的转变，更贴近临床患者\"卒中后长期功能障碍\"的真实需求，可能为脑卒中的康复治疗提供新策略。"));

children.push(new PageBreak());

// 5. 年度研究计划
children.push(heading1("5. 年度研究计划"));

children.push(heading3("第一年（202X年1月 - 202X年12月）"));

children.push(para([
  { text: "研究目标：", bold: true }, "建立CIRI铁依赖性SIPS研究体系，明确铁衰老的时空分布特征和关键分子开关。"
]));

children.push(para([
  { text: "主要研究内容：", bold: true }
]));
children.push(bullet("建立小鼠MCAO/R模型，优化造模条件，确保模型稳定性和可重复性。"));
children.push(bullet("系统检测再灌注后24 h、3 d、7 d、28 d四个关键时间点半暗带组织中铁死亡和SIPS标志物的动态变化。"));
children.push(bullet("运用免疫荧光共定位、透射电镜等技术，明确铁依赖性SIPS细胞的细胞类型和空间分布特征。"));
children.push(bullet("检测4-HNE含量、p53羰基化水平、SLC7A11表达的动态变化，分析4-HNE-p53羰基化-SLC7A11分子开关的相关性。"));
children.push(bullet("构建AAV-shACSL4载体，通过脑立体定位注射验证ACSL4在铁依赖性SIPS中的核心驱动作用。AAV-shACSL4载体已于前期完成设计，将在项目启动后立即委托公司包装，预计第4个月获得病毒，第6个月完成立体定位注射，第10个月完成全部功能验证。"));
children.push(bullet("完成桂艾挥发油的提取、GC-MS成分分析和含量测定。"));

children.push(para([
  { text: "预期进展：", bold: true }
]));
children.push(bullet("证实CIRI半暗带中存在铁依赖性SIPS现象，明确其时空分布规律。"));
children.push(bullet("初步验证4-HNE-p53羰基化-SLC7A11分子开关的存在。"));
children.push(bullet("建立桂艾挥发油的质量控制方法。"));
children.push(bullet("发表SCI论文1-2篇，培养硕士研究生1名。"));

children.push(heading3("第二年（202X年1月 - 202X年12月）"));

children.push(para([
  { text: "研究目标：", bold: true }, "明确BCP抗铁依赖性SIPS的细胞机制和分子靶点。"
]));

children.push(para([
  { text: "主要研究内容：", bold: true }
]));
children.push(bullet("建立原代神经元和星形胶质细胞的铁依赖性SIPS体外模型（低剂量Erastin/OGD/R诱导），并与经典H₂O₂诱导的SIPS模型进行比较。"));
children.push(bullet("在细胞水平评价BCP抗铁依赖性SIPS的药效，确定量效关系。"));
children.push(bullet("运用Nrf2抑制剂ML385和Nrf2 siRNA，验证BCP抗铁依赖性SIPS是否依赖Nrf2通路。"));
children.push(bullet("验证BCP对4-HNE-p53羰基化-SLC7A11分子开关的调控作用。"));
children.push(bullet("CETSA和分子对接验证BCP与Keap1的直接结合。"));
children.push(bullet("桂艾挥发油与BCP单体的细胞水平药效初步比较。"));

children.push(para([
  { text: "预期进展：", bold: true }
]));
children.push(bullet("建立稳定的体外铁依赖性SIPS细胞模型。"));
children.push(bullet("明确BCP抗铁依赖性SIPS的Nrf2依赖机制。"));
children.push(bullet("验证4-HNE-p53羰基化-SLC7A11分子开关的调控。"));
children.push(bullet("发表SCI论文2-3篇，培养硕士研究生1-2名。"));

children.push(heading3("第三年（202X年1月 - 202X年12月）"));

children.push(para([
  { text: "研究目标：", bold: true }, "在整体动物水平验证BCP通过抑制铁依赖性SIPS改善CIRI远期预后的药效，并完成项目总结。"
]));

children.push(para([
  { text: "主要研究内容：", bold: true }
]));
children.push(bullet("在小鼠MCAO/R模型上，系统评价BCP对短期（24 h、72 h）和远期（7 d、14 d、28 d）神经功能恢复的影响。"));
children.push(bullet("通过行为学检测（转棒、足误、水迷宫）评价远期运动功能和认知功能。"));
children.push(bullet("在体验证BCP对铁依赖性SIPS的抑制作用（SA-β-gal、p53羰基化、SASP、铁死亡标志物等）。"));
children.push(bullet("利用Nrf2⁻/⁻小鼠，在体验证BCP抗铁依赖性SIPS和神经保护的Nrf2依赖性。"));
children.push(bullet("桂艾挥发油与BCP单体的动物水平药效初步比较。"));
children.push(bullet("整合所有实验数据，完善铁依赖性SIPS理论和BCP干预机制，撰写研究总结和结题报告。"));

children.push(para([
  { text: "预期进展：", bold: true }
]));
children.push(bullet("证实靶向铁依赖性SIPS可有效改善CIRI远期预后。"));
children.push(bullet("完成Nrf2依赖性的在体验证。"));
children.push(bullet("初步比较桂艾挥发油与BCP单体的药效差异。"));
children.push(bullet("发表高影响力SCI论文1-2篇（IF>5）。"));
children.push(bullet("培养研究生2-3名。申请发明专利1项。"));

children.push(new PageBreak());

// 6. 预期研究结果
children.push(heading1("6. 预期研究结果"));

children.push(heading3("（1）预期成果"));

children.push(para([
  { text: "1. 理论成果：", bold: true }
]));
children.push(bullet("首次系统证实脑缺血再灌注损伤中存在\"铁死亡驱动的应激诱导早熟性衰老\"（铁依赖性SIPS）病理轴。"));
children.push(bullet("明确\"4-HNE-p53羰基化-SLC7A11\"是铁依赖性SIPS的关键分子开关。"));
children.push(bullet("揭示桂艾活性成分BCP通过Nrf2通路抑制铁依赖性SIPS的双重保护机制。"));
children.push(bullet("建立\"道地药材-壮瑶医功效-核心成分-铁衰老靶点-信号通路\"的整合研究模式。"));

children.push(para([
  { text: "2. 学术论文：", bold: true }
]));
children.push(bullet("发表SCI论文4-6篇，其中IF>5的论文2-3篇。"));
children.push(bullet("发表中文核心期刊论文1-2篇。"));
children.push(bullet("在国内外学术会议上交流研究成果2-3次。"));

children.push(para([
  { text: "3. 知识产权：", bold: true }
]));
children.push(bullet("申请国家发明专利1项（β-石竹烯/桂艾挥发油在制备抑制铁依赖性SIPS药物中的应用）。"));

children.push(para([
  { text: "4. 人才培养：", bold: true }
]));
children.push(bullet("培养硕士研究生3-4名。"));
children.push(bullet("培养青年科研骨干1-2名。"));

children.push(para([
  { text: "5. 潜在转化价值：", bold: true }
]));
children.push(bullet("为桂艾治疗缺血性脑卒中的临床应用提供科学依据。"));
children.push(bullet("为开发新型铁衰老抑制剂提供候选化合物和作用靶点。"));
children.push(bullet("为缺血性脑卒中的康复治疗提供新策略。"));

children.push(heading3("（2）预期可解决的关键科学技术问题"));

children.push(bullet("阐明铁依赖性SIPS在CIRI慢性损伤中的病理作用，丰富对CIRI病理机制的认识。"));
children.push(bullet("明确BCP抗铁依赖性SIPS的分子靶点，深化对其神经保护机制的理解。"));
children.push(bullet("阐释桂艾道地性的科学内涵，为壮瑶药现代化研究提供范例。"));
children.push(bullet("验证靶向铁依赖性SIPS改善CIRI远期预后的可行性，为脑卒中治疗提供新思路。"));

children.push(heading3("（3）预期社会经济效益"));

children.push(bullet("为缺血性脑卒中患者提供新的治疗思路和药物候选，有望改善卒中后远期功能障碍，提高患者生活质量。"));
children.push(bullet("推动广西道地药材桂艾的开发利用，带动地方中药材产业发展，助力乡村振兴。"));
children.push(bullet("促进壮瑶医药理论的现代化阐释，推动民族医药的传承创新发展。"));

children.push(new PageBreak());

// 参考文献
children.push(heading1("参考文献"));

const refs = [
  "[1] Dixon SJ, Lemberg KM, Lamprecht MR, et al. Ferroptosis: an iron-dependent form of nonapoptotic cell death. Cell, 2012, 149(5): 1060-1072.",
  "[2] Hu Q, Zuo T, Deng L, et al. β-Caryophyllene suppresses ferroptosis induced by cerebral ischemia reperfusion via activation of the NRF2/HO-1 signaling pathway in MCAO/R rats. Phytomedicine, 2022, 104: 154112.",
  "[3] Toussaint O, Royer V, Salmon M, et al. Stress-induced premature senescence and tissue aging. Sub-cellular Biochemistry, 2002, 33: 285-307.",
  "[4] Baixauli-Martín J, Burguete MC, López-Morales MA, et al. Spatio-Temporal Characterization of Cellular Senescence Hallmarks in Experimental Ischemic Stroke. International Journal of Molecular Sciences, 2025, 26(5): 2364.",
  "[5] Liu L, Zheng Z, You W, et al. Vitamin C inhibits ACSL4 to alleviate ferro-aging in primates. Cell Metabolism, 2026, 38(4): 567-584.e8.",
  "[6] Monroe TB, Hertzel AV, Dickey DM, et al. Lipid peroxidation products induce carbonyl stress, mitochondrial dysfunction, and cellular senescence in human and murine cells. Aging Cell, 2025, 24(1): e14367.",
  "[7] 宋叶, 梅全喜, 吴孟华, 等. 不同产地艾叶挥发油GC-MS分析及解热作用比较. 中药材, 2019, 42(8): 1789-1793.",
  "[8] Guo D, Yang Y, Wu Y, et al. Chemical Composition Analysis and Discrimination of Essential Oils of Artemisia Argyi Folium from Different Germplasm Resources Based on Electronic Nose and GC/MS Combined with Chemometrics. Chemistry & Biodiversity, 2023, 20(3): e202200991.",
  "[9] Gertsch J, Leonti M, Raduner S, et al. Beta-caryophyllene is a dietary cannabinoid. Proceedings of the National Academy of Sciences, 2008, 105(26): 9099-9104.",
  "[10] Yang YC, Lv GH, Chen S, et al. β-caryophyllene alleviates ischemic stroke injury in mice by inhibiting HMGB1-TLR4 signaling pathway. Journal of Ethnopharmacology, 2017, 204: 1-10.",
  "[11] Bahi A, Al Mansouri S, Al Memari E, et al. β-caryophyllene, a CB2 receptor agonist produces multiple behavioral changes relevant to anxiety and depression in mice. Physiology & Behavior, 2014, 135: 193-200.",
  "[12] Rathod SS, Agrawal YO. β-Caryophyllene (CB2 agonist) mitigates rotenone-induced neurotoxicity and apoptosis in SH-SY5Y neuroblastoma cells via modulation of GSK-3β/NRF2/HO-1 axis. Naunyn-Schmiedeberg's Archives of Pharmacology, 2025, 398(11): 2987-3003.",
  "[13] Bolat I, Yildirim S, Saglam YS, et al. β-Caryophyllene attenuates cadmium induced neurotoxicity in rats by modulating different cellular signaling pathways. Neurotoxicology, 2025, 80: 131-144.",
  "[14] Iorio R, Celenza G, Petricca S. Multi-Target Effects of ß-Caryophyllene and Carnosic Acid at the Crossroads of Mitochondrial Dysfunction and Neurodegeneration: From Oxidative Stress to Microglia-Mediated Neuroinflammation. Antioxidants, 2022, 11(6): 1199.",
  "[15] Liu Y, He Y, Wang F, et al. From longevity grass to contemporary soft gold: Explore the chemical constituents, pharmacology, and toxicology of Artemisia argyi H.Lév. & vaniot essential oil. Journal of Ethnopharmacology, 2021, 281: 114404.",
  "[16] Hossain R, Lee HJ, Hossain MS, et al. Ferroptosis-Driven Senescence Loop as a Central Amplifier of Osteoarthritis Progression. Biomolecules & Therapeutics, 2026, 34(3): 234-248.",
  "[17] Zhang Y, Zhang D, Xiao H. Regulated neuronal death in Alzheimer's disease: Crosstalk and convergence of apoptosis, pyroptosis, senescence, and ferroptosis. Journal of Alzheimer's Disease, 2026, 88(2): 345-368.",
  "[18] Li C, Gong Z, Ye J, et al. Vitamin C, ACSL4, and Ferro-Aging: Mechanistic Insights and Translational Perspectives from Primate Studies. Aging and Disease, 2026, 17(3): 456-472.",
  "[19] Stockwell BR, Jiang X, Gu W. Emerging mechanisms and disease relevance of ferroptosis. Cell, 2020, 181(2): 296-313.",
  "[20] Tsvetkov PO, Coy S, Petrova B, et al. Copper induces cell death by targeting lipoylated TCA cycle proteins. Science, 2022, 375(6586): 1254-1261.",
];

refs.forEach(ref => {
  children.push(new Paragraph({
    children: [new TextRun({ text: ref, size: 20 })],
    spacing: { line: 320, after: 80 },
    indent: { left: 480, hanging: 480 }
  }));
});

// ===== 构建文档 =====
const doc = new Document({
  styles: {
    default: { document: { run: { font: "宋体", size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "黑体" },
        paragraph: { spacing: { before: 360, after: 240 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "黑体" },
        paragraph: { spacing: { before: 280, after: 200 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "黑体" },
        paragraph: { spacing: { before: 200, after: 160 }, outlineLevel: 2 } },
    ]
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
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
      default: new Header({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "国家自然科学基金申请书（修订版）", size: 18, color: "666666" })]
      })] })
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "第 ", size: 18 }),
          new TextRun({ children: [PageNumber.CURRENT], size: 18 }),
          new TextRun({ text: " 页", size: 18 })]
      })] })
    },
    children
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("标书_修订版V2_广西道地壮药桂艾BCP靶向Nrf2抑制铁依赖性SIPS改善CIRI.docx", buffer);
  console.log("标书Word文档V2生成成功！");
  console.log("文件路径：d:\\铁衰老 绝不重蹈覆辙\\标书_修订版V2_广西道地壮药桂艾BCP靶向Nrf2抑制铁依赖性SIPS改善CIRI.docx");
});
