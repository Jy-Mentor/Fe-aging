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

children.push(h3('1. 脑缺血再灌注损伤的临床困境与病理机制复杂性'));
children.push(p('脑卒中是全球范围内导致成人死亡和长期残疾的首要原因之一。据世界卫生组织统计，全球每年约有1500万人发生脑卒中，其中约500万人死亡，另有500万人遗留永久性残疾。在我国，脑卒中已成为首位致死和致残病因，每年新发病例超过200万，现存患者逾1300万，给家庭和社会带来沉重的疾病负担。急性缺血性脑卒中约占全部脑卒中的70%-80%，其治疗的核心在于尽早恢复血流灌注。静脉溶栓和血管内机械取栓是目前唯一被循证医学证实有效的再灌注治疗手段，然而严格的时间窗限制了其临床获益人群——溶栓治疗的时间窗仅为发病后4.5小时，机械取栓可延长至6-24小时，但即便是在时间窗内接受再灌注治疗的患者中，仍有相当比例出现不良预后。'));
children.push(p('导致这一现象的重要原因之一是脑缺血再灌注损伤（cerebral ischemia-reperfusion injury, CIRI）。再灌注在挽救缺血半暗带的同时，也触发了一系列复杂的继发性损伤级联反应，包括氧化应激爆发、兴奋性毒性、神经炎症、血脑屏障破坏以及多种形式的调节性细胞死亡。这些病理过程相互交织、互为因果，共同推动缺血性损伤从急性期向慢性期演变，最终导致神经功能缺损的持续存在。尽管过去数十年间针对CIRI的单一靶点干预策略在动物实验中屡获成功，但转化至临床研究却屡屡失败，提示我们对CIRI病理机制的系统性认识仍存在不足，特别是急性期损伤向慢性期演变的分子纽带尚未阐明。寻找能够同时干预急性期损伤和慢性期修复的关键病理节点，是改善缺血性脑卒中远期预后的重要科学问题。'));
children.push(p('半暗带概念的提出为CIRI的干预提供了重要的理论框架。缺血核心区细胞发生快速、不可逆的坏死，而周围半暗带的神经细胞虽遭受缺血应激但仍维持膜完整性，具有潜在的可逆性。然而，随着再灌注时间延长，半暗带细胞可通过多种机制发生进行性死亡或进入功能异常状态，导致梗死灶扩大和神经功能恶化。除了经典的坏死、凋亡和自噬性死亡外，铁死亡（ferroptosis）、焦亡（pyroptosis）、坏死性凋亡（necroptosis）等新型调节性细胞死亡方式在CIRI中的作用逐渐被揭示。这些细胞死亡方式在时间和空间上呈现差异化分布，共同构成了缺血后脑损伤的细胞死亡网络。与此同时，越来越多的证据表明，半暗带中存在一类既未发生死亡、也未恢复正常功能的细胞群体，它们进入一种持续性的应激状态，通过旁分泌效应影响周围微环境，阻碍神经修复。细胞衰老（cellular senescence）正是这种状态的典型代表。'));

children.push(h3('2. 铁死亡：CIRI急性期神经元损伤的关键执行者'));
children.push(p('铁死亡是Stockwell团队于2012年正式命名的一种铁依赖性、脂质过氧化驱动的调节性细胞死亡方式，其形态学、生化和遗传学特征均区别于凋亡、坏死和自噬[4]。铁死亡的核心机制是细胞抗氧化防御系统失能，导致含多不饱和脂肪酸的磷脂发生毒性过氧化产物的大量堆积，最终破坏质膜完整性引发细胞死亡。在生化层面，铁死亡涉及三大核心通路的失衡：System Xc\u207b/GPX4抗氧化轴、ACSL4/LPCAT3脂质重塑轴以及铁代谢调控轴。GPX4是目前已知唯一能够直接还原磷脂氢过氧化物的酶，被认为是铁死亡的核心守门分子；System Xc\u207b通过摄取胱氨酸维持细胞内GSH合成，为GPX4提供还原当量；ACSL4和LPCAT3则参与催化多不饱和脂肪酸掺入膜磷脂，决定了细胞对铁死亡的敏感性；铁代谢的紊乱则为Fenton反应提供了催化剂，促进脂质过氧化的启动和传播[5]。'));
children.push(p('近年来，铁死亡在CIRI中的作用得到了广泛验证。多项独立研究表明，脑缺血再灌注后，缺血脑组织中出现典型的铁死亡特征，包括游离铁水平升高、GSH耗竭、GPX4活性下降、脂质过氧化产物4-HNE和MDA堆积以及线粒体皱缩等[7,13]。时间进程研究显示，铁死亡标志物在再灌注后数小时内即可检测到升高，其时间动力学与急性期神经元死亡高度吻合。药理学干预实验进一步证实了铁死亡的因果作用：铁死亡特异性抑制剂Ferrostatin-1、Liproxstatin-1以及铁螯合剂DFO均可显著减小MCAO模型的梗死体积、改善神经功能评分，其神经保护效应在不同物种（小鼠、大鼠）、不同缺血模型（MCAO、四血管阻断）中均得到重复。基因水平的证据同样支持这一结论：神经元特异性GPX4敲除加剧缺血性脑损伤，而GPX4过表达或SLC7A11上调则发挥保护作用。这些研究共同确立了铁死亡作为CIRI急性期神经元死亡关键形式的地位。'));
children.push(p('然而，铁死亡在CIRI中的作用并非仅限于急性期的细胞杀伤效应。越来越多的研究提示，铁死亡相关的氧化应激和脂质过氧化产物可能具有更广泛的生物学效应，包括对基因表达调控、信号通路活化以及细胞命运决定的长期影响。特别是亚致死剂量的铁死亡应激——即氧化应激水平升高但尚未达到致死阈值——是否会触发细胞进入其他应激状态，如细胞衰老，目前尚缺乏系统研究。回答这一问题，对于理解缺血后脑损伤从急性期向慢性期演变的机制具有重要意义。如果亚致死性铁死亡应激确实能够驱动细胞衰老，那么铁死亡就不仅仅是一个急性期的治疗靶点，更是连接急性期损伤与慢性期功能恶化的关键病理节点。'));

children.push(h3('3. 细胞衰老：缺血半暗带慢性化的重要推手'));
children.push(p('细胞衰老是指细胞在各种应激因素作用下，退出细胞周期并进入一种稳定的增殖停滞状态，同时伴随广泛的基因表达和代谢重编程，形成特征性的衰老相关分泌表型（senescence-associated secretory phenotype, SASP）[6]。经典的细胞衰老包括端粒依赖性的复制性衰老和应激诱导的早熟性衰老（stress-induced premature senescence, SIPS）。前者与细胞分裂次数相关，主要发生于生理性衰老过程；后者则可由DNA损伤、氧化应激、癌基因激活、炎症因子等多种急性应激因素在数小时至数天内快速触发，其表型与复制性衰老高度相似但发生机制和时间尺度截然不同。SIPS的核心分子通路包括p53/p21CIP1轴和p16INK4a/Rb轴，二者共同驱动细胞周期停滞的建立和维持。SASP是衰老细胞最具病理影响力的特征之一，其成分包括促炎细胞因子、趋化因子、生长因子、基质金属蛋白酶以及多种可溶性受体配体，通过旁分泌和内分泌方式重塑组织微环境。'));
children.push(p('传统观点认为，细胞衰老主要与机体老化和年龄相关疾病有关，在急性损伤中作用有限。然而，近年来这一观念正在被逐步修正。在脑缺血领域，越来越多的证据表明，缺血性损伤可在梗死周围半暗带诱导多种细胞类型发生衰老样改变。在啮齿类动物MCAO模型中，再灌注后数天内即可在半暗带检测到SA-\u03b2-半乳糖苷酶（SA-\u03b2-gal）阳性细胞，同时伴随p21和p16表达上调、\u03b3H2AX焦点增加等衰老标志物的激活[8]。发生衰老的细胞类型包括神经元、星形胶质细胞、小胶质细胞以及血管内皮细胞，提示缺血性应激诱导的衰老具有细胞普遍性。临床样本研究同样提供了支持性证据：缺血性脑卒中患者的脑脊液和外周血中可检测到衰老相关生物标志物的升高，且其水平与梗死体积和功能预后相关。'));
children.push(p('缺血诱导的衰老细胞并非被动的旁观者，而是通过多种机制积极参与病理进程。首先，SASP中的促炎因子和趋化因子可招募外周免疫细胞浸润、激活脑内固有免疫细胞，形成慢性炎症微环境，加剧继发性损伤。其次，衰老细胞分泌的基质金属蛋白酶可降解细胞外基质，破坏血脑屏障完整性和神经环路结构。再次，SASP因子可通过旁分泌方式诱导周围正常细胞发生衰老（衰老的传播效应），扩大损伤范围。最后，衰老的神经元和胶质细胞丧失正常生理功能，阻碍突触重塑和神经发生，影响脑损伤后的修复与重塑。从干预角度，利用衰老细胞清除剂（senolytics）选择性清除衰老细胞或抑制SASP分泌，在多种脑损伤模型中均显示出改善功能预后的效果，为靶向衰老的治疗策略提供了原理验证。然而，脑缺血后细胞衰老的触发因素和上游驱动机制尚未完全阐明，特别是氧化应激在衰老启动中的具体作用形式和分子通路仍有待深入揭示。'));

children.push(h3('4. 铁死亡与细胞衰老的交汇：铁衰老概念及其在CIRI中的研究空白'));
children.push(p('铁死亡与细胞衰老并非两个独立的病理过程，二者之间存在密切的交互作用和共同的上游驱动因素——氧化应激和铁代谢紊乱。近年来，研究者开始将这两个领域联系起来，提出了铁衰老（ferro-aging/ferro-senescence）的概念框架。2026年，Liu等[1]在Cell Metabolism发表的灵长类动物研究为铁衰老概念提供了系统性的体内证据：该研究发现，随着灵长类年龄增长，肝脏和血清中铁水平逐渐升高，同时伴随ACSL4表达上调和脂质过氧化产物积累；铁过载可通过ACSL4介导的脂质过氧化通路驱动肝细胞和造血干细胞发生衰老，形成铁代谢紊乱\u2192脂质过氧化\u2192细胞衰老的完整级联；反之，抑制ACSL4或敲除铁调素调控基因可减轻衰老表型、延长健康寿命。这一研究首次在高等哺乳动物中确立了ferro-aging作为生理性衰老驱动因素的地位，其时间尺度为月至年级，属于慢性衰老过程。'));
children.push(pNoIndent([
  new TextRun({ text: '本项目对铁衰老概念的界定：', size: 24, font: '宋体', bold: true }),
  new TextRun({ text: '本项目中的铁衰老特指缺血再灌注后由亚致死铁死亡压力驱动的应激诱导早熟性衰老（ischemia-induced iron-dependent SIPS），与Liu等在自然衰老模型中描述的慢性ferro-aging在时间尺度、触发因素和生理背景上存在本质差异。前者是急性病理损伤后的继发性事件（小时至数天级），后者是生理性衰老过程的慢性积累（月至年级）。尽管存在这些差异，二者共享核心的分子通路——ACSL4/脂质过氧化/p53轴，均以铁依赖性脂质过氧化为上游驱动力。本项目的研究重点是探索这一分子轴在急性脑缺血再灌注损伤中的病理作用，验证铁依赖性SIPS是否是连接急性期铁死亡与慢性期衰老相关炎症的关键中间环节。', size: 24, font: '宋体' })
]));
children.push(p('4-羟基壬烯醛（4-hydroxynonenal, 4-HNE）是\u03c9-6多不饱和脂肪酸过氧化的主要毒性醛类产物之一，具有高度亲电性，可与蛋白质的半胱氨酸、组氨酸和赖氨酸残基发生Michael加成反应，形成稳定的蛋白质羰基化修饰[14]。4-HNE在细胞内具有浓度依赖性的生物学效应：低浓度时可作为信号分子参与多种信号通路的调控，高浓度时则引发细胞毒性和细胞死亡。在铁死亡过程中，4-HNE是脂质过氧化的重要下游产物和标志物，其水平与铁死亡严重程度正相关。近年来，4-HNE在细胞衰老中的作用逐渐被认识。Monroe等[2]在Aging Cell发表的研究系统证实，4-HNE等脂质过氧化产物可在人成纤维细胞和小鼠脂肪干细胞中剂量依赖性地诱导衰老表型，包括SA-\u03b2-gal阳性率升高、\u03b3H2AX焦点积累、p53磷酸化增强、p21表达上调以及SASP因子分泌增加。这些发现提示，脂质过氧化产物4-HNE可能是连接铁死亡与细胞衰老的关键分子介质。'));
children.push(p('p53作为细胞应激反应的核心转录因子，在铁死亡和细胞衰老中均发挥关键调控作用，是连接二者的理想候选分子。在铁死亡调控方面，p53具有双重角色：一方面，p53可通过转录抑制SLC7A11（System Xc\u207b的催化亚基）促进铁死亡；另一方面，p53也可通过调控铁代谢相关基因影响细胞对铁死亡的敏感性。在细胞衰老调控方面，p53/p21轴是经典的衰老启动通路，DNA损伤和氧化应激均可激活p53，进而上调p21诱导细胞周期停滞。值得注意的是，4-HNE可通过对p53的羰基化修饰影响其活性和稳定性——4-HNE修饰的p53呈现出构象改变和半衰期延长，可能导致p53靶基因的持续激活。基于这些文献线索，我们推测存在一个4-HNE-p53-SLC7A11正反馈环路：铁死亡产生的4-HNE修饰并激活p53，活化的p53转录抑制SLC7A11，进一步削弱GSH合成和抗氧化防御，加剧脂质过氧化和4-HNE生成，最终将亚致死性氧化应激转化为持续性的衰老表型。这一环路如果得到验证，将为理解铁死亡如何驱动SIPS提供分子水平的机制解释。'));
children.push(p('然而，目前关于铁死亡驱动细胞衰老的研究主要集中于肿瘤细胞和体外细胞系，在神经系统特别是缺血性脑损伤中的研究极为有限。CIRI半暗带中是否确实存在铁依赖性SIPS？其时空分布特征如何？4-HNE-p53-SLC7A11分子轴是否在其中发挥关键作用？这些问题目前均缺乏直接的实验证据，有待系统研究加以阐明。'));

children.push(h3('5. Nrf2通路：铁死亡防御与衰老抑制的共同枢纽'));
children.push(p('核因子E2相关因子2（nuclear factor erythroid 2-related factor 2, Nrf2）是细胞抗氧化反应的主调控因子，属于碱性亮氨酸拉链转录因子家族。在生理状态下，Nrf2被Keap1（Kelch样ECH关联蛋白1）锚定在胞浆中，并经Cullin-3介导的泛素-蛋白酶体途径快速降解，保持较低的基础水平。当细胞暴露于氧化应激或亲电试剂时，Keap1分子中的关键半胱氨酸残基（如Cys151、Cys273、Cys288）发生共价修饰，导致Keap1构象改变，丧失对Nrf2的泛素连接酶活性。新生的Nrf2逃脱降解并转位入核，与小Maf蛋白形成异二聚体，结合于抗氧化反应元件（antioxidant response element, ARE）的保守序列上，启动下游数百个靶基因的转录。Nrf2靶基因涵盖了抗氧化酶、解毒酶、物质代谢转运、蛋白质稳态、铁代谢调控等多个功能类别，共同构成细胞的综合性防御体系。'));
children.push(p('Nrf2通路是铁死亡的重要防御机制。Nrf2可通过多层面的调控抑制铁死亡发生：在抗氧化层面，Nrf2上调GPX4的表达直接增强磷脂过氧化物清除能力，同时上调SLC7A11促进胱氨酸摄取和GSH合成，间接维持GPX4活性；在铁代谢层面，Nrf2转录激活铁蛋白重链（FTH1）和铁蛋白轻链（FTL），促进游离铁的螯合储存，还可通过上调铁输出蛋白FPN1降低细胞内不稳定铁池水平；在脂质代谢层面，Nrf2调控ACSL4等脂质重塑相关酶的表达，影响膜磷脂的脂肪酸组成，降低细胞对铁死亡的敏感性[12]。这些多维度的调控使Nrf2成为铁死亡防御网络中的核心节点。'));
children.push(p('Nrf2同样是抑制细胞衰老和SASP的关键因子。多项研究表明，Nrf2的表达和活性随年龄增长而下降，这种下降与衰老表型的出现和年龄相关疾病的发生发展密切相关。在细胞水平，Nrf2缺失或功能低下可加速氧化应激诱导的SIPS，表现为SA-\u03b2-gal阳性率升高、端粒缩短加速、SASP分泌增强；反之，Nrf2激活则可延缓多种细胞类型的衰老进程，减轻SASP相关的炎症反应。Nrf2抑制衰老的机制包括：直接清除活性氧减轻DNA损伤、上调II相解毒酶增强细胞防御、抑制NF-\u03baB通路减少SASP因子转录、调节自噬-溶酶体通路维持蛋白质稳态等。鉴于Nrf2在铁死亡防御和衰老抑制中的双重核心作用，我们认为Nrf2通路是一个理想的干预靶点——激活Nrf2有望同时阻断铁死亡的急性期损伤和铁依赖性SIPS的慢性期损害，实现对CIRI病理进程的双阶段干预。'));

children.push(h3('6. 壮瑶药艾叶/桂艾及其活性成分\u03b2-石竹烯：从民族药经验到现代药理机制'));
children.push(p('艾叶（Artemisia argyi L\u00e9vl. et Vant.）为菊科蒿属植物的干燥叶，是我国传统中药和广西道地壮瑶药材。艾叶入药历史悠久，始载于《名医别录》，被列为中品，其性温，味辛、苦，归肝、脾、肾经，具有温经止血、散寒止痛、外用祛湿止痒等功效。在壮瑶医药理论体系中，艾叶具有独特的地位：壮语称\u300c挨\u300d，瑶语称\u300c各艾\u300d，被视为\u300c通龙路火路、除风毒寒毒、逐湿邪\u300d之要药。壮医\u300c三道两路\u300d理论认为，脑（巧坞）为神明之府，龙路火路网络密布；中风（麻邦）是由于风毒、火毒、痧毒循龙路火路上攻巧坞，致三道两路不通、气血失衡、天地人三气不能同步。艾叶的\u300c通龙路火路、除毒邪\u300d功效，恰中中风病的病机要害，因此在壮瑶医临床中常用于麻痹、头痛、眩晕等脑病的防治，常用方法包括艾熏、艾灸、煎汤内服等。艾叶的\u300c解毒除蛊\u300d功效，可从现代药理学角度解读为清除自由基、抗炎、调节细胞死亡与免疫功能等多重作用。'));
children.push(p('桂艾是产于广西地区的艾叶道地药材，因独特的地理气候条件而具有挥发油含量高、活性成分丰富的特点。桂艾挥发油是艾叶的主要药效物质基础，采用水蒸气蒸馏法或超临界CO2萃取法制备，得油率约为0.45%-1.2%。已从桂艾挥发油中分离鉴定出数十种化学成分，主要包括倍半萜类、单萜类、黄酮类及酚酸类化合物，其中\u03b2-石竹烯（\u03b2-caryophyllene, BCP）的含量可高达15%-35%，是桂艾挥发油中最重要的倍半萜成分之一。\u03b2-石竹烯是一种天然存在的双环倍半萜化合物，广泛分布于多种植物的挥发油中，具有独特的丁香香气。由于其良好的安全性，BCP已被美国FDA和欧洲食品安全局（EFSA）批准为食品添加剂，广泛应用于食品、化妆品和香料工业。'));
children.push(p('现代药理学研究揭示了BCP的多种生物活性，包括抗炎、抗氧化、镇痛、抗肿瘤、保肝、护胃以及神经保护等。其中，抗炎和抗氧化活性是BCP最受关注的药理作用。BCP是大麻素CB2受体的选择性激动剂，可通过CB2受体依赖的信号通路发挥抗炎和免疫调节作用。在抗氧化方面，近年来的研究发现BCP可显著激活Nrf2/ARE信号通路，上调多种抗氧化酶和Ⅱ相解毒酶的表达。具体而言，BCP可通过修饰Keap1的半胱氨酸残基或激活PI3K/Akt、ERK等上游信号激酶，促进Nrf2核转位和转录激活，进而上调HO-1、NQO1、GCLC、GCLM等靶基因的表达，增强细胞的抗氧化防御能力。'));
children.push(p('在神经系统疾病领域，BCP的神经保护作用已在多种模型中得到验证。Hu等[3]在Phytomedicine发表的研究明确证实，BCP可通过激活Nrf2/HO-1通路减轻大鼠脑缺血再灌注损伤：BCP预处理可剂量依赖性地减小MCAO/R模型的梗死体积、降低脑水肿程度、改善神经功能评分，同时伴随脑组织中Nrf2核转位增加、HO-1表达上调以及氧化应激指标（MDA、SOD）的改善；而Nrf2抑制剂ML385可显著逆转BCP的神经保护效应。此外，BCP在帕金森病、阿尔茨海默病、癫痫、糖尿病神经病变等神经系统疾病模型中也显示出良好的保护作用。近年来，BCP与铁死亡的关系开始受到关注。有研究报道BCP可通过激活Nrf2通路抑制肝纤维化模型中的铁死亡[15]，在顺铂耳毒性模型中也观察到BCP对铁死亡的抑制作用[16]。这些研究为BCP调控铁死亡提供了初步的实验支持。然而，BCP是否能够通过激活Nrf2通路同时阻断铁死亡和铁依赖性SIPS，是否能够通过干预铁死亡-衰老正反馈环路改善CIRI的远期预后，目前尚未见文献报道。'));

children.push(h3('7. 本项目拟解决的核心科学问题及研究意义'));
children.push(p('综上所述，脑缺血再灌注损伤是急性缺血性脑卒中治疗中的重大临床难题，其病理机制涉及急性期多种调节性细胞死亡和慢性期神经炎症与修复障碍。铁死亡作为急性期神经元死亡的关键形式，已得到广泛认可；细胞衰老则被认为是缺血半暗带慢性化的重要推手。然而，二者之间的因果联系——即铁死亡是否以及如何驱动缺血半暗带中的细胞衰老（铁依赖性SIPS）——目前尚不清楚。基于文献线索和我们前期的网络药理学研究，我们推测：CIRI半暗带中存在亚致死性铁死亡压力驱动的铁依赖性SIPS，4-HNE-p53-SLC7A11正反馈环路是其核心分子机制；Nrf2作为铁死亡防御与衰老抑制的共同枢纽，是理想的干预靶点；广西道地壮药桂艾的活性成分\u03b2-石竹烯可通过激活Nrf2通路，同时阻断铁死亡急性期损伤和铁依赖性SIPS慢性损害，从而改善CIRI远期预后。'));
children.push(p('本项目拟围绕上述科学假说，综合运用动物模型、细胞生物学、分子生物学、网络药理学等多学科技术手段，系统验证CIRI中铁依赖性SIPS的存在，解析其分子调控机制，明确\u03b2-石竹烯通过Nrf2通路干预铁依赖性SIPS的药效与机制。本研究的理论意义在于：提出并验证缺血诱导的铁依赖性SIPS这一新型病理概念，丰富对CIRI慢性化机制的认识，为理解铁死亡与细胞衰老的交互作用提供新的实验证据。实践意义在于：阐明壮瑶药桂艾活性成分\u03b2-石竹烯的神经保护新机制，为开发以铁衰老为靶点的脑卒中治疗药物提供理论依据和先导化合物，也为民族医药现代化研究提供可借鉴的研究范式。'));

children.push(h2('（二）项目组网药研究基础'));

children.push(p('为验证上述科学假说，项目组前期整合多维度公开数据集与生物信息学方法，系统开展了脑缺血-铁衰老-石竹烯的网络药理学与机器学习研究，完成了从疾病特征识别到药物靶点预测的全链条计算分析，为后续实验验证奠定了坚实的前期基础。'));

children.push(h3('1. 多数据集铁衰老转录特征的识别与验证'));

children.push(pNoIndent([
  new TextRun({ text: '（1）四数据集铁衰老评分的疾病关联性', size: 24, font: '宋体', bold: true })
]));
children.push(p('本研究整合了4个脑缺血时间进程数据集——GSE104036（小鼠MCAO，RNA-seq，27样本，0-72h）、GSE16561（人缺血性脑卒中，Illumina微阵列，63样本）、GSE61616（大鼠MCAO，Affymetrix微阵列，15样本）及GSE97537（大鼠MCAO，Affymetrix微阵列，12样本）。采用单样本基因集富集分析（ssGSEA），基于Barbie等（Nature Protocols, 2009）[9]的秩加权富集统计算法，计算每个样本的铁死亡、细胞衰老及铁衰老（96基因集）评分。结果显示，在全部4个数据集中，铁衰老评分的疾病-对照效应量（Cohen\'s d）均大于铁死亡评分和衰老评分，表明铁衰老基因集捕获的转录信号与缺血性脑损伤的关联最为紧密，在跨物种、跨平台中具有稳健性。在GSE104036小鼠MCAO模型中，同侧脑组织铁衰老评分为0.167\u00b10.032，显著高于假手术组（0.113\u00b10.001），效应量Cohen\'s d = 1.84（P = 4.40 \u00d7 10\u207b\u00b3），同时高于对侧组（0.118\u00b10.016），d = 1.94（P = 3.84 \u00d7 10\u207b\u2074）。'));

children.push(pNoIndent([
  new TextRun({ text: '（2）时序特征与铁死亡-衰老关系重定义', size: 24, font: '宋体', bold: true })
]));
children.push(p('在GSE104036小鼠MCAO模型中，铁死亡与铁衰老评分均在再灌注后6小时达到峰值，随后下降；而经典的细胞衰老评分并未在后期时间点升高，反而在急性期至亚急性期呈下降趋势。时序分析显示，同侧铁衰老评分随再灌注时间呈递增趋势：3小时为0.144\u00b10.016，6小时升至0.169\u00b10.002，12小时略降至0.156\u00b10.037，24小时达峰值0.200\u00b10.038。Spearman秩相关分析显示时间与铁衰老评分呈正相关趋势（\u03c1 = 0.497，P = 0.101），与缺血后铁死亡级联反应的时程特征一致。这一发现不支持经典的铁死亡上升\u2192衰老上升\u2192铁衰老过渡序列模式。据此，本项目将分析目标重新定义为高铁衰老活性状态而非过渡窗口，即铁衰老是一个急性的铁死亡相关转录状态，而非迟发性衰老转变。这一发现为后续实验研究指明了方向——铁依赖性SIPS可能在再灌注后数小时至数天内启动，与急性铁死亡在时间上存在重叠而非先后序列关系。'));

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
  new TextRun({ text: '（2）PPI网络拓扑特征与功能模块解析', size: 24, font: '宋体', bold: true })
]));
children.push(p('基于STRING v12.0构建的核心基因PPI子网包含311个节点及1,867条边，网络密度为0.039，平均聚类系数为0.458，提示存在显著的模块化结构。最大连通分量包含306个节点，平均最短路径长度为2.91，网络直径为8，符合小世界网络特征。度中心性排名前10的Hub基因为：TP53（Degree=116）、EGFR（66）、EP300（63）、STAT3（63）、IL6（52）、TNF（51）、HSP90AA1（47）、H3C12（46）、H3C13（46）、SIRT1（45）。其中TP53作为度最高的节点，同时介数中心性亦最高（0.217），是网络信息传递的核心枢纽。NFE2L2（Nrf2）度中心性为38，处于网络核心位置，其下游靶基因GPX4、FTH1、SLC7A11等均在核心网络中。'));
children.push(p('采用贪心模块度算法在核心PPI子网中识别出8个紧密连接的功能模块：模块1（TP53种子，83基因，转录调控/细胞周期/应激反应，含E2F、FOXO、Nrf2通路）、模块2（EGFR种子，59基因，炎症免疫/细胞因子信号/NLRP3炎症小体）、模块3（MTOR种子，44基因，自噬-溶酶体通路/线粒体质量控制/铁自噬）、模块4（CAV1种子，30基因，铁代谢调控/脂质代谢）、模块5（HSP90AA1种子，28基因，氧化应激/分子伴侣）、模块6（H3C12种子，27基因，表观遗传调控/组蛋白修饰）、模块7（PRKCA种子，15基因，信号转导/RNA结合）及模块8（ALOX15种子，7基因，花生四烯酸代谢/脂氧合酶/脂质过氧化执行）。模块4（铁代谢）与模块3（自噬/铁自噬）直接呼应铁死亡的核心病理机制，模块8（ALOX15）直接参与脂质过氧化执行，模块1（含Nrf2通路）是抗氧化防御的核心。功能富集分析进一步支持上述结论：KEGG通路富集排名前列的包括细胞衰老（29基因，adjusted P = 5.47 \u00d7 10\u207b\u00b2\u00b9）、TNF信号通路（38基因）、IL-17信号通路（28基因）、细胞凋亡（33基因）及NF-\u03baB信号通路（27基因）等。WikiPathways中，Ferroptosis通路（WP4313，40基因）排名第4位（adjusted P = 9.19 \u00d7 10\u207b\u207b\u00b2\u00b3），进一步支持核心基因集与铁死亡机制的高度关联。'));

children.push(h3('3. 免疫浸润与铁衰老的协同激活及WGCNA验证'));
children.push(p('基于特征基因集的ssGSEA免疫浸润分析显示，在GSE104036的27个样本中，铁衰老评分与多种免疫细胞丰度存在显著相关。其中，中性粒细胞（r = 0.651，P = 2.3 \u00d7 10\u207b\u2074）和M2型巨噬细胞（r = 0.613，P = 6.7 \u00d7 10\u207b\u2074）与铁衰老评分呈强正相关；而小胶质细胞稳态标志（r = -0.738，P = 1.1 \u00d7 10\u207b\u2075）和星形胶质细胞（r = -0.567，P = 0.002）呈显著负相关。在检测的18个关键炎症因子中，15个与铁衰老评分呈显著正相关，排名前列的包括Ccl2（r = 0.903）、Icam1（r = 0.890）、Cxcl10（r = 0.877）、Stat3（r = 0.875）及Il1b（r = 0.847），强烈支持铁衰老与神经炎症的协同激活机制。这一发现提示，铁依赖性SIPS不仅是细胞自主的过程，还可能通过SASP分泌招募外周免疫细胞、激活胶质细胞，形成炎症-衰老正反馈环路。'));
children.push(p('在样本量最大的人脑缺血数据集GSE16561（63样本）中，加权基因共表达网络（WGCNA）验证显示，337个核心基因中有97个被分配至有意义的共表达模块。其中turquoise模块包含的核心基因最多，且与铁衰老表型显著相关。核心基因在turquoise模块中的平均模块身份（MM）为0.936，平均基因显著性（GS）为0.236，证实核心基因集在人脑缺血的共表达调控网络中处于核心位置，具有高模块身份和高表型相关性。'));

children.push(h3('4. 核心科学假说的提出'));
children.push(p('基于上述国内外研究现状和本项目组前期网络药理学研究基础，我们提出以下科学假说：\u03b2-石竹烯通过激活Nrf2通路，上调GPX4、FTH1和SLC7A11等下游靶基因，增强细胞抗氧化防御能力，减少4-HNE生成，解除4-HNE对p53的修饰和活化，从而阻断缺血诱导的铁依赖性SIPS的启动和SASP的分泌，打破铁死亡\u21924-HNE\u2192p53\u2192SLC7A11\u2193\u2192更多铁死亡的正反馈环路，最终改善脑缺血再灌注损伤的远期预后。桂艾挥发油因多成分整合可能呈现一定程度的药效增益。本项目拟通过系统的体内外实验验证这一假说，为脑缺血再灌注损伤的治疗提供新的靶点和候选药物。'));

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
  new TextRun({ text: '具体目标：', size: 24, font: '宋体', bold: true })
]));
children.push(bullet('证实CIRI半暗带中存在缺血诱导的铁依赖性SIPS，并明确4-HNE-p53-SLC7A11分子轴在其中的关键作用。'));
children.push(bullet('阐明BCP通过Nrf2通路阻断铁依赖性SIPS的细胞分子机制，明确其作用靶点和信号调控网络。'));
children.push(bullet('在整体动物水平验证BCP改善CIRI远期预后的药效，并确认其神经保护效应的Nrf2依赖性。'));
children.push(bullet('初步探索桂艾挥发油多成分的药效增益现象，为民族药现代化研究提供实验依据。'));

children.push(h1('三、研究内容'));

children.push(h2('研究内容一：CIRI中铁依赖性SIPS的时空特征及分子轴研究'));

children.push(pNoIndent([
  new TextRun({ text: '1. 铁依赖性SIPS的时空调定位', size: 24, font: '宋体', bold: true })
]));
children.push(p('本部分旨在明确CIRI后半暗带中是否存在铁依赖性SIPS，及其时空分布特征和细胞类型。以C57BL/6J小鼠为研究对象，采用改良线栓法制备MCAO/R模型（缺血60 min后再灌注），设置假手术组作为对照。在再灌注后24 h、3 d、7 d、28 d四个时间点进行脑组织取材，涵盖急性期到慢性期的完整时间窗。采用免疫荧光多重标记技术，分别标记神经元（NeuN）、星形胶质细胞（GFAP）、小胶质细胞（Iba-1），结合铁死亡标志物（GPX4、4-HNE、FTH1）和衰老标志物（p21、\u03b3H2AX、SA-\u03b2-gal），通过激光共聚焦显微镜观察和定量分析，明确共定位细胞的比例、分布区域（核心区、半暗带、对侧区）及时间动力学。利用透射电镜观察半暗带细胞的超微结构，重点观察线粒体形态（铁死亡典型的皱缩、膜密度增高 vs 衰老细胞的特征性改变）和铁沉积情况。采用铁含量比色法和普鲁士蓝染色检测组织铁水平的时空变化。综合上述结果，绘制CIRI中铁依赖性SIPS的时空分布图谱，为后续机制研究和药物干预提供时间和空间靶点依据。'));

children.push(pNoIndent([
  new TextRun({ text: '2. 4-HNE-p53-SLC7A11分子轴的在体验证', size: 24, font: '宋体', bold: true })
]));
children.push(p('本部分旨在验证4-HNE-p53-SLC7A11正反馈环路是否是铁死亡驱动SIPS的核心分子机制。在上述MCAO/R模型的不同时间点，取半暗带脑组织，采用多种分子生物学技术进行时序检测：采用比色法和ELISA检测4-HNE蛋白加合物水平；采用Western Blot和免疫组化检测p53的总表达、磷酸化水平及核转位情况；检测p21、p16等衰老标志物以及SLC7A11、GPX4等铁死亡相关蛋白的表达变化；采用qRT-PCR检测IL-6、IL-1\u03b2、TNF-\u03b1等SASP因子的mRNA水平。关键实验是采用免疫沉淀（IP）联合抗4-HNE抗体或抗DNP（二硝基苯肼，检测蛋白质羰基化）抗体，检测p53的羰基化修饰水平，明确4-HNE是否在缺血后对p53进行共价修饰。技术难点与对策：4-HNE修饰的检测在组织样本中由于4-HNE的高反应性和不稳定性，容易产生假阴性。为克服这一问题，所有组织样本在取材后将立即用含5 mM DTT和蛋白酶抑制剂的裂解液处理，并在氮气保护下进行后续操作；同时设置4-HNE处理的细胞裂解液作为阳性对照，确保IP/IB体系的灵敏度。此外，通过脑立体定位注射AAV-shACSL4下调半暗带ACSL4表达，观察铁死亡和SIPS标志物的变化，验证ACSL4作为铁死亡上游调控分子在铁依赖性SIPS启动中的作用。'));

children.push(h2('研究内容二：BCP抗铁依赖性SIPS的细胞机制研究'));

children.push(pNoIndent([
  new TextRun({ text: '1. 体外铁依赖性SIPS模型的建立与BCP药效评价', size: 24, font: '宋体', bold: true })
]));
children.push(p('本部分旨在建立稳定的体外铁依赖性SIPS细胞模型，并评价BCP的干预效果。采用新生24 h内C57BL/6J小鼠的原代皮层神经元和星形胶质细胞为研究对象，分别使用低剂量铁死亡诱导剂Erastin（0.5 \u03bcM）和氧糖剥夺/复氧（OGD/R，2 h OGD + 24 h复氧）两种方法诱导亚致死量铁死亡压力。通过优化诱导剂浓度和处理时间，建立细胞死亡率<15%但出现典型衰老表型的SIPS模型。衰老表型的鉴定指标包括：SA-\u03b2-gal染色阳性率、p21/p16蛋白和mRNA表达水平、SASP因子（IL-6、IL-1\u03b2、MMP3等）分泌、增殖能力检测（BrdU掺入或Ki67染色）以及细胞周期分析。在成功建立模型的基础上，设置BCP不同浓度组（1、10、50 \u03bcM），观察其对铁依赖性SIPS的干预效果。以铁死亡特异性抑制剂Liproxstatin-1（200 nM）和铁螯合剂DFO（100 \u03bcM）作为工具对照，以确认BCP的作用是否通过抑制铁死亡实现。'));

children.push(pNoIndent([
  new TextRun({ text: '2. Nrf2在BCP抗铁依赖性SIPS中的关键作用验证', size: 24, font: '宋体', bold: true })
]));
children.push(p('本部分旨在明确BCP的抗SIPS效应是否依赖于Nrf2通路的激活。采用功能获得和功能缺失策略：在Nrf2过表达（质粒转染）和Nrf2沉默（siRNA干扰或药理学抑制剂ML385）的细胞中，观察BCP的抗SIPS效应是否增强或减弱。检测指标包括：Nrf2核转位水平（免疫荧光、核浆分离Western Blot）、下游靶基因（GPX4、HO-1、FTH1、SLC7A11、NQO1）的mRNA和蛋白表达水平、铁死亡相关指标（细胞内Fe\u00b2\u207a、脂质ROS、GSH/GSSG比值、MDA含量）以及衰老相关指标（SA-\u03b2-gal阳性率、p21/p16表达、SASP分泌）。为进一步探究BCP激活Nrf2的上游机制，进行以下实验：采用分子对接预测BCP与Keap1疏水口袋的结合模式（PDB: 4IQK）；采用细胞热位移分析（CETSA）验证BCP与Keap1的潜在结合。若CETSA结果不理想，采用分子对接+点突变验证的备用策略——预测BCP与Keap1疏水口袋的关键结合残基，通过点突变破坏结合位点，观察BCP是否还能激活Nrf2，从而间接验证BCP-Keap1的相互作用。'));

children.push(pNoIndent([
  new TextRun({ text: '3. BCP对4-HNE-p53-SLC7A11分子轴的调控', size: 24, font: '宋体', bold: true })
]));
children.push(p('本部分旨在解析BCP调控铁依赖性SIPS的下游分子机制，重点验证BCP\u2192Nrf2\u2192GPX4\u2191\u21924-HNE\u2193\u2192p53活化\u2193\u2192SLC7A11\u2191这一信号轴。在铁依赖性SIPS细胞模型中，检测BCP处理后不同时间点的4-HNE含量、p53羰基化水平、p53核转位、SLC7A11表达的时序变化，分析各指标间的相关性。采用挽救实验验证信号通路的上下游关系：在GPX4抑制剂（RSL3）处理的细胞中，观察BCP对4-HNE和p53的调控是否被逆转；在p53过表达细胞中，观察BCP对SLC7A11和SIPS的调控是否减弱。通过这些实验，系统验证BCP通过Nrf2-GPX4-4-HNE-p53-SLC7A11通路解除铁死亡-衰老正反馈环路的分子机制。延伸实验（条件允许时开展）：在p53敲低/过表达细胞中进一步验证BCP的抗SIPS效应是否依赖p53通路；通过质谱鉴定p53的4-HNE修饰位点，并构建位点突变体质粒，验证羰基化修饰对p53活性和功能的影响。'));

children.push(h2('研究内容三：BCP改善CIRI远期预后的整体药效与机制验证'));

children.push(pNoIndent([
  new TextRun({ text: '1. BCP对MCAO/R模型的整体药效评价', size: 24, font: '宋体', bold: true })
]));
children.push(p('本部分旨在整体动物水平验证BCP对CIRI的神经保护效应，重点关注远期功能预后。采用C57BL/6J小鼠MCAO/R模型，设置以下实验组：假手术组、模型组、BCP低剂量组（102 mg/kg）、BCP中剂量组（204 mg/kg）、BCP高剂量组（408 mg/kg）、Liproxstatin-1阳性药组（10 mg/kg）、BCP+ML385组（30 mg/kg），每组12只动物。给药方案：再灌注后立即腹腔注射首次给药，随后每日灌胃给药，连续14天。短期评价（再灌注后1-3 d）：TTC染色测定梗死体积、干湿重法测定脑水肿程度、mNSS神经功能评分。远期评价（再灌注后28 d）：采用多种行为学实验全面评估神经功能恢复情况，包括足误实验（前肢感觉运动功能）、转棒实验（运动协调能力）、Morris水迷宫（空间学习记忆能力）。通过短期和远期指标的综合评价，明确BCP是否不仅能减轻急性期损伤，更能改善远期功能预后。'));

children.push(pNoIndent([
  new TextRun({ text: '2. Nrf2依赖性的在体验证与铁依赖性SIPS抑制的机制确认', size: 24, font: '宋体', bold: true })
]));
children.push(p('本部分旨在在整体动物水平验证BCP神经保护效应的Nrf2依赖性，并确认其对铁依赖性SIPS的在体抑制作用。使用Nrf2基因敲除小鼠（C57BL/6J背景），设置以下实验组：野生型+假手术组、野生型+模型组、野生型+BCP组、Nrf2\u207b/\u207b+模型组、Nrf2\u207b/\u207b+BCP组、Nrf2\u207b/\u207b+溶媒组，每组10-12只。比较各组的梗死体积、脑水肿程度以及远期行为学功能恢复情况，验证BCP的神经保护效应是否依赖于Nrf2通路。在各组28 d脑组织标本中，进行以下检测以确认BCP对铁依赖性SIPS的在体抑制效应：SA-\u03b2-gal染色评估衰老细胞负荷、免疫荧光共定位检测铁死亡-衰老双阳性细胞比例、Western Blot检测p53羰基化水平及衰老标志物（p21、p16）和SASP因子的表达、铁死亡相关指标（4-HNE、GPX4、FTH1）检测。通过这些实验，将BCP的整体药效与铁依赖性SIPS抑制以及Nrf2通路激活直接关联起来。'));

children.push(pNoIndent([
  new TextRun({ text: '3. 桂艾挥发油与BCP单体的药效初步比较', size: 24, font: '宋体', bold: true })
]));
children.push(p('本部分作为延伸目标，旨在初步探索桂艾挥发油多成分的药效增益现象，为后续深入研究民族药配伍机制提供线索。在细胞水平（铁依赖性SIPS模型），头对头比较桂艾挥发油（按其中BCP含量折算等剂量）与BCP单体的药效差异，检测SA-\u03b2-gal阳性率、p21表达、脂质过氧化水平等指标。在动物水平（MCAO/R模型），设置桂艾挥发油组与等剂量BCP单体组，初步比较梗死体积、神经功能评分和SIPS标志物的差异。通过上述比较，评估是否存在药效增益现象。若观察到药效增益，将为后续深入研究桂艾挥发油多成分协同作用机制提供实验依据，也为壮瑶药多成分协同作用的现代科学阐释奠定基础。'));

children.push(h1('四、拟解决的关键科学问题'));
children.push(p('1. CIRI半暗带中铁依赖性SIPS的存在性及其4-HNE-p53-SLC7A11分子轴的阐明。这一问题的解决将为理解缺血性脑损伤从急性期向慢性期演变的机制提供新视角，丰富铁死亡与细胞衰老交互作用的理论体系。'));
children.push(p('2. BCP通过Nrf2通路阻断铁依赖性SIPS、改善CIRI远期预后的分子机制解析。这一问题的解决将为开发以铁衰老为靶点的脑卒中治疗药物提供先导化合物和理论基础，也为天然小分子同时靶向急性死亡和慢性衰老的干预策略提供实验范式。'));
children.push(p('3. 壮瑶药桂艾多成分药效增益现象的初步验证。这一问题的探索将为民族药现代化研究提供新的思路，有助于揭示壮瑶药传统经验的现代科学内涵。'));

children.push(h1('五、研究方案'));

children.push(h2('5.1 实验材料'));
children.push(bullet('实验动物：SPF级C57BL/6J小鼠（22-25 g，雄性），购自北京维通利华实验动物技术有限公司；Nrf2基因敲除小鼠（C57BL/6J背景）购自赛业生物。动物饲养于屏障环境，12 h光暗循环，自由摄食饮水。'));
children.push(bullet('药物与试剂：\u03b2-石竹烯（Sigma-Aldrich，纯度\u226598.5%）、桂艾挥发油（本项目提取，经GC-MS鉴定）、Erastin（MCE）、RSL3（MCE）、Liproxstatin-1（MCE）、DFO（Sigma）、ML385（MCE）、4-HNE（Sigma）、CCK-8试剂盒（同仁化学）、SA-\u03b2-gal染色试剂盒（Cell Signaling）、C11-BODIPY 581/591（Invitrogen）。'));
children.push(bullet('主要抗体：抗-GPX4、抗-ACSL4、抗-FTH1、抗-TFR1、抗-p53、抗-磷酸化p53（Ser15）、抗-p21、抗-p16、抗-\u03b3H2AX、抗-4-HNE、抗-DNP、抗-Nrf2、抗-Keap1、抗-HO-1、抗-SLC7A11、抗-NeuN、抗-GFAP、抗-Iba-1、抗-\u03b2-actin等，均购自CST、Abcam或Proteintech等知名供应商。'));
children.push(bullet('道地药材：广西道地桂艾（Artemisia argyi）采自广西药用植物园艾叶GAP种植基地，经广西中医药大学中药鉴定教研室鉴定为菊科植物艾的干燥叶。水蒸气蒸馏法提取挥发油，GC-MS进行成分分析。'));

children.push(h2('5.2 主要实验方法'));
children.push(bullet('小鼠MCAO/R模型制备：改良线栓法制备小鼠大脑中动脉阻塞/再灌注模型，缺血60 min后再灌注。激光多普勒血流仪监测脑血流变化确保模型成功。假手术组仅分离颈总动脉不插线。'));
children.push(bullet('原代神经细胞培养：新生24 h内C57BL/6J小鼠皮层神经元和星形胶质细胞的分离与原代培养。神经元采用Neurobasal+B27培养基，星形胶质细胞采用DMEM+10%FBS培养基，纯度鉴定采用MAP2和GFAP免疫荧光染色。'));
children.push(bullet('OGD/R模型：将细胞置于缺氧培养箱（1% O2、5% CO2、94% N2），用无糖Earle\'s液孵育2 h，然后恢复正常培养基和常氧条件继续培养24 h。'));
children.push(bullet('细胞活力检测：CCK-8法，按试剂盒说明书操作，酶标仪测定450 nm吸光度。'));
children.push(bullet('细胞内Fe\u00b2\u007a检测：Phen Green SK探针，流式细胞术检测荧光强度。'));
children.push(bullet('脂质过氧化检测：C11-BODIPY 581/591探针，流式细胞术检测590 nm/510 nm荧光比值。'));
children.push(bullet('GSH/GSSG测定：比色法试剂盒，检测还原型和氧化型谷胱甘肽含量及比值。'));
children.push(bullet('MDA含量测定：TBA法，检测丙二醛含量作为脂质过氧化指标。'));
children.push(bullet('SA-\u03b2-gal染色：细胞衰老\u03b2-半乳糖苷酶染色试剂盒，光学显微镜下计数阳性细胞比例。'));
children.push(bullet('免疫荧光共定位：冰冻切片4%多聚甲醛固定，0.3% Triton X-100透化，5% BSA封闭，一抗4\u2103孵育过夜，荧光二抗室温孵育1 h，DAPI染核，激光共聚焦显微镜观察和图像采集。'));
children.push(bullet('透射电镜：脑组织2.5%戊二醛4\u2103固定，1%锇酸后固定，梯度丙酮脱水，环氧树脂包埋，超薄切片（60-80 nm），醋酸铀-柠檬酸铅双重染色，透射电镜观察。'));
children.push(bullet('Western Blot：RIPA裂解液（含蛋白酶和磷酸酶抑制剂）提取总蛋白，BCA法定量，SDS-PAGE电泳分离，PVDF膜转膜，5%脱脂乳封闭，一抗4\u2103孵育过夜，HRP标记二抗室温孵育1 h，ECL化学发光显影，ImageJ软件进行灰度定量。'));
children.push(bullet('免疫沉淀（IP）：细胞或组织裂解液与抗-p53抗体4\u2103孵育过夜，加入Protein A/G琼脂糖珠沉淀免疫复合物，洗涤后洗脱，进行Western Blot检测，用抗-4-HNE或抗-DNP抗体检测p53的羰基化修饰。'));
children.push(bullet('qRT-PCR：Trizol提取总RNA，反转录合成cDNA，SYBR Green荧光定量PCR检测目标基因mRNA水平，GAPDH为内参，2\u207b\u0394\u0394Ct法计算相对表达量。'));
children.push(bullet('ELISA：组织匀浆或细胞培养上清中IL-6、IL-1\u03b2、TNF-\u03b1含量检测，按试剂盒说明书操作。'));
children.push(bullet('脑立体定位注射：小鼠脑立体定位仪，坐标：前囟后0.5 mm、旁开3.0 mm、深3.5 mm，微量注射泵以0.5 \u03bcL/min速度注射AAV载体，总量2 \u03bcL。注射后留针5 min以防反流。'));
children.push(bullet('Morris水迷宫：圆形水池（直径120 cm），水温（22\u00b11）\u2103，平台置于水面下1 cm。定位航行实验5天（每天4次），记录逃逸潜伏期；第6天撤除平台进行空间探索实验，记录平台穿越次数和目标象限停留时间。'));
children.push(bullet('转棒实验：小鼠置于转棒仪上，转速从4 rpm匀速加速至40 rpm，记录小鼠从转棒上跌落的潜伏期，连续测试3次取平均值。'));
children.push(bullet('足误实验：小鼠在水平栅格（栅条间距1 cm）上自由行走5 min，记录前足误伸入栅格的次数，以总行走步数校正。'));

children.push(h2('5.3 统计学分析'));
children.push(p('所有数据采用GraphPad Prism 9.0和SPSS 26.0软件进行统计分析。计量资料以均数\u00b1标准差（x\u0304 \u00b1 s）表示。两组间比较采用独立样本t检验（满足正态分布和方差齐性）或Mann-Whitney U检验（不满足参数检验条件）。多组间比较采用单因素方差分析（one-way ANOVA），组间两两比较采用Tukey法（各组间均比较）或Dunnett法（与对照组比较）；重复测量数据采用重复测量方差分析。相关性分析采用Pearson相关（参数数据）或Spearman秩相关（非参数数据）。P < 0.05认为差异具有统计学意义。涉及多重比较的实验采用Benjamini-Hochberg法进行错误发现率校正。样本量依据前期预实验结果和文献报道的效应量，采用Power分析确定，确保统计效能达到80%以上。'));

children.push(h1('六、技术路线'));

const techRows = [
  [
    { text: '第一部分\n现象验证\n（第1年）', fill: 'E8F5E9' },
    { text: 'CIRI半暗带铁依赖性SIPS的存在性与时空特征鉴定', fill: 'E8F5E9' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: '\u2193 时间：24 h / 3 d / 7 d / 28 d\n\u2193 空间：核心区 / 半暗带 / 对侧区\n\u2193 细胞：神经元 / 星形胶质细胞 / 小胶质细胞', fill: 'FFFFFF' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: '关键技术：免疫荧光共定位 | SA-\u03b2-gal | 普鲁士蓝 | 透射电镜 | Western Blot | IP/IB', fill: 'FFF3E0' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: '\u2193 分子轴验证：4-HNE\u2192p53\u2192SLC7A11正反馈环路 + AAV-shACSL4功能验证 \u2193', fill: 'FFFFFF' }
  ],
  [
    { text: '第二部分\n机制解析\n（第2年）', fill: 'E3F2FD' },
    { text: 'BCP通过Nrf2通路阻断铁依赖性SIPS的细胞分子机制', fill: 'E3F2FD' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: '\u2193 体外SIPS模型建立 \u2193            \u2193 Nrf2依赖性验证 \u2193            \u2193 分子轴调控 \u2193', fill: 'FFFFFF' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: 'Erastin / OGD/R\nSA-\u03b2-gal | p21/p16 | SASP\n\nNrf2过表达/沉默\n分子对接 | CETSA（探索）\n\n            BCP\u2192Nrf2\u2192GPX4\u2191\u21924-HNE\u2193\u2192p53\u2193\u2192SLC7A11\u2191\n            挽救实验 | 时序分析', fill: 'FFFFFF' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: '\u2193\n阐明BCP通过Nrf2通路阻断铁死亡-衰老正反馈环路的分子机制', fill: 'FFFFFF' }
  ],
  [
    { text: '第三部分\n药效验证\n（第3年）', fill: 'FCE4EC' },
    { text: 'BCP改善CIRI远期预后的整体药效 + 桂艾药效增益探索', fill: 'FCE4EC' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: '\u2193 整体药效 \u2193            \u2193 Nrf2依赖（Nrf2 KO）\u2193            \u2193 桂艾比较（延伸）\u2193', fill: 'FFFFFF' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: 'MCAO/R模型\nTTC | 脑水肿 | mNSS\n转棒 | 足误 | 水迷宫（28d）\n\nNrf2\u207b/\u207b小鼠\n功能 + SIPS标志物验证\n\n            桂艾挥发油 vs BCP单体            \n            （条件允许时开展）', fill: 'FFFFFF' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: '\u2193\n最终结论：BCP通过激活Nrf2抑制铁依赖性SIPS，改善CIRI远期预后', fill: 'F3E5F5' }
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
children.push(p('铁死亡和细胞衰老是当前CIRI研究的两个热点领域，二者的交互作用（铁衰老）是前沿交叉方向，具有坚实的理论基础。铁死亡在CIRI中的作用已被国内外众多实验室独立验证；细胞衰老参与缺血后脑损伤的慢性化进程也得到了越来越多证据的支持；4-HNE作为脂质过氧化产物诱导衰老的效应已在多种细胞类型中得到证实；p53在铁死亡和衰老调控中的双重角色已有充分文献支撑；Nrf2作为二者共同防御枢纽的地位已经确立。本项目组前期通过4个脑缺血数据集的整合分析，进一步证实了铁衰老转录特征在脑缺血中的稳健存在：铁衰老评分在4个数据集中均表现出最大的疾病-对照效应量，五基因CIRI-铁衰老签名在3个独立数据集中得到验证（Spearman \u03c1 = 0.56-0.88）。网络药理学分析预测了BCP靶点与铁死亡调控网络的高度拓扑关联（超几何检验P = 2.48 \u00d7 10\u207b\u2074\u00b3），并初步确定SAT1-Nrf2轴为关键入口节点。最短路径分析显示SAT1通过NFE2L2桥接至铁死亡调控网络，平均距离2.84步。这些计算生物学结果为后续实验验证提供了坚实的理论依据和明确的研究方向。'));

children.push(h2('7.2 材料可行性'));
children.push(p('本研究所需的实验材料均可稳定获取。广西道地桂艾采自广西药用植物园艾叶GAP种植基地，来源明确，品种鉴定可靠，活性成分提取和分析平台成熟。BCP标准品为Sigma-Aldrich商品化试剂，纯度\u226598.5%，质量有保障。Nrf2基因敲除小鼠已商用可购（赛业生物、Jackson Laboratory等），遗传背景清晰，基因型鉴定方法成熟。原代神经细胞培养、MCAO/R模型等所需动物和细胞材料均可稳定获取。STRING v12.0、FerrDb v2、GEO等公共数据库资源开放获取，为网络药理学分析提供了数据支撑。各种分子生物学试剂、抗体、试剂盒均有多家商业供应商可供选择，不存在材料供应障碍。'));

children.push(h2('7.3 技术可行性'));
children.push(p('项目涉及的实验技术均为成熟方法，具有良好的可重复性。动物模型方面，小鼠MCAO/R模型是脑缺血研究的经典模型，技术操作规范成熟，项目组已掌握该模型的制备技术，成功率稳定在75%以上。细胞生物学方面，原代神经元和星形胶质细胞培养、OGD/R模型、质粒转染、siRNA干扰等均为常规技术，方法学成熟可靠。分子生物学方面，Western Blot、qRT-PCR、免疫荧光共定位、免疫沉淀、ELISA等技术在多数生命科学实验室均已常规开展。铁死亡和衰老的特异性检测方法（C11-BODIPY、SA-\u03b2-gal染色等）已有大量文献支持，试剂盒商品化程度高。行为学检测（Morris水迷宫、转棒、足误实验）均为神经科学研究的标准范式。网络药理学分析方面，项目组已建立包括ssGSEA评分、LASSO特征选择、PPI拓扑分析、功能富集、WGCNA共表达网络验证等在内的完整分析流程，具备持续的数据挖掘和分析能力。'));

children.push(h2('7.4 前期工作基础'));
children.push(p('本团队前期已在脑缺血、铁死亡和天然药物神经保护领域积累了扎实的研究基础。在计算生物学层面，已完成4个脑缺血数据集的整合分析，建立了CIRI-铁衰老五基因签名（SAT1、EBF3、KLF6、LIFR、CD74），验证了BCP靶点与铁死亡调控网络的高度拓扑富集（超几何检验P = 2.48 \u00d7 10\u207b\u2074\u00b3）。构建了包含311个节点、1,867条边的核心PPI子网，识别出8个功能模块，完成了免疫浸润和炎症因子相关性分析，以及WGCNA共表达网络验证。在实验层面，项目组已建立小鼠MCAO/R模型和原代神经细胞培养体系，具备开展体内外实验的基础条件。Hu等[3]已证实BCP通过Nrf2/HO-1通路减轻大鼠脑缺血再灌注损伤，为本项目的延伸研究提供了直接的前期支持。此外，项目组与广西中医药大学中药鉴定教研室、广西药用植物园等单位建立了良好的合作关系，可为桂艾药材的来源鉴定和成分分析提供技术支持。'));

children.push(h1('八、特色与创新之处'));

children.push(h2('8.1 理论创新：提出并验证CIRI中铁依赖性SIPS新假说'));
children.push(p('现状不足：现有研究多将铁死亡和细胞衰老视为CIRI中两个独立的病理过程，分别关注急性期和慢性期，对二者之间的因果联系缺乏系统探讨。铁死亡驱动细胞衰老（铁衰老）的概念主要在肿瘤和自然衰老领域被提出，在缺血性脑损伤中的研究几近空白。我们的不同设计：本项目率先提出CIRI半暗带中存在缺血诱导的铁依赖性SIPS的科学假说，将铁死亡的急性损伤与细胞衰老的慢性损害通过4-HNE-p53-SLC7A11分子轴有机联系。独特优势：这一假说将CIRI的急性期损伤机制与慢性期恶化机制贯通起来，为理解缺血性脑损伤的慢性化提供了新的理论视角，也为寻找能够同时干预急慢性阶段的治疗靶点开辟了新思路。与经典ferro-aging概念的明确区分（时间尺度、触发因素、病理背景），体现了概念的精确性和科学严谨性。'));

children.push(h2('8.2 机制创新：揭示BCP通过Nrf2双阻断铁死亡-衰老的新机制'));
children.push(p('现状不足：BCP的神经保护作用已有报道，但研究多停留在宏观药效层面，对其细胞分子机制的解析不够深入；现有研究多关注单一靶点或单一病理过程，缺乏对信号调控网络的系统认识；BCP对铁死亡和衰老的研究多为独立报道，尚未将二者联系起来。我们的不同设计：本项目系统解析BCP通过激活Nrf2通路协同抑制铁死亡与铁依赖性SIPS的分子机制，明确SAT1-Nrf2桥接的关键作用，验证BCP对4-HNE-p53-SLC7A11正反馈环路的调控。独特优势：首次阐明壮瑶药桂艾活性成分BCP通龙路火路、除毒邪功效的铁衰老干预科学内涵，为天然小分子同时靶向急性死亡和慢性衰老提供了新的作用范式，也为脑卒中治疗药物的研发提供了新的候选靶点和先导化合物。'));

children.push(h2('8.3 模式创新：构建壮瑶药-铁衰老的整合研究新模式'));
children.push(p('现状不足：民族药现代化研究常停留在成分鉴定和活性筛选层面，与前沿生物学问题结合不够紧密，难以充分揭示民族药传统经验的科学内涵；传统的网络药理学研究多为描述性分析，与后续实验验证的衔接不够紧密。我们的不同设计：构建基于壮瑶医药理论的道地药材-功效-核心成分-铁衰老靶点-信号通路精准整合研究模式，将壮瑶医\u300c通龙路火路、除毒邪\u300d的传统功效与铁死亡-衰老交互的现代前沿生物学有机结合。前期网药分析的系统布局（多数据集验证、机器学习筛选、PPI拓扑分析、功能模块解析、免疫浸润关联、WGCNA验证）为后续实验提供了明确的靶点和通路方向，体现了从计算预测到实验验证的转化医学研究思路。独特优势：为民族药现代化研究提供了可复制的范例，有助于推动壮瑶医药从经验医学向循证医学的转变，也为从民族药宝库中挖掘创新药物提供了新的研究范式。'));

children.push(h1('九、年度研究计划及预期研究成果'));

children.push(h2('9.1 第一年（2027年1月-2027年12月）'));
children.push(bullet('完成小鼠MCAO/R模型的建立和方法学优化，确保模型稳定性和重复性。'));
children.push(bullet('完成铁依赖性SIPS的时空定位研究（4个时间点、3个脑区、3种细胞类型）。'));
children.push(bullet('完成4-HNE-p53-SLC7A11分子轴的时序变化检测和相关性分析。'));
children.push(bullet('完成AAV-shACSL4载体的包装、立体定位注射和功能验证。'));
children.push(bullet('完成桂艾挥发油的提取和GC-MS成分分析鉴定。'));
children.push(bullet('预期成果：发表SCI收录论文1篇，申请发明专利1项，培养硕士研究生1名。'));

children.push(h2('9.2 第二年（2028年1月-2028年12月）'));
children.push(bullet('建立稳定的体外铁依赖性SIPS细胞模型（神经元和星形胶质细胞）。'));
children.push(bullet('完成BCP抗铁依赖性SIPS的药效评价（剂量效应、时间效应）。'));
children.push(bullet('完成Nrf2在BCP抗铁依赖性SIPS中关键作用的细胞水平验证（过表达/沉默）。'));
children.push(bullet('完成BCP对4-HNE-p53-SLC7A11分子轴调控的系统检测（时序分析+挽救实验）。'));
children.push(bullet('完成分子对接和CETSA探索性实验。'));
children.push(bullet('预期成果：发表SCI收录论文1-2篇，培养硕士研究生1名。'));

children.push(h2('9.3 第三年（2029年1月-2029年12月）'));
children.push(bullet('完成BCP对MCAO/R模型的整体药效评价（短期指标+远期行为学）。'));
children.push(bullet('完成Nrf2基因敲除小鼠的在体验证实验，确认Nrf2依赖性。'));
children.push(bullet('完成脑组织中铁依赖性SIPS标志物的检测，确认在体抑制效应。'));
children.push(bullet('完成桂艾挥发油与BCP单体的药效初步比较（细胞+动物水平，条件允许时）。'));
children.push(bullet('数据整理、统计分析、论文撰写、项目结题验收。'));
children.push(bullet('预期成果：发表高影响力SCI论文1篇，培养研究生2-3名，申请发明专利1项。'));

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
  fs.writeFileSync('D:/铁衰老 绝不重蹈覆辙/标书_国自然标准_final_桂艾BCP靶向Nrf2抑制铁依赖性SIPS改善CIRI.docx', buffer);
  console.log('Done: proposal generated');
  console.log('File: D:/铁衰老 绝不重蹈覆辙/标书_国自然标准_final_桂艾BCP靶向Nrf2抑制铁依赖性SIPS改善CIRI.docx');
});
