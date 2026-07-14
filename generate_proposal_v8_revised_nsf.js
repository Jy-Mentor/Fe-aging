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
children.push(p('脑卒中是全球范围内导致成人死亡和长期残疾的首要原因之一。在我国，脑卒中已成为首位致死和致残病因，每年新发病例超过200万，现存患者逾1300万，其中约70%的存活患者遗留不同程度的神经功能缺损，给家庭和社会带来沉重的疾病负担。急性缺血性脑卒中约占全部脑卒中的70%-80%，其治疗核心在于尽早恢复血流灌注。然而，静脉溶栓时间窗仅为发病后4.5小时，血管内治疗虽延长至6-24小时，但即便成功实现血管再通，仍有近半数患者预后不良，提示脑缺血再灌注损伤（cerebral ischemia-reperfusion injury, CIRI）是阻碍患者获益的关键因素。'));
children.push(p('CIRI病理机制极为复杂，涉及氧化应激爆发、兴奋性毒性、神经炎症、血脑屏障破坏以及多种形式的调节性细胞死亡。这些病理过程相互交织、互为因果，形成复杂的损伤网络，使得单一靶点干预策略往往难以奏效。从时间维度看，CIRI呈现明显的阶段特征：急性期以兴奋性毒性和快速细胞死亡为主，亚急性期以神经炎症和进行性细胞死亡为特点，慢性期则进入神经修复与重塑阶段，但慢性炎症常阻碍功能恢复。半暗带概念的提出为干预提供了理论框架——缺血核心区细胞发生快速坏死，而半暗带细胞虽受应激但仍维持膜完整性，具有可逆性。然而，随着再灌注时间延长，半暗带细胞可通过多种机制发生进行性死亡或进入功能异常状态。除经典的坏死、凋亡外，铁死亡（ferroptosis）、焦亡、坏死性凋亡等新型调节性细胞死亡方式在CIRI中的作用逐渐被揭示。与此同时，越来越多证据表明，半暗带中存在一类既未死亡、也未恢复正常的细胞群体，它们进入持续性应激状态，通过旁分泌效应影响周围微环境，阻碍神经修复。细胞衰老（cellular senescence）正是这种状态的典型代表。'));

children.push(h3('2. 铁死亡：CIRI急性期神经元损伤的关键执行者'));
children.push(p('铁死亡是Stockwell团队于2012年正式命名的一种铁依赖性、脂质过氧化驱动的调节性细胞死亡方式，其形态学、生化和遗传学特征均区别于凋亡、坏死和自噬[4]。铁死亡的核心机制是细胞抗氧化防御系统失能，导致含多不饱和脂肪酸的磷脂发生毒性过氧化产物大量堆积，最终破坏质膜完整性引发细胞死亡。其调控涉及三大核心通路的协同失衡：System Xc\u207b/GPX4抗氧化轴、ACSL4/LPCAT3脂质重塑轴以及铁代谢调控轴。GPX4是目前已知唯一能直接还原磷脂氢过氧化物的酶，被视为铁死亡的核心守门分子；System Xc\u207b通过摄取胱氨酸维持GSH合成，为GPX4提供还原当量；ACSL4催化多不饱和脂肪酸掺入膜磷脂，决定细胞对铁死亡的敏感性；铁代谢紊乱导致的游离铁升高则为Fenton反应提供催化剂，是铁死亡发生的必要条件[5]。除三大核心通路外，FSP1/CoQ10、GCH1/BH4等不依赖GPX4的防御系统以及铁自噬等也参与铁死亡的精细调控。'));
children.push(p('近年来，铁死亡在CIRI中的作用得到广泛验证。多项独立研究表明，脑缺血再灌注后缺血脑组织出现典型的铁死亡特征，包括游离铁升高、GSH耗竭、GPX4活性下降、4-HNE和MDA堆积以及线粒体皱缩等[7,13]。胡晴雯等[3]在大鼠MCAO/R模型中系统证实了铁死亡的存在：再灌注后不同时间点铁死亡标志物呈时间依赖性升高，透射电镜观察到缺血皮层神经元线粒体皱缩、膜密度增高等铁死亡典型形态学特征，铁死亡抑制剂DFO可显著减小梗死体积、改善神经功能评分。时间进程研究显示，铁死亡标志物在再灌注后数小时内即可升高，与急性期神经元死亡高度吻合。药理学干预证实了铁死亡的因果作用：Ferrostatin-1、Liproxstatin-1及铁螯合剂DFO均可显著减小MCAO模型梗死体积、改善神经功能评分，其保护效应在不同物种和模型中均得到重复。基因水平证据同样支持这一结论：神经元特异性GPX4敲除加剧缺血性损伤，而GPX4过表达或SLC7A11上调则发挥保护作用。细胞特异性研究进一步揭示了铁死亡的复杂性——神经元因高代谢率和高多不饱和脂肪酸含量对铁死亡高度敏感，星形胶质细胞、小胶质细胞及血管内皮细胞的铁死亡也参与病理进程。然而，铁死亡在CIRI中的作用并非仅限于急性期细胞杀伤——亚致死剂量的铁死亡应激是否会触发细胞进入其他应激状态（如细胞衰老），目前尚缺乏系统研究。'));

children.push(h3('3. 细胞衰老：缺血半暗带慢性化的重要推手'));
children.push(p('细胞衰老是指细胞在应激因素作用下退出细胞周期，进入稳定的增殖停滞状态，同时伴随广泛的基因表达和代谢重编程，形成特征性的衰老相关分泌表型（senescence-associated secretory phenotype, SASP）[6]。经典的细胞衰老包括端粒依赖性的复制性衰老和应激诱导的早熟性衰老（stress-induced premature senescence, SIPS）。前者与细胞分裂次数相关，主要发生于生理性衰老；后者则可由DNA损伤、氧化应激、癌基因激活、炎症因子等多种急性应激在数小时至数天内快速触发，其表型与复制性衰老高度相似但发生机制和时间尺度截然不同。SIPS的核心分子通路包括p53/p21CIP1轴和p16INK4a/Rb轴，二者共同驱动细胞周期停滞的建立和维持。SASP是衰老细胞最具病理影响力的特征，其成分包括促炎细胞因子、趋化因子、生长因子、基质金属蛋白酶等，通过旁分泌和内分泌方式重塑组织微环境。SASP的分泌受精密时空调控，涉及NF-\u03baB、C/EBP\u03b2、p38 MAPK、mTOR等多条信号通路。'));
children.push(p('传统观点认为细胞衰老主要与机体老化和年龄相关疾病有关，在急性损伤中作用有限。然而，近年来这一观念正在被修正。在脑缺血领域，越来越多证据表明，缺血性损伤可在梗死周围半暗带诱导多种细胞类型发生衰老样改变。啮齿类动物MCAO模型中，再灌注后数天内即可在半暗带检测到SA-\u03b2-gal阳性细胞，伴随p21和p16表达上调、\u03b3H2AX焦点增加等衰老标志物激活[8]。发生衰老的细胞类型包括神经元、星形胶质细胞、小胶质细胞及血管内皮细胞，提示缺血性应激诱导的衰老具有细胞普遍性。道吉吉等[16]在D-半乳糖诱导的神经元衰老模型中证实，铁死亡相关分子（Nrf2、SLC7A11、GPX4下调，TFRC上调）与衰老标志物（p53、p21、p16上调）共存，铁死亡诱导剂Erastin可进一步加剧衰老表型，而靶向ErbB4受体的小分子激动剂可通过Akt/Nrf2通路同时抑制铁死亡和衰老。临床样本研究同样提供了支持性证据：缺血性脑卒中患者脑脊液和外周血中可检测到衰老相关生物标志物升高，且其水平与梗死体积和功能预后相关。缺血诱导的衰老细胞并非被动旁观者，而是通过多种机制积极参与病理进程：SASP促炎因子招募免疫细胞浸润、激活胶质细胞形成慢性炎症微环境；基质金属蛋白酶降解细胞外基质、破坏血脑屏障；衰老的传播效应扩大损伤范围；衰老细胞丧失正常生理功能，阻碍突触重塑和神经发生。从干预角度，衰老细胞清除剂（senolytics）在多种脑损伤模型中显示出改善功能预后的效果，为靶向衰老的治疗策略提供了原理验证。然而，脑缺血后细胞衰老的触发因素和上游驱动机制尚未完全阐明，特别是氧化应激在衰老启动中的具体作用形式和分子通路仍有待深入揭示。'));

children.push(h3('4. 铁死亡与细胞衰老的交汇：铁衰老概念及其在CIRI中的研究空白'));
children.push(p('铁死亡与细胞衰老并非两个独立的病理过程，二者存在密切的交互作用和共同的上游驱动因素——氧化应激和铁代谢紊乱。近年来，研究者开始将这两个领域联系起来，提出了铁衰老（ferro-aging/ferro-senescence）的概念框架。2026年，Liu等[1]在Cell Metabolism发表的灵长类动物研究为铁衰老概念提供了系统性体内证据：随着灵长类年龄增长，肝脏和血清中铁水平逐渐升高，伴随ACSL4表达上调和脂质过氧化产物积累；铁过载可通过ACSL4介导的脂质过氧化通路驱动肝细胞和造血干细胞衰老，形成铁代谢紊乱\u2192脂质过氧化\u2192细胞衰老的完整级联；反之，抑制ACSL4或敲除铁调素调控基因可减轻衰老表型、延长健康寿命。该研究首次在高等哺乳动物中确立了ferro-aging作为生理性衰老驱动因素的地位，其时间尺度为月至年级，属于慢性衰老过程。在其他组织器官中，铁死亡驱动衰老的证据也在积累：肾近端小管细胞中，铁死亡诱导剂可引发SIPS样表型；在肺纤维化模型中，铁过载诱导的肺泡上皮细胞衰老与铁死亡标志物共存；心血管领域亦有铁依赖性血管内皮细胞衰老的报道。周永昌等[15]综述了铁死亡与衰老的复杂关系，指出二者存在双向调控的可能性：铁死亡既可驱动衰老发生，衰老细胞的铁代谢重编程也可能影响铁死亡敏感性，具体方向取决于细胞类型和应激强度。这些跨组织的一致性发现提示，铁死亡驱动衰老可能是一种普遍的病理生理机制。'));
children.push(pNoIndent([
  new TextRun({ text: '本项目对铁衰老概念的界定：', size: 24, font: '宋体', bold: true }),
  new TextRun({ text: '本项目中的铁衰老特指缺血再灌注后由亚致死铁死亡压力驱动的应激诱导早熟性衰老（ischemia-induced iron-dependent SIPS），与Liu等在自然衰老模型中描述的慢性ferro-aging在时间尺度、触发因素和生理背景上存在本质差异。前者是急性病理损伤后的继发性事件（小时至数天级），后者是生理性衰老过程的慢性积累（月至年级）。尽管存在差异，二者共享核心的分子通路——ACSL4/脂质过氧化/p53轴，均以铁依赖性脂质过氧化为上游驱动力。本项目将通过时序分析明确CIRI中铁依赖性SIPS的启动时间窗和持续时间，以期在急性损伤背景下验证这一概念。', size: 24, font: '宋体' })
]));
children.push(p('4-羟基壬烯醛（4-hydroxynonenal, 4-HNE）是\u03c9-6多不饱和脂肪酸过氧化的主要毒性醛类产物之一，具有高度亲电性，可与蛋白质的半胱氨酸、组氨酸和赖氨酸残基发生Michael加成反应，形成稳定的蛋白质羰基化修饰[14]。4-HNE在细胞内具有浓度依赖性的生物学效应：低浓度时作为信号分子参与多种通路调控，高浓度时则引发细胞毒性和细胞死亡。在铁死亡过程中，4-HNE是脂质过氧化的重要下游产物和标志物，其水平与铁死亡严重程度正相关。Monroe等[2]在Aging Cell发表的研究系统证实，4-HNE等脂质过氧化产物可在人成纤维细胞和小鼠脂肪干细胞中剂量依赖性地诱导衰老表型，包括SA-\u03b2-gal阳性率升高、\u03b3H2AX焦点积累、p53磷酸化增强、p21表达上调及SASP因子分泌增加。机制研究表明，4-HNE主要通过诱导氧化性DNA损伤、激活DNA损伤应答（DDR）通路导致p53 Ser15磷酸化，进而启动p21介导的细胞周期阻滞——这是一种间接的p53激活机制，而非4-HNE直接共价修饰p53蛋白所致。'));

children.push(pNoIndent([
  new TextRun({ text: '4-HNE作用的机制选择性问题：', size: 24, font: '宋体', bold: true }),
  new TextRun({ text: '值得注意的是，4-HNE作为亲电试剂，更倾向于修饰Keap1的半胱氨酸残基（已知激活Nrf2的经典机制），而非优先修饰p53并使其活化。为何在铁死亡情境下4-HNE不主要激活Nrf2（细胞防御）而是激活p53（促衰老/促铁死亡）？这是本项目拟回答的核心机制问题之一。我们推测存在"防御失效后的灾难性转向"机制：在铁死亡早期，4-HNE首先修饰Keap1激活Nrf2，启动细胞防御反应；但当脂质过氧化持续进展、4-HNE积累超过防御阈值时，Nrf2通路被过度消耗或失活，此时4-HNE转而通过DNA损伤和DDR通路激活p53，将细胞从"防御修复"状态推向"衰老阻滞"状态。这种时间依赖性的机制转换，可能是亚致死铁死亡压力最终驱动SIPS的关键分子开关。', size: 24, font: '宋体' })
]));

children.push(p('基于上述文献线索，我们提出一个经过修正的4-HNE-p53-SLC7A11正反馈环路假说：铁死亡产生的4-HNE首先通过Keap1修饰激活Nrf2防御通路；当4-HNE持续积累超过防御阈值时，转而通过氧化性DNA损伤激活DDR-p53-p21通路驱动细胞衰老；同时，活化的p53可转录抑制SLC7A11，进一步削弱GSH合成和抗氧化防御，加剧脂质过氧化和4-HNE生成，形成"铁死亡\u21924-HNE\u2192p53\u2192SLC7A11\u2193\u2192更多铁死亡"的正反馈环路，最终将亚致死性氧化应激转化为持续性的衰老表型。除4-HNE-p53轴外，HMGB1/铁自噬环路是另一个受到关注的连接通路：铁死亡细胞释放的HMGB1可结合TLR4受体，激活NF-\u03baB通路驱动SASP形成和炎症反应；同时，SASP中的IL-6、TNF-\u03b1等可上调转铁蛋白受体（TFR1），促进铁摄取，进一步加剧铁过载和脂质过氧化，形成恶性循环。'));
children.push(p('然而，目前关于铁死亡驱动细胞衰老的研究主要集中于肿瘤细胞和体外细胞系，在神经系统特别是缺血性脑损伤中的研究极为有限。CIRI半暗带中是否确实存在铁依赖性SIPS？其时空分布特征如何？4-HNE作用的机制选择性（Keap1-Nrf2防御 vs p53衰老转向）如何决定细胞命运？4-HNE-p53-SLC7A11分子轴是否在其中发挥关键作用？这些问题目前均缺乏直接的实验证据，有待系统研究加以阐明。'));

children.push(h3('5. Nrf2通路：铁死亡防御与衰老抑制的共同枢纽'));
children.push(p('核因子E2相关因子2（nuclear factor erythroid 2-related factor 2, Nrf2）是细胞抗氧化反应的主调控因子，属于碱性亮氨酸拉链转录因子家族。生理状态下，Nrf2被Keap1锚定在胞浆中，经Cullin-3介导的泛素-蛋白酶体途径快速降解，保持较低基础水平。当细胞暴露于氧化应激或亲电试剂时，Keap1关键半胱氨酸残基发生共价修饰，导致构象改变，丧失对Nrf2的泛素连接酶活性。新生的Nrf2逃脱降解并转位入核，与小Maf蛋白形成异二聚体，结合于抗氧化反应元件（ARE）保守序列，启动下游数百个靶基因转录。Nrf2靶基因涵盖抗氧化酶、解毒酶、物质代谢转运、蛋白质稳态、铁代谢调控等多个功能类别，共同构成细胞的综合性防御体系。除经典Keap1-Nrf2通路外，Nrf2还受PI3K/Akt、MAPK、GSK-3\u03b2等激酶磷酸化修饰以及自噬-溶酶体途径对Keap1降解等非经典机制调控。'));
children.push(p('Nrf2通路是铁死亡的重要防御机制。Nrf2可通过多层面调控抑制铁死亡发生：在抗氧化层面，上调GPX4表达直接增强磷脂过氧化物清除能力，上调SLC7A11促进胱氨酸摄取和GSH合成，间接维持GPX4活性；在铁代谢层面，转录激活铁蛋白（FTH1/FTL）促进游离铁螯合储存，上调铁输出蛋白FPN1降低不稳定铁池水平，下调TFR1减少铁摄取；在脂质代谢层面，调控ACSL4等脂质重塑相关酶表达，影响膜磷脂脂肪酸组成，降低铁死亡敏感性[12]。胡晴雯等[3]在大鼠MCAO/R模型中证实，BCP可显著增加Nrf2核转位，上调HO-1和GPX4表达，抑制ACSL4，从而减轻铁死亡和脑损伤；Nrf2抑制剂ML385可显著逆转BCP的神经保护效应，直接证明了Nrf2通路的关键作用。此外，Nrf2还可通过上调FSP1、GCH1等不依赖GPX4的防御分子构建多层防御网络。这些多维度调控使Nrf2成为铁死亡防御网络的核心节点。'));
children.push(p('Nrf2同样是抑制细胞衰老和SASP的关键因子。多项研究表明，Nrf2的表达和活性随年龄增长而下降，这种下降与衰老表型出现和年龄相关疾病发生发展密切相关。在细胞水平，Nrf2缺失或功能低下可加速氧化应激诱导的SIPS，表现为SA-\u03b2-gal阳性率升高、端粒缩短加速、SASP分泌增强；反之，Nrf2激活则可延缓多种细胞类型的衰老进程，减轻SASP相关炎症反应。道吉吉等[16]在D-半乳糖诱导的神经元衰老模型中证实，Nrf2抑制剂可显著削弱ErbB4激动剂对SLC7A11和GPX4的上调作用，同时阻碍其逆转神经元衰老的效果，直接证明了Nrf2在铁死亡-衰老调控轴中的核心地位。Nrf2抑制衰老的机制是多方面的：直接清除活性氧减轻DNA损伤，维持基因组稳定性；上调II相解毒酶增强细胞防御；抑制NF-\u03baB通路减少SASP因子转录，减轻慢性炎症；调节自噬-溶酶体通路维持蛋白质稳态和细胞器质量控制。这些机制共同作用，使Nrf2成为联系氧化应激、炎症和衰老的核心调控分子。值得注意的是，Nrf2与NF-\u03baB之间存在相互拮抗的串话：Nrf2激活可抑制NF-\u03baB核转位和转录活性，从而减少SASP促炎因子的表达；反之，NF-\u03baB的持续激活则可通过促进Keap1表达或直接竞争转录共激活因子而抑制Nrf2功能。这种拮抗关系在铁依赖性SIPS中可能具有重要意义——铁死亡产生的氧化应激在激活Nrf2防御反应的同时，也通过炎症因子激活NF-\u03baB，后者可能反过来削弱Nrf2的防御能力，形成另一个层面的恶性循环。鉴于Nrf2在铁死亡防御和衰老抑制中的双重核心作用，我们认为Nrf2通路是理想的干预靶点——激活Nrf2有望同时阻断铁死亡的急性期损伤和铁依赖性SIPS的慢性期损害，实现对CIRI病理进程的双阶段干预。'));

children.push(h3('6. 壮瑶药艾叶/桂艾及其活性成分\u03b2-石竹烯：从民族药经验到现代药理机制'));
children.push(p('艾叶（Artemisia argyi L\u00e9vl. et Vant.）为菊科蒿属植物的干燥叶，是我国传统中药和广西道地壮瑶药材。艾叶入药历史悠久，始载于《名医别录》，其性温，味辛、苦，归肝、脾、肾经，具有温经止血、散寒止痛、外用祛湿止痒等功效。在壮瑶医药理论体系中，艾叶具有独特地位：壮语称\u300c挨\u300d，瑶语称\u300c各艾\u300d，被视为\u300c通龙路火路、除风毒寒毒、逐湿邪\u300d之要药。壮医\u300c三道两路\u300d理论认为，脑（巧坞）为神明之府，龙路火路网络密布；中风（麻邦）是由于风毒、火毒、痧毒循龙路火路上攻巧坞，致三道两路不通、气血失衡、天地人三气不能同步。艾叶的\u300c通龙路火路、除毒邪\u300d功效恰中中风病病机要害，因此在壮瑶医临床中常用于麻痹、头痛、眩晕等脑病防治。艾叶的\u300c解毒除蛊\u300d功效，可从现代药理学角度解读为清除自由基、抗炎、调节细胞死亡与免疫功能等多重作用。'));
children.push(p('桂艾是产于广西地区的艾叶道地药材，因独特的地理气候条件而具有挥发油含量高、活性成分丰富的特点。桂艾挥发油是艾叶的主要药效物质基础，得油率约为0.45%-1.2%，已从中分离鉴定出数十种化学成分，主要包括倍半萜类、单萜类、黄酮类及酚酸类化合物，其中\u03b2-石竹烯（\u03b2-caryophyllene, BCP）含量可高达15%-35%，是桂艾挥发油中最重要的倍半萜成分之一。BCP是一种天然存在的双环倍半萜化合物，具有独特的丁香香气，已被美国FDA和欧洲食品安全局批准为食品添加剂，安全性极高。'));

children.push(pNoIndent([
  new TextRun({ text: '桂艾挥发油与BCP单体的关系定位：', size: 24, font: '宋体', bold: true }),
  new TextRun({ text: '本项目选择BCP作为主要研究对象，基于以下考虑：（1）BCP是桂艾挥发油中含量最高、活性最明确的倍半萜成分，其药理作用研究最为深入，可作为桂艾药效物质基础的代表性分子；（2）单体成分研究有助于明确作用靶点和分子机制，符合现代药理学研究规范；（3）桂艾挥发油代表了民族药"多成分、多靶点、整合效应"的传统用药特色。二者关系可概括为：BCP是桂艾挥发油的核心药效物质基础之一，但桂艾挥发油的整体药效可能因其多成分协同作用而呈现一定的增益效应。需要说明的是，若桂艾挥发油中BCP含量为15%-35%，则408 mg/kg BCP单体相当于约1,200-2,700 mg/kg挥发油，这一剂量已超出挥发油的常规安全剂量范围。因此，本项目中桂艾挥发油的研究将采用其自身的剂量梯度（基于文献和预实验确定），而非与BCP单体进行等剂量比较，二者的药效对比应理解为"民族药整体观"与"单体精准机制"两个研究层面的互补，而非简单的效价比较。', size: 24, font: '宋体' })
]));

children.push(p('现代药理学研究揭示了BCP的多种生物活性，包括抗炎、抗氧化、镇痛、抗肿瘤、保肝、护胃以及神经保护等。其中，抗炎和抗氧化活性是BCP最受关注的药理作用。BCP是大麻素CB2受体的选择性激动剂，可通过CB2受体依赖的信号通路发挥抗炎和免疫调节作用。在抗氧化方面，近年来研究发现BCP可显著激活Nrf2/ARE信号通路，上调多种抗氧化酶和II相解毒酶的表达。具体而言，BCP可通过修饰Keap1半胱氨酸残基或激活PI3K/Akt、ERK等上游信号激酶，促进Nrf2核转位和转录激活，进而上调HO-1、NQO1、GCLC、GCLM等靶基因表达，增强细胞抗氧化防御能力。在神经系统疾病领域，BCP的神经保护作用已在多种模型中得到验证。刘京东等[17]报道BCP通过抑制Notch1/NF-\u03baB信号轴减少炎性因子释放，减轻大鼠脑缺血再灌注损伤；左天睿等[18]证实BCP通过激活HSF1/HSP70通路减轻氧化应激和细胞凋亡。胡晴雯等[3]在Phytomedicine发表的研究明确证实，BCP可通过激活Nrf2/HO-1通路减轻大鼠脑缺血再灌注损伤：BCP预处理可剂量依赖性地减小MCAO/R模型梗死体积、降低脑水肿程度、改善神经功能评分，同时伴随脑组织中Nrf2核转位增加、HO-1表达上调及氧化应激指标改善；而Nrf2抑制剂ML385可显著逆转BCP的神经保护效应。药代动力学研究表明，BCP具有良好的血脑屏障穿透能力，口服给药后可迅速进入脑组织并维持有效浓度，为其中枢神经系统疾病治疗应用提供了药代动力学基础。'));
children.push(p('近年来，BCP与铁死亡的关系开始受到关注。有研究报道BCP可通过激活Nrf2通路抑制肝纤维化模型中的铁死亡[15]，在顺铂耳毒性模型中也观察到BCP对铁死亡的抑制作用[16]。这些研究为BCP调控铁死亡提供了初步实验支持。然而，BCP是否能够通过激活Nrf2通路同时阻断铁死亡和铁依赖性SIPS，是否能够通过干预铁死亡-衰老正反馈环路改善CIRI远期预后，目前尚未见文献报道。考虑到BCP明确的Nrf2激活活性、良好的血脑屏障穿透性以及极高的安全性，系统研究BCP对铁依赖性SIPS的干预作用及其分子机制，具有重要的理论价值和转化潜力。'));

children.push(h3('7. 本项目拟解决的核心科学问题及研究意义'));
children.push(p('综上所述，脑缺血再灌注损伤是急性缺血性脑卒中治疗中的重大临床难题，其病理机制涉及急性期多种调节性细胞死亡和慢性期神经炎症与修复障碍。铁死亡作为急性期神经元死亡的关键形式已得到广泛认可；细胞衰老则被认为是缺血半暗带慢性化的重要推手。然而，二者之间的因果联系——即铁死亡是否以及如何驱动缺血半暗带中的细胞衰老（铁依赖性SIPS）——目前尚不清楚。特别是，4-HNE作为脂质过氧化的核心产物，其作用的机制选择性（优先激活Keap1-Nrf2防御还是转向p53介导的衰老）是决定亚致死铁死亡压力最终细胞命运的关键科学问题。基于文献线索和我们前期的网络药理学研究，我们推测：CIRI半暗带中存在亚致死性铁死亡压力驱动的铁依赖性SIPS；4-HNE作用存在时间依赖性的机制转换——早期激活Keap1-Nrf2防御，晚期超过阈值后通过DDR-p53通路驱动衰老；4-HNE-p53-SLC7A11正反馈环路是铁依赖性SIPS维持和放大的核心分子机制；Nrf2作为铁死亡防御与衰老抑制的共同枢纽，是理想的干预靶点；广西道地壮药桂艾的活性成分\u03b2-石竹烯可通过激活Nrf2通路，同时阻断铁死亡急性期损伤和铁依赖性SIPS慢性损害，从而改善CIRI远期预后。'));
children.push(p('本研究的理论意义在于：提出并验证缺血诱导的铁依赖性SIPS这一新型病理概念，丰富对CIRI慢性化机制的认识，为理解铁死亡与细胞衰老的交互作用提供新的实验证据；阐明4-HNE作用的机制选择性及其在铁依赖性SIPS启动中的开关作用，揭示4-HNE-p53-SLC7A11正反馈环路的核心作用，为铁死亡驱动衰老的分子机制提供新的视角。实践意义在于：阐明壮瑶药桂艾活性成分\u03b2-石竹烯的神经保护新机制，为开发以铁衰老为靶点的脑卒中治疗药物提供理论依据和先导化合物，也为民族医药现代化研究提供可借鉴的研究范式。'));

children.push(h2('（二）项目组网药研究基础'));

children.push(p('为验证上述科学假说，项目组前期整合多维度公开数据集与生物信息学方法，系统开展了脑缺血-铁衰老-石竹烯的网络药理学与机器学习研究，完成了从疾病特征识别到药物靶点预测的全链条计算分析，为后续实验验证奠定了坚实的前期基础。'));

children.push(h3('1. 多数据集铁衰老转录特征的识别与验证'));

children.push(pNoIndent([
  new TextRun({ text: '（1）铁衰老96基因集的构建策略', size: 24, font: '宋体', bold: true })
]));
children.push(p('本研究中铁衰老基因集的构建遵循"多源整合、严格筛选、功能验证"的原则，具体流程如下：首先，从FerrDb数据库（http://www.zhounan.org/ferrdb/）获取经文献 curation 的铁死亡相关基因（驱动基因和抑制基因），从CellAge数据库（https://genomics.senescence.info/cells/）获取细胞衰老相关基因（Senescence-Associated Genes, SAGs），从Gene Ontology数据库获取"cellular response to oxidative stress"、"iron ion homeostasis"、"lipid peroxidation"等功能条目下的基因。其次，取铁死亡基因集与细胞衰老基因集的交集，获得"双身份基因"（同时与铁死亡和衰老相关）；进一步通过文本挖掘PubMed文献中同时提及"ferroptosis"和"senescence"的基因，补充入基因集。第三，通过STRING v12.0蛋白质相互作用网络分析，保留在网络中具有至少3个连接的基因，去除孤立节点，确保基因集的功能连通性。最终获得96个铁衰老相关基因，涵盖铁代谢调控（FTH1、FTL、TFRC、SLC40A1、HEPH等）、脂质过氧化与抗氧化防御（GPX4、SLC7A11、ACSL4、SOD1、CAT等）、DNA损伤与衰老调控（TP53、CDKN1A/p21、CDKN2A/p16、RB1、ATM等）以及SASP相关因子（IL6、TNF、CXCL8、MMP3等）四大功能类别。基因集的完整列表见附录1。'));

children.push(pNoIndent([
  new TextRun({ text: '（2）四数据集铁衰老评分的疾病关联性', size: 24, font: '宋体', bold: true })
]));
children.push(p('本研究整合了4个脑缺血时间进程数据集——GSE104036（小鼠MCAO，RNA-seq，27样本，0-72h）、GSE16561（人缺血性脑卒中，Illumina微阵列，63样本）、GSE61616（大鼠MCAO，Affymetrix微阵列，15样本）及GSE97537（大鼠MCAO，Affymetrix微阵列，12样本）。采用单样本基因集富集分析（ssGSEA），基于Barbie等（Nature Protocols, 2009）[9]的秩加权富集统计算法，计算每个样本的铁死亡、细胞衰老及铁衰老（96基因集）评分。结果显示，在全部4个数据集中，铁衰老评分的疾病-对照效应量（Cohen\'s d）均大于铁死亡评分和衰老评分，表明铁衰老基因集捕获的转录信号与缺血性脑损伤的关联最为紧密，在跨物种、跨平台中具有稳健性。在GSE104036小鼠MCAO模型中，同侧脑组织铁衰老评分为0.167\u00b10.032，显著高于假手术组（0.113\u00b10.001），效应量Cohen\'s d = 1.84（P = 4.40 \u00d7 10\u207b\u00b3）。'));

children.push(pNoIndent([
  new TextRun({ text: '（3）基因集特异性的置换检验验证', size: 24, font: '宋体', bold: true })
]));
children.push(p('为排除铁衰老评分仅仅是泛氧化应激信号的反映，我们进行了严格的置换检验：从全基因组中随机抽取与铁衰老基因集同等大小（96个）的基因集，重复1000次，计算每个随机基因集的ssGSEA评分与疾病状态的关联强度。结果显示，铁衰老基因集的疾病-对照效应量（Cohen\'s d = 1.84）显著高于随机基因集的分布（随机集平均d = 0.32 \u00b1 0.21），置换检验P = 0.001，表明铁衰老基因集捕获的疾病关联信号具有高度特异性，并非随机选择的氧化应激基因的泛化效应。'));

children.push(pNoIndent([
  new TextRun({ text: '（4）时序特征与LASSO稳定性筛选', size: 24, font: '宋体', bold: true })
]));
children.push(p('时序分析显示，铁死亡与铁衰老评分均在再灌注后6小时达到峰值，随后下降；而经典细胞衰老评分并未在后期时间点升高，反而在急性期至亚急性期呈下降趋势。这一发现不支持经典的铁死亡上升\u2192衰老上升\u2192铁衰老过渡序列模式。据此，本项目明确指出：当前基于转录组的铁衰老评分主要捕获的是铁死亡相关转录状态，而非经典意义上的衰老程序——评分中"衰老"成分的贡献主要来自p53/p21通路的早期激活（DNA损伤应答的即时反应），而非完全建立的衰老表型。因此，本项目将后续实验验证的核心定位为"铁死亡压力是否驱动细胞进入衰老状态"这一因果关系的证明，而非将转录组评分作为铁衰老存在的直接证据。以GSE104036中位数划分的高铁衰老活性组为因变量，以铁衰老基因集表达矩阵为预测变量，采用L1正则化逻辑回归进行特征选择。为确保稳定性，重复50次随机子采样的6折交叉验证，保留选择频率大于50%的基因。结果筛选出5个稳定的CIRI-铁衰老特征基因：SAT1（96%）、EBF3（88%）、KLF6（88%）、LIFR（72%）及CD74（70%）。内部交叉验证AUC为0.73 \u00b1 0.09，置换检验P = 0.002。将训练好的模型应用于3个独立数据集，预测概率与独立计算的铁衰老评分呈显著正相关：GSE16561（\u03c1 = 0.56，P < 0.0001）、GSE61616（\u03c1 = 0.75，P < 0.0001）、GSE97537（\u03c1 = 0.88，P < 0.0001），有力支持了五基因特征对铁死亡相关转录活性的预测效能。五个特征基因的功能内涵丰富：SAT1为精脒/精胺N1-乙酰转移酶1，是p53驱动铁死亡的核心介质，其高选择频率（96%）提示p53-铁死亡轴在铁衰老转录特征中占主导地位；KLF6经Nrf2/HO-1轴参与MCAO后铁死亡调控；CD74参与卒中后小胶质细胞活化和神经炎症；LIFR介导脑缺血后的神经保护信号；EBF3是神经元分化因子。'));

children.push(h3('2. 石竹烯靶点与铁衰老调控网络的拓扑收敛'));
children.push(p('采用交集+网络邻近扩展双路径策略确定石竹烯干预脑缺血铁衰老的最终核心基因集：最终核心基因集 = （CIRI-铁衰老候选基因 \u2229 石竹烯高置信度靶点） \u222a 网络邻近扩展的铁死亡关键基因。Part A通过五基因CIRI-铁衰老签名与石竹烯427个高置信度靶点（SwissTargetPrediction + STITCH联合预测）取交集，仅包含SAT1（选择频率96%），提示SAT1是石竹烯直接作用于铁衰老调控的核心靶点。Part B通过一阶网络邻近分析，在STRING v12.0（人，combined score > 700）[10]网络中，石竹烯靶点的直接邻居中识别出337个铁死亡驱动基因邻近基因（44个直接靶点 + 293个一阶邻居）。超几何检验显示该富集具有极高度统计学意义（P = 2.48 \u00d7 10\u207b\u2074\u00b3），表明石竹烯靶点群与铁死亡调控网络存在高度拓扑关联。'));
children.push(p('基于STRING v12.0构建的核心基因PPI子网包含311个节点及1,867条边，网络密度为0.039，平均聚类系数为0.458，提示存在显著的模块化结构。最大连通分量包含306个节点，平均最短路径长度为2.91，符合小世界网络特征。度中心性排名前10的Hub基因为：TP53（Degree=116）、EGFR（66）、EP300（63）、STAT3（63）、IL6（52）、TNF（51）、HSP90AA1（47）、H3C12（46）、H3C13（46）、SIRT1（45）。其中TP53作为度最高的节点，同时介数中心性亦最高（0.217），是网络信息传递的核心枢纽。NFE2L2（Nrf2）度中心性为38，处于网络核心位置，其下游靶基因GPX4、FTH1、SLC7A11等均在核心网络中。采用贪心模块度算法识别出8个紧密连接的功能模块：模块1（TP53种子，83基因，转录调控/细胞周期/应激反应，含E2F、FOXO、Nrf2通路）、模块2（EGFR种子，59基因，炎症免疫/细胞因子信号/NLRP3炎症小体）、模块3（MTOR种子，44基因，自噬-溶酶体通路/铁自噬）、模块4（CAV1种子，30基因，铁代谢调控/脂质代谢）、模块5（HSP90AA1种子，28基因，氧化应激/分子伴侣）、模块6（H3C12种子，27基因，表观遗传调控/组蛋白修饰）、模块7（PRKCA种子，15基因，信号转导）及模块8（ALOX15种子，7基因，花生四烯酸代谢/脂质过氧化执行）。功能富集分析进一步支持上述结论：KEGG通路富集排名前列的包括细胞衰老（29基因，adjusted P = 5.47 \u00d7 10\u207b\u00b2\u00b9）、TNF信号通路（38基因）、细胞凋亡（33基因）及NF-\u03baB信号通路（27基因）等。WikiPathways中，Ferroptosis通路（WP4313，40基因）排名第4位（adjusted P = 9.19 \u00d7 10\u207b\u007b\u00b2\u00b3\u007d），进一步支持核心基因集与铁死亡机制的高度关联。刘胜伟等[19]的网络药理学研究同样证实BCP作用于CIRI的靶点富集于p53、MAPK、NF-\u03baB等通路，关键靶点包括IL-6、TNF、TP53等，与本项目的分析结果高度一致。'));

children.push(h3('3. 免疫浸润与铁衰老的协同激活及WGCNA验证'));
children.push(p('基于特征基因集的ssGSEA免疫浸润分析显示，在GSE104036的27个样本中，铁衰老评分与多种免疫细胞丰度存在显著相关。其中，中性粒细胞（r = 0.651，P = 2.3 \u00d7 10\u207b\u2074）和M2型巨噬细胞（r = 0.613，P = 6.7 \u00d7 10\u207b\u2074）与铁衰老评分呈强正相关；而小胶质细胞稳态标志（r = -0.738，P = 1.1 \u00d7 10\u207b\u2075）和星形胶质细胞（r = -0.567，P = 0.002）呈显著负相关。需要说明的是，ssGSEA方法在区分M2型巨噬细胞与小胶质细胞亚型时存在一定局限性——CD206等标志物在某些活化状态的小胶质细胞中也有表达，可能导致算法对细胞类型的判定存在交叉。为弥补这一缺陷，后续实验验证阶段将采用多色免疫荧光（Iba-1/CD206/CD68）进行细胞类型的精确鉴定。在检测的18个关键炎症因子中，15个与铁衰老评分呈显著正相关，排名前列的包括Ccl2（r = 0.903）、Icam1（r = 0.890）、Cxcl10（r = 0.877）、Stat3（r = 0.875）及Il1b（r = 0.847），强烈支持铁衰老与神经炎症的协同激活机制。这一发现提示，铁依赖性SIPS不仅是细胞自主的过程，还可能通过SASP分泌招募外周免疫细胞、激活胶质细胞，形成炎症-衰老正反馈环路。特别值得注意的是，Ccl2和Icam1是单核细胞招募和黏附的关键分子，其与铁衰老评分的高度正相关（r > 0.89）暗示铁衰老细胞可能通过分泌这些趋化因子主动招募外周巨噬细胞，从而放大神经炎症反应。这与文献中SASP招募免疫细胞的报道一致，也为靶向铁衰老减轻脑缺血后慢性炎症提供了理论依据。在样本量最大的人脑缺血数据集GSE16561（63样本）中，加权基因共表达网络（WGCNA）验证显示，337个核心基因中有97个被分配至有意义的共表达模块。核心基因在turquoise模块中的平均模块身份（MM）为0.936，平均基因显著性（GS）为0.236，证实核心基因集在人脑缺血的共表达调控网络中处于核心位置。'));

children.push(h3('4. 核心科学假说的提出'));
children.push(p('基于上述国内外研究现状和本项目组前期网络药理学研究基础，我们提出以下科学假说：\u03b2-石竹烯通过激活Nrf2通路，上调GPX4、FTH1和SLC7A11等下游靶基因，增强细胞抗氧化防御能力，减少4-HNE生成，阻断4-HNE从Keap1-Nrf2防御向DDR-p53衰老的机制转换，从而抑制缺血诱导的铁依赖性SIPS的启动和SASP的分泌，打破铁死亡\u21924-HNE\u2192p53\u2192SLC7A11\u2193\u2192更多铁死亡的正反馈环路，最终改善脑缺血再灌注损伤的远期预后。BCP单体是桂艾挥发油的核心药效物质基础，桂艾挥发油因多成分整合可能呈现一定程度的整体药效增益，二者分别代表民族药研究的"精准机制"与"整体观"两个层面。本项目拟通过系统的体内外实验验证这一假说，为脑缺血再灌注损伤的治疗提供新的靶点和候选药物。'));

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
  '[15] Zhou Y, Tao S, Chen S, et al. Advances in ferroptosis-based therapy for aging and aging-related diseases. Chinese Journal of Comparative Medicine, 2023, 33(9): 121-131, 154.',
  '[16] Daoji J, Zhao W. Small molecule agonist targeting ErbB4 receptor inhibits neuronal senescence by regulating ferroptosis via Akt/Nrf2 pathway. Master Thesis, Jiangnan University, 2025.',
  '[17] Liu J, Chen S, Wang Y, et al. \u03b2-caryophyllene improves cerebral ischemia reperfusion injury in rats via Notch1/NF-\u03baB signal axis. Journal of Third Military Medical University, 2021, 43(2): 109-117.',
  '[18] Zuo T, Dong Z. \u03b2-Caryophyllene alleviates cerebral ischemia-reperfusion injury in rats by activating HSF1/HSP70 signaling pathway. Master Thesis, Chongqing Medical University, 2022.',
  '[19] Liu S, Shen Z, Ren Z, et al. Network pharmacology-based study and in-vivo verification of \u03b2-caryophyllene against cerebral ischemia/reperfusion injury. Journal of New Chinese Medicine, 2024, 56(11): 63-69.'
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
children.push(bullet('证实CIRI半暗带中存在缺血诱导的铁依赖性SIPS，阐明4-HNE作用的机制选择性（Keap1-Nrf2防御 vs p53衰老转向）在铁依赖性SIPS启动中的开关作用，明确4-HNE-p53-SLC7A11分子轴的关键作用。'));
children.push(bullet('阐明BCP通过Nrf2通路阻断铁依赖性SIPS的细胞分子机制，明确其作用靶点和信号调控网络，重点验证BCP\u2192Nrf2\u2192GPX4\u2191\u21924-HNE\u2193\u2192阻断p53衰老转向\u2192SLC7A11\u2191这一信号轴。'));
children.push(bullet('在整体动物水平验证BCP改善CIRI远期预后的药效，确认其神经保护效应的Nrf2依赖性（药理学抑制+脑区特异性敲降双重验证）。'));
children.push(bullet('比较桂艾挥发油与BCP单体的药效差异，探讨民族药多成分整合效应，为壮瑶药现代化研究提供实验依据。'));

children.push(h1('三、研究内容'));

children.push(h2('研究内容一：CIRI中铁依赖性SIPS的时空特征及分子轴研究（重点）'));
children.push(p('本部分为项目研究重心，旨在明确CIRI后半暗带中是否存在铁依赖性SIPS，及其时空分布特征、细胞类型和核心分子机制，重点解答4-HNE作用的机制选择性问题。采用C57BL/6J小鼠MCAO/R模型（缺血60 min），在再灌注后6 h、24 h、3 d、7 d、28 d五个时间点取材（增加6h时间点以捕捉4-HNE作用的早期动态）。运用免疫荧光多重标记技术（NeuN/GFAP/Iba-1分别与铁死亡标志物GPX4/4-HNE/FTH1及衰老标志物p21/\u03b3H2AX共定位）、透射电镜、铁含量比色法及普鲁士蓝染色，系统定量分析铁衰老双阳性细胞的时空分布规律及细胞类型。'));

children.push(pNoIndent([
  new TextRun({ text: '4-HNE机制选择性的时序验证：', size: 24, font: '宋体', bold: true }),
  new TextRun({ text: '采用免疫共沉淀（IP）联合Western Blot，时序检测4-HNE修饰Keap1与4-HNE修饰p53的相对水平变化，绘制时间动力学曲线。若观察到"Keap1修饰先于p53修饰达峰"的时序模式，则支持"防御失效后转向衰老"的机制假说。同时检测Nrf2核转位和HO-1/NQO1转录（防御通路激活标志）与p53 Ser15磷酸化和p21转录（衰老通路激活标志）的时序关系，明确两条通路的激活顺序和转换时间窗。', size: 24, font: '宋体' })
]));

children.push(pNoIndent([
  new TextRun({ text: 'p53修饰位点的功能验证：', size: 24, font: '宋体', bold: true }),
  new TextRun({ text: '在体外培养的HEK293T细胞中，转染野生型p53及半胱氨酸突变体（Cys124Ala、Cys135Ala、Cys141Ala等候选位点），给予不同浓度4-HNE处理，通过IP检测各突变体的4-HNE修饰水平，确定4-HNE修饰p53的关键半胱氨酸残基。进一步在原代神经元中，通过AAV介导的突变体p53过表达，观察关键位点突变是否影响4-HNE诱导的衰老表型（SA-\u03b2-gal、p21表达）。', size: 24, font: '宋体' })
]));

children.push(pNoIndent([
  new TextRun({ text: 'p53对SLC7A11转录抑制的功能验证：', size: 24, font: '宋体', bold: true }),
  new TextRun({ text: '采用ChIP-qPCR验证p53与SLC7A11启动子的直接结合；在p53野生型和p53敲低细胞中，检测4-HNE处理后SLC7A11的表达变化，明确p53依赖的转录抑制程度；通过SLC7A11过表达回复实验，验证p53介导的SLC7A11下调是否为4-HNE诱导铁死亡增敏的必要条件。', size: 24, font: '宋体' })
]));

children.push(pNoIndent([
  new TextRun({ text: 'ACSL4的双向调控验证：', size: 24, font: '宋体', bold: true }),
  new TextRun({ text: '除AAV-shACSL4敲低实验外，增加AAV-ACSL4过表达的正交验证：在半暗带区过表达ACSL4，观察是否可单独诱导铁死亡-衰老双阳性细胞增加，与敲低实验互为补充，增强因果推断的可靠性。', size: 24, font: '宋体' })
]));

children.push(h2('研究内容二：BCP抗铁依赖性SIPS的细胞机制研究（重点）'));
children.push(p('本部分为另一研究重心，旨在建立稳定的体外铁依赖性SIPS细胞模型，评价BCP的干预效果，并明确Nrf2在其中的关键作用。采用新生24 h内C57BL/6J小鼠原代皮层神经元和星形胶质细胞，以低剂量Erastin和OGD/R（2 h OGD + 24 h复氧）两种方法诱导亚致死量铁死亡压力。'));

children.push(pNoIndent([
  new TextRun({ text: '铁依赖性SIPS模型的严格验证：', size: 24, font: '宋体', bold: true }),
  new TextRun({ text: '在确定诱导条件前，进行多层次验证以确保"铁依赖性"而非泛氧化应激：（1）铁死亡指标确证：C11-BODIPY 581/591探针检测脂质ROS、Phen Green SK检测游离Fe\u00b2\u007a、TBA法测MDA、比色法测GSH/GSSG比值，确认选定条件下确实存在铁死亡相关的脂质过氧化和铁蓄积；（2）铁死亡抑制剂逆转实验：Liproxstatin-1（200 nM）或DFO（100 \u03bcM）预处理后，观察衰老表型（SA-\u03b2-gal、p21）是否被显著抑制，若可被抑制则支持"铁依赖性"；（3）4-HNE剂量-效应曲线：绘制4-HNE浓度梯度（0.1-50 \u03bcM）与细胞死亡率、铁死亡标志物、衰老标志物的量效关系，确定诱导铁死亡的浓度阈值与诱导衰老的浓度窗口，为BCP干预提供直接的剂量参考。优化诱导条件，建立细胞死亡率<15%但呈现典型衰老表型、且可被铁死亡抑制剂逆转的SIPS模型。评价BCP（1、10、50 \u03bcM）的干预效应。', size: 24, font: '宋体' })
]));

children.push(pNoIndent([
  new TextRun({ text: 'Nrf2依赖性的双重验证：', size: 24, font: '宋体', bold: true }),
  new TextRun({ text: '采用功能获得和缺失策略：Nrf2过表达（质粒转染）和Nrf2沉默（siRNA + ML385药理学抑制）细胞中，观察BCP抗SIPS效应的变化。检测Nrf2核转位、下游靶基因（GPX4、HO-1、FTH1、SLC7A11、NQO1）表达及铁死亡和衰老相关指标。上游机制采用分子对接预测BCP与Keap1疏水口袋结合模式（PDB: 4IQK），结合CETSA验证潜在结合；若不理想则以点突变验证。下游机制重点验证BCP\u2192Nrf2\u2192GPX4\u2191\u21924-HNE\u2193\u2192阻断p53衰老转向\u2192SLC7A11\u2191信号轴，通过时序分析和RSL3/p53过表达挽救实验确认上下游关系。', size: 24, font: '宋体' })
]));

children.push(pNoIndent([
  new TextRun({ text: '双重染色的即时共定位验证：', size: 24, font: '宋体', bold: true }),
  new TextRun({ text: '在低剂量Erastin处理后12-24 h（脂质ROS峰值时点），采用C11-BODIPY（脂质ROS，铁死亡标志）与SA-\u03b2-gal（衰老标志）的双重染色，通过共聚焦显微镜观察同一细胞中两种信号的共存情况。若在脂质ROS信号峰值时刻即可检测到衰老标志物的早期出现，则为"铁死亡驱动衰老"提供最直接的细胞学证据，比单纯依赖转录组数据更具说服力。', size: 24, font: '宋体' })
]));

children.push(h2('研究内容三：BCP改善CIRI远期预后的整体药效与机制验证'));
children.push(p('本部分在整体动物水平验证BCP对CIRI的神经保护效应，重点关注远期功能预后，确认Nrf2依赖性。考虑到本项目研究重心定位于"现象确认"和"分子机制"（研究内容一和二），整体药效验证适当简化，聚焦核心指标。'));

children.push(pNoIndent([
  new TextRun({ text: '给药方案的优化与依据：', size: 24, font: '宋体', bold: true }),
  new TextRun({ text: '关于BCP给药剂量和方案，需说明以下几点：（1）剂量设置依据：前期文献中BCP抗CIRI研究多采用预处理模式（缺血前连续7天灌胃），剂量范围为102-408 mg/kg[3,17,18]。本项目拟采用治疗模式（再灌注后即刻首次给药），因此设置预实验确定有效剂量范围：设5个剂量梯度（50、100、200、400、600 mg/kg），以24 h梗死体积为主要终点，n=6/组，确定治疗模式下的ED50和最大有效剂量，为正式实验提供剂量依据。（2）给药频率：BCP半衰期约2-4 h，为评估每日一次给药的合理性，预实验中选取3个时间点（给药后2、8、24 h）测定脑组织BCP浓度（GC-MS法），绘制脑组织药物浓度-时间曲线。若谷浓度低于有效浓度，则调整为每日两次给药。（3）给药周期：正式实验给药14天，聚焦亚急性期至慢性早期的恢复过程，28天进行行为学终点评价。（4）安全性监测：记录体重变化、血清ALT/AST（肝功能）、BUN/Cr（肾功能），评估高剂量组的肝肾安全性。', size: 24, font: '宋体' })
]));

children.push(p('采用C57BL/6J小鼠MCAO/R模型，设假手术组、模型组、BCP低/中/高剂量组（基于预实验结果确定）、Liproxstatin-1阳性药组（10 mg/kg）、BCP+ML385组（30 mg/kg，药理学Nrf2抑制），每组12只。再灌注后即刻首次给药，随后按预实验确定的频率灌胃。短期评价（1-3 d）：TTC染色测梗死体积、干湿重法测脑水肿、mNSS评分。远期评价（28 d）：足误实验、转棒实验评估运动功能恢复。'));

children.push(pNoIndent([
  new TextRun({ text: 'Nrf2依赖性的双重验证策略：', size: 24, font: '宋体', bold: true }),
  new TextRun({ text: '考虑到全身性Nrf2敲除存在发育代偿和系统性混杂效应，本项目采用"药理学抑制+脑区特异性敲降"的双重验证策略：（1）药理学抑制（主要）：ML385（30 mg/kg，腹腔注射）作为Nrf2特异性抑制剂，与BCP联合给药，观察BCP效应是否被逆转，该方法可排除发育代偿问题，但存在一定的脱靶风险。（2）脑区特异性敲降（辅助验证）：通过脑立体定位注射AAV-shNrf2至MCA供血区（纹状体+皮层），实现半暗带区Nrf2的局部敲降，结合BCP给药，验证BCP的神经保护是否依赖于脑组织中Nrf2的激活。这两种方法互为补充，可更严谨地证明BCP效应的Nrf2通路特异性。同时，增设Nrf2\u207b/\u207b+溶媒组作为基础水平对照，以排除Nrf2敲除本身对梗死体积的影响。', size: 24, font: '宋体' })
]));

children.push(p('28 d脑组织行SA-\u03b2-gal染色、铁死亡-衰老双阳性细胞定量、p53羰基化及衰老和SASP标志物检测、铁死亡指标检测，将BCP药效与铁依赖性SIPS抑制及Nrf2通路激活直接关联。'));

children.push(h2('研究内容四：桂艾挥发油与BCP单体的药效比较研究'));
children.push(p('本部分旨在从民族药整体观角度，比较桂艾挥发油与BCP单体的药效差异，探讨多成分整合效应。考虑到研究重心和资源分配，本部分为探索性研究，适当控制规模。'));
children.push(pNoIndent([
  new TextRun({ text: '研究内容：', size: 24, font: '宋体', bold: true }),
  new TextRun({ text: '（1）桂艾挥发油的制备与质量控制：采用水蒸气蒸馏法提取广西道地桂艾挥发油，GC-MS进行成分分析和含量测定，确保BCP含量在15%-35%的正常范围内，建立指纹图谱用于质量控制。（2）剂量设置说明：由于桂艾挥发油中BCP仅占15%-35%，若以BCP含量计算等效剂量，挥发油的总给药量将远超出安全范围。因此，桂艾挥发油组将设置独立的剂量梯度（基于挥发油的文献安全剂量和预实验确定，预计为50-200 mg/kg），与BCP单体组的比较应理解为"等摩尔BCP含量"和"等药效"两个层面的比较，而非简单的等剂量比较。（3）药效比较：在MCAO/R模型中，比较桂艾挥发油组与等BCP摩尔含量的BCP单体组在梗死体积、铁死亡标志物、衰老标志物及短期神经功能评分方面的差异，初步判断是否存在多成分协同效应。（4）机制初步探索：比较两组对Nrf2通路激活强度的差异，若挥发油组显示出更强的Nrf2激活或更广的靶基因谱，则提示可能存在多成分协同作用。', size: 24, font: '宋体' })
]));

children.push(h1('四、拟解决的关键科学问题'));
children.push(p('1. CIRI半暗带中铁依赖性SIPS的存在性及其4-HNE作用机制选择性的阐明。核心是回答"4-HNE为何在铁死亡情境下不主要激活Nrf2防御，而是转向p53介导的衰老"这一关键问题。这一问题的解决将为理解缺血性脑损伤从急性期向慢性期演变的机制提供新视角，丰富铁死亡与细胞衰老交互作用的理论体系。'));
children.push(p('2. BCP通过Nrf2通路阻断铁依赖性SIPS、改善CIRI远期预后的分子机制解析。重点验证BCP\u2192Nrf2\u2192GPX4\u2191\u21924-HNE\u2193\u2192阻断p53衰老转向\u2192SLC7A11\u2191这一完整信号轴。这一问题的解决将为开发以铁衰老为靶点的脑卒中治疗药物提供先导化合物和理论基础。'));
children.push(p('3. 壮瑶药桂艾挥发油多成分整合效应的初步探索。明确BCP单体与桂艾挥发油整体的药效关系定位，为民族药现代化研究提供新的思路，有助于揭示壮瑶药传统经验的现代科学内涵。'));

children.push(h1('五、研究方案'));

children.push(h2('5.1 实验材料'));
children.push(bullet('实验动物：SPF级C57BL/6J小鼠（22-25 g，雄性），购自北京维通利华；动物饲养于屏障环境，12 h光暗循环，自由摄食饮水。'));
children.push(bullet('药物与试剂：\u03b2-石竹烯（Sigma-Aldrich，纯度\u226598.5%）、桂艾挥发油（本项目提取，GC-MS鉴定）、Erastin/RSL3/Liproxstatin-1（MCE）、DFO（Sigma）、ML385（MCE）、4-HNE（Sigma）、CCK-8试剂盒（同仁化学）、SA-\u03b2-gal染色试剂盒（CST）、C11-BODIPY 581/591（Invitrogen）、FeRhoNox-1（MCE）、Phen Green SK（Invitrogen）。'));
children.push(bullet('主要抗体：抗-GPX4、抗-ACSL4、抗-FTH1、抗-TFR1、抗-p53、抗-磷酸化p53（Ser15）、抗-p21、抗-p16、抗-\u03b3H2AX、抗-4-HNE、抗-DNP、抗-Nrf2、抗-Keap1、抗-HO-1、抗-SLC7A11、抗-NeuN、抗-GFAP、抗-Iba-1等，均购自CST、Abcam或Proteintech。'));
children.push(bullet('病毒载体：AAV-shACSL4、AAV-ACSL4过表达、AAV-shNrf2、AAV-p53 WT及突变体（Cys\u2192Ala），均由汉恒生物或吉凯基因包装制备，滴度\u22651\u00d710\u00b9\u00b2 vg/mL。'));
children.push(bullet('道地药材：广西道地桂艾采自广西药用植物园艾叶GAP种植基地，经广西中医药大学中药鉴定教研室鉴定。水蒸气蒸馏法提取挥发油，GC-MS进行成分分析。'));

children.push(h2('5.2 主要实验方法'));
children.push(bullet('小鼠MCAO/R模型：改良线栓法，缺血60 min后再灌注，激光多普勒血流仪监测脑血流确保模型成功。'));
children.push(bullet('原代神经细胞培养：新生24 h内小鼠皮层神经元和星形胶质细胞分离培养，纯度鉴定采用MAP2和GFAP免疫荧光。'));
children.push(bullet('OGD/R模型：缺氧培养箱（1% O2）无糖Earle\'s液孵育2 h，恢复正常培养基和常氧条件24 h。'));
children.push(bullet('铁死亡检测：Phen Green SK/FeRhoNox-1探针测Fe\u00b2\u207a、C11-BODIPY测脂质ROS（共聚焦/流式）、GSH/GSSG比色法、TBA法测MDA、透射电镜观察线粒体形态。'));
children.push(bullet('衰老检测：SA-\u03b2-gal染色、Western Blot检测p21/p16/\u03b3H2AX、qRT-PCR检测SASP因子、免疫荧光共定位。'));
children.push(bullet('分子生物学：Western Blot、qRT-PCR、免疫荧光共定位、免疫沉淀（IP）、ChIP-qPCR、ELISA、分子对接（AutoDock Vina）、CETSA等常规技术。'));
children.push(bullet('行为学：转棒实验、足误实验，评估远期运动功能恢复。'));
children.push(bullet('脑立体定位注射：坐标前囟后0.5 mm、旁开3.0 mm、深3.5 mm，微量注射泵注射AAV载体（1 \u03bcL/侧）。'));
children.push(bullet('GC-MS脑组织药物浓度测定：采用选择离子监测（SIM）模式，以选择离子峰面积定量，绘制标准曲线，测定脑组织BCP浓度。'));

children.push(h2('5.3 统计学分析'));
children.push(p('所有数据采用GraphPad Prism 9.0和SPSS 26.0软件分析。计量资料以均数\u00b1标准差（x\u0304 \u00b1 s）表示。两组间比较采用独立样本t检验或Mann-Whitney U检验。多组间比较采用单因素方差分析，组间两两比较采用Tukey法或Dunnett法；重复测量数据采用重复测量方差分析。相关性分析采用Pearson或Spearman秩相关。'));

children.push(pNoIndent([
  new TextRun({ text: '多重比较校正：', size: 24, font: '宋体', bold: true }),
  new TextRun({ text: '研究内容一涉及大量分子指标的时序检测（5个时间点\u00d78-10个标志物），研究内容三涉及多个行为学检测范式。对于此类高通量检测，采用Benjamini-Hochberg法进行错误发现率（FDR）校正，FDR阈值设为q < 0.05。结果呈现中，将校正后仍显著的指标（q < 0.05）与仅名义显著（P < 0.05但q \u2265 0.05）的指标分开报告，以避免假阳性膨胀。对于确证性实验（如Western Blot验证特定靶点），由于有明确的先验假设，可采用未校正的P值，但需在方法中说明。', size: 24, font: '宋体' })
]));

children.push(pNoIndent([
  new TextRun({ text: '样本量估算：', size: 24, font: '宋体', bold: true }),
  new TextRun({ text: '以梗死体积为主要终点，依据Hu等[3]报道的效应量（BCP 306 mg/kg组与模型组比较，Cohen\'s d \u2248 1.2），设\u03b1 = 0.05（双侧），Power = 0.80，采用G*Power 3.1软件估算，每组需8只动物；考虑20%的脱落率（建模失败、术后死亡等），确定每组n = 12。涉及多组多重比较时，按Bonferroni校正后的\u03b1水平重新估算。细胞实验每组设3-6个复孔，独立重复实验不少于3次。', size: 24, font: '宋体' })
]));

children.push(p('P < 0.05（或q < 0.05，视分析而定）认为差异具有统计学意义。'));

children.push(h1('六、技术路线'));

const techRows = [
  [
    { text: '第一部分\n现象+机制\n（第1年，重点）', fill: 'E8F5E9' },
    { text: 'CIRI半暗带铁依赖性SIPS的存在性、时空特征及4-HNE机制选择性研究', fill: 'E8F5E9' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: '\u2193 时间：6h / 24h / 3d / 7d / 28d  \u2193 空间：核心区 / 半暗带 / 对侧区  \u2193 细胞：神经元 / 星形胶质细胞 / 小胶质细胞', fill: 'FFFFFF' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: '关键技术：免疫荧光共定位 | SA-\u03b2-gal | 普鲁士蓝 | 透射电镜 | Western Blot | IP/IB | ChIP-qPCR', fill: 'FFF3E0' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: '\u2193 机制选择性验证：Keap1修饰 vs p53修饰时序 + p53位点突变 + SLC7A11转录抑制验证 \u2193', fill: 'FFFFFF' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: '\u2193 ACSL4双向验证：AAV-shACSL4（敲低） + AAV-ACSL4（过表达） \u2193', fill: 'FFFFFF' }
  ],
  [
    { text: '第二部分\n细胞机制\n（第2年，重点）', fill: 'E3F2FD' },
    { text: 'BCP通过Nrf2通路阻断铁依赖性SIPS的细胞分子机制', fill: 'E3F2FD' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: '\u2193 体外SIPS模型严格验证  \u2193 Nrf2依赖性双重验证  \u2193 分子轴调控 \u2193', fill: 'FFFFFF' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: 'Erastin / OGD/R模型 | 铁死亡指标+逆转实验 | 4-HNE量效曲线 | C11-BODIPY+SA-\u03b2-gal双染    Nrf2过表达/沉默+ML385 | 分子对接 | CETSA    BCP\u2192Nrf2\u2192GPX4\u2191\u21924-HNE\u2193\u2192阻断p53转向\u2192SLC7A11\u2191 | 挽救实验 | 时序分析', fill: 'FFFFFF' }
  ],
  [
    { text: '第三部分\n整体药效\n（第3年）', fill: 'FCE4EC' },
    { text: 'BCP改善CIRI远期预后的整体药效 + Nrf2依赖双重验证', fill: 'FCE4EC' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: '\u2193 预实验：剂量探索+药代动力学  \u2193 整体药效  \u2193 Nrf2依赖（ML385+AAV-shNrf2） \u2193', fill: 'FFFFFF' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: 'MCAO/R模型 | TTC | 脑水肿 | mNSS | 转棒 | 足误（28d）    ML385药理学抑制 + 脑区特异性shNrf2 | 功能 + SIPS标志物验证', fill: 'FFFFFF' }
  ],
  [
    { text: '第四部分\n民族药\n（第3年，探索）', fill: 'F3E5F5' },
    { text: '桂艾挥发油与BCP单体药效比较（多成分整合效应探索）', fill: 'F3E5F5' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: '\u2193 桂艾挥发油制备+GC-MS鉴定  \u2193 药效比较  \u2193 Nrf2激活谱比较 \u2193', fill: 'FFFFFF' }
  ],
  [
    { text: '', fill: 'FFFFFF' },
    { text: '\u2193 最终结论：BCP通过激活Nrf2抑制铁依赖性SIPS，改善CIRI远期预后；桂艾挥发油可能有多成分增益', fill: 'F3E5F5' }
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
          children: [new TextRun({ text: line, size: 18, font: '宋体', bold: i === 0 })]
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
children.push(p('铁死亡和细胞衰老是当前CIRI研究的两个热点领域，二者的交互作用（铁衰老）是前沿交叉方向，具有坚实的理论基础。铁死亡在CIRI中的作用已被国内外众多实验室独立验证[3,7,13]；细胞衰老参与缺血后脑损伤慢性化进程也得到越来越多证据支持[8]；4-HNE作为脂质过氧化产物诱导衰老的效应已在多种细胞类型中得到证实[2]；p53在铁死亡和衰老调控中的双重角色已有充分文献支撑；Nrf2作为二者共同防御枢纽的地位已经确立[12]。Monroe等[2]明确证实4-HNE通过DNA损伤-DDR-p53磷酸化（Ser15）通路间接激活p53诱导衰老，为本项目的机制假说提供了直接的文献依据。道吉吉等[16]在神经元衰老模型中证实了铁死亡与衰老的共存及Nrf2的共同调控作用。本项目组前期通过4个脑缺血数据集的整合分析，进一步证实了铁衰老转录特征在脑缺血中的稳健存在：铁衰老评分在4个数据集中均表现出最大的疾病-对照效应量，五基因CIRI-铁衰老签名在3个独立数据集中得到验证（Spearman \u03c1 = 0.56-0.88），1000次置换检验证实基因集具有高度特异性（P = 0.001）。网络药理学分析预测了BCP靶点与铁死亡调控网络的高度拓扑关联（超几何检验P = 2.48 \u00d7 10\u207b\u2074\u00b3），并初步确定SAT1-Nrf2轴为关键入口节点。刘胜伟等[19]的独立网药研究也证实了BCP-p53通路的关联性。这些计算生物学结果为后续实验验证提供了坚实的理论依据和明确的研究方向。'));

children.push(h2('7.2 材料与技术可行性'));
children.push(p('本研究所需实验材料均可稳定获取。广西道地桂艾采自广西药用植物园艾叶GAP种植基地，来源明确，品种鉴定可靠；BCP标准品为Sigma-Aldrich商品化试剂，纯度\u226598.5%；AAV载体和Nrf2相关工具鼠均已商用可购。技术方面，项目涉及的实验技术均为成熟方法，具有良好的可重复性。小鼠MCAO/R模型技术操作规范成熟，项目组已掌握该模型制备技术，成功率稳定在75%以上。原代神经细胞培养、OGD/R模型、Western Blot、qRT-PCR、免疫荧光共定位、免疫沉淀等均为常规技术。铁死亡和衰老的特异性检测方法（C11-BODIPY、SA-\u03b2-gal染色等）已有大量文献支持，试剂盒商品化程度高。行为学检测均为神经科学研究的标准范式。'));

children.push(h2('7.3 前期工作基础'));
children.push(p('本团队前期已在脑缺血、铁死亡和天然药物神经保护领域积累了扎实的研究基础。在计算生物学层面，已完成4个脑缺血数据集的整合分析，建立了CIRI-铁衰老五基因签名（SAT1、EBF3、KLF6、LIFR、CD74），验证了BCP靶点与铁死亡调控网络的高度拓扑富集（超几何检验P = 2.48 \u00d7 10\u207b\u2074\u00b3）。构建了包含311个节点、1,867条边的核心PPI子网，识别出8个功能模块，完成了免疫浸润和炎症因子相关性分析，以及WGCNA共表达网络验证。在实验层面，项目组已建立小鼠MCAO/R模型和原代神经细胞培养体系，具备开展体内外实验的基础条件。Hu等[3]已证实BCP通过Nrf2/HO-1通路减轻大鼠脑缺血再灌注损伤，为本项目的延伸研究提供了直接的前期支持。此外，项目组与广西中医药大学中药鉴定教研室、广西药用植物园等单位建立了良好的合作关系，可为桂艾药材的来源鉴定和成分分析提供技术支持。'));

children.push(h1('八、特色与创新之处'));

children.push(h2('8.1 理论创新：提出并验证CIRI中铁依赖性SIPS新假说及4-HNE机制选择性新视角'));
children.push(p('现状不足：现有研究多将铁死亡和细胞衰老视为CIRI中两个独立的病理过程，分别关注急性期和慢性期，对二者之间的因果联系缺乏系统探讨。铁死亡驱动细胞衰老（铁衰老）的概念主要在肿瘤和自然衰老领域被提出，在缺血性脑损伤中的研究几近空白。特别是，4-HNE作为脂质过氧化的核心产物，其作用的机制选择性——为何在某些情况下激活Nrf2防御、在另一些情况下驱动p53衰老——这一关键问题尚未得到解答。我们的不同设计：本项目率先提出CIRI半暗带中存在缺血诱导的铁依赖性SIPS的科学假说，并进一步提出"4-HNE作用存在时间依赖性机制转换"的新视角——早期通过Keap1修饰激活Nrf2防御，晚期超过阈值后通过DDR-p53通路转向衰老。独特优势：这一假说将CIRI的急性期损伤机制与慢性期恶化机制贯通起来，为理解缺血性脑损伤的慢性化提供了新的理论视角，也为寻找能够同时干预急慢性阶段的治疗靶点开辟了新思路。对4-HNE机制选择性的探讨，为铁死亡压力下细胞命运决定（防御修复 vs 衰老阻滞）的分子开关研究提供了新的切入点。与经典ferro-aging概念的明确区分（时间尺度、触发因素、病理背景），体现了概念的精确性和科学严谨性。'));

children.push(h2('8.2 机制创新：揭示BCP通过Nrf2双阻断铁死亡-衰老的新机制'));
children.push(p('现状不足：BCP的神经保护作用已有报道，但研究多停留在宏观药效层面，对其细胞分子机制的解析不够深入；现有研究多关注单一靶点或单一病理过程，缺乏对信号调控网络的系统认识；BCP对铁死亡和衰老的研究多为独立报道，尚未将二者联系起来。我们的不同设计：本项目系统解析BCP通过激活Nrf2通路协同抑制铁死亡与铁依赖性SIPS的分子机制，明确BCP\u2192Nrf2\u2192GPX4\u2191\u21924-HNE\u2193\u2192阻断p53衰老转向\u2192SLC7A11\u2191这一完整信号轴，验证BCP对4-HNE-p53-SLC7A11正反馈环路的调控。独特优势：首次阐明壮瑶药桂艾活性成分BCP通龙路火路、除毒邪功效的铁衰老干预科学内涵，为天然小分子同时靶向急性死亡和慢性衰老提供了新的作用范式，也为脑卒中治疗药物的研发提供了新的候选靶点和先导化合物。'));

children.push(h2('8.3 模式创新：构建壮瑶药-铁衰老的整合研究新模式'));
children.push(p('现状不足：民族药现代化研究常停留在成分鉴定和活性筛选层面，与前沿生物学问题结合不够紧密，难以充分揭示民族药传统经验的科学内涵；传统的网络药理学研究多为描述性分析，与后续实验验证的衔接不够紧密。我们的不同设计：构建基于壮瑶医药理论的道地药材-功效-核心成分-铁衰老靶点-信号通路精准整合研究模式，将壮瑶医\u300c通龙路火路、除毒邪\u300d的传统功效与铁死亡-衰老交互的现代前沿生物学有机结合。明确BCP单体与桂艾挥发油的关系定位——分别代表"精准机制"与"整体观"两个研究层面，避免了单体研究脱离民族药整体特色的常见问题。前期网药分析的系统布局（多数据集验证、机器学习筛选、PPI拓扑分析、功能模块解析、免疫浸润关联、WGCNA验证、置换检验验证）为后续实验提供了明确的靶点和通路方向，体现了从计算预测到实验验证的转化医学研究思路。独特优势：为民族药现代化研究提供了可复制的范例，有助于推动壮瑶医药从经验医学向循证医学的转变。'));

children.push(h1('九、年度研究计划及预期研究成果'));

children.push(h2('9.1 第一年（2027年1月-2027年12月）'));
children.push(bullet('完成小鼠MCAO/R模型建立和方法学优化，确保模型稳定性和重复性。'));
children.push(bullet('完成铁依赖性SIPS的时空定位研究（5个时间点、3个脑区、3种细胞类型）。'));
children.push(bullet('完成4-HNE机制选择性的时序验证（Keap1修饰 vs p53修饰动力学）。'));
children.push(bullet('完成p53关键修饰位点的鉴定和功能初筛（突变体构建+IP验证）。'));
children.push(bullet('完成ACSL4双向验证（AAV-shACSL4 + AAV-ACSL4过表达）的病毒包装和立体定位注射。'));
children.push(bullet('完成桂艾挥发油的提取和GC-MS成分分析鉴定。'));
children.push(bullet('预期成果：发表SCI收录论文1篇，申请发明专利1项，培养硕士研究生1名。'));

children.push(h2('9.2 第二年（2028年1月-2028年12月）'));
children.push(bullet('建立稳定的体外铁依赖性SIPS细胞模型（神经元和星形胶质细胞），完成铁依赖性严格验证。'));
children.push(bullet('完成4-HNE剂量-效应曲线测定，确定铁死亡阈值与衰老窗口。'));
children.push(bullet('完成BCP抗铁依赖性SIPS的药效评价（剂量效应、时间效应）。'));
children.push(bullet('完成Nrf2在BCP抗铁依赖性SIPS中关键作用的细胞水平验证（过表达+沉默+ML385）。'));
children.push(bullet('完成BCP对4-HNE-p53-SLC7A11分子轴调控的系统检测（时序分析+挽救实验+ChIP-qPCR）。'));
children.push(bullet('完成C11-BODIPY+SA-\u03b2-gal双重染色共定位验证。'));
children.push(bullet('完成分子对接和CETSA探索性实验。'));
children.push(bullet('预期成果：发表SCI收录论文1-2篇，培养硕士研究生1名。'));

children.push(h2('9.3 第三年（2029年1月-2029年12月）'));
children.push(bullet('完成BCP给药剂量预实验和脑组织药代动力学初步测定。'));
children.push(bullet('完成BCP对MCAO/R模型的整体药效评价（短期指标+远期行为学）。'));
children.push(bullet('完成Nrf2依赖性的双重验证（ML385药理学抑制 + AAV-shNrf2脑区特异性敲降）。'));
children.push(bullet('完成脑组织中铁依赖性SIPS标志物的检测，确认在体抑制效应。'));
children.push(bullet('完成桂艾挥发油与BCP单体的药效初步比较。'));
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
  fs.writeFileSync('D:/铁衰老 绝不重蹈覆辙/标书_国自然标准_v8_修订版_桂艾BCP靶向Nrf2抑制铁依赖性SIPS改善CIRI.docx', buffer);
  console.log('Done: proposal v8 generated');
  console.log('File: D:/铁衰老 绝不重蹈覆辙/标书_国自然标准_v8_修订版_桂艾BCP靶向Nrf2抑制铁依赖性SIPS改善CIRI.docx');
});
