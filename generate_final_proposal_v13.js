const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageBreak, LevelFormat, Header, Footer, PageNumber
} = require("docx");

const FIGS_DIR = "D:/铁衰老 绝不重蹈覆辙/figures";

// ========== Helpers ==========
function p(texts, opts = {}) {
  const children = texts.map(t => {
    if (typeof t === "string") return new TextRun(t);
    return new TextRun({ text: t.text, ...t.opts });
  });
  return new Paragraph({ children, ...opts });
}

function boldP(texts, opts = {}) {
  const children = texts.map(t => {
    if (typeof t === "string") return new TextRun({ text: t, bold: true });
    return new TextRun({ text: t.text, bold: true, ...t.opts });
  });
  return new Paragraph({ children, ...opts });
}

function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, bold: true, font: "SimHei", size: 32 })]
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, bold: true, font: "SimHei", size: 28 })]
  });
}

function heading3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [new TextRun({ text, bold: true, font: "SimHei", size: 24 })]
  });
}

function spacedText(text) {
  return new Paragraph({
    spacing: { line: 360 },
    children: [new TextRun({ text, font: "SimSun", size: 24 })]
  });
}

function smallText(text) {
  return new Paragraph({
    spacing: { line: 300 },
    children: [new TextRun({ text, font: "SimSun", size: 20 })]
  });
}

function imagePara(name, width, height) {
  const imgPath = `${FIGS_DIR}/${name}`;
  if (!fs.existsSync(imgPath)) {
    console.log(`WARNING: Image not found: ${imgPath}`);
    return p([`[图片未找到: ${name}]`]);
  }
  const buf = fs.readFileSync(imgPath);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 200, after: 100 },
    children: [new ImageRun({
      type: "png",
      data: buf,
      transformation: { width, height },
      altText: { title: name, description: name, name }
    })]
  });
}

function figureCaption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 200 },
    children: [new TextRun({ text, font: "SimHei", size: 18, italics: false })]
  });
}

// Scale image to fit page width. docx-js ImageRun transformation expects pixels
// and converts internally using 96 DPI (1 px = 9525 EMU). A4 content width is
// ~6.27 inches, so at 96 DPI that's ~600 px.
function scaleImage(pxW, pxH, targetWpx = 600) {
  const ratio = pxH / pxW;
  const w = targetWpx;
  const h = Math.round(targetWpx * ratio);
  return { width: w, height: h };
}

// ========== Build Document ==========
async function main() {
  const doc = new Document({
    styles: {
      default: {
        document: {
          run: { font: "SimSun", size: 24 }
        }
      },
      paragraphStyles: [
        {
          id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 32, bold: true, font: "SimHei" },
          paragraph: { spacing: { before: 360, after: 240 }, outlineLevel: 0 }
        },
        {
          id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 28, bold: true, font: "SimHei" },
          paragraph: { spacing: { before: 240, after: 180 }, outlineLevel: 1 }
        },
        {
          id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 24, bold: true, font: "SimHei" },
          paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 }
        }
      ]
    },
    sections: [{
      properties: {
        page: {
          size: { width: 11906, height: 16838 }, // A4
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
        }
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ text: "国家自然科学基金申请书", font: "SimHei", size: 18, color: "888888" })]
          })]
        })
      },
      children: [
        // ============ TITLE ============
        new Paragraph({ spacing: { before: 600 } }),
        boldP([
          { text: "国家自然科学基金申请书", opts: { size: 36 } }
        ], { alignment: AlignmentType.CENTER }),
        new Paragraph({ spacing: { before: 200 } }),
        boldP([
          { text: "（地区科学基金项目）", opts: { size: 28 } }
        ], { alignment: AlignmentType.CENTER }),
        new Paragraph({ spacing: { before: 400 } }),
        boldP([
          { text: "广西道地壮药桂艾活性成分\u03B2-石竹烯靶向Nrf2通路", opts: { size: 28 } }
        ], { alignment: AlignmentType.CENTER }),
        boldP([
          { text: "抑制缺血诱导的铁依赖性SIPS改善脑缺血再灌注损伤的机制研究", opts: { size: 28 } }
        ], { alignment: AlignmentType.CENTER }),
        new Paragraph({ children: [new PageBreak()] }),

        // ============ 一、立项依据与研究内容 ============
        heading1("一、立项依据与研究内容"),
        heading2("（一）国内外研究现状与进展"),
        heading3("1. 脑缺血再灌注损伤的临床困境与病理机制复杂性"),

        spacedText("脑卒中是全球成人死亡和长期残疾的首要原因之一。我国脑卒中已成为首位致死和致残病因，每年新发病例超200万，现存患者逾1300万，约70%存活患者遗留神经功能缺损。急性缺血性脑卒中约占全部脑卒中的70%-80%，治疗核心在于尽早恢复血流灌注。然而，静脉溶栓时间窗仅4.5小时，血管内治疗虽延长至6-24小时，但即便成功血管再通，仍有近半数患者预后不良，提示脑缺血再灌注损伤（cerebral ischemia-reperfusion injury, CIRI）是阻碍患者获益的关键因素。"),

        spacedText("CIRI病理机制涉及氧化应激、兴奋性毒性、神经炎症、血脑屏障破坏及多种调节性细胞死亡。半暗带概念为干预提供了理论框架——缺血核心区细胞快速坏死，半暗带细胞虽受应激但维持膜完整性，具有可逆性。然而随再灌注时间延长，半暗带细胞可通过多种机制发生进行性死亡或进入功能异常状态。除经典坏死、凋亡外，铁死亡（ferroptosis）、焦亡等新型调节性细胞死亡方式在CIRI中的作用逐渐被揭示。同时，越来越多证据表明半暗带中存在一类既未死亡也未恢复正常的细胞群体，进入持续性应激状态，通过旁分泌效应影响微环境，阻碍神经修复。细胞衰老（cellular senescence）正是这种状态的典型代表。"),

        heading3("2. 铁死亡：CIRI急性期神经元损伤的关键执行者"),

        spacedText("铁死亡是Stockwell团队于2012年正式命名的一种铁依赖性、脂质过氧化驱动的调节性细胞死亡方式，其形态学、生化和遗传学特征均区别于凋亡、坏死和自噬[4]。铁死亡核心机制是抗氧化防御系统失能，导致含多不饱和脂肪酸的磷脂过氧化物堆积至致死量，最终破坏质膜完整性。其调控涉及三大核心通路：System Xc\u207B/GPX4抗氧化轴、ACSL4/LPCAT3脂质重塑轴以及铁代谢调控轴。GPX4是唯一能直接还原磷脂氢过氧化物的酶，被视为铁死亡核心守门分子；System Xc\u207B摄取胱氨酸维持GSH合成；ACSL4催化多不饱和脂肪酸掺入膜磷脂，决定细胞对铁死亡的敏感性[5]。"),

        spacedText("近年来，铁死亡在CIRI中的作用得到广泛验证。脑缺血再灌注后缺血脑组织出现游离铁升高、GSH耗竭、GPX4活性下降、4-HNE和MDA堆积及线粒体皱缩等典型铁死亡特征[7,11,13]。胡晴雯等[3]在大鼠MCAO/R模型中系统证实铁死亡的存在：再灌注后铁死亡标志物呈时间依赖性升高，TEM观察到缺血皮层神经元线粒体皱缩、膜密度增高；铁死亡抑制剂DFO可显著减小梗死体积、改善神经功能。药理学干预证实Ferrostatin-1、Liproxstatin-1及DFO均可减小MCAO模型梗死体积[7,11]。然而，铁死亡在CIRI中的作用并非仅限于急性期细胞杀伤——亚致死剂量的铁死亡应激是否会触发细胞进入其他应激状态（如细胞衰老），目前尚缺乏系统研究。"),

        heading3("3. 细胞衰老：缺血半暗带慢性化的重要推手"),

        spacedText("细胞衰老是指细胞在应激因素作用下退出细胞周期，进入稳定的增殖停滞状态，同时伴随广泛的基因表达和代谢重编程，形成特征性的衰老相关分泌表型（senescence-associated secretory phenotype, SASP）[6]。经典细胞衰老包括端粒依赖性的复制性衰老和应激诱导的早熟性衰老（stress-induced premature senescence, SIPS）。SIPS核心分子通路包括p53/p21^CIP1轴和p16^INK4a/Rb轴，二者共同驱动细胞周期停滞。SASP是衰老细胞最具病理影响力的特征，包括促炎细胞因子、趋化因子、生长因子、基质金属蛋白酶等，通过旁分泌方式重塑组织微环境。"),

        spacedText("传统观点认为细胞衰老主要与机体老化和年龄相关疾病有关，在急性损伤中作用有限。然而，近年来这一观念正在被修正。在脑缺血领域，越来越多证据表明缺血性损伤可在梗死周围半暗带诱导多种细胞类型发生衰老样改变。啮齿类动物MCAO模型中，再灌注后数天内即可在半暗带检测到SA-\u03B2-gal阳性细胞，伴随p21和p16表达上调、\u03B3H2AX焦点增加[8]。发生衰老的细胞类型包括神经元、星形胶质细胞、小胶质细胞及血管内皮细胞[8]。道吉吉等[16]在D-半乳糖诱导的神经元衰老模型中证实，铁死亡相关分子（Nrf2、SLC7A11、GPX4下调，TFRC上调）与衰老标志物（p53、p21、p16上调）共存，Erastin可进一步加剧衰老表型，而靶向ErbB4受体的小分子激动剂可通过Akt/Nrf2通路同时抑制铁死亡和衰老。缺血诱导的衰老细胞通过SASP促炎因子招募免疫细胞、激活胶质细胞形成慢性炎症微环境，基质金属蛋白酶破坏血脑屏障，扩大损伤范围。衰老细胞清除剂（senolytics）在多种脑损伤模型中显示改善功能预后的效果。然而，脑缺血后细胞衰老的触发因素和上游驱动机制尚未完全阐明，特别是氧化应激在衰老启动中的具体作用形式和分子通路仍有待深入揭示。"),

        heading3("4. 铁死亡与细胞衰老的交汇：铁衰老概念及其在CIRI中的研究空白"),

        spacedText("铁死亡与细胞衰老并非两个独立的病理过程，二者存在密切的交互作用和共同的上游驱动因素——氧化应激和铁代谢紊乱。近年来，研究者开始将这两个领域联系起来，提出了\u201C铁衰老\u201D（ferro-aging/ferro-senescence）的概念框架。2026年，Liu等[1]在Cell Metabolism发表的灵长类动物研究为铁衰老概念提供了系统性体内证据：随着灵长类年龄增长，肝脏和血清中铁水平逐渐升高，伴随ACSL4表达上调和脂质过氧化产物积累；铁过载可通过ACSL4介导的脂质过氧化通路驱动肝细胞和造血干细胞衰老；抑制ACSL4可减轻衰老表型、延长健康寿命。该研究首次在高等哺乳动物中确立了ferro-aging作为生理性衰老驱动因素的地位。"),

        spacedText("4-羟基壬烯醛（4-hydroxynonenal, 4-HNE）是\u03C9-6多不饱和脂肪酸过氧化的主要毒性醛类产物之一，具有高度亲电性，可与蛋白质的半胱氨酸、组氨酸和赖氨酸残基发生Michael加成反应，形成稳定的蛋白质羰基化修饰[14]。Monroe等[2]在Aging Cell发表的研究证实，4-HNE等脂质过氧化产物可在人成纤维细胞和小鼠脂肪干细胞中剂量依赖性地诱导衰老表型，包括SA-\u03B2-gal阳性率升高、\u03B3H2AX焦点积累、p53磷酸化增强、p21表达上调及SASP因子分泌增加。机制研究表明，4-HNE主要通过诱导氧化性DNA损伤、激活DDR通路导致p53 Ser15磷酸化，进而启动p21介导的细胞周期阻滞。"),

        spacedText("基于上述文献线索，我们提出一个经过修正的4-HNE-p53-SLC7A11正反馈环路假说：铁死亡产生的4-HNE首先通过Keap1修饰激活Nrf2防御通路；当4-HNE持续积累超过防御阈值时，转而通过氧化性DNA损伤激活DDR-p53-p21通路驱动细胞衰老；同时，活化的p53可转录抑制SLC7A11[27]，进一步削弱GSH合成和抗氧化防御，加剧脂质过氧化和4-HNE生成，形成\u201C铁死亡\u21924-HNE\u2192p53\u2192SLC7A11\u2193\u2192更多铁死亡\u201D的正反馈环路，最终将亚致死性氧化应激转化为持续性的衰老表型。然而，目前关于铁死亡驱动细胞衰老的研究主要集中于肿瘤细胞和体外细胞系，在神经系统特别是缺血性脑损伤中的研究极为有限。CIRI半暗带中是否确实存在铁依赖性SIPS？其时空分布特征如何？4-HNE作用的机制选择性如何决定细胞命运？这些问题目前均缺乏直接的实验证据，有待系统研究加以阐明。"),

        heading3("5. Nrf2通路：铁死亡防御与衰老抑制的共同枢纽"),

        spacedText("核因子E2相关因子2（nuclear factor erythroid 2-related factor 2, Nrf2）是细胞抗氧化反应的主调控因子。生理状态下，Nrf2被Keap1锚定在胞浆中，经Cullin-3介导的泛素-蛋白酶体途径快速降解。当细胞暴露于氧化应激或亲电试剂时，Keap1关键半胱氨酸残基发生共价修饰，Nrf2逃脱降解并转位入核，与小Maf蛋白形成异二聚体，结合于抗氧化反应元件（ARE），启动下游数百个靶基因转录。"),

        spacedText("Nrf2通路是铁死亡的重要防御机制。Nrf2可通过多层面调控抑制铁死亡发生：在抗氧化层面，上调GPX4表达直接增强磷脂过氧化物清除能力，上调SLC7A11促进胱氨酸摄取和GSH合成；在铁代谢层面，转录激活铁蛋白（FTH1/FTL）促进游离铁螯合储存，上调铁输出蛋白FPN1降低不稳定铁池水平，下调TFR1减少铁摄取[12]。胡晴雯等[3]在大鼠MCAO/R模型中证实，BCP可显著增加Nrf2核转位，上调HO-1和GPX4表达，抑制ACSL4，从而减轻铁死亡和脑损伤；Nrf2抑制剂ML385可显著逆转BCP的神经保护效应，直接证明了Nrf2通路的关键作用。"),

        spacedText("Nrf2同样是抑制细胞衰老和SASP的关键因子。多项研究表明，Nrf2的表达和活性随年龄增长而下降，与衰老表型出现和年龄相关疾病发展密切相关。在细胞水平，Nrf2缺失或功能低下可加速氧化应激诱导的SIPS；反之，Nrf2激活则可延缓多种细胞类型的衰老进程，减轻SASP相关炎症反应[15]。道吉吉等[16]在D-半乳糖诱导的神经元衰老模型中证实，Nrf2抑制剂可显著削弱ErbB4激动剂对SLC7A11和GPX4的上调作用，同时阻碍其逆转神经元衰老的效果，直接证明了Nrf2在铁死亡-衰老调控轴中的核心地位。王淦民等[28]在骨骼肌衰老模型中进一步证实，Sirt6通过Nrf2/HO-1信号途径抑制铁死亡并延缓衰老，Nrf2 siRNA可显著减弱Sirt6激动剂的抗铁死亡和抗衰老效果，为Nrf2作为铁死亡-衰老共同调控枢纽提供了新证据。鉴于Nrf2在铁死亡防御和衰老抑制中的双重核心作用，我们认为Nrf2通路是理想的干预靶点——激活Nrf2有望同时阻断铁死亡的急性期损伤和铁依赖性SIPS的慢性期损害，实现对CIRI病理进程的双阶段干预。"),

        heading3("6. 壮瑶药艾叶/桂艾及其活性成分\u03B2-石竹烯"),

        spacedText("艾叶（Artemisia argyi L\u00E9vl. et Vant.）为菊科蒿属植物的干燥叶，是我国传统中药和广西道地壮瑶药材。艾叶入药历史悠久，始载于《名医别录》，其性温，味辛、苦，归肝、脾、肾经，具有温经止血、散寒止痛、外用祛湿止痒等功效。在壮瑶医药理论体系中，艾叶具有独特地位：壮语称\u201C挨\u201D，瑶语称\u201C各艾\u201D，被视为\u201C通龙路火路、除风毒寒毒、逐湿邪\u201D之要药。壮医\u201C三道两路\u201D理论认为，脑（巧坞）为神明之府，龙路火路网络密布；中风（麻邦）是由于风毒、火毒、痧毒循龙路火路上攻巧坞，致三道两路不通、气血失衡、天地人三气不能同步。艾叶的\u201C通龙路火路、除毒邪\u201D功效恰中中风病病机要害，因此在壮瑶医临床中常用于麻痹、头痛、眩晕等脑病防治。"),

        spacedText("桂艾是产于广西地区的艾叶道地药材，因独特的地理气候条件而具有挥发油含量高、活性成分丰富的特点。桂艾挥发油是艾叶的主要药效物质基础，得油率约为0.45%-1.2%，已从中分离鉴定出数十种化学成分，其中\u03B2-石竹烯（\u03B2-caryophyllene, BCP）含量可高达15%-35%，是桂艾挥发油中最重要的倍半萜成分之一。BCP是一种天然存在的双环倍半萜化合物，已被美国FDA和欧洲食品安全局批准为食品添加剂，安全性极高。"),

        spacedText("现代药理学研究揭示了BCP的多种生物活性。BCP是大麻素CB2受体的选择性激动剂，可通过CB2受体依赖的信号通路发挥抗炎和免疫调节作用。在神经系统疾病领域，BCP的神经保护作用已在多种模型中得到验证。Chang等[17]首次报道BCP可显著减小小鼠MCAO模型皮层梗死体积（67%），机制与抑制小胶质细胞炎症介质释放有关。Zhang等[18]证实BCP预处理通过激活PI3K/Akt通路抑制凋亡减轻大鼠CIRI。Yang等[19]发现BCP通过抑制HMGB1/TLR4/NF-\u03BAB通路减轻小鼠CIRI炎症反应。Tian等[20]进一步揭示BCP通过TLR4通路促进小胶质细胞向M2表型极化。Rao等[21]发现BCP通过激活Pink1/Parkin2信号促进线粒体自噬保护神经元。刘京东等[22]报道BCP通过抑制Notch1/NF-\u03BAB信号轴减轻大鼠CIRI。左天睿等[23]证实BCP通过激活HSF1/HSP70通路减轻氧化应激和凋亡。胡晴雯等[3]在Phytomedicine发表的研究明确证实BCP可通过激活Nrf2/HO-1通路减轻大鼠CIRI：BCP预处理剂量依赖性地减小MCAO/R梗死体积、降低脑水肿、改善神经功能评分，伴随Nrf2核转位增加、HO-1表达上调及氧化应激指标改善；Nrf2抑制剂ML385可显著逆转BCP的神经保护效应。刘胜伟等[24]的网络药理学研究证实BCP作用于CIRI的靶点富集于p53、MAPK、NF-\u03BAB、PPAR等通路。赵微等[29]进一步在血管性痴呆大鼠模型中报道BCP可上调GPX4、下调ACSL4，降低脑组织Fe\u00B2\u207A和MDA水平，改善认知功能，为BCP在脑病中抑制铁死亡提供了新证据。李尤[30]在系统性铁死亡研究中证实BCP可通过多靶点抑制铁死亡。Shen等[31]发现BCP可通过JAK2/STAT3通路抑制中性粒细胞胞外陷阱形成，减轻CIRI后的炎症和氧化应激。药代动力学研究表明BCP具有良好的血脑屏障穿透能力，口服给药后可迅速进入脑组织。然而，BCP是否能够通过激活Nrf2同时阻断铁死亡和铁依赖性SIPS，是否能够通过干预铁死亡-衰老正反馈环路改善CIRI远期预后，目前尚未见文献报道。"),

        heading3("7. 本项目拟解决的核心科学问题及研究意义"),

        spacedText("综上所述，脑缺血再灌注损伤是急性缺血性脑卒中治疗中的重大临床难题，其病理机制涉及急性期多种调节性细胞死亡和慢性期神经炎症与修复障碍。铁死亡作为急性期神经元死亡的关键形式已得到广泛认可[7,13]；细胞衰老则被认为是缺血半暗带慢性化的重要推手[8]。然而，目前尚不清楚铁死亡是否以及如何驱动缺血半暗带中的细胞衰老（铁依赖性SIPS）。特别是，4-HNE作为脂质过氧化的核心产物，其作用的机制选择性（优先激活Keap1-Nrf2防御还是转向p53介导的衰老）是决定亚致死铁死亡压力最终细胞命运的关键科学问题。"),

        spacedText("基于文献线索和我们前期的网络药理学研究，我们推测：CIRI半暗带中存在亚致死性铁死亡压力驱动的铁依赖性SIPS；4-HNE作用存在时间依赖性的机制转换——早期激活Keap1-Nrf2防御，晚期超过阈值后通过DDR-p53通路驱动衰老；4-HNE-p53-SLC7A11正反馈环路是铁依赖性SIPS维持和放大的核心分子机制；Nrf2作为铁死亡防御与衰老抑制的共同枢纽，是理想的干预靶点；广西道地壮药桂艾的活性成分\u03B2-石竹烯可通过激活Nrf2通路，同时阻断铁死亡急性期损伤和铁依赖性SIPS慢性损害，从而改善CIRI远期预后。"),

        spacedText("本研究的理论意义在于：提出并验证铁依赖性SIPS这一新型病理概念，丰富对CIRI慢性化机制的认识；阐明4-HNE作用的机制选择性及其在铁依赖性SIPS启动中的开关作用。实践意义在于：阐明壮瑶药桂艾活性成分BCP的神经保护新机制，为开发以铁衰老为靶点的脑卒中治疗药物提供理论依据和先导化合物，也为民族药现代化研究提供可借鉴范式。"),

        // ============ 项目组前期研究基础 ============
        heading2("（二）项目组前期研究基础"),

        spacedText("为验证上述科学假说，项目组前期整合多维度公开数据集与生物信息学方法，系统开展了脑缺血-铁衰老-石竹烯的多组学整合研究，完成了从单细胞图谱构建、疾病特征识别、药物靶点预测到细胞通讯解析的全链条计算分析。"),

        boldP(["1. 单细胞转录组图谱与铁衰老表型鉴定"]),
        spacedText("项目组基于GSE233815小鼠脑缺血snRNA-seq数据集（7414个单核，25种细胞类型），系统构建了CIRI单细胞转录组图谱。采用AddModuleScore方法计算96基因铁衰老评分，在单细胞水平揭示了铁衰老活性的细胞类型异质性（图1）。结果显示，铁衰老评分在MCAO组显著高于对照组，且在小胶质细胞、星形胶质细胞和神经元中呈现差异化分布模式。小胶质细胞亚群时序分析进一步揭示了再灌注后1天至7天铁衰老活性的动态演变规律（图4B）。"),

        imagePara("Fig1_Composite_singlecell_atlas.png", ...Object.values(scaleImage(4200, 3300))),
        figureCaption("图1 CIRI单细胞转录组图谱与铁衰老表型鉴定"),

        boldP(["2. CIRI-铁衰老转录特征的识别与验证"]),
        spacedText("项目组整合4个脑缺血数据集（GSE104036、GSE16561、GSE61616、GSE97537，共117样本），采用ssGSEA方法[9]计算铁衰老评分。结果显示，在全部4个数据集中，铁衰老评分的疾病-对照效应量均大于单纯铁死亡或衰老评分，提示铁衰老基因集与缺血性脑损伤关联更为紧密，具有跨物种、跨平台稳健性。1000次置换检验证实该基因集具有高度特异性（P = 0.001）。时序分析显示铁衰老评分在再灌注后6h达峰（图3A）。经LASSO稳定性筛选（50次重复6折交叉验证），获得SAT1、EBF3、KLF6、LIFR、CD74共5个CIRI-铁衰老特征基因（图2B），在3个独立数据集中得到验证（Spearman相关系数：GSE16561 rho=0.56，GSE61616 rho=0.75，GSE97537 rho=0.88；图3C）。GSE61616差异表达分析中，85/96个铁衰老基因被检测到，差异显著者标注于火山图（图3B）。PPI网络分析识别出TP53、STAT3、HIF1A、TNF、IL6等25个Hub基因，度中心性和介数中心性排名前25（图3D）。"),

        imagePara("Fig3_Composite_transcriptomic_validation.png", ...Object.values(scaleImage(3600, 4800))),
        figureCaption("图3 CIRI-铁衰老转录特征的跨数据集验证与PPI网络分析"),

        boldP(["3. GNN药物筛选与石竹烯靶点拓扑关联"]),
        spacedText("项目组构建了基于图神经网络（GNN）的CIRI-铁衰老化合物-靶点预测模型，系统比较了SAGE、HGT、SimpleHGN三种架构的预测性能（图2A）。基于517种中药化合物的虚拟筛选识别出前25个候选化合物，BCP位列其中（图2C-D）。化合物理化性质分析表明BCP符合Lipinski五规则，具有良好成药性（图4A）。采用交集+网络邻近扩展双路径策略，在STRING v12.0网络[10]中，BCP靶点群与铁死亡调控网络存在高度拓扑关联（超几何检验P = 2.48 \u00D7 10\u207B\u2074\u00B3），其中SAT1为BCP的直接靶点（选择频率96%），CD74、KLF6、LIFR为BCP靶点的一阶PPI邻居（图4C）。刘胜伟等[24]的网络药理学研究同样证实BCP作用于CIRI的靶点富集于p53、MAPK、NF-\u03BAB等通路，与本分析结果高度一致。"),

        imagePara("Fig2_Composite_gnn_compound_screening.png", ...Object.values(scaleImage(3600, 4800))),
        figureCaption("图2 GNN药物筛选与CIRI-铁衰老特征基因鉴定"),

        imagePara("Fig4_Composite_chemistry_microglia.png", ...Object.values(scaleImage(3900, 3300))),
        figureCaption("图4 BCP理化性质与铁死亡靶点网络拓扑关联"),

        boldP(["4. SCISSOR表型关联与细胞通讯网络解析"]),
        spacedText("项目组采用SCISSOR算法[25]整合GSE233815单细胞转录组与GSE61616批量转录组表型，识别出与CIRI表型显著相关的Scissor+细胞亚群（图5A）。细胞类型富集分析揭示了Scissor+细胞在特定细胞类型中的显著富集模式（图5B-C），Scissor+与Scissor-细胞间差异表达基因分析为进一步机制研究提供了候选分子（图5D）。基于CellChatDB.mouse配体-受体数据库[26]的细胞通讯网络分析识别了CIRI后762968对L-R相互作用，构建了细胞间通讯弦图（图6A）、通讯强度热图（图6B）和信号通路贡献排名（图6D），其中HMGB1、TNF、IL-6等促炎通路在再灌注后显著激活，为BCP的抗炎神经保护机制提供了系统层面的证据。"),

        imagePara("Fig5_Composite_SCISSOR.png", ...Object.values(scaleImage(4200, 4800))),
        figureCaption("图5 SCISSOR表型关联细胞亚群鉴定"),

        imagePara("Fig6_Composite_cell_communication.png", ...Object.values(scaleImage(4800, 6000))),
        figureCaption("图6 CellChat细胞通讯网络解析"),

        boldP(["5. 核心科学假说"]),
        spacedText("基于上述研究基础，我们提出以下科学假说：\u03B2-石竹烯通过激活Nrf2通路，上调GPX4、FTH1和SLC7A11等下游靶基因，增强细胞抗氧化防御能力，减少4-HNE生成，阻断4-HNE从Keap1-Nrf2防御向DDR-p53衰老的机制转换，从而抑制缺血诱导的铁依赖性SIPS的启动和SASP的分泌，打破铁死亡\u21924-HNE\u2192p53\u2192SLC7A11\u2193\u2192更多铁死亡的正反馈环路，最终改善脑缺血再灌注损伤的远期预后。BCP单体是桂艾挥发油的核心药效物质基础，桂艾挥发油因多成分整合可能呈现整体药效增益。"),

        // ============ 参考文献 ============
        heading2("参考文献"),
        smallText("[1] Liu L, Zheng Z, You W, et al. Vitamin C inhibits ACSL4 to alleviate ferro-aging in primates. Cell Metab, 2026, 38(4): 673-693."),
        smallText("[2] Monroe TB, Hertzel AV, Dickey DM, et al. Lipid peroxidation products induce carbonyl stress, mitochondrial dysfunction, and cellular senescence in human and murine cells. Aging Cell, 2025, 24(1): e14367."),
        smallText("[3] Hu Q, Zuo T, Deng L, et al. \u03B2-Caryophyllene suppresses ferroptosis induced by cerebral ischemia reperfusion via activation of the NRF2/HO-1 signaling pathway in MCAO/R rats. Phytomedicine, 2022, 102: 154112."),
        smallText("[4] Dixon SJ, Lemberg KM, Lamprecht MR, et al. Ferroptosis: an iron-dependent form of nonapoptotic cell death. Cell, 2012, 149(5): 1060-1072."),
        smallText("[5] Stockwell BR, Friedmann Angeli JP, Bayir H, et al. Ferroptosis: a regulated cell death nexus linking metabolism, redox biology, and disease. Cell, 2017, 171(2): 273-285."),
        smallText("[6] Tchkonia T, Zhu Y, van Deursen J, et al. Cellular senescence and the senescent secretory phenotype: therapeutic opportunities. J Clin Invest, 2013, 123(3): 966-972."),
        smallText("[7] Guo J, Tuo QZ, Lei P. Iron, ferroptosis, and ischemic stroke. J Neurochem, 2023, 165(4): 487-520."),
        smallText("[8] Real MGC, Falcione SR, Boghozian R, et al. Endothelial cell senescence effect on the blood-brain barrier in stroke and cognitive impairment. Neurology, 2024, 103(24): e210063."),
        smallText("[9] Barbie DA, Tamayo P, Boehm JS, et al. Systematic RNA interference reveals that oncogenic KRAS-driven cancers require TBK1. Nature, 2009, 462(7269): 108-112."),
        smallText("[10] Szklarczyk D, Kirsch R, Koutrouli M, et al. The STRING database in 2023: protein-protein association networks and functional enrichment analyses for any sequenced genome of interest. Nucleic Acids Res, 2023, 51(D1): D638-D646."),
        smallText("[11] Fang X, Wang H, Han D, et al. Ferroptosis as a target for protection against cardiomyopathy. Proc Natl Acad Sci U S A, 2019, 116(7): 2672-2680."),
        smallText("[12] Wang L, Zhang X, Xiong X, et al. Nrf2 regulates oxidative stress and its role in cerebral ischemic stroke. Antioxidants (Basel), 2022, 11(12): 2377."),
        smallText("[13] Tuo QZ, Lei P. Ferroptosis in ischemic stroke: animal models and mechanisms. Zool Res, 2024, 45(6): 1235-1248."),
        smallText("[14] Uchida K. 4-Hydroxy-2-nonenal: a product and mediator of oxidative stress. Prog Lipid Res, 2003, 42(4): 318-343."),
        smallText("[15] Guan WB, Liu XG, Zhao W. Ferroptosis and aging. Chinese Journal of Biochemistry and Molecular Biology, 2023, 39(10): 1233-1242."),
        smallText("[16] Dao JJ, Zhang W, Liu C, et al. Targeted ErbB4 receptor activation prevents D-galactose-induced neuronal senescence via inhibiting ferroptosis pathway. Front Pharmacol, 2025, 16: 1528604."),
        smallText("[17] Chang HJ, Kim JM, Lee JC, et al. Protective effect of \u03B2-caryophyllene, a natural bicyclic sesquiterpene, against cerebral ischemic injury. J Med Food, 2013, 16(6): 471-480."),
        smallText("[18] Zhang Q, An R, Tian X, et al. \u03B2-Caryophyllene pretreatment alleviates focal cerebral ischemia-reperfusion injury by activating PI3K/Akt signaling pathway. Neurochem Res, 2017, 42(5): 1459-1467."),
        smallText("[19] Yang M, An R, Li M, et al. \u03B2-Caryophyllene mitigates cerebral ischemia reperfusion injury in mice by inhibiting HMGB1/TLR4/NF-\u03BAB pathway. Chinese Journal of Immunology, 2017, 33(7): 1009-1013."),
        smallText("[20] Tian X, Liu H, Xiang F, et al. \u03B2-Caryophyllene protects against ischemic stroke by promoting polarization of microglia toward M2 phenotype via the TLR4 pathway. Life Sci, 2019, 237: 116915."),
        smallText("[21] Rao J, Wu Y, Fan X, et al. Facilitating mitophagy via Pink1/Parkin2 signaling is essential for the neuroprotective effect of \u03B2-caryophyllene against CIR-induced neuronal injury. Brain Sci, 2022, 12(7): 868."),
        smallText("[22] Liu J, Chen S, Wang Y, et al. \u03B2-caryophyllene improves cerebral ischemia reperfusion injury in rats via Notch1/NF-\u03BAB signal axis. Journal of Third Military Medical University, 2021, 43(2): 109-117."),
        smallText("[23] Zuo TR, Hu QW, Liu JD, et al. \u03B2-Caryophyllene reduces cerebral ischemia-reperfusion injury in rats by activating HSF1/HSP70 pathway. Chinese Journal of New Drugs, 2023, 32(5): 562-570."),
        smallText("[24] Liu S, Shen Z, Ren Z, et al. Network pharmacology-based study and in-vivo verification of \u03B2-caryophyllene against cerebral ischemia/reperfusion injury. Journal of New Chinese Medicine, 2024, 56(11): 63-69."),
        smallText("[25] Sun D, Guan X, Moran AE, et al. Identifying phenotype-associated subpopulations by integrating bulk and single-cell sequencing data. Nat Biotechnol, 2022, 40(4): 527-538."),
        smallText("[26] Jin S, Guerrero-Juarez CF, Zhang L, et al. Inference and analysis of cell-cell communication using CellChat. Nat Commun, 2021, 12(1): 1088."),
        smallText("[27] Jiang L, Kon N, Li T, et al. Ferroptosis as a p53-mediated activity during tumour suppression. Nature, 2015, 520(7545): 57-62."),
        smallText("[28] Wang GM, Wang Y, Chen SK, et al. Sirt6 inhibits ferroptosis and attenuates D-Gal-induced skeletal muscle aging in mice via Nrf2/HO-1 signaling pathway. Chinese Journal of Pathophysiology, 2025, 41(7): 1276-1286."),
        smallText("[29] Zhao W, Wu J. The investigation on the effect of \u03B2-caryophyllene on ferroptosis and neuroprotection mechanisms in rats with vascular dementia. Journal of Jinzhou Medical University, 2025, 46(4): 27-32."),
        smallText("[30] Li Y. Study on the function and mechanism of natural product \u03B2-caryophyllene in inhibiting ferroptosis. Hangzhou Normal University Master Thesis, 2024."),
        smallText("[31] Shen ZZ, Xu ZL, Liu QS, et al. \u03B2-Caryophyllene modulates the JAK2/STAT3 signaling pathway to downregulate neutrophil extracellular traps and alleviate cerebral ischemia-reperfusion injury. Mol Neurobiol, 2025, Epub ahead of print."),

        new Paragraph({ children: [new PageBreak()] }),

        // ============ 二、研究目标 ============
        heading1("二、研究目标"),
        p(["• 证实CIRI半暗带中铁依赖性SIPS的存在，阐明4-HNE作用的机制选择性，明确4-HNE-p53-SLC7A11分子轴。"]),
        p(["• 阐明BCP通过Nrf2通路阻断铁依赖性SIPS的细胞分子机制，验证BCP\u2192Nrf2\u2192GPX4\u2191\u21924-HNE\u2193\u2192阻断p53衰老转向\u2192SLC7A11\u2191信号轴。"]),
        p(["• 在整体动物水平验证BCP改善CIRI远期预后的药效，确认Nrf2依赖性（药理学抑制+脑区特异性敲降双重验证）。"]),
        p(["• 比较桂艾挥发油与BCP单体的药效差异，探讨民族药多成分整合效应。"]),

        // ============ 三、研究内容 ============
        heading1("三、研究内容"),
        heading2("研究内容一：CIRI中铁依赖性SIPS的时空特征及分子轴研究"),
        spacedText("明确CIRI后半暗带中铁依赖性SIPS的存在性、时空分布及核心分子机制，重点解答4-HNE作用的机制选择性。采用C57BL/6J小鼠MCAO/R模型（缺血60 min），再灌注后6 h/24 h/3 d/7 d/28 d五个时间点取材。运用IF多重标记（NeuN/GFAP/Iba-1分别与GPX4/4-HNE/FTH1及p21/\u03B3H2AX共定位）、TEM、铁含量比色法，系统定量分析铁衰老双阳性细胞的时空分布及细胞类型。"),

        heading2("研究内容二：BCP抗铁依赖性SIPS的细胞机制研究"),
        spacedText("建立稳定的体外铁依赖性SIPS模型，评价BCP干预效果，明确Nrf2关键作用。采用新生C57BL/6J小鼠原代皮层神经元和星形胶质细胞，以低剂量Erastin和OGD/R（2 h OGD + 24 h复氧）诱导亚致死量铁死亡压力。"),

        heading2("研究内容三：BCP改善CIRI远期预后的整体药效与机制验证"),
        spacedText("验证BCP对CIRI的神经保护效应，重点关注远期功能预后，确认Nrf2依赖性。采用C57BL/6J小鼠MCAO/R模型，设假手术组、模型组、BCP低/中/高剂量组、Liproxstatin-1阳性药组、BCP+ML385组，每组12只。再灌注后即刻给药，14天，28天行为学终点评价。"),

        heading2("研究内容四：桂艾挥发油与BCP单体的药效比较研究"),
        spacedText("水蒸气蒸馏法提取桂艾挥发油，GC-MS分析（BCP含量15%-35%）。MCAO/R模型中比较桂艾挥发油与等BCP摩尔含量单体在梗死体积、铁死亡和衰老标志物、Nrf2通路激活差异。"),

        // ============ 四、拟解决的关键科学问题 ============
        heading1("四、拟解决的关键科学问题"),
        p(["1. CIRI半暗带中铁依赖性SIPS的存在性及其4-HNE作用机制选择性的阐明。核心是回答\u201C4-HNE为何在铁死亡情境下转向p53介导的衰老而非激活Nrf2防御\u201D这一关键问题。"]),
        p(["2. BCP通过Nrf2通路阻断铁依赖性SIPS、改善CIRI远期预后的分子机制解析。验证BCP\u2192Nrf2\u2192GPX4\u2191\u21924-HNE\u2193\u2192阻断p53衰老转向\u2192SLC7A11\u2191信号轴。"]),
        p(["3. 壮瑶药桂艾挥发油多成分整合效应的初步探索，明确BCP单体与桂艾挥发油整体的药效关系定位。"]),

        // ============ 五、研究方案 ============
        heading1("五、研究方案"),
        heading2("5.1 研究内容一方案：CIRI中铁依赖性SIPS的时空特征及分子轴研究"),
        boldP(["动物模型"]),
        p(["C57BL/6J雄性小鼠，MCAO/R线栓法（缺血60 min），LDF监测血流。假手术组、模型组（6 h/24 h/3 d/7 d/28 d），n=12/组。"]),
        boldP(["检测"]),
        p(["（1）IF多重标记：NeuN/GFAP/Iba-1与GPX4/4-HNE/FTH1及p21/\u03B3H2AX共定位。（2）TEM观察线粒体形态。（3）铁含量比色法。（4）WB：GPX4、ACSL4、FTH1、p53/p-p53、p21、\u03B3H2AX、SLC7A11。（5）IP/IB检测4-HNE对Keap1和p53修饰时序。（6）ChIP-qPCR：p53对Slc7a11启动子结合及H3K27ac/H3K4me3修饰。（7）p53-K117R/K386R突变体验证，AAV注射后MCAO/R检测SIPS表型。"]),

        heading2("5.2 研究内容二方案：BCP抗铁依赖性SIPS的细胞机制"),
        boldP(["细胞模型"]),
        p(["新生C57BL/6J小鼠皮层原代神经元和星形胶质细胞，纯度>95%。铁依赖性SIPS诱导：（1）低剂量Erastin（0.1-0.5 \u03BCM，24 h\u2192洗脱\u219272 h）；（2）OGD/R（2 h+24 h）。BCP（10-100 \u03BCM）同时加入。"]),
        boldP(["实验分组"]),
        p(["对照组、模型组、BCP低/中/高剂量组、BCP+ML385（5 \u03BCM）组、BCP+siNrf2组、Fer-1（1 \u03BCM）阳性对照组、Navitoclax（1 \u03BCM）阳性对照组。n=3复孔。"]),
        boldP(["检测"]),
        p(["（1）铁死亡：C11-BODIPY（流式+荧光）、FerroOrange（Fe\u00B2\u207A）、MDA、GSH/GSSG、GPX4/ACSL4/FTH1/SLC7A11 WB。（2）衰老：SA-\u03B2-gal、EdU、p21/p16/\u03B3H2AX WB、SASP因子（IL-6/IL-1\u03B2/TNF-\u03B1/MMP-3）ELISA/qPCR。（3）Nrf2：核浆分离WB、ARE荧光素酶报告基因、HO-1/NQO1。（4）分子对接（AutoDock Vina）和CETSA验证BCP与Keap1结合。"]),

        heading2("5.3 研究内容三方案：BCP整体药效与Nrf2依赖性验证"),
        boldP(["动物分组"]),
        p(["C57BL/6J小鼠8组（n=12/组）：假手术组、MCAO/R组、BCP低/中/高（36/72/144 mg/kg）组、Liproxstatin-1（10 mg/kg）组、BCP+ML385（30 mg/kg）组、BCP+AAV-shNrf2组（术前14天注射）。BCP灌胃，再灌注后即刻，每日1次\u00D714天。"]),
        boldP(["药效评价"]),
        p(["24 h TTC测梗死体积，72 h脑水肿；1/3/7/14/28天mNSS评分；28天转棒和足误实验；Morris水迷宫（28-32天）。28天处死，TUNEL、SA-\u03B2-gal染色、WB检测铁死亡和SIPS标志物、ELISA检测SASP。"]),

        heading2("5.4 研究内容四方案：桂艾挥发油与BCP单体比较"),
        p(["桂艾采自广西药用植物园，水蒸气蒸馏法提取挥发油，GC-MS分析（BCP含量15%-35%）。MCAO/R模型增设桂艾挥发油组（等BCP摩尔含量），比较梗死体积、铁死亡及SIPS标志物、Nrf2通路激活差异。"]),

        heading2("5.5 实验材料"),
        p(["C57BL/6J小鼠（北京维通利华）；\u03B2-石竹烯（Sigma，\u226598.5%）、Erastin/Liproxstatin-1（MCE）、DFO、ML385、4-HNE、Navitoclax（Selleck）；SA-\u03B2-gal试剂盒（CST）、C11-BODIPY（Invitrogen）、FerroOrange（Dojindo）。抗体：GPX4、ACSL4、FTH1、p53/p-p53、p21、\u03B3H2AX、4-HNE、Nrf2、Keap1、HO-1、SLC7A11、NeuN、GFAP、Iba-1（CST/Abcam/Proteintech）。AAV-shNrf2、AAV-p53突变体（汉恒/吉凯）。"]),

        heading2("5.6 统计学分析"),
        p(["数据以mean\u00B1SD表示，GraphPad Prism 9.0分析。两组比较用t检验或Mann-Whitney U检验；多组比较用单因素方差分析+Tukey/Dunnett法；重复测量用重复测量方差分析。Benjamini-Hochberg法FDR校正（q<0.05）。\u03B1=0.05，Power=0.80，n=12/组。P<0.05为差异有统计学意义。"]),

        // ============ 六、技术路线 ============
        heading1("六、技术路线"),

        // Technical route table
        (() => {
          const border = { style: BorderStyle.SINGLE, size: 1, color: "000000" };
          const borders = { top: border, bottom: border, left: border, right: border };
          const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };
          const col1W = 2800, col2W = 6226;
          const tableWidth = col1W + col2W;

          function cell(text, opts = {}) {
            return new TableCell({
              borders,
              width: { size: opts.w || col2W, type: WidthType.DXA },
              margins: cellMargins,
              shading: opts.shading ? { fill: opts.shading, type: ShadingType.CLEAR } : undefined,
              children: [new Paragraph({
                children: [new TextRun({ text, font: "SimSun", size: 20, bold: opts.bold })],
                spacing: { line: 280 }
              })]
            });
          }

          const rows = [
            ["第一部分（第1年）", "CIRI半暗带铁依赖性SIPS存在性、时空特征及4-HNE机制选择性研究"],
            ["\u2193 时间：6h/24h/3d/7d/28d \u2193 空间：核心/半暗带/对侧 \u2193 细胞：神经元/星形胶质细胞/小胶质细胞"],
            ["关键技术：IF共定位 | SA-\u03B2-gal | TEM | WB | IP/IB | ChIP-qPCR"],
            ["\u2193 机制选择性：Keap1修饰vs p53修饰时序 + p53位点突变 + SLC7A11转录抑制验证"],
            ["第二部分（第2年）", "BCP通过Nrf2通路阻断铁依赖性SIPS的细胞分子机制"],
            ["\u2193 Erastin/OGD/R模型 | 铁死亡指标+逆转 | Nrf2过表达/沉默+ML385 | 分子对接 | CETSA | BCP\u2192Nrf2\u2192GPX4\u2191\u21924-HNE\u2193\u2192阻断p53\u2192SLC7A11\u2191"],
            ["第三部分（第3年）", "BCP改善CIRI远期预后的整体药效 + Nrf2依赖双重验证"],
            ["\u2193 MCAO/R模型 | TTC | 脑水肿 | mNSS | 转棒 | 足误（28d） | ML385 + AAV-shNrf2 | 功能 + SIPS标志物验证"],
            ["第四部分（第3年）", "桂艾挥发油与BCP单体药效比较"],
            ["\u2193 桂艾挥发油制备+GC-MS \u2193 药效比较 \u2193 Nrf2激活谱"],
            ["\u2193 结论：BCP激活Nrf2抑制铁依赖性SIPS，改善CIRI远期预后"],
          ];

          return new Table({
            width: { size: tableWidth, type: WidthType.DXA },
            columnWidths: [col1W, col2W],
            rows: rows.map((row, i) => {
              if (row.length === 1) {
                // Full-width cell
                return new TableRow({
                  children: [new TableCell({
                    borders,
                    width: { size: tableWidth, type: WidthType.DXA },
                    margins: cellMargins,
                    gridSpan: 2,
                    children: [new Paragraph({
                      children: [new TextRun({ text: row[0], font: "SimSun", size: 20 })],
                      spacing: { line: 280 }
                    })]
                  })]
                });
              }
              const isHeader = i % 2 === 0;
              return new TableRow({
                children: [
                  new TableCell({
                    borders,
                    width: { size: col1W, type: WidthType.DXA },
                    margins: cellMargins,
                    shading: isHeader ? { fill: "D9E2F3", type: ShadingType.CLEAR } : undefined,
                    children: [new Paragraph({
                      children: [new TextRun({ text: row[0], font: "SimSun", size: 20, bold: isHeader })],
                      spacing: { line: 280 }
                    })]
                  }),
                  new TableCell({
                    borders,
                    width: { size: col2W, type: WidthType.DXA },
                    margins: cellMargins,
                    children: [new Paragraph({
                      children: [new TextRun({ text: row[1], font: "SimSun", size: 20 })],
                      spacing: { line: 280 }
                    })]
                  })
                ]
              });
            })
          });
        })(),

        // ============ 七、可行性分析 ============
        heading1("七、可行性分析"),
        heading2("7.1 理论与技术可行性"),
        spacedText("铁死亡和细胞衰老是CIRI研究热点，二者交互作用是前沿方向。铁死亡在CIRI中已被多实验室验证[3,7,13]；细胞衰老参与缺血后脑损伤慢性化获证据支持[8]；4-HNE诱导衰老效应已证实[2]；Nrf2作为二者共同防御枢纽已确立[12]。前期计算分析：铁衰老评分效应量最大，五基因签名3个独立数据集验证，置换检验P=0.001；BCP靶点与铁死亡网络高度拓扑关联（P=2.48\u00D710\u207B\u2074\u00B3）。SCISSOR[25]和CellChat[26]为CIRI表型关联细胞亚群和通讯网络提供系统证据。实验材料成熟：桂艾来源明确，BCP及AAV均为商品化试剂；MCAO/R、原代细胞培养、WB、IF等均为常规技术。"),

        heading2("7.2 前期工作基础"),
        spacedText("本团队在脑缺血、铁死亡和天然药物神经保护领域积累扎实基础。计算生物学：完成4个脑缺血数据集整合分析，构建CIRI单细胞图谱（7414个单核，25种细胞类型），建立CIRI-铁衰老五基因签名，完成GNN药物筛选，通过SCISSOR和CellChat解析CIRI表型关联细胞亚群和通讯网络。实验层面：已建立小鼠MCAO/R模型和原代神经细胞培养体系。Hu等[3]证实BCP通过Nrf2/HO-1通路减轻大鼠CIRI，为本项目延伸提供直接支持。与广西中医药大学、广西药用植物园等单位建立了良好合作。"),

        // ============ 八、特色与创新之处 ============
        heading1("八、特色与创新之处"),
        heading2("8.1 理论创新：提出CIRI中铁依赖性SIPS新假说"),
        spacedText("现有研究多将铁死亡和细胞衰老视为CIRI中两个独立过程，对二者因果联系缺乏系统探讨。铁死亡驱动衰老的概念主要在肿瘤和自然衰老领域提出，在缺血性脑损伤中研究几近空白。特别是4-HNE作用的机制选择性——为何在某些情况下激活Nrf2防御、另一些情况下驱动p53衰老——尚未解答。本项目率先提出CIRI半暗带中铁依赖性SIPS假说，并进一步提出4-HNE作用存在时间依赖性机制转换的新视角，为理解缺血性脑损伤慢性化提供新视角。"),

        heading2("8.2 机制创新：揭示BCP通过Nrf2双阻断铁死亡-衰老的新机制"),
        spacedText("BCP的神经保护作用已有报道，但多停留在药效层面，对细胞分子机制解析不够深入；BCP对铁死亡和衰老的研究多为独立报道，尚未将二者联系起来。本项目系统解析BCP通过激活Nrf2协同抑制铁死亡与铁依赖性SIPS的分子机制，明确BCP\u2192Nrf2\u2192GPX4\u2191\u21924-HNE\u2193\u2192阻断p53衰老转向\u2192SLC7A11\u2191信号轴，验证BCP对4-HNE-p53-SLC7A11正反馈环路的调控。首次阐明壮瑶药桂艾活性成分BCP通龙路火路、除毒邪功效的铁衰老干预科学内涵。"),

        heading2("8.3 模式创新：构建壮瑶药-铁衰老的整合研究新模式"),
        spacedText("民族药现代化研究常停留在成分鉴定和活性筛选层面，与前沿生物学问题结合不够紧密。本项目构建基于壮瑶医药理论的道地药材-功效-核心成分-铁衰老靶点-信号通路精准整合研究模式，将壮瑶医\u201C通龙路火路、除毒邪\u201D传统功效与铁死亡-衰老交互的前沿生物学结合。前期通过单细胞转录组学、GNN药物筛选、SCISSOR表型关联及CellChat细胞通讯等系统生物学手段构建了从分子到组织多层次证据链。明确BCP单体与桂艾挥发油分别代表\u201C精准机制\u201D与\u201C整体观\u201D两个研究层面，为民族药现代化研究提供可复制范例。"),

        // ============ 九、年度研究计划及预期研究成果 ============
        heading1("九、年度研究计划及预期研究成果"),
        heading2("9.1 第一年（2027年）"),
        p(["• 建立小鼠MCAO/R模型，完成铁依赖性SIPS时空定位（5个时间点、3个脑区、3种细胞类型）。"]),
        p(["• 完成4-HNE机制选择性时序验证（Keap1修饰vs p53修饰动力学）及p53关键修饰位点鉴定。"]),
        p(["• 完成桂艾挥发油提取与GC-MS成分分析。"]),
        p(["• 预期：发表SCI论文1篇，申请发明专利1项，培养硕士1名。"]),

        heading2("9.2 第二年（2028年）"),
        p(["• 建立体外铁依赖性SIPS细胞模型，完成BCP药效评价及Nrf2依赖性细胞水平验证。"]),
        p(["• 完成BCP对4-HNE-p53-SLC7A11分子轴调控的系统检测。"]),
        p(["• 预期：发表SCI论文1-2篇，培养硕士1名。"]),

        heading2("9.3 第三年（2029年）"),
        p(["• 完成BCP整体药效评价，Nrf2依赖性双重验证（ML385 + AAV-shNrf2）。"]),
        p(["• 完成桂艾挥发油与BCP药效比较，数据整理与论文撰写。"]),
        p(["• 预期：发表高影响力SCI论文1篇，培养研究生2-3名，申请发明专利1项。"]),
      ]
    }]
  });

  const buffer = await Packer.toBuffer(doc);
  const outPath = "D:/铁衰老 绝不重蹈覆辙/标书_终版_v13_含图表_fixed.docx";
  fs.writeFileSync(outPath, buffer);
  console.log("Done: " + outPath);
}

main().catch(err => { console.error(err); process.exit(1); });