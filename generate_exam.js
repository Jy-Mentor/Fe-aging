const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
  WidthType, ShadingType, PageNumber, PageBreak, TableOfContents
} = require("docx");

// ====== 数据定义 ======

// 30道简答题
const shortAnswerQuestions = [
  // --- 心血管内科 (4题) ---
  {
    id: 1, knowledge: "急性心肌梗死心电图演变", subject: "心血管内科",
    stem: "简述急性ST段抬高型心肌梗死（STEMI）的心电图动态演变过程。",
    answer: "超急性期T波高尖→ST段弓背向上抬高→异常Q波形成→T波倒置→ST段回落。数小时至数天内完成演变，T波改变可持续数月。",
    rubric: "超急性期T波改变（1分），ST段抬高特征（1分），Q波形成（1分），T波倒置（1分），时间顺序（1分），共5分。"
  },
  {
    id: 2, knowledge: "心力衰竭的NYHA分级", subject: "心血管内科",
    stem: "简述纽约心脏协会（NYHA）心功能分级的四级标准。",
    answer: "I级：日常活动不受限；II级：日常活动轻度受限；III级：低于日常活动即出现症状；IV级：休息时也有症状，任何活动均加重。",
    rubric: "每级1分（共4分），描述准确、无歧义（1分），共5分。"
  },
  {
    id: 3, knowledge: "高血压急症处理原则", subject: "心血管内科",
    stem: "简述高血压急症与高血压亚急症的区别及处理原则。",
    answer: "急症伴靶器官损害（脑、心、肾），需静脉降压1小时内降25%；亚急症无靶器官损害，口服降压24-48小时内达标。",
    rubric: "急症定义及靶器官损害（2分），亚急症定义（1分），处理原则区分（2分），共5分。"
  },
  {
    id: 4, knowledge: "心房颤动的抗凝治疗指征", subject: "心血管内科",
    stem: "简述非瓣膜性心房颤动患者启动抗凝治疗的CHA2DS2-VASc评分标准。",
    answer: "充血性心衰1分，高血压1分，年龄≥75岁2分，糖尿病1分，卒中/TIA/血栓栓塞2分，血管疾病1分，年龄65-74岁1分，女性1分。≥2分（男）/≥3分（女）启动抗凝。",
    rubric: "7个危险因素列举正确（3分），评分阈值说明（1分），性别差异说明（1分），共5分。"
  },

  // --- 呼吸内科 (3题) ---
  {
    id: 5, knowledge: "慢性阻塞性肺疾病诊断标准", subject: "呼吸内科",
    stem: "简述慢性阻塞性肺疾病（COPD）的诊断标准及GOLD分级依据。",
    answer: "吸入支气管舒张剂后FEV1/FVC<0.7确诊。GOLD 1级：FEV1≥80%预计值；2级：50%-80%；3级：30%-50%；4级：<30%。",
    rubric: "肺功能诊断标准（2分），GOLD四级数值（2分），前提条件（吸入支气管舒张剂后）（1分），共5分。"
  },
  {
    id: 6, knowledge: "社区获得性肺炎严重度评估", subject: "呼吸内科",
    stem: "简述CURB-65评分在社区获得性肺炎（CAP）中的应用。",
    answer: "意识障碍（Confusion），尿素氮>7mmol/L，呼吸频率≥30次/分，低血压（SBP<90或DBP≤60），年龄≥65岁。0-1分门诊，2分住院，≥3分重症。",
    rubric: "五项指标各1分（共5分），评分分层与处置建议（1分），共6分。"
  },
  {
    id: 7, knowledge: "支气管哮喘急性发作严重度分级", subject: "呼吸内科",
    stem: "简述支气管哮喘急性发作的严重度分级及轻、重度主要判别指标。",
    answer: "分轻、中、重、危重四级。轻度：步行气短，能平卧，PEF>80%；重度：静息气短，端坐呼吸，PEF≤50%，PaO2<60mmHg，PaCO2>45mmHg。",
    rubric: "四级分类（1分），轻度关键指标（2分），重度关键指标（2分），共5分。"
  },

  // --- 消化内科 (3题) ---
  {
    id: 8, knowledge: "上消化道出血病因鉴别", subject: "消化内科",
    stem: "简述上消化道出血最常见的五大病因及各自特征性表现。",
    answer: "消化性溃疡（节律性上腹痛），食管胃底静脉曲张破裂（肝硬化体征、呕血量大），急性胃黏膜病变（应激/NSAIDs史），胃癌（消瘦、黑便持续），Mallory-Weiss综合征（剧烈呕吐后）。",
    rubric: "五个病因各1分（共5分），特征性表现描述准确（1分），共6分。"
  },
  {
    id: 9, knowledge: "急性胰腺炎诊断标准", subject: "消化内科",
    stem: "简述急性胰腺炎的诊断标准及Ranson评分的主要项目。",
    answer: "诊断需符合以下三项中两项：①典型腹痛，②血淀粉酶/脂肪酶>正常上限3倍，③影像学符合。Ranson评分入院时5项（年龄、WBC、血糖、LDH、AST），48h后6项。",
    rubric: "诊断三项标准（2分），Ranson入院时五项（2分），48h后概念（1分），共5分。"
  },
  {
    id: 10, knowledge: "肝硬化Child-Pugh分级", subject: "消化内科",
    stem: "简述肝硬化Child-Pugh分级的五项指标及其临床意义。",
    answer: "胆红素、白蛋白、凝血酶原时间延长、腹水、肝性脑病。A级5-6分（代偿期），B级7-9分（中度），C级10-15分（失代偿期），用于评估肝功能储备和手术风险。",
    rubric: "五项指标各1分（共5分），分级与临床意义（1分），共6分。"
  },

  // --- 内分泌科 (3题) ---
  {
    id: 11, knowledge: "糖尿病诊断标准", subject: "内分泌科",
    stem: "简述糖尿病的诊断标准（WHO 1999/2020）。",
    answer: "①典型症状+随机血糖≥11.1mmol/L；②空腹血糖≥7.0mmol/L；③OGTT 2h血糖≥11.1mmol/L；④HbA1c≥6.5%。需重复确认，无症状者需两次异常。",
    rubric: "四条标准各1分（共4分），重复确认原则（1分），共5分。"
  },
  {
    id: 12, knowledge: "糖尿病酮症酸中毒病理生理", subject: "内分泌科",
    stem: "简述糖尿病酮症酸中毒（DKA）的病理生理机制。",
    answer: "胰岛素绝对/相对缺乏→葡萄糖利用障碍→脂肪分解加速→酮体（β-羟丁酸、乙酰乙酸、丙酮）大量生成→代谢性酸中毒→渗透性利尿→脱水及电解质紊乱。",
    rubric: "胰岛素缺乏起点（1分），脂肪分解与酮体生成（2分），酸中毒机制（1分），脱水电解质紊乱（1分），共5分。"
  },
  {
    id: 13, knowledge: "甲状腺功能亢进症诊断", subject: "内分泌科",
    stem: "简述Graves病的主要临床表现和实验室诊断依据。",
    answer: "高代谢症候群、弥漫性甲状腺肿、突眼、胫前黏液性水肿。实验室：TSH降低，FT3/FT4升高，TRAb/TSAb阳性。甲状腺摄碘率增高且高峰前移。",
    rubric: "临床表现（2分），实验室检查（2分），摄碘率特征（1分），共5分。"
  },

  // --- 肾脏内科 (2题) ---
  {
    id: 14, knowledge: "肾病综合征诊断标准", subject: "肾脏内科",
    stem: "简述肾病综合征的诊断标准及常见并发症。",
    answer: "①大量蛋白尿>3.5g/24h；②低白蛋白血症<30g/L；③水肿；④高脂血症。①②为必备。并发症：感染、血栓栓塞、急性肾损伤、蛋白质营养不良。",
    rubric: "四项诊断标准各0.5分（2分），必备条件说明（1分），四种并发症各0.5分（2分），共5分。"
  },
  {
    id: 15, knowledge: "急性肾损伤分期", subject: "肾脏内科",
    stem: "简述急性肾损伤（AKI）的KDIGO分期标准（基于血清肌酐）。",
    answer: "1期：肌酐升高≥26.5μmol/L或1.5-1.9倍基线；2期：2.0-2.9倍基线；3期：≥3.0倍基线或≥353.6μmol/L或开始肾脏替代治疗。",
    rubric: "1期标准（2分），2期标准（1分），3期标准（2分），共5分。"
  },

  // --- 血液内科 (2题) ---
  {
    id: 16, knowledge: "缺铁性贫血诊断", subject: "血液内科",
    stem: "简述缺铁性贫血的实验室诊断指标及铁代谢各阶段变化。",
    answer: "小细胞低色素性贫血。储存铁耗尽期：铁蛋白↓；缺铁性红细胞生成期：血清铁↓、TIBC↑、转铁蛋白饱和度↓；IDA期：Hb↓、MCV↓、MCH↓。",
    rubric: "贫血形态学特征（1分），三期各阶段指标变化（3分），诊断顺序（1分），共5分。"
  },
  {
    id: 17, knowledge: "急性白血病分型", subject: "血液内科",
    stem: "简述急性白血病的FAB分型及急性早幼粒细胞白血病（M3）的临床特点。",
    answer: "ALL分L1-L3，AML分M0-M7。M3特征：异常早幼粒细胞增生，t(15;17)形成PML-RARα融合基因，DIC发生率高，全反式维甲酸和三氧化二砷治疗有效。",
    rubric: "FAB分型简述（1分），M3细胞形态（1分），遗传学特征（1分），DIC风险（1分），靶向治疗（1分），共5分。"
  },

  // --- 神经内科 (3题) ---
  {
    id: 18, knowledge: "急性缺血性脑卒中治疗", subject: "神经内科",
    stem: "简述急性缺血性脑卒中静脉溶栓的适应证和禁忌证。",
    answer: "适应证：发病≤4.5h，年龄≥18岁，有神经功能缺损。禁忌证：颅内出血史、近3月卒中/头外伤、可疑蛛网膜下腔出血、活动性出血、血小板<10万、INR>1.7。",
    rubric: "时间窗（1分），适应证（2分），主要禁忌证（2分，至少列出4项），共5分。"
  },
  {
    id: 19, knowledge: "癫痫发作分类", subject: "神经内科",
    stem: "简述ILAE 2017癫痫发作分类中局灶性发作与全面性发作的主要区别。",
    answer: "局灶性发作起源于一侧半球网络，可分知觉保留/损害，可伴运动/非运动症状，可继发全面化。全面性发作双侧半球同时起源，包括失神、肌阵挛、强直-阵挛等。",
    rubric: "起源部位区别（2分），局灶性特征（1.5分），全面性特征（1.5分），共5分。"
  },
  {
    id: 20, knowledge: "帕金森病核心症状", subject: "神经内科",
    stem: "简述帕金森病的四大核心运动症状及其病理基础。",
    answer: "静止性震颤、肌强直（铅管样/齿轮样）、运动迟缓（核心）、姿势步态异常。病理基础：黑质多巴胺能神经元变性丢失，路易小体形成，纹状体多巴胺显著减少。",
    rubric: "四大症状各1分（共4分），病理基础（1分），共5分。"
  },

  // --- 感染科 (2题) ---
  {
    id: 21, knowledge: "病毒性肝炎血清学标志", subject: "感染科",
    stem: "简述乙肝五项（两对半）各项指标及其临床意义。",
    answer: "HBsAg：现症感染；抗-HBs：保护性抗体；HBeAg：病毒复制活跃；抗-HBe：复制减弱；抗-HBc：IgM近期感染，IgG既往感染。大三阳：HBsAg+HBeAg+抗-HBc阳性。",
    rubric: "五项指标各1分（共5分），大三阳定义（1分），共6分。"
  },
  {
    id: 22, knowledge: "结核病诊断方法", subject: "感染科",
    stem: "简述肺结核的病原学诊断方法及各自的优缺点。",
    answer: "痰涂片抗酸染色（快速但敏感性低），分枝杆菌培养（金标准但需4-8周），分子检测（GeneXpert快速且可检测利福平耐药，推荐首选），T-SPOT（辅助诊断，不能区分活动与潜伏）。",
    rubric: "四种方法各1分（共4分），优缺点分析（1分），共5分。"
  },

  // --- 普外科 (3题) ---
  {
    id: 23, knowledge: "急腹症鉴别诊断", subject: "普外科",
    stem: "简述急性阑尾炎的典型临床表现及Alvarado评分主要项目。",
    answer: "转移性右下腹痛、麦氏点压痛反跳痛、恶心呕吐、低热、白细胞升高。Alvarado评分：症状3项（转移痛、厌食、恶心呕吐），体征3项（压痛、反跳痛、体温），实验室2项（白细胞、核左移）。",
    rubric: "典型临床表现（2分），Alvarado评分项目（2分），评分意义（1分），共5分。"
  },
  {
    id: 24, knowledge: "肠梗阻分类与诊断", subject: "普外科",
    stem: "简述肠梗阻的病因分类及机械性肠梗阻的典型X线表现。",
    answer: "分类：机械性、动力性（麻痹性/痉挛性）、血运性。X线：阶梯状气液平面，结肠无气（完全性），小肠扩张>3cm，结肠扩张>6cm。绞窄性可见孤立胀大肠袢。",
    rubric: "三种分类（1.5分），X线特征（2.5分），绞窄性特征（1分），共5分。"
  },
  {
    id: 25, knowledge: "胆囊结石并发症", subject: "普外科",
    stem: "简述胆囊结石的常见并发症及其临床特点。",
    answer: "急性胆囊炎（右上腹痛、Murphy征阳性），胆总管结石（黄疸、胆管炎），急性胰腺炎，胆囊十二指肠瘘（肠梗阻），胆囊癌（长期结石史恶变）。",
    rubric: "五种并发症各1分（共5分），临床特点描述准确（1分），共6分。"
  },

  // --- 骨科 (2题) ---
  {
    id: 26, knowledge: "骨折愈合过程", subject: "骨科",
    stem: "简述骨折愈合的四个阶段及其主要病理过程。",
    answer: "血肿炎症机化期（血肿形成→炎症反应→肉芽组织），软骨痂形成期（纤维软骨痂），骨性骨痂形成期（膜内成骨+软骨内成骨），骨痂改造塑形期（Wolff定律）。",
    rubric: "四个阶段各1分（共4分），各阶段病理过程（1分），共5分。"
  },
  {
    id: 27, knowledge: "骨筋膜室综合征", subject: "骨科",
    stem: "简述骨筋膜室综合征的5P征及其紧急处理原则。",
    answer: "Pain（疼痛，被动牵拉加剧）、Pallor（苍白）、Paresthesia（感觉异常）、Paralysis（麻痹）、Pulselessness（无脉，晚期）。处理：立即切开减压，不可抬高患肢。",
    rubric: "5P征各1分（共5分），紧急处理原则（1分），共6分。"
  },

  // --- 妇产科 (2题) ---
  {
    id: 28, knowledge: "子痫前期诊断标准", subject: "妇产科",
    stem: "简述子痫前期的诊断标准及重度子痫前期的特征。",
    answer: "妊娠20周后新发高血压（≥140/90mmHg）伴蛋白尿（≥0.3g/24h）或靶器官损害。重度：BP≥160/110mmHg，蛋白尿≥5g/24h，血小板<10万，肝酶升高，肺水肿，视觉障碍。",
    rubric: "诊断标准（2分），重度特征（2分），时间节点（1分），共5分。"
  },
  {
    id: 29, knowledge: "产后出血处理", subject: "妇产科",
    stem: "简述产后出血的定义、四大病因（4T）及初步处理措施。",
    answer: "胎儿娩出24h内出血≥500ml（剖宫产≥1000ml）。4T：Tone（宫缩乏力）、Trauma（产道损伤）、Tissue（胎盘残留）、Thrombin（凝血障碍）。处理：按摩子宫、宫缩剂、宫腔填塞。",
    rubric: "定义及出血量（1分），4T各0.5分（2分），初步处理（2分），共5分。"
  },

  // --- 儿科 (1题) ---
  {
    id: 30, knowledge: "小儿腹泻病液体疗法", subject: "儿科",
    stem: "简述小儿腹泻病脱水程度评估及口服补液盐（ORS）的使用原则。",
    answer: "轻度（失水<5%）：眼窝稍凹，皮肤弹性可；中度（5-10%）：眼窝凹陷，皮肤弹性差；重度（>10%）：休克。ORS：无脱水预防，轻中度首选ORS口服，重度静脉补液。",
    rubric: "三度脱水评估（3分），ORS使用原则（2分），共5分。"
  }
];

// 7道案例分析题
const caseQuestions = [
  {
    id: 1, subject: "心血管内科", title: "急性心肌梗死",
    scenario: "患者，男性，62岁，因"持续性胸骨后压榨样疼痛2小时"急诊入院。患者2小时前无明显诱因突然出现胸骨后压榨样疼痛，向左肩放射，伴大汗、恶心。既往有高血压病史10年，长期吸烟史40年（1包/天）。入院查体：BP 150/95mmHg，HR 96次/分，双肺底可闻及少量湿啰音。心电图示：V1-V4导联ST段弓背向上抬高0.3-0.5mV。",
    questions: [
      "根据上述临床表现，该患者最可能的诊断是什么？列出诊断依据。",
      "该患者应立即进行哪些实验室检查？预期结果如何？",
      "简述该患者的急诊处理原则（包括再灌注治疗策略）。",
      "患者住院期间应如何管理并发症风险？"
    ],
    answers: [
      "最可能诊断为急性广泛前壁ST段抬高型心肌梗死（STEMI），Killip II级。诊断依据：①典型缺血性胸痛>20分钟；②心电图V1-V4导联ST段抬高>0.2mV；③心血管危险因素（高血压、吸烟、年龄）；④双肺底湿啰音提示轻度心衰。",
      "应急查：①心肌肌钙蛋白I/T（cTnI/cTnT）——STEMI早期可正常，3-6h后升高，12-24h达峰；②CK-MB——4-6h升高，24h达峰；③电解质、肾功能、血糖、凝血功能；④血常规、血脂。预期cTnI和CK-MB将显著升高，动态监测呈上升趋势。",
      "急诊处理：①立即心电监护、吸氧（SpO2<90%时）、建立静脉通路；②双联抗血小板（阿司匹林300mg+替格瑞洛180mg嚼服）；③抗凝（肝素）；④再灌注治疗：发病<12h，首选急诊PCI（门-球时间<90min）；若无法120min内行PCI，且无禁忌证，予静脉溶栓；⑤硝酸酯类（SBP>90mmHg）、β受体阻滞剂（无禁忌证时）。",
      "并发症管理：①心律失常：持续心电监护，备除颤仪，室颤立即电除颤；②心力衰竭：控制液体入量，利尿剂，必要时血管活性药物；③心源性休克：维持血压，评估机械循环支持；④机械并发症（室间隔穿孔、乳头肌断裂）：超声心动图监测，外科干预；⑤再梗死/支架血栓：双抗依从性。危险分层与二级预防教育贯穿住院全程。"
    ],
    rubric: "诊断及依据（5分）：诊断正确2分，依据充分3分。实验室检查（3分）。急诊处理（6分）：一般处理1分，抗血小板1分，抗凝1分，再灌注策略2分，其他药物1分。并发症管理（6分）：每项1.5分。共20分。"
  },
  {
    id: 2, subject: "呼吸内科", title: "慢性阻塞性肺疾病急性加重",
    scenario: "患者，男性，72岁，因"反复咳嗽咳痰20年，气促加重3天"入院。患者有长期吸烟史（50包年），既往诊断为COPD（GOLD 3级），平时规律使用噻托溴铵吸入。3天前受凉后出现咳嗽咳痰增多，痰液变黄脓痰，伴发热（Tmax 38.5℃），静息时即感气促，不能平卧。查体：T 38.3℃，R 28次/分，BP 138/86mmHg，桶状胸，双肺可闻及散在哮鸣音及湿啰音。SpO2 85%（吸空气）。",
    questions: [
      "该患者AECOPD的诊断依据及严重度评估是什么？",
      "该患者应进行哪些辅助检查？预期结果如何？",
      "简述该患者的综合治疗方案。",
      "患者出院时应如何制定长期管理方案？"
    ],
    answers: [
      "诊断依据：①COPD病史（GOLD 3级）；②症状恶化（咳嗽咳痰增多、脓痰、气促加重）；③体征（发热、桶状胸、啰音）。严重度：静息气促、SpO2 85%，属重度加重，需住院治疗。判断是否存在呼吸衰竭需动脉血气分析。",
      "辅助检查：①动脉血气分析——评估有无II型呼衰（PaO2<60mmHg，PaCO2>50mmHg）；②血常规、CRP——WBC↑、中性粒细胞↑、CRP↑提示细菌感染；③痰培养+药敏——指导抗生素选择，常见流感嗜血杆菌、肺炎链球菌、卡他莫拉菌；④胸部X线——排除肺炎、气胸；⑤心电图——排除心脏合并症。",
      "综合治疗：①控制性氧疗：目标SpO2 88%-92%，避免高浓度氧导致CO2潴留；②支气管舒张剂：短效β2激动剂+短效抗胆碱能药雾化吸入，每4-6h；③全身糖皮质激素：泼尼松40mg/d×5天；④抗生素：脓痰+发热+CRP升高，覆盖社区获得性病原菌；⑤无创正压通气（NIPPV）：若出现呼吸性酸中毒（pH<7.35，PaCO2>45mmHg）。",
      "出院管理：①戒烟（核心措施）；②长期吸入治疗：LABA+LAMA+ICS三联（根据GOLD分组）；③肺康复计划：包括运动训练、营养指导、自我管理教育；④疫苗接种：流感疫苗、肺炎球菌疫苗；⑤定期随访：每3-6个月评估肺功能、症状、急性加重频率；⑥家庭氧疗评估：若静息PaO2≤55mmHg。"
    ],
    rubric: "诊断及严重度（4分）。辅助检查（4分）：每项1分。治疗方案（7分）：氧疗1分，支气管舒张剂1分，激素1分，抗生素1分，NIPPV指征2分，其他1分。出院管理（5分）：每项约1分。共20分。"
  },
  {
    id: 3, subject: "消化内科", title: "上消化道出血",
    scenario: "患者，男性，48岁，因"呕血、黑便6小时"急诊入院。患者6小时前无明显诱因突然呕血约500ml，为暗红色血液，混有血块，随后解柏油样黑便2次，伴头晕、乏力、心慌。既往有"十二指肠球部溃疡"病史5年，间断服用NSAIDs（因关节炎）。查体：P 110次/分，BP 90/60mmHg，面色苍白，皮肤湿冷，上腹部轻压痛。",
    questions: [
      "该患者最可能的诊断及出血严重程度评估。",
      "该患者急诊处理的首要步骤是什么？",
      "简述该患者的药物治疗方案。",
      "患者何时应行急诊内镜检查？内镜下Forrest分级的意义是什么？",
      "患者出院后应如何预防再出血？"
    ],
    answers: [
      "最可能诊断为十二指肠球部溃疡并上消化道出血（NSAIDs相关）。严重度评估：心率110次/分、血压90/60mmHg（休克指数1.22），面色苍白、皮肤湿冷，提示中度失血性休克（估计失血量800-1500ml），属高危出血，需紧急处理。",
      "急诊处理：①气道保护（防误吸），吸氧，建立双静脉通路；②快速补液：晶体液（林格液/生理盐水）扩容，目标SBP>100mmHg、HR<100次/分；③交叉配血，Hb<70g/L时输血；④禁食、留置胃管（观察活动性出血，可冰盐水洗胃）；⑤监测生命体征、尿量、中心静脉压。",
      "药物治疗：①PPI：奥美拉唑80mg静推后8mg/h持续泵入（大剂量PPI），胃内pH>6可稳定血痂；②生长抑素/奥曲肽：减少内脏血流（尤其怀疑静脉曲张时）；③止血药：氨甲环酸可辅助；④根除Hp（若阳性）：四联疗法。NSAIDs停药。",
      "急诊内镜时机：血流动力学稳定后，24h内行急诊胃镜（高危征象应12h内）。Forrest分级：Ia（动脉喷射）、Ib（渗血）——需内镜治疗，再出血率>55%；IIa（裸露血管）、IIb（血凝块）——内镜治疗，再出血率20-40%；IIc（黑色基底）、III（洁净基底）——无需内镜治疗，再出血率<5%。",
      "预防再出血：①根除Hp并确认根除成功；②避免NSAIDs，必要时改用COX-2抑制剂+PPI；③PPI维持治疗（高危者长期）；④生活方式：戒酒、规律饮食；⑤定期随访，出现黑便、呕血立即就医。"
    ],
    rubric: "诊断及严重度（4分）。急诊处理（4分）：每项0.8分。药物治疗（4分）。内镜指征及Forrest分级（5分）。预防再出血（3分）。共20分。"
  },
  {
    id: 4, subject: "内分泌科", title: "糖尿病酮症酸中毒",
    scenario: "患者，女性，22岁，因"恶心呕吐、腹痛2天，意识模糊2小时"急诊入院。患者1周前无明显诱因出现多饮、多尿、多食，体重下降约5kg，2天前出现恶心呕吐、上腹部疼痛，今晨家属发现患者呼之不应。既往体健，无糖尿病史。查体：T 36.5℃，P 120次/分，BP 88/56mmHg，R 28次/分（深大呼吸），皮肤干燥弹性差，呼气可闻及烂苹果味，意识模糊。",
    questions: [
      "该患者最可能的诊断是什么？列出诊断依据。",
      "该患者应立即进行哪些实验室检查？预期结果如何？",
      "简述该患者的补液及胰岛素治疗原则。",
      "治疗过程中需密切监测哪些指标？可能发生哪些并发症？"
    ],
    answers: [
      "最可能诊断为1型糖尿病并糖尿病酮症酸中毒（DKA），重度。诊断依据：①三多一少典型症状；②消化系统症状（恶心呕吐、腹痛）；③意识障碍；④脱水体征及休克；⑤Kussmaul深大呼吸；⑥呼气烂苹果味（丙酮）；⑦年轻患者，急性起病，提示1型糖尿病。",
      "应急查：①血糖——显著升高（通常16.7-33.3mmol/L）；②血酮/尿酮——强阳性；③血气分析——代谢性酸中毒（pH<7.3，HCO3-<15mmol/L），阴离子间隙增高；④电解质——钾可正常或偏高（酸中毒时细胞内钾外移），钠可降低（假性低钠）；⑤尿素氮/肌酐——肾前性升高；⑥糖化血红蛋白——反映近期血糖水平，非诊断必需。",
      "补液：①第1h：生理盐水15-20ml/kg/h（约1-1.5L）；②随后4h：250-500ml/h，根据脱水程度调整；③血糖降至13.9mmol/L时改5%葡萄糖+胰岛素。24h总量约4-6L。胰岛素：①持续静脉泵入短效胰岛素0.1U/kg/h；②血糖下降速度3.9-6.1mmol/L/h；③不可骤停胰岛素，酮体清除后方可过渡皮下注射。补钾：血钾<5.5mmol/L且尿量正常时开始补钾。",
      "监测指标：①血糖每小时一次；②血酮/尿酮每2-4h一次；③血气分析每2-4h一次；④电解质（尤血钾）每2-4h一次；⑤生命体征、意识状态、出入量每小时记录。并发症：①低血糖（胰岛素过量）；②低钾血症（纠酸后钾细胞内移）；③脑水肿（儿童多见，血糖下降过快、补液不当）；④急性肾损伤；⑤ARDS；⑥血栓栓塞。"
    ],
    rubric: "诊断及依据（5分）：诊断2分，依据3分。实验室检查（4分）。补液及胰岛素（5分）：补液2分，胰岛素2分，补钾1分。监测及并发症（6分）：监测3分，并发症3分。共20分。"
  },
  {
    id: 5, subject: "神经内科", title: "急性缺血性脑卒中",
    scenario: "患者，男性，68岁，因"突发右侧肢体无力伴言语不清3小时"急诊入院。患者今晨8:00起床时无明显诱因突然出现右侧肢体无力，不能站立，伴口角歪斜、言语含糊不清，无头痛、呕吐、意识障碍。既往有高血压病史15年，心房颤动史5年（未规律抗凝），糖尿病史8年。查体：BP 170/95mmHg，P 88次/分（心律绝对不齐），嗜睡，运动性失语，右侧中枢性面舌瘫，右侧上下肢肌力II级，右侧巴氏征阳性。NIHSS评分16分。",
    questions: [
      "该患者最可能的诊断及分型是什么？",
      "该患者是否适合静脉溶栓治疗？请说明理由。",
      "若患者不适合溶栓，应如何选择抗栓治疗？",
      "简述该患者急性期的综合管理方案。",
      "患者出院后二级预防的核心措施有哪些？"
    ],
    answers: [
      "最可能诊断为急性缺血性脑卒中（心源性栓塞型），左侧大脑中动脉供血区。分型（TOAST）为心源性栓塞。诊断依据：①突发局灶性神经功能缺损；②有心房颤动（栓塞源）；③右侧偏瘫、失语符合左侧MCA综合征；④NIHSS 16分为中重度卒中。",
      "该患者发病3小时，在4.5h时间窗内，但需评估溶栓条件：①排除脑出血（需急诊头颅CT）；②无近期手术、活动性出血、出血倾向；③BP 170/95mmHg，需降压至<185/110mmHg方可溶栓；④NIHSS 16分，明确神经功能缺损。若CT排除出血且无其他禁忌，BP控制后可行静脉溶栓（阿替普酶0.9mg/kg）。但心源性栓塞溶栓效果有限，应评估血管内治疗（机械取栓）适应证。",
      "若不适合/不选择溶栓：①血管内治疗：发病6h内，NIHSS≥6分，大血管闭塞（CTA/DSA证实），可直接机械取栓；②抗血小板：阿司匹林150-300mg/d（溶栓后24h方可启用）；③抗凝：心源性栓塞患者，急性期一般不立即抗凝（大面积梗死出血转化风险高），待病情稳定（通常1-2周后）启动华法林或DOAC。",
      "急性期管理：①监护：生命体征、神经功能（NIHSS动态评估）、心电监护；②血压管理：准备溶栓者控制<185/110mmHg，不溶栓者SBP>220或DBP>120时谨慎降压，避免过度降压；③血糖管理：控制在7.8-10mmol/L；④防治并发症：吞咽困难筛查（防误吸），深静脉血栓预防，早期康复；⑤病因检查：头颅MRI/MRA、颈动脉超声、心脏超声（确认心源性栓塞）。",
      "二级预防：①抗凝治疗：非瓣膜性房颤CHA2DS2-VASc≥2分，长期口服华法林（INR 2-3）或DOAC；②降压治疗：目标<140/90mmHg；③他汀治疗：强化降脂（LDL-C<1.8mmol/L）；④降糖治疗：HbA1c<7%；⑤生活方式干预：戒烟、限酒、合理饮食、规律运动；⑥健康教育与定期随访。"
    ],
    rubric: "诊断及分型（3分）。溶栓评估（5分）：时间窗1分，CT排除出血1分，血压控制1分，适应证1分，血管内治疗提及1分。抗栓治疗（4分）。急性期管理（4分）：每项约1分。二级预防（4分）：每项约0.8分。共20分。"
  },
  {
    id: 6, subject: "普外科", title: "急性阑尾炎",
    scenario: "患者，男性，28岁，因"转移性右下腹痛12小时"入院。患者12小时前无明显诱因出现上腹部隐痛，伴恶心，6小时前腹痛转移至右下腹，呈持续性胀痛，无放射，伴发热（T 37.8℃）。查体：腹平坦，右下腹麦克伯尼点压痛、反跳痛阳性，腰大肌试验阳性，结肠充气试验阴性。血常规：WBC 13.5×10⁹/L，中性粒细胞85%。",
    questions: [
      "该患者最可能的诊断是什么？诊断依据为何？",
      "该患者应进行哪些鉴别诊断？",
      "简述该患者的治疗原则及围手术期处理。",
      "若术中发现阑尾穿孔合并弥漫性腹膜炎，应如何处理？"
    ],
    answers: [
      "最可能诊断为急性阑尾炎（化脓性）。诊断依据：①转移性右下腹痛（典型病史）；②右下腹固定压痛、反跳痛（腹膜炎体征）；③腰大肌试验阳性（提示盲肠后位阑尾炎）；④WBC及中性粒细胞明显升高；⑤Alvarado评分≥7分（转移痛1分+厌食1分+恶心1分+压痛2分+反跳痛1分+体温1分+WBC 2分），属高危。",
      "鉴别诊断：①消化性溃疡穿孔（突发上腹痛，板状腹，膈下游离气体）；②急性肠系膜淋巴结炎（多见于儿童，腹痛范围广，随体位改变）；③右侧输尿管结石（放射至会阴，血尿）；④Meckel憩室炎（症状相似，位置偏中腹部）；⑤急性胆囊炎（右上腹痛，Murphy征阳性）；⑥克罗恩病（慢性病程，腹泻，瘘管形成）。",
      "治疗原则：确诊后尽早行阑尾切除术（腹腔镜首选）。围手术期处理：①术前：禁食、补液、抗生素（头孢类+甲硝唑）术前30min单次给药；②术中：腹腔镜探查，阑尾切除，脓液吸引冲洗；③术后：抗生素治疗（化脓性/穿孔性继续3-5天），早期下床活动，排气后进食，观察切口及腹腔感染征象。",
      "穿孔合并弥漫性腹膜炎处理：①彻底清除脓液及纤维蛋白渗出物，大量温盐水冲洗腹腔（至冲洗液清亮）；②放置腹腔引流管（盆腔及右结肠旁沟）；③术后加强抗感染（广谱抗生素覆盖需氧菌和厌氧菌）；④延长禁食时间，胃肠减压，营养支持；⑤监测有无腹腔脓肿（术后超声/CT），若形成脓肿需穿刺引流；⑥切口感染预防（延迟一期缝合或二期缝合）。"
    ],
    rubric: "诊断及依据（5分）：诊断2分，依据3分。鉴别诊断（4分）：每项约0.7分。治疗原则及围手术期（5分）：手术1分，抗生素1分，术前准备1分，术后管理2分。穿孔处理（6分）：腹腔冲洗引流2分，抗感染1分，营养支持1分，并发症监测1分，切口处理1分。共20分。"
  },
  {
    id: 7, subject: "妇产科", title: "子痫前期",
    scenario: "患者，女性，32岁，G1P0，孕34周，因"头痛、视物模糊2天，上腹部不适1天"入院。患者孕期于社区医院常规产检，孕28周发现血压升高（140/90mmHg），未规律服药。近2天出现持续性头痛，视物模糊，1天前感上腹部不适。查体：BP 170/110mmHg，双下肢水肿（++），宫高28cm，腹围98cm，胎心140次/分。辅助检查：尿蛋白（+++），24h尿蛋白定量4.8g，血小板计数95×10⁹/L，ALT 85U/L，AST 78U/L。",
    questions: [
      "该患者最可能的诊断是什么？列出诊断依据。",
      "该患者是否为重度子痫前期？请说明理由。",
      "该患者的处理原则是什么？何时应终止妊娠？",
      "子痫前期患者应警惕哪些严重并发症？",
      "简述硫酸镁使用的注意事项。"
    ],
    answers: [
      "最可能诊断为重度子痫前期。诊断依据：①孕34周，孕20周后新发；②BP 170/110mmHg（重度标准）；③尿蛋白+++，24h尿蛋白4.8g（>2g）；④血小板减少（95×10⁹/L）；⑤肝酶升高（ALT、AST均升高）；⑥头痛、视物模糊（中枢神经系统受累）；⑦上腹部不适（肝包膜下血肿/HELLP综合征前兆）。",
      "该患者符合重度子痫前期诊断：①BP≥160/110mmHg；②血小板<100×10⁹/L；③肝酶升高>正常上限2倍；④24h尿蛋白>2g；⑤头痛、视物模糊等神经系统症状。多项指标达重度标准，且存在HELLP综合征倾向（血小板减少+肝酶升高），需紧急处理。期待治疗（延缓分娩）不适用于此患者，应立即终止妊娠。",
      "处理原则：①立即住院，绝对卧床休息，监测血压每15-30min；②硫酸镁：4-6g负荷量（20min内静注），1-2g/h维持，预防子痫抽搐；③控制血压：拉贝洛尔或硝苯地平，目标140-150/90-100mmHg；④促胎肺成熟：地塞米松6mg q12h×4次（<34周）；⑤终止妊娠：患者已34周，病情较重，应在控制病情后尽快终止妊娠，首选剖宫产。产后仍需监测血压及实验室指标48-72h。",
      "严重并发症：①子痫（抽搐发作）；②HELLP综合征（溶血、肝酶升高、血小板减少三联征）；③胎盘早剥（腹痛、阴道出血、胎儿窘迫）；④急性肾衰竭（少尿/无尿，肌酐进行性升高）；⑤肺水肿（呼吸困难、低氧）；⑥颅内出血（严重高血压并发症）；⑦弥散性血管内凝血（DIC）；⑧胎儿生长受限、胎儿窘迫、胎死宫内。",
      "硫酸镁注意事项：①必备条件：膝反射存在、呼吸≥16次/分、尿量≥25ml/h（或≥100ml/4h）；②中毒监测：血镁浓度治疗窗1.7-3.5mmol/L，>3.5mmol/L中毒风险；③中毒表现：膝反射消失（最早）、呼吸抑制、心跳骤停；④解毒剂：10%葡萄糖酸钙10ml缓慢静推；⑤肾功能不全者减量，监测血镁浓度；⑥持续用药不超过5-7天。"
    ],
    rubric: "诊断及依据（4分）。重度判断（4分）：判断正确2分，理由2分。处理原则（5分）：硫酸镁1.5分，降压1分，促胎肺1分，终止妊娠1.5分。并发症（4分）：每项0.5分。硫酸镁注意事项（3分）：每项0.6分。共20分。"
  }
];

// ====== 文档构建函数 ======

function createTable(headers, rows, colWidths) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const headerShading = { fill: "2B579A", type: ShadingType.CLEAR };
  const altShading = { fill: "F2F6FC", type: ShadingType.CLEAR };

  const totalWidth = colWidths.reduce((a, b) => a + b, 0);

  const headerRow = new TableRow({
    children: headers.map((h, i) =>
      new TableCell({
        borders,
        width: { size: colWidths[i], type: WidthType.DXA },
        shading: headerShading,
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: h, bold: true, font: "Microsoft YaHei", size: 20, color: "FFFFFF" })]
        })]
      })
    )
  });

  const dataRows = rows.map((row, ri) =>
    new TableRow({
      children: row.map((cell, ci) =>
        new TableCell({
          borders,
          width: { size: colWidths[ci], type: WidthType.DXA },
          shading: ri % 2 === 0 ? altShading : undefined,
          margins: { top: 60, bottom: 60, left: 100, right: 100 },
          children: [new Paragraph({
            children: [new TextRun({ text: String(cell), font: "Microsoft YaHei", size: 18 })]
          })]
        })
      )
    })
  );

  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows]
  });
}

function sectionTitle(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 240 },
    children: [new TextRun({ text, font: "Microsoft YaHei", size: 32, bold: true, color: "2B579A" })]
  });
}

function subTitle(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 180 },
    children: [new TextRun({ text, font: "Microsoft YaHei", size: 26, bold: true, color: "2B579A" })]
  });
}

function subSubTitle(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 120 },
    children: [new TextRun({ text, font: "Microsoft YaHei", size: 22, bold: true, color: "333333" })]
  });
}

function normalPara(text, options = {}) {
  return new Paragraph({
    spacing: { after: 120, line: 360 },
    ...options,
    children: [new TextRun({ text, font: "Microsoft YaHei", size: 21, ...options.run })]
  });
}

function boldPara(text) {
  return normalPara(text, { run: { bold: true } });
}

function indentPara(text) {
  return new Paragraph({
    spacing: { after: 100, line: 340 },
    indent: { left: 360 },
    children: [new TextRun({ text, font: "Microsoft YaHei", size: 20, color: "555555" })]
  });
}

// ====== 构建文档 ======

const children = [];

// 封面
children.push(new Paragraph({ spacing: { before: 3000 } }));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 200 },
  children: [new TextRun({ text: "临床医学综合押题密卷", font: "Microsoft YaHei", size: 48, bold: true, color: "2B579A" })]
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 100 },
  children: [new TextRun({ text: "（30道简答题 + 7道案例分析题）", font: "Microsoft YaHei", size: 28, color: "666666" })]
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 600 },
  children: [new TextRun({ text: "含详细评分标准与答案解析", font: "Microsoft YaHei", size: 24, color: "888888" })]
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 200 },
  children: [new TextRun({ text: "适用对象：临床医学专业考试备考", font: "Microsoft YaHei", size: 22, color: "555555" })]
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "2026年7月", font: "Microsoft YaHei", size: 22, color: "555555" })]
}));
children.push(new Paragraph({ children: [new PageBreak()] }));

// 目录
children.push(sectionTitle("目  录"));
children.push(new TableOfContents("目录", { hyperlink: true, headingStyleRange: "1-3" }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ====== 第一部分：简答题 ======
children.push(sectionTitle("第一部分  简答题（共30题）"));
children.push(normalPara("说明：本部分共30道简答题，每题5分（部分题目为6分），共150分。每道题需在80字左右回答，要求简明扼要、要点齐全。"));
children.push(new Paragraph({ children: [new PageBreak()] }));

// 按科目分组输出简答题
const subjects = [...new Set(shortAnswerQuestions.map(q => q.subject))];
subjects.forEach(subject => {
  children.push(subTitle(subject));

  const subjectQuestions = shortAnswerQuestions.filter(q => q.subject === subject);
  subjectQuestions.forEach(q => {
    children.push(subSubTitle(`第${q.id}题  【知识点：${q.knowledge}】`));
    children.push(boldPara(`题干：${q.stem}`));
    children.push(normalPara(`参考答案：${q.answer}`));
    children.push(new Paragraph({
      spacing: { after: 200, line: 340 },
      children: [
        new TextRun({ text: "评分标准：", font: "Microsoft YaHei", size: 20, bold: true, color: "C00000" }),
        new TextRun({ text: q.rubric, font: "Microsoft YaHei", size: 20, color: "C00000" })
      ]
    }));
  });
});

children.push(new Paragraph({ children: [new PageBreak()] }));

// ====== 第二部分：案例分析题 ======
children.push(sectionTitle("第二部分  案例分析题（共7题）"));
children.push(normalPara("说明：本部分共7道案例分析题，每题20分，共140分。每道题包含3-5个问题，每题要求200字左右的回答，要求结合临床情景进行分析。"));
children.push(new Paragraph({ children: [new PageBreak()] }));

caseQuestions.forEach(c => {
  children.push(subTitle(`案例${c.id}：${c.title}（${c.subject}）`));

  // 临床情景
  children.push(subSubTitle("临床情景"));
  children.push(new Paragraph({
    spacing: { after: 120, line: 360 },
    indent: { left: 180 },
    children: [new TextRun({ text: c.scenario, font: "Microsoft YaHei", size: 21, italics: true })]
  }));

  // 问题
  children.push(subSubTitle("问题"));
  c.questions.forEach((q, qi) => {
    children.push(new Paragraph({
      spacing: { after: 80, line: 340 },
      indent: { left: 360 },
      children: [
        new TextRun({ text: `${qi + 1}. `, font: "Microsoft YaHei", size: 21, bold: true }),
        new TextRun({ text: q, font: "Microsoft YaHei", size: 21 })
      ]
    }));
  });

  // 参考答案
  children.push(subSubTitle("参考答案"));
  c.answers.forEach((a, ai) => {
    children.push(new Paragraph({
      spacing: { after: 80, line: 340 },
      indent: { left: 180 },
      children: [
        new TextRun({ text: `第${ai + 1}问：`, font: "Microsoft YaHei", size: 20, bold: true, color: "2B579A" }),
      ]
    }));
    children.push(new Paragraph({
      spacing: { after: 120, line: 340 },
      indent: { left: 360 },
      children: [new TextRun({ text: a, font: "Microsoft YaHei", size: 20 })]
    }));
  });

  // 评分标准
  children.push(new Paragraph({
    spacing: { before: 120, after: 200, line: 340 },
    children: [
      new TextRun({ text: "评分标准：", font: "Microsoft YaHei", size: 20, bold: true, color: "C00000" }),
      new TextRun({ text: c.rubric, font: "Microsoft YaHei", size: 20, color: "C00000" })
    ]
  }));

  // 案例间分页
  children.push(new Paragraph({ children: [new PageBreak()] }));
});

// ====== 简答题速查表 ======
children.push(sectionTitle("附录：简答题速查表"));
children.push(normalPara("以下表格汇总全部30道简答题的核心信息，方便快速查阅复习。"));

const summaryHeaders = ["编号", "科目", "知识点", "题干摘要"];
const summaryColWidths = [800, 1600, 2600, 4360];
const summaryRows = shortAnswerQuestions.map(q => [
  q.id.toString(),
  q.subject,
  q.knowledge,
  q.stem.length > 50 ? q.stem.substring(0, 50) + "..." : q.stem
]);
children.push(createTable(summaryHeaders, summaryRows, summaryColWidths));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ====== 案例分析题速查表 ======
children.push(sectionTitle("附录：案例分析题速查表"));

const caseHeaders = ["编号", "科目", "案例标题", "问题数", "总分"];
const caseColWidths = [800, 1800, 3000, 1200, 800];
const caseRows = caseQuestions.map(c => [
  c.id.toString(),
  c.subject,
  c.title,
  c.questions.length.toString(),
  "20分"
]);
children.push(createTable(caseHeaders, caseRows, caseColWidths));

// ====== 生成文档 ======
const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: "Microsoft YaHei", size: 21 }
      }
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Microsoft YaHei", color: "2B579A" },
        paragraph: { spacing: { before: 360, after: 240 }, outlineLevel: 0 }
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Microsoft YaHei", color: "2B579A" },
        paragraph: { spacing: { before: 240, after: 180 }, outlineLevel: 1 }
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "Microsoft YaHei", color: "333333" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 }
      }
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1440, right: 1200, bottom: 1440, left: 1200 }
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "临床医学综合押题密卷", font: "Microsoft YaHei", size: 18, color: "999999" })]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "第 ", font: "Microsoft YaHei", size: 18, color: "999999" }),
            new TextRun({ children: [PageNumber.CURRENT], font: "Microsoft YaHei", size: 18, color: "999999" }),
            new TextRun({ text: " 页", font: "Microsoft YaHei", size: 18, color: "999999" })
          ]
        })]
      })
    },
    children
  }]
});

const outputPath = "C:\\Users\\Jy-Mentor-7\\Desktop\\新建文件夹 (2)\\临床医学综合押题密卷.docx";

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(outputPath, buffer);
  console.log("文档已生成：" + outputPath);
  console.log("简答题：" + shortAnswerQuestions.length + " 道");
  console.log("案例分析题：" + caseQuestions.length + " 道");
}).catch(err => {
  console.error("生成失败：" + err.message);
  process.exit(1);
});