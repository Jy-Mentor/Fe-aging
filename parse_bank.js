const mammoth = require("mammoth");
const fs = require("fs");
const path = require("path");

const DOCS_DIR = "D:\\微信聊天记录\\xwechat_files\\wxid_1gtfoi7l8op622_8a67\\msg\\file\\2026-07";

const DOCS = [
  { file: "生理学第一章绪论_整合精简题库(1).docx", chapter: "第一章 绪论" },
  { file: "生理学第二章细胞_复习试题（去重整理）.docx", chapter: "第二章 细胞" },
  { file: "生理学第三章血液_复习试题（去重完整版）.docx", chapter: "第三章 血液" },
  { file: "生理学第四章循环_单选+多选汇总.docx", chapter: "第四章 循环" },
  { file: "第五章呼吸复习题（去重完整版）(1).docx", chapter: "第五章 呼吸" },
  { file: "第六章消化自测习题（去重整理版）(1).docx", chapter: "第六章 消化" },
  { file: "生理学泌尿生理_单选题（去重整合完整版）.docx", chapter: "第八章 泌尿" },
  { file: "生理学第十章神经生理学_单选汇总（去重整理）.docx", chapter: "第十章 神经" },
  { file: "生理学第十一章内分泌_单选题（去重完整版）.docx", chapter: "第十一章 内分泌" },
];

function stripHtml(html) {
  return html.replace(/<[^>]*>/g, "").replace(/&nbsp;/g, " ").replace(/&amp;/g, "&").replace(/&#x2060;/g, "").trim();
}

function parseQuestionsFromHtml(html, chapterName) {
  const lines = html.split(/<\/p>|<\/h[1-6]>/).map(l => stripHtml(l)).filter(l => l.trim());
  
  // Step 1: Extract all answers in order (sequential, ignoring question numbers)
  const allAnswers = [];
  let inAnswerSection = false;
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    
    if (line.match(/答案|标准答案|参考答案|答案汇总/i)) {
      inAnswerSection = true;
      continue;
    }
    
    if (inAnswerSection) {
      // Extract all answer patterns like "1.B", "2.ABCD", "3.E", etc.
      const matches = line.match(/(\d+)[\.．、]\s*([A-Z]+)/g);
      if (matches) {
        matches.forEach(m => {
          const match = m.match(/(\d+)[\.．、]\s*([A-Z]+)/);
          if (match) {
            allAnswers.push(match[2]);
          }
        });
      }
    }
  }
  
  // Step 2: Extract all questions in order
  const rawQuestions = [];
  let inQuestionSection = false;
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    
    if (line.match(/单选题|一、单选题/i) && !line.match(/答案/i)) {
      inQuestionSection = true;
      continue;
    }
    if (line.match(/多选题|二、多选题/i) && !line.match(/答案/i)) {
      inQuestionSection = true;
      continue;
    }
    if (line.match(/答案|标准答案|参考答案|答案汇总/i)) {
      break;
    }
    
    if (!inQuestionSection) continue;
    
    const qMatch = line.match(/^(\d+)[\.．、]\s*(.+)$/);
    if (qMatch) {
      const qText = qMatch[2].trim();
      
      const options = {};
      let j = i + 1;
      while (j < lines.length) {
        const optLine = lines[j].trim();
        const optMatch = optLine.match(/^([A-Z])[\.．、]\s*(.+)$/);
        if (optMatch) {
          options[optMatch[1]] = optMatch[2].trim();
          j++;
        } else {
          break;
        }
      }
      
      if (Object.keys(options).length >= 2) {
        rawQuestions.push({ question: qText, options: options });
      }
      
      i = j - 1;
    }
  }
  
  // Step 3: Match questions to answers by sequential order
  // Answer length determines type: len > 1 → multi, len = 1 → single
  const questions = [];
  for (let idx = 0; idx < rawQuestions.length; idx++) {
    const answer = idx < allAnswers.length ? allAnswers[idx] : "";
    const qType = answer.length > 1 ? "multi" : "single";
    
    questions.push({
      question: rawQuestions[idx].question,
      options: rawQuestions[idx].options,
      answer: answer,
      type: qType,
      chapter: chapterName,
      source: "生理学题库",
    });
  }
  
  return questions;
}

async function main() {
  let allQuestions = [];
  
  for (const doc of DOCS) {
    const filePath = path.join(DOCS_DIR, doc.file);
    console.log("Parsing:", doc.chapter, "-", doc.file);
    
    try {
      const result = await mammoth.convertToHtml({ path: filePath });
      const html = result.value;
      const questions = parseQuestionsFromHtml(html, doc.chapter);
      
      const singleCount = questions.filter(q => q.type === "single").length;
      const multiCount = questions.filter(q => q.type === "multi").length;
      const noAnswerCount = questions.filter(q => !q.answer).length;
      
      console.log(`  总题数: ${questions.length}, 单选: ${singleCount}, 多选: ${multiCount}, 无答案: ${noAnswerCount}`);
      
      if (noAnswerCount > 0) {
        console.log("  ⚠️ 无答案题目前5题:");
        questions.filter(q => !q.answer).slice(0, 5).forEach((q, i) => {
          console.log(`    [${i}] ${q.question.substring(0, 40)}`);
        });
      }
      
      allQuestions = allQuestions.concat(questions);
    } catch (e) {
      console.error("  解析失败:", e.message);
    }
  }
  
  console.log("\n=== 汇总 ===");
  console.log("总题数:", allQuestions.length);
  console.log("单选题:", allQuestions.filter(q => q.type === "single").length);
  console.log("多选题:", allQuestions.filter(q => q.type === "multi").length);
  
  // Validation
  const errors = [];
  allQuestions.forEach((q, i) => {
    const optKeys = Object.keys(q.options).sort();
    for (const letter of q.answer) {
      if (!optKeys.includes(letter)) {
        errors.push({ idx: i, chapter: q.chapter, answer: q.answer, badLetter: letter, opts: optKeys, q: q.question.substring(0, 40) });
      }
    }
    if (q.type === 'multi' && q.answer.length <= 1) {
      errors.push({ idx: i, chapter: q.chapter, problem: 'multi但答案=1字母', answer: q.answer, q: q.question.substring(0, 40) });
    }
    if (q.type === 'single' && q.answer.length > 1) {
      errors.push({ idx: i, chapter: q.chapter, problem: 'single但答案多字母', answer: q.answer, q: q.question.substring(0, 40) });
    }
  });
  
  console.log("\n验证错误:", errors.length);
  if (errors.length > 0) {
    errors.slice(0, 10).forEach(e => console.log("  ", JSON.stringify(e)));
  }
  
  const output = allQuestions.map(q => ({
    question: q.question,
    options: q.options,
    answer: q.answer,
    type: q.type,
    chapter: q.chapter,
    source: q.source,
  }));
  
  fs.writeFileSync("parsed_questions.json", JSON.stringify(output, null, 2), "utf-8");
  console.log("\n已保存到 parsed_questions.json");
}

main().catch(console.error);