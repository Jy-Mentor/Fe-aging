const fs = require("fs");

const htmlPath = "physiology_quiz.html";
const jsonPath = "parsed_questions.json";

let html = fs.readFileSync(htmlPath, "utf-8");
const questions = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));

// Find boundaries
const startIdx = html.indexOf("const questions = [");
const corruptIdx = html.indexOf("], a[j]] = [a[j], a[i]];");

if (startIdx === -1) { console.error("找不到起始"); process.exit(1); }
if (corruptIdx === -1) { console.error("找不到破坏点"); process.exit(1); }

console.log("start:", startIdx, "corrupt:", corruptIdx);

const before = html.substring(0, startIdx);
const after = html.substring(corruptIdx + 1);

const newQuestions = "const questions = " + JSON.stringify(questions) + ";\n\n        // ==================== STATE ====================\n\n        function shuffleArray(a) {\n            for (let i = a.length - 1; i > 0; i--) {\n                const j = Math.floor(Math.random() * (i + 1));\n                [a[i]";

const newHtml = before + newQuestions + after;

fs.writeFileSync(htmlPath, newHtml, "utf-8");

// Verify
const verify = fs.readFileSync(htmlPath, "utf-8");
const funcs = verify.match(/function shuffle\w*/g);
console.log("Functions found:", funcs ? [...new Set(funcs)].join(", ") : "none");
console.log("Total:", questions.length, "Single:", questions.filter(q => q.type === "single").length, "Multi:", questions.filter(q => q.type === "multi").length);