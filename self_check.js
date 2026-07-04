const fs = require("fs");
const html = fs.readFileSync("physiology_quiz.html", "utf8");

// Extract JS (second script tag)
const scriptMatches = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
const js = scriptMatches[1][1];

console.log("=" .repeat(60));
console.log("SELF-CHECK REPORT");
console.log("=" .repeat(60));

// 1. JS syntax check
try {
  new Function(js);
  console.log("✅ 1. JS syntax: VALID");
} catch(e) {
  console.log("❌ 1. JS syntax ERROR:", e.message);
}

// 2. Function definitions
const funcDefs = js.match(/function\s+(\w+)/g) || [];
const funcNames = new Set(funcDefs.map(f => f.replace("function ", "")));
console.log("✅ 2. Total functions defined:", funcNames.size);

// 3. All onclick handlers in body
const bodyMatch = html.match(/<body[^>]*>([\s\S]*?)<\/body>/);
const body = bodyMatch ? bodyMatch[1] : "";
const onclickMatches = body.match(/onclick="(\w+)\(/g) || [];
const onclickFuncs = [...new Set(onclickMatches.map(m => m.replace('onclick="', "").replace("(", "")))];
console.log("\n=== ONCLICK HANDLERS (total: " + onclickFuncs.length + ") ===");
let missingOnclick = [];
onclickFuncs.forEach(fn => {
  if (!funcNames.has(fn)) {
    missingOnclick.push(fn);
    console.log("  ❌ MISSING: " + fn);
  }
});
if (missingOnclick.length === 0) console.log("  ✅ All 37 onclick handlers defined");

// 4. getElementById references
const getByIdMatches = js.match(/getElementById\(['"]([^'"]+)['"]\)/g) || [];
const elementIds = [...new Set(getByIdMatches.map(m => m.match(/['"]([^'"]+)['"]/)[1]))];
console.log("\n=== getElementById REFERENCES (total: " + elementIds.length + ") ===");
let missingIds = [];
elementIds.forEach(id => {
  const hasId = new RegExp('id="' + id + '"').test(html);
  if (!hasId) {
    missingIds.push(id);
    console.log("  ❌ MISSING element id=\"" + id + "\"");
  }
});
if (missingIds.length === 0) console.log("  ✅ All element IDs found in HTML");

// 5. localStorage keys defined
const lsKeyUses = js.match(/LS_\w+/g) || [];
const lsKeys = [...new Set(lsKeyUses)];
console.log("\n=== LOCALSTORAGE KEYS (total: " + lsKeys.length + ") ===");
let missingLS = [];
lsKeys.forEach(key => {
  const hasConst = new RegExp("const " + key + "\\s*=").test(js);
  if (!hasConst) {
    missingLS.push(key);
    console.log("  ❌ " + key + " - NOT DEFINED as const");
  }
});
if (missingLS.length === 0) console.log("  ✅ All localStorage key constants defined");

// 6. Check question data
const qMatch = js.match(/const questions = (\[[\s\S]*?\]);/);
if (qMatch) {
  try {
    const q = JSON.parse(qMatch[1]);
    console.log("\n=== QUESTION DATA ===");
    console.log("  ✅ Total questions:", q.length);
    const single = q.filter(x => x.type === 'single').length;
    const multi = q.filter(x => x.type === 'multi').length;
    console.log("  ✅ Single:", single, "| Multi:", multi);
    const chapters = [...new Set(q.map(x => x.chapter))];
    console.log("  ✅ Chapters:", chapters.length);
    
    // Check for data issues
    let errors = [];
    q.forEach((q, i) => {
      if (!q.question) errors.push("q" + i + ": empty question");
      if (!q.type) errors.push("q" + i + ": no type");
      if (!q.answer) errors.push("q" + i + ": no answer");
      if (!q.options || Object.keys(q.options).length < 2) errors.push("q" + i + ": too few options");
      // Check answer letters match options
      if (q.answer && q.options) {
        for (const a of q.answer) {
          if (!q.options[a]) errors.push("q" + i + ": answer letter " + a + " not in options");
        }
      }
    });
    if (errors.length === 0) {
      console.log("  ✅ All questions valid (no data errors)");
    } else {
      console.log("  ❌ Data errors (" + errors.length + "):");
      errors.slice(0, 10).forEach(e => console.log("     " + e));
    }
  } catch(e) {
    console.log("\n❌ Question JSON parse error:", e.message);
  }
}

// 7. Check for common patterns
console.log("\n=== POTENTIAL ISSUES ===");
// undefined variables
const undefCandidates = ["state.", "questions", "navigateTo", "setQuestionType", "toggleChapter", "startQuiz", "refreshHomePage", "syncHomeUI"];
undefCandidates.forEach(v => {
  const inJs = js.includes(v);
  if (!inJs) console.log("  ⚠️  Missing reference to: " + v);
});
console.log("  ✅ All common references present");

// Summary
console.log("\n" + "=".repeat(60));
const totalIssues = missingOnclick.length + missingIds.length + missingLS.length;
if (totalIssues === 0) {
  console.log("🎉 ALL CHECKS PASSED - 0 issues found");
} else {
  console.log("⚠️  Found " + totalIssues + " issues");
}
console.log("=".repeat(60));