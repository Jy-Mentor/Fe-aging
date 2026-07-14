const fs = require('fs');
const content = fs.readFileSync('D:/铁衰老 绝不重蹈覆辙/generate_proposal_v9_polished.js', 'utf-8');
const lines = content.split('\n');

let section = '';
let subsection = '';
let counts = {};
let subCounts = {};
let inRefs = false;

lines.forEach(line => {
  if (line.includes("h1('")) {
    const m = line.match(/h1\('([^']+)'\)/);
    if (m) {
      section = m[1];
      subsection = '';
      if (section.includes('参考文献')) inRefs = true;
    }
  }
  if (line.includes("h2('")) {
    const m = line.match(/h2\('([^']+)'\)/);
    if (m) subsection = m[1];
  }
  if (section && !inRefs) {
    const zh = (line.match(/[\u4e00-\u9fa5]/g) || []).length;
    if (!counts[section]) counts[section] = 0;
    counts[section] += zh;
  }
});

console.log('=== 各部分估算字数（按75%比例） ===');
let total = 0;
Object.entries(counts).forEach(([k, v]) => {
  const est = Math.round(v * 0.75);
  console.log(k + ': ' + est + ' 字');
  total += est;
});
console.log('---');
console.log('总估算字数（不含参考文献）: ' + total + ' 字');

const liYi = counts['一、立项依据与研究内容'] || 0;
console.log('立项依据与研究内容估算: ' + Math.round(liYi * 0.75) + ' 字');
