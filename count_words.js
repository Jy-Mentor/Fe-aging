const fs = require('fs');
const { Document, Packer } = require('docx');

function extractText(docxPath) {
  const AdmZip = require('adm-zip');
  const zip = new AdmZip(docxPath);
  const xmlContent = zip.readAsText('word/document.xml');
  const text = xmlContent
    .replace(/<[^>]+>/g, '')
    .replace(/\s+/g, '')
    .trim();
  return text;
}

const filePath = 'D:/铁衰老 绝不重蹈覆辙/标书_整合版V3_桂艾BCP靶向Nrf2抑制铁依赖性SIPS改善CIRI.docx';
const fullText = extractText(filePath);

// 找到各部分的位置
const section1Start = fullText.indexOf('一、立项依据与研究内容');
const section1End = fullText.indexOf('二、研究目标');
const refStart = fullText.indexOf('参考文献');

const section1Text = fullText.substring(section1Start, section1End);
const section1WithoutRefs = fullText.substring(section1Start, refStart);
const totalText = fullText;

// 统计中文字符（基本就是总字数）
function countChinese(str) {
  return (str.match(/[\u4e00-\u9fa5]/g) || []).length;
}

console.log('=== 字数统计 ===');
console.log(`立项依据总字数（含参考文献）: ${countChinese(section1Text)}`);
console.log(`立项依据总字数（不含参考文献）: ${countChinese(section1WithoutRefs)}`);
console.log(`全文总字数（含参考文献）: ${countChinese(totalText)}`);

// 去掉参考文献后的总字数
const beforeRef = fullText.substring(0, refStart);
const afterRef = fullText.substring(refStart);
const refChinese = countChinese(afterRef.substring(0, afterRef.indexOf('二、研究目标')));
console.log(`参考文献部分中文字数: ${refChinese}`);
console.log(`全文总字数（不含参考文献）: ${countChinese(beforeRef) + countChinese(fullText.substring(section1End))}`);
