// Stage-0 result report as a Word file (numbers copied from results/report.md, 2026-10-20 run).
// Usage: node make_stage0_docx.js <out.docx>
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, HeadingLevel,
  AlignmentType, WidthType, ShadingType, BorderStyle, LevelFormat, PageBreak,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const OUT = process.argv[2] || path.join(ROOT, "docs", "记忆压缩实验_阶段0_结果报告.docx");
const PAGE_W = 11906, MARGIN = 1134, CONTENT_W = PAGE_W - 2 * MARGIN;  // A4, 2 cm margins

const FONT = { ascii: "Arial", hAnsi: "Arial", eastAsia: "Microsoft YaHei" };
const border = { style: BorderStyle.SINGLE, size: 4, color: "B0B0B0" };
const borders = { top: border, bottom: border, left: border, right: border };

const p = (text, opts = {}) => new Paragraph({ spacing: { after: 120 }, ...opts, children: runs(text) });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const bullet = (text) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: runs(text) });
const numbered = (text) => new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 }, children: runs(text) });

// "**bold**" spans inside a string
function runs(text) {
  return text.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((s) =>
    s.startsWith("**") ? new TextRun({ text: s.slice(2, -2), bold: true }) : new TextRun(s));
}

function table(header, rows, weights, highlight = []) {
  const total = weights.reduce((a, b) => a + b, 0);
  const widths = weights.map((w) => Math.floor((CONTENT_W * w) / total));
  widths[widths.length - 1] += CONTENT_W - widths.reduce((a, b) => a + b, 0);
  const cell = (text, i, head, hl) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: head ? { fill: "E8EEF4", type: ShadingType.CLEAR } : hl ? { fill: "FFF6DA", type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: String(text).split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((s) => s.startsWith("**")
        ? new TextRun({ text: s.slice(2, -2), bold: true, size: 19 })
        : new TextRun({ text: s, bold: head, size: 19 })) })],
  });
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, true, false)) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((t, i) => cell(t, i, false, highlight.includes(ri))) })),
    ],
  });
}

function image(file, w, h) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(ROOT, "results", file)),
      transformation: { width: w, height: h }, altText: { title: file, description: file, name: file } })] });
}
const caption = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [new TextRun({ text: t, italics: true, size: 18, color: "555555" })] });

function pngSize(file) {
  const b = fs.readFileSync(path.join(ROOT, "results", file));
  return [b.readUInt32BE(16), b.readUInt32BE(20)];
}
function fitImage(file, maxW) {
  const [w, h] = pngSize(file);
  return image(file, maxW, Math.round((maxW * h) / w));
}

const children = [
  new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("记忆压缩实验 · 阶段 0 结果报告")] }),
  p("模型：Qwen3-4B（冻结）　数据：UCI Combined Cycle Power Plant（CCPP）　方法：Cartridges（打补丁版）　日期：2026-10-21",
    { spacing: { after: 240 } }),

  h1("一、一句话结论"),
  p("把约 1.1 万词元的完整记忆 M（256 行数据）压缩成只有 64 个位置的 Z（KV 位置数为原来的 1/171.5），模型用 Z 回答测试题的 RMSE 为 **10.01 ± 0.19**（3 个种子），用完整记忆 B 为 **9.79**；相对“没有记忆”的 RMSE 改进，Z 恢复了 B 的大约 **97%**（95% CI 94.6%–99.3%）。"),
  p("**对照实验（第七节）的结论**：Z 会忠实地跟着老师随记忆变化的行为走。把记忆打乱、整体平移 +10，或者只平移低温区的 +10，Z 都跟着变；只平移低温区时，Z 只在低温区移动（+9.18），高温区基本不动（−0.21），并且逐题复现老师的整体预测比用同样老师答案拟合的 OLS / 5-NN 更准（平均平移量上 5-NN 跟得和 Z 一样好）。另一方面，信息是通过老师答案进入 Z 的：用无关文本初始化 Z 结果不变，蒸馏数据从 1024 条减到 64 条时 Z 明显变差。这符合蒸馏的预期，所以应该表述为“行为蒸馏”，而不是“Z 独立压缩了 M”。"),
  p("**汇报时需说明**：只用训练 Z 时的 1024 条老师答案拟合一个线性回归，测试 RMSE 就有 8.80，比 Z 还低；用真实标签拟合 M 的 OLS 更是 4.59。所以 CCPP 回归适合作为技术可行性实验，不能用来说明 KV 记忆是回归任务的最佳压缩方式。"),

  h1("二、实验设置"),
  h2("2.1 数据划分（各部分互不重叠）"),
  table(["用途", "条数", "说明"], [
    ["记忆池 / 记忆 M", "1024 / 256", "从记忆池抽 256 行作为 M（r0_n256），约 10,979 词元"],
    ["蒸馏查询", "1024", "老师 B 回答这些问题，作为训练 Z 的监督"],
    ["开发集 dev", "256", "按开发集 KL 选 checkpoint 和学习率"],
    ["测试集 test", "2000", "最终结果只在这里报告"],
    ["tune（辅助）", "256", "训练前检查老师是否真的利用记忆"],
    ["reserve（备用）", "5008", "未使用"],
  ], [3, 2, 7]),
  p("去重后共 9568 行，跨集合输入泄漏为 0。", { spacing: { before: 120, after: 120 } }),
  h2("2.2 实验分组"),
  bullet("**A　无记忆**：只给问题。"),
  bullet("**B　完整记忆**：问题前放入完整 M（老师）。"),
  bullet("**C　压缩记忆 Z**：问题前放入 64 位置的可训练 KV 缓存，通过蒸馏模仿 B（学生）。"),
  bullet("对照：B-打乱（M 中输入与 PE 的对应被打乱）、C-初始（训练前的 Z）、传统方法（均值、线性回归 OLS、5-NN，都拟合同一份 256 行记忆）。"),
  h2("2.3 训练配置"),
  p("k = 64；学习率在 {5e-3, 2e-2} 中用种子 42 按开发集 KL 选出 2e-2，再在该学习率下补跑种子 43、44；损失为老师 top-20 概率上的蒸馏损失；按开发集上全词表 KL(B‖C) 选最佳 checkpoint。"),

  h1("三、主结果（测试集 2000 条）"),
  table(["条件", "缓存位置数", "缓存大小 (MB)", "解析率", "RMSE", "MAE", "与 B 的数字差", "KL(B‖X)"], [
    ["A 无记忆", "0", "0", "拒答", "17.12*", "14.91*", "—", "3.519"],
    ["B 完整记忆", "10,979", "1543.9", "1.000", "9.79", "7.17", "0", "0"],
    ["B-打乱", "10,979", "1543.9", "1.000", "15.97", "12.67", "10.21", "0.621"],
    ["C-初始", "64", "9.0", "0.000", "17.12*", "14.86*", "—", "1.886"],
    ["C 种子 42", "64", "9.0", "1.000", "9.80", "7.13", "1.62", "0.096"],
    ["C 种子 43", "64", "9.0", "1.000", "10.09", "7.44", "1.59", "0.095"],
    ["C 种子 44", "64", "9.0", "1.000", "10.15", "7.49", "1.78", "0.107"],
    ["均值基线", "—", "—", "—", "17.12", "14.86", "—", "—"],
    ["OLS（拟合 M 的 256 行真实数据）", "—", "—", "—", "4.59", "3.69", "—", "—"],
    ["5-NN（拟合 M 的 256 行真实数据）", "—", "—", "—", "5.36", "4.21", "—", "—"],
  ], [3, 2, 2.2, 1.6, 1.6, 1.6, 2, 1.8], [4, 5, 6]),
  p("* A 和 C-初始都不输出数字（A 在长回答中 100% 明确拒答），按“预测记忆均值”回退计算。缓存大小为理论值，未实测显存。",
    { spacing: { before: 120, after: 120 } }),
  p("**C 三个种子汇总**：RMSE 10.014 ± 0.188；MAE 7.351 ± 0.191；KL(B‖C) 0.0996 ± 0.0067；与 B 的数字差 1.663 ± 0.100 MW。整条答案与 B 完全相同的比例为 39.5%–44.2%。"),

  h1("四、统计检验（配对 bootstrap，5000 次，95% CI）"),
  table(["比较", "差值", "95% CI", "解读"], [
    ["RMSE(C 三种子平均) − RMSE(B)", "+0.221", "[+0.049, +0.399]", "C 略差于 B，差异显著但很小"],
    ["RMSE(C 种子 42) − RMSE(B)", "+0.009", "[−0.206, +0.216]", "无显著差异（该种子参与了选学习率）"],
    ["RMSE(C 种子 43) − RMSE(B)", "+0.294", "[+0.085, +0.503]", "略差"],
    ["RMSE(C 种子 44) − RMSE(B)", "+0.362", "[+0.145, +0.592]", "略差"],
    ["RMSE(C) − RMSE(B-打乱)", "−5.954", "[−6.539, −5.368]", "C 远好于打乱记忆"],
    ["RMSE(C) − RMSE(均值)", "−7.107", "[−7.423, −6.790]", "C 远好于无记忆"],
    ["RMSE(B) − RMSE(B-打乱)", "−6.176", "[−6.778, −5.571]", "老师确实在用记忆内容"],
  ], [4, 1.6, 2.4, 4]),
  p("**恢复率** R = (RMSE均值 − RMSE_C) / (RMSE均值 − RMSE_B) = **97.0%** [94.6%, 99.3%]。只包含测试查询抽样的不确定性。",
    { spacing: { before: 120, after: 120 } }),

  h1("五、训练前的检查"),
  h2("5.1 老师 B 是否真的利用了记忆（tune 集 256 条，每个长度 3 份随机抽样）"),
  table(["记忆行数", "词元数", "B 解析率", "B RMSE", "B-打乱 RMSE", "OLS", "5-NN"], [
    ["16", "689", "0.05", "16.42", "16.71", "4.82", "7.90"],
    ["64", "2,747", "1.00", "10.17", "22.17", "4.36", "6.10"],
    ["128", "5,493", "1.00", "8.15", "21.29", "4.20", "5.76"],
    ["256", "10,980", "1.00", "8.39", "19.35", "4.13", "5.23"],
    ["512", "21,953", "1.00", "6.44", "19.13", "4.12", "4.58"],
  ], [1.6, 1.6, 1.6, 1.6, 2, 1.4, 1.4], [3]),
  p("记忆越长 B 越好，且明显好于 B-打乱，说明老师在利用记忆。16 行时不通过是输出格式问题（16 词元内基本不给数字），不是记忆无效。预注册的主记忆 r0_n256 上，B 相对 B-打乱的 ΔRMSE = −5.06 [−6.72, −3.51]。",
    { spacing: { before: 120, after: 120 } }),
  h2("5.2 实现正确性"),
  bullet("先算 M 的 KV 再接问题，与整串一次前向：KL 0.0065，与 bf16 本身的数值噪声（0.0074）相当，说明两种方式等价。"),
  bullet("Cartridges 专用模型与 HuggingFace 模型输出一致（k=64 初始化 KL 0.0026，完整 M KL 0.0123）。"),
  bullet("老师、学生、测试三处的输入逐 token 核对一致，Qwen3 思考模式已关闭（含空思考标记）。"),
  bullet("修复了 Cartridges 的 3 个问题：缓存读回位置数错误（P1）、到达步数不停止训练（P2）、bf16 参数导致 33%–57% 的更新被舍入为 0（P4，改为 fp32 主参数后降到 1e-5 以下）。"),
  bullet("老师 top-20 概率覆盖率最小值 0.999998，只保留 top-20 基本不损失信息。"),

  h1("六、伪标签回归对照：同样的老师答案，普通回归能做到多少？"),
  p("不用任何记忆，只用训练 C 时同样的 1024 个（输入，老师答案）拟合传统回归器，在测试集上评估："),
  table(["方法", "测试 RMSE", "与 B 的相关", "与 B 的数字差"], [
    ["OLS（老师答案）", "8.80", "0.918", "3.34"],
    ["1-NN（老师答案）", "9.59", "0.915", "2.61"],
    ["5-NN（老师答案）", "9.29", "0.940", "2.64"],
    ["OLS（1024 条蒸馏输入 + 真实标签，参照）", "4.59", "0.884", "6.16"],
    ["5-NN（1024 条蒸馏输入 + 真实标签，参照）", "4.74", "0.880", "6.11"],
    ["C（64 位置 Z，三种子平均）", "10.01", "0.953", "1.66"],
  ], [4, 2, 2, 2], [5]),
  p("注：这里的“真实标签”回归器用的是 1024 条蒸馏输入的真实发电量，训练数据是第三节中“拟合 M 的 256 行”的 4 倍，所以 5-NN 为 4.74，低于第三节的 5.36，两者不矛盾。",
    { spacing: { before: 120, after: 0 } }),
  p("**解读**：只看 RMSE，用同样老师答案拟合的 OLS（8.80）已经不差于 C 和 B，所以“保留 97% 收益”单独不能证明 C 压缩了 M。不过 C 比这些回归器更贴近 B 的具体行为（相关 0.953 对 ≤0.940，数字差 1.66 对 ≥2.61 MW），说明它学到的是 B 的特定输出，而不只是一个平滑回归。",
    { spacing: { before: 120, after: 120 } }),

  h1("七、对照实验结果（2026-10-20 第二轮，均为种子 42）"),
  p("设置与主实验相同（r0_n256，k=64，学习率 2e-2，640 次更新，按开发集 KL 选 checkpoint）。每个 Z 都与“用这个 Z 的同一批（输入，老师答案）拟合的 OLS / 5-NN”对比，这些回归器不用任何记忆。"),
  h2("7.1 打乱老师、初始化、蒸馏量、查询来源"),
  table(["实验", "蒸馏条数", "老师", "Z 测试 RMSE", "Z 与老师相关", "Z 与老师数字差", "OLS RMSE", "5-NN RMSE", "5-NN 与老师相关", "5-NN 与老师数字差"], [
    ["主实验 C", "1024", "B", "9.80", "0.955", "1.62", "8.80", "9.29", "0.940", "2.63"],
    ["打乱老师", "1024", "B-打乱", "16.25", "0.771", "3.81", "12.40", "14.26", "0.662", "6.07"],
    ["无关文本初始化", "1024", "B", "9.77", "0.943", "1.83", "8.80", "9.29", "0.940", "2.63"],
    ["蒸馏 256 条", "256", "B", "10.20", "0.930", "2.15", "9.08", "9.80", "0.930", "2.97"],
    ["蒸馏 64 条", "64", "B", "11.49", "0.857", "3.25", "10.43", "11.31", "0.903", "3.96"],
    ["随机输入蒸馏", "1024", "B", "9.33", "0.948", "1.86", "9.69", "9.28", "0.920", "3.21"],
  ], [2.6, 1.3, 1.4, 1.5, 1.5, 1.5, 1.3, 1.3, 1.5, 1.6]),
  p("“与老师相关 / 数字差”按各自的老师计算（打乱老师一行对照 B-打乱）。参照：B 9.79，B-打乱 15.97；无关文本初始化在训练前不输出数字（RMSE 17.12）。",
    { spacing: { before: 120, after: 120 } }),
  bullet("**打乱老师**：Z 跟着变差（16.25，接近 B-打乱的 15.97），而且比 5-NN 更贴近这个老师（相关 0.771 对 0.662），说明 Z 学的是老师具体给出的内容。"),
  bullet("**无关文本初始化**：结果和从 M 初始化几乎一样（9.77 对 9.80），所以 Z 的信息不是来自“用 M 的前 64 个词元初始化”。"),
  bullet("**蒸馏量**：64 条时 Z 明显变差（11.49），与老师的相关甚至低于 5-NN（0.857 对 0.903）。数据少时 Z 不比普通回归器更能还原老师。"),
  bullet("**随机输入蒸馏**：用在特征范围内独立均匀采样、与测试分布无关的输入蒸馏，测试 RMSE 9.33，与老师相关 0.948，基本不输给用真实分布查询蒸馏。说明不需要知道测试查询长什么样。"),
  h2("7.2 反事实记忆：老师变，Z 跟着变吗？"),
  p("先确认前提：在 1024 条蒸馏题上，整体平移后老师 B′ 相对 B 平均移动 +10.86（标准差 3.09），老师确实在用记忆内容。"),
  table(["测试集平均平移", "整体 +10", "低温区（AT<21.13，1024 题）", "高温区（976 题）"], [
    ["老师 B′ − B", "+10.84", "+9.55", "+0.26"],
    ["Z′ − Z（Z′ 从 M′ 初始化）", "+11.91", "+9.18", "−0.21"],
    ["Z′ − Z（Z′ 从原 M 初始化）", "+10.61", "—", "—"],
    ["OLS（B′ 答案）− OLS（B 答案）", "+10.91", "+8.87", "+1.43"],
    ["5-NN（B′ 答案）− 5-NN（B 答案）", "—", "+9.55", "+0.52"],
  ], [4, 1.8, 3, 2.4], [1]),
  p("区域平移时，各方法与老师 B″ 的 RMSE：**Z 3.95**，5-NN 4.76，OLS 5.81。",
    { spacing: { before: 120, after: 120 } }),
  bullet("**整体 +10**：Z 跟着平移，从原 M 初始化也一样（+10.61），再次说明信息来自老师答案而不是初始化。但 OLS 改个截距就能跟上，所以这一项区分不了 Z 和回归器。"),
  bullet("**只平移低温区**：Z 只在低温区移动、高温区不动，和老师一致；OLS 会把变化“抹”到高温区（+1.43）。这是目前最直接的证据：Z 能保留老师对局部记忆变化的响应。"),
  bullet("**两项证据要分开看**：只看两个区域的平均平移，5-NN（+9.55 / +0.52）跟得和 Z 一样好，Z 在这一项上并不占优；Z 的优势在于逐题复现老师的整体预测更准（与 B″ 的 RMSE 3.95 对 5-NN 4.76）。准确的说法是：Z 能保留老师对局部记忆变化的响应，并且在该实验中比 5-NN 更准确地复现老师的整体预测行为。"),
  h2("7.3 小结"),
  p("Z 是对老师行为的高保真蒸馏：老师因记忆而变化（包括局部变化），Z 就跟着变化，而且在蒸馏数据足够时比同样监督下的回归器更贴近老师。信息通过老师答案进入 Z，与初始化无关；蒸馏数据太少（64 条）时这种优势消失。"),

  h1("八、第一类验证（2026-10-20 第三轮）：补种子、换记忆、换笔记大小"),
  p("设置与主实验相同（k=64，学习率 2e-2，640 次更新，按开发集 KL 选 checkpoint），只改动下面每一项说明的那个因素。“± ”为 3 个种子（42、43、44）的标准差。"),
  h2("8.1 关键对照补到 3 个种子"),
  table(["实验", "老师", "Z 测试 RMSE", "Z 与老师相关", "Z 与老师数字差"], [
    ["主实验 C", "B", "10.01 ± 0.19", "0.95 ± 0.00", "1.66 ± 0.10"],
    ["打乱老师", "B-打乱", "15.81 ± 0.57", "0.74 ± 0.03", "4.23 ± 0.36"],
    ["蒸馏 64 条", "B", "11.39 ± 0.68", "0.86 ± 0.04", "3.33 ± 0.24"],
    ["随机输入蒸馏", "B", "9.71 ± 0.34", "0.95 ± 0.00", "1.86 ± 0.03"],
  ], [3, 2, 2.4, 2.4, 2.4]),
  p("区域平移（低温区 +10）3 个种子：老师低温 +9.55 / 高温 +0.26；Z 低温 **+9.94 ± 0.78** / 高温 **−0.15 ± 0.05**；Z 与 B″ 的 RMSE **3.93 ± 0.07**（5-NN 4.76，OLS 5.81）。",
    { spacing: { before: 120, after: 120 } }),
  bullet("第七节的单种子结论在 3 个种子下都成立：打乱老师 Z 跟着变差，蒸馏 64 条明显变差，随机输入蒸馏基本不受影响；区域平移时 Z 只动低温区，逐题复现老师比 5-NN 更准，且种子间波动很小。"),
  h2("8.2 换两份记忆（r1、r2，同样 256 行，从记忆池独立抽样）"),
  table(["记忆", "B", "B-打乱", "Z（3 种子）", "恢复率 R", "Z 与 B 相关", "OLS（老师答案）", "OLS（M 真实数据）"], [
    ["r0（主实验）", "9.79", "15.97", "10.01 ± 0.19", "0.97 ± 0.03", "0.95", "8.80", "4.59"],
    ["r1", "8.56", "23.26", "7.89 ± 0.13", "1.08 ± 0.02", "0.95", "6.97", "4.59"],
    ["r2", "8.48", "20.99", "8.24 ± 0.15", "1.03 ± 0.02", "0.97", "7.60", "4.60"],
  ], [2.2, 1.2, 1.4, 2, 1.8, 1.6, 2, 2]),
  table(["区域平移（种子 42）", "老师 低温 / 高温", "Z 低温 / 高温", "与 B″ 的 RMSE：Z / 5-NN / OLS"], [
    ["r0", "+9.55 / +0.26", "+9.18 / −0.21", "3.95 / 4.76 / 5.81"],
    ["r1", "+10.16 / +0.50", "+10.33 / +0.10", "5.15 / 5.59 / 6.22"],
    ["r2", "+9.77 / +3.26", "+9.97 / +4.79", "5.19 / 5.31 / 5.92"],
  ], [2.6, 2.6, 2.6, 3.4]),
  bullet("三份记忆上结论一致：Z 都恢复了相对无记忆的大部分 RMSE 改进（r1、r2 上 R 略大于 1，即 Z 的 RMSE 比 B 还略低一点），与 B 的相关都在 0.95 以上。"),
  bullet("区域平移在三份记忆上都复现：Z 跟着老师移动低温区，并且逐题复现老师比 5-NN、OLS 都更准（r2 上领先 5-NN 的幅度较小：5.19 对 5.31）。r2 上老师自己在高温区也移动了 +3.26，Z 也跟着移动（+4.79），说明 Z 跟的是老师的实际行为，而不是“规则上应该怎么变”。"),
  bullet("伪标签限定同样成立：在 r1、r2 上，用同样老师答案拟合的 OLS（6.97、7.60）的测试 RMSE 都比 Z 低。"),
  h2("8.3 笔记大小（k = 16 / 64 / 256，记忆 r0）"),
  table(["k", "压缩倍数（KV 位置数）", "Z 测试 RMSE", "Z 与 B 数字差"], [
    ["16", "686.2×", "10.05 ± 0.21", "1.74 ± 0.07"],
    ["64", "171.5×", "10.01 ± 0.19", "1.66 ± 0.10"],
    ["256", "42.9×", "10.04 ± 0.60", "1.67 ± 0.15"],
  ], [1.4, 3, 3, 3]),
  bullet("16 到 256 个位置结果几乎相同：16 个位置（686× 压缩）就够用了。这说明在这个任务上，瓶颈不是 Z 的大小，而是老师本身和蒸馏数据；也说明 CCPP 这个任务需要的信息量很小，要检验容量极限需要更难的任务或更大的记忆。"),
  h2("8.4 小结"),
  p("第一类验证全部通过：关键对照在 3 个种子下稳定，换两份记忆后主结果和区域平移结论都复现，笔记大小从 16 到 256 不影响结果。结论仍然是“行为蒸馏”：Z 稳定地复现老师随记忆变化的行为，信息来自老师答案。"),

  h1("九、汇报时请保留的限定"),
  numbered("打乱老师、蒸馏 64 条、随机输入和区域平移已补到 3 个种子（第八节）；无关文本初始化、蒸馏 256 条和整体平移仍为单种子（42）。主实验中种子间 RMSE 标准差约 0.19，无关文本初始化的差异（9.77 对 9.80）在这个范围内。"),
  numbered("整体平移实验中 Z′（从 M′ 初始化）在第 193 次更新就被开发集选中，开发 KL 0.16 高于其他组，训练可能不够充分。"),
  numbered("设定仍然有限：CCPP、Qwen3-4B、256 行记忆（r0 为预注册主记忆，r1、r2 为补充）、1024 条蒸馏、贪心解码；512 行记忆尚未测试。"),
  numbered("B 不是上限：B 与 1-NN 相关 0.85，约 57%–61% 的预测恰好是 M 中某一行的 PE，更像“查表”；97% 衡量的是对一个较弱老师的保真度。"),
  numbered("不参与选学习率的两个种子都显著差于 B（+0.29、+0.36）；种子 42 与 B 无差异，但它参与了学习率选择。"),
  numbered("学习率 2e-2 下种子 44 训练中途发散，靠开发集早停选到发散前的 checkpoint。"),
  numbered("171.5× 是 KV 位置数之比；显存、延迟未实测，训练成本未计入（老师数据约 4 分钟，每个训练约 15 分钟 H100）。"),
  numbered("实现为 Cartridges + 补丁 P1/P2/P4，偏离原版 bf16 参数；P4 未在完整训练上做消融。"),

  h1("十、下一步"),
  bullet("**扩大实验数据**：更大的记忆（如 512 行及以上）和更难的任务，检验 Z 的容量在什么时候开始成为瓶颈（第八节显示 CCPP 上 16 个位置就够）。"),
  bullet("**衔接后续研究**：区域平移可以自然扩展为分段记忆，即不同区域分别压缩成 Z，再检验合并。"),

  new Paragraph({ children: [new PageBreak()] }),
  h1("附：图"),
  fitImage("fig_test_rmse.png", 600),
  caption("图 1　测试集 RMSE 对比"),
  fitImage("fig_kl_curves.png", 600),
  caption("图 2　训练过程中开发集 KL(B‖C) 曲线"),
];

const doc = new Document({
  creator: "MZ project",
  title: "记忆压缩实验 阶段0 结果报告",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal",
        run: { font: FONT, size: 36, bold: true, color: "1F3864" }, paragraph: { spacing: { after: 120 } } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true, color: "1F3864" },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 23, bold: true, color: "2E5597" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 600, hanging: 300 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 600, hanging: 360 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, buf.length, "bytes"); });
