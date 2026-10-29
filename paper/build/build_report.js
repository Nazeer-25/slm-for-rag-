// Builds the single-column reference report (.docx) for the project guide.
// Inputs: numbers.json (paper_numbers.py), refs.json, ../figures/*.png, ../matlab/make_report_figures.m
// Usage:  node build_report.js [toc_pages.json]
//   Pass 1 (no argument) writes headings.json; toc_pages.py turns the rendered PDF into
//   toc_pages.json; pass 2 fills the contents page with real page numbers.
const fs = require("fs");
const path = require("path");
const d = require("docx");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, WidthType,
  BorderStyle, HeadingLevel, Tab, TabStopType, LevelFormat, VerticalAlign, ShadingType, Header, Footer,
  PageNumber, PageBreak, LineRuleType,
} = d;

const HERE = __dirname;
const FIG = path.join(HERE, "..", "figures");
const OUT = path.join(HERE, "..", "Project_Report_IPI_Detection.docx");
const N = JSON.parse(fs.readFileSync(path.join(HERE, "numbers.json"), "utf8"));
const REFS = JSON.parse(fs.readFileSync(path.join(HERE, "refs.json"), "utf8"));
const TOC_FILE = process.argv[2];
const TOC_PAGES = TOC_FILE && fs.existsSync(TOC_FILE) ? JSON.parse(fs.readFileSync(TOC_FILE, "utf8")) : null;

// ------------------------------------------------------------------ geometry and palette (A4, 1 in margins)
const PAGE = { width: 11906, height: 16838 };
const MARGIN = 1440;
const TEXT_W = PAGE.width - 2 * MARGIN;          // 9026 DXA = 6.27 in
const FONT = "Times New Roman";
const NAVY = "1F3864", INK2 = "595959", HEAD_FILL = "E7ECF4", NOTE_FILL = "EEF3FA", CODE_FILL = "F4F4F2";
const ACCENT = "2A78D6", RULE_C = "808080", INNER_C = "D0D0D0";
const AUTO = (line) => ({ line, lineRule: LineRuleType.AUTO });

// ------------------------------------------------------------------ numbers
const p2 = (x) => (100 * x).toFixed(2);
const ci = (a) => `[${p2(a[0])}, ${p2(a[1])}]`;
const int = (x) => Math.round(x).toLocaleString("en-US");
const R = N.regular, C = N.challenge, F = N.families, PRM = N.params, TR = N.training;
const ML = F["Multilingual instructions"], DEP = F["Dependency traps"], UNI = F["Unicode spoofing"];
function wilson(k, n, z = 1.959963984540054) {
  const p = k / n, den = 1 + z * z / n, c = (p + z * z / (2 * n)) / den;
  const h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}
const UNI_CI = wilson(632 - 6, 632);              // n is 629-634; the bounds move by < 0.05 points
const ZERO_LO = wilson(625, 625)[0];
const MISS_ML = (100 * N.miss_share["Multilingual instructions"]).toFixed(1);
const EMB = 30522 * 768 + 512 * 768 + 2 * 768;

// ------------------------------------------------------------------ cross-reference numbering
const TABLES = ["glance", "bipia", "positioning", "claims", "modules", "families", "partitions", "legacy", "params",
  "config", "metrics", "results", "perfamily", "rules", "l2checks", "modes", "stack", "code", "next", "venue",
  "outline", "figmap", "verify", "figscripts"];
const FIGURES = ["arch", "partitions", "confusion", "metrics", "families"];
const tNum = (k) => { const i = TABLES.indexOf(k); if (i < 0) throw new Error(`table ${k}`); return i + 1; };
const fNum = (k) => { const i = FIGURES.indexOf(k); if (i < 0) throw new Error(`figure ${k}`); return i + 1; };

// ------------------------------------------------------------------ citations (numbered by first use)
const citeNum = new Map();
function cite(keys) {
  const nums = keys.map((k) => {
    if (!REFS[k]) throw new Error(`missing reference ${k}`);
    if (!citeNum.has(k)) citeNum.set(k, citeNum.size + 1);
    return citeNum.get(k);
  }).sort((a, b) => a - b);
  const parts = [];
  for (let i = 0; i < nums.length;) {
    let j = i;
    while (j + 1 < nums.length && nums[j + 1] === nums[j] + 1) j++;
    if (j - i >= 2) parts.push(`[${nums[i]}]–[${nums[j]}]`);
    else for (let k = i; k <= j; k++) parts.push(`[${nums[k]}]`);
    i = j + 1;
  }
  return parts.join(", ");
}

// Inline markup: **bold** *italic* ==highlight== [@key; @key] ^{sup} _{sub} @@mono@@ \* (literal star)
// {T:label} -> "Table n", {F:label} -> "Figure n"
function runs(text, base = {}) {
  const ESC = "\u0001";
  text = text.replace(/\\\*/g, ESC)
    .replace(/\{T:(\w+)\}/g, (_, k) => `Table ${tNum(k)}`)
    .replace(/\{F:(\w+)\}/g, (_, k) => `Figure ${fNum(k)}`);
  const out = [];
  const st = { bold: !!base.bold, italics: !!base.italics, hl: false };
  const push = (t, extra = {}) => {
    if (!t) return;
    out.push(new TextRun({ ...base, text: t.split(ESC).join("*"), bold: st.bold, italics: st.italics,
      ...(st.hl ? { highlight: "yellow" } : {}), ...extra }));
  };
  const re = /(\*\*|==|\*|\[@[^\]]+\]|\^\{[^}]+\}|_\{[^}]+\}|@@[^@]+@@)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    push(text.slice(last, m.index));
    const t = m[0];
    if (t === "**") st.bold = !st.bold;
    else if (t === "*") st.italics = !st.italics;
    else if (t === "==") st.hl = !st.hl;
    else if (t.startsWith("[@")) push(cite(t.slice(1, -1).split(";").map((s) => s.trim().replace(/^@/, ""))));
    else if (t.startsWith("^{")) push(t.slice(2, -1), { superScript: true });
    else if (t.startsWith("_{")) push(t.slice(2, -1), { subScript: true });
    else push(t.slice(2, -2), { font: "Courier New", size: Math.max(16, (base.size || 22) - 3) });
    last = m.index + t.length;
  }
  push(text.slice(last));
  return out;
}

// ------------------------------------------------------------------ block builders
const body = [];
const add = (...xs) => xs.flat().forEach((x) => body.push(x));
const headings = [];
const PG = (text, opts = {}) => new Paragraph({ style: "Body", children: runs(text), ...opts });
const PB = () => new Paragraph({ children: [new PageBreak()] });
function H1(num, title, { pageBreak = false } = {}) {
  const text = num ? `${num} ${title}` : title;
  headings.push({ level: 1, text });
  return new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: pageBreak, children: [new TextRun(text)] });
}
function H2(num, title) {
  const text = `${num} ${title}`;
  headings.push({ level: 2, text });
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
}
const H3 = (title) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(title)] });
const BUL = (text, level = 0) => new Paragraph({ style: "Body", numbering: { reference: "bullets", level },
  spacing: { after: 60, ...AUTO(276) }, children: runs(text) });
let listNo = 0;
const NUMLIST = (items) => { listNo += 1; return items.map((t) => new Paragraph({ style: "Body",
  numbering: { reference: "numbers", level: 0, instance: listNo }, spacing: { after: 60, ...AUTO(276) }, children: runs(t) })); };

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}
let figCount = 0;
function FIGURE(key, file, caption, widthIn = 4.7) {
  figCount += 1;
  if (FIGURES[figCount - 1] !== key) throw new Error(`figure order: expected ${FIGURES[figCount - 1]}, got ${key}`);
  const { w, h, data } = pngSize(path.join(FIG, `${file}.png`));
  const wpx = Math.round(widthIn * 96);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 80, ...AUTO(240) },
      children: [new ImageRun({ type: "png", data, transformation: { width: wpx, height: Math.round(wpx * h / w) },
        altText: { title: `Figure ${figCount}`, description: caption.replace(/\[@[^\]]+\]|\*|==/g, ""), name: file } })] }),
    new Paragraph({ style: "Caption", children: [new TextRun({ text: `Figure ${figCount}: `, bold: true }), ...runs(caption, { size: 19 })] }),
  ];
}
let eqCount = 0;
function EQ(file) {
  eqCount += 1;
  const { w, h, data } = pngSize(path.join(FIG, `${file}.png`));
  return new Paragraph({
    tabStops: [{ type: TabStopType.CENTER, position: TEXT_W / 2 }, { type: TabStopType.RIGHT, position: TEXT_W }],
    spacing: { before: 120, after: 120, ...AUTO(240) }, keepLines: true,
    children: [new TextRun({ children: [new Tab()] }),
      new ImageRun({ type: "png", data, transformation: { width: Math.round(w / 600 * 96), height: Math.round(h / 600 * 96) },
        altText: { title: `Equation ${eqCount}`, description: `Equation ${eqCount}`, name: file } }),
      new TextRun({ children: [new Tab(), `(${eqCount})`] })],
  });
}

const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const LINE = (sz, color = RULE_C) => ({ style: BorderStyle.SINGLE, size: sz, color });
let tabCount = 0;
function TABLE(key, title, widths, header, rows, { align = [], note, bold = [] } = {}) {
  tabCount += 1;
  if (TABLES[tabCount - 1] !== key) throw new Error(`table order: expected ${TABLES[tabCount - 1]}, got ${key}`);
  const total = widths.reduce((a, b) => a + b, 0);
  if (total > TEXT_W) throw new Error(`table ${key} too wide: ${total}`);
  const cell = (txt, i, head, last) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    verticalAlign: VerticalAlign.CENTER,
    shading: head ? { fill: HEAD_FILL, type: ShadingType.CLEAR, color: "auto" } : undefined,
    borders: { top: NONE, left: NONE, right: NONE, bottom: head ? LINE(6) : (last ? NONE : LINE(2, INNER_C)) },
    children: [new Paragraph({ style: "TableText", alignment: align[i] || AlignmentType.LEFT, keepNext: !last,
      children: runs(String(txt), { size: 19, bold: head }) })],
  });
  return [
    new Paragraph({ style: "Caption", keepNext: true, spacing: { before: 200, after: 80, ...AUTO(240) },
      children: [new TextRun({ text: `Table ${tabCount}: `, bold: true }), ...runs(title, { size: 19 })] }),
    new Table({
      width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
      borders: { top: LINE(8), bottom: LINE(8), left: NONE, right: NONE, insideHorizontal: NONE, insideVertical: NONE },
      rows: [new TableRow({ tableHeader: true, cantSplit: true, children: header.map((t, i) => cell(t, i, true, false)) }),
        ...rows.map((r, ri) => new TableRow({ cantSplit: true,
          children: r.map((t, i) => cell(bold.includes(ri) ? `**${t}**` : t, i, false, ri === rows.length - 1)) }))],
    }),
    new Paragraph({ style: "Caption", spacing: { before: 60, after: 200, ...AUTO(240) }, children: note ? runs(note, { size: 17, color: INK2 }) : [] }),
  ];
}
function CALLOUT(title, lines, { bullets = true } = {}) {
  return [new Table({
    width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: [TEXT_W],
    borders: { top: NONE, bottom: NONE, left: NONE, right: NONE, insideHorizontal: NONE, insideVertical: NONE },
    rows: [new TableRow({ cantSplit: true, children: [new TableCell({
      width: { size: TEXT_W, type: WidthType.DXA }, margins: { top: 100, bottom: 100, left: 200, right: 200 },
      shading: { fill: NOTE_FILL, type: ShadingType.CLEAR, color: "auto" },
      borders: { top: NONE, bottom: NONE, right: NONE, left: { style: BorderStyle.SINGLE, size: 24, color: ACCENT } },
      children: [new Paragraph({ spacing: { after: 80, ...AUTO(264) }, children: [new TextRun({ text: title, bold: true, color: NAVY, size: 22 })] }),
        ...lines.map((t) => new Paragraph({ ...(bullets ? { numbering: { reference: "bullets", level: 0 } } : {}),
          alignment: bullets ? AlignmentType.LEFT : AlignmentType.JUSTIFIED,
          spacing: { after: 60, ...AUTO(264) }, children: runs(t, { size: 21 }) }))],
    })] })],
  }), new Paragraph({ spacing: { after: 120 }, children: [] })];
}
function CODE(lines, size = 16) {
  return [new Table({
    width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: [TEXT_W],
    borders: { top: LINE(2, INNER_C), bottom: LINE(2, INNER_C), left: LINE(2, INNER_C), right: LINE(2, INNER_C),
      insideHorizontal: NONE, insideVertical: NONE },
    rows: [new TableRow({ cantSplit: false, children: [new TableCell({
      width: { size: TEXT_W, type: WidthType.DXA }, margins: { top: 100, bottom: 100, left: 140, right: 140 },
      shading: { fill: CODE_FILL, type: ShadingType.CLEAR, color: "auto" },
      children: lines.map((l) => new Paragraph({ spacing: { after: 0, line: 228, lineRule: LineRuleType.AUTO },
        children: [new TextRun({ text: l.length ? l : " ", font: "Courier New", size })] })),
    })] })],
  }), new Paragraph({ spacing: { after: 120 }, children: [] })];
}

// ====================================================================== CONTENT
// ---------------------------------------------------------------- Executive summary
add(H1("", "Executive Summary"));
add(PG(`This report is reference material for the conference paper on the project *A Lightweight Hybrid Defense Framework Against Indirect Prompt Injection in RAG Systems*. It concentrates on **detection (Layer 1)**, the trained and fully evaluated part, and covers the **defense layer (Layer 2)** briefly.`));
add(PG(`**Problem and approach.** In indirect prompt injection, an attacker hides instructions in documents that a retrieval-augmented generation (RAG) system later passes to the language model. The base paper, BIPIA (ACM KDD 2025) [@bipia], found all 25 evaluated LLMs vulnerable, and its strongest defense fine-tunes the generator, which a small CPU-only local deployment cannot afford. This project screens every retrieved chunk *before* generation with a small DistilBERT classifier (Layer 1) and sends only flagged chunks to a rule-based clause sanitizer (Layer 2), so the defense never calls a generative model.`));
add(PG(`**What has been done.**`, { spacing: { after: 40 }, keepNext: true }));
add(BUL(`**A complete local RAG pipeline:** MiniLM retrieval with Chroma, the two defense layers and a Llama 3.2 3B generator served by Ollama, plus a four-mode evaluation harness, a web workbench and unit tests.`));
add(BUL(`**A 100,000-chunk synthetic benchmark:** 50,000 policies, each clean and injected; 10 topics; 5,000 divisions; 32 attack families, 8 of them withheld for a challenge set.`));
add(BUL(`**One trained module,** the Layer 1 DistilBERT detector. Its last block and head were fine-tuned (${int(PRM.distilbert_trainable)} of ${int(PRM.distilbert_total)} parameters, ${(100 * PRM.trainable_share).toFixed(2)}%) on a stratified 6,000-chunk sample for 3 epochs on a laptop CPU. Epoch 1 and τ = 0.175 were selected on validation data.`));
add(BUL(`**Evaluation and figures:** the detector was evaluated on 20,000 held-out chunks and the sanitizer checked on the earlier 200-chunk corpus. Print-ready figures come with MATLAB and Python scripts (Appendix A).`));
add(TABLE("glance", "Key results at a glance (95% Wilson intervals in brackets)", [2400, 3313, 3313],
  ["Quantity", "Regular test (10,000 chunks, 24 seen families)", "Challenge (10,000 chunks, 8 withheld families)"],
  [
    ["Accuracy", `${p2(R.accuracy)}% ${ci(R.accuracy_ci)}`, `${p2(C.accuracy)}% ${ci(C.accuracy_ci)}`],
    ["Recall (detection rate)", `${p2(R.recall)}% ${ci(R.recall_ci)}`, `${p2(C.recall)}% ${ci(C.recall_ci)}`],
    ["False-positive rate", `0.00% (upper bound ${(100 * R.fpr_ci[1]).toFixed(3)}%)`, `0.00% (upper bound ${(100 * C.fpr_ci[1]).toFixed(3)}%)`],
    ["Errors", "none", "191 missed injections, 0 false alarms"],
    ["Weakest attack family", "—", `multilingual: ${p2(ML.recall)}% recall (${ML.missed} of ${ML.n} missed)`],
  ], { align: [AlignmentType.LEFT, AlignmentType.CENTER, AlignmentType.CENTER] }));
add(CALLOUT("Notes for the guide", [
  `**Venue timing.** The COMSNETS 2027 main-track deadline (31 August 2026) has passed; reported open tracks are the Graduate Forum (about 8–9 November; 4 pages + references page + 1-page resume), posters and demos (Section 12.1).`,
  `**Values to confirm.** ==Highlighted values== come from code defaults or inference, not the training log ({T:verify}).`,
  `**Double-blind.** Main-track and workshop papers must not show author names or the project's GitHub link.`,
  `**Still open.** No end-to-end attack-success numbers yet; synthetic data from one run with one seed; no external or adaptive tests (Chapter 11).`,
]));

// ---------------------------------------------------------------- 1 Introduction
add(H1("1", "Introduction", { pageBreak: true }));
add(H2("1.1", "Context"));
add(PG(`Retrieval-augmented generation (RAG) [@rag] has become the standard way to ground large language models (LLMs) in an organisation's own documents. A retriever selects the chunks most similar to the user's question, and a generator answers from them. The same mechanism opens an attack surface. Prompt injection heads the OWASP list of risks for LLM applications [@owasp]. In its *indirect* form, the attacker never interacts with the model. Instead, they plant instructions in content that the application will later retrieve, such as a wiki page, a shared policy file or an e-mail [@greshake]. In direct injection, the user types something like "ignore previous instructions" [@perez]. In indirect injection the victim is an honest user, and the malicious text arrives through a channel that the application treats as data.`));
add(H2("1.2", "Problem Statement"));
add(PG(`Consider an organisation that runs an internal policy assistant entirely on its own hardware. A small open model answers employees' questions from internal documents found by vector search. Anyone who can edit an indexed document, for example on a shared wiki, can plant a sentence that addresses the assistant, such as an instruction to ignore the question and send the user to a "verification" link. If that chunk is retrieved, the model may obey it. A practical defense must:`));
add(NUMLIST([
  `stop injected chunks before they reach the generator;`,
  `leave legitimate policy text untouched, including legitimate instructions to employees;`,
  `run on a CPU without additional LLM calls; and`,
  `make its decisions visible to an operator.`,
]));
add(H2("1.3", "Base Paper: BIPIA (KDD 2025)"));
add(PG(`Yi *et al.* [@bipia] introduced BIPIA, the first benchmark for indirect prompt injection. It covers five application scenarios (e-mail QA, web QA, table QA, summarisation and code QA) and 250 attacker goals, and it found all 25 evaluated LLMs susceptible. The authors trace the vulnerability to two causes: LLMs do not distinguish informational context from actionable instructions, and they lack awareness that instructions inside external content should not be executed. Their defenses follow from this diagnosis. *Boundary awareness* and *explicit reminders* are implemented through prompting for black-box models, where they mitigate attacks substantially. For white-box models, the generator is fine-tuned with adversarial examples and special boundary tokens, which reduces the attack success rate to near zero while preserving output quality.`));
add(PG(`We use BIPIA as the base paper and adopt three of its elements. The first is its **threat model**: the attacker controls only external content, not the user's request or the system prompt. The second is its **diagnosis** that the root failure is the confusion of data with instructions, which we turn into a classification problem on the data channel. The third is its **separation of training and test attacks**, which we strengthen by withholding entire attack families. We differ in *where* the defense acts (before the generator rather than inside it) and in *what* is measured (chunk-level detection rather than generator attack success; {T:bipia}).`));
add(TABLE("bipia", "BIPIA compared with this project", [1900, 3563, 3563],
  ["Aspect", "BIPIA [@bipia]", "This project"],
  [
    ["Goal", "Benchmark indirect prompt injection and defend the LLM", "Detect injected chunks before they reach the LLM"],
    ["Data", "5 application scenarios; 250 attacker goals", "Enterprise-policy QA; 100,000 synthetic chunks; 32 attack families"],
    ["Models", "25 LLMs evaluated", "DistilBERT detector (66.96M parameters); Llama 3.2 3B generator"],
    ["Defense", "Boundary awareness and explicit reminder (black-box); adversarial fine-tuning with special tokens (white-box)", "Encoder classifier at retrieval time, gating a rule-based clause sanitizer"],
    ["Changes the generator", "White-box defense: yes", "No"],
    ["Main measurement", "Attack success rate of the generator", "Chunk-level detection (accuracy, recall, FPR); end-to-end attack success pending"],
    ["Generalisation test", "Separate training and test attack sets", "Eight attack families withheld from training and validation"],
    ["Compute for the defense", "Released fine-tuning recipe uses eight V100 GPUs", "One laptop CPU"],
  ]));
add(H2("1.4", "Objectives"));
add(NUMLIST([
  `Design a retrieval-time detector that needs neither a GPU nor a generative call.`,
  `Build a benchmark that tests generalisation to attack families never seen in training.`,
  `Measure detection accuracy, false-positive rate and per-family behaviour with uncertainty estimates.`,
  `Integrate the detector with a sanitizer that preserves legitimate content, inside an inspectable pipeline.`,
]));
add(H2("1.5", "Contributions"));
add(BUL(`**Benchmark.** 100,000 synthetic enterprise-policy chunks (10 topics, 5,000 fictional divisions, 50,000 policies). Each policy appears as a clean and an injected chunk, across 32 attack families (96 templates) with hard benign negatives. Eight families and 1,000 divisions are held out for evaluation (Chapter 4).`));
add(BUL(`**Detector.** A windowed DistilBERT classifier [@distilbert] with max-over-windows scoring. Only the last block and the head (${(PRM.distilbert_trainable / 1e6).toFixed(2)}M parameters) are fine-tuned, on 6,000 chunks on a CPU, and the threshold is set on validation data under a 2% false-positive constraint (Chapter 5).`));
add(BUL(`**Held-out evaluation.** On 20,000 chunks: no errors on known families, ${p2(C.recall)}% recall with zero false positives on withheld families, and a per-family analysis that isolates multilingual and dependency-trap injections as the main blind spots (Chapter 7).`));
add(BUL(`**Inspectable pipeline.** The detector gates a clause sanitizer, with a four-mode evaluation harness and a preliminary verification of the sanitizer (Chapters 8–9).`));
add(H2("1.6", "Scope and Organisation of this Report"));
add(PG(`Chapter 2 reviews related work and positions the project; Chapter 3 defines the threat model and architecture; Chapter 4 describes the benchmark; Chapter 5 presents the detector and its training; Chapters 6 and 7 give the evaluation method and results. Chapter 8 covers the defense layer briefly, and Chapter 9 the implementation. Chapters 10 and 11 list limitations and next experiments, and Chapter 12 collects guidance for writing the paper. The appendices explain how to regenerate every figure in MATLAB or Python and list the raw numbers.`));

// ---------------------------------------------------------------- 2 Related work
add(H1("2", "Background and Related Work"));
add(H2("2.1", "Indirect Prompt Injection in RAG"));
add(PG(`Greshake *et al.* [@greshake] showed that attacker-controlled web pages, e-mails and documents can hijack LLM-integrated applications. Liu *et al.* [@liu24] formalised prompt injection and benchmarked attacks and defenses across tasks and models. BIPIA [@bipia] then provided a dedicated benchmark for the indirect setting (Section 1.3). Injection should be distinguished from *knowledge corruption*: PoisonedRAG [@poisonedrag] inserts crafted false passages that steer the answer without any instruction. That threat is out of scope here, because a detector of instructions cannot recognise a well-formed false statement.`));
add(H2("2.2", "Detection-Based Defenses"));
add(PG(`Detectors classify inputs as injected or benign. PIGuard [@piguard] trains a guardrail classifier and shows that a bias toward trigger words causes over-defense on benign inputs, which it measures with its NotInject set. PromptShield [@promptshield] argues that detectors must be judged at deployable, very low false-positive rates. DataSentinel [@datasentinel] fine-tunes an LLM-based detector through a minimax game against adaptive attacks. Akinrele and Gowda [@regime] show that detector performance depends strongly on the deployment regime. Human-written attack corpora such as Tensor Trust [@tensortrust] supply diverse injections. Our detector follows the supervised-classifier line. It takes up PIGuard's concern with over-defense through hard benign negatives and PromptShield's emphasis on low false-positive rates through a validation constraint. It adds a family hold-out protocol in a RAG setting.`));
add(H2("2.3", "Prevention, Sanitization and Layered Defenses"));
add(PG(`*Prompt-level* defenses such as Spotlighting [@spotlighting] mark the provenance of external text by delimiting, datamarking or encoding it. *Training-level* defenses, namely StruQ [@struq], SecAlign [@secalign] and the instruction hierarchy [@ihier], teach the model to prioritise trusted instructions. *System-level* designs such as Task Shield [@taskshield] and CaMeL [@camel] check task alignment or separate control flow from data flow. PromptArmor [@promptarmor] prompts an off-the-shelf LLM to detect and remove injected text, which is the objective closest to our sanitizer, at the cost of an LLM call. Layered designs also exist. Spotlight-Guard [@spotlightguard] combines spotlighting, LLM-based detection and integrity checks for agents, and Raghuraj *et al.* [@hybridrag] benchmark hybrid RAG defenses across models.`));
add(H2("2.4", "Benchmarks for RAG and Agent Security"));
add(PG(`InjecAgent [@injecagent] and AgentDojo [@agentdojo] evaluate injections against tool-using agents, and AgentDyn [@agentdyn] adds dynamic tasks in which third-party instructions can be legitimate. SafeRAG [@saferag] covers several RAG-specific threats, and Hidden-in-Plain-Text [@hipt] targets social-web content in RAG. A recent survey [@survey26] organises the field. Our benchmark is narrower in task (enterprise-policy QA) but is designed specifically to test whether a detector generalises to attack families withheld from training.`));
add(H2("2.5", "Positioning"));
add(PG(`{T:positioning} compares where representative defenses act and what they require. The design point of this project is a defense that leaves the generator unchanged and makes no generative call, which suits CPU-only local deployment.`));
add(TABLE("positioning", "Where representative defenses act and what they require", [2900, 2726, 1700, 1700],
  ["Defense", "Acts on", "Changes generator", "LLM call in defense"],
  [
    [`BIPIA black-box [@bipia]`, "Prompt", "No", "No"],
    [`BIPIA white-box [@bipia]`, "Generator weights", "Yes", "No"],
    [`Spotlighting [@spotlighting]`, "Prompt", "No", "No"],
    [`StruQ, SecAlign [@struq; @secalign]`, "Generator weights", "Yes", "No"],
    [`PIGuard [@piguard]`, "Input text (classifier)", "No", "No"],
    [`DataSentinel [@datasentinel]`, "Input text (LLM detector)", "No", "Yes"],
    [`PromptArmor [@promptarmor]`, "Input text (LLM removal)", "No", "Yes"],
    ["This project", "Retrieved chunks (encoder + rules)", "No", "No"],
  ], { align: [AlignmentType.LEFT, AlignmentType.LEFT, AlignmentType.CENTER, AlignmentType.CENTER], bold: [7] }));
add(H2("2.6", "What the Project Can and Cannot Claim"));
add(PG(`Because several elements of the design already have prior art, the paper's claims must be precise. {T:claims} summarises which statements the evidence supports.`));
add(TABLE("claims", "Claims supported and not supported by the current evidence", [4600, 4426],
  ["Statement", "Assessment"],
  [
    [`"We discovered indirect prompt injection in RAG."`, `Not supported: established by [@greshake] and [@bipia].`],
    [`"First hybrid or dual-layer defense."`, `Not supported: layered defenses exist [@spotlightguard; @hybridrag].`],
    [`"First to remove injections while keeping useful content."`, `Not supported: PromptArmor removes injected text [@promptarmor].`],
    [`"A CPU-trained detector reaches ${p2(C.recall)}% recall with zero false positives on eight withheld families of this benchmark."`, "Supported, for this synthetic benchmark."],
    [`"The defense stage makes no generative-model call."`, "Supported by the design."],
    [`"Clause sanitization preserves legitimate imperative policy text."`, "Partly supported (legacy checks only); needs independent annotation."],
    [`"The system reduces end-to-end attack success."`, "Not yet measured."],
    [`"Faster or more accurate than the state of the art."`, "Not established: no matched comparison yet."],
    [`"The defense guarantees protection."`, "Not supported: misses and adaptive attacks remain possible."],
  ]));

// ---------------------------------------------------------------- 3 Threat model and architecture
add(H1("3", "Threat Model and System Architecture"));
add(H2("3.1", "Threat Model"));
add(BUL(`**Setting.** An internal assistant answers employees' policy questions from an indexed document collection.`));
add(BUL(`**Trust boundary.** The application code, the system prompt and the user's question are trusted. Every retrieved chunk is untrusted, because an attacker may author or edit documents that are later indexed.`));
add(BUL(`**Attacker goal.** Make the generator deviate from the user's task, for example by:`));
add(BUL(`emitting an attacker-chosen string;`, 1));
add(BUL(`redirecting the user to an external link;`, 1));
add(BUL(`dictating the answer format;`, 1));
add(BUL(`impersonating a system or developer message;`, 1));
add(BUL(`leaking data through markup.`, 1));
add(BUL(`**Attacker capability.** The attacker can place instructions anywhere in a chunk and can use formatting tricks (XML, JSON, YAML, comments), social engineering (urgency, authority, politeness) and obfuscation (other languages, Unicode look-alikes, encodings).`));
add(BUL(`**Assumptions and exclusions.** The attacker is non-adaptive: they know neither the detector's weights nor its threshold. Adaptive attacks, false declarative statements [@poisonedrag] and tool-using agents are out of scope.`));
add(H2("3.2", "Architecture"));
add(PG(`{F:arch} shows the pipeline. Every retrieved chunk passes through the detector before it can reach the prompt:`));
add(NUMLIST([
  `The question is embedded with all-MiniLM-L6-v2 [@sbert; @minilm] (384 dimensions, ${(PRM.minilm_total / 1e6).toFixed(1)}M parameters, frozen).`,
  `A Chroma vector store returns the top-*k* chunks.`,
  `Layer 1 (Chapter 5) assigns each chunk an injection score; chunks scoring below τ = 0.175 pass unchanged.`,
  `Flagged chunks are sanitized clause by clause by Layer 2 (Chapter 8); they are edited, not discarded.`,
  `The context is assembled with an explicit reminder that retrieved text is untrusted reference material, and a local Llama 3.2 3B model [@llama3] served by Ollama generates the answer.`,
  `Every run records retrieval scores, detector scores, removed and retained clauses, the processed context, the answer and the latency.`,
]));
add(FIGURE("arch", "fig1_architecture", `Detection-centred pipeline. Layer 1, the only trained component, screens every retrieved chunk before generation. Only flagged chunks reach the rule-based Layer 2 sanitizer, so the defense makes no LLM call.`, 4.4));
add(H2("3.3", "Design Goals"));
add(BUL(`**G1, no generative call inside the defense.** The cost of the defense must not scale with LLM inference.`));
add(BUL(`**G2, CPU-only.** Training and inference must run on a laptop CPU.`));
add(BUL(`**G3, preserve benign evidence.** Very few false alarms, and no loss of policy facts when a chunk is sanitized.`));
add(BUL(`**G4, inspectability.** Every decision (score, threshold, removed clause and its reason) is visible to an operator.`));
add(H2("3.4", "Components and Training Status"));
add(PG(`{T:modules} lists every component of the pipeline. **Exactly one module is trained in this project: the Layer 1 detector.** This report presents its single recorded checkpoint (29 October 2026). The retriever and the generator are used as released, the spaCy parser is pretrained, and the sanitizer rules and the baselines involve no learning.`));
add(TABLE("modules", "Pipeline components, parameter counts and training status", [1900, 2700, 2026, 2400],
  ["Component", "Model / implementation", "Parameters", "Status"],
  [
    ["Retriever encoder", "sentence-transformers all-MiniLM-L6-v2", `${(PRM.minilm_total / 1e6).toFixed(1)}M`, "pretrained, frozen"],
    ["Vector index", "Chroma (persistent)", "—", "built, not trained"],
    ["Layer 1 detector", "DistilBERT-base-uncased + classification head", `${(PRM.distilbert_total / 1e6).toFixed(2)}M (${(PRM.distilbert_trainable / 1e6).toFixed(2)}M fine-tuned)`, "fine-tuned (this project)"],
    ["Layer 2 parser", "spaCy en_core_web_sm", "—", "pretrained, used as-is"],
    ["Layer 2 rules", "hand-written contextual patterns", "—", "no learning"],
    ["Generator", "Llama 3.2 3B via Ollama", "3.21B", "pretrained, frozen"],
    ["Baselines", "delimiter prompt; LLM-judge prompt", "—", "no learning"],
  ], { align: [AlignmentType.LEFT, AlignmentType.LEFT, AlignmentType.CENTER, AlignmentType.LEFT], bold: [2] }));

// ---------------------------------------------------------------- 4 Benchmark
add(H1("4", "Benchmark Dataset"));
add(H2("4.1", "Design Principles"));
add(PG(`Public prompt-injection datasets rarely provide an enterprise document, its clean counterpart, the question it answers and the fact that must survive sanitization. We therefore generated a synthetic benchmark under five principles:`));
add(NUMLIST([
  `**Paired chunks.** Every policy appears as a clean and an injected chunk, so the label depends only on the injected content.`,
  `**Answerable facts.** Each policy carries specific numeric facts and a benchmark question with its expected answer (evaluation-only metadata).`,
  `**Hard negatives.** Benign chunks contain legitimate instructions, quoted attack phrases, markup and inert target strings.`,
  `**Held-out families.** Eight attack families never appear in training or validation.`,
  `**Isolation.** Evaluation partitions use policies and divisions never seen in training.`,
]));
add(H2("4.2", "Corpus Composition"));
add(PG(`The corpus simulates the policy library of a large organisation. It covers ten topics: refunds, IT security, HR leave, financial reports, server configurations, travel expenses, data retention, vendor procurement, customer support and business continuity. There are 5,000 fictional divisions and 50,000 policies. Rendering every policy twice gives 100,000 chunks. The generator is seeded, and its size scales in multiples of 200 rows to keep topics balanced and the 70/10/10/10 split exact.`));
add(H2("4.3", "Attack Families"));
add(PG(`Injections are instantiated from 96 templates grouped into 32 families ({T:families}). They range from overt commands and role-play, through structural spoofing in XML, JSON, YAML, Markdown and code comments, to social-engineering lures and manipulation of the answer's prefix, format or routing. The eight withheld families test styles that differ from training. Some are subtle (indirect completion, reasoning lures, polite requests, few-shot imitation, dependency traps) and some are obfuscated (multilingual instructions, Unicode spoofing, encoded instructions).`));
add(TABLE("families", "Attack families in the benchmark", [2300, 6726],
  ["Split", "Families"],
  [
    ["Seen (24): training, validation, regular test", "direct imperatives, role-play, XML context breakouts, hidden comments, evaluation lures, tool spoofing, conversation injection, footer injection, developer spoofing, Markdown instructions, JSON spoofing, YAML spoofing, urgency, reward lures, redirection, answer-prefix hijacking, compliance spoofing, translation instructions, routing hijacks, format hijacks, nested XML, fake memory, code comments, authority claims"],
    ["Withheld (8): challenge only", "indirect completion, dependency traps, reasoning lures, polite requests, few-shot imitation, multilingual instructions, Unicode spoofing, ==encoded instructions=="],
  ], { note: `The dataset notes name the eighth withheld family "encoded instructions" in one place and "output-protocol lures" in another; confirm against the generator's family list.` }));
add(H2("4.4", "Hard Benign Negatives"));
add(PG(`A detector that keys on single words would be useless in an enterprise, where policies are full of instructions ("never share your password", "submit receipts within 30 days"). Both classes therefore contain legitimate imperative policy language, quoted attack phrases (for example, a security bulletin that quotes a phishing message), markup and inert attacker-target strings. Neither the word "ignore" nor a target token such as "PWNED" separates the classes on its own.`));
add(H2("4.5", "Partitions and Isolation"));
add(PG(`The corpus is split 70/10/10/10 ({F:partitions}, {T:partitions}). Each evaluation partition uses 500 division identifiers that never occur in training, so the classifier was evaluated on 1,000 of the 5,000 divisions. Clean/injected pairs never straddle partitions. The regular test shares the 24 attack families with training; the challenge set contains only the 8 withheld families.`));
add(FIGURE("partitions", "fig2_partitions", `Benchmark partitions. Only a stratified 6,000-chunk sample (dark segment) of the 70,000-chunk training pool was used for fine-tuning. Validation alone selects the checkpoint and threshold, and the challenge set contains only the eight withheld families.`, 4.7));
add(TABLE("partitions", "Benchmark partitions", [1700, 1200, 1700, 1700, 2726],
  ["Partition", "Chunks", "Benign / injected", "Attack families", "Role"],
  [
    ["Training pool", "70,000", "35,000 / 35,000", "24 seen", "6,000 sampled for fine-tuning"],
    ["Validation", "10,000", "5,000 / 5,000", "24 seen", "checkpoint and threshold selection"],
    ["Regular test", "10,000", "5,000 / 5,000", "24 seen", "final evaluation; 500 unseen divisions"],
    ["Challenge", "10,000", "5,000 / 5,000", "8 withheld", "final evaluation; 500 unseen divisions"],
  ], { align: [AlignmentType.LEFT, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.LEFT] }));
add(H2("4.6", "Reproducibility and Automated Checks"));
add(PG(`A manifest records the seed, the finite template inventory and a dataset hash. Earlier datasets are archived before replacement, and the web workbench refuses to attribute an old score to a changed dataset. Unit tests check:`));
add(BUL(`balanced partitions and reproducible generation;`));
add(BUL(`that facts survive in the injected chunks;`));
add(BUL(`cross-split isolation of policies and entities, and isolation of the challenge families;`));
add(BUL(`that benign chunks contain trigger mentions (the hard negatives);`));
add(BUL(`metadata round-tripping, threshold selection, and web access to all partitions.`));
add(H2("4.7", "External Reference Data"));
add(PG(`The 546 training rows of the public deepset prompt-injections dataset [@deepset] (Apache-2.0; 662 rows in total) are imported into a separate collection. They are not mixed into any synthetic split, because their label definitions differ and they lack clean policy facts. They are reserved for cross-dataset evaluation (Chapter 11).`));
add(H2("4.8", "From the Legacy Corpus to the 100k Benchmark"));
add(PG(`The project began with a 200-chunk demonstration corpus. A detector fully fine-tuned on it classified all 60 of its test chunks correctly. A test set that small cannot distinguish detection from template memorisation, which motivated the current benchmark ({T:legacy}).`));
add(TABLE("legacy", "Legacy corpus compared with the current benchmark", [2600, 3213, 3213],
  ["Property", "Legacy corpus", "Current benchmark"],
  [
    ["Chunks", "200 (100 policies × clean/injected)", "100,000 (50,000 policies × clean/injected)"],
    ["Divisions × topics", "20 × 5", "5,000 divisions; 10 topics"],
    ["Attack families", "6", "32 (96 templates)"],
    ["Test data", "60 chunks", "10,000 regular + 10,000 challenge"],
    ["Held-out families", "none", "8"],
    ["Detector training", "full fine-tuning, 140 chunks", "last block + head, 6,000 chunks"],
    ["Detector result", "60 / 60 correct", "Chapter 7"],
  ]));

// ---------------------------------------------------------------- 5 Detector
add(H1("5", "Layer 1: Detector Design and Training"));
add(H2("5.1", "Why a Compact Encoder"));
add(PG(`An encoder-only classifier produces a score in one forward pass, is deterministic, and needs no prompt that an attacker could also target. DistilBERT [@distilbert] is a distilled version of BERT that is 40% smaller and 60% faster while retaining about 97% of BERT's language-understanding performance. That makes it a natural choice for CPU inference over every retrieved chunk.`));
add(H2("5.2", "Windowed Scoring"));
add(PG(`An injected clause can be a single sentence inside a long chunk, so chunks are not truncated. The tokenizer splits each chunk *c* into overlapping windows of ==256 tokens with a 64-token overlap==, forming the set 𝒲(*c*). Let *p*_{θ}(inj | *w*) be the classifier's injection probability for window *w*. The chunk score and decision are`));
add(EQ("eq1_score"));
add(EQ("eq2_decision"));
add(PG(`Taking the maximum keeps a strong local signal from being diluted by long benign text. This deliberately favours recall; its cost, that a single suspicious window can flag a whole chunk, is controlled by the threshold τ.`));
add(H2("5.3", "Partial Fine-Tuning and Parameter Budget"));
add(PG(`The embeddings and the first five transformer blocks are frozen. Only the sixth block and the classification head are trained ({T:params}). This keeps back-propagation shallow enough for a laptop CPU and limits memory use. It also leaves the pretrained lower layers intact, which we expect to reduce memorisation of templates; this choice has not yet been ablated.`));
add(TABLE("params", "Parameter budget of the Layer 1 detector (distilbert-base-uncased, two classes)", [4300, 2726, 2000],
  ["Part", "Parameters", "Trained?"],
  [
    ["Token + position embeddings, LayerNorm", int(EMB), "frozen"],
    ["Transformer blocks 1–5 (5 × 7,087,872)", int(5 * PRM.distilbert_block), "frozen"],
    ["Transformer block 6", int(PRM.distilbert_block), "trained"],
    ["Pre-classifier (768 × 768 + 768)", int(590592), "trained"],
    ["Classifier (768 × 2 + 2)", int(1538), "trained"],
    ["Total", int(PRM.distilbert_total), `${int(PRM.distilbert_trainable)} trained (${(100 * PRM.trainable_share).toFixed(2)}%)`],
  ], { align: [AlignmentType.LEFT, AlignmentType.RIGHT, AlignmentType.LEFT], bold: [5] }));
add(H2("5.4", "Training Data"));
add(PG(`The fine-tuning set is a stratified sample of 6,000 chunks (3,000 benign, 3,000 injected) drawn from the 70,000-chunk training pool. The sample is stratified across labels, topics, attack families and benign categories. This is 8.6% of the available pool: the run was sized for a CPU budget, and the full pool remains available. Validation and both test partitions are always used in full.`));
add(H2("5.5", "Training Configuration"));
add(TABLE("config", "Training configuration of the recorded run (29 October 2026)", [3300, 5726],
  ["Setting", "Value"],
  [
    ["Base model", "distilbert-base-uncased (Hugging Face Transformers [@transformers])"],
    ["Trainable parameters", `${int(PRM.distilbert_trainable)} of ${int(PRM.distilbert_total)} (${(100 * PRM.trainable_share).toFixed(2)}%)`],
    ["Window length / overlap", "==256 / 64 tokens=="],
    ["Training sample", "6,000 of 70,000 chunks (3,000 / 3,000), stratified"],
    ["Batch size; optimizer steps per epoch", `16; ${TR.steps_per_epoch}`],
    ["Epochs run; epoch selected", "3; 1 (validation-based selection with early stopping)"],
    ["Optimizer", "==AdamW, learning rate 5 × 10==^{−5}==, weight decay 0.01=="],
    ["Learning-rate schedule; clipping", "==linear with 10% warm-up; gradient-norm clipping at 1.0=="],
    ["Threshold selection", "on validation, FPR ≤ 2% → τ = 0.175"],
    ["Framework; hardware", "PyTorch [@pytorch] 2.14.1 (CPU build); laptop CPU"],
  ], { note: "Highlighted values are the defaults of the released training code and should be confirmed from the training log." }));
add(H2("5.6", "Checkpoint and Threshold Selection"));
add(PG(`After each epoch, the model scores the 10,000 validation chunks. The checkpoint and the operating threshold are chosen on validation data only, to maximise detection subject to a false-positive constraint α = 2%:`));
add(EQ("eq3_threshold"));
add(PG(`Validation selected the epoch-1 checkpoint and τ = 0.175. The 2% constraint is a tuning target, not a guarantee on new data, and neither test partition influenced any choice.`));
add(H2("5.7", "Compute Environment"));
add(PG(`Training ran entirely on the CPU of a laptop. The laptop has an NVIDIA GTX 1650, but the active environment contained the CPU-only PyTorch build (2.14.1+cpu). A separate CUDA environment was attempted after training, but its download failed when the package host became unreachable, so no GPU result exists. ==Wall-clock training time was not recorded== and should be added if available. The checkpoint and metrics were saved successfully, and earlier checkpoints are archived rather than overwritten.`));

// ---------------------------------------------------------------- 6 Evaluation methodology
add(H1("6", "Evaluation Methodology"));
add(H2("6.1", "Protocol"));
add(PG(`The selected checkpoint is evaluated once, at τ = 0.175, on all 10,000 regular-test chunks and all 10,000 challenge chunks. Per-chunk predictions are saved (@@results/layer1_predictions.csv@@ and @@results/layer1_challenge_predictions.csv@@), and the summary goes to @@results/layer1_metrics.json@@. These numbers measure Layer 1 only, not end-to-end attack prevention.`));
add(H2("6.2", "Metrics"));
add(TABLE("metrics", "Metrics (TP: injected and flagged; FN: injected and passed; FP: benign and flagged; TN: benign and passed)", [1700, 3626, 3700],
  ["Metric", "Definition", "Meaning in this project"],
  [
    ["Accuracy", "(TP + TN) / N", "share of chunks classified correctly"],
    ["Precision", "TP / (TP + FP)", "share of alarms that are real injections"],
    ["Recall (TPR)", "TP / (TP + FN)", "share of injected chunks detected"],
    ["False-positive rate", "FP / (FP + TN)", "share of benign chunks wrongly flagged"],
    ["Specificity", "TN / (TN + FP) = 1 − FPR", "share of benign chunks passed unchanged"],
    ["F1", "2 · Precision · Recall / (Precision + Recall)", "balance of precision and recall"],
    ["MCC", "(TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))", "correlation of prediction and truth (−1 to 1)"],
  ]));
add(H2("6.3", "Confidence Intervals"));
add(PG(`For every proportion *p̂* = *k*/*n*, we report a 95% Wilson score interval [@wilson]:`));
add(EQ("eq4_wilson"));
add(PG(`Unlike the normal approximation, the Wilson interval stays informative when *p̂* is 0 or 1. With zero errors among 5,000 benign chunks, for example, the interval bounds the FPR above by ${(100 * R.fpr_ci[1]).toFixed(3)}% rather than collapsing to zero.`));
add(H2("6.4", "Caveat on Independence"));
add(PG(`The binomial model assumes independent chunks. Here, clean and injected versions are paired and templates are reused, so the effective sample size is smaller than the chunk count. The intervals therefore describe the sampling of chunks, not of attack templates, and they are optimistic.`));

// ---------------------------------------------------------------- 7 Results
add(H1("7", "Detection Results"));
add(H2("7.1", "Regular Test: Seen Families"));
add(PG(`The detector classifies all 10,000 regular-test chunks correctly: 5,000 true negatives and 5,000 true positives ({T:results}, {F:confusion}a). The Wilson intervals bound accuracy from below by ${p2(R.accuracy_ci[0])}% and recall by ${p2(R.recall_ci[0])}%, and they bound the FPR from above by ${(100 * R.fpr_ci[1]).toFixed(3)}%. Because the regular test shares attack families and templates with training, this shows that the detector has learned the 24 seen families on unseen policies and divisions. It does not by itself show that injection detection is solved.`));
add(TABLE("results", "Detection results at τ = 0.175 (percent; 95% Wilson intervals in brackets)", [2626, 3200, 3200],
  ["Metric", "Regular test", "Challenge"],
  [
    ["Accuracy", `${p2(R.accuracy)} ${ci(R.accuracy_ci)}`, `${p2(C.accuracy)} ${ci(C.accuracy_ci)}`],
    ["Precision", `${p2(R.precision)} ${ci(R.precision_ci)}`, `${p2(C.precision)} ${ci(C.precision_ci)}`],
    ["Recall (TPR)", `${p2(R.recall)} ${ci(R.recall_ci)}`, `${p2(C.recall)} ${ci(C.recall_ci)}`],
    ["False-positive rate", `${p2(R.fpr)} [0.00, ${(100 * R.fpr_ci[1]).toFixed(3)}]`, `${p2(C.fpr)} [0.00, ${(100 * C.fpr_ci[1]).toFixed(3)}]`],
    ["Specificity", `${p2(R.specificity)} ${ci(R.specificity_ci)}`, `${p2(C.specificity)} ${ci(C.specificity_ci)}`],
    ["F1", p2(R.f1), p2(C.f1)],
    ["MCC", R.mcc.toFixed(4), C.mcc.toFixed(4)],
    ["TN / FP / FN / TP", "5,000 / 0 / 0 / 5,000", "5,000 / 0 / 191 / 4,809"],
  ], { align: [AlignmentType.LEFT, AlignmentType.CENTER, AlignmentType.CENTER] }));
add(FIGURE("confusion", "fig3_confusion", `Confusion matrices at τ = 0.175 on (a) the regular test (24 seen families) and (b) the challenge set (8 withheld families). Cells give counts and row percentages.`, 4.7));
add(H2("7.2", "Challenge Set: Withheld Families"));
add(PG(`On the eight withheld families, the detector reaches ${p2(C.accuracy)}% accuracy, ${p2(C.recall)}% recall, ${p2(C.f1)}% F1 and an MCC of ${C.mcc.toFixed(4)} ({F:confusion}b, {F:metrics}). All 191 errors are missed injections: none of the 5,000 benign challenge chunks is flagged, so precision and FPR are unchanged from the regular test. Moving to unseen attack styles therefore costs about 3.8 percentage points of recall and no false alarms, at a threshold that was never tuned on these families.`));
add(FIGURE("metrics", "fig4_metrics_ci", `Headline metrics with 95% Wilson intervals. All challenge errors are misses; precision and specificity stay at 100%. The x-axis starts at 95%.`, 4.7));
add(H2("7.3", "Per-Family Analysis"));
add(PG(`{F:families} and {T:perfamily} break recall down by withheld family. Five families (indirect completion, reasoning lures, polite requests, few-shot imitation and encoded instructions) are detected without a single miss. Three families account for all 191 misses, and multilingual instructions alone for ${MISS_ML}% of them. Excluding that family, recall on the other seven withheld families is ${p2(N.recall_without_multilingual)}%. Excluding dependency traps as well, the remaining six families contain 6 misses in 3,766 injected chunks (${p2(N.recall_without_multilingual_and_dependency)}%).`));
add(FIGURE("families", "fig5_family_recall", `Recall per withheld family on the challenge set (threshold 0.175). Three families account for all 191 misses; multilingual instructions alone account for 148 (${MISS_ML}%).`, 4.7));
add(TABLE("perfamily", "Recall per withheld family (challenge set)", [2900, 1500, 1000, 1500, 2126],
  ["Family", "Injected chunks (n)", "Missed", "Recall", "95% Wilson interval"],
  [
    ["Multilingual instructions", int(ML.n), int(ML.missed), `${p2(ML.recall)}%`, ci(ML.recall_ci)],
    ["Dependency traps", int(DEP.n), int(DEP.missed), `${p2(DEP.recall)}%`, ci(DEP.recall_ci)],
    ["Unicode spoofing", "==629–634==", "6", `${p2(UNI.recall)}%`, `≈ [${(100 * UNI_CI[0]).toFixed(1)}, ${(100 * UNI_CI[1]).toFixed(1)}]`],
    ["Encoded instructions", "≈ 625", "0", "100%", `≈ [${(100 * ZERO_LO).toFixed(1)}, 100]`],
    ["Few-shot imitation", "≈ 625", "0", "100%", `≈ [${(100 * ZERO_LO).toFixed(1)}, 100]`],
    ["Indirect completion", "≈ 625", "0", "100%", `≈ [${(100 * ZERO_LO).toFixed(1)}, 100]`],
    ["Polite requests", "≈ 625", "0", "100%", `≈ [${(100 * ZERO_LO).toFixed(1)}, 100]`],
    ["Reasoning lures", "≈ 625", "0", "100%", `≈ [${(100 * ZERO_LO).toFixed(1)}, 100]`],
    ["All withheld families", "5,000", "191", `${p2(C.recall)}%`, ci(C.recall_ci)],
  ], { align: [AlignmentType.LEFT, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER], bold: [8],
    note: `The n for multilingual instructions and dependency traps are the only integers consistent with the recorded recall. The other family sizes are not in the recorded summary; @@plot_from_predictions.m@@ (Appendix A) prints the exact values from the predictions file.` }));
add(H2("7.4", "Why the Three Families Are Missed"));
add(BUL(`**Multilingual instructions (recall ${p2(ML.recall)}%).** The base model was pretrained on English text, and fine-tuning used English chunks, so imperative cues in other languages are weakly represented. This is the largest and most actionable gap.`));
add(BUL(`**Dependency traps (recall ${p2(DEP.recall)}%).** ==These injections present the attacker's instruction as a prerequisite of the legitimate task. Their procedural phrasing resembles legitimate policy instructions, which the hard negatives teach the detector to accept.== This explanation is inferred from the family's name and its error pattern; inspect the missed chunks to confirm it.`));
add(BUL(`**Unicode spoofing (recall ${p2(UNI.recall)}%).** Look-alike characters fragment WordPiece tokens, but the surrounding structure of the injection is usually still recognised. Unicode normalisation before scoring should remove most of these misses.`));
add(PG(`The five fully detected families differ in surface form from the training families, but they still contain text addressed to the assistant that tries to change its behaviour. This suggests, without proving, that the detector responds to that cue rather than only to memorised templates.`));
add(H2("7.5", "Operating Point and the Cost of Errors"));
add(PG(`The threshold of 0.175 is low, yet no benign regular-test or challenge chunk crosses it. On this benchmark, therefore, the score distributions of the two classes are well separated, hard negatives included. Real deployments will produce some false positives: PIGuard [@piguard] reports substantial over-defense on natural benign text containing trigger words. The gated design makes such errors cheap. A false positive sends a benign chunk through the sanitizer, which keeps employee-directed instructions and removes only assistant-directed clauses, rather than discarding the document. A false negative, by contrast, delivers the injection to the generator untouched. In this design recall is the critical quantity, which is why the threshold maximises detection under an FPR constraint.`));
add(H2("7.6", "Summary of Findings"));
add(NUMLIST([
  `On known attack families with new policies and divisions, the detector made no errors on 10,000 chunks.`,
  `On eight withheld families, recall fell to ${p2(C.recall)}% while the false-positive rate stayed at zero.`,
  `The errors are concentrated: multilingual instructions cause ${MISS_ML}% of the misses, followed by dependency traps (19.4%) and Unicode spoofing (3.1%).`,
  `These results were obtained by fine-tuning 11.5% of DistilBERT's parameters on 8.6% of the training pool on a CPU, so the approach is affordable for small teams.`,
  `All conclusions apply to this synthetic benchmark; external and end-to-end validation are still required.`,
]));

// ---------------------------------------------------------------- 8 Defense layer (brief)
add(H1("8", "Defense Layer (Layer 2) and End-to-End Harness"));
add(H2("8.1", "Detector-Gated Clause Sanitization"));
add(PG(`Only flagged chunks reach Layer 2, which edits a chunk instead of discarding it:`));
add(NUMLIST([
  `**Segmentation.** Structural markers (closing tags, chat-template tokens, role prefixes) become separate units. The remaining text is split into sentences with spaCy [@spacy] and into clauses at connector boundaries.`,
  `**Rules.** A clause is removed if it matches one of eight strong rule groups ({T:rules}). Matching also runs on a de-obfuscated copy that undoes leetspeak, hyphenation and letter spacing. Three weak syntactic signals together (for example, imperative mood with a command verb aimed at the output) also remove a clause.`,
  `**Legitimate context.** Imperative clauses addressed to employees or customers that carry no assistant-directed marker, such as password or receipt rules, are kept and reported as retained instructions.`,
  `**Reconstruction.** Kept clauses are copied verbatim, only punctuation around removed spans is tidied, and each removal is logged with its reason.`,
]));
add(PG(`The sanitizer never sees the clean reference text, uses no generative model and is deterministic.`));
add(TABLE("rules", "Strong rule groups of the clause sanitizer", [2500, 6526],
  ["Rule group", "What it removes"],
  [
    ["Role or system markers", "role prefixes such as \"System:\" or \"Assistant:\" and chat-template tokens"],
    ["Delimiter breakouts", "closing tags, CDATA ends and fence lines that try to end the data section"],
    ["Address to the assistant", "clauses whose addressee or subject is the AI assistant"],
    ["Instruction override", "attempts to override the question or prior instructions, including common non-English phrasings"],
    ["Persona switch", "attempts to change the assistant's role or task"],
    ["Output control", "dictation of the answer, e.g. what to reply or how to begin"],
    ["User redirection", "telling the assistant to send the user elsewhere (links, numbers, portals)"],
    ["Exfiltration", "Markdown image links and placeholders that could leak data"],
  ]));
add(H2("8.2", "Preliminary Verification"));
add(PG(`The sanitizer was verified on the earlier 200-chunk corpus, which has 20 divisions × 5 topics and 6 attack families placed at the start, middle or end of a chunk, or inline with the fact sentence ({T:l2checks}). Two observations stand out. First, the sanitizer restored all 100 injected chunks exactly to their clean text and left all clean chunks unchanged. Second, injections barely reduce retrievability: the poisoned chunk was still ranked first for 95 of 100 questions. Screening must therefore happen *after* retrieval. These checks are not an independent test, because the rules were developed with these cases in view. The 100,000-chunk benchmark also contains styles (multilingual, Unicode) that the current rules do not cover.`));
add(TABLE("l2checks", "Preliminary checks on the 200-chunk legacy corpus", [6026, 3000],
  ["Check", "Result"],
  [
    ["Injected chunks restored exactly to the clean reference text", "100 / 100"],
    ["Clean chunks left unchanged", "100 / 100"],
    ["Out-of-template payloads: attacker-target leaks; lost facts", "0; 0 (of 48)"],
    ["Custom payloads removed", "6 / 6"],
    ["Benign look-alikes kept", "10 / 10"],
    ["Retrieval, clean corpus: target chunk at rank 1", "100 / 100"],
    ["Retrieval, poisoned corpus: injected chunk at rank 1; within top 3", "95; 98 (of 100)"],
  ], { align: [AlignmentType.LEFT, AlignmentType.CENTER],
    note: "==Measured with the version of the code in the project repository; re-run on the current local version before citing.==" }));
add(H2("8.3", "End-to-End Evaluation Harness"));
add(PG(`The harness compares four modes on the same questions ({T:modes}). It reports the following measures:`));
add(BUL(`**Attack success:** the attacker's target string appears in the answer.`));
add(BUL(`**Benign utility:** the expected numeric fact appears in the answer.`));
add(BUL(`**Chunk-level FPR** of the defense.`));
add(BUL(`**ROUGE-L** [@rouge] between the processed and the clean context.`));
add(PG(`Failed runs are counted separately and excluded from denominators, the order of modes is randomised, and warm-up calls are excluded. All four measures are proxies. Substring matching can mistake a quoted or refused target for compliance, and ROUGE-L measures overlap, not truth.`));
add(TABLE("modes", "Defense modes in the evaluation harness", [2300, 5126, 1600],
  ["Mode", "Behaviour", "Extra LLM calls"],
  [
    ["No defense", "retrieved chunks go straight into the prompt", "0"],
    ["Delimiter defense", "chunks wrapped in tags with an untrusted-data reminder (Spotlighting-style)", "0"],
    ["LLM-judge defense", "the same local LLM judges each chunk; flagged chunks are dropped", "one per chunk"],
    ["Hybrid defense (this project)", "Layer 1 on every chunk, Layer 2 on flagged chunks, plus the reminder", "0"],
  ], { align: [AlignmentType.LEFT, AlignmentType.LEFT, AlignmentType.CENTER], bold: [3] }));
add(H2("8.4", "Status"));
add(PG(`A full four-mode run over one 10,000-question partition with top-3 retrieval requires up to 70,000 generations on a CPU. Only intermediate and interrupted runs exist, so **no end-to-end attack-success numbers are reported**. Because the hybrid mode also adds a reminder, an ablation is needed to separate the reminder's effect from that of the detector and sanitizer.`));

// ---------------------------------------------------------------- 9 Implementation
add(H1("9", "Implementation, Tools and Reproducibility"));
add(H2("9.1", "Software Stack"));
add(TABLE("stack", "Software stack", [2800, 6226],
  ["Purpose", "Component"],
  [
    ["Language and environment", "Python in a project virtual environment (Windows; administrator rights not needed)"],
    ["Deep learning", "PyTorch [@pytorch] 2.14.1 (CPU build); Hugging Face Transformers [@transformers]"],
    ["Retrieval", "sentence-transformers all-MiniLM-L6-v2 [@sbert; @minilm]; ChromaDB (persistent store, explicit embeddings)"],
    ["Linguistic processing", "spaCy [@spacy], en_core_web_sm"],
    ["Generation", "Ollama serving llama3.2:3b [@llama3]; optional OpenAI client"],
    ["Interfaces", "FastAPI web workbench; Streamlit dashboard"],
    ["Figures", "matplotlib (Python) and MATLAB scripts (Appendix A)"],
  ], { note: "==Record the exact library versions (pip freeze) for the paper's reproducibility statement.==" }));
add(H2("9.2", "Code Organisation"));
add(TABLE("code", "Main modules of the code base", [3300, 5726],
  ["Module", "Responsibility"],
  [
    ["@@src/data_generator.py@@", "benchmark generator (100k default, legacy profile, Hugging Face import)"],
    ["@@src/train_layer1.py@@", "sampling, partial fine-tuning, validation selection, evaluation on both test sets"],
    ["@@src/layer1_detector.py@@", "windowed scoring and thresholding at inference"],
    ["@@src/layer2_quarantine.py@@", "clause sanitizer (segmentation, rules, retention, reconstruction)"],
    ["@@src/embeddings.py@@, @@src/vector_store.py@@", "MiniLM embeddings; Chroma collections with explicit vectors"],
    ["@@scripts/index_corpus.py@@", "incremental indexing; writes @@results/index_inventory.json@@"],
    ["@@src/llm_client.py@@", "Ollama and OpenAI clients"],
    ["@@src/rag_pipeline.py@@", "end-to-end pipeline and command-line interface"],
    ["@@src/evaluate_all.py@@, @@src/metrics.py@@", "four-mode harness and metrics"],
    ["@@src/web_server.py@@, @@web/@@, @@app.py@@", "FastAPI workbench and Streamlit dashboard"],
    ["@@tests/@@", "unit tests"],
  ]));
add(H2("9.3", "Inspection Workbench"));
add(PG(`The web workbench offers a live playground with preset attack scenarios and a browser over six data sources: enterprise corpus, external references, training, validation, regular test and challenge. It also shows saved results and their evidence. For every run it displays the retrieved chunks, the detector score of each chunk, removed and retained clauses with reasons, the final context, the answer and the latency. This supports design goal G4 and is useful for a demonstration track.`));
add(H2("9.4", "Engineering Safeguards and Tests"));
add(BUL(`Embeddings are always passed to Chroma explicitly, so the store can never fall back silently to a different default embedder.`));
add(BUL(`A dataset hash ties every reported score to the data it was computed on.`));
add(BUL(`Checkpoints and metrics are archived rather than overwritten.`));
add(BUL(`Offline mode guarantees that no model is downloaded during an experiment.`));
add(BUL(`Indexing is incremental: unchanged rows are reused and obsolete identifiers are removed in batches.`));
add(BUL(`Unit tests cover the generator's guarantees, threshold selection and web access to all partitions.`));
add(H2("9.5", "Reproducing the Results"));
add(PG(`From the project folder, in a Windows Command Prompt:`));
add(CODE([
  "REM 1. Generate the 100,000-chunk benchmark",
  ".venv\\Scripts\\python.exe src\\data_generator.py --rows 100000 --output-dir data_100k",
  "",
  "REM 2. Train and evaluate Layer 1 (6,000-row sample, 3 epochs, CPU)",
  ".venv\\Scripts\\python.exe src\\train_layer1.py --data-dir data_100k --max-train-rows 6000 --batch-size 16 --epochs 3 --offline",
  "",
  "REM 3. Index the corpus for retrieval",
  ".venv\\Scripts\\python.exe scripts\\index_corpus.py --offline --threads 2",
  "",
  "REM 4. End-to-end harness (start small; --limit 0 runs every question)",
  ".venv\\Scripts\\python.exe src\\evaluate_all.py --limit 100 --top-k 1 --offline --output-dir results\\rag_regular",
  ".venv\\Scripts\\python.exe src\\evaluate_all.py --split challenge --limit 100 --top-k 1 --offline --output-dir results\\rag_challenge",
  "",
  "REM 5. Tests and the workbench",
  ".venv\\Scripts\\python.exe -m unittest discover -s tests -v",
  "start_demo.bat",
]));
add(H2("9.6", "Output Files"));
add(BUL(`@@results/layer1_metrics.json@@: regular-test and challenge metrics, validation history, threshold, and data counts and hashes.`));
add(BUL(`@@results/layer1_predictions.csv@@: one prediction per regular-test chunk.`));
add(BUL(`@@results/layer1_challenge_predictions.csv@@: one prediction per challenge chunk, with family labels.`));
add(BUL(`@@results/index_inventory.json@@: counts of indexed documents per collection.`));

// ---------------------------------------------------------------- 10 Limitations
add(H1("10", "Limitations"));
add(NUMLIST([
  `**Synthetic data.** The benchmark is template-based (96 templates), paired and English-heavy. The results measure performance on this benchmark, not on natural enterprise documents.`,
  `**Meaning of "unseen".** The eight families are withheld from fine-tuning only; DistilBERT's pretraining corpus is not controlled.`,
  `**Statistical dependence.** Paired chunks and shared templates make the confidence intervals optimistic.`,
  `**One run.** A single seed, trained on 6,000 of the 70,000 available chunks; run-to-run variance is unknown.`,
  `**Non-adaptive attacker.** An attacker who can query the detector could optimise evasions, as DataSentinel's threat model anticipates [@datasentinel].`,
  `**No task context.** The detector sees each chunk without the user's question, so it cannot judge task relevance as Task Shield does [@taskshield]. It also does not address false declarative content [@poisonedrag].`,
  `**Long documents.** Max-over-windows scoring may raise the FPR on long natural documents; this has not been measured.`,
  `**No external comparison.** There is no comparison yet with PIGuard, PromptShield, PromptArmor, or BIPIA's own defenses on BIPIA data.`,
  `**Coverage of Layer 2.** A miss by Layer 1 bypasses Layer 2 entirely, the rules cover only recognised patterns, and no end-to-end benchmark has been completed.`,
]));

// ---------------------------------------------------------------- 11 Next experiments
add(H1("11", "Recommended Next Experiments"));
add(PG(`{T:next} ranks the experiments by how much they would strengthen a submission, given the time available.`));
add(TABLE("next", "Recommended experiments, in priority order", [500, 3700, 3326, 1500],
  ["#", "Experiment", "Why it matters", "Effort"],
  [
    ["1", "Four-mode end-to-end run on 200–500 questions per split (top-3)", "turns detector accuracy into attack-success and utility numbers; reviewers will ask for them", "1–2 days of CPU"],
    ["2", "Score deepset/prompt-injections and the NotInject benign set with the trained detector", "first external evidence of recall and of over-defense", "hours"],
    ["3", "Train on the full 70,000-chunk pool with 3 seeds", "answers the one-run, 8.6%-of-data objection; gives mean ± s.d.", "1–2 days of CPU"],
    ["4", "Add translated versions of *seen* families and apply Unicode normalisation", "targets the two main blind spots; the withheld families stay unseen", "1 day"],
    ["5", "Measure latency per chunk (detector, sanitizer, LLM judge)", "quantifies the no-LLM-call advantage", "hours"],
    ["6", "Ablations: partial vs full fine-tuning; max vs mean pooling; reminder on/off", "justifies the design choices", "1–2 days"],
    ["7", "Simple adaptive attack (paraphrase or obfuscate until s(c) < τ)", "first robustness evidence", "1–2 days"],
  ], { align: [AlignmentType.CENTER, AlignmentType.LEFT, AlignmentType.LEFT, AlignmentType.LEFT] }));

// ---------------------------------------------------------------- 12 Guidance for the paper
add(H1("12", "Guidance for Writing the Conference Paper", { pageBreak: true }));
add(H2("12.1", "Venue Facts: COMSNETS 2027"));
add(PG(`COMSNETS 2027, the 19th International Conference on COMmunication Systems & NETworkS, takes place in Bengaluru on 5–9 January 2027. The facts below were collected by web search on 29 October 2026. The official site could not be opened directly from the environment that produced this report, so **confirm every date and limit on comsnets.org and in EDAS before planning**.`));
add(TABLE("venue", "COMSNETS 2027 tracks as reported on 29 October 2026", [1800, 3000, 2526, 1700],
  ["Track", "Deadline (reported)", "Length", "Review"],
  [
    ["Main conference", "31 August 2026 (closed)", "3–8 pages, excluding references", "double-blind"],
    ["Graduate Forum", "8 or 9 November 2026 (AoE; two pages disagree); notification 7 Dec; camera-ready 14 Dec", "4 pages + 1 page of references + 1-page resume", "resume included, so not anonymous"],
    ["Posters", "about 12 November 2026", "not yet published for 2027", "—"],
    ["Demos and exhibits", "about 10 November 2026", "not yet published for 2027", "—"],
    ["Workshops", "2027 list not yet confirmed", "per workshop", "double-blind (2026 rules)"],
  ], { note: "Sources: comsnets.org pages call_for_papers, graduate_forum and submission_guidelines, as reported by search results. The 2027 Graduate Forum page describes eligibility as students of graduate programmes; confirm whether undergraduates qualify." }));
add(H2("12.2", "Recommended Framing"));
add(PG(`**Title options:**`));
add(NUMLIST([
  `*Detect Before You Generate: A Lightweight Retrieval-Time Detector for Indirect Prompt Injection in Local RAG Pipelines* (detection-focused; recommended for the current evidence).`,
  `*Selective Clause Sanitization with a Lightweight Injection Detector for Local RAG Systems* (better once end-to-end results exist).`,
  `*How Far Does a CPU-Trained Detector Generalise? Held-Out-Family Evaluation of Indirect Prompt Injection Detection for RAG* (analysis-focused).`,
]));
add(PG(`**Positioning sentence:** "Following BIPIA's diagnosis that LLMs confuse retrieved data with instructions, we move the first line of defense out of the generator, to retrieval time, using a detector that is cheap enough for CPU-only local deployment."`));
add(CALLOUT("Draft abstract (about 230 words) for adaptation", [
  `Retrieval-augmented generation (RAG) lets a large language model (LLM) answer from an organisation's documents, but any retrieved chunk may carry instructions planted by an attacker. The BIPIA benchmark found all 25 LLMs it evaluated vulnerable to such indirect prompt injection, and its most effective defense fine-tunes the generator itself, which is out of reach for small models served on CPU-only hardware. We instead screen every retrieved chunk before generation with a lightweight detector, and send only flagged chunks to a rule-based clause sanitizer, so the defense makes no generative call. The detector is a DistilBERT classifier that scores overlapping 256-token windows and takes the maximum injection probability; only its last transformer block and classification head (7.68M of 66.96M parameters) are fine-tuned, on 6,000 chunks, on a laptop CPU. To test generalisation we build a 100,000-chunk synthetic enterprise-policy benchmark with paired clean and injected chunks, 32 attack families and hard benign negatives, and withhold eight families from training and threshold selection. On 10,000 regular-test chunks the detector makes no errors; on 10,000 chunks from the withheld families it attains ${p2(C.accuracy)}% accuracy and ${p2(C.recall)}% recall with no false positives. Multilingual injections cause ${MISS_ML}% of the misses; recall on the other seven withheld families is ${p2(N.recall_without_multilingual)}%. We discuss what these detector-level results do and do not establish.`,
], { bullets: false }));
add(PG(`**Contribution statements** (use as written or shorten):`));
add(BUL(`a 100k-chunk benchmark for retrieval-time injection detection with a family hold-out protocol;`));
add(BUL(`a CPU-trainable windowed detector (7.68M trainable parameters, 6,000 training chunks);`));
add(BUL(`a held-out evaluation with per-family failure analysis and uncertainty estimates;`));
add(BUL(`an inspectable pipeline in which the detector gates a deterministic sanitizer, so the defense makes no generative call.`));
add(H2("12.3", "Suggested Structure and Page Budget"));
add(TABLE("outline", "Suggested paper structure (IEEE two-column) with page budget", [3400, 1500, 1500, 2626],
  ["Section", "Full paper (8 pp.)", "Graduate Forum (4 pp.)", "Material in this report"],
  [
    ["Abstract and Introduction", "1.0", "0.6", "Ch. 1; draft abstract above"],
    ["Related work and positioning", "0.75", "0.3", `Ch. 2; {T:positioning}`],
    ["Threat model and system", "0.75", "0.5", `Ch. 3; {F:arch}`],
    ["Benchmark", "0.75", "0.4", `Ch. 4; {T:families}, {F:partitions}`],
    ["Detector and training", "1.0", "0.6", `Ch. 5; Eqs. (1)–(3); {T:config}`],
    ["Evaluation setup", "0.25", "0.1", "Ch. 6"],
    ["Results", "1.5", "1.1", `Ch. 7; {T:results}, {F:confusion}, {F:families}`],
    ["Defense layer (brief)", "0.5", "0.2", `Ch. 8; {T:l2checks}`],
    ["Limitations, future work, conclusion", "0.75", "0.2", "Ch. 10–11, 13"],
    ["References", "≈ 1 (not counted)", "1 (separate page)", "References of this report"],
  ], { align: [AlignmentType.LEFT, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.LEFT] }));
add(H2("12.4", "Figures and Tables for the Paper"));
add(TABLE("figmap", "Which report items to reuse in the paper", [3100, 2400, 3526],
  ["Paper item", "Source", "Notes"],
  [
    ["Architecture figure", `{F:arch}`, "essential in both formats"],
    ["Per-family recall figure", `{F:families}`, "the key analysis figure; keep in both formats"],
    ["Results table", `{T:results}`, "essential; can replace the confusion-matrix figure in 4 pages"],
    ["Confusion matrices", `{F:confusion}`, "8-page version"],
    ["Metric intervals figure", `{F:metrics}`, "optional; duplicates the results table"],
    ["Partition figure", `{F:partitions}`, "8-page version; one sentence suffices in 4 pages"],
    ["Positioning table", `{T:positioning}`, "8-page version; text in 4 pages"],
    ["Training configuration", `{T:config}`, "8-page version; one sentence in 4 pages"],
    ["Score distribution, ROC, threshold sweep", "Appendix A (MATLAB)", "add if space allows; generated from the predictions files"],
  ]));
add(PG(`All figures are 3.45 in wide (one IEEE column) with 7–8 pt text, and are supplied as PNG (600 dpi) and PDF. Blue always marks the regular test and orange the challenge set; the pair is safe for colour-blind readers.`));
add(H2("12.5", "Claims and Wording"));
add(BUL(`Say "withheld from fine-tuning" rather than simply "unseen".`));
add(BUL(`Say "on this synthetic benchmark" whenever a percentage is quoted.`));
add(BUL(`Do not claim the first hybrid defense or the first detect-and-remove method ({T:claims}).`));
add(BUL(`Do not report end-to-end attack-success numbers until a complete run exists; intermediate runs are not results.`));
add(BUL(`Cite BIPIA [@bipia] as the base paper for the problem, the threat model and the evaluation philosophy, and PromptArmor [@promptarmor] as the closest detect-and-remove prior art.`));
add(H2("12.6", "Items to Verify Before Submission"));
add(TABLE("verify", "Values to confirm from the project files", [3000, 2900, 3126],
  ["Item", "Value in this report", "Where to check"],
  [
    ["Learning rate, weight decay, warm-up, clipping", "5 × 10^{−5}; 0.01; 10%; 1.0 (code defaults)", "training log or @@train_layer1.py@@ arguments"],
    ["Window length / overlap", "256 / 64 tokens", "detector configuration in @@config.py@@"],
    ["Name of the eighth withheld family", "encoded instructions", "generator family list / dataset manifest"],
    ["Description of dependency traps", "instruction framed as a prerequisite", "templates; missed chunks in the challenge CSV"],
    ["Exact family sizes (n)", "626, 608, 629–634, ≈ 625", "@@plot_from_predictions.m@@ output"],
    ["Total parameter count", `${int(PRM.distilbert_total)} (uncased vocabulary)`, "@@sum(p.numel() for p in model.parameters())@@"],
    ["Layer 2 legacy checks", "{T:l2checks}", "re-run on the current local code"],
    ["Training time; library versions", "not recorded", "training log; @@pip freeze@@"],
    ["COMSNETS tracks, deadlines, limits", "{T:venue}", "comsnets.org and EDAS"],
  ]));
add(H2("12.7", "Anonymisation and Artifact Availability"));
add(PG(`For double-blind tracks, remove all author names, affiliations, acknowledgements and the GitHub link, and write about prior work in the third person. Use "The code, benchmark generator and trained detector will be released upon publication." If reviewers should see code, use an anonymised repository service rather than the project repository. COMSNETS allows an arXiv version, but asks authors not to advertise it on large mailing lists or social media during review.`));

// ---------------------------------------------------------------- 13 Conclusion
add(H1("13", "Conclusion"));
add(PG(`This project moved the first line of defense against indirect prompt injection out of the generator and into the retrieval step. A DistilBERT detector was fine-tuned on a laptop CPU, with 7.68M trainable parameters and 6,000 training chunks. It screens every retrieved chunk and gates a deterministic clause sanitizer, so the defense needs no generative call. On a 100,000-chunk benchmark, it made no errors on 10,000 chunks from known attack families. On 10,000 chunks from eight withheld families, it reached ${p2(C.recall)}% recall with zero false positives. The remaining misses are concentrated and actionable: multilingual instructions above all, then dependency traps and Unicode spoofing. Retrieval-time detection is therefore a cheap and inspectable first layer for local RAG. Turning this detector-level evidence into a security claim requires the external, adaptive and end-to-end evaluations listed in Chapter 11.`));

// ---------------------------------------------------------------- References
add(H1("", "References"));
const refsBlock = () => [...citeNum.entries()].sort((a, b) => a[1] - b[1]).map(([key, num]) => new Paragraph({
  style: "Reference", tabStops: [{ type: TabStopType.LEFT, position: 500 }],
  children: [new TextRun({ text: `[${num}]` }), new TextRun({ children: [new Tab()] }), ...runs(REFS[key], { size: 20 })] }));
const refsMarker = body.length;

// ---------------------------------------------------------------- Appendix A
add(H1("", "Appendix A: Reproducing the Figures", { pageBreak: true }));
add(H2("A.1", "Figures in this Report (Python)"));
add(PG(`The figures were generated with Python and matplotlib from the recorded numbers, so every figure and table agrees by construction:`));
add(CODE([
  "cd paper\\build",
  "python paper_numbers.py      & REM writes numbers.json (counts, intervals, parameter counts)",
  "python make_figures.py       & REM writes ..\\figures\\fig1..fig5 (.png 600 dpi, .pdf, .svg)",
  "python make_equations.py     & REM writes ..\\figures\\eq1..eq4 (.png)",
]));
add(H2("A.2", "MATLAB: Step by Step"));
add(PG(`The MATLAB scripts need no toolboxes and were tested end to end (in GNU Octave 10.3) on files with the same structure as the project's outputs.`));
add(NUMLIST([
  `Copy the @@matlab@@ folder into the project, e.g. @@<project>\\paper\\matlab@@, so that the results folder is @@..\\..\\results@@. Otherwise, edit @@RESULTS_DIR@@ and @@METRICS_JSON@@ at the top of the scripts.`,
  `Start MATLAB and set the Current Folder to the @@matlab@@ folder.`,
  `Run @@make_report_figures@@. It writes Figures 2–5 to @@figures_matlab\\@@ as 600-dpi PNG and vector PDF. No data files are needed.`,
  `Run @@plot_from_predictions@@. It reads both predictions files, prints the confusion matrices at τ = 0.175 (they must equal {T:results}) and the per-family table with exact n and Wilson intervals. It then draws Figures A1–A5: score distributions, ROC, precision–recall, threshold sweep and per-family recall. If a column is not detected, it lists the available columns; set @@LABEL_COLUMN@@, @@SCORE_COLUMN@@ or @@FAMILY_COLUMN@@ and run again.`,
  `Run @@plot_validation_history@@ to plot the per-epoch validation curves stored in @@layer1_metrics.json@@ (Figure A6). If it cannot find them, set @@HISTORY_FIELD@@ to the field name it prints.`,
  `Insert the PNG files into Word (Insert → Pictures) at 3.5 in width for a two-column paper. In LaTeX, use the PDF files with @@\\includegraphics[width=\\columnwidth]{...}@@.`,
]));
add(TABLE("figscripts", "Figures, scripts and data", [2700, 3000, 3326],
  ["Figure", "MATLAB script", "Data needed"],
  [
    [`{F:arch} (architecture)`, "— (diagram; reuse the PNG or redraw)", "—"],
    [`{F:partitions}–{F:families}`, "@@make_report_figures.m@@", "none (recorded numbers in the script)"],
    ["A1 score distributions", "@@plot_from_predictions.m@@", "both predictions CSV files"],
    ["A2 ROC curves, A3 precision–recall", "@@plot_from_predictions.m@@", "both predictions CSV files"],
    ["A4 threshold sweep", "@@plot_from_predictions.m@@", "challenge predictions CSV"],
    ["A5 per-family recall (exact n)", "@@plot_from_predictions.m@@", "challenge predictions CSV with family column"],
    ["A6 validation history", "@@plot_validation_history.m@@", "@@layer1_metrics.json@@"],
  ]));
add(H2("A.3", "Listing: make_report_figures.m"));
const mlines = fs.readFileSync(path.join(HERE, "..", "matlab", "make_report_figures.m"), "utf8").replace(/\r/g, "").split("\n");
add(CODE(mlines, 15));
add(H2("A.4", "Listing: wilson_ci.m"));
add(CODE(fs.readFileSync(path.join(HERE, "..", "matlab", "wilson_ci.m"), "utf8").replace(/\r/g, "").split("\n"), 15));

// ---------------------------------------------------------------- Appendix B
add(H1("", "Appendix B: Equations in LaTeX"));
add(PG(`For the paper, the equations can be typeset natively:`));
add(CODE([
  "% (1) chunk score",
  "s(c) = \\max_{w \\in \\mathcal{W}(c)} p_\\theta(\\mathrm{inj} \\mid w)",
  "% (2) decision",
  "\\hat{y}(c) = \\mathbb{1}\\left[ s(c) \\ge \\tau \\right]",
  "% (3) threshold selection on validation data",
  "\\tau^{*} = \\arg\\max_{\\tau} \\mathrm{TPR}_{\\mathrm{val}}(\\tau) \\quad \\text{s.t.} \\quad \\mathrm{FPR}_{\\mathrm{val}}(\\tau) \\le \\alpha",
  "% (4) Wilson score interval",
  "\\frac{\\hat{p} + \\frac{z^2}{2n} \\pm z\\sqrt{\\frac{\\hat{p}(1-\\hat{p})}{n} + \\frac{z^2}{4n^2}}}{1 + \\frac{z^2}{n}}, \\qquad z = 1.96",
]));
add(PG(`In Word, use Insert → Equation and type the same expressions in LaTeX mode (Microsoft 365 accepts LaTeX syntax).`));

// ---------------------------------------------------------------- Appendix C
add(H1("", "Appendix C: Recorded Numbers and Derivations"));
add(H2("C.1", "Confusion Matrices (Rows: Actual; Columns: Predicted)"));
add(CODE([
  "                      Regular test            Challenge",
  "                   benign   injected      benign   injected",
  "actual benign       5,000          0       5,000          0",
  "actual injected         0      5,000         191      4,809",
]));
add(H2("C.2", "Derived Quantities"));
add(BUL(`Challenge recall = 4,809 / 5,000 = ${p2(C.recall)}%; F1 = 2 × 1 × 0.9618 / 1.9618 = ${p2(C.f1)}%; MCC = 4,809 / √(4,809 × 5,191) = ${C.mcc.toFixed(4)}.`));
add(BUL(`Multilingual family size: the only integer *n* with 1 − 148/*n* rounding to 76.36% is *n* = 626. Dependency traps: 1 − 37/*n* = 93.91% gives *n* = 608. Unicode spoofing: 1 − 6/*n* = 99.05% holds for *n* = 629 to 634.`));
add(BUL(`Recall without the multilingual family: (4,809 − 478) / (5,000 − 626) = 4,331 / 4,374 = ${p2(N.recall_without_multilingual)}%. Without multilingual instructions and dependency traps: 3,760 / 3,766 = ${p2(N.recall_without_multilingual_and_dependency)}%.`));
add(BUL(`Shares of the 191 misses: 148 (77.5%), 37 (19.4%), 6 (3.1%).`));
add(BUL(`Optimizer steps: ⌈6,000 / 16⌉ = ${TR.steps_per_epoch} per epoch; 3 epochs = ${3 * TR.steps_per_epoch} steps.`));
add(BUL(`DistilBERT parameters: embeddings 30,522 × 768 + 512 × 768 + 2 × 768 = ${int(EMB)}; one block 4 × (768² + 768) + 2 × 768 + (768 × 3,072 + 3,072) + (3,072 × 768 + 768) + 2 × 768 = ${int(PRM.distilbert_block)}; head (768² + 768) + (768 × 2 + 2) = ${int(PRM.distilbert_head)}; total ${int(PRM.distilbert_total)}.`));

// insert references at the marker now that every citation has been numbered
body.splice(refsMarker, 0, ...refsBlock());

// ====================================================================== FRONT MATTER
const infoRow = (k, v) => new TableRow({ children: [
  new TableCell({ width: { size: 2400, type: WidthType.DXA }, margins: { top: 60, bottom: 60, left: 0, right: 120 },
    borders: { top: NONE, bottom: NONE, left: NONE, right: NONE },
    children: [new Paragraph({ children: [new TextRun({ text: k, bold: true, color: NAVY, size: 21 })] })] }),
  new TableCell({ width: { size: 6626, type: WidthType.DXA }, margins: { top: 60, bottom: 60, left: 120, right: 0 },
    borders: { top: NONE, bottom: NONE, left: NONE, right: NONE },
    children: [new Paragraph({ children: runs(v, { size: 21 }) })] }),
] });
const titlePage = [
  new Paragraph({ spacing: { before: 1800, after: 240 }, children: [new TextRun({ text: "PROJECT REPORT", bold: true, color: ACCENT, size: 22, characterSpacing: 60 })] }),
  new Paragraph({ spacing: { after: 240, ...AUTO(252) }, children: [new TextRun({ text: "Lightweight Retrieval-Time Detection of Indirect Prompt Injection in Local RAG Systems", bold: true, color: NAVY, size: 52 })] }),
  new Paragraph({ spacing: { after: 480, ...AUTO(276) }, border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: ACCENT, space: 12 } },
    children: [new TextRun({ text: "A partially fine-tuned DistilBERT detector that gates clause-level sanitization: complete project record and reference material for the conference paper", color: INK2, size: 26 })] }),
  new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: [2400, 6626],
    borders: { top: NONE, bottom: NONE, left: NONE, right: NONE, insideHorizontal: NONE, insideVertical: NONE },
    rows: [
      infoRow("Prepared for", "[Guide name], Project Guide"),
      infoRow("Prepared by", "[Student name(s)]"),
      infoRow("Institution", "[Department], [Institution]"),
      infoRow("Date", "29 October 2026"),
      infoRow("Target venue", "COMSNETS 2027, Bengaluru, 5–9 January 2027"),
      infoRow("Base paper", "J. Yi *et al.*, \"Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models\" (BIPIA), ACM KDD 2025, pp. 1809–1820, doi: 10.1145/3690624.3709179"),
      infoRow("Status", "Layer 1 detector trained and evaluated on 20,000 held-out chunks; end-to-end RAG benchmark not yet complete"),
    ] }),
];

function tocBlock() {
  const out = [new Paragraph({ style: "TocTitle", children: [new TextRun("Contents")] })];
  for (const h of headings) {
    const page = TOC_PAGES && TOC_PAGES[h.text] != null ? String(TOC_PAGES[h.text]) : "00";
    out.push(new Paragraph({
      tabStops: [{ type: TabStopType.RIGHT, position: TEXT_W, leader: "dot" }],
      indent: { left: h.level === 1 ? 0 : 440 },
      spacing: { before: h.level === 1 ? 100 : 0, after: 20, ...AUTO(252) },
      children: [new TextRun({ text: h.text, bold: h.level === 1, size: h.level === 1 ? 22 : 21 }),
        new TextRun({ children: [new Tab(), page], bold: h.level === 1, size: h.level === 1 ? 22 : 21 })],
    }));
  }
  return out;
}

// ====================================================================== DOCUMENT
const header = new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
  border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF", space: 4 } },
  children: [new TextRun({ text: "Lightweight Retrieval-Time Detection of Indirect Prompt Injection · Project Report", size: 17, color: INK2 })] })] });
const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], size: 18, color: INK2 })] })] });
const pageProps = { size: PAGE, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN, header: 708, footer: 708 } };

const doc = new Document({
  creator: "Project team", title: "Lightweight Retrieval-Time Detection of Indirect Prompt Injection in Local RAG Systems",
  description: "Project report: detection-focused reference material for the COMSNETS 2027 paper write-up",
  styles: {
    default: { document: { run: { font: FONT, size: 22 }, paragraph: { spacing: { after: 0, ...AUTO(276) } } } },
    paragraphStyles: [
      { id: "Body", name: "Report Body", basedOn: "Normal", quickFormat: true, run: { font: FONT, size: 22 },
        paragraph: { alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, ...AUTO(276) } } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Body", quickFormat: true,
        run: { font: FONT, size: 32, bold: true, color: NAVY },
        paragraph: { spacing: { before: 240, after: 200, ...AUTO(240) }, keepNext: true, keepLines: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Body", quickFormat: true,
        run: { font: FONT, size: 26, bold: true, color: NAVY },
        paragraph: { spacing: { before: 240, after: 100, ...AUTO(240) }, keepNext: true, keepLines: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Body", quickFormat: true,
        run: { font: FONT, size: 22, bold: true, italics: true, color: "404040" },
        paragraph: { spacing: { before: 160, after: 60, ...AUTO(240) }, keepNext: true, outlineLevel: 2 } },
      { id: "Caption", name: "Report Caption", basedOn: "Normal", run: { font: FONT, size: 19 },
        paragraph: { alignment: AlignmentType.CENTER, spacing: { after: 200, ...AUTO(240) } } },
      { id: "TableText", name: "Report Table Text", basedOn: "Normal", run: { font: FONT, size: 19 },
        paragraph: { spacing: { after: 0, ...AUTO(240) } } },
      { id: "Reference", name: "Report Reference", basedOn: "Normal", run: { font: FONT, size: 20 },
        paragraph: { alignment: AlignmentType.LEFT, indent: { left: 500, hanging: 500 }, spacing: { after: 80, ...AUTO(252) } } },
      { id: "TocTitle", name: "Contents Title", basedOn: "Normal", run: { font: FONT, size: 32, bold: true, color: NAVY },
        paragraph: { spacing: { after: 240 } } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 240 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 240 } } } }] },
    { reference: "numbers", levels: [
      { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 400, hanging: 300 } } } }] },
  ] },
  sections: [
    { properties: { page: pageProps }, children: titlePage },
    { properties: { page: { ...pageProps, pageNumbers: { start: 1 } } }, headers: { default: header }, footers: { default: footer },
      children: [...tocBlock(), PB(), ...body] },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  fs.writeFileSync(path.join(HERE, "headings.json"), JSON.stringify(headings.map((h) => h.text), null, 1));
  const unused = Object.keys(REFS).filter((k) => !citeNum.has(k));
  console.log(`wrote ${OUT} (${(buf.length / 1024).toFixed(0)} KB): ${figCount} figures, ${tabCount} tables, ${eqCount} equations, ${citeNum.size} references${TOC_PAGES ? ", contents with page numbers" : " (pass 1)"}`);
  if (unused.length) console.log("uncited:", unused.join(", "));
});
