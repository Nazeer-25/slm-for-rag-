// Review of RAG_COMSNETS.pdf against the project: corrections and additions (.docx)
const fs = require("fs");
const path = require("path");
const d = require("docx");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, AlignmentType, WidthType, BorderStyle,
  HeadingLevel, ShadingType, LevelFormat, LineRuleType, Footer, PageNumber, VerticalAlign } = d;

const OUT = path.join(__dirname, "Paper_Review_RAG_COMSNETS.docx");
const FONT = "Times New Roman", NAVY = "1F3864", GREY = "595959";
const W = 9026;                                   // A4 text width with 1 in margins
const AUTO = (line) => ({ line, lineRule: LineRuleType.AUTO });
const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const LINE = (c = "BFBFBF", s = 4) => ({ style: BorderStyle.SINGLE, size: s, color: c });

// inline markup: **bold**, *italic*, @@mono@@
function runs(text, base = {}) {
  const out = [];
  const st = { b: !!base.bold, i: !!base.italics };
  const re = /(\*\*|\*|@@[^@]+@@)/g;
  let last = 0, m;
  const push = (t, extra = {}) => t && out.push(new TextRun({ ...base, text: t, bold: st.b, italics: st.i, ...extra }));
  while ((m = re.exec(text))) {
    push(text.slice(last, m.index));
    if (m[0] === "**") st.b = !st.b;
    else if (m[0] === "*") st.i = !st.i;
    else push(m[0].slice(2, -2), { font: "Courier New", size: (base.size || 22) - 3 });
    last = m.index + m[0].length;
  }
  push(text.slice(last));
  return out;
}
const body = [];
const add = (...x) => x.flat().forEach((y) => body.push(y));
const P = (t, o = {}) => new Paragraph({ style: "Body", children: runs(t), ...o });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const BUL = (t) => new Paragraph({ style: "Body", numbering: { reference: "b", level: 0 }, spacing: { after: 60, ...AUTO(264) }, children: runs(t) });

function grid(widths, rows, { head = true, fills = {} } = {}) {
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths,
    borders: { top: LINE("808080", 8), bottom: LINE("808080", 8), left: NONE, right: NONE, insideHorizontal: LINE(), insideVertical: NONE },
    rows: rows.map((r, i) => new TableRow({ cantSplit: true, tableHeader: head && i === 0, children: r.map((c, j) => new TableCell({
      width: { size: widths[j], type: WidthType.DXA }, margins: { top: 60, bottom: 60, left: 100, right: 100 },
      verticalAlign: VerticalAlign.TOP,
      shading: (head && i === 0) ? { fill: "E7ECF4", type: ShadingType.CLEAR, color: "auto" } : (fills[j] ? { fill: fills[j], type: ShadingType.CLEAR, color: "auto" } : undefined),
      children: String(c).split("\n").map((line) => new Paragraph({ spacing: { after: 20, ...AUTO(240) },
        children: runs(line, { size: 19, bold: head && i === 0 }) })),
    })) })),
  });
}
let changeNo = 0;
function CHANGE(prefix, title, where, current, replacement, why) {
  changeNo += 1;
  return [
    new Paragraph({ keepNext: true, spacing: { before: 200, after: 80 }, children: runs(`**${prefix}${changeNo}. ${title}**`, { size: 22, color: NAVY }) }),
    grid([1500, 7526], [
      ["Where", where],
      ["Current text", current],
      ["Replace with", replacement],
      ["Why", why],
    ], { head: false, fills: { 0: "F2F4F8" } }),
  ];
}
function CODE(lines) {
  return [new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: [W],
    borders: { top: LINE(), bottom: LINE(), left: LINE(), right: LINE(), insideHorizontal: NONE, insideVertical: NONE },
    rows: [new TableRow({ children: [new TableCell({ width: { size: W, type: WidthType.DXA }, margins: { top: 80, bottom: 80, left: 120, right: 120 },
      shading: { fill: "F4F4F2", type: ShadingType.CLEAR, color: "auto" },
      children: lines.map((l) => new Paragraph({ spacing: { after: 0, ...AUTO(228) }, children: [new TextRun({ text: l || " ", font: "Courier New", size: 16 })] })) })] })] }),
    new Paragraph({ spacing: { after: 120 }, children: [] })];
}

// ============================================================ content
add(new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "Review of RAG_COMSNETS.pdf against the project", bold: true, size: 36, color: NAVY })] }));
add(P("*Scope: text, equations, tables and references. Graphs, diagrams and the architecture figure were not reviewed, as requested. Checked against the recorded training run (29 October 2026), the benchmark files validation.csv, test.csv and challenge.csv, and the verified reference list.*"));

add(H1("1  Summary"));
add(P("The paper matches the project. Every headline number agrees with the recorded run, and no numerical error was found. It also describes the pipeline, the threshold rule and the evaluation status correctly. The changes below are of two kinds:"));
add(BUL("**Required (10):** leftover notes-to-self, a broken figure reference, a claim that confidence intervals are reported when Table I has none, and a statement attributed to BIPIA that BIPIA does not make. They also cover protocol wording that differs from the ablation that will actually be run, a generalisation claim that the data does not fully support, a typo and one unverifiable reference."));
add(BUL("**Recommended (8):** project details the paper omits (training settings, partition isolation, hard negatives, family names and counts), intervals in Table I, closely related prior work, and citations for tools the paper names."));

add(H1("2  What already matches"));
add(grid([3000, 3426, 2600], [
  ["Item", "Paper", "Project / data"],
  ["Corpus", "100,000 chunks, 50,000 policies, 10 topics, 32 families, 96 templates", "same"],
  ["Partitions", "70:10:10:10; 6,000 stratified training chunks", "same (70k pool; 3,000 / 3,000 sample)"],
  ["Trainable parameters", "7,680,002 of 66,955,010 (11.47%)", "same"],
  ["Scoring and threshold", "256/64 windows, max pooling, τ = 0.175 at validation FPR ≤ 2%", "same"],
  ["Regular test", "100% accuracy, precision, recall; FPR 0", "TN 5,000, FP 0, FN 0, TP 5,000"],
  ["Challenge", "98.09% accuracy, 100% precision, 96.18% recall, FPR 0, 191 misses", "TN 5,000, FP 0, FN 191, TP 4,809"],
  ["Per-family recall", "76.36%, 93.91%, 99.05%; five families at 100%", "same (n = 626, 608, 629)"],
  ["Shares of misses", "148 (77.5%), 37 (19.4%), 6 (3.1%)", "same"],
  ["Components", "MiniLM + Chroma, spaCy + rules, Llama 3.2 3B via Ollama, reminder", "same"],
  ["End-to-end status", "not claimed", "correct: no completed run"],
  ["References [1], [3]–[8]", "authors, venues, pages", "verified"],
]));

add(H1("3  Required changes"));
add(CHANGE("R", "Broken figure reference", "§III-A, end of the first paragraph",
  "“…context construction, and answer generation, as shown in Fig. III-A.”",
  "“…context construction, and answer generation (Fig. 1).”\nIn LaTeX, place @@\\label{fig:architecture}@@ after the figure's @@\\caption@@ and write @@Fig.~\\ref{fig:architecture}@@.",
  "The reference resolves to a section number, not a figure. The architecture figure itself does not appear in the PDF; once it is added, the reference must point to it and the later figures shift by one."));
add(CHANGE("R", "Typo", "§III-B, second paragraph", "“…7,680,002 of the model’s66,955,010 parameters…”", "“…7,680,002 of the model’s 66,955,010 parameters…”", "Missing space."));
add(CHANGE("R", "Note-to-self about reproducibility", "§IV-A, last sentence",
  "“Software versions and hardware specifications should be documented to facilitate reproducibility.”",
  "“All training and inference were performed on a single laptop CPU using PyTorch 2.14.1 (CPU build) and Hugging Face Transformers; the laptop’s NVIDIA GTX 1650 GPU was not used.”\nAdd the exact library versions from @@pip freeze@@.",
  "A submitted paper should state the setup, not a reminder to do so. These are the recorded facts."));
add(CHANGE("R", "Note-to-self and empty Table II", "§V-C, last two sentences, and Table II",
  "“Once the evaluation is complete, Table II can be populated with measured results. Valid trial counts and excluded failures should also be recorded for each configuration.” Table II contains only dashes.",
  "Option A, no ablation results by the deadline: delete Table II and both sentences, and end the subsection with “The protocol of Section IV-C is fully specified; its measurements are left to future work.”\nOption B, after running @@ablation_four_configs.py@@: fill Table II from @@ablation_summary.csv@@ and replace both sentences with “Table II reports the four configurations on the same items in every row; ASR is the percentage of injected items whose answer contains the attack’s target phrase, and correctness the percentage of answers that contain every expected value.”",
  "An empty results table and future-tense instructions read as an unfinished draft."));
add(CHANGE("R", "Intervals claimed but not shown", "§IV-D first paragraph, and Table I",
  "“Where appropriate, 95% Wilson confidence intervals are calculated for reported proportions.” Table I reports point estimates only.",
  "Use the Table I in Appendix A, which adds the intervals and an F1 row, and write: “Table I reports point estimates with 95% Wilson intervals in brackets [Wilson 1927].”",
  "The text and the table must agree. The intervals were computed from the recorded counts."));
add(CHANGE("R", "Statement attributed to the base paper", "§II-A, second and third sentences",
  "“Yi et al. introduced BIPIA to benchmark such attacks and evaluate defenses across different LLMs [8]. Their work highlights the need to assess defenses against diverse attack patterns, including families excluded from training, rather than relying solely on performance against previously observed attacks.”",
  "“Yi et al. introduced BIPIA, the first benchmark for indirect prompt injection, covering five application scenarios and 250 attacker goals, and found all 25 evaluated LLMs vulnerable [8]. They attribute this to the inability of LLMs to distinguish informational context from actionable instructions, and propose boundary-awareness and explicit-reminder defenses: prompt-based variants for black-box models and adversarial fine-tuning with special boundary tokens for white-box models, the latter reducing attack success to near zero. Both act inside the generator. The present work instead screens retrieved content before generation, leaving the generator unchanged, and strengthens BIPIA’s separation of training and test attacks by withholding entire attack families from training and threshold selection.”",
  "BIPIA separates training and test attacks; family-level hold-out is this paper’s protocol. Since BIPIA is the base paper, its findings, its defenses and the difference from this work should be stated."));
add(CHANGE("R", "Ablation protocol wording", "§III-E, second sentence; §IV-C, first paragraph, last two sentences",
  "“The same retrieval configuration, generator settings, and questions are retained across experimental configurations to support a fair comparison.”\n“The same questions, attack cases, retrieval settings, and generator configuration are maintained across the configurations. The use of the untrusted-data reminder and the handling of flagged chunks are also recorded to clarify the role of each defense component.”",
  "§III-E: “In the four-configuration comparison, every configuration receives the same questions and chunks: each question is answered with its own policy chunk as the only context document, so retrieval exposure is fixed, and all configurations use the same prompt and generator settings (temperature 0, fixed seed).”\n§IV-C: “The same questions, attack cases, context chunks and generator configuration are used in every configuration. To isolate the detector and the sanitizer, the untrusted-data reminder r is omitted in all four configurations; the deployed pipeline (Algorithm 1) adds it on top. Detector-only drops a flagged chunk, sanitizer-only sanitizes every chunk, and the hybrid sanitizes flagged chunks only. Each test set contributes 200 injected items, stratified by attack family, and 100 clean items that are the clean versions of injected items, with the same questions.”",
  "This is how @@ablation_four_configs.py@@ runs: oracle context, no reminder in any configuration, 200 + 100 items. With @@--quick@@ the counts are 60 + 30. Keeping the reminder in the hybrid only would confound the comparison, as §III-D itself notes."));
add(CHANGE("R", "Correctness definition", "§IV-C, sentence before Eq. (8)",
  "“Answer correctness is measured separately as the percentage of valid question trials for which the expected answer is provided:”",
  "“Answer correctness is measured separately as the percentage of valid question trials whose answer contains every expected numeric value of the question (each question asks for two policy facts):”",
  "Each benchmark question has two expected values, and the scoring requires both."));
add(CHANGE("R", "Generalisation claim needs a qualification", "§V-A, sentence after the recall values; also §VI, limitations paragraph",
  "“These results suggest that the detector handles several withheld attack patterns effectively, but its performance varies with the attack characteristics.”",
  "“The detector therefore transfers to most withheld families. This transfer should be interpreted with care: in every partition, including the challenge set, each injected chunk contains one of the same ten attacker target phrases, so the withheld families differ from the training families in their framing rather than in their payload vocabulary. A trivial rule that flags any known target phrase inside quotation marks already reaches 100% recall on the challenge set, but at an FPR of 16.8%, because the hard negatives quote the same phrases as inert examples. The detector’s advantage is its ability to separate instruction-like use of these phrases from inert use at 0% FPR; detection of attacks with unseen target phrases has not yet been evaluated.”\nAdd to §VI: “In addition, all partitions share the same ten attacker target phrases, so the challenge set tests unseen attack framings rather than unseen payload vocabulary.”",
  "Measured on the uploaded files: 5,000 of 5,000 injected chunks in each partition contain their target phrase, and all ten challenge phrases also occur in validation and test. The quoted-phrase rule gives 97.6%, 97.3% and 100% recall at 16.7%, 16.6% and 16.8% FPR on validation, regular test and challenge. Reviewers who notice this would otherwise question the held-out claim."));
add(CHANGE("R", "Reference [2] cannot be confirmed as written", "References, [2] CausalArmor",
  "“…In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 58855–58877. PMLR, 2026.”",
  "Check the volume and pages on proceedings.mlr.press. If they cannot be confirmed, cite: “M. Kim, M. Parmar, P. Wallis, L. Miculicich, K. Jung, K. Dvijotham, L. T. Le, and T. Pfister, ‘CausalArmor: Efficient indirect prompt injection guardrails via causal attribution,’ in Proc. 43rd Int. Conf. Machine Learning (ICML), 2026, arXiv:2602.07918.”",
  "The ICML 2026 acceptance and author list are confirmed; no PMLR listing with these volume and page numbers was found. All other references are correct."));

add(H1("4  Recommended additions"));
changeNo = 0;
add(CHANGE("A", "Training details", "§III-B, after “…sampled from the synthetic training pool.”", "—",
  "“The sample is balanced (3,000 benign and 3,000 injected chunks) and stratified by topic, attack family and benign category. Training ran for three epochs with a batch size of 16, gradient clipping and a learning-rate schedule; the epoch-1 checkpoint was selected on the validation set.”",
  "These are the recorded settings; the paper gives none."));
add(CHANGE("A", "Partition isolation, hard negatives and family names", "§IV-A, after “…the latter contains the eight withheld families.” (replace the sentence on identifiers)",
  "“Policy and division identifiers are separated across the partitions to limit overlap between training and evaluation.”",
  "“The policies belong to 5,000 fictional divisions. The regular-test and challenge partitions each contain 500 divisions held out from training, and the validation, regular-test and challenge partitions share no divisions or policies. Injected instructions are placed at the beginning, middle or end of a chunk. To discourage shortcut learning, benign chunks include hard negatives from twelve categories, such as legitimate imperative policy text, quoted attack phrases, markup and inert target strings. The regular-test set contains the 24 attack families used in training; the challenge set contains only the eight withheld families: indirect completion, dependency traps, reasoning lures, polite requests, few-shot imitation, multilingual instructions, Unicode spoofing and encoded instructions.”",
  "Checked on the uploaded files: 500 divisions per partition with no overlap, no shared policies, 12 hard-negative categories, 24 seen families, and placements prefix/middle/suffix. The withheld families are never named in the current text."));
add(CHANGE("A", "Exact family counts", "§V-A, sentence with the recall values",
  "“…the reported recall is 76.36% for multilingual instructions, 93.91% for dependency traps, and 99.05% for Unicode spoofing. Five other families are reported at 100% recall.”",
  "“…recall is 76.36% for multilingual instructions (148 of 626 missed; 95% CI 72.87–79.52%), 93.91% for dependency traps (37 of 608; 91.72–95.55%) and 99.05% for Unicode spoofing (6 of 629; 97.93–99.56%). The other five families, indirect completion (608 chunks), reasoning lures (637), polite requests (637), few-shot imitation (626) and encoded instructions (629), are detected without a miss.”",
  "Family sizes come from challenge.csv; “reported” suggests the numbers come from elsewhere."));
add(CHANGE("A", "Bound on the false-positive rate", "§V, sentence after “No false positives were observed in either test set.”",
  "“However, the observed zero FPR does not guarantee the absence of false positives in deployment, since each test set contains only 5,000 benign chunks.”",
  "“However, the observed zero FPR does not guarantee the absence of false positives in deployment: with 5,000 benign chunks per test set, the 95% Wilson upper bound on the FPR is 0.077%.”",
  "Quantifies the statement."));
add(CHANGE("A", "Benchmark as a contribution", "§I, contribution list (first item)", "—",
  "“A 100,000-chunk synthetic enterprise-policy benchmark with paired clean and injected chunks, 32 attack families and twelve categories of hard negatives, with eight families withheld for a challenge partition.”",
  "The benchmark is a substantial part of the work but is not listed."));
add(CHANGE("A", "Closely related prior work", "§II-B or §II-C, new paragraph",
  "—",
  "“Supervised detectors such as PIGuard [PIGuard] and PromptShield [PromptShield] classify inputs as benign or injected, with attention to over-defense and to deployable false-positive rates, and DataSentinel [DataSentinel] trains an LLM-based detector against adaptive attacks. Prompt-level defenses such as Spotlighting [Spotlighting] mark the provenance of untrusted text. Layered designs that combine several mechanisms have been proposed for agents [Spotlight-Guard] and for RAG [Raghuraj et al.]. This work does not claim novelty for combining detection and sanitization; it studies a CPU-trainable detector under family hold-out that gates a sanitizer without additional generative calls.”",
  "With eight references, reviewers may see the novelty as unclear. The 2026 RAG hybrid-defense paper (Raghuraj et al.) has a very similar title and topic."));
add(CHANGE("A", "Citations for named tools and methods", "§I (first mention of indirect prompt injection); §III-A/§III-D (Llama 3.2, spaCy); §IV-D (Wilson intervals)", "—",
  "Cite Greshake et al. (2023) for indirect prompt injection, Grattafiori et al. (2024) for Llama 3.2, Honnibal et al. (2020) for spaCy, and Wilson (1927) for the intervals. BibTeX entries are in Appendix B.",
  "Each is used or named without a citation."));
add(CHANGE("A", "F1 in Table I", "Table I", "—", "Add an F1 row: 100.00 (regular) and 98.05 (challenge). It is included in Appendix A.",
  "F1 is the usual summary for a balanced detector; it was recorded."));

add(H1("5  Venue and format checks"));
add(BUL("**Length:** the paper has 6 A4 pages including references. The COMSNETS 2027 main track (3–8 pages, deadline 31 August 2026) has closed. The Graduate Forum requires exactly 4 pages of content + 1 page of references + a 1-page resume, and non-adherence disqualifies. Confirm the track and its limits on comsnets.org."));
add(BUL("**Paper size:** A4. IEEE conference templates default to US Letter; check whether EDAS for this track requires Letter."));
add(BUL("**Anonymity:** no author names are visible, which is correct for double-blind tracks. The Graduate Forum includes a resume and is not anonymous."));
add(BUL("**Page number:** a page number appears at the top of page 1. Remove it for the camera-ready version, since IEEE templates do not print page numbers."));

add(H1("Appendix A: Table I with intervals (LaTeX)"));
add(P("Requires @@\\usepackage{booktabs}@@; otherwise replace @@\\toprule@@, @@\\midrule@@ and @@\\bottomrule@@ with @@\\hline@@."));
const TABLE_TEX = [
  "\\begin{table}[t]",
  "\\caption{Detector performance on the held-out test sets (95\\% Wilson intervals in brackets).}",
  "\\label{tab:detector}",
  "\\centering",
  "\\begin{tabular}{lcc}",
  "\\toprule",
  "Metric & Regular test & Challenge test \\\\",
  "\\midrule",
  "Test chunks     & 10{,}000            & 10{,}000            \\\\",
  "Accuracy (\\%)  & 100.00 [99.96, 100] & 98.09 [97.80, 98.34] \\\\",
  "Precision (\\%) & 100.00 [99.92, 100] & 100.00 [99.92, 100]  \\\\",
  "Recall (\\%)    & 100.00 [99.92, 100] & 96.18 [95.61, 96.68] \\\\",
  "F1 (\\%)        & 100.00              & 98.05               \\\\",
  "FPR (\\%)       & 0.00 [0, 0.077]     & 0.00 [0, 0.077]     \\\\",
  "False negatives & 0                   & 191                 \\\\",
  "False positives & 0                   & 0                   \\\\",
  "\\bottomrule",
  "\\end{tabular}",
  "\\end{table}",
];
add(CODE(TABLE_TEX));

add(H1("Appendix B: BibTeX for the added references"));
const BIB = [
  "@inproceedings{greshake2023,",
  "  author    = {Greshake, Kai and Abdelnabi, Sahar and Mishra, Shailesh and Endres, Christoph and Holz, Thorsten and Fritz, Mario},",
  "  title     = {Not What You've Signed Up For: Compromising Real-World {LLM}-Integrated Applications with Indirect Prompt Injection},",
  "  booktitle = {Proc. 16th ACM Workshop on Artificial Intelligence and Security (AISec '23)},",
  "  pages     = {79--90}, year = {2023}, doi = {10.1145/3605764.3623985}}",
  "",
  "@article{wilson1927,",
  "  author  = {Wilson, Edwin B.},",
  "  title   = {Probable Inference, the Law of Succession, and Statistical Inference},",
  "  journal = {Journal of the American Statistical Association}, volume = {22}, number = {158},",
  "  pages   = {209--212}, year = {1927}, doi = {10.1080/01621459.1927.10502953}}",
  "",
  "@misc{honnibal2020spacy,",
  "  author = {Honnibal, Matthew and Montani, Ines and Van Landeghem, Sofie and Boyd, Adriane},",
  "  title  = {{spaCy}: Industrial-strength Natural Language Processing in {Python}},",
  "  publisher = {Zenodo}, year = {2020}, doi = {10.5281/zenodo.1212303}}",
  "",
  "@article{grattafiori2024llama3,",
  "  author  = {Grattafiori, Aaron and others},",
  "  title   = {The {Llama} 3 Herd of Models},",
  "  journal = {arXiv preprint arXiv:2407.21783}, year = {2024}}",
  "",
  "@article{raghuraj2026hybrid,",
  "  author  = {Raghuraj, Geetha and N, Veena and Bharathi, Jennie and Kadavad, Geetha Rani},",
  "  title   = {Hybrid Defense Against Indirect Prompt Injection in {RAG}: A Cross-Model Benchmark},",
  "  journal = {International Journal of Artificial Intelligence and Machine Learning},",
  "  volume  = {6}, number = {3}, pages = {1148--1156}, year = {2026}}",
  "",
  "@article{demirol2026spotlightguard,",
  "  author  = {Demirol, Doygun and Aydogan, Murat},",
  "  title   = {Balancing Security and Performance in {LLM} Agents: {Spotlight-Guard}, a Layered Defense Against Indirect Prompt Injection},",
  "  journal = {Applied Sciences}, volume = {16}, number = {15}, pages = {7662}, year = {2026}, doi = {10.3390/app16157662}}",
  "",
  "@inproceedings{li2025piguard,",
  "  author    = {Li, Hao and Liu, Xiaogeng and Zhang, Ning and Xiao, Chaowei},",
  "  title     = {{PIGuard}: Prompt Injection Guardrail via Mitigating Overdefense for Free},",
  "  booktitle = {Proc. 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},",
  "  pages     = {30420--30437}, year = {2025}, doi = {10.18653/v1/2025.acl-long.1468}}",
  "",
  "@inproceedings{jacob2025promptshield,",
  "  author    = {Jacob, D. and Alzahrani, H. and Hu, Z. and Alomair, B. and Wagner, D.},",
  "  title     = {{PromptShield}: Deployable Detection for Prompt Injection Attacks},",
  "  booktitle = {Proc. 15th ACM Conference on Data and Application Security and Privacy (CODASPY '25)},",
  "  pages     = {341--352}, year = {2025}, doi = {10.1145/3714393.3726501}}",
  "",
  "@inproceedings{liu2025datasentinel,",
  "  author    = {Liu, Yupei and Jia, Yuqi and Jia, Jinyuan and Song, Dawn and Gong, Neil Zhenqiang},",
  "  title     = {{DataSentinel}: A Game-Theoretic Detection of Prompt Injection Attacks},",
  "  booktitle = {Proc. 2025 IEEE Symposium on Security and Privacy (SP)},",
  "  pages     = {2190--2208}, year = {2025}, doi = {10.1109/SP61157.2025.00250}}",
  "",
  "@inproceedings{hines2024spotlighting,",
  "  author    = {Hines, K. and Lopez, G. and Hall, M. and Zarfati, F. and Zunger, Y. and K{\\i}c{\\i}man, E.},",
  "  title     = {Defending Against Indirect Prompt Injection Attacks With Spotlighting},",
  "  booktitle = {Proc. Conference on Applied Machine Learning in Information Security (CAMLIS 2024)},",
  "  series    = {CEUR Workshop Proceedings}, volume = {3920}, pages = {48--62}, year = {2024}}",
  "",
  "@inproceedings{kim2026causalarmor,",
  "  author    = {Kim, Minbeom and Parmar, Mihir and Wallis, Phillip and Miculicich, Lesly and Jung, Kyomin and Dvijotham, Krishnamurthy and Le, Long T. and Pfister, Tomas},",
  "  title     = {{CausalArmor}: Efficient Indirect Prompt Injection Guardrails via Causal Attribution},",
  "  booktitle = {Proc. 43rd International Conference on Machine Learning (ICML)},",
  "  year      = {2026}, note = {arXiv:2602.07918}}",
];
add(CODE(BIB));
fs.writeFileSync(path.join(__dirname, "paper_corrections_table1.tex"), TABLE_TEX.join("\n") + "\n");
fs.writeFileSync(path.join(__dirname, "paper_additions.bib"), BIB.join("\n") + "\n");

const doc = new Document({
  creator: "Project team", title: "Review of RAG_COMSNETS.pdf against the project",
  styles: {
    default: { document: { run: { font: FONT, size: 22 }, paragraph: { spacing: { after: 0, ...AUTO(264) } } } },
    paragraphStyles: [
      { id: "Body", name: "Body", basedOn: "Normal", run: { font: FONT, size: 22 }, paragraph: { alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, ...AUTO(276) } } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Body", run: { font: FONT, size: 28, bold: true, color: NAVY },
        paragraph: { spacing: { before: 280, after: 120 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Body", run: { font: FONT, size: 24, bold: true, color: NAVY },
        paragraph: { spacing: { before: 200, after: 80 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 360, hanging: 240 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1300, bottom: 1300, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], size: 18, color: GREY })] })] }) },
    children: body }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log("wrote", OUT, b.length); });
