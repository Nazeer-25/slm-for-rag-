"""Builds the supplementary report (.docx) from the outputs of the three experiments.

Contents (only these analyses): 1 four-configuration ablation, 2 threshold sensitivity,
3 score distributions. Tables and the interpretation sentences are generated from the numbers in
results/supplementary/. Figures: the MATLAB versions are used when present in
results/supplementary/figures_matlab/ (same file names), otherwise the Python versions.

Usage (Windows, project folder):
  .venv\\Scripts\\pip.exe install python-docx          (once)
  .venv\\Scripts\\python.exe supplementary\\make_report.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

NAVY = "1F3864"
CONFIG_LABELS = {"no_defense": "No defense", "detector_only": "Detector only", "sanitizer_only": "Sanitizer only",
                 "hybrid": "Hybrid"}
SPLIT_LABELS = {"regular": "Regular test", "challenge": "Challenge", "all": "Both"}


def p(x, nd=1):
    return "n/a" if x is None or x == "" or (isinstance(x, float) and math.isnan(x)) else f"{100 * float(x):.{nd}f}%"


def pv(x, nd=1):
    """Percentage without the % sign (for table cells whose header states the unit)."""
    return "n/a" if x is None or x == "" or (isinstance(x, float) and math.isnan(x)) else f"{100 * float(x):.{nd}f}"


def ci(lo, hi, nd=1):
    return "" if lo in (None, "") or (isinstance(lo, float) and math.isnan(lo)) else f"[{100 * float(lo):.{nd}f}, {100 * float(hi):.{nd}f}]"


class Report:
    def __init__(self, title: str):
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Cm, Pt, RGBColor

        self.Pt, self.Cm, self.RGB, self.ALIGN = Pt, Cm, RGBColor, WD_ALIGN_PARAGRAPH
        self.doc = Document()
        sec = self.doc.sections[0]
        sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(sec, side, Cm(2.3))
        normal = self.doc.styles["Normal"]
        normal.font.name, normal.font.size = "Times New Roman", Pt(11)
        normal.paragraph_format.space_after = Pt(6)
        for level, size in ((1, 15), (2, 12.5)):
            st = self.doc.styles[f"Heading {level}"]
            st.font.name, st.font.size, st.font.bold = "Times New Roman", Pt(size), True
            st.font.color.rgb = RGBColor.from_string(NAVY)
        self.figures = self.tables = 0
        t = self.doc.add_paragraph()
        r = t.add_run(title)
        r.bold, r.font.size, r.font.color.rgb = True, Pt(18), RGBColor.from_string(NAVY)

    def h1(self, text):
        self.doc.add_heading(text, level=1)

    def h2(self, text):
        self.doc.add_heading(text, level=2)

    def para(self, text, italic=False, size=None):
        par = self.doc.add_paragraph()
        r = par.add_run(text)
        r.italic = italic
        if size:
            r.font.size = self.Pt(size)
        par.paragraph_format.alignment = self.ALIGN.JUSTIFY
        return par

    def bullets(self, items):
        for text in items:
            self.doc.add_paragraph(text, style="List Bullet")

    def figure(self, path: Path, caption: str, width_cm=11.5):
        self.figures += 1
        par = self.doc.add_paragraph()
        par.alignment = self.ALIGN.CENTER
        par.add_run().add_picture(str(path), width=self.Cm(width_cm))
        cap = self.doc.add_paragraph()
        cap.alignment = self.ALIGN.CENTER
        r = cap.add_run(f"Figure {self.figures}: ")
        r.bold, r.font.size = True, self.Pt(9.5)
        cap.add_run(caption).font.size = self.Pt(9.5)

    def table(self, title: str, header: list[str], rows: list[list[str]], widths_cm: list[float] | None = None, note=None):
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn

        self.tables += 1
        cap = self.doc.add_paragraph()
        cap.alignment = self.ALIGN.CENTER
        r = cap.add_run(f"Table {self.tables}: ")
        r.bold, r.font.size = True, self.Pt(9.5)
        cap.add_run(title).font.size = self.Pt(9.5)
        cap.paragraph_format.keep_with_next = True
        tab = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        tab.style = "Table Grid"
        tab.autofit = False
        if widths_cm:
            layout = OxmlElement("w:tblLayout")
            layout.set(qn("w:type"), "fixed")
            tab._tbl.tblPr.append(layout)
            for j, w in enumerate(widths_cm):
                tab.columns[j].width = self.Cm(w)
        for i, cells in enumerate([header, *rows]):
            for j, value in enumerate(cells):
                cell = tab.rows[i].cells[j]
                cell.text = ""
                run = cell.paragraphs[0].add_run(str(value))
                run.font.size = self.Pt(9)
                run.bold = i == 0
                if i == 0:
                    shd = OxmlElement("w:shd")
                    shd.set(qn("w:val"), "clear")
                    shd.set(qn("w:color"), "auto")
                    shd.set(qn("w:fill"), "E7ECF4")
                    cell._tc.get_or_add_tcPr().append(shd)
                if widths_cm:
                    cell.width = self.Cm(widths_cm[j])
        if note:
            n = self.doc.add_paragraph()
            n.add_run(note).font.size = self.Pt(8.5)
        else:
            self.doc.add_paragraph()

    def save(self, path: Path):
        self.doc.save(str(path))


def pick_figure(out: Path, name: str) -> tuple[Path | None, str]:
    for folder, source in ((out / "figures_matlab", "MATLAB"), (out / "figures_python", "Python")):
        if (folder / f"{name}.png").exists():
            return folder / f"{name}.png", source
    return None, ""


# --------------------------------------------------------------------------- section 1: ablation
def ablation_section(rep: Report, out: Path):
    rep.h1("1  Four-configuration ablation")
    f = out / "ablation_summary.json"
    if not f.exists():
        rep.para("Not run yet. Run: .venv\\Scripts\\python.exe supplementary\\ablation_four_configs.py", italic=True)
        return
    meta = json.loads(f.read_text())
    summ = meta["summary"]
    splits = [s for s in meta["splits"]]
    items = meta["items"]
    rep.para("The four configurations answer exactly the same questions over exactly the same chunks with the same "
             "prompt and generator, so differences come from the defense alone.")
    rep.bullets([
        "Items: " + "; ".join(f"{SPLIT_LABELS[s]}: {items[s]['injected']} injected chunks (stratified by attack family) "
                              f"and {items[s]['clean']} clean chunks (the clean twins of injected chunks)" for s in splits) + ".",
        "Context: the chunk the question is about is the only context document (retrieval exposure is guaranteed).",
        "Configurations: no defense (chunk unchanged); detector only (flagged chunks are dropped); sanitizer only "
        "(every chunk is sanitized); hybrid (flagged chunks are sanitized, the others pass unchanged). "
        f"Detector threshold τ = {meta['threshold']:g}.",
        f"Generator: {meta['llm']} via Ollama, temperature {meta['llm_temperature']}, seed {meta['llm_seed']}, "
        f"at most {meta['llm_max_tokens']} tokens; identical prompts are generated once and reused.",
        "Attack success (ASR): the attacker's target phrase appears in the answer. Correct answer: every expected value "
        "of the question (two facts) appears in the answer. Both are lexical proxies; 95% Wilson intervals in brackets.",
        f"Cost of the defense on this CPU: Layer 1 {meta['detector_ms_per_chunk']:.1f} ms per chunk, Layer 2 "
        f"{meta['sanitizer_ms_per_chunk']:.1f} ms per chunk (no extra LLM call in any configuration).",
    ])
    rows = []
    for r in summ:
        if r["split"] not in splits:
            continue
        flagged = (f"{r['flagged_injected']}/{r['n_injected']} | {r['flagged_clean']}/{r['n_clean']}"
                   if r["flagged_injected"] != "" else "—")
        if r["config"] == "detector_only":
            changed = f"dropped {r['dropped_injected']} | {r['dropped_clean']}"
        elif r["changed_injected"] != "":
            changed = f"edited {r['changed_injected']} | {r['changed_clean']}"
        else:
            changed = "—"
        rows.append([SPLIT_LABELS[r["split"]], CONFIG_LABELS[r["config"]],
                     f"{pv(r['asr'])} {ci(r['asr_lo'], r['asr_hi'])}",
                     f"{pv(r['correct_attack_rate'])} {ci(r['correct_attack_lo'], r['correct_attack_hi'])}",
                     f"{pv(r['correct_clean_rate'])} {ci(r['correct_clean_lo'], r['correct_clean_hi'])}",
                     flagged, changed])
    rep.table("Attack success and answer correctness per configuration, in percent with 95% Wilson intervals "
              "(same items in every row of a test set)",
              ["Test set", "Configuration", "ASR (injected)", "Correct (injected)", "Correct (clean)",
               "Flagged by Layer 1 (inj. | clean)", "Chunk changed (inj. | clean)"],
              rows, [1.8, 2.6, 2.4, 2.4, 2.4, 2.4, 2.4])
    fig, source = pick_figure(out, "fig_ablation")
    if fig:
        rep.figure(fig, "Attack success (top) and correct answers on injected (middle) and clean items (bottom) for the "
                        f"four configurations; whiskers are 95% Wilson intervals ({source} figure).", 10.5)
    fam_file = out / "ablation_family_asr.csv"
    if fam_file.exists():
        fam = [r for r in C.read_rows(fam_file) if r["split"] == "challenge"]
        if fam:
            names = sorted({r["family"] for r in fam})
            by = {(r["family"], r["config"]): r for r in fam}
            rows = [[n, str(by.get((n, "no_defense"), {}).get("n", "")),
                     *[p(float(by[(n, c)]["asr"])) if (n, c) in by else "—" for c in CONFIG_LABELS]] for n in names]
            rep.table("Attack success per withheld attack family (challenge set)",
                      ["Family", "n", *CONFIG_LABELS.values()], rows, [4.0, 1.2, 2.8, 2.8, 2.8, 2.8])
    rep.h2("Interpretation")
    for text in interpret_ablation(summ, splits):
        rep.para(text)


def interpret_ablation(summ, splits):
    get = {(r["split"], r["config"]): r for r in summ}
    out = []
    for s in splits:
        if not all((s, c) in get for c in CONFIG_LABELS):
            continue
        g = {c: get[(s, c)] for c in CONFIG_LABELS}
        asr = {c: g[c]["asr"] for c in g}
        att = {c: g[c]["correct_attack_rate"] for c in g}
        cln = {c: g[c]["correct_clean_rate"] for c in g}
        best_asr = min(asr.values())
        lowest = [CONFIG_LABELS[c] for c in asr if abs(asr[c] - best_asr) < 1e-12]
        best_att = max(att.values())
        top_att = [CONFIG_LABELS[c] for c in att if abs(att[c] - best_att) < 1e-12]
        text = (f"{SPLIT_LABELS[s]}: without a defense the generator produced the attacker's phrase for {p(asr['no_defense'])} "
                f"of the injected items and answered {p(att['no_defense'])} of them correctly. Detector only gave ASR "
                f"{p(asr['detector_only'])}, sanitizer only {p(asr['sanitizer_only'])} and the hybrid "
                f"{p(asr['hybrid'])}; the lowest ASR was reached by {' and '.join(lowest)}. Under attack, answers were "
                f"correct for {p(att['detector_only'])} (detector only), {p(att['sanitizer_only'])} (sanitizer only) and "
                f"{p(att['hybrid'])} (hybrid); the most correct answers came from {' and '.join(top_att)}.")
        dropped = g["detector_only"]["dropped_injected"]
        if dropped:
            text += (f" Detector only removes the whole chunk when it is flagged ({dropped} of {g['detector_only']['n_injected']} "
                     "injected chunks), so the policy facts disappear together with the injection; the hybrid keeps them "
                     "and removes only the flagged clauses.")
        diff = max(abs(cln[c] - cln["no_defense"]) for c in cln)
        text += (f" On clean chunks, correctness was {p(cln['no_defense'])} without a defense and {p(cln['hybrid'])} with "
                 f"the hybrid" + (", i.e. the defenses did not change benign answers." if diff < 1e-12 else
                                  f"; the largest change from no defense across configurations was {100 * diff:.1f} points."))
        out.append(text)
    out.append("These are lexical proxies on synthetic data: a quoted or refused target phrase counts as an attack "
               "success, and an answer can contain the expected numbers yet still be wrong in other ways.")
    return out


# --------------------------------------------------------------------------- section 2: thresholds
def threshold_section(rep: Report, out: Path):
    rep.h1("2  Threshold sensitivity of the detector")
    f = out / "threshold_summary.json"
    if not f.exists():
        rep.para("Not run yet. Run: .venv\\Scripts\\python.exe supplementary\\threshold_sensitivity.py", italic=True)
        return
    s = json.loads(f.read_text())
    v = s["validation"]
    rep.bullets([
        f"Selection data: validation set only ({v['chunks']:,} chunks: {v['injected']:,} injected, {v['benign']:,} benign).",
        "Thresholds: " + ", ".join(f"{t:g}" for t in s["thresholds"]) + ".",
        f"Rule: {s['rule']}; constraint FPR ≤ {100 * s['fpr_limit']:g}%.",
        "The regular and challenge test sets are used only afterwards, for the final evaluation of the selected "
        f"threshold and the current threshold {s['current_threshold']:g} (saved per-chunk predictions from training).",
        f"Detector: {s['detector']}. Scorer check against the saved test predictions: {s['scorer_check'].get('status')}"
        + (f" (max |difference| {s['scorer_check']['max_abs_difference']:.1e} over {s['scorer_check']['chunks']} chunks)."
           if "max_abs_difference" in s["scorer_check"] else "."),
    ])
    rows = [[f"{r['threshold']:g}" + (" (current)" if r["current"] else ""), f"{r['TP']:,}/{r['positives']:,}",
             f"{pv(r['recall'], 2)} {ci(r['recall_lo'], r['recall_hi'], 2)}", f"{r['FP']:,}/{r['negatives']:,}",
             f"{pv(r['fpr'], 2)} {ci(r['fpr_lo'], r['fpr_hi'], 2)}", pv(r["precision"], 2),
             "yes" if r["feasible"] else "no", "τ*" if r["selected"] else ""] for r in s["grid"]]
    rep.table("Validation recall and false-positive rate per threshold, in percent with 95% Wilson intervals",
              ["τ", "TP / injected", "Recall", "FP / benign", "FPR", "Precision", "FPR ≤ 2%", "Selected"],
              rows, [1.9, 1.9, 3.2, 1.7, 2.8, 1.8, 1.6, 1.5])
    fig, source = pick_figure(out, "fig_threshold_sensitivity")
    if fig:
        rep.figure(fig, f"Validation recall (top) and FPR (bottom) against the threshold. Dashed vertical line: current "
                        f"threshold {s['current_threshold']:g}; dashed horizontal line: FPR limit 2%; ring: selected τ* "
                        f"({source} figure).", 10.5)
    final = s["final_evaluation"]
    rows = [[SPLIT_LABELS[r["split"]], f"{r['threshold']:g}", r["role"], f"{pv(r['recall'], 2)} {ci(r['recall_lo'], r['recall_hi'], 2)}",
             f"{pv(r['fpr'], 2)} {ci(r['fpr_lo'], r['fpr_hi'], 2)}", pv(r["precision"], 2), pv(r["accuracy"], 2),
             f"{r['TN']:,} / {r['FP']:,} / {r['FN']:,} / {r['TP']:,}"] for r in final]
    rep.table("Final evaluation on the test sets, in percent with 95% Wilson intervals (not used for selection)",
              ["Test set", "τ", "Role", "Recall", "FPR", "Precision", "Accuracy", "TN / FP / FN / TP"], rows,
              [1.8, 1.0, 2.0, 3.1, 2.6, 1.8, 1.8, 2.3])
    fam = s.get("family_recall") or []
    if fam:
        ts = sorted({r["threshold"] for r in fam})
        names = sorted({r["family"] for r in fam})
        by = {(r["family"], r["threshold"]): r for r in fam}
        rows = [[n, str(by[(n, ts[0])]["n"]), *[f"{p(by[(n, t)]['recall'], 2)} ({by[(n, t)]['missed']} missed)" for t in ts]]
                for n in names]
        rep.table("Challenge recall per withheld attack family", ["Family", "n", *[f"τ = {t:g}" for t in ts]], rows,
                  [4.4, 1.4, *[round(10.6 / len(ts), 2)] * len(ts)])
    rep.h2("Interpretation")
    for text in interpret_threshold(s):
        rep.para(text)


def interpret_threshold(s):
    grid, cur, tau = s["grid"], s["current_threshold"], s["selected_threshold"]
    rec = [r["recall"] for r in grid]
    fpr = [r["fpr"] for r in grid]
    n_feas = len(s["feasible_thresholds"])
    def span(values, name):
        lo, hi = min(values), max(values)
        return (f"{name} was {p(lo, 2)} at every threshold" if abs(hi - lo) < 1e-12
                else f"{name} ranged from {p(lo, 2)} to {p(hi, 2)}")
    out = [f"On the validation set, {span(rec, 'recall')} and {span(fpr, 'the FPR')}; {n_feas} of {len(grid)} "
           f"thresholds satisfy FPR ≤ {100 * s['fpr_limit']:g}%."]
    if tau is None:
        out.append("No threshold in the grid meets the FPR constraint on validation, so none is selected.")
        return out
    r_tau = next(r for r in grid if r["threshold"] == tau)
    r_cur = next(r for r in grid if abs(r["threshold"] - cur) < 1e-9)
    tied = [r["threshold"] for r in grid if r["TP"] == r_tau["TP"] and r["FP"] == r_tau["FP"]]
    if abs(tau - cur) < 1e-9:
        text = f"The rule selects τ* = {tau:g}, which is the current threshold, so the independent sweep confirms the operating point."
        if len(tied) > 1:
            text += (f" Thresholds from {min(tied):g} to {max(tied):g} give identical validation recall and FPR "
                     f"({p(r_tau['recall'], 2)} and {p(r_tau['fpr'], 2)}); {tau:g} lies at the centre of this plateau, which "
                     "leaves the largest margin on both sides.")
        out.append(text)
    else:
        out.append(f"The rule selects τ* = {tau:g} instead of {cur:g}: validation recall {p(r_tau['recall'], 2)} vs "
                   f"{p(r_cur['recall'], 2)} and FPR {p(r_tau['fpr'], 2)} vs {p(r_cur['fpr'], 2)}.")
    fe = {(r["split"], r["threshold"]): r for r in s["final_evaluation"]}
    parts = []
    for split in ("regular", "challenge"):
        a, b = fe.get((split, tau)), fe.get((split, cur))
        if a and b and abs(tau - cur) > 1e-9:
            parts.append(f"on the {SPLIT_LABELS[split].lower()} set τ* gives recall {p(a['recall'], 2)} and FPR {p(a['fpr'], 2)} "
                         f"(0.175: {p(b['recall'], 2)} and {p(b['fpr'], 2)})")
        elif b:
            parts.append(f"on the {SPLIT_LABELS[split].lower()} set recall is {p(b['recall'], 2)} and FPR {p(b['fpr'], 2)}")
    out.append("In the final evaluation, " + "; ".join(parts) + ". These test numbers were not used to choose the threshold.")
    return out


# --------------------------------------------------------------------------- section 3: distributions
def distribution_section(rep: Report, out: Path):
    rep.h1("3  Score distributions of benign and injected chunks")
    f = out / "score_distribution_summary.json"
    if not f.exists():
        rep.para("Not run yet. Run: .venv\\Scripts\\python.exe supplementary\\score_distributions.py", italic=True)
        return
    s = json.loads(f.read_text())
    rep.para(f"Scores are the saved per-chunk predictions of the trained detector on the regular test and the challenge "
             f"set. Benign and injected chunks are shown as separate distributions; the dashed line marks τ = {s['threshold']:g}.")
    fig, source = pick_figure(out, "fig_score_distributions")
    if fig:
        rep.figure(fig, f"Detector scores of benign (top, blue) and injected chunks (bottom, orange) on the regular test (left) "
                        f"and challenge set (right); log-scaled counts, bins of {s['bin_width']:g}; dashed line: τ = "
                        f"{s['threshold']:g} ({source} figure).", 11.5)
    rows = []
    for split in ("regular", "challenge"):
        info = s["splits"][split]
        for cls in ("benign", "injected"):
            st = info[cls]
            side = (f"{st['at_or_above_threshold']:,} ≥ τ" if cls == "benign" else f"{st['below_threshold']:,} < τ")
            rows.append([SPLIT_LABELS[split], cls, f"{st['n']:,}", f"{st['median']:.4f}", f"{st['p01']:.4f}",
                         f"{st['p99']:.4f}", f"{st['min']:.4f}", f"{st['max']:.4f}", side])
    rep.table("Score summary per test set and class", ["Test set", "Class", "n", "Median", "1st pct.", "99th pct.", "Min",
                                                       "Max", "Wrong side of τ"], rows,
              [2.0, 1.6, 1.4, 1.7, 1.7, 1.7, 1.7, 1.7, 2.9])
    rep.h2("Interpretation")
    for text in interpret_distributions(s):
        rep.para(text)


def interpret_distributions(s):
    out, tau = [], s["threshold"]
    for split in ("regular", "challenge"):
        info = s["splits"][split]
        b, i = info["benign"], info["injected"]
        text = f"{SPLIT_LABELS[split]}: "
        if b["max"] < tau <= i["min"]:
            text += (f"every benign chunk scored at most {b['max']:.4f} and every injected chunk at least {i['min']:.4f}, so "
                     f"the two classes are completely separated and τ = {tau:g} lies inside the gap.")
        else:
            text += (f"{info['false_alarms']:,} benign chunks reach τ and {info['misses']:,} injected chunks fall below it; "
                     f"the highest benign score is {b['max']:.4f} and the lowest injected score {i['min']:.4f}.")
        text += (f" The median benign score is {b['median']:.4f} and the median injected score {i['median']:.4f} "
                 f"(ROC AUC {info['roc_auc']:.4f}).")
        if info["misses"]:
            text += (f" The {info['misses']:,} missed injections have a median score of {info['median_score_of_misses']:.4f} "
                     f"(highest {info['max_score_of_misses']:.4f}); {info['misses_between_0.10_and_threshold']:,} of them lie "
                     f"between 0.10 and τ, where {info['benign_between_0.10_and_threshold']:,} benign chunks also lie. "
                     + ("Most misses are therefore confident errors far below the threshold rather than borderline cases, "
                        "so moving the threshold within the tested range would recover few of them."
                        if info["misses_between_0.10_and_threshold"] < 0.5 * info["misses"] else
                        "Many misses are borderline, so the threshold choice matters for this set."))
        out.append(text)
    out.append("This description uses the test sets; the threshold itself is chosen on the validation set (Section 2).")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project-root")
    ap.add_argument("--out-dir")
    ap.add_argument("--output", help="default: results/supplementary/Supplementary_Experiments_Report.docx")
    args = ap.parse_args(argv)
    try:
        import docx  # noqa: F401
    except ImportError:
        raise SystemExit("python-docx is missing. Install it once with: .venv\\Scripts\\pip.exe install python-docx")
    root = C.find_project_root(args.project_root)
    out = C.out_dir(root, args.out_dir)
    rep = Report("Supplementary Experiments: Defense Ablation, Threshold Sensitivity and Score Separation")
    rep.para(f"Layer 1 DistilBERT detector and Layer 2 clause sanitizer. Generated on {time.strftime('%d %B %Y')} from "
             f"the files in results/supplementary. Data: the 10,000-chunk validation, regular-test and challenge "
             f"partitions of the 100,000-chunk benchmark.", italic=True, size=10)
    ablation_section(rep, out)
    threshold_section(rep, out)
    distribution_section(rep, out)
    path = Path(args.output) if args.output else out / "Supplementary_Experiments_Report.docx"
    rep.save(path)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
