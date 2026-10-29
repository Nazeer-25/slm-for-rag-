"""Streamlit dashboard: an alternative to the custom website (python src/web_server.py).

    streamlit run app.py

Same engine as the website (src/workspace.py): every answer comes from real inference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import config  # noqa: E402
from config import CORPORA, MODE_DESCRIPTIONS, MODE_LABELS, MODES  # noqa: E402
from data_generator import read_csv  # noqa: E402
from workspace import Workspace  # noqa: E402

st.set_page_config(page_title="Context - Hybrid RAG Defense Workbench", layout="wide")


@st.cache_resource
def get_workspace() -> Workspace:
    return Workspace()


ws = get_workspace()

# ------------------------------------------------------------------ sidebar: workspace
with st.sidebar:
    st.title("Context")
    st.caption("Hybrid RAG Defense Workbench")
    status = ws.status()
    if status["initialized"]:
        st.success(f"Workspace ready - {status['llm']['label']}")
    else:
        st.info("Workspace not initialized")
    if st.button("Initialize workspace" if not status["initialized"] else "Re-initialize", type="primary",
                 width="stretch"):
        with st.spinner("Loading models and indexing (the first time takes longer)..."):
            ws.initialize()
        st.rerun()
    for step in status["steps"]:
        icon = {"ok": "✅", "error": "❌", "skipped": "➖", "running": "⏳"}.get(step["status"], "▫️")
        st.markdown(f"{icon} **{step['label']}**  \n<small>{step['detail'] or step['status']}</small>",
                    unsafe_allow_html=False)

playground, knowledge, evidence = st.tabs(["Live playground", "Knowledge base", "Results & evidence"])

# ------------------------------------------------------------------ playground
with playground:
    st.subheader("Live playground")
    left, right = st.columns([2, 3])
    with left:
        test_questions = sorted({r["question"] for r in read_csv(ws.paths.test_csv)}) if ws.paths.test_csv.exists() else []
        sample = st.selectbox("Sample question", ["(type your own)"] + test_questions)
        question = st.text_area("Question", value="" if sample == "(type your own)" else sample,
                                placeholder="What is the Cedar division refund policy?")
        available = ws.available_modes() or list(MODES)
        mode = st.radio("Defense mode", available, index=len(available) - 1, format_func=MODE_LABELS.get,
                        captions=[MODE_DESCRIPTIONS[m] for m in available])
        corpus = st.radio("Corpus", list(CORPORA), index=1, format_func=lambda c: CORPORA[c]["label"], horizontal=True)
        top_k = st.slider("Retrieved chunks (top-k)", 1, 10, config.DEFAULT_TOP_K)
        with st.expander("Temporary custom poison"):
            custom_poison = st.text_area("Custom document (live_payloads, this run only)")
            custom_target = st.text_input("Attacker target to watch for (optional)")
        run = st.button("Run query", type="primary", disabled=not status["initialized"])
        if not status["initialized"]:
            st.caption("Initialize the workspace first (sidebar).")
    with right:
        if run and question.strip():
            with st.spinner(f"Running {MODE_LABELS[mode]}..."):
                st.session_state["trace"] = ws.run(question.strip(), mode, corpus, top_k, custom_poison or None,
                                                   custom_target or None)
        trace = st.session_state.get("trace")
        if trace:
            ev = trace.get("evaluation") or {}
            st.markdown(f"**{trace['mode_label']}** · {trace['corpus_label']} · top-{trace['top_k']}")
            if trace["status"] == "ok":
                st.info(trace["answer"] or "(empty answer)")
            else:
                st.error(f"Run failed: {trace['error']}")
            cols = st.columns(4)
            cols[0].metric("Total latency", f"{trace['latency_ms']['total'] / 1000:.2f} s")
            cols[1].metric("LLM calls", trace["llm_calls"])
            cols[2].metric("Hijacked", "-" if ev.get("hijacked") is None else ("yes" if ev["hijacked"] else "no"))
            cols[3].metric("Expected fact", "-" if ev.get("fact_found") is None else ("found" if ev["fact_found"] else "missing"))
            for c in trace["chunks"]:
                gt = c.get("ground_truth") or {}
                truth = "custom poison" if c.get("is_live") else {0: "clean", 1: "poisoned"}.get(gt.get("label"), "unknown")
                p = f" · P(inj) {c['layer1']['probability']:.2f}" if c.get("layer1") else ""
                with st.expander(f"#{c['rank']} {c['chunk_id']} · {truth} · {c['action']}{p}"):
                    if c.get("layer2"):
                        for r in c["layer2"]["removed_clauses"]:
                            st.markdown(f"~~{r['text']}~~  \n*{'; '.join(r['reasons'])}*")
                        for r in c["layer2"]["retained_instructions"]:
                            st.markdown(f"Kept: {r['text']}  \n*{r['reason']}*")
                    st.caption("Retrieved text")
                    st.text(c["text"])
                    if c["action"] == "sanitized":
                        st.caption("Delivered to the LLM")
                        st.text(c["delivered_text"])
            st.download_button("Download trace (JSON)", json.dumps(trace, indent=1), f"trace-{trace['run_id']}.json",
                               "application/json")

# ------------------------------------------------------------------ knowledge base
with knowledge:
    st.subheader("Knowledge base")
    counts = status["collections"] if isinstance(status["collections"], dict) else {}
    cols = st.columns(3)
    for col, name in zip(cols, ("enterprise_policies", "external_examples", "live_payloads")):
        col.metric(f"Chroma · {name}", "-" if counts.get(name) is None else counts[name])
    files = {k: v for k, v in ws.paths.dataset_files().items() if v.exists()}
    if files:
        name = st.selectbox("Dataset", list(files), format_func=lambda k: f"data/{files[k].name}")
        rows = read_csv(files[name])
        query = st.text_input("Search")
        if query:
            rows = [r for r in rows if query.lower() in " ".join(map(str, r.values())).lower()]
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.warning("No datasets yet. Run: python src/data_generator.py")

# ------------------------------------------------------------------ results & evidence
with evidence:
    st.subheader("Layer 1 detector")
    if ws.paths.detector_metrics_json.exists():
        d = json.loads(ws.paths.detector_metrics_json.read_text(encoding="utf-8"))
        cols = st.columns(6)
        for col, key in zip(cols, ("accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate")):
            col.metric(key.replace("_", " "), "n/a" if d[key] is None else f"{d[key]:.3f}")
        st.caption(d["caveat"])
    else:
        st.info("No saved detector metrics. Run: python src/train_layer1.py")
    st.subheader("Benchmark")
    if ws.paths.benchmark_results_json.exists():
        b = json.loads(ws.paths.benchmark_results_json.read_text(encoding="utf-8"))
        table = []
        for m, x in b["modes"].items():
            table.append({"mode": MODE_LABELS[m], "ASR": x["asr"]["value"], "benign utility": x["benign_utility"]["value"],
                          "utility under attack": x["utility_under_attack"]["value"], "FPR": x["fpr"]["value"],
                          "ROUGE-L": x["rouge_l_preservation_poisoned"], "mean latency (ms)": x["latency_ms_mean"],
                          "failed": x["failed"]["total"]})
        st.dataframe(table, width="stretch", hide_index=True)
        if ws.paths.benchmark_chart_png.exists():
            st.image(str(ws.paths.benchmark_chart_png))
    else:
        st.info("No benchmark yet. Run: python src/evaluate_all.py")
    st.subheader("Query history")
    summaries = ws.history.summaries()
    if summaries:
        st.dataframe(summaries, width="stretch", hide_index=True)
        st.download_button("Download all traces", json.dumps(ws.history.all(), indent=1), "demo_history.json",
                           "application/json")
    else:
        st.caption("No experiments yet.")
