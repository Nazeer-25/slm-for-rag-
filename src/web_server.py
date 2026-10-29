"""The presentation website: http://127.0.0.1:8000 (FastAPI + a static front end in web/).

  python src/web_server.py                  start (start_demo.bat does this for you)
  python src/web_server.py --open-browser   start and open the browser
  python src/web_server.py --check          print the readiness checks and exit

The server starts in a second; models load when you press "Initialize workspace".
Every answer comes from real inference by the configured LLM (local Ollama by default).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import socket
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

import config
from config import CORPORA, MODE_DESCRIPTIONS, MODE_LABELS, MODES, PATHS, Paths

class RunRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    mode: str
    corpus: str = "enterprise_poisoned"
    top_k: int = Field(default=config.DEFAULT_TOP_K, ge=1, le=10)
    custom_poison: Optional[str] = Field(default=None, max_length=4000)
    custom_target: Optional[str] = Field(default=None, max_length=200)


KB_COLUMNS = {
    "synthetic": ["chunk_id", "pair_id", "split", "division", "topic", "variant", "label", "template_family",
                  "injection_position", "attacker_target", "question", "expected_fact", "text", "clean_reference"],
    "external": ["chunk_id", "source", "variant", "label", "text"],
}


def architecture(ws) -> dict:
    from layer2_quarantine import rule_catalog

    det_cfg = {}
    cfg_path = ws.paths.detector_dir / "detector_config.json"
    if cfg_path.exists():
        det_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return {
        "embedding": {"model": config.EMBEDDING_MODEL, "dimensions": config.EMBEDDING_DIM,
                      "cache": "models/embedding_cache"},
        "vector_store": {"engine": "Chroma (persistent)", "path": "data/chroma", "similarity": "cosine",
                         "collections": [config.ENTERPRISE_COLLECTION, config.EXTERNAL_COLLECTION,
                                         config.LIVE_COLLECTION]},
        "layer1": {"model": det_cfg.get("base_model", config.DETECTOR_BASE_MODEL) + " (fine-tuned)",
                   "threshold": det_cfg.get("threshold", config.DETECTOR_THRESHOLD),
                   "window_tokens": det_cfg.get("max_length", config.DETECTOR_MAX_LENGTH),
                   "stride_tokens": det_cfg.get("stride", config.DETECTOR_STRIDE),
                   "trained": bool(det_cfg), "trained_at": det_cfg.get("trained_at"),
                   "train_examples": det_cfg.get("train_examples")},
        "layer2": {"parser": f"spaCy {config.SPACY_MODEL}", "rules": rule_catalog()},
        "llm": {"label": ws.status()["llm"]["label"], "provider": config.LLM_PROVIDER,
                "temperature": config.LLM_TEMPERATURE, "max_tokens": config.LLM_MAX_TOKENS},
        "top_k_default": config.DEFAULT_TOP_K,
        "modes": [{"id": m, "label": MODE_LABELS[m], "description": MODE_DESCRIPTIONS[m],
                   "extra_llm_calls": "one per retrieved chunk" if m == "llm_judge_defense" else "none"}
                  for m in MODES],
    }


def kb_dataset(paths: Paths, name: str) -> dict:
    from data_generator import read_csv

    files = paths.dataset_files()
    if name not in files:
        raise KeyError(name)
    path = files[name]
    if not path.exists():
        return {"name": name, "file": f"data/{path.name}", "exists": False, "rows": [], "columns": []}
    rows = read_csv(path)
    columns = KB_COLUMNS["external" if name == "external" else "synthetic"]
    return {"name": name, "file": f"data/{path.name}", "exists": True, "columns": columns,
            "rows": [{c: r.get(c, "") for c in columns} for r in rows]}


def results_payload(paths: Paths) -> dict:
    def load(p: Path):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    return {"detector": load(paths.detector_metrics_json), "benchmark": load(paths.benchmark_results_json),
            "benchmark_runs_available": paths.benchmark_runs_json.exists(),
            "chart_available": paths.benchmark_chart_png.exists()}


def create_app(ws):
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles

    from scenarios import build_scenarios

    app = FastAPI(title="Context - Hybrid RAG Defense Workbench", docs_url=None, redoc_url=None)
    web_dir = ws.paths.web_dir

    @app.get("/api/status")
    def status():
        return ws.status()

    @app.post("/api/initialize")
    def initialize():
        return {"started": ws.start_initialize()}

    @app.post("/api/run")
    def run(req: RunRequest):
        if req.mode not in MODES:
            raise HTTPException(400, f"Unknown mode {req.mode}")
        if req.corpus not in CORPORA:
            raise HTTPException(400, f"Unknown corpus {req.corpus}")
        if ws.initializing:
            raise HTTPException(409, "The workspace is still initializing.")
        try:
            return ws.run(req.question.strip(), req.mode, req.corpus, req.top_k, req.custom_poison,
                          req.custom_target)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/kb/{name}")
    def kb(name: str):
        try:
            return kb_dataset(ws.paths, name)
        except KeyError:
            raise HTTPException(404, f"Unknown dataset {name}") from None

    @app.get("/api/architecture")
    def arch():
        return architecture(ws)

    @app.get("/api/scenarios")
    def scenarios():
        try:
            return build_scenarios(ws.paths)
        except (OSError, LookupError) as exc:
            raise HTTPException(409, f"Scenarios need data/dataset.csv ({exc})") from exc

    @app.get("/api/results")
    def results():
        return results_payload(ws.paths)

    @app.get("/api/download/{name}")
    def download(name: str):
        allowed = {"detector_metrics.json": ws.paths.detector_metrics_json,
                   "benchmark_results.json": ws.paths.benchmark_results_json,
                   "benchmark_runs.json": ws.paths.benchmark_runs_json}
        path = allowed.get(name)
        if path is None or not path.exists():
            raise HTTPException(404, f"{name} has not been generated yet")
        return FileResponse(path, media_type="application/json", filename=name)

    @app.get("/api/history")
    def history():
        return ws.history.summaries()

    @app.get("/api/history/{run_id}")
    def history_item(run_id: str, download: bool = False):
        trace = ws.history.get(run_id)
        if trace is None:
            raise HTTPException(404, "Run not found in history")
        headers = {"Content-Disposition": f'attachment; filename="trace-{run_id}.json"'} if download else {}
        return JSONResponse(trace, headers=headers)

    @app.get("/api/history-export")
    def history_export():
        return Response(json.dumps(ws.history.all(), indent=1), media_type="application/json",
                        headers={"Content-Disposition": 'attachment; filename="demo_history.json"'})

    @app.delete("/api/history")
    def history_clear(confirm: bool = Body(False, embed=True)):
        if not confirm:
            raise HTTPException(400, "Send {\"confirm\": true} to clear the history")
        ws.history.clear()
        return {"cleared": True}

    @app.get("/results/benchmark_comparison.png")
    def chart():
        if not ws.paths.benchmark_chart_png.exists():
            raise HTTPException(404, "No benchmark chart yet")
        return FileResponse(ws.paths.benchmark_chart_png, media_type="image/png",
                            headers={"Cache-Control": "no-store"})

    app.mount("/static", StaticFiles(directory=web_dir / "static"), name="static")

    @app.get("/")
    def index():
        return FileResponse(web_dir / "index.html", headers={"Cache-Control": "no-store"})

    return app


def preflight(paths: Paths = PATHS) -> list[tuple[str, bool, str]]:
    """Fast readiness checks (no model loading)."""
    from embeddings import embedding_cached
    from layer1_detector import detector_available
    from llm_client import make_llm_client

    checks = [("Python", sys.version_info >= (3, 10), sys.version.split()[0])]
    missing = [p.name for p in (paths.dataset_csv, paths.train_csv, paths.test_csv) if not p.exists()]
    checks.append(("Source CSVs", not missing, "present" if not missing else
                   f"missing {', '.join(missing)} - run: python src/data_generator.py"))
    checks.append(("Layer 1 detector", detector_available(paths.detector_dir),
                   "models/layer1_detector" if detector_available(paths.detector_dir) else
                   "not trained - run: python src/train_layer1.py (hybrid mode needs it)"))
    spacy_ok = importlib.util.find_spec(config.SPACY_MODEL) is not None
    checks.append(("English parser", spacy_ok, config.SPACY_MODEL if spacy_ok else
                   f"missing - run: python -m spacy download {config.SPACY_MODEL}"))
    cached = embedding_cached(paths.embedding_cache)
    checks.append(("Embedding cache", True, "models/embedding_cache" if cached else
                   "empty - downloads on first Initialize (needs internet once)"))
    try:
        health = make_llm_client().health()
        ok = health["reachable"] and health["model_present"]
        checks.append(("LLM", ok, f"{config.OLLAMA_MODEL if config.LLM_PROVIDER == 'ollama' else config.OPENAI_MODEL} "
                                  f"ready" if ok else health.get("error") or "not reachable"))
    except Exception as exc:  # noqa: BLE001
        checks.append(("LLM", False, str(exc)))
    return checks


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=config.WEB_HOST)
    ap.add_argument("--port", type=int, default=config.WEB_PORT)
    ap.add_argument("--open-browser", action="store_true")
    ap.add_argument("--check", action="store_true", help="print readiness checks and exit")
    args = ap.parse_args(argv)

    print("Context - Hybrid RAG Defense Workbench")
    for name, ok, detail in preflight():
        print(f"  [{'ok' if ok else '!!'}] {name:<17} {detail}")
    if args.check:
        return 0
    if not _port_free(args.host, args.port):
        print(f"\nPort {args.port} is already in use - is the website already running? "
              f"Open http://{args.host}:{args.port} or stop the other window first.")
        return 1

    import uvicorn

    from workspace import Workspace

    local = "127.0.0.1" if args.host in ("0.0.0.0", "::", "") else args.host
    url = f"http://{local}:{args.port}"
    print(f"\nOpen {url}  (keep this window and Ollama running; Ctrl+C stops the website)")
    if local != args.host:
        print(f"Listening on all interfaces: others on your network use http://<this-PC's-IP>:{args.port}")
    if args.open_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(Workspace()), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
