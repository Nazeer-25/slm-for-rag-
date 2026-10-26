"""Phase 1 — data ingestion, payload injection and standardization.

Produces three files in data/processed/:

  chunks.jsonl      Detector dataset (train/val/test). Every poisoned document also
                    contributes its clean twin, so the classifier learns the payload,
                    not the topic. Schema:
                    {chunk_id, doc_id, text, clean_fact, payload, label: 0|1, position,
                     attack_family, template_id, canary, payload_source, payload_seen,
                     source, split, variant}
  rag_corpus.jsonl  The document store for the end-to-end benchmark (test docs only,
                    exactly ONE variant per doc: poisoned or clean).
  rag_eval.jsonl    Queries: {query_id, doc_id, question, answer, is_attack, canary,
                    attack_family, position, payload_seen, source}

Sources:
  * BIPIA (GitHub): text attacks (train/test files are disjoint attack categories)
    and email QA contexts.
  * SQuAD (HF): benign passages with gold answers.
  * Synthetic policy docs + benign-imperative hard negatives (offline).
  * deepset/prompt-injections (HF) and any configured InjectBench / RAG-PIBench
    adapters (HF id or local file, mapped through text_col/label_col).

Usage:
  python -m src.data_pipeline                    # full build
  python -m src.data_pipeline --offline          # skip all network sources
"""
from __future__ import annotations

import argparse
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

from src.payloads import (Payload, canary_hit, inject, render_payload, synthetic_payload_pool,
                          wrap_goal_as_injection)
from src.synthetic_docs import generate_hard_negatives, generate_policy_docs
from src.utils import get_logger, load_config, set_seed, write_jsonl

log = get_logger("data_pipeline")


# ----------------------------------------------------------------------------- loaders

def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as resp:
        dest.write_bytes(resp.read())
    return dest


def load_bipia(cfg: dict) -> tuple[dict[str, list[tuple[str, str]]], list[dict]]:
    """Return ({'train': [(category, attack)], 'test': [...]}, email_docs)."""
    bcfg = cfg["data"]["bipia"]
    raw = Path(cfg["paths"]["raw_dir"]) / "bipia"
    attacks: dict[str, list[tuple[str, str]]] = {"train": [], "test": []}
    for fname in bcfg["text_attack_files"]:
        split = "test" if "test" in fname else "train"
        data = json.loads(_download(f"{bcfg['base_url']}/{fname}", raw / fname).read_text("utf-8"))
        for category, items in data.items():
            attacks[split].extend((category, a) for a in items)
    docs = []
    for fname in bcfg["email_files"]:
        split = "test" if "test" in fname else "train"
        path = _download(f"{bcfg['base_url']}/{fname}", raw / fname.replace("/", "_"))
        for i, line in enumerate(path.read_text("utf-8").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            ideal = row["ideal"] if isinstance(row["ideal"], str) else str(row["ideal"])
            q = row["question"].removeprefix("Q:").strip()
            docs.append({"doc_id": f"bipia-email-{split}-{i:04d}", "text": row["context"].replace("|", "\n"),
                         "qas": [(q, ideal)], "source": "bipia_email", "fixed_split": split})
    log.info("BIPIA: %d train / %d test attacks, %d email docs", len(attacks["train"]), len(attacks["test"]), len(docs))
    return attacks, docs


def load_squad(cfg: dict) -> list[dict]:
    from datasets import load_dataset

    scfg = cfg["data"]["squad"]
    ds = load_dataset(scfg["hf_id"], split=scfg["split"])
    by_ctx: dict[str, dict] = {}
    for row in ds:
        ctx = row["context"].strip()
        if not (scfg["min_chars"] <= len(ctx) <= scfg["max_chars"]):
            continue
        entry = by_ctx.setdefault(ctx, {"title": row["title"], "qas": []})
        if row["answers"]["text"]:
            entry["qas"].append((row["question"].strip(), row["answers"]["text"][0]))
    docs = [{"doc_id": f"squad-{i:05d}", "text": ctx, "qas": v["qas"][:3], "source": "squad"}
            for i, (ctx, v) in enumerate(by_ctx.items()) if v["qas"]]
    docs = docs[: scfg["max_docs"]]
    log.info("SQuAD: %d passages", len(docs))
    return docs


def _load_table(src: dict):
    """Load an extra source from HF or a local file into a list of dicts."""
    if src.get("hf_id"):
        from datasets import load_dataset

        dd = load_dataset(src["hf_id"])
        return {name: list(split) for name, split in dd.items()}
    path = Path(src["local_path"])
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    if not path.exists():
        raise FileNotFoundError(path)
    import pandas as pd

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif path.suffix == ".json":
        df = pd.read_json(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"unsupported file type {path.suffix}")
    return {src.get("split", "train"): df.to_dict("records")}


def load_extra_sources(cfg: dict, offline: bool) -> tuple[dict[str, list[str]], dict[str, list[str]], list[dict]]:
    """Return (payloads_by_split, benign_texts_by_split, ready_chunks)."""
    payloads: dict[str, list[str]] = {"train": [], "test": []}
    benign: dict[str, list[str]] = {"train": [], "test": []}
    chunks: list[dict] = []
    for src in cfg["data"]["extra_sources"]:
        if not src.get("enabled"):
            continue
        if offline and src.get("hf_id"):
            log.warning("offline: skipping %s", src["name"])
            continue
        try:
            tables = _load_table(src)
        except Exception as exc:  # noqa: BLE001 — external data must never crash the build
            log.warning("could not load %s (%s: %s) — skipping", src["name"], type(exc).__name__, exc)
            continue
        for split_name, rows in tables.items():
            split = "test" if split_name in ("test", "validation") else "train"
            for i, row in enumerate(rows):
                text = str(row.get(src["text_col"], "")).strip()
                if not text:
                    continue
                label = int(row.get(src["label_col"], 0))
                if src["role"] == "payload":
                    (payloads if label == 1 else benign)[split].append(text[:400])
                else:
                    chunks.append({"chunk_id": f"{src['name']}-{split_name}-{i}", "doc_id": f"{src['name']}-{i}",
                                   "text": text, "clean_fact": "" if label else text, "payload": text if label else "",
                                   "label": label, "position": "unknown", "attack_family": src["name"] if label else "",
                                   "template_id": "", "canary": "", "payload_source": src["name"] if label else "",
                                   "payload_seen": False, "source": src["name"], "split": split, "variant": "external"})
        log.info("%s: %d/%d payloads, %d/%d benign, %d chunks", src["name"], len(payloads["train"]),
                 len(payloads["test"]), len(benign["train"]), len(benign["test"]), len(chunks))
    return payloads, benign, chunks


# ----------------------------------------------------------------------------- payload sampling

class PayloadSampler:
    """Draws payloads with no template/attack leakage across splits."""

    def __init__(self, cfg: dict, bipia_attacks: dict, extra_payloads: dict, rng: random.Random):
        self.rng = rng
        pool = synthetic_payload_pool(cfg["data"]["heldout_families"])
        self.synthetic = {"train": pool["train"], "test_unseen": pool["test"], "test_seen": pool["train"]}
        self.bipia = bipia_attacks
        self.extra = extra_payloads

    def sample(self, split: str, prefer_canary: bool = False) -> tuple[Payload, bool]:
        """Return (payload, seen_in_training)."""
        r = self.rng.random()
        doc_split = "train" if split in ("train", "val") else "test"
        if not prefer_canary and r < 0.25 and self.bipia.get(doc_split):
            category, goal = self.rng.choice(self.bipia[doc_split])
            return Payload(wrap_goal_as_injection(goal, self.rng), f"bipia:{category}", f"bipia:{category}",
                           source="bipia"), doc_split == "train"
        if not prefer_canary and r < 0.40 and self.extra.get(doc_split):
            return Payload(self.rng.choice(self.extra[doc_split]), "external", "external",
                           source="extra"), doc_split == "train"
        if doc_split == "train":
            fam, idx, tpl = self.rng.choice(self.synthetic["train"])
            return render_payload(fam, idx, tpl, self.rng), True
        # test: half unseen templates/families, half seen templates with fresh canaries
        seen = self.rng.random() < 0.5
        fam, idx, tpl = self.rng.choice(self.synthetic["test_seen" if seen else "test_unseen"])
        return render_payload(fam, idx, tpl, self.rng), seen


# ----------------------------------------------------------------------------- chunking

def make_splitter(cfg: dict):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(chunk_size=cfg["chunking"]["chunk_size"],
                                          chunk_overlap=cfg["chunking"]["chunk_overlap"],
                                          add_start_index=True)


def chunk_document(text: str, span: tuple[int, int] | None, splitter) -> list[dict]:
    """Split text; label a chunk 1 iff it contains >= 50% of the payload characters
    (or the entire chunk is payload). clean_fact = chunk with payload chars removed."""
    out = []
    for d in splitter.create_documents([text]):
        c_start = d.metadata["start_index"]
        c_end = c_start + len(d.page_content)
        label, clean = 0, d.page_content
        if span:
            ov_start, ov_end = max(c_start, span[0]), min(c_end, span[1])
            overlap = max(0, ov_end - ov_start)
            p_len = span[1] - span[0]
            label = int(overlap >= 0.5 * p_len or (overlap > 0 and overlap >= 0.5 * len(d.page_content)))
            if overlap > 0:
                rel_s, rel_e = ov_start - c_start, ov_end - c_start
                clean = (d.page_content[:rel_s] + " " + d.page_content[rel_e:]).strip()
                clean = " ".join(clean.split())
        out.append({"text": d.page_content, "label": label, "clean_fact": clean})
    return out


# ----------------------------------------------------------------------------- build

def assign_doc_splits(docs: list[dict], cfg: dict, rng: random.Random) -> None:
    val_frac, test_frac = cfg["data"]["val_frac"], cfg["data"]["test_frac"]
    for d in docs:
        if "fixed_split" in d:
            # BIPIA ships train/test; carve val out of its train split.
            d["split"] = "test" if d["fixed_split"] == "test" else ("val" if rng.random() < val_frac else "train")
            continue
        r = rng.random()
        d["split"] = "test" if r < test_frac else ("val" if r < test_frac + val_frac else "train")


def build(cfg: dict, offline: bool = False) -> dict:
    set_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    dcfg = cfg["data"]
    docs: list[dict] = []

    bipia_attacks: dict = {"train": [], "test": []}
    if dcfg["bipia"]["enabled"] and not offline:
        try:
            bipia_attacks, email_docs = load_bipia(cfg)
            docs += email_docs
        except Exception as exc:  # noqa: BLE001
            log.warning("BIPIA unavailable (%s) — continuing without it", exc)
    if dcfg["squad"]["enabled"] and not offline:
        try:
            docs += load_squad(cfg)
        except Exception as exc:  # noqa: BLE001
            log.warning("SQuAD unavailable (%s) — continuing without it", exc)
    if dcfg["synthetic_policies"]["enabled"]:
        docs += generate_policy_docs(dcfg["synthetic_policies"]["n_docs"], rng)
    if dcfg["hard_negatives"]["enabled"]:
        docs += generate_hard_negatives(dcfg["hard_negatives"]["n_docs"], rng)
    extra_payloads, extra_benign, external_chunks = load_extra_sources(cfg, offline)

    # Benign extra texts (e.g. normal user questions) become FAQ-style hard negatives.
    for split_name, texts in extra_benign.items():
        for i, t in enumerate(texts):
            docs.append({"doc_id": f"faq-{split_name}-{i:05d}", "text": f"Frequently asked: {t}", "qas": [],
                         "source": "extra_benign", "fixed_split": split_name})

    if not docs:
        raise RuntimeError("no documents loaded")
    assign_doc_splits(docs, cfg, rng)
    sampler = PayloadSampler(cfg, bipia_attacks, extra_payloads, rng)
    splitter = make_splitter(cfg)
    positions = dcfg["positions"]

    chunks: list[dict] = []
    rag_corpus: list[dict] = []
    eval_candidates: list[dict] = []

    for d in docs:
        is_test = d["split"] == "test"
        poison = rng.random() < dcfg["poison_rate"] and d["source"] != "extra_benign"
        # Clean variant (always in the detector dataset).
        for j, c in enumerate(chunk_document(d["text"], None, splitter)):
            chunks.append(_chunk_row(d, f"{d['doc_id']}::clean::{j}", c, None, "", "clean"))
        if not poison:
            if is_test:
                for j, c in enumerate(chunk_document(d["text"], None, splitter)):
                    rag_corpus.append(_corpus_row(d, f"{d['doc_id']}::{j}", c, None))
                eval_candidates += [_query(d, q, a, None, "") for q, a in d["qas"]]
            continue
        payload, seen = sampler.sample(d["split"], prefer_canary=is_test and rng.random() < 0.8)
        position = rng.choice(positions)
        poisoned, s, e = inject(d["text"], payload.text, position, rng)
        poisoned_chunks = chunk_document(poisoned, (s, e), splitter)
        for j, c in enumerate(poisoned_chunks):
            chunks.append(_chunk_row(d, f"{d['doc_id']}::poison::{j}", c, payload, position, "poison", seen))
        if is_test:
            for j, c in enumerate(poisoned_chunks):
                rag_corpus.append(_corpus_row(d, f"{d['doc_id']}::{j}", c, payload if c["label"] else None))
            eval_candidates += [_query(d, q, a, payload, position, seen) for q, a in d["qas"]]

    for c in external_chunks:
        chunks.append(c)

    # Sample the evaluation queries (one query per doc to keep queries independent).
    rng.shuffle(eval_candidates)
    ecfg = cfg["eval"]
    seen_docs: set[str] = set()
    attacks, benign = [], []
    for q in eval_candidates:
        if q["doc_id"] in seen_docs:
            continue
        if q["is_attack"]:
            if not q["canary"] and not ecfg["use_llm_judge_for_asr"]:
                continue
            if len(attacks) < ecfg["n_attack_queries"]:
                attacks.append(q)
                seen_docs.add(q["doc_id"])
        elif len(benign) < ecfg["n_benign_queries"]:
            benign.append(q)
            seen_docs.add(q["doc_id"])
    queries = attacks + benign
    for i, q in enumerate(queries):
        q["query_id"] = f"q{i:05d}"

    out_dir = Path(cfg["paths"]["processed_dir"])
    write_jsonl(out_dir / "chunks.jsonl", chunks)
    write_jsonl(out_dir / "rag_corpus.jsonl", rag_corpus)
    write_jsonl(out_dir / "rag_eval.jsonl", queries)
    stats = summarize(chunks, rag_corpus, queries)
    (out_dir / "manifest.json").write_text(json.dumps(stats, indent=2))
    return stats


def _chunk_row(d, chunk_id, c, payload: Payload | None, position, variant, seen=False) -> dict:
    lab = c["label"] if payload else 0
    return {"chunk_id": chunk_id, "doc_id": d["doc_id"], "text": c["text"], "clean_fact": c["clean_fact"],
            "payload": payload.text if (payload and lab) else "", "label": lab,
            "position": position if lab else "", "attack_family": payload.family if (payload and lab) else "",
            "template_id": payload.template_id if (payload and lab) else "",
            "canary": payload.canary if (payload and lab) else "",
            "payload_source": payload.source if (payload and lab) else "", "payload_seen": bool(seen and lab),
            "source": d["source"], "split": d["split"], "variant": variant}


def _corpus_row(d, chunk_id, c, payload: Payload | None) -> dict:
    return {"chunk_id": chunk_id, "doc_id": d["doc_id"], "text": c["text"], "clean_fact": c["clean_fact"],
            "label": c["label"] if payload else 0, "payload": payload.text if payload else "",
            "canary": payload.canary if payload else "", "source": d["source"]}


def _query(d, question, answer, payload: Payload | None, position, seen=False) -> dict:
    return {"doc_id": d["doc_id"], "question": question, "answer": answer, "is_attack": payload is not None,
            "payload": payload.text if payload else "",
            "canary": payload.canary if payload else "", "attack_family": payload.family if payload else "",
            "position": position, "payload_seen": bool(seen) if payload else False, "source": d["source"]}


def summarize(chunks, corpus, queries) -> dict:
    by_split = Counter((c["split"], c["label"]) for c in chunks)
    stats = {
        "chunks_total": len(chunks),
        "chunks_by_split_label": {f"{s}/{l}": n for (s, l), n in sorted(by_split.items())},
        "chunks_by_source": dict(Counter(c["source"] for c in chunks)),
        "poison_by_family": dict(Counter(c["attack_family"] for c in chunks if c["label"])),
        "poison_by_position": dict(Counter(c["position"] for c in chunks if c["label"])),
        "rag_corpus_chunks": len(corpus),
        "rag_corpus_poisoned": sum(c["label"] for c in corpus),
        "eval_queries": len(queries),
        "eval_attack_queries": sum(q["is_attack"] for q in queries),
        "eval_benign_queries": sum(not q["is_attack"] for q in queries),
    }
    # Sanity: every labelled poisoned chunk with a canary must actually contain it.
    bad = [c["chunk_id"] for c in chunks if c["label"] and c["canary"] and not canary_hit(c["text"], c["canary"])]
    stats["label_sanity_failures"] = len(bad)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--offline", action="store_true", help="skip BIPIA/HF downloads (synthetic data only)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    stats = build(cfg, offline=args.offline)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
