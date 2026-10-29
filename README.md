# Context / Hybrid RAG Defense Workbench

A local research demonstration of **A Lightweight Hybrid Defense Framework Against Indirect Prompt Injection in RAG Systems**. The website makes retrieval, classification, quarantine and generation inspectable. All model answers come from real inference by a local model. No paid API key is needed.

```
Question → MiniLM embedding → Chroma top-k → Layer 1 detector ─┬─ P(injection) < 0.50 → clean bypass ─┐
                                                               └─ P(injection) ≥ 0.50 → Layer 2 ──────┴→ Ollama (llama3.2:3b) → answer
                                                                                        (removes assistant-directed
                                                                                         clauses, keeps the facts)
```

The filter sits strictly between vector retrieval and generation. It never needs the language model's weights, logits or hidden states, and the hybrid mode adds no extra LLM calls.

## Start on this computer

After the one-time installation below, open Command Prompt in this project folder:

```bat
start_demo.bat
```

Open **http://127.0.0.1:8000** (the script opens it for you). Keep the terminal and Ollama running. Choose **Initialize workspace**, then **Presentation guide** for six ready-to-use scenarios. The first model load takes longer than later queries.

Once installed, the trained detector, the cached embedding model, the English parser and the source CSVs stay on disk. You do not need to retrain just to start the website.

On macOS or Linux, use `./start_demo.sh` instead.

## Website screens

| Screen | What you can demonstrate |
|---|---|
| Live playground | Four defense modes; corpus selection; temporary custom poison; real retrieval, flags, removed clauses, answer and latency |
| Knowledge base | All CSV datasets; actual Chroma collection counts; labels; search; raw text; clean evaluation references and attacker targets |
| System architecture | Question → MiniLM → Chroma → Layer 1 → clean bypass or Layer 2 → Ollama |
| Results & evidence | Saved detector metrics, actual query history, downloadable traces, benchmark metrics when generated |
| Presentation guide | Six staged demonstrations, expected facts and interpretation |

The presentation uses a restrained forest-green and ivory theme, serif headings and readable document views. It has no frontend build step, CDN fonts or external JavaScript dependencies.

## Fresh installation on another Windows PC

Python 3.11 is recommended (3.10 to 3.13 work). Use the virtual environment to avoid permission errors in `C:\Python311\Lib\site-packages`.

```bat
git clone https://github.com/Nazeer-25/slm-for-rag-.git
cd slm-for-rag-
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m spacy download en_core_web_sm
.venv\Scripts\python.exe src\data_generator.py
.venv\Scripts\python.exe src\train_layer1.py
```

The source CSVs are already in the repository; `data_generator.py` regenerates identical files. `train_layer1.py` downloads `distilbert-base-uncased` once (about 270 MB), fine-tunes it on the 140 training examples (a few minutes on a typical CPU), evaluates it on the 60 test examples and saves the result.

Install [Ollama for Windows](https://ollama.com/download/windows), open it, then run:

```bat
ollama pull llama3.2:3b
ollama run llama3.2:3b "Reply with READY."
start_demo.bat
```

If `ollama` is not recognized in an older terminal, open a new terminal or use `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`. The smoke command checks actual generation; seeing a model in `ollama list` alone does not establish runtime health. The website runs the same check during **Initialize workspace**.

The first run needs internet to install dependencies and download the models; the embedding model downloads into `models/embedding_cache` on the first **Initialize workspace**. After caching and indexing, use `set HF_HUB_OFFLINE=1` before `start_demo.bat` for Hugging Face offline mode. Ollama still needs to be running locally. Synthetic data generation needs no internet.

<details>
<summary>macOS / Linux</summary>

```bash
git clone https://github.com/Nazeer-25/slm-for-rag-.git && cd slm-for-rag-
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_sm
.venv/bin/python src/data_generator.py
.venv/bin/python src/train_layer1.py
# install Ollama from https://ollama.com/download, then:
ollama pull llama3.2:3b && ollama run llama3.2:3b "Reply with READY."
./start_demo.sh
```
On Linux, `pip install torch` pulls the large CUDA build by default. For a CPU-only machine, install the CPU wheel first: `.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.
</details>

Optional public reference dataset import:

```bat
.venv\Scripts\python.exe src\data_generator.py --hf-dataset deepset/prompt-injections
```

This keeps a balanced sample of 200 examples (100 injection, 100 benign) in `data/external_dataset.csv`; use `--external-limit` to change it. External examples are stored separately and are never silently substituted for synthetic training data. Close CSVs in Excel before regenerating them if Windows reports a file lock; the generator tells you when that happens.

## Data and models

| File / collection | Role |
|---|---|
| `data/dataset.csv` | 200 synthetic enterprise chunks: 100 benign, 100 poisoned; five policy topics (refund, travel, password, leave, data retention) across 20 divisions |
| `data/train.csv` | 140 examples, balanced, with clean/poison pairs kept together |
| `data/test.csv` | 60 held-out examples, balanced (6 pairs per topic); shares synthetic template families with training |
| `data/external_dataset.csv` | Optional imported reference examples (created by `--hf-dataset`) |
| `data/chroma/` | One persistent Chroma database |
| `enterprise_policies` | Enterprise vectors (both variants; the corpus selector filters clean or poisoned) |
| `external_examples` | External-reference vectors when available |
| `live_payloads` | Temporary per-run custom documents; deleted in `finally` after the run |
| `models/layer1_detector/` | Fine-tuned DistilBERT model, tokenizer and detector configuration |
| `models/embedding_cache/` | all-MiniLM-L6-v2 cache, 384-dimensional embeddings |
| `data/demo_history.json` | Up to 30 actual website experiments |
| `results/` | Detector evaluation, benchmark reports and chart |

The three source CSVs are committed to the repository. Everything else in this table is created on your machine by the scripts and is not committed.

Every clean policy contains exactly one number (unique per topic), so "the answer contains the expected fact" is unambiguous. Every poisoned twin carries one of six injection families (direct override, role marker, delimiter breakout, phishing link, fake update, exfiltration) at the start, middle or end of the document, or **inline**: appended as a clause to the fact sentence itself, so only clause-level removal can keep the fact.

## How the defense works

Layer 1 scans overlapping token windows (256 tokens, 64 overlap; the chunk score is the maximum over its windows) and routes at injection probability 0.50. Layer 2 combines spaCy syntax with contextual rules to remove assistant-directed clauses: role and system markers, delimiter breakouts, clauses addressed to the AI, attempts to override the question, dictating the assistant's output, redirecting the user, and exfiltration links. It also reads common obfuscations (leetspeak, hyphenation, `IGNORE_PREVIOUS`) and a few multilingual overrides. Ordinary enterprise instructions such as “Never share your password” are retained when the rules recognize their legitimate context, and the website lists them as retained instructions. Clean reference text is used only to score preservation, never to decide what to sanitize or to answer a query; Chroma metadata holds only identifiers and retrieval filters.

| Mode | What happens | Extra LLM calls |
|---|---|---|
| `no_defense` | Retrieved chunks go into the prompt unchanged | none |
| `delimiter_defense` | Chunks are wrapped in `<context>` tags and the model is told to treat them as data | none |
| `llm_judge_defense` | The LLM screens every retrieved chunk; flagged chunks are excluded | one per retrieved chunk |
| `hybrid_defense` | Layer 1 scores every chunk; flagged chunks go to Layer 2; the rest bypass it | none |

## Benchmark and verification

On an 8 GB PC, stop the website and Streamlit with Ctrl+C before benchmarking so only one copy of the models is loaded. Keep Ollama running. The website reads the results files when you open **Results & evidence**, so start it again afterwards to view them.

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe src\evaluate_all.py --limit 2 --top-k 1
.venv\Scripts\python.exe src\evaluate_all.py
```

The tests run offline in well under a minute. They use stand-ins for the embedding model and the LLM, so they need neither Ollama nor model downloads.

The second command is a small smoke benchmark. The last runs the full evaluation: the 30 held-out test questions in all four modes, each against both the poisoned and the clean corpus (240 runs). It can take substantial time on CPU. It writes `results/benchmark_runs.json`, `results/benchmark_results.json` and `results/benchmark_comparison.png`. Mode order is randomized per case and a model warm-up is excluded from timing.

ASR uses known attacker target-string matches. Benign utility counts expected numeric facts in answers. FPR counts benign retrieved chunks flagged or excluded. ROUGE-L measures context preservation against the clean reference. These are explicit proxies: a quoted target can match without obedience, and an answer without a target can still be wrong. Failed queries have explicit counts and are excluded from metric denominators.

The detector usually scores perfectly on the 60 synthetic test examples. That is not evidence of robustness against unseen injection styles. Neither layer is a complete security boundary: a chunk that Layer 1 does not flag bypasses Layer 2. Use the external corpus for exploration; it has no clean reference or known target for every record.

## Other entry points

```bat
.venv\Scripts\python.exe -m streamlit run app.py
.venv\Scripts\python.exe src\rag_pipeline.py "What is the Cedar division refund policy?" --mode hybrid_defense --variant poisoned
.venv\Scripts\python.exe scripts\export_showcase.py --output dist
```

Streamlit provides the alternative dashboard. The custom website on port 8000 is the recommended presentation UI. The command-line pipeline prints the full trace: retrieved chunks, Layer 1 scores, removed and retained clauses, the answer and latency. Add `--json` for the raw trace or `--custom-poison "..."` to test your own payload. The export is a static, browseable snapshot with actual recorded runs; it cannot generate new answers. Open `dist\index.html` directly or host it for free.

Default generation is local Ollama. The optional remote provider is enabled explicitly with `LLM_PROVIDER=openai`, `OPENAI_API_KEY` and optionally `OPENAI_MODEL`; that provider may incur API charges. The website's default model labels describe the Ollama configuration.

Read [the presentation script](docs/DEMO_GUIDE.md) and [free hosting instructions](docs/HOSTING.md).

## Configuration

All settings have working defaults. Override them with environment variables (`set NAME=value` in Command Prompt before `start_demo.bat`) or with `NAME=value` lines in a `.env` file in the project folder (see `.env.example`).

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2:3b` | Local generation model (pull it first) |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Where Ollama listens |
| `LLM_PROVIDER` | `ollama` | `openai` switches to the remote provider |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | -, `gpt-4o-mini` | Only for `LLM_PROVIDER=openai` |
| `TOP_K` | `3` | Default number of retrieved chunks |
| `LLM_TIMEOUT_S` | `300` | Per-request timeout; the first model load can be slow |
| `WORKBENCH_HOST`, `WORKBENCH_PORT` | `127.0.0.1`, `8000` | Website address |
| `HF_HUB_OFFLINE` | unset | `1` = load Hugging Face models from the local cache only |

## Project structure

```
start_demo.bat / start_demo.sh   launch the website
app.py                           Streamlit dashboard
src/config.py                    paths, model names, settings
src/data_generator.py            synthetic corpus and optional external import
src/train_layer1.py              fine-tune and evaluate the Layer 1 detector
src/layer1_detector.py           windowed DistilBERT detector (routes at 0.50)
src/layer2_quarantine.py         clause-level sanitizer (spaCy + contextual rules)
src/embeddings.py, vector_store.py  MiniLM embeddings and the Chroma collections
src/llm_client.py                Ollama (default) and optional OpenAI client
src/rag_pipeline.py              the four modes, ground-truth annotation, CLI
src/workspace.py                 loads components once; status for the UIs
src/evaluate_all.py, metrics.py  benchmark and metric definitions
src/web_server.py, web/          website backend and front end
src/scenarios.py, history.py     presentation scenarios, experiment history
scripts/export_showcase.py       static snapshot export
tests/                           offline unittest suite
docs/                            presentation script, hosting guide
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `start_demo.bat` says `.venv` was not found | Run the installation commands above in this folder |
| Initialize: "Cannot reach Ollama" | Open the Ollama app, then check `ollama run llama3.2:3b "Reply with READY."` |
| Initialize: model not pulled | `ollama pull llama3.2:3b` |
| Initialize: "No trained detector" | `.venv\Scripts\python.exe src\train_layer1.py` (only the hybrid mode needs it) |
| Initialize: embedding model not cached | Connect to the internet once and unset `HF_HUB_OFFLINE` |
| "Port 8000 is already in use" | The website is already running in another window, or start it with `set WORKBENCH_PORT=8001` |
| "Cannot write data\dataset.csv" | Close the CSV in Excel and run the generator again |
| The first query is slow | Ollama is loading the model into memory; later queries are faster |
| A scenario's attack does not succeed | Small models do not obey every payload; the guide says how to present that |

The earlier research-scale version of this repository (BIPIA/SQuAD data, QLoRA option, Colab notebook) is preserved in the git history at commit `ab2e389`.
