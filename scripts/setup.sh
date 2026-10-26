#!/usr/bin/env bash
# One-shot local setup: venv + dependencies + spaCy model + .env template.
# Usage: bash scripts/setup.sh            (from the repository root)
set -euo pipefail
PY=${PYTHON:-python3}
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
if [[ "${CPU_ONLY:-0}" == "1" ]]; then
  # ~200 MB CPU wheel instead of the ~3 GB CUDA build
  pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
pip install -r requirements.txt
python -m spacy download en_core_web_sm
[[ -f .env ]] || cp .env.example .env
python - <<'PY'
import spacy, torch, transformers, langchain_chroma
spacy.load("en_core_web_sm")
print(f"OK  torch={torch.__version__} cuda={torch.cuda.is_available()}  transformers={transformers.__version__}")
PY
echo "Setup complete. Edit .env, then: source .venv/bin/activate"
