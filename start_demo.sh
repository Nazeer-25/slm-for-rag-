#!/usr/bin/env bash
# Context - Hybrid RAG Defense Workbench (macOS / Linux): starts the website on http://127.0.0.1:8000
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "The virtual environment .venv was not found. See 'Fresh installation' in README.md."
  exit 1
fi
export PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false
exec .venv/bin/python src/web_server.py --open-browser "$@"
