"""Runs the supplementary experiments and builds the report.

  .venv\\Scripts\\python.exe supplementary\\run_all.py                    threshold sensitivity, score distributions, report
  .venv\\Scripts\\python.exe supplementary\\run_all.py --ablation         also the four-configuration ablation (hours on a CPU)
  .venv\\Scripts\\python.exe supplementary\\run_all.py --ablation --quick  smaller ablation (60 injected + 30 clean per test set)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablation_four_configs  # noqa: E402
import make_report  # noqa: E402
import score_distributions  # noqa: E402
import threshold_sensitivity  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Supplementary experiments and report")
    ap.add_argument("--ablation", action="store_true", help="also run the four-configuration ablation (needs Ollama)")
    ap.add_argument("--quick", action="store_true", help="smaller ablation sample")
    for opt in ("project-root", "out-dir", "data-dir", "detector-dir", "regular-predictions", "challenge-predictions",
                "sanitizer", "ollama-host", "ollama-model"):
        ap.add_argument(f"--{opt}")
    a = ap.parse_args(argv)
    opt = lambda *names: [x for n in names if getattr(a, n.replace("-", "_")) for x in (f"--{n}", getattr(a, n.replace("-", "_")))]  # noqa: E731
    print("== Threshold sensitivity (validation selects, test sets evaluate)")
    threshold_sensitivity.main(opt("project-root", "out-dir", "data-dir", "detector-dir", "regular-predictions", "challenge-predictions"))
    print("\n== Score distributions (saved predictions)")
    score_distributions.main(opt("project-root", "out-dir", "regular-predictions", "challenge-predictions"))
    if a.ablation:
        print("\n== Four-configuration ablation")
        ablation_four_configs.main(opt("project-root", "out-dir", "data-dir", "detector-dir", "regular-predictions", "sanitizer",
                                       "ollama-host", "ollama-model") + (["--quick"] if a.quick else []))
    print("\n== Report")
    make_report.main(opt("project-root", "out-dir"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
