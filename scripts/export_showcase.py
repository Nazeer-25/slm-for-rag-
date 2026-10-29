"""Export a static, browseable snapshot of the website with the actual recorded runs.

    python scripts/export_showcase.py --output dist

The snapshot contains the datasets, Chroma collection counts, architecture, saved detector
and benchmark results, the six scenarios and up to 30 recorded experiments. Open
dist/index.html directly (no server needed) or host the folder for free (docs/HOSTING.md).
It cannot generate new answers.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import PATHS, Paths  # noqa: E402


def build_snapshot(paths: Paths) -> dict:
    from scenarios import build_scenarios
    from web_server import architecture, kb_dataset, results_payload
    from workspace import Workspace

    ws = Workspace(paths)
    traces = ws.history.all()
    status = ws.status()
    status.update(initialized=False, initializing=False, available_modes=[])
    return {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "kb": {name: kb_dataset(paths, name) for name in paths.dataset_files()},
        "architecture": architecture(ws),
        "scenarios": build_scenarios(paths) if paths.dataset_csv.exists() else [],
        "results": results_payload(paths),
        "history": ws.history.summaries(),
        "traces": {t["run_id"]: t for t in traces},
    }


def export(output: Path, paths: Paths = PATHS) -> Path:
    output = Path(output).resolve()
    if output == ROOT or output in ROOT.parents or output == paths.root.resolve():
        raise SystemExit(f"Refusing to export into {output}: choose a dedicated folder such as dist.")
    if output.exists() and any(output.iterdir()):
        if not (output / "snapshot.js").exists():
            raise SystemExit(f"Refusing to overwrite {output}: it is not empty and is not a previous export.")
        shutil.rmtree(output)
    shutil.copytree(paths.web_dir / "static", output / "static")
    html = (paths.web_dir / "index.html").read_text(encoding="utf-8")
    marker = '<script src="static/app.js"></script>'
    if marker not in html:
        raise RuntimeError("web/index.html no longer loads static/app.js; update export_showcase.py")
    (output / "index.html").write_text(html.replace(marker, '<script src="snapshot.js"></script>\n' + marker),
                                       encoding="utf-8")
    snapshot = build_snapshot(paths)
    (output / "snapshot.js").write_text("window.WORKBENCH_SNAPSHOT = " + json.dumps(snapshot) + ";\n",
                                        encoding="utf-8")
    if paths.benchmark_chart_png.exists():
        (output / "results").mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths.benchmark_chart_png, output / "results" / paths.benchmark_chart_png.name)
    (output / ".nojekyll").write_text("", encoding="utf-8")  # GitHub Pages: serve files as-is
    return output


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="dist")
    args = ap.parse_args(argv)
    out = export(Path(args.output) if Path(args.output).is_absolute() else ROOT / args.output)
    snap = json.loads((out / "snapshot.js").read_text(encoding="utf-8").split("=", 1)[1].rstrip(";\n"))
    print(f"Exported a static snapshot to {out}")
    print(f"  {len(snap['history'])} recorded runs, benchmark: {'yes' if snap['results']['benchmark'] else 'no'}, "
          f"detector metrics: {'yes' if snap['results']['detector'] else 'no'}")
    print(f"Open {out / 'index.html'} in a browser, or host the folder (see docs/HOSTING.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
