import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import _testkit as tk

sys.path.insert(0, str(tk.ROOT / "scripts"))
import export_showcase  # noqa: E402

Q = "What is the Cedar division refund policy?"


class ExportShowcaseTest(unittest.TestCase):
    def test_static_snapshot_contains_recorded_runs(self):
        paths = tk.trained_workspace()
        try:
            ws = tk.offline_workspace(paths)
            ws.initialize()
            ws.run(Q, "hybrid_defense", "enterprise_poisoned", 3)
            out = export_showcase.export(Path(tempfile.mkdtemp()) / "dist", paths)
            html = (out / "index.html").read_text(encoding="utf-8")
            self.assertLess(html.index("snapshot.js"), html.index("static/app.js"))
            for f in ("static/app.js", "static/styles.css", ".nojekyll"):
                self.assertTrue((out / f).exists(), f)
            snap = json.loads((out / "snapshot.js").read_text(encoding="utf-8").split("=", 1)[1].rstrip(";\n"))
            self.assertEqual(len(snap["history"]), 1)
            self.assertIn(snap["history"][0]["run_id"], snap["traces"])
            self.assertEqual(len(snap["kb"]["dataset"]["rows"]), 200)
            self.assertEqual(len(snap["scenarios"]), 6)
            self.assertFalse(snap["status"]["initialized"])
        finally:
            tk.remove_workspace(paths)


class ExportSafetyTest(unittest.TestCase):
    def test_refuses_to_overwrite_folders_that_are_not_exports(self):
        keep = Path(tempfile.mkdtemp())
        (keep / "notes.md").write_text("important", encoding="utf-8")
        for target in (keep, tk.ROOT, tk.ROOT / "docs"):
            with self.assertRaises(SystemExit):
                export_showcase.export(target)
        self.assertTrue((keep / "notes.md").exists())
        self.assertTrue((tk.ROOT / "docs" / "DEMO_GUIDE.md").exists())


@unittest.skipIf(importlib.util.find_spec("streamlit") is None, "streamlit not installed")
class StreamlitAppTest(unittest.TestCase):
    def test_dashboard_renders_without_errors(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(tk.ROOT / "app.py"), default_timeout=120).run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertEqual([t.label for t in at.tabs], ["Live playground", "Knowledge base", "Results & evidence"])


if __name__ == "__main__":
    unittest.main()
