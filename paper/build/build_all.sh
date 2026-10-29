#!/bin/bash
# Full build: numbers -> figures -> equations -> report (two passes for the contents page) -> PDF preview.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
OUTDIR=${RENDER_DIR:-/tmp/report_render}
mkdir -p "$OUTDIR"
$PY paper_numbers.py > /dev/null
$PY make_figures.py > /dev/null
$PY make_equations.py > /dev/null
node build_report.js
soffice --headless --convert-to pdf --outdir "$OUTDIR" ../Project_Report_IPI_Detection.docx > /dev/null 2>&1
python3 toc_pages.py "$OUTDIR/Project_Report_IPI_Detection.pdf" headings.json toc_pages.json
node build_report.js toc_pages.json
python3 fix_docx.py ../Project_Report_IPI_Detection.docx
soffice --headless --convert-to pdf --outdir "$OUTDIR" ../Project_Report_IPI_Detection.docx > /dev/null 2>&1
python3 toc_pages.py "$OUTDIR/Project_Report_IPI_Detection.pdf" headings.json toc_check.json
cmp -s toc_pages.json toc_check.json && echo "contents page numbers verified" || echo "WARNING: page numbers shifted between passes"
pdfinfo "$OUTDIR/Project_Report_IPI_Detection.pdf" | grep Pages
