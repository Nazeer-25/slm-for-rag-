"""Find the page of every heading in the rendered report PDF (for the contents page).

Usage: python toc_pages.py report.pdf headings.json toc_pages.json
Displayed page = physical page - 1 (the title page is unnumbered; numbering starts at 1 on the contents page).
"""
import json
import re
import subprocess
import sys

pdf, headings_file, out_file = sys.argv[1:4]
pages = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout).group(1))
norm = lambda s: re.sub(r"\s+", " ", s).strip()
text = [norm(subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), "-layout", pdf, "-"],
                            capture_output=True, text=True).stdout) for i in range(1, pages + 1)]
headings = json.load(open(headings_file))
# skip the title page and the contents pages: start at the first page that has the first heading but no "Contents"
start = next(i for i in range(1, pages) if norm(headings[0]) in text[i] and "Contents" not in text[i])
result, cur = {}, start
for h in headings:
    key = norm(h)
    found = next((i for i in range(cur, pages) if key in text[i]), None)
    if found is None:
        sys.exit(f"heading not found in PDF: {h!r}")
    result[h] = found  # physical index is 0-based, so displayed page = (found + 1) - 1
    cur = found
json.dump(result, open(out_file, "w"), indent=1, ensure_ascii=False)
print(f"{len(result)} headings located across {pages} pages; last heading on page {max(result.values())}")
