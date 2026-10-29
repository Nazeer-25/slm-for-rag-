"""Post-process the generated .docx: drop w:highlightCs, which docx-js emits but the
WordprocessingML schema does not allow (w:highlight itself is kept)."""
import re
import shutil
import sys
import tempfile
import zipfile

src = sys.argv[1]
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx").name
with zipfile.ZipFile(src) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename.endswith(".xml"):
            text = data.decode("utf-8")
            text, n = re.subn(r"<w:highlightCs\b[^>]*/>", "", text)
            if n:
                print(f"{item.filename}: removed {n} w:highlightCs")
            data = text.encode("utf-8")
        zout.writestr(item, data)
shutil.copymode(src, tmp)
shutil.move(tmp, src)
