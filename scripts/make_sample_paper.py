"""Writes a fake DDU-style OS paper to data/raw/OS/OS_2025_11_END.pdf so you can try the pipeline
before collecting real papers. Digital text only, so `extract` and `split` work without an API key.
Delete it once your real papers are in data/raw/."""
import sys
from pathlib import Path
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exam_prep import config  # noqa: E402
from tests.conftest import PAGE1, PAGE2_OCR  # noqa: E402

out = config.RAW_DIR / "OS" / "OS_2025_11_END.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
doc = pymupdf.open()
for text in (PAGE1, PAGE2_OCR):
    doc.new_page().insert_text((40, 50), text, fontsize=8)
doc.save(out)
print(f"wrote {out}")
