"""Paper metadata: parsed from file names, optionally overridden by data/papers.csv.

File name convention:  <SUBJECT>_<YYYY>_<MM>_<EXAM>.pdf   e.g.  OS_2025_11_END.pdf, DBMS_2024_09_MID.pdf
EXAM is one of MID, END, REM (remedial), OTHER.
Papers from other universities live in data/external/<SUBJECT>/ and carry a source prefix:
<SRC>-<SUBJECT>_<YYYY>_<MM>_<EXAM>.pdf, e.g. BITM-AI_2019_11_END.pdf (provenance in data/external/SOURCES.csv).
They are used for retrieval and practice only; DDU-only analyses filter on source == "DDU".

papers.csv (optional) has the header: file,subject,year,month,exam_type
Use it for papers whose names you don't want to rename.
"""
import csv, re
from pathlib import Path
from typing import Optional
from .schema import PaperMeta
from . import config

DDU = "DDU"
_NAME = re.compile(r"^(?:(?P<source>[A-Za-z]+)-)?(?P<subject>[A-Za-z0-9]+)_(?P<year>\d{4})_(?P<month>\d{1,2})_(?P<exam>MID|END|REM|REMEDIAL|OTHER)$", re.I)
_EXAM = {"mid": "mid", "end": "end", "rem": "remedial", "remedial": "remedial", "other": "other"}


def parse_filename(path: Path) -> Optional[PaperMeta]:
    m = _NAME.match(path.stem)
    if not m:
        return None
    subject, source = m["subject"].upper(), (m["source"] or DDU).upper()
    year, month, exam = int(m["year"]), int(m["month"]), _EXAM[m["exam"].lower()]
    prefix = "" if source == DDU else f"{source}-"
    return PaperMeta(paper_id=f"{prefix}{subject}_{year}_{month:02d}_{m['exam'].upper()}", subject=subject, source=source, year=year, month=month, exam_type=exam, source_file=str(path))


def load_manifest(manifest: Optional[Path] = None) -> dict[str, dict]:
    manifest = manifest or config.MANIFEST
    if not manifest.exists():
        return {}
    with manifest.open(newline="", encoding="utf-8") as f:
        return {Path(row["file"]).name: row for row in csv.DictReader(f)}


def paper_meta(path: Path, manifest: Optional[dict[str, dict]] = None) -> PaperMeta:
    row = (manifest or {}).get(path.name)
    if row:
        subject = row["subject"].strip().upper()
        year = int(row["year"])
        month = int(row["month"]) if row.get("month") else None
        exam = _EXAM.get((row.get("exam_type") or "end").strip().lower(), "other")
        tag = {"mid": "MID", "end": "END", "remedial": "REM", "other": "OTHER"}[exam]
        pid = f"{subject}_{year}_{month:02d}_{tag}" if month else f"{subject}_{year}_{tag}"
        return PaperMeta(paper_id=pid, subject=subject, year=year, month=month, exam_type=exam, source_file=str(path))
    meta = parse_filename(path)
    if meta is None:
        raise ValueError(f"Can't read metadata from {path.name}. Rename it like OS_2025_11_END.pdf or add a row to {config.MANIFEST.name}.")
    return meta


def discover_papers(raw_dir: Optional[Path] = None, subject: Optional[str] = None, external: bool = True, source: Optional[str] = None) -> list[PaperMeta]:
    """DDU papers from data/raw, plus other universities' papers from data/external unless external=False.
    source filters by origin: "DDU", "external" (everything not DDU) or a code such as "BITM"."""
    raw_dir = raw_dir or config.RAW_DIR
    manifest = load_manifest()
    metas, errors = [], []
    pdfs = sorted(raw_dir.rglob("*.pdf"))
    ext_pdfs = sorted(config.EXTERNAL_DIR.rglob("*.pdf")) if external and config.EXTERNAL_DIR.exists() else []
    for pdf in pdfs + ext_pdfs:
        try:
            meta = paper_meta(pdf, manifest)
        except ValueError as e:
            errors.append(str(e))
            continue
        if (pdf in ext_pdfs) != (meta.source != DDU):
            errors.append(f"{pdf.name}: papers in data/external need a source prefix (BITM-AI_...), and DDU papers in data/raw must not have one")
            continue
        if source and not (meta.source != DDU if source.lower() == "external" else meta.source == source.upper()):
            continue
        if subject is None or meta.subject == subject.upper():
            metas.append(meta)
    ids = [m.paper_id for m in metas]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Two files map to the same paper_id: {sorted(dupes)}")
    for e in errors:
        print(f"  skipped: {e}")
    return sorted(metas, key=lambda m: m.paper_id)
