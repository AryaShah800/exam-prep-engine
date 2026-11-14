"""Step 3: syllabus -> topic taxonomy (unit -> topics), saved as data/taxonomy/<SUBJECT>.json.

Put the syllabus in data/syllabus/<SUBJECT>.txt (copy-paste from the DDU syllabus PDF) or <SUBJECT>.pdf.
The JSON is meant to be edited by hand afterwards: merge, rename or split topics until they match how
exam questions are actually asked. Topic IDs look like OS-U3-T2 (subject, unit 3, topic 2).
"""
import json
from pathlib import Path
from string import Template
from typing import Optional
import pymupdf
from . import config
from .extract import page_is_scanned, render_page_png
from .llm import LLM, load_prompt
from .schema import LLMPage, LLMSyllabus, Taxonomy, Topic


def taxonomy_path(subject: str) -> Path:
    return config.TAXONOMY_DIR / f"{subject.upper()}.json"


def read_syllabus_text(subject: str, vision_llm: Optional[LLM] = None) -> str:
    base = config.SYLLABUS_DIR / subject.upper()
    for ext in (".txt", ".md"):
        if base.with_suffix(ext).exists():
            return base.with_suffix(ext).read_text(encoding="utf-8")
    pdf = base.with_suffix(".pdf")
    if not pdf.exists():
        raise FileNotFoundError(f"Put the syllabus at {base}.txt or {base}.pdf")
    chunks = []
    ocr_prompt = None
    with pymupdf.open(pdf) as doc:
        for page in doc:
            text = page.get_text("text", sort=True)
            if page_is_scanned(text):
                if vision_llm is None:
                    raise RuntimeError("Scanned syllabus PDF needs GEMINI_API_KEY, or paste it into a .txt file.")
                ocr_prompt = ocr_prompt or load_prompt("ocr")[0]
                text = vision_llm.generate_json(ocr_prompt, LLMPage, images=[render_page_png(page)]).text
            chunks.append(text)
    return "\n".join(chunks)


def build_taxonomy(subject: str, syllabus_text: str, llm: LLM) -> Taxonomy:
    template, _ = load_prompt("syllabus")
    prompt = Template(template).safe_substitute(subject=subject.upper(), syllabus_text=syllabus_text)
    result = llm.generate_json(prompt, LLMSyllabus)
    topics = []
    for unit in sorted(result.units, key=lambda u: u.unit_no):
        for n, t in enumerate(unit.topics, start=1):
            topics.append(Topic(topic_id=f"{subject.upper()}-U{unit.unit_no}-T{n}", unit_no=unit.unit_no, unit_title=unit.title.strip(), name=t.name.strip(), keywords=[k.strip() for k in t.keywords if k.strip()]))
    if not topics:
        raise ValueError("Gemini returned no topics; check the syllabus text.")
    return Taxonomy(subject=subject.upper(), subject_name=result.subject_name.strip(), topics=topics)


def save_taxonomy(tax: Taxonomy) -> Path:
    path = taxonomy_path(tax.subject)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tax.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_taxonomy(subject: str) -> Taxonomy:
    path = taxonomy_path(subject)
    if not path.exists():
        raise FileNotFoundError(f"No taxonomy for {subject}; run `python -m exam_prep syllabus --subject {subject}`")
    tax = Taxonomy.model_validate_json(path.read_text(encoding="utf-8"))
    dupes = [i for i in [t.topic_id for t in tax.topics] if [t.topic_id for t in tax.topics].count(i) > 1]
    if dupes:
        raise ValueError(f"{path.name}: duplicate topic_id {sorted(set(dupes))}; give each topic a unique ID")
    return tax


def run(subject: str, llm: LLM, vision_llm: Optional[LLM] = None, overwrite: bool = False) -> Taxonomy:
    path = taxonomy_path(subject)
    if path.exists() and not overwrite:
        print(f"  {path.name} exists (edit it by hand, or pass --overwrite to rebuild)")
        return load_taxonomy(subject)
    tax = build_taxonomy(subject, read_syllabus_text(subject, vision_llm), llm)
    save_taxonomy(tax)
    units = len({t.unit_no for t in tax.topics})
    print(f"  {path.name}: {units} units, {len(tax.topics)} topics. Review and edit it before tagging.")
    return tax
