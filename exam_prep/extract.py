"""Step 1: PDF -> page text.

Digital pages use the PDF text layer (PyMuPDF, free and instant). A page with almost no text layer is a
scan: it is rendered to PNG and transcribed by Gemini vision, which also describes any diagram or table
in words so it becomes searchable later. Output: data/interim/pages/<paper_id>.jsonl
"""
from pathlib import Path
from typing import Optional
import re
import pymupdf
from . import config
from .store import read_jsonl, write_jsonl
from .llm import LLM, load_prompt
from .schema import LLMPage, PageText, PaperMeta


HEX_WATERMARK = re.compile(r"[0-9A-F]{16,}")   # runs of upper-case hex: per-copy IDs printed as a watermark
# AKTU prints a paper code and the downloader's IP address in the page gutter: "QP23DP1_290     117.55.242.132"
IP_WATERMARK = re.compile(r"\bQP\d{2}\w*\b|(?<=\s{3})\d{1,3}(?:\.\d{1,3}){3}(?=\s{3}|\s*$)")


def clean_text_layer(text: str) -> str:
    return "\n".join(IP_WATERMARK.sub("", HEX_WATERMARK.sub("", ln)).rstrip() for ln in text.splitlines()).strip()


def page_is_scanned(text: str, min_chars: int = config.SCANNED_PAGE_MIN_CHARS) -> bool:
    return len("".join(text.split())) < min_chars


def render_page_png(page: "pymupdf.Page", dpi: int = config.OCR_DPI) -> bytes:
    return page.get_pixmap(dpi=dpi).tobytes("png")


def extract_paper(meta: PaperMeta, vision_llm: Optional[LLM] = None, force_ocr: bool = False) -> list[PageText]:
    pages: list[PageText] = []
    ocr_prompt = None
    with pymupdf.open(meta.source_file) as doc:
        for i, page in enumerate(doc, start=1):
            text = clean_text_layer(page.get_text("text", sort=True))
            if not force_ocr and not page_is_scanned(text):
                pages.append(PageText(paper_id=meta.paper_id, page=i, method="text", text=text.strip()))
                continue
            if vision_llm is None:
                raise RuntimeError(f"{meta.paper_id} page {i} looks scanned but no Gemini key is configured for OCR.")
            if ocr_prompt is None:
                ocr_prompt, _ = load_prompt("ocr")
            png = render_page_png(page)
            out = vision_llm.generate_json(ocr_prompt, LLMPage, images=[png])
            pages.append(PageText(paper_id=meta.paper_id, page=i, method="ocr", text=out.text.strip(), figures=out.figures))
    return pages


def pages_path(paper_id: str) -> Path:
    return config.PAGES_DIR / f"{paper_id}.jsonl"


def run(metas: list[PaperMeta], vision_llm: Optional[LLM], overwrite: bool = False, force_ocr: bool = False) -> dict[str, int]:
    stats = {"papers": 0, "skipped": 0, "text_pages": 0, "ocr_pages": 0}
    for meta in metas:
        out = pages_path(meta.paper_id)
        if out.exists() and not overwrite:
            stats["skipped"] += 1
            continue
        pages = extract_paper(meta, vision_llm, force_ocr=force_ocr)
        write_jsonl(out, pages)
        stats["papers"] += 1
        stats["text_pages"] += sum(p.method == "text" for p in pages)
        stats["ocr_pages"] += sum(p.method == "ocr" for p in pages)
        print(f"  {meta.paper_id}: {len(pages)} pages ({sum(p.method == 'ocr' for p in pages)} via Gemini vision)")
    return stats


def load_pages(paper_id: str) -> list[PageText]:
    return read_jsonl(pages_path(paper_id), PageText)
