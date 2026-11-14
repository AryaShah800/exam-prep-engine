"""Step 2: page text -> one row per question (or sub-part).

Two splitters:
- rules: free and instant; reads "Q.1", "(a)", "OR" and marks like "[5]" / "(07)" / "5 M".
- llm:   Gemini with structured output, for layouts the rules can't read. Every returned question is checked
         against the paper text, so a rephrased or invented question is flagged instead of trusted.
- auto (default): rules first; falls back to llm when the rules result fails a sanity check.
"""
import re
from string import Template
from typing import Optional
from . import config
from .extract import load_pages
from .store import read_jsonl, write_jsonl
from .llm import LLM, load_prompt
from .schema import LLMSplit, PageText, PaperMeta, Question

Q_START = re.compile(r"^\s*Q(?:ue(?:s(?:tion)?)?)?\s*[.\-:]?\s*(\d{1,2})\s*[.):\-]?\s*(.*)$", re.I)
NUM_START = re.compile(r"^\s*(\d{1,2})\s*[.)]\s+(.*)$")
LETTER = re.compile(r"^\s*(?:\(\s*([a-h])\s*\)|([a-h])\s*[.)])\s*(.*)$", re.I)
ROMAN = re.compile(r"^\s*(?:\(\s*(i{1,3}|iv|v|vi{1,3}|ix|x)\s*\)|(i{1,3}|iv|v|vi{1,3}|ix|x)\s*[.)])\s+(.*)$", re.I)
OR_LINE = re.compile(r"^\s*[-–—(\[]*\s*OR\s*[-–—)\]]*\s*$")
MARKS_END = re.compile(r"(?:[\[(]\s*(\d{1,2}(?:\.5)?)\s*(?:M|Marks?)?\s*[\])]|\b(\d{1,2})\s*(?:M|Marks)\.?)\s*$", re.I)
NOISE = re.compile(r"^\s*(?:page\s*\d+(?:\s*(?:of|/)\s*\d+)?|\d{1,2}|p\.?\s*t\.?\s*o\.?|[-*_=+\s]+|[RUANEC]|\*+\s*best of luck\s*\*+|all the best)\s*$", re.I)
INSTRUCTION_STEM = re.compile(r"^(?:attempt|answer|do as directed)\b", re.I)
# DDU papers print the course outcome and Bloom level left of each sub-part: "CO1  R  (a) ...", "CO3(A)   (a) ..."
CO_PREFIX = re.compile(r"^\s*(CO\s*\d+)\s*(?:\(\s*([RUANEC])\s*\)|([RUANEC])(?=\s))?\s*")
MARGIN_CODE = re.compile(r"\bCO\s*\d+\s*(?:\(\s*[RUANEC]\s*\))?(?:\s+[RUANEC]\b(?=\s))?|(?<=\s)\+(?=\s)")
PAGE_FOOTER = re.compile(r"\s*\bpage\s+\d+\s+of\s+\d+\b\s*", re.I)   # footer merged into a text line by the layout
ZERO_WIDTH = re.compile("[​‌‍﻿]")
NESTED_INDENT = 4      # a label indented more than this past the shallowest label is a list inside a sub-part


BRACKET_MARKS = re.compile(r"[\[(]\s*\d{1,2}(?:\.5)?\s*[\])]|\b\d\s*\+(?=\s)")
VERIFY_CHUNK, VERIFY_MIN = 20, 0.8
FRAGMENT_CHARS = 60      # a model-split item shorter than this with no marks is merged into the previous item


def _locate(qnorm: str, page_norms: list[tuple[int, str]]) -> Optional[int]:
    """Page holding the question: >= 80% of its 20-char chunks appear there. Tolerates layout tokens printed
    between the words (marks, margin codes) but not rephrased or invented text. None if no page qualifies."""
    if not qnorm:
        return None
    chunks = [qnorm[i:i + VERIFY_CHUNK] for i in range(0, max(1, len(qnorm) - VERIFY_CHUNK + 1), VERIFY_CHUNK)]
    best_page, best = None, 0.0
    for page, txt in page_norms:
        share = sum(c in txt for c in chunks) / len(chunks)
        if share > best:
            best_page, best = page, share
    return best_page if best >= VERIFY_MIN else None


def _strip_margin(text: str) -> str:
    return "\n".join(MARGIN_CODE.sub(" ", ln) for ln in text.splitlines() if not re.fullmatch(r"\s*[RUANEC+]\s*", ln))


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


MARKS_MID = re.compile(r"\s{2,}\[\s*(\d{1,2}(?:\.5)?)\s*\](?=\s{3,})")   # "[06]" with a table printed to its right


def _take_marks(line: str, first: bool = False) -> tuple[str, Optional[float], bool]:
    m = MARKS_END.search(line)
    if m:
        return line[: m.start()].rstrip(), float(m.group(1) or m.group(2)), m.group(0).lstrip().startswith("[")
    m = MARKS_MID.search(line) if first else None
    if m:
        return line[: m.start()] + " " + line[m.end():], float(m.group(1)), True
    return line, None, False


class _Item:
    def __init__(self, qno: str, sub: Optional[str], page: int, alt: bool, co: Optional[str] = None, bloom: Optional[str] = None):
        self.qno, self.sub, self.page, self.alt, self.co, self.bloom = qno, sub, page, alt, co, bloom
        self.lines: list[str] = []
        self.marks: Optional[float] = None
        self._square = False

    def add(self, text: str) -> None:
        text, marks, square = _take_marks(text, first=not self.lines and self.sub is not None)
        if marks is not None:
            if self.marks is not None and square and self._square:
                self.marks += marks                    # "1. ... [1]  2. ... [1]" inside one sub-part
            else:
                self.marks, self._square = marks, square
        if text.strip():
            self.lines.append(text.strip())

    @property
    def text(self) -> str:
        return " ".join(self.lines).strip()


def _body_line(page: int, raw: str, offset: int = 0) -> dict:
    """Strip a CO/Bloom prefix and note where a sub-part label like "(a)" sits, so nested lists can be told apart."""
    co = bloom = None
    m = CO_PREFIX.match(raw)
    rest, start = raw, 0
    if m:
        co, bloom = m.group(1).replace(" ", "").upper(), (m.group(2) or m.group(3))
        rest, start = raw[m.end():], m.end()
    col = offset + start + len(rest) - len(rest.lstrip())
    return {"page": page, "text": rest, "col": col, "co": co, "bloom": bloom}


def split_rules(meta: PaperMeta, pages: list[PageText]) -> list[Question]:
    lines = [(p.page, PAGE_FOOTER.sub(" ", ZERO_WIDTH.sub("", ln)).rstrip()) for p in pages for ln in p.text.splitlines()]
    q_style = any(Q_START.match(ln) for _, ln in lines)
    start_re = Q_START if q_style else NUM_START

    # blocks: one per main question (an "OR" between whole questions starts a new block)
    blocks: list[dict] = []
    for page, raw in lines:
        if NOISE.match(raw):
            continue
        m = start_re.match(raw)
        if m:
            qno = str(int(m.group(1)))
            # "Q.3 ... OR Q.3 ...": a repeated number is the alternative even when the OR line is missing from the text layer
            alt = bool(blocks) and blocks[-1]["qno"] == qno
            blocks.append({"qno": qno, "page": page, "alt": alt, "lines": [_body_line(page, m.group(2), m.start(2))]})
            continue
        if not blocks:
            continue                                   # header and instructions before Q.1
        if OR_LINE.match(raw):
            blocks[-1]["lines"].append({"page": page, "text": "\x00OR"})
            continue
        blocks[-1]["lines"].append(_body_line(page, raw))

    items: list[_Item] = []
    for b in blocks:
        body = [ln for ln in b["lines"] if ln["text"] != "\x00OR"]
        has_letters = any(LETTER.match(ln["text"]) for ln in body)
        sub_re = LETTER if has_letters else ROMAN
        label_cols = [ln["col"] for ln in body if sub_re.match(ln["text"])]
        max_col = min(label_cols) + NESTED_INDENT if label_cols else 0
        stem = _Item(b["qno"], None, b["page"], b["alt"])
        current: Optional[_Item] = None
        next_alt = rest_alt = False
        block_items: list[_Item] = []
        for ln in b["lines"]:
            t = ln["text"]
            if t == "\x00OR":
                next_alt = True
                continue
            m = sub_re.match(t)
            if m and ln["col"] <= max_col:
                label = (m.group(1) or m.group(2)).lower()
                # "(a) (b) OR (a) (b)": labels restart after OR, so the rest of the block replaces the whole question
                if next_alt and len(block_items) > 1 and label == block_items[0].sub:
                    rest_alt = True
                current = _Item(b["qno"], label, ln["page"], b["alt"] or next_alt or rest_alt, ln["co"], ln["bloom"])
                next_alt = False
                current.add(m.group(3))
                block_items.append(current)
                continue
            target = stem if current is None else current
            if target.co is None and ln.get("co"):     # CO printed on a continuation line (column layouts)
                target.co, target.bloom = ln["co"], ln["bloom"]
            target.add(t)
        if not block_items:
            block_items = [stem]
        elif stem.text and not INSTRUCTION_STEM.match(stem.text):
            for it in block_items:                    # shared data/context in the stem belongs to every sub-part
                it.lines.insert(0, stem.text)
        items.extend(block_items)

    return _to_questions(meta, [(i.qno, i.sub, i.marks, i.alt, i.text, i.page, i.co, i.bloom) for i in items if i.text], "rules")


def _to_questions(meta: PaperMeta, rows: list[tuple], method: str) -> list[Question]:
    out, seen = [], set()
    for qno, sub, marks, alt, text, page, *extra in rows:
        co, bloom = (extra + [None, None])[:2]
        base = f"{meta.paper_id}-Q{qno}{sub or ''}" + ("-OR" if alt else "")
        qid, n = base, 2
        while qid in seen:
            qid, n = f"{base}-{n}", n + 1
        seen.add(qid)
        out.append(Question(question_id=qid, paper_id=meta.paper_id, subject=meta.subject, source=meta.source, exam_type=meta.exam_type, year=meta.year, month=meta.month, question_no=qno, sub_part=sub, marks=marks, is_or_alternative=alt, text=text, page=page, split_method=method, co=co, bloom=bloom))
    return out


def question_order(q: Question) -> tuple:
    return (q.paper_id, int(q.question_no) if q.question_no.isdigit() else 0, q.is_or_alternative, q.sub_part or "", q.question_id)


def rules_look_ok(questions: list[Question]) -> bool:
    if len({q.question_no for q in questions}) < 3:
        return False
    substantive = sum(len(q.text) >= 15 for q in questions)
    with_marks = sum(q.marks is not None for q in questions)
    if 0 < with_marks < 0.8 * len(questions):      # paper prints marks but the rules missed many: layout misread
        return False
    return substantive >= 0.6 * len(questions)


def split_llm(meta: PaperMeta, pages: list[PageText], llm: LLM) -> tuple[list[Question], int]:
    template, _ = load_prompt("split")
    paper_text = "\n\n".join(f"[page {p.page}]\n{p.text}" for p in pages)
    result = llm.generate_json(Template(template).safe_substitute(paper_text=paper_text), LLMSplit)
    # compare like with like: margin codes, lone Bloom letters and [marks] are stripped from the paper as from the model's text
    page_norms = [(p.page, _norm(BRACKET_MARKS.sub(" ", _strip_margin(p.text)) + " " + " ".join(p.figures))) for p in pages]
    rows, unverified = [], 0
    for q in result.questions:
        q.text = re.sub(r"\s+", " ", PAGE_FOOTER.sub(" ", MARGIN_CODE.sub(" ", q.text))).strip()
        page = _locate(_norm(q.text), page_norms)
        if page is None:
            unverified += 1
            print(f"  warning: {meta.paper_id} Q{q.question_no}{q.sub_part or ''} text not found verbatim in the paper; check it")
        qno = str(int(re.sub(r"\D", "", q.question_no) or 0))
        sub = (q.sub_part or "").strip("() .").lower() or None
        prev = rows[-1] if rows else None
        if prev and q.marks is None and len(q.text) < FRAGMENT_CHARS and prev[0] == qno and prev[3] == q.is_or_alternative:
            # a criterion or list entry the model split off ("Methodology"): it belongs to the item before it
            rows[-1] = prev[:4] + (f"{prev[4]} {q.text.strip()}",) + prev[5:]
            continue
        rows.append((qno, sub, q.marks, q.is_or_alternative, q.text.strip(), page))
    return _to_questions(meta, rows, "llm"), unverified


def split_paper(meta: PaperMeta, method: str = "auto", llm: Optional[LLM] = None) -> list[Question]:
    pages = load_pages(meta.paper_id)
    if not pages:
        raise FileNotFoundError(f"No extracted pages for {meta.paper_id}; run `extract` first.")
    if method in ("rules", "auto"):
        qs = split_rules(meta, pages)
        if method == "rules" or rules_look_ok(qs):
            return qs
        if llm is None:
            print(f"  {meta.paper_id}: rules split looks wrong ({len(qs)} items) and no Gemini key; keeping it, check by hand")
            return qs
        print(f"  {meta.paper_id}: rules split looks wrong ({len(qs)} items); using Gemini")
    if llm is None:
        raise RuntimeError("method=llm needs GEMINI_API_KEY")
    qs, _ = split_llm(meta, pages, llm)
    return qs


def run(metas: list[PaperMeta], method: str = "auto", llm: Optional[LLM] = None) -> dict[str, int]:
    done_ids = {m.paper_id for m in metas}
    kept = [q for q in read_jsonl(config.QUESTIONS_FILE, Question) if q.paper_id not in done_ids]
    new: list[Question] = []
    done: set[str] = set()
    try:
        for meta in metas:
            qs = split_paper(meta, method, llm)
            with_marks = sum(q.marks is not None for q in qs)
            print(f"  {meta.paper_id}: {len(qs)} questions via {qs[0].split_method if qs else '-'}, {with_marks} with marks")
            new.extend(qs)
            done.add(meta.paper_id)
    finally:
        # save finished papers even if a later one fails (e.g. a Gemini 503); unfinished papers keep their old rows
        old = [q for q in read_jsonl(config.QUESTIONS_FILE, Question) if q.paper_id in done_ids - done]
        write_jsonl(config.QUESTIONS_FILE, sorted(kept + old + new, key=question_order))
    return {"papers": len(metas), "questions": len(new), "total_questions": len(kept) + len(new)}
