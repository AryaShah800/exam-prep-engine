"""Phase B: grounded answers with validated citations.

The model sees the retrieved questions, each with a tag like [DC_2025_09_MID, Q2b], and must end every claim
with the tag(s) it used. After generation we check:
  1. every cited tag is one of the retrieved questions (no invented citations),
  2. every claim (each bullet, and each sentence inside it) carries at least one tag,
  3. at least one tag is present.
A failing answer is retried with the problems spelled out; after ANSWER_MAX_ATTEMPTS it is refused, and the
caller gets the retrieved questions instead of an unverified answer.
"""
import re
from dataclasses import dataclass, field
from string import Template
from typing import Optional
from pydantic import BaseModel
from . import config
from .llm import LLM, load_prompt
from .retrieval import Filters, Hit, Mode, Retriever

BULLET_BREAK = re.compile(r"(?:\s+|(?<=[.!?\]]))(?=[-*•]\s)")   # bullets the model ran together: "...]. - next" or "...].- next"
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
CITATION = re.compile(r"\[([A-Za-z0-9_\-]+),\s*(Q[^\]\s,]+)\]")
NO_MATCH = "NO_MATCH"
NO_DDU_NOTE = "No DDU paper in the bank matches this; all sources below are other universities' papers (extra practice)."


class LLMAnswer(BaseModel):
    answer: str


@dataclass
class Answer:
    query: str
    text: str
    grounded: bool                         # True only if every check passed
    citations: list[str] = field(default_factory=list)   # cited chunk_ids, in order of first use
    hits: list[Hit] = field(default_factory=list)
    attempts: int = 0
    problems: list[str] = field(default_factory=list)   # what failed on the last attempt
    prompt_version: str = ""


def format_sources(hits: list[Hit]) -> str:
    lines = []
    for h in hits:
        c = h.chunk
        marks = f"{c.marks:g} marks" if c.marks is not None else "marks not printed"
        lines.append(f"{c.citation} {c.source}, {c.year}, {c.exam_type} exam, {marks}: {' '.join(c.text.split())}")
    return "\n".join(lines)


def tidy(text: str) -> str:
    """One bullet per line."""
    return "\n".join(b.strip() for ln in text.splitlines() for b in BULLET_BREAK.split(ln) if b.strip())


def claims(text: str) -> list[str]:
    """Bullets and sentences that state something (headings ending in ':' and very short fragments skipped)."""
    out = []
    for bullet in tidy(text).splitlines():
        for sent in SENTENCE_END.split(bullet.lstrip("-*• ").strip()):
            if len(sent) >= 25 and not sent.endswith(":"):
                out.append(sent)
    return out


def check_citations(text: str, hits: list[Hit]) -> tuple[list[str], list[str]]:
    """Returns (cited chunk_ids, problems). An empty problems list means the answer passes."""
    allowed = {(h.chunk.paper_id, h.chunk.label): h.chunk.chunk_id for h in hits}
    cited, problems = [], []
    for pid, label in CITATION.findall(text):
        cid = allowed.get((pid, label))
        if cid is None:
            problems.append(f"[{pid}, {label}] is not one of the sources")
        elif cid not in cited:
            cited.append(cid)
    for claim in claims(text):
        if not CITATION.search(claim):
            problems.append(f"no citation on: {claim[:80]}")
    if not cited and not problems:
        problems.append("the answer cites no source")
    return cited, problems


def answer(query: str, retriever: Retriever, llm: LLM, filters: Optional[Filters] = None, k: int = 8,
           mode: Mode = "hybrid", max_attempts: int = config.ANSWER_MAX_ATTEMPTS) -> Answer:
    hits = retriever.search(query, filters, k=k, mode=mode)
    template, version = load_prompt("answer")
    result = Answer(query=query, text="", grounded=False, hits=hits, prompt_version=version)
    if not hits:
        result.text = "No past questions match these filters."
        return result
    feedback = ""
    for attempt in range(1, max_attempts + 1):
        prompt = Template(template).safe_substitute(query=query, sources=format_sources(hits), feedback=feedback)
        text = tidy(llm.generate_json(prompt, LLMAnswer).answer.strip())
        result.attempts = attempt
        if text == NO_MATCH:
            result.text, result.problems = "None of the retrieved past questions match this request.", []
            return result
        cited, problems = check_citations(text, hits)
        if not problems:
            if not any(h.chunk.source == "DDU" for h in hits):   # a fact about the data, so code says it, not the model
                text = f"{NO_DDU_NOTE}\n{text}"
            result.text, result.citations, result.grounded, result.problems = text, cited, True, []
            return result
        result.problems = problems
        feedback = ("\nYour previous reply failed the citation check. Fix these problems:\n- " + "\n- ".join(problems[:8])
                    + f"\nAllowed tags: {', '.join(h.chunk.citation for h in hits)}")
    result.text = "Could not write an answer whose every claim is backed by a retrieved question. The most relevant past questions are listed below."
    return result
