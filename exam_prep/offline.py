"""No-LLM practice and grading for the public demo and as a transparent baseline.

Practice: picks a past question from the bank (BM25) and attaches the authored rubric of its topic.
Grading: a keyword-coverage grader. Every rubric point carries keyword stems; the score is the share of the
point's distinct keyword stems found in the answer (stemmed, stop words removed), with partial credit.

Honest limits: it measures vocabulary coverage, not correctness. Keyword stuffing raises the score and a correct
answer in other words can lose marks. tests/test_offline.py and scripts/grader_validation.py quantify this on
synthetic answers of controlled quality; it has NOT been compared with human graders.
"""
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from . import config
from .intelligence import NotEnoughData, RubricPoint, topic_names
from .retrieval import STOPWORDS, Filters, Hit, tokenize

OFFLINE_PRACTICE_NOTICE = ("Practice question chosen from the bundled synthetic sample bank (written for this demo, "
                           "not from any university paper). The rubric is authored per topic.")
OFFLINE_GRADING_NOTICE = ("Keyword-coverage grading: counts how many expected terms your answer uses. It does not check "
                          "correctness and has not been validated against human graders.")
MIN_ANSWER_WORDS = 15
PUNCT = ".,;:!?()[]'\""
MIN_FUNCTION_WORD_SHARE = 0.12   # prose has "the", "of", "is"...; a bare keyword list almost never does


def load_rubrics(path: Optional[Path] = None) -> dict:
    path = path or config.DATA / "rubrics.json"
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _scaled(points: list[dict], total: float) -> list[dict]:
    base = sum(p["marks"] for p in points)
    scaled = [{**p, "marks": round(p["marks"] * total / base, 2)} for p in points]
    scaled[-1]["marks"] = round(total - sum(p["marks"] for p in scaled[:-1]), 2)   # exact total
    return scaled


def generate_practice_offline(topic: str, retriever, subject: str, marks: Optional[float] = None,
                              source: Optional[str] = None, k: int = 5, variant: int = 0) -> dict:
    subject = subject.upper()
    hits: list[Hit] = retriever.search(topic, Filters(subject=subject, source=source), k=k, mode="bm25")
    if not hits:
        raise NotEnoughData(f"No sample {subject} questions match this topic; try other words or another subject.")
    rubrics = load_rubrics()
    candidates = [h for h in hits if h.chunk.topic_ids and h.chunk.topic_ids[0] in rubrics]
    if not candidates:
        raise NotEnoughData("The sample bank has no rubric for the matching topics.")
    pick = candidates[variant % len(candidates)]
    topic_id = pick.chunk.topic_ids[0]
    total = float(marks if marks is not None else (pick.chunk.marks or 8))
    rubric = _scaled(rubrics[topic_id], total)
    names = topic_names([pick.chunk])
    return {"subject": subject, "query": topic, "question": pick.chunk.text, "marks": total, "topic_ids": [topic_id],
            "topic_names": {topic_id: names.get(topic_id, topic_id)},
            "rubric": [{"criterion": r["criterion"], "marks": r["marks"]} for r in rubric],
            "references": [], "attempts": 1, "prompt_version": "offline", "model": "none", "ai_generated": False,
            "notice": OFFLINE_PRACTICE_NOTICE, "source_question": pick.chunk.citation}


def _stems(text: str) -> set[str]:
    return set(tokenize(text))


def keywords_for(criterion: str, rubrics: Optional[dict] = None) -> list[str]:
    for points in (rubrics if rubrics is not None else load_rubrics()).values():
        for p in points:
            if p["criterion"].strip() == criterion.strip():
                return p["keywords"]
    return sorted(_stems(criterion))      # unknown rubric: use the criterion's own content words


@dataclass
class OfflinePoint:
    point: int
    criterion: str
    max_marks: float
    score: float
    feedback: str


def grade_keywords(rubric: list[RubricPoint], answer: str, keyword_lists: Optional[list[list[str]]] = None) -> dict:
    """Score `answer` against `rubric`. keyword_lists overrides the lookup (used by the validation script)."""
    rubrics = load_rubrics()
    raw_words = [w.strip(PUNCT).lower() for w in answer.split()]
    words = len(raw_words)
    prose = words and sum(w in STOPWORDS for w in raw_words) / words >= MIN_FUNCTION_WORD_SHARE
    answer_stems = _stems(answer)
    points = []
    for i, point in enumerate(rubric, 1):
        keywords = keyword_lists[i - 1] if keyword_lists else keywords_for(point.criterion, rubrics)
        wanted = sorted(_stems(" ".join(keywords)))
        need = max(2, math.ceil(0.4 * len(wanted))) if wanted else 1
        found = [w for w in wanted if w in answer_stems]
        share = 0.0 if not wanted else min(1.0, len(found) / need)
        if words < MIN_ANSWER_WORDS or not prose:
            share = min(share, 0.5)    # a short answer or a bare list of terms is not an answer
        score = round(point.marks * share, 2)
        missing = [w for w in wanted if w not in answer_stems][:4]
        feedback = (f"Used {len(found)} of {len(wanted)} expected terms." +
                    (f" Not mentioned: {', '.join(missing)}." if share < 1 and missing else ""))
        points.append(OfflinePoint(i, point.criterion, point.marks, score, feedback))
    total = round(sum(p.score for p in points), 2)
    max_total = round(sum(p.marks for p in rubric), 2)
    overall = ("Short answer or a bare list of terms: write full sentences." if words < MIN_ANSWER_WORDS or not prose else
               "Covers most expected ideas." if total >= 0.75 * max_total else
               "Some expected ideas are missing; compare with the rubric points below.")
    return {"points": [asdict(p) for p in points], "total": total, "max_total": max_total, "overall_feedback": overall,
            "attempts": 1, "prompt_version": "offline", "model": "none", "grader": "keyword",
            "human_validated": False, "notice": OFFLINE_GRADING_NOTICE}
