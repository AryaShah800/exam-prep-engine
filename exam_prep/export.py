"""Step 6: merge questions + tags (+ your overrides) into questions_tagged.jsonl, the input to Phase B,
and print the dataset numbers you will later quote in the README."""
import json
from collections import Counter
from . import config
from .store import read_jsonl
from .schema import PageText, Question
from .syllabus import load_taxonomy
from .tag import effective_tags


def export() -> int:
    questions = read_jsonl(config.QUESTIONS_FILE, Question)
    tags = effective_tags()
    names: dict[str, dict] = {}
    config.EXPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with config.EXPORT_FILE.open("w", encoding="utf-8") as f:
        for q in questions:
            if q.subject not in names:
                try:
                    names[q.subject] = load_taxonomy(q.subject).by_id()
                except FileNotFoundError:
                    names[q.subject] = {}
            t = tags.get(q.question_id)
            row = q.model_dump()
            row["topic_ids"] = t.topic_ids if t else []
            row["topics"] = [names[q.subject][i].name for i in row["topic_ids"] if i in names[q.subject]]
            row["units"] = sorted({names[q.subject][i].unit_no for i in row["topic_ids"] if i in names[q.subject]})
            row["tag_source"] = t.source if t else None
            row["tag_confidence"] = t.confidence if t else None
            row["off_syllabus"] = bool(t and t.off_syllabus)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"  wrote {n} questions to {config.EXPORT_FILE}")
    return n


def stats() -> dict:
    """Dataset numbers for the README. Headline counts are DDU only (the resume says "DDU past papers");
    other universities' papers are reported separately."""
    all_questions = read_jsonl(config.QUESTIONS_FILE, Question)
    questions = [q for q in all_questions if q.source == "DDU"]
    external = [q for q in all_questions if q.source != "DDU"]
    tags = effective_tags()
    ddu_papers = {q.paper_id for q in questions}
    pages = [p for f in sorted(config.PAGES_DIR.glob("*.jsonl")) for p in read_jsonl(f, PageText) if p.paper_id in ddu_papers]
    paper_subject = {q.paper_id: q.subject for q in questions}
    s = {
        "ddu_papers": len(ddu_papers),
        "ddu_pages": len(pages),
        "pages_via_gemini_vision": sum(p.method == "ocr" for p in pages),
        "ddu_questions": len(questions),
        "questions_with_marks": sum(q.marks is not None for q in questions),
        "questions_tagged": sum(1 for q in questions if tags.get(q.question_id) and tags[q.question_id].topic_ids),
        "tags_needing_review": sum(1 for q in questions if tags.get(q.question_id) and tags[q.question_id].needs_review),
        "hand_corrected_tags": sum(1 for q in questions if tags.get(q.question_id) and tags[q.question_id].source == "human"),
        "ai_review_corrected_tags": sum(1 for q in questions if tags.get(q.question_id) and tags[q.question_id].source == "claude"),
        "by_subject": dict(Counter(q.subject for q in questions)),
        "papers_by_subject": dict(Counter(paper_subject.values())),
        "years": sorted({q.year for q in questions}),
        "external_papers": dict(Counter(src for _, src in {(q.paper_id, q.source) for q in external})),
        "external_questions": len(external),
        "external_off_syllabus": sum(1 for q in external if tags.get(q.question_id) and tags[q.question_id].off_syllabus),
    }
    for k, v in s.items():
        print(f"  {k:<26} {v}")
    return s
