"""Step 4: question -> 1 or 2 syllabus topics, with Gemini structured output.

Questions are sent in batches (TAG_BATCH_SIZE, default 8) to save free-tier requests. Every returned
topic ID is checked against the taxonomy; a question with no valid ID is retried alone, then marked
needs_review. Hand corrections in tag_overrides.jsonl always win over the model (see review.py).
"""
from string import Template
from typing import Callable, Iterable, Optional
from . import config
from .store import read_jsonl, write_jsonl
from .llm import LLM, load_prompt
from .schema import LLMTagBatch, Question, Taxonomy, TopicTag
from .syllabus import load_taxonomy


def topic_table(tax: Taxonomy) -> str:
    return "\n".join(f"{t.topic_id} | U{t.unit_no} {t.unit_title} | {t.name} | {', '.join(t.keywords)}" for t in tax.topics)


def _format_questions(qs: Iterable[Question]) -> str:
    return "\n\n".join(f"[{q.question_id}] ({q.marks:g} marks) {q.text}" if q.marks else f"[{q.question_id}] {q.text}" for q in qs)


def _batches(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def tag_batch(qs: list[Question], tax: Taxonomy, llm: LLM, template: str, version: str) -> list[TopicTag]:
    prompt = Template(template).safe_substitute(subject=tax.subject, subject_name=tax.subject_name or tax.subject, topic_table=topic_table(tax), questions=_format_questions(qs))
    result = llm.generate_json(prompt, LLMTagBatch)
    valid, by_id = tax.ids(), {q.question_id: q for q in qs}
    tags: dict[str, TopicTag] = {}
    for t in result.tags:
        if t.question_id not in by_id or t.question_id in tags:
            continue
        ids = list(dict.fromkeys(i.strip() for i in t.topic_ids if i.strip() in valid))[:2]
        if ids:
            tags[t.question_id] = TopicTag(question_id=t.question_id, topic_ids=ids, confidence=t.confidence, needs_review=t.confidence == "low", model=llm.model, prompt_version=version)
        elif t.off_syllabus and not t.topic_ids:
            # a DDU question outside the DDU syllabus means the taxonomy is missing something: review it
            is_ddu = by_id[t.question_id].source == "DDU"
            tags[t.question_id] = TopicTag(question_id=t.question_id, topic_ids=[], confidence=t.confidence, needs_review=is_ddu, off_syllabus=True, model=llm.model, prompt_version=version)
    return [tags[q.question_id] for q in qs if q.question_id in tags]


def tag_questions(qs: list[Question], tax: Taxonomy, llm: LLM, batch_size: int = config.TAG_BATCH_SIZE, on_batch: Optional[Callable[[list[TopicTag]], None]] = None) -> list[TopicTag]:
    template, version = load_prompt("tag")
    out: list[TopicTag] = []
    for batch in _batches(qs, batch_size):
        got = tag_batch(batch, tax, llm, template, version)
        done = {t.question_id for t in got}
        for q in batch:
            if q.question_id in done:
                continue
            retry = tag_batch([q], tax, llm, template, version) if len(batch) > 1 else []
            got.extend(retry or [TopicTag(question_id=q.question_id, topic_ids=[], confidence="low", needs_review=True, model=llm.model, prompt_version=version)])
        out.extend(got)
        if on_batch:
            on_batch(out)                              # save as we go: a killed or failed run keeps finished batches
        print(f"  tagged {len(out)}/{len(qs)}")
    return out


def load_overrides() -> dict[str, TopicTag]:
    return {t.question_id: t for t in read_jsonl(config.OVERRIDES_FILE, TopicTag)}


def effective_tags() -> dict[str, TopicTag]:
    """Model tags with your hand corrections applied on top."""
    tags = {t.question_id: t for t in read_jsonl(config.TAGS_FILE, TopicTag)}
    tags.update(load_overrides())
    return tags


def run(subject: str, llm: LLM, retag: bool = False, limit: Optional[int] = None) -> dict[str, int]:
    tax = load_taxonomy(subject)
    _, version = load_prompt("tag")
    questions = [q for q in read_jsonl(config.QUESTIONS_FILE, Question) if q.subject == subject.upper()]
    existing = {t.question_id: t for t in read_jsonl(config.TAGS_FILE, TopicTag)}
    todo = [q for q in questions if retag or q.question_id not in existing or existing[q.question_id].prompt_version != version]
    if limit:
        todo = todo[:limit]
    def save(done: list[TopicTag]) -> None:
        merged = {**existing, **{t.question_id: t for t in done}}
        write_jsonl(config.TAGS_FILE, sorted(merged.values(), key=lambda t: t.question_id))

    new = tag_questions(todo, tax, llm, on_batch=save) if todo else []
    save(new)
    flagged = sum(t.needs_review for t in new)
    print(f"  {subject.upper()}: tagged {len(new)} questions with {version}; {flagged} flagged for review")
    return {"tagged": len(new), "needs_review": flagged, "questions": len(questions)}
