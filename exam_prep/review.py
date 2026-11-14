"""Step 5: the hand check. Sample tags, mark them right or wrong in a CSV, then score the prompt.

  python -m exam_prep review sample --n 50      -> data/review/tag_review_tag_v1.csv
  (open it in Excel / Sheets / LibreOffice, fill the `verdict` column with y, p (partly right) or n,
   and for p / n rows put the right topic IDs in `correct_topic_ids`, separated by ;)
  python -m exam_prep review score              -> accuracy report + your corrections saved as overrides

Change the prompt by adding prompts/tag_v2.txt, re-run `tag`, sample again and compare reports.
"""
import csv, random
from collections import defaultdict
from pathlib import Path
from typing import Optional
from . import config
from .store import read_jsonl, write_jsonl
from .llm import load_prompt
from .schema import Question, TopicTag
from .syllabus import load_taxonomy
from .tag import load_overrides

FIELDS = ["question_id", "subject", "marks", "question_text", "predicted_topic_ids", "predicted_topics", "confidence", "verdict", "correct_topic_ids", "notes"]


def review_path(version: str) -> Path:
    return config.REVIEW_DIR / f"tag_review_{version}.csv"


def sample(n: int = 50, seed: int = 7, version: Optional[str] = None) -> Path:
    version = version or load_prompt("tag")[1]
    # DDU questions only: the accuracy number describes the DDU question bank
    questions = {q.question_id: q for q in read_jsonl(config.QUESTIONS_FILE, Question) if q.source == "DDU"}
    tags = [t for t in read_jsonl(config.TAGS_FILE, TopicTag) if t.prompt_version == version and t.question_id in questions]
    if not tags:
        raise RuntimeError(f"No DDU tags made with {version}; run `tag` first.")
    # Stratify by subject so each pilot subject is checked; every low-confidence tag is included.
    rng = random.Random(seed)
    by_subject: dict[str, list[TopicTag]] = defaultdict(list)
    for t in tags:
        by_subject[questions[t.question_id].subject if t.question_id in questions else "?"].append(t)
    picked = [t for t in tags if t.confidence == "low"][:n]
    remaining = n - len(picked)
    pools = {s: [t for t in ts if t not in picked] for s, ts in by_subject.items()}
    for s in pools:
        rng.shuffle(pools[s])
    while remaining > 0 and any(pools.values()):
        for s in sorted(pools):
            if pools[s] and remaining > 0:
                picked.append(pools[s].pop())
                remaining -= 1
    taxes = {s: load_taxonomy(s).by_id() for s in by_subject if s != "?"}
    path = review_path(version)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for t in sorted(picked, key=lambda t: t.question_id):
            q = questions.get(t.question_id)
            names = taxes.get(q.subject, {}) if q else {}
            w.writerow({"question_id": t.question_id, "subject": q.subject if q else "", "marks": q.marks if q and q.marks is not None else "", "question_text": q.text if q else "", "predicted_topic_ids": ";".join(t.topic_ids), "predicted_topics": "OFF-SYLLABUS (no topic fits)" if t.off_syllabus else " | ".join(names[i].name for i in t.topic_ids if i in names), "confidence": t.confidence, "verdict": "", "correct_topic_ids": "", "notes": ""})
    print(f"  wrote {len(picked)} rows to {path}")
    return path


def score(version: Optional[str] = None, save_overrides: bool = True, reviewer: str = "human") -> dict:
    """reviewer: "human" for a hand check, "claude" when an AI model did the review (recorded on every correction and in the report)."""
    version = version or load_prompt("tag")[1]
    path = review_path(version)
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    judged = [r for r in rows if r["verdict"].strip().lower() in ("y", "p", "n")]
    if not judged:
        raise RuntimeError(f"No verdicts in {path.name}; fill the verdict column with y, p or n.")
    pts = {"y": 1.0, "p": 0.5, "n": 0.0}
    strict = sum(r["verdict"].strip().lower() == "y" for r in judged) / len(judged)
    lenient = sum(pts[r["verdict"].strip().lower()] for r in judged) / len(judged)
    by_conf: dict[str, list[float]] = defaultdict(list)
    for r in judged:
        by_conf[r["confidence"]].append(pts[r["verdict"].strip().lower()] == 1.0)

    overrides = load_overrides()
    valid = {}
    for r in judged:
        ids = [i.strip() for i in r["correct_topic_ids"].replace(",", ";").split(";") if i.strip()]
        if r["verdict"].strip().lower() in ("p", "n") and ids:
            subject = r["subject"]
            if subject not in valid:
                valid[subject] = load_taxonomy(subject).ids()
            bad = [i for i in ids if i not in valid[subject]]
            if bad:
                print(f"  warning: {r['question_id']}: unknown topic IDs {bad}, correction skipped")
                continue
            overrides[r["question_id"]] = TopicTag(question_id=r["question_id"], topic_ids=ids[:2], confidence="high", source="human" if reviewer == "human" else "claude", prompt_version=version)
    if save_overrides:
        write_jsonl(config.OVERRIDES_FILE, sorted(overrides.values(), key=lambda t: t.question_id))

    wrong = [r for r in judged if r["verdict"].strip().lower() == "n"]
    report = {"version": version, "judged": len(judged), "strict_accuracy": round(strict, 3), "lenient_accuracy": round(lenient, 3), "by_confidence": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_conf.items())}, "corrections_saved": len(overrides), "reviewer": reviewer}
    who = "human hand check" if reviewer == "human" else f"{reviewer} (AI second-model review, not a human hand check)"
    lines = [f"# Topic tag review — {version}", "", f"Reviewed by: {who}", "", "| Metric | Value |", "| --- | --- |", f"| Tags checked | {len(judged)} |", f"| Fully right (y) | {strict:.1%} |", f"| Right counting partial as half | {lenient:.1%} |"]
    lines += [f"| Fully right, {k} confidence | {v} |" for k, v in report["by_confidence"].items()]
    lines += ["", "## Wrong tags", "", "| Question | Predicted | Correct | Notes |", "| --- | --- | --- | --- |"]
    lines += [f"| {r['question_id']} | {r['predicted_topic_ids']} | {r['correct_topic_ids']} | {r['notes']} |" for r in wrong] or ["| none | | | |"]
    out = config.REVIEW_DIR / f"report_{version}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {version}: {strict:.1%} fully right, {lenient:.1%} counting partial as half, on {len(judged)} tags -> {out.name}")
    return report
