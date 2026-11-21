"""Build the synthetic demo bank in demo/data/ from scripts/sample_corpus_data.py (deterministic, seed 7).

    python scripts/build_sample_corpus.py
Writes demo/data/processed/questions_tagged.jsonl, demo/data/taxonomy/<SUBJECT>.json, demo/data/rubrics.json.
All questions are authored for this repo; none is taken from a university paper.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_corpus_data import SUBJECTS

OUT = ROOT / "demo" / "data"
EXAMS = [(year, month, kind) for year in range(2021, 2026) for kind, month in (("mid", 9), ("end", 11))]


def main():
    rng = random.Random(7)
    (OUT / "processed").mkdir(parents=True, exist_ok=True)
    (OUT / "taxonomy").mkdir(parents=True, exist_ok=True)
    rows, rubrics = [], {}
    for subject, (subject_name, topics) in SUBJECTS.items():
        taxonomy = {"subject": subject, "subject_name": subject_name, "topics": []}
        pool = []
        for unit, (topic_name, questions, rubric) in enumerate(topics, 1):
            topic_id = f"{subject}-U{unit}-T1"
            taxonomy["topics"].append({"topic_id": topic_id, "unit_no": unit, "unit_title": topic_name, "name": topic_name,
                                       "keywords": sorted({k for _, _, ks in rubric for k in ks})[:8]})
            rubrics[topic_id] = [{"criterion": c, "marks": m, "keywords": ks} for c, m, ks in rubric]
            pool += [(topic_id, topic_name, unit, q) for q in questions]
        rng.shuffle(pool)
        counters = {}
        for index, (topic_id, topic_name, unit, text) in enumerate(pool):
            year, month, kind = EXAMS[index % len(EXAMS)]
            paper_id = f"{subject}_{year}_{month:02d}_{kind.upper()}"
            counters[paper_id] = counters.get(paper_id, 0) + 1
            number = counters[paper_id]
            rows.append({"question_id": f"{paper_id}-Q{number}", "paper_id": paper_id, "subject": subject, "source": "SAMPLE",
                         "exam_type": kind, "year": year, "month": month, "question_no": str(number), "sub_part": None,
                         "marks": rng.choice([4, 5, 6, 8, 10]), "is_or_alternative": False, "text": text,
                         "topic_ids": [topic_id], "topics": [topic_name], "units": [unit], "tag_source": "authored",
                         "tag_confidence": 1.0, "off_syllabus": False})
        (OUT / "taxonomy" / f"{subject}.json").write_text(json.dumps(taxonomy, indent=2), encoding="utf-8")
    with (OUT / "processed" / "questions_tagged.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / "rubrics.json").write_text(json.dumps(rubrics, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} sample questions, {len(rubrics)} topic rubrics to {OUT}")


if __name__ == "__main__":
    main()
