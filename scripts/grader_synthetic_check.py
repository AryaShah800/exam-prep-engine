"""Sanity and adversarial check of the keyword grader on SYNTHETIC answers of controlled quality.

    python scripts/grader_synthetic_check.py        # writes reports/grader_synthetic_check.json

For each of the 24 sample topics it builds (deterministically, no LLM) four answers:
  full      one full sentence per rubric point, using that point's keywords
  partial   a sentence for the first rubric point only
  offtopic  the full answer of a different topic (same subject, next topic)
  stuffed   every keyword of the rubric as a bare list, no sentences (keyword-stuffing attack)
This is circular by design for "full" (answers are built from the grader's own keywords), so it checks that the
implementation orders tiers sensibly and exposes its weaknesses. It does NOT measure agreement with human graders.
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exam_prep.intelligence import RubricPoint
from exam_prep.offline import grade_keywords

RUBRICS = json.loads((ROOT / "demo/data/rubrics.json").read_text(encoding="utf-8"))
FILLER = "In an examination answer a student would explain this idea in the following way, using the standard terms:"


def sentence(keywords):
    return f"{FILLER} {', '.join(keywords)}. These terms are related to each other in the explanation."


def grade(topic_id, answer):
    points = RUBRICS[topic_id]
    rubric = [RubricPoint(criterion=p["criterion"], marks=p["marks"]) for p in points]
    result = grade_keywords(rubric, answer, [p["keywords"] for p in points])
    return result["total"] / result["max_total"]


def main():
    ids = list(RUBRICS)
    tiers = {"full": [], "partial": [], "offtopic": [], "stuffed": []}
    for i, topic_id in enumerate(ids):
        points = RUBRICS[topic_id]
        full = " ".join(sentence(p["keywords"]) for p in points)
        subject = topic_id.split("-")[0]
        mates = [t for t in ids if t.startswith(subject + "-") and t != topic_id]
        other = mates[i % len(mates)]
        tiers["full"].append(grade(topic_id, full))
        tiers["partial"].append(grade(topic_id, sentence(points[0]["keywords"])))
        tiers["offtopic"].append(grade(topic_id, " ".join(sentence(p["keywords"]) for p in RUBRICS[other])))
        tiers["stuffed"].append(grade(topic_id, " ".join(k for p in points for k in p["keywords"])))
    summary = {name: {"mean_share_of_marks": round(statistics.mean(v), 3), "min": round(min(v), 3), "max": round(max(v), 3)}
               for name, v in tiers.items()}
    ordered = sum(f > p > o for f, p, o in zip(tiers["full"], tiers["partial"], tiers["offtopic"]))
    report = {"topics": len(ids), "tiers": summary, "topics_where_full_gt_partial_gt_offtopic": ordered,
              "caveat": "Synthetic answers built from the grader's own keyword lists; checks ordering and keyword-stuffing "
                        "behaviour only. Not a validation against human graders."}
    out = ROOT / "reports" / "grader_synthetic_check.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
