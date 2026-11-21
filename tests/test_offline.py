import json
import pytest
from exam_prep import config, offline
from exam_prep.intelligence import RubricPoint
from exam_prep.retrieval import Filters, MemoryBackend, Retriever, chunk_from_row

RUBRIC = [RubricPoint(criterion="Defines mutual exclusion and circular wait", marks=3),
          RubricPoint(criterion="Explains avoidance with safe state", marks=3)]
KEYWORDS = [["mutual", "exclusion", "hold", "wait", "circular", "preemption"], ["safe", "banker", "available", "need"]]
GOOD = ("A deadlock needs mutual exclusion, hold and wait, no preemption and circular wait. Avoidance keeps the system in a "
        "safe state, as the Banker algorithm checks available resources against the maximum need of each process.")


def test_full_answer_scores_full_marks():
    g = offline.grade_keywords(RUBRIC, GOOD, KEYWORDS)
    assert g["total"] == g["max_total"] == 6 and g["grader"] == "keyword" and g["human_validated"] is False


def test_irrelevant_answer_scores_zero_and_feedback_names_missing_terms():
    g = offline.grade_keywords(RUBRIC, "I like to study operating systems in the library every evening after lunch.", KEYWORDS)
    assert g["total"] == 0 and "Not mentioned" in g["points"][0]["feedback"]


def test_bare_keyword_list_is_capped_at_half_marks():
    g = offline.grade_keywords(RUBRIC, "mutual exclusion hold wait circular safe banker available need", KEYWORDS)
    assert g["total"] <= 3.0
    assert "full sentences" in g["overall_feedback"]


def test_long_keyword_list_without_prose_is_capped_too():
    g = offline.grade_keywords(RUBRIC, " ".join(KEYWORDS[0] + KEYWORDS[1]) * 3, KEYWORDS)
    assert g["total"] <= 3.0


def test_partial_answer_is_between_zero_and_full():
    g = offline.grade_keywords(RUBRIC, "A deadlock occurs when there is mutual exclusion and a circular wait between several processes.", KEYWORDS)
    assert 0 < g["total"] < 6


def test_unknown_criterion_falls_back_to_criterion_words():
    r = [RubricPoint(criterion="Explains photosynthesis and chlorophyll", marks=2)]
    g = offline.grade_keywords(r, "Photosynthesis uses chlorophyll to turn light into chemical energy in the leaf cells of the plant during the day.")
    assert g["total"] == 2


def test_practice_offline_uses_bank_question_and_scaled_rubric(tmp_path, monkeypatch):
    rubrics = {"OS-U2-T1": [{"criterion": "a", "marks": 3, "keywords": ["x"]}, {"criterion": "b", "marks": 5, "keywords": ["y"]}]}
    (tmp_path / "rubrics.json").write_text(json.dumps(rubrics), encoding="utf-8")
    monkeypatch.setattr(config, "DATA", tmp_path)
    row = {"question_id": "OS_2021_09_MID-Q1", "paper_id": "OS_2021_09_MID", "subject": "OS", "source": "SAMPLE", "year": 2021,
           "month": 9, "exam_type": "mid", "marks": 10, "topic_ids": ["OS-U2-T1"], "topics": ["Deadlocks"],
           "text": "State the four necessary conditions for a deadlock."}
    retriever = Retriever(MemoryBackend([chunk_from_row(row)], None), None)
    p = offline.generate_practice_offline("deadlock", retriever, "os", marks=7)
    assert p["question"].startswith("State the four") and p["ai_generated"] is False
    assert sum(r["marks"] for r in p["rubric"]) == pytest.approx(7)
