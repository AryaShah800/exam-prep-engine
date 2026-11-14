"""Phase C tests: metric maths, bootstrap, golden-set loading and the report. No network."""
import json
import pytest
from exam_prep import config, evaluate
from exam_prep.evaluate import GoldenQuery, Setup, bootstrap_diff, query_metrics
from test_retrieval import FakeEmbedder, _write_export


def test_query_metrics_by_hand():
    m = query_metrics(["x", "a", "y", "b", "z", "c"], {"a", "b", "c"})
    assert m["recall@5"] == pytest.approx(2 / 3)        # a, b in top 5; c at rank 6
    assert m["hit@5"] == 1.0 and m["mrr@10"] == pytest.approx(0.5)
    ideal = 1 + 1 / 1.5849625 + 1 / 2.0                   # log2(2), log2(3), log2(4)
    got = 1 / 1.5849625 + 1 / 2.3219281 + 1 / 2.8073549  # ranks 2, 4, 6
    assert m["ndcg@10"] == pytest.approx(got / ideal, rel=1e-6)
    assert query_metrics(["x", "y"], {"a"}) == {"recall@5": 0.0, "hit@5": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0}


def test_bootstrap_ci_contains_mean_and_detects_clear_gain():
    d, lo, hi = bootstrap_diff([1.0] * 30, [0.0] * 30)
    assert d == lo == hi == 1.0
    d, lo, hi = bootstrap_diff([1, 0] * 20, [0, 1] * 20)
    assert d == 0 and lo < 0 < hi


def test_run_and_report(workspace):
    _write_export()
    golden = [GoldenQuery("DC-1", "DC", "topic", "claude", "important questions on deadlock", ["DC_2025_09_MID-Q2b", "BITM-DC_2019_11_END-Q4a"]),
              GoldenQuery("AI-1", "AI", "specific", "arya", "A* search", ["AI_2025_07_MID-Q2a"])]
    setups = [Setup("bm25", "bm25"), Setup("dense", "dense"), Setup("hybrid", "hybrid"), Setup("hybrid (no topic names)", "hybrid", with_topics=False)]
    results = evaluate.run(golden, FakeEmbedder(), setups)
    hybrid = results[2].summary()
    assert hybrid["hit@5"] == 1.0 and hybrid["recall@5"] == 1.0
    assert all(c.startswith("AI_") for c in results[0].ranked["AI-1"])      # subject filter applied
    md = evaluate.report(golden, "golden_v9", results, "fake-embed")
    assert "| hybrid |" in md and "hybrid − bm25" in md and "written and judged by an AI" in md
    assert "| arya | 1 |" in md


def test_bm25_only_run_needs_no_embedder(workspace):
    _write_export()
    golden = [GoldenQuery("DC-1", "DC", "topic", "claude", "deadlock", ["DC_2025_09_MID-Q2b"])]
    assert evaluate.run(golden, None, [Setup("bm25", "bm25")])[0].per_query["DC-1"]["hit@5"] == 1.0
    with pytest.raises(ValueError):
        evaluate.run(golden, None, [Setup("dense", "dense")])


def test_load_golden_picks_highest_version(workspace):
    config.GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    row = {"qid": "X", "subject": "DC", "type": "topic", "author": "arya", "query": "q", "relevant": ["a"]}
    for v in (1, 2, 10):
        (config.GOLDEN_DIR / f"golden_v{v}.jsonl").write_text(json.dumps({**row, "qid": f"X{v}"}) + "\n", encoding="utf-8")
    rows, version = evaluate.load_golden()
    assert version == "golden_v10" and rows[0].qid == "X10"


def test_pool_then_import_adds_human_queries(workspace):
    _write_export()
    config.GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    (config.GOLDEN_DIR / "golden_v1.jsonl").write_text(json.dumps({"qid": "DC-1", "subject": "DC", "type": "topic", "author": "claude",
                                                                   "query": "deadlock", "relevant": ["DC_2025_09_MID-Q2b"]}) + "\n", encoding="utf-8")
    src = config.GOLDEN_DIR / "classmates.csv"
    src.write_text("subject,query,author\nDC,how to write a docker file,classmate1\nDC,quantum entanglement,classmate2\n", encoding="utf-8")
    sheet = evaluate.pool(src, FakeEmbedder())
    import csv as _csv
    rows = list(_csv.DictReader(sheet.open(encoding="utf-8")))
    assert {r["question_id"] for r in rows if r["n"] == "1"} >= {"DC_2025_10_MID-Q3a-OR"}
    with pytest.raises(ValueError):
        evaluate.import_judgements(sheet)                         # nothing judged yet
    for r in rows:
        r["relevant"] = "y" if (r["n"] == "1" and "Dockerfile" in r["question_text"]) else "n"
    with sheet.open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = evaluate.import_judgements(sheet)
    golden, version = evaluate.load_golden()
    assert out.name == "golden_v2.jsonl" and version == "golden_v2" and len(golden) == 2      # query 2 had nothing relevant
    assert golden[1].qid == "H-001" and golden[1].author == "classmate1" and golden[1].relevant == ["DC_2025_10_MID-Q3a-OR"]


def test_answer_eval_with_fake_judge(workspace):
    from exam_prep import answer_eval
    from exam_prep.retrieval import MemoryBackend, Retriever, load_chunks
    from test_retrieval import ScriptedLLM
    _write_export()
    golden = [GoldenQuery("DC-1", "DC", "topic", "claude", "deadlock", ["DC_2025_09_MID-Q2b"]),
              GoldenQuery("DC-2", "DC", "topic", "claude", "dockerfile", ["DC_2025_10_MID-Q3a-OR"]),
              GoldenQuery("AI-1", "AI", "topic", "claude", "A* search", ["AI_2025_07_MID-Q2a"])]
    emb = FakeEmbedder()
    retr = Retriever(MemoryBackend(load_chunks(), emb), emb)
    llm = ScriptedLLM(["- A* is asked for 6 marks [AI_2025_07_MID, Q2a].",         # AI-1: grounded
                       "- Banker's algorithm, 6 marks [DC_2025_09_MID, Q2b].",       # DC-1: grounded
                       "NO_MATCH"])                                                # DC-2
    calls = []
    def judge(q, resp, ctx):
        calls.append(q)
        assert any("AI_2025_07_MID, Q2a" in c or "DC_2025_09_MID, Q2b" in c for c in ctx)
        return 1.0 if "A*" in resp else 0.5
    out = config.DATA / "eval" / "answers.jsonl"
    records = answer_eval.run(golden, retr, llm, judge, per_subject=5, k=3, out_path=out, pause_s=0)
    s = answer_eval.summarise(records)
    assert s["answers"] == 3 and s["grounded_rate"] == pytest.approx(2 / 3) and s["no_match_rate"] == pytest.approx(1 / 3)
    assert s["faithfulness_mean"] == pytest.approx(0.75) and s["faithfulness_n"] == 2 and len(calls) == 2
    again = answer_eval.run(golden, retr, ScriptedLLM([]), judge, per_subject=5, k=3, out_path=out, pause_s=0)   # resumes from disk
    assert len(calls) == 2 and [r.qid for r in again] == [r.qid for r in records]
    assert "RAGAS faithfulness" in answer_eval.report(records, "golden_v1", "gen", "judge", "answer_v1")
