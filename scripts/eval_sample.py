"""Retrieval evaluation on the synthetic sample bank with a hand-written PARAPHRASE query set.

    python scripts/eval_sample.py [--no-dense]       # writes reports/retrieval_eval_sample_paraphrase.md
Queries (two per topic, 48 total) were written by hand to avoid the wording of the bank's questions; relevance =
every question tagged with the topic. Reports BM25 / dense (local bge-base-en-v1.5) / hybrid with bootstrap CIs and the
mean share of query words that also occur in the relevant questions (lexical overlap). Synthetic bank, so absolute
numbers say nothing about real exam papers.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exam_prep import config
config.set_root(ROOT / "demo")
from exam_prep import evaluate
from exam_prep.retrieval import load_chunks, tokenize

QUERIES = {
    "OS-U1-T1": ["which job runs next on the processor and how to rank algorithms", "ordering of tasks for the cpu so nobody waits forever"],
    "OS-U2-T1": ["programs stuck forever each holding something the other needs", "how a system avoids ending up in a standstill over resources"],
    "OS-U3-T1": ["mapping program addresses to physical ram using fixed-size blocks", "wasted space inside and between allocated regions of main storage"],
    "OS-U4-T1": ["what happens when a needed block of a program is not in ram", "swapping out blocks and the system spending its time swapping"],
    "OS-U5-T1": ["making sure two programs do not touch shared data at the same moment", "coordinating cooperating programs with counters and signals"],
    "OS-U6-T1": ["how files are laid out on storage and found again", "organizing folders and tracking which storage blocks are unused"],
    "OS-U7-T1": ["ordering read requests so the disk arm moves less", "mechanical delays of spinning storage and redundant arrays"],
    "OS-U8-T1": ["lightweight units of execution inside a program versus whole programs", "ways running programs exchange data with one another"],
    "DB-U1-T1": ["splitting tables to remove duplicated information", "why storing the same fact in many rows causes update trouble"],
    "DB-U2-T1": ["all-or-nothing guarantees when money moves between accounts", "correct interleaving of concurrent operations on stored data"],
    "DB-U3-T1": ["stopping two users from changing the same record at once", "validating or timestamping operations instead of locking"],
    "DB-U4-T1": ["balanced tree structures that speed up lookups on disk", "auxiliary structures so searching a table avoids a full scan"],
    "DB-U5-T1": ["combining rows from two tables on matching columns", "counting and averaging per category with a declarative language"],
    "DB-U6-T1": ["diagramming real-world things and how they relate", "turning a conceptual picture of data into actual tables"],
    "DB-U7-T1": ["how the engine decides the cheapest way to run a request", "rewriting a request so filters happen early"],
    "DB-U8-T1": ["restoring a consistent state after the machine crashes", "keeping a journal of changes so they can be redone or reversed"],
    "ML-U1-T1": ["predicting a number from features with a straight line fit", "squashing a score into a probability for yes or no outcomes"],
    "ML-U2-T1": ["model too simple versus model memorizing the training set", "holding out data to pick a model fairly"],
    "ML-U3-T1": ["splitting data by asking yes/no questions about features", "combining many weak learners into a strong predictor"],
    "ML-U4-T1": ["finding the widest street between two classes of points", "mapping data to a higher dimensional space without computing it"],
    "ML-U5-T1": ["grouping unlabeled points around centers", "discovering groups of arbitrary shape from dense regions"],
    "ML-U6-T1": ["how layers of neurons learn by passing errors backward", "why deep stacks stop learning and which nonlinearity helps"],
    "ML-U7-T1": ["sliding small filters over pictures to detect patterns", "reusing a network trained on millions of photos for a new task"],
    "ML-U8-T1": ["of the items flagged, how many were right, and how many were found", "measuring a classifier fairly when one class is rare"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-dense", action="store_true")
    args = parser.parse_args()
    chunks = load_chunks()
    by_topic = {}
    for c in chunks:
        by_topic.setdefault(c.topic_ids[0], []).append(c)
    golden_rows, overlaps = [], []
    for n, (topic_id, qs) in enumerate(QUERIES.items(), 1):
        for j, q in enumerate(qs, 1):
            rel = by_topic[topic_id]
            qt = set(tokenize(q, query=True))
            rt = set().union(*(set(tokenize(c.text)) for c in rel))
            overlaps.append(len(qt & rt) / max(1, len(qt)))
            golden_rows.append({"qid": f"{topic_id}-{j}", "subject": topic_id.split("-")[0], "type": "paraphrase", "author": "human-synthetic",
                                "query": q, "relevant": [c.chunk_id for c in rel]})
    (ROOT / "demo/data/golden/golden_v1.jsonl").write_text("\n".join(json.dumps(r) for r in golden_rows) + "\n", encoding="utf-8")
    golden, version = evaluate.load_golden()
    embedder = None
    model = "none (bm25 only)"
    setups = [s for s in evaluate.DEFAULT_SETUPS if s.mode == "bm25" and s.with_topics] if args.no_dense else None
    if not args.no_dense:
        from exam_prep.embed import make_embedder
        embedder = make_embedder("local")
        model = embedder.model
    results = evaluate.run(golden, embedder, setups)
    path = evaluate.save(golden, "sample_paraphrase", results, model)
    print(path, "mean query-word overlap with relevant questions:", round(sum(overlaps) / len(overlaps), 3))
    for r in results:
        pm = r.per_query.values()
        print(r.setup.name, {k: round(sum(m[k] for m in pm) / len(pm), 3) for k in ("recall@5", "hit@5", "mrr@10", "ndcg@10")})


if __name__ == "__main__":
    main()
