"""Phase C: retrieval evaluation on a golden query set.

Golden set: data/golden/golden_v<N>.jsonl, one row per query:
    {"qid": "DC-01", "subject": "DC", "type": "topic|specific|paraphrase", "author": "claude|arya|<classmate>",
     "query": "important questions on deadlock", "relevant": ["AKTU-DC_2023_01_END-Q1c", ...]}
Relevance is judged from the question TEXT, never from the topic tags (the index contains tag names, so labels
built from tags would be circular). Every query is searched with its subject filter, as the app does.

Metrics per query, averaged (macro) over queries:
    Recall@5   share of the query's relevant questions found in the top 5
    Hit@5      1 if at least one relevant question is in the top 5
    MRR@10     1 / rank of the first relevant question (0 if none in the top 10)
    nDCG@10    binary-gain nDCG
Differences between setups get a paired bootstrap 95% confidence interval (2,000 resamples of queries, seed 0).
"""
import csv, json, math, random, re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from . import config
from .embed import Embedder
from .retrieval import Filters, MemoryBackend, Retriever, load_chunks

K_RECALL, K_RANK = 5, 10
REPORTS_DIR = config.ROOT / "reports"


@dataclass
class GoldenQuery:
    qid: str
    subject: str
    type: str
    author: str
    query: str
    relevant: list[str]


def load_golden(path: Optional[Path] = None) -> tuple[list[GoldenQuery], str]:
    if path is None:
        files = sorted(config.GOLDEN_DIR.glob("golden_v*.jsonl"), key=lambda p: int(p.stem.rsplit("_v", 1)[1]))
        if not files:
            raise FileNotFoundError(f"No golden set in {config.GOLDEN_DIR}")
        path = files[-1]
    with path.open(encoding="utf-8") as f:
        rows = [GoldenQuery(**json.loads(line)) for line in f if line.strip()]
    return rows, path.stem


def query_metrics(ranked: list[str], relevant: set[str]) -> dict[str, float]:
    top5 = ranked[:K_RECALL]
    found = sum(1 for c in top5 if c in relevant)
    rr = next((1.0 / i for i, c in enumerate(ranked[:K_RANK], 1) if c in relevant), 0.0)
    dcg = sum(1.0 / math.log2(i + 1) for i, c in enumerate(ranked[:K_RANK], 1) if c in relevant)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), K_RANK) + 1))
    return {"recall@5": found / len(relevant), "hit@5": float(found > 0), "mrr@10": rr, "ndcg@10": dcg / idcg if idcg else 0.0}


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_diff(a: list[float], b: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) over queries and its paired bootstrap 95% CI."""
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    stats = sorted(mean([d[rng.randrange(len(d))] for _ in d]) for _ in range(n))
    return mean(d), stats[int(0.025 * n)], stats[int(0.975 * n) - 1]


@dataclass
class Setup:
    name: str
    mode: str
    with_topics: bool = True
    external_weight: float = config.EXTERNAL_WEIGHT


DEFAULT_SETUPS = [
    Setup("bm25", "bm25"), Setup("dense", "dense"), Setup("hybrid", "hybrid"),
    Setup("bm25 (no topic names)", "bm25", with_topics=False), Setup("dense (no topic names)", "dense", with_topics=False),
    Setup("hybrid (no topic names)", "hybrid", with_topics=False),
    Setup("hybrid, external weight 1.0", "hybrid", external_weight=1.0),
    Setup("hybrid, external weight 0.9", "hybrid", external_weight=0.9),
]


@dataclass
class RunResult:
    setup: Setup
    per_query: dict[str, dict[str, float]] = field(default_factory=dict)
    ranked: dict[str, list[str]] = field(default_factory=dict)

    def summary(self, qids: Optional[list[str]] = None) -> dict[str, float]:
        rows = [self.per_query[q] for q in (qids or list(self.per_query))]
        return {m: mean([r[m] for r in rows]) for m in ("recall@5", "hit@5", "mrr@10", "ndcg@10")}


def run(golden: list[GoldenQuery], embedder: Optional[Embedder], setups: Optional[list[Setup]] = None) -> list[RunResult]:
    setups = setups or DEFAULT_SETUPS
    if embedder is not None:
        embedder.embed([g.query for g in golden], "RETRIEVAL_QUERY")   # one batched call, then every search hits the cache
    elif any(s.mode != "bm25" for s in setups):
        raise ValueError("dense and hybrid setups need an embedder")
    backends: dict[bool, MemoryBackend] = {}
    results = []
    for s in setups:
        if s.with_topics not in backends:
            backends[s.with_topics] = MemoryBackend(load_chunks(with_topics=s.with_topics), embedder)
        retr = Retriever(backends[s.with_topics], embedder, external_weight=s.external_weight)
        res = RunResult(s)
        for g in golden:
            hits = retr.search(g.query, Filters(subject=g.subject), k=K_RANK, mode=s.mode)
            res.ranked[g.qid] = [h.chunk.chunk_id for h in hits]
            res.per_query[g.qid] = query_metrics(res.ranked[g.qid], set(g.relevant))
        results.append(res)
    return results


def _table(rows: list[tuple[str, dict[str, float]]]) -> list[str]:
    out = ["| Setup | Recall@5 | Hit@5 | MRR@10 | nDCG@10 |", "| --- | --- | --- | --- | --- |"]
    for name, s in rows:
        out.append(f"| {name} | {s['recall@5']:.1%} | {s['hit@5']:.1%} | {s['mrr@10']:.3f} | {s['ndcg@10']:.3f} |")
    return out


def report(golden: list[GoldenQuery], version: str, results: list[RunResult], embed_model: str) -> str:
    by = {r.setup.name: r for r in results}
    authors = sorted({g.author for g in golden})
    lines = [f"# Retrieval evaluation — {version}", "",
             f"- Run: {datetime.now():%Y-%m-%d %H:%M}, backend `memory` (exact cosine + Okapi BM25), embeddings `{embed_model}`, RRF k={config.RRF_K}, subject filter on every query.",
             f"- Golden set: {len(golden)} queries over {len({g.subject for g in golden})} subjects, {sum(len(g.relevant) for g in golden)} relevant (query, question) pairs; written by: {', '.join(authors)}.",
             "- Relevance was judged from question text against the full question bank of each subject (not from topic tags).",
             "", "## All queries", ""]
    lines += _table([(r.setup.name, r.summary()) for r in results])
    if {"hybrid", "bm25", "dense"} <= by.keys():
        lines += ["", "## Is hybrid better? (paired bootstrap, 95% CI)", "", "| Comparison | Metric | Mean difference | 95% CI |", "| --- | --- | --- | --- |"]
        qids = [g.qid for g in golden]
        for other in ("bm25", "dense"):
            for m in ("recall@5", "mrr@10"):
                d, lo, hi = bootstrap_diff([by["hybrid"].per_query[q][m] for q in qids], [by[other].per_query[q][m] for q in qids])
                verdict = "significant" if lo > 0 or hi < 0 else "not significant"
                lines.append(f"| hybrid − {other} | {m} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] ({verdict}) |")
    main = [r for r in results if r.setup.name in ("bm25", "dense", "hybrid")]
    for label, key in (("query type", "type"), ("subject", "subject"), ("author", "author")):
        lines += ["", f"## By {label}", "", "| " + label.capitalize() + " | n | " + " | ".join(f"{r.setup.name} R@5 / MRR" for r in main) + " |",
                  "| --- | --- | " + " | ".join("---" for _ in main) + " |"]
        for val in sorted({getattr(g, key) for g in golden}):
            qids = [g.qid for g in golden if getattr(g, key) == val]
            cells = [f"{r.summary(qids)['recall@5']:.1%} / {r.summary(qids)['mrr@10']:.3f}" for r in main]
            lines.append(f"| {val} | {len(qids)} | " + " | ".join(cells) + " |")
    if "hybrid" in by:
        worst = sorted(golden, key=lambda g: (by["hybrid"].per_query[g.qid]["recall@5"], by["hybrid"].per_query[g.qid]["mrr@10"]))[:8]
        lines += ["", "## Hardest queries for hybrid", "", "| Query | Subject | Relevant | Recall@5 | MRR@10 |", "| --- | --- | --- | --- | --- |"]
        for g in worst:
            m = by["hybrid"].per_query[g.qid]
            lines.append(f"| {g.query} | {g.subject} | {len(g.relevant)} | {m['recall@5']:.0%} | {m['mrr@10']:.2f} |")
    lines += ["", "## Caveats", "",
              "- Recall@5 is capped below 100% for queries with more than 5 relevant questions; Hit@5 and MRR are not.",
              "- Queries written after reading the question bank tend to reuse its wording, which favours keyword search. The `paraphrase` rows avoid that wording on purpose.",
              "- One semester of DDU papers (2025) plus other universities' papers; numbers will move as papers are added.",
              "- The external-weight rows were not used to pick the default weight from this same set (that would overfit); they show sensitivity only."]
    if "claude" in authors:
        lines.append("- Queries marked `claude` were written and judged by an AI (Claude), not by students. Say so wherever these numbers are quoted, until a human-written set is added.")
    return "\n".join(lines) + "\n"


def save(golden: list[GoldenQuery], version: str, results: list[RunResult], embed_model: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^a-z0-9]+", "-", embed_model.lower().split("/")[-1]).strip("-")
    md = REPORTS_DIR / f"retrieval_eval_{version}_{tag}.md"
    md.write_text(report(golden, version, results, embed_model), encoding="utf-8")
    with (REPORTS_DIR / f"retrieval_eval_{version}_{tag}_per_query.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "subject", "type", "author", "query", "n_relevant", "setup", "recall@5", "hit@5", "mrr@10", "ndcg@10", "top5"])
        for r in results:
            for g in golden:
                m = r.per_query[g.qid]
                w.writerow([g.qid, g.subject, g.type, g.author, g.query, len(g.relevant), r.setup.name,
                            f"{m['recall@5']:.4f}", f"{m['hit@5']:.0f}", f"{m['mrr@10']:.4f}", f"{m['ndcg@10']:.4f}", ";".join(r.ranked[g.qid][:5])])
    _log_mlflow(version, results, embed_model, md)
    return md


def _log_mlflow(version: str, results: list[RunResult], embed_model: str, report_path: Path) -> None:
    try:
        import mlflow
    except ImportError:
        print("  (mlflow not installed: skipped experiment logging)")
        return
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment("exam-prep-retrieval")
    for r in results:
        with mlflow.start_run(run_name=f"{version} | {r.setup.name}"):
            mlflow.log_params({"golden_set": version, "mode": r.setup.mode, "with_topics": r.setup.with_topics,
                               "external_weight": r.setup.external_weight, "embed_model": embed_model,
                               "rrf_k": config.RRF_K, "candidates": config.RETRIEVE_CANDIDATES, "backend": "memory"})
            mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in r.summary().items()})
            mlflow.log_artifact(str(report_path))
    print(f"  logged {len(results)} runs to MLflow ({config.MLFLOW_URI})")


# ---------------------------------------------------------------- growing the golden set with human queries
#
# 1. People write queries in a CSV with columns: subject, query, author[, type]   (they must NOT browse the papers first)
# 2. `golden pool --input that.csv` -> judge_<name>.csv: for each query, the union of the top 10 from bm25, dense and
#    hybrid (TREC-style pooling), one row per (query, candidate question), with an empty `relevant` column.
# 3. Someone who knows the subject fills `relevant` with y/n.
# 4. `golden import judge_<name>.csv` -> golden_v<N+1>.jsonl = previous set + the new queries (qid prefix H-).
# Pooled relevance only covers what some ranker returned, so recall on these queries is measured against that pool.

POOL_DEPTH = 10


def pool(input_csv: Path, embedder: Optional[Embedder]) -> Path:
    with input_csv.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("query") or "").strip()]
    modes = ["bm25"] + (["dense", "hybrid"] if embedder is not None else [])
    if embedder is not None:
        embedder.embed([r["query"].strip() for r in rows], "RETRIEVAL_QUERY")
    else:
        print("  no embedder: pooling from bm25 only (judgements will favour keyword search; re-pool when quota allows)")
    retr = Retriever(MemoryBackend(load_chunks(), embedder), embedder)
    out = input_csv.with_name(f"judge_{input_csv.stem}.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "subject", "author", "type", "query", "question_id", "source", "question_text", "relevant"])
        for n, r in enumerate(rows, 1):
            subj = r["subject"].strip().upper()
            seen: list[str] = []
            for mode in modes:
                for h in retr.search(r["query"].strip(), Filters(subject=subj), k=POOL_DEPTH, mode=mode):
                    if h.chunk.chunk_id not in seen:
                        seen.append(h.chunk.chunk_id)
            for cid in sorted(seen):         # sorted, so the judge cannot tell which ranker put it first
                c = retr.backend.chunks[cid]
                w.writerow([n, subj, r.get("author", "").strip() or "unknown", (r.get("type") or "human").strip(),
                            r["query"].strip(), cid, c.source, " ".join(c.text.split()), ""])
    return out


def import_judgements(judge_csv: Path) -> Path:
    golden, version = load_golden()
    with judge_csv.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    blank = [r for r in rows if (r.get("relevant") or "").strip().lower() not in ("y", "n", "yes", "no")]
    if blank:
        raise ValueError(f"{len(blank)} rows in {judge_csv.name} have no y/n in 'relevant' (first: query {blank[0]['n']}, {blank[0]['question_id']})")
    by_n: dict[str, list[dict]] = {}
    for r in rows:
        by_n.setdefault(r["n"], []).append(r)
    start = sum(1 for g in golden if g.qid.startswith("H-")) + 1
    added, skipped = [], 0
    for i, (_, group) in enumerate(sorted(by_n.items(), key=lambda kv: int(kv[0]))):
        rel = [r["question_id"] for r in group if r["relevant"].strip().lower() in ("y", "yes")]
        if not rel:
            skipped += 1                   # nothing relevant in the bank: cannot score recall for it
            continue
        r0 = group[0]
        added.append(GoldenQuery(f"H-{start + len(added):03d}", r0["subject"], r0["type"] or "human", r0["author"], r0["query"], rel))
    new_version = int(version.rsplit("_v", 1)[1]) + 1
    out = config.GOLDEN_DIR / f"golden_v{new_version}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for g in golden + added:
            f.write(json.dumps(g.__dict__, ensure_ascii=False) + "\n")
    print(f"  {version} + {len(added)} new queries -> {out.name} ({skipped} queries had no relevant question and were left out)")
    return out
