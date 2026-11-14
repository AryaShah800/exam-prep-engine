"""Phase C, part 2: how faithful are generated answers to the retrieved questions?

For a fixed, seeded sample of golden-set queries (N per subject), the app's own pipeline runs end to end:
hybrid retrieval -> Gemini answer -> our citation check. Then RAGAS `Faithfulness` (ragas.metrics.collections)
splits each grounded answer into atomic statements and asks an LLM judge whether each statement is supported by
the retrieved questions; the score is supported / total statements.

Also reported, from our own pipeline: how often an answer passed the citation check, how many attempts it took,
and how often the model said none of the retrieved questions matched.

Quota: 1 generation call per query (+ retries on failed citation checks) and 2 judge calls per grounded answer, on
the Flash-Lite free tier. Everything is cached (our LLM cache for answers, a RAGAS disk cache for judge calls),
so a rerun after a 503 or a quota stop continues where it stopped.

RAGAS needs the LangChain 0.3 line, which conflicts with LangGraph (Phase E), so it lives in requirements-eval.txt.
"""
import asyncio, json, math, random, time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
from . import config
from .answer import answer, format_sources
from .evaluate import REPORTS_DIR, GoldenQuery, mean
from .llm import LLM
from .retrieval import Filters, Retriever

Judge = Callable[[str, str, list[str]], float]     # (query, response, contexts) -> faithfulness in [0, 1] or nan


@dataclass
class AnswerRecord:
    qid: str
    subject: str
    query: str
    grounded: bool
    no_match: bool
    attempts: int
    n_citations: int
    response: str
    contexts: list[str]
    faithfulness: Optional[float] = None


def sample_queries(golden: list[GoldenQuery], per_subject: int, seed: int = 0) -> list[GoldenQuery]:
    rng = random.Random(seed)
    out = []
    for subj in sorted({g.subject for g in golden}):
        pool = sorted((g for g in golden if g.subject == subj), key=lambda g: g.qid)
        out += sorted(rng.sample(pool, min(per_subject, len(pool))), key=lambda g: g.qid)
    return out


GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
TRANSIENT = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "timed out", "Timeout", "Connection")


def ragas_judge(model: Optional[str] = None) -> Judge:
    """RAGAS Faithfulness with a Gemini judge. RAGAS calls the LLM asynchronously, and its own source recommends
    Gemini's OpenAI-compatible endpoint over the google-genai instructor adapter (upstream safety-settings bug),
    so the judge uses AsyncOpenAI pointed at Gemini. Same API key, same free tier."""
    from openai import AsyncOpenAI
    from ragas.cache import DiskCacheBackend
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness
    client = AsyncOpenAI(api_key=config.GEMINI_API_KEY, base_url=GEMINI_OPENAI_URL, timeout=config.GEMINI_TIMEOUT_S, max_retries=0)
    cache = DiskCacheBackend(cache_dir=str(config.CACHE_DIR.parent / "ragas"))
    llm = llm_factory(model or config.GEMINI_MODEL, provider="openai", client=client, cache=cache)
    metric = Faithfulness(llm=llm)
    # One event loop for the judge's lifetime: the async client's connection pool is bound to the loop that first used
    # it, so a fresh asyncio.run() per call fails later calls with "Connection error".
    loop = asyncio.new_event_loop()

    def judge(query: str, response: str, contexts: list[str]) -> float:
        last = None
        for attempt in range(config.GEMINI_MAX_RETRIES):
            try:
                return float(loop.run_until_complete(metric.ascore(user_input=query, response=response, retrieved_contexts=contexts)).value)
            except Exception as e:
                msg = str(e)
                if "PerDay" in msg or not any(t in msg for t in TRANSIENT):
                    raise                     # daily quota or a real bug: retrying cannot help
                last = e
                wait = min(60, 2 ** attempt * 4)
                print(f"  judge call failed ({type(e).__name__}); retrying in {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"RAGAS judge failed after {config.GEMINI_MAX_RETRIES} attempts: {last}")
    return judge


def run(golden: list[GoldenQuery], retriever: Retriever, llm: LLM, judge: Judge, per_subject: int = 10, k: int = 8,
        out_path: Optional[Path] = None, pause_s: Optional[float] = None) -> list[AnswerRecord]:
    out_path = out_path or (config.DATA / "eval" / "answers.jsonl")
    done: dict[str, AnswerRecord] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = AnswerRecord(**json.loads(line))
                done[r.qid] = r
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    gap = 2 * 60.0 / max(config.GEMINI_RPM, 1) if pause_s is None else pause_s   # 2 judge calls per answer, under the per-minute limit
    for g in sample_queries(golden, per_subject):
        rec = done.get(g.qid)
        if rec is None or (rec.grounded and rec.faithfulness is None):
            res = answer(g.query, retriever, llm, Filters(subject=g.subject), k=k)
            no_match = (not res.grounded) and res.attempts == 1 and not res.problems
            rec = AnswerRecord(g.qid, g.subject, g.query, res.grounded, no_match, res.attempts, len(res.citations),
                               res.text, format_sources(res.hits).splitlines())
            if rec.grounded:
                rec.faithfulness = judge(g.query, rec.response, rec.contexts)
                time.sleep(gap)
            done[g.qid] = rec
            with out_path.open("w", encoding="utf-8") as f:     # rewrite after each query: safe to stop at any point
                for r in done.values():
                    f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            print(f"  {g.qid}: grounded={rec.grounded} attempts={rec.attempts} faithfulness={rec.faithfulness}")
        records.append(rec)
    return records


def summarise(records: list[AnswerRecord]) -> dict[str, float]:
    grounded = [r for r in records if r.grounded]
    scored = [r.faithfulness for r in grounded if r.faithfulness is not None and not math.isnan(r.faithfulness)]
    return {
        "answers": len(records),
        "grounded_rate": len(grounded) / len(records) if records else 0.0,
        "no_match_rate": sum(r.no_match for r in records) / len(records) if records else 0.0,
        "refused_rate": sum((not r.grounded) and (not r.no_match) for r in records) / len(records) if records else 0.0,
        "first_try_rate": sum(r.grounded and r.attempts == 1 for r in records) / len(records) if records else 0.0,
        "faithfulness_mean": mean(scored),
        "faithfulness_n": len(scored),
        "fully_faithful_rate": sum(1 for s in scored if s >= 0.999) / len(scored) if scored else 0.0,
    }


def report(records: list[AnswerRecord], golden_version: str, gen_model: str, judge_model: str, prompt_version: str, embed_model: str = "") -> str:
    s = summarise(records)
    worst = sorted((r for r in records if r.faithfulness is not None and not math.isnan(r.faithfulness)), key=lambda r: r.faithfulness)[:5]
    lines = [f"# Answer faithfulness — {golden_version}", "",
             f"- Run: {datetime.now():%Y-%m-%d %H:%M}. Pipeline: hybrid retrieval (k=8, embeddings `{embed_model or '?'}`) -> `{gen_model}` with `{prompt_version}` -> citation check (max {config.ANSWER_MAX_ATTEMPTS} attempts).",
             f"- Judge: RAGAS `Faithfulness` (ragas.metrics.collections), LLM `{judge_model}`. Contexts = the retrieved questions with source, year, exam and marks.",
             f"- Sample: {s['answers']} golden-set queries, a fixed seeded sample per subject.", "",
             "| Metric | Value |", "| --- | --- |",
             f"| RAGAS faithfulness (mean over {s['faithfulness_n']} grounded answers) | {s['faithfulness_mean']:.1%} |",
             f"| Answers with every statement supported | {s['fully_faithful_rate']:.1%} |",
             f"| Passed our citation check | {s['grounded_rate']:.1%} |",
             f"| ...on the first attempt | {s['first_try_rate']:.1%} |",
             f"| Model said nothing retrieved matched | {s['no_match_rate']:.1%} |",
             f"| Refused after failing the citation check {config.ANSWER_MAX_ATTEMPTS} times | {s['refused_rate']:.1%} |",
             "", "## Lowest-scoring answers", ""]
    for r in worst:
        lines += [f"**{r.qid}** ({r.faithfulness:.0%}) — {r.query}", "", "```", r.response, "```", ""]
    lines += ["## Caveats", "",
              "- The judge is an LLM; when it is the same model family as the generator, it may be lenient on its own phrasing.",
              "- Faithfulness is measured only on answers that passed the citation check; refused answers are counted above, not scored.",
              "- The queries come from the golden set, written by Claude (AI) unless the author column says otherwise."]
    return "\n".join(lines) + "\n"


def save(records: list[AnswerRecord], golden_version: str, gen_model: str, judge_model: str, prompt_version: str, embed_model: str = "") -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"answer_eval_{golden_version}.md"
    path.write_text(report(records, golden_version, gen_model, judge_model, prompt_version, embed_model), encoding="utf-8")
    try:
        import mlflow
        mlflow.set_tracking_uri(config.MLFLOW_URI)
        mlflow.set_experiment("exam-prep-answers")
        with mlflow.start_run(run_name=f"{golden_version} | faithfulness"):
            mlflow.log_params({"golden_set": golden_version, "gen_model": gen_model, "judge_model": judge_model,
                               "prompt": prompt_version, "retrieval": "hybrid", "k": 8, "embed_model": embed_model})
            mlflow.log_metrics({k: float(v) for k, v in summarise(records).items()})
            mlflow.log_artifact(str(path))
    except ImportError:
        pass
    return path
