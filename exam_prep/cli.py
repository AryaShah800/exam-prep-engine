"""Command line for Phases A and B. Run `python -m exam_prep --help`."""
import argparse, sys
from typing import Optional
from . import config, export, extract, review, split, syllabus, tag
from .llm import GeminiLLM, LLMError
from .metadata import discover_papers


def _llm(vision: bool = False, required: bool = True) -> Optional[GeminiLLM]:
    try:
        if vision:
            return GeminiLLM(model=config.GEMINI_VISION_MODEL, rpm=config.GEMINI_VISION_RPM)
        return GeminiLLM(model=config.GEMINI_MODEL)
    except LLMError:
        if required:
            raise
        return None


def _subjects(arg: Optional[str]) -> list[str]:
    if arg:
        return [arg.upper()]
    return sorted({m.subject for m in discover_papers()})


def _index(backend: Optional[str], embedder: Optional[str] = None) -> None:
    from .embed import make_embedder
    from .retrieval import load_chunks
    emb = make_embedder(embedder)
    backend = backend or config.RETRIEVAL_BACKEND
    if backend == "pg":
        from .pgstore import build_index
        n = build_index(emb)
        print(f"  loaded {n} questions into Postgres ({emb.model}, {emb.dim} dims, {emb.api_calls} embedding calls)")
    else:
        chunks = load_chunks()
        emb.embed([c.index_text for c in chunks], "RETRIEVAL_DOCUMENT")
        print(f"  {len(chunks)} questions embedded ({emb.model}, {emb.dim} dims, {emb.api_calls} new embedding calls; the rest came from the cache)")


def _query(args) -> None:
    from .answer import answer
    from .embed import make_embedder
    from .retrieval import Filters, build_retriever
    embedder = make_embedder(args.embedder) if args.mode != "bm25" else None
    retriever = build_retriever(embedder, args.backend)
    f = Filters(subject=args.subject, source=args.source, year_from=args.year_from, year_to=args.year_to, exam_type=args.exam_type)
    if args.cmd == "search":
        hits = retriever.search(args.query, f, k=args.k, mode=args.mode)
        for n, h in enumerate(hits, 1):
            c = h.chunk
            marks = f"{c.marks:g}m" if c.marks is not None else "?m"
            print(f"{n}. {c.citation} {c.source} {c.year} {marks}  score={h.score:.4f} bm25#{h.bm25_rank or '-'} dense#{h.dense_rank or '-'}")
            print(f"   {' '.join(c.text.split())[:220]}")
            if c.topics:
                print(f"   topics: {'; '.join(c.topics)}")
        if not hits:
            print("No matching questions.")
        return
    res = answer(args.query, retriever, _llm(), f, k=args.k, mode=args.mode)
    print(res.text)
    status = "citations verified" if res.grounded else "NOT grounded: " + "; ".join(res.problems[:3])
    print(f"\n[{status}; attempts={res.attempts}; {res.prompt_version}]")
    if not res.grounded:
        for h in res.hits:
            print(f"  {h.chunk.citation} {' '.join(h.chunk.text.split())[:160]}")


def _evaluate(args) -> None:
    from pathlib import Path
    from . import evaluate
    from .embed import make_embedder
    golden, version = evaluate.load_golden(Path(args.golden) if args.golden else None)
    if args.bm25_only:
        emb, setups = None, [s for s in evaluate.DEFAULT_SETUPS if s.mode == "bm25"]
    else:
        emb = make_embedder(args.embedder)
        setups = evaluate.DEFAULT_SETUPS[:3] if args.quick else None
    results = evaluate.run(golden, emb, setups)
    for r in results:
        s = r.summary()
        print(f"  {r.setup.name:<30} Recall@5 {s['recall@5']:.1%}  Hit@5 {s['hit@5']:.1%}  MRR@10 {s['mrr@10']:.3f}  nDCG@10 {s['ndcg@10']:.3f}")
    if emb is None:
        print("  bm25-only run: not saved as a report (run without --bm25-only for the full comparison)")
        return
    print(f"  report: {evaluate.save(golden, version, results, emb.model)}  ({emb.api_calls} new embedding calls)")


def main(argv: Optional[list[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):   # Windows consoles default to cp1252, which cannot print maths symbols in questions
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="exam_prep", description="Phase A: past papers -> tagged questions")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="PDF -> page text (Gemini vision for scanned pages)")
    e.add_argument("--subject")
    e.add_argument("--source", help="DDU, external, or a source code like BITM")
    e.add_argument("--overwrite", action="store_true")
    e.add_argument("--force-ocr", action="store_true", help="send every page to Gemini vision")

    s = sub.add_parser("split", help="page text -> questions")
    s.add_argument("--subject")
    s.add_argument("--source", help="DDU, external, or a source code like BITM")
    s.add_argument("--method", choices=["auto", "rules", "llm"], default="auto")

    y = sub.add_parser("syllabus", help="syllabus -> topic taxonomy JSON")
    y.add_argument("--subject", required=True)
    y.add_argument("--overwrite", action="store_true")

    t = sub.add_parser("tag", help="questions -> 1-2 topics each")
    t.add_argument("--subject")
    t.add_argument("--retag", action="store_true")
    t.add_argument("--limit", type=int, help="tag only the first N untagged questions (to try a prompt cheaply)")

    r = sub.add_parser("review", help="hand-check tags")
    r.add_argument("action", choices=["sample", "score"])
    r.add_argument("--n", type=int, default=50)
    r.add_argument("--version", help="prompt version, e.g. tag_v1 (default: latest)")
    r.add_argument("--reviewer", default="human", choices=["human", "claude"], help="who filled the verdicts (default: human)")

    sub.add_parser("export", help="write questions_tagged.jsonl for Phase B")
    sub.add_parser("stats", help="dataset numbers for the README")

    i = sub.add_parser("index", help="Phase B: embed every question (cached) and build the search index")
    i.add_argument("--backend", choices=["memory", "pg"], default=None, help="default: RETRIEVAL_BACKEND (memory)")
    i.add_argument("--embedder", choices=["gemini", "local"], default=None, help="default: EMBED_PROVIDER (gemini)")

    for name, helptext in (("search", "Phase B: hybrid search over past questions"), ("ask", "Phase B: answer from past questions, with checked citations")):
        q = sub.add_parser(name, help=helptext)
        q.add_argument("query")
        q.add_argument("--subject")
        q.add_argument("--source", help="all (default), DDU, external, or a code like BITM")
        q.add_argument("--year-from", type=int)
        q.add_argument("--year-to", type=int)
        q.add_argument("--exam-type", choices=["mid", "end", "remedial", "other"])
        q.add_argument("--k", type=int, default=5 if name == "search" else 8)
        q.add_argument("--mode", choices=["hybrid", "bm25", "dense"], default="hybrid")
        q.add_argument("--backend", choices=["memory", "pg"], default=None)
        q.add_argument("--embedder", choices=["gemini", "local"], default=None, help="default: EMBED_PROVIDER (gemini)")

    ev = sub.add_parser("evaluate", help="Phase C: Recall@5 / MRR for bm25, dense and hybrid on the golden set")
    ev.add_argument("--golden", help="path to a golden_vN.jsonl (default: highest N in data/golden/)")
    ev.add_argument("--quick", action="store_true", help="only bm25, dense and hybrid (skip ablations)")
    ev.add_argument("--bm25-only", action="store_true", help="keyword setups only: needs no embedding quota")
    ev.add_argument("--embedder", choices=["gemini", "local"], default=None, help="default: EMBED_PROVIDER (gemini)")

    ea = sub.add_parser("evaluate-answers", help="Phase C: RAGAS faithfulness of generated answers (pip install -r requirements-eval.txt)")
    ea.add_argument("--per-subject", type=int, default=10)
    ea.add_argument("--embedder", choices=["gemini", "local"], default=None, help="default: EMBED_PROVIDER (gemini)")

    gs = sub.add_parser("golden", help="Phase C: add human-written queries to the golden set")
    gs.add_argument("action", choices=["pool", "import"])
    gs.add_argument("--embedder", choices=["gemini", "local"], default=None, help="default: EMBED_PROVIDER (gemini)")
    gs.add_argument("--input", required=True, help="pool: CSV with subject,query,author[,type]; import: the filled judge_*.csv")

    a = sub.add_parser("all", help="extract -> split -> tag -> export (taxonomies must exist)")
    a.add_argument("--subject")

    args = p.parse_args(argv)
    try:
        if args.cmd == "extract":
            metas = discover_papers(subject=args.subject, source=args.source)
            print(f"Extracting {len(metas)} papers")
            print(extract.run(metas, _llm(vision=True, required=False), overwrite=args.overwrite, force_ocr=args.force_ocr))
        elif args.cmd == "split":
            metas = discover_papers(subject=args.subject, source=args.source)
            print(f"Splitting {len(metas)} papers ({args.method})")
            print(split.run(metas, args.method, _llm(required=args.method == "llm")))
        elif args.cmd == "syllabus":
            syllabus.run(args.subject, _llm(vision=True), _llm(vision=True), overwrite=args.overwrite)
        elif args.cmd == "tag":
            llm = _llm()
            for subj in _subjects(args.subject):
                tag.run(subj, llm, retag=args.retag, limit=args.limit)
        elif args.cmd == "review":
            if args.action == "sample":
                review.sample(n=args.n, version=args.version)
            else:
                review.score(version=args.version, reviewer=args.reviewer)
        elif args.cmd == "export":
            export.export()
        elif args.cmd == "stats":
            export.stats()
        elif args.cmd == "index":
            _index(args.backend, args.embedder)
        elif args.cmd in ("search", "ask"):
            _query(args)
        elif args.cmd == "evaluate":
            _evaluate(args)
        elif args.cmd == "evaluate-answers":
            from . import answer_eval, evaluate
            from .embed import make_embedder
            from .llm import load_prompt
            from .retrieval import build_retriever
            golden, version = evaluate.load_golden()
            llm = _llm()
            emb = make_embedder(args.embedder)
            records = answer_eval.run(golden, build_retriever(emb, "memory"), llm, answer_eval.ragas_judge(), args.per_subject)
            for k, v in answer_eval.summarise(records).items():
                print(f"  {k:<22} {v:.3f}" if isinstance(v, float) else f"  {k:<22} {v}")
            print(f"  report: {answer_eval.save(records, version, llm.model, config.GEMINI_MODEL, load_prompt('answer')[1], emb.model)}")
        elif args.cmd == "golden":
            from pathlib import Path
            from . import evaluate
            if args.action == "pool":
                try:
                    from .embed import make_embedder
                    emb = make_embedder(args.embedder)
                except LLMError:
                    emb = None
                print(f"  judging sheet: {evaluate.pool(Path(args.input), emb)}  (fill the 'relevant' column with y/n)")
            else:
                evaluate.import_judgements(Path(args.input))
        elif args.cmd == "all":
            metas = discover_papers(subject=args.subject)
            llm = _llm(required=False)
            extract.run(metas, _llm(vision=True, required=False))
            split.run(metas, "auto", llm)
            if llm is None:
                print("No GEMINI_API_KEY: stopping before tagging.")
            else:
                for subj in sorted({m.subject for m in metas}):
                    tag.run(subj, llm)
            export.export()
            export.stats()
    except (FileNotFoundError, RuntimeError, ValueError, LLMError) as err:
        print(f"error: {err}")
        return 1
    return 0
