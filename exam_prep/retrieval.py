"""Phase B: hybrid retrieval over past-paper questions.

One chunk = one question (or sub-part), read from questions_tagged.jsonl. Two rankers run on the same filtered
set and are fused with Reciprocal Rank Fusion:

    keyword  BM25 (memory backend) or Postgres full-text search (pg backend)
    dense    Gemini embeddings, cosine similarity (exact in memory, HNSW in pgvector)
    hybrid   RRF(k=60): score(d) = sum over rankers of 1 / (k + rank), times EXTERNAL_WEIGHT for non-DDU questions

`mode` picks bm25 / dense / hybrid so Phase C can compare all three on one golden set.
"""
import json, math, re
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol
import numpy as np
from pydantic import BaseModel
from . import config
from .embed import Embedder

Mode = Literal["bm25", "dense", "hybrid"]


class Chunk(BaseModel):
    chunk_id: str                      # = question_id, e.g. DC_2025_09_MID-Q2b or BITM-AI_2019_11_END-Q3a-OR
    paper_id: str
    label: str                         # question label inside the paper: Q2b, Q3a-OR
    subject: str
    source: str                        # DDU or a university code (BITM, UPES, AKTU, MU)
    year: int
    month: Optional[int] = None
    exam_type: str
    marks: Optional[float] = None
    topic_ids: list[str] = []
    topics: list[str] = []
    off_syllabus: bool = False
    text: str                          # the question as printed
    index_text: str                    # what both rankers see: question text + topic names

    @property
    def citation(self) -> str:
        return f"[{self.paper_id}, {self.label}]"


# Watermark that survived into already-split questions (extract.py now strips it at the source).
LEFTOVER_WATERMARK = re.compile(r"\s*\bQP\d{2}\w*\s+\d{1,3}(?:\.\d{1,3}){3}\b")


def chunk_from_row(row: dict, with_topics: bool = True) -> Chunk:
    qid, pid = row["question_id"], row["paper_id"]
    row = {**row, "text": LEFTOVER_WATERMARK.sub("", row["text"])}
    label = qid[len(pid) + 1:] if qid.startswith(pid + "-") else qid
    topics = row.get("topics") or []
    index_text = row["text"] + (f"\nTopics: {'; '.join(topics)}" if with_topics and topics else "")
    return Chunk(chunk_id=qid, paper_id=pid, label=label, subject=row["subject"], source=row.get("source", "DDU"),
                 year=row["year"], month=row.get("month"), exam_type=row["exam_type"], marks=row.get("marks"),
                 topic_ids=row.get("topic_ids") or [], topics=topics, off_syllabus=bool(row.get("off_syllabus")),
                 text=row["text"], index_text=index_text)


def load_chunks(with_topics: bool = True) -> list[Chunk]:
    if not config.EXPORT_FILE.exists():
        raise FileNotFoundError(f"{config.EXPORT_FILE} not found. Run Phase A first: python -m exam_prep export")
    with config.EXPORT_FILE.open(encoding="utf-8") as f:
        return [chunk_from_row(json.loads(line), with_topics) for line in f if line.strip()]


# ---------------------------------------------------------------- filters

@dataclass
class Filters:
    subject: Optional[str] = None
    source: Optional[str] = None       # None/"all", "DDU", "external", or a code like BITM
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    exam_type: Optional[str] = None    # mid / end / remedial / other

    def match(self, c: Chunk) -> bool:
        if self.subject and c.subject != self.subject.upper():
            return False
        src = (self.source or "all").upper()
        if src == "EXTERNAL" and c.source == "DDU":
            return False
        if src not in ("ALL", "EXTERNAL") and c.source.upper() != src:
            return False
        if self.year_from and c.year < self.year_from:
            return False
        if self.year_to and c.year > self.year_to:
            return False
        if self.exam_type and c.exam_type != self.exam_type.lower():
            return False
        return True


# ---------------------------------------------------------------- text processing for the keyword ranker

STOPWORDS = set("""a an the and or of to in on for with by from at as is are was were be been being it its this that these
those what which who whom how why when where do does did can could should would will shall may might must not no
than then there their them they you your we our i me my he she his her into about over under between any each all
some such using use used following given give""".split())
# Words students type that say nothing about the topic ("important questions on deadlock").
QUERY_NOISE = set("""important imp question questions ques asked ask previous past year years pyq pyqs exam exams paper
papers mark marks sessional sessionals midsem endsem semester related topic topics frequently repeated most show find
explain describe discuss define write short note notes""".split())

_stemmer = None


def _stem(word: str) -> str:
    global _stemmer
    if _stemmer is None:
        import snowballstemmer
        _stemmer = snowballstemmer.stemmer("english")
    return _stemmer.stemWord(word)


def tokenize(text: str, query: bool = False) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    drop = STOPWORDS | QUERY_NOISE if query else STOPWORDS
    return [_stem(w) for w in words if w not in drop and (len(w) > 1 or w.isdigit())]


class BM25:
    """Okapi BM25 (k1=1.5, b=0.75) with the non-negative IDF used by Lucene."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.len.mean()) if docs else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf), dtype=np.float32)
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avgdl))
        return out


# ---------------------------------------------------------------- backends

class Backend(Protocol):
    chunks: dict[str, Chunk]

    def keyword(self, query: str, filters: Filters, n: int) -> list[str]: ...

    def dense(self, qvec: np.ndarray, filters: Filters, n: int) -> list[str]: ...


class MemoryBackend:
    """Everything in RAM: exact cosine search and BM25. Fine for a few thousand questions; used by tests and Phase C."""

    def __init__(self, chunks: list[Chunk], embedder: Optional[Embedder]):
        self.order = [c.chunk_id for c in chunks]
        self.chunks = {c.chunk_id: c for c in chunks}
        self.bm25 = BM25([tokenize(c.index_text) for c in chunks])
        self.matrix = embedder.embed([c.index_text for c in chunks], "RETRIEVAL_DOCUMENT") if embedder else None

    def _mask(self, filters: Filters) -> np.ndarray:
        return np.array([filters.match(self.chunks[i]) for i in self.order], dtype=bool)

    def keyword(self, query: str, filters: Filters, n: int) -> list[str]:
        terms = tokenize(query, query=True)
        if not terms:
            return []
        s = self.bm25.scores(terms)
        s[~self._mask(filters)] = 0
        idx = [i for i in np.argsort(-s, kind="stable")[:n] if s[i] > 0]
        return [self.order[i] for i in idx]

    def dense(self, qvec: np.ndarray, filters: Filters, n: int) -> list[str]:
        if self.matrix is None:
            raise RuntimeError("dense search needs an embedder")
        s = self.matrix @ qvec
        s[~self._mask(filters)] = -np.inf
        idx = [i for i in np.argsort(-s, kind="stable")[:n] if np.isfinite(s[i])]
        return [self.order[i] for i in idx]


# ---------------------------------------------------------------- fusion + search

def rrf(rankings: list[list[str]], k: int = config.RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None


@dataclass
class Retriever:
    backend: Backend
    embedder: Optional[Embedder] = None
    rrf_k: int = config.RRF_K
    candidates: int = config.RETRIEVE_CANDIDATES
    external_weight: float = config.EXTERNAL_WEIGHT
    last_lists: dict = field(default_factory=dict)

    def search(self, query: str, filters: Optional[Filters] = None, k: int = 5, mode: Mode = "hybrid") -> list[Hit]:
        filters = filters or Filters()
        kw = self.backend.keyword(query, filters, self.candidates) if mode in ("bm25", "hybrid") else []
        dn: list[str] = []
        if mode in ("dense", "hybrid"):
            if self.embedder is None:
                raise RuntimeError("dense/hybrid search needs an embedder (GEMINI_API_KEY)")
            qvec = self.embedder.embed([query], "RETRIEVAL_QUERY")[0]
            dn = self.backend.dense(qvec, filters, self.candidates)
        self.last_lists = {"bm25": kw, "dense": dn}
        kw_rank = {c: r for r, c in enumerate(kw, 1)}
        dn_rank = {c: r for r, c in enumerate(dn, 1)}
        if mode == "bm25":
            scores = {c: 1.0 / (self.rrf_k + r) for c, r in kw_rank.items()}
        elif mode == "dense":
            scores = {c: 1.0 / (self.rrf_k + r) for c, r in dn_rank.items()}
        else:
            scores = rrf([kw, dn], self.rrf_k)
        for cid in scores:
            if self.backend.chunks[cid].source != "DDU":
                scores[cid] *= self.external_weight
        best = sorted(scores, key=lambda c: (-scores[c], c))[:k]
        return [Hit(self.backend.chunks[c], scores[c], kw_rank.get(c), dn_rank.get(c)) for c in best]


def build_retriever(embedder: Optional[Embedder], backend: Optional[str] = None, with_topics: bool = True) -> Retriever:
    backend = backend or config.RETRIEVAL_BACKEND
    if backend == "pg":
        from .pgstore import PgBackend
        return Retriever(PgBackend(), embedder)
    return Retriever(MemoryBackend(load_chunks(with_topics), embedder), embedder)
