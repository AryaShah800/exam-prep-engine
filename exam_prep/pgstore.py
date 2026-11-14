"""Postgres backend: pgvector (HNSW, cosine) for dense search and Postgres full-text search for keywords.

Start a database for free with Docker (port 5433 so it does not clash with a local Postgres install):

    docker run -d --name examprep-pg -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=examprep -p 5433:5432 pgvector/pgvector:pg17

Then: `python -m exam_prep index --backend pg` loads every chunk with its embedding, and
`--backend pg` (or RETRIEVAL_BACKEND=pg) on search/ask queries it. Keyword ranking here is ts_rank_cd over a
tsvector (stemmed English), not BM25; the memory backend has true BM25. Phase C reports which backend it ran on.
"""
from typing import Optional
import numpy as np
from . import config
from .embed import Embedder
from .retrieval import Chunk, Filters, QUERY_NOISE, load_chunks

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   text PRIMARY KEY,
    paper_id   text NOT NULL,
    label      text NOT NULL,
    subject    text NOT NULL,
    source     text NOT NULL,
    year       int  NOT NULL,
    month      int,
    exam_type  text NOT NULL,
    marks      real,
    topic_ids  text[] NOT NULL DEFAULT '{{}}',
    topics     text[] NOT NULL DEFAULT '{{}}',
    off_syllabus boolean NOT NULL DEFAULT false,
    text       text NOT NULL,
    index_text text NOT NULL,
    embedding  vector({dim}) NOT NULL,
    tsv        tsvector GENERATED ALWAYS AS (to_tsvector('english', index_text)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_filters ON chunks (subject, source, year);
"""

COLUMNS = ["chunk_id", "paper_id", "label", "subject", "source", "year", "month", "exam_type", "marks",
           "topic_ids", "topics", "off_syllabus", "text", "index_text"]


def connect(dsn: Optional[str] = None):
    import psycopg
    from pgvector.psycopg import register_vector
    conn = psycopg.connect(dsn or config.DATABASE_URL, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def _where(f: Filters) -> tuple[str, list]:
    sql, args = [], []
    if f.subject:
        sql.append("subject = %s"); args.append(f.subject.upper())
    src = (f.source or "all").upper()
    if src == "EXTERNAL":
        sql.append("source <> 'DDU'")
    elif src != "ALL":
        sql.append("upper(source) = %s"); args.append(src)
    if f.year_from:
        sql.append("year >= %s"); args.append(f.year_from)
    if f.year_to:
        sql.append("year <= %s"); args.append(f.year_to)
    if f.exam_type:
        sql.append("exam_type = %s"); args.append(f.exam_type.lower())
    return (" AND ".join(sql) or "TRUE"), args


def build_index(embedder: Embedder, dsn: Optional[str] = None, with_topics: bool = True) -> int:
    """Drop and reload the chunks table from questions_tagged.jsonl (embeddings come from the cache)."""
    chunks = load_chunks(with_topics)
    vecs = embedder.embed([c.index_text for c in chunks], "RETRIEVAL_DOCUMENT")
    with connect(dsn) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks")
        for stmt in SCHEMA.format(dim=embedder.dim).split(";"):
            if stmt.strip():
                conn.execute(stmt)
        cols = ", ".join(COLUMNS + ["embedding"])
        marks = ", ".join(["%s"] * (len(COLUMNS) + 1))
        with conn.cursor() as cur:
            cur.executemany(f"INSERT INTO chunks ({cols}) VALUES ({marks})",
                            [[getattr(c, k) for k in COLUMNS] + [v] for c, v in zip(chunks, vecs)])
        conn.execute("ANALYZE chunks")
    return len(chunks)


class PgBackend:
    def __init__(self, dsn: Optional[str] = None):
        self.conn = connect(dsn)
        try:   # pgvector >= 0.8: keep scanning the HNSW graph until enough rows pass the WHERE filters
            self.conn.execute("SET hnsw.iterative_scan = relaxed_order")
        except Exception:
            pass
        rows = self.conn.execute(f"SELECT {', '.join(COLUMNS)} FROM chunks").fetchall()
        self.chunks = {r[0]: Chunk(**dict(zip(COLUMNS, r))) for r in rows}

    def keyword(self, query: str, filters: Filters, n: int) -> list[str]:
        words = [w for w in query.split() if w.lower().strip("?.,!") not in QUERY_NOISE]
        if not words:
            return []
        where, args = _where(filters)
        # plainto_tsquery ANDs the words; swap & for | so a document matching some of the words still ranks.
        sql = (f"WITH q AS (SELECT NULLIF(replace(plainto_tsquery('english', %s)::text, '&', '|'), '')::tsquery AS q) "
               f"SELECT chunk_id FROM chunks, q WHERE q.q IS NOT NULL AND tsv @@ q.q AND {where} "
               f"ORDER BY ts_rank_cd(tsv, q.q) DESC, chunk_id LIMIT %s")
        return [r[0] for r in self.conn.execute(sql, [" ".join(words), *args, n]).fetchall()]

    def dense(self, qvec: np.ndarray, filters: Filters, n: int) -> list[str]:
        where, args = _where(filters)
        sql = f"SELECT chunk_id FROM chunks WHERE {where} ORDER BY embedding <=> %s LIMIT %s"
        return [r[0] for r in self.conn.execute(sql, [*args, np.asarray(qvec, dtype=np.float32), n]).fetchall()]
