"""Postgres + pgvector backend. Runs only when EXAM_PREP_PG_TEST_URL points at a database with pgvector
(CI starts pgvector/pgvector:pg17 as a service); skipped otherwise. The test drops and recreates the chunks table."""
import os
import pytest
from exam_prep.retrieval import Filters, Retriever
from test_retrieval import FakeEmbedder, _write_export

PG_URL = os.getenv("EXAM_PREP_PG_TEST_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="set EXAM_PREP_PG_TEST_URL to a pgvector database to run")


@pytest.fixture
def pg_retriever(workspace):
    from exam_prep.pgstore import PgBackend, build_index
    _write_export()
    emb = FakeEmbedder()
    assert build_index(emb, dsn=PG_URL) == 5
    return Retriever(PgBackend(PG_URL), emb, external_weight=0.95)


def test_pg_index_has_hnsw_and_fts(pg_retriever):
    names = {r[0] for r in pg_retriever.backend.conn.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'chunks'").fetchall()}
    assert {"chunks_embedding_hnsw", "chunks_tsv_gin"} <= names


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_pg_search_matches_memory_behaviour(pg_retriever, mode):
    ids = [h.chunk.chunk_id for h in pg_retriever.search("important questions on deadlock", Filters(subject="DC"), k=3, mode=mode)]
    assert ids[0] == "DC_2025_09_MID-Q2b"
    if mode != "dense":
        assert ids[1] == "BITM-DC_2019_11_END-Q4a"


def test_pg_filters(pg_retriever):
    def ids(**kw):
        return {h.chunk.chunk_id for h in pg_retriever.search("banker deadlock dockerfile search", Filters(**kw), k=10)}
    assert all(i.startswith("AI_") for i in ids(subject="AI"))
    assert ids(source="external") == {"BITM-DC_2019_11_END-Q4a"}
    assert ids(subject="DC", year_to=2020) == {"BITM-DC_2019_11_END-Q4a"}


def test_pg_query_of_only_noise_words(pg_retriever):
    assert pg_retriever.backend.keyword("important questions", Filters(), 10) == []
