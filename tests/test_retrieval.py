"""Phase B tests: chunking, BM25, RRF, filters, DDU preference, embedding cache, citation checking. No network."""
import hashlib, json
import numpy as np
import pytest
from exam_prep import config
from exam_prep.answer import LLMAnswer, answer, check_citations, tidy
from exam_prep.embed import CachedEmbedder
from exam_prep.retrieval import BM25, Filters, MemoryBackend, Retriever, chunk_from_row, load_chunks, rrf, tokenize

ROWS = [
    ("DC_2025_09_MID-Q2b", "DC_2025_09_MID", "DC", "DDU", 2025, 6, "Explain the Banker's algorithm for deadlock avoidance.", ["Deadlock"]),
    ("DC_2025_10_MID-Q3a-OR", "DC_2025_10_MID", "DC", "DDU", 2025, 6, "Write a Dockerfile for a Python web app and build the image.", ["Dockerfile, Docker Images and Registry"]),
    ("DC_2025_07_MID-Q1a", "DC_2025_07_MID", "DC", "DDU", 2025, 2, "Define container orchestration.", ["Kubernetes"]),
    ("BITM-DC_2019_11_END-Q4a", "BITM-DC_2019_11_END", "DC", "BITM", 2019, 5, "Explain the Banker's algorithm for deadlock avoidance.", ["Deadlock"]),
    ("AI_2025_07_MID-Q2a", "AI_2025_07_MID", "AI", "DDU", 2025, 6, "Apply A* search to the graph and show the open list.", ["Informed search"]),
]


def _write_export():
    config.EXPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with config.EXPORT_FILE.open("w", encoding="utf-8") as f:
        for qid, pid, subj, src, year, marks, text, topics in ROWS:
            f.write(json.dumps({"question_id": qid, "paper_id": pid, "subject": subj, "source": src, "year": year, "month": 9,
                                "exam_type": "mid", "marks": marks, "text": text, "topics": topics, "topic_ids": []}) + "\n")


class FakeEmbedder(CachedEmbedder):
    """Hashed bag of stemmed words: texts sharing words get similar vectors. Counts would-be API calls."""

    def __init__(self):
        super().__init__("fake-embed", 64)

    def _embed_batch(self, texts, task):
        self.api_calls += 1
        out = []
        for t in texts:
            v = np.zeros(self.dim)
            for w in tokenize(t, query=True):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
            out.append(v.tolist())
        return out


@pytest.fixture
def retriever(workspace):
    _write_export()
    emb = FakeEmbedder()
    return Retriever(MemoryBackend(load_chunks(), emb), emb, external_weight=0.95)


def test_chunk_label_and_citation():
    c = chunk_from_row({"question_id": "BITM-AI_2019_11_END-Q3a-OR", "paper_id": "BITM-AI_2019_11_END", "subject": "AI",
                        "source": "BITM", "year": 2019, "exam_type": "end", "text": "Explain A*.", "topics": ["Informed search"]})
    assert c.label == "Q3a-OR" and c.citation == "[BITM-AI_2019_11_END, Q3a-OR]"
    assert c.index_text.endswith("Topics: Informed search")


def test_query_noise_is_dropped_for_keywords():
    assert tokenize("important questions on deadlock", query=True) == ["deadlock"]
    assert "question" not in tokenize("Important previous year questions", query=True)


def test_bm25_prefers_rarer_matching_term():
    bm = BM25([tokenize("deadlock avoidance banker"), tokenize("docker image build"), tokenize("docker deadlock")])
    s = bm.scores(tokenize("banker deadlock", query=True))
    assert s.argmax() == 0 and s[1] == 0


def test_rrf_formula():
    s = rrf([["a", "b"], ["b", "c"]], k=60)
    assert s["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert s["a"] == pytest.approx(1 / 61) and s["c"] == pytest.approx(1 / 62)


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_search_finds_topic_and_prefers_ddu(retriever, mode):
    hits = retriever.search("important questions on deadlock", Filters(subject="DC"), k=3, mode=mode)
    ids = [h.chunk.chunk_id for h in hits]
    assert ids[0] == "DC_2025_09_MID-Q2b"            # identical text at DDU and BITM: DDU ranks first
    if mode != "dense":                              # a keyword match keeps the BITM copy above unrelated DDU questions
        assert ids[1] == "BITM-DC_2019_11_END-Q4a"


def test_filters(retriever):
    def ids(**kw):
        return {h.chunk.chunk_id for h in retriever.search("banker deadlock dockerfile search", Filters(**kw), k=10)}
    assert all(i.startswith("AI_") for i in ids(subject="AI"))
    assert ids(source="external") == {"BITM-DC_2019_11_END-Q4a"}
    assert "BITM-DC_2019_11_END-Q4a" not in ids(source="DDU")
    assert ids(subject="DC", year_to=2020) == {"BITM-DC_2019_11_END-Q4a"}


def test_embeddings_are_cached_on_disk(workspace):
    emb = FakeEmbedder()
    emb.embed(["alpha beta", "gamma"], "RETRIEVAL_DOCUMENT")
    assert emb.api_calls == 1
    again = FakeEmbedder()                       # new object: must read the disk cache
    v = again.embed(["gamma", "alpha beta"], "RETRIEVAL_DOCUMENT")
    assert again.api_calls == 0 and np.allclose(np.linalg.norm(v, axis=1), 1)
    again.embed(["gamma"], "RETRIEVAL_QUERY")    # different task type is a different vector
    assert again.api_calls == 1


class ScriptedLLM:
    model = "scripted"

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def generate_json(self, prompt, schema, images=None):
        assert schema is LLMAnswer
        self.prompts.append(prompt)
        return LLMAnswer(answer=self.replies.pop(0))


def test_check_citations(retriever):
    hits = retriever.search("deadlock", Filters(subject="DC"), k=3)
    cited, problems = check_citations("- Banker's algorithm asked for 6 marks [DC_2025_09_MID, Q2b]", hits)
    assert cited == ["DC_2025_09_MID-Q2b"] and problems == []
    _, problems = check_citations("- Banker's algorithm asked for 6 marks [DC_2024_11_END, Q1]", hits)
    assert any("not one of the sources" in p for p in problems)
    _, problems = check_citations("- Banker's algorithm is asked often in sessionals.", hits)
    assert any("no citation" in p for p in problems)


def test_answer_retries_after_invented_citation(retriever):
    llm = ScriptedLLM(["- Asked for 6 marks [DC_2023_11_END, Q9]", "- Banker's algorithm, 6 marks [DC_2025_09_MID, Q2b]"])
    res = answer("deadlock", retriever, llm, Filters(subject="DC"), k=3)
    assert res.grounded and res.attempts == 2 and res.citations == ["DC_2025_09_MID-Q2b"]
    assert "DC_2023_11_END, Q9] is not one of the sources" in llm.prompts[1]


def test_answer_refuses_when_citations_never_check_out(retriever):
    llm = ScriptedLLM(["- Deadlock is asked every year."] * 3)
    res = answer("deadlock", retriever, llm, Filters(subject="DC"), k=3, max_attempts=3)
    assert not res.grounded and res.attempts == 3 and res.hits and "Could not" in res.text


def test_answer_no_match(retriever):
    res = answer("deadlock", retriever, ScriptedLLM(["NO_MATCH"]), Filters(subject="DC"), k=3)
    assert not res.grounded and res.attempts == 1


def test_every_sentence_needs_a_citation(retriever):
    hits = retriever.search("deadlock", Filters(subject="DC"), k=3)
    one_line = "- Banker's algorithm is asked for 6 marks [DC_2025_09_MID, Q2b]. - It is also asked every single year in the end-sem."
    _, problems = check_citations(one_line, hits)
    assert problems == ["no citation on: It is also asked every single year in the end-sem."]
    two_sentences = "- Banker's algorithm is asked for 6 marks [DC_2025_09_MID, Q2b]. It also appears in every end-sem paper."
    assert check_citations(two_sentences, hits)[1]


def test_answer_text_is_one_bullet_per_line(retriever):
    llm = ScriptedLLM(["- Banker's algorithm, 6 marks [DC_2025_09_MID, Q2b]. - Also asked at BITM [BITM-DC_2019_11_END, Q4a]."])
    res = answer("deadlock", retriever, llm, Filters(subject="DC"), k=3)
    assert res.grounded and res.text.count("\n") == 1 and len(res.citations) == 2


def test_no_ddu_note_comes_from_code(retriever):
    llm = ScriptedLLM(["- Banker's algorithm, 5 marks [BITM-DC_2019_11_END, Q4a]."])
    res = answer("deadlock", retriever, llm, Filters(subject="DC", source="external"), k=3)
    assert res.grounded and res.text.startswith("No DDU paper")


def test_aktu_watermark_removed():
    from exam_prep.extract import clean_text_layer
    line = "Describe the Path-Pushing dead lock          QP23DP1_290     117.55.242.132               detection algorithm."
    assert "117.55" not in clean_text_layer(line) and "QP23" not in clean_text_layer(line)
    assert "192.168.1.1" in clean_text_layer("Ping the host 192.168.1.1 and explain the reply.")
    c = chunk_from_row({"question_id": "X-Q1", "paper_id": "X", "subject": "DC", "year": 2023, "exam_type": "end",
                        "text": "Describe the Path-Pushing deadlock QP23DP1_290 117.55.242.132 detection algorithm."})
    assert c.text == "Describe the Path-Pushing deadlock detection algorithm."


def test_embedder_factory_rejects_unknown_provider():
    from exam_prep.embed import make_embedder
    with pytest.raises(ValueError):
        make_embedder("openai")


def test_cache_files_are_separate_per_model(workspace):
    a, b = FakeEmbedder(), FakeEmbedder()
    b.model = "other-model"
    from exam_prep.embed import EmbeddingCache
    assert EmbeddingCache("fake-embed", 64).path != EmbeddingCache("BAAI/bge-base-en-v1.5", 768).path
    assert "/" not in EmbeddingCache("BAAI/bge-base-en-v1.5", 768).path.name
