"""Local API/UI tests (exam_prep/web.py). Real retrieval over a synthetic question bank in a temp folder; the
embedding model and Gemini are replaced by fakes, so there is no network, no API key and no model download."""
import hashlib, json, logging
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")   # TestClient needs it; both come from requirements-web.txt

from fastapi.testclient import TestClient
from exam_prep import agent, config, web
from exam_prep.answer import LLMAnswer
from exam_prep.intelligence import LLMGrade, LLMPointScore, LLMPractice, LLMRubricPoint
from exam_prep.embed import CachedEmbedder
from exam_prep.llm import LLMError
from exam_prep.retrieval import tokenize

SECRET = "AIzaSy-TEST-SECRET-do-not-leak-0123456789"

ROWS = [   # question_id, paper_id, subject, source, year, marks, text, topics
    ("DC_2025_09_MID-Q2b", "DC_2025_09_MID", "DC", "DDU", 2025, 6, "Explain the Banker's algorithm for deadlock avoidance.", ["Deadlock"]),
    ("DC_2025_10_MID-Q3a-OR", "DC_2025_10_MID", "DC", "DDU", 2025, 6, "Write a Dockerfile for a Python web app and build the image.", ["Dockerfile, Docker Images and Registry"]),
    ("DC_2025_07_MID-Q1a", "DC_2025_07_MID", "DC", "DDU", 2025, 2, "Define container orchestration.", ["Kubernetes"]),
    ("BITM-DC_2019_11_END-Q4a", "BITM-DC_2019_11_END", "DC", "BITM", 2019, 5, "Explain the Banker's algorithm for deadlock avoidance.", ["Deadlock"]),
    ("AI_2025_07_MID-Q2a", "AI_2025_07_MID", "AI", "DDU", 2025, 6, "Apply A* search to the graph and show the open list.", ["Informed search"]),
    ("WT_2025_09_MID-Q1a", "WT_2025_09_MID", "WT", "DDU", 2025, None, "Where is the connection string stored in web.config?", ["Configuration"]),
]


def write_export():
    config.EXPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with config.EXPORT_FILE.open("w", encoding="utf-8") as f:
        for qid, pid, subj, src, year, marks, text, topics in ROWS:
            f.write(json.dumps({"question_id": qid, "paper_id": pid, "subject": subj, "source": src, "year": year, "month": 9,
                                "exam_type": "mid", "marks": marks, "text": text, "topics": topics,
                                "topic_ids": [TOPIC_IDS[t] for t in topics]}) + "\n")


TOPIC_IDS = {"Deadlock": "DC-U3-T2", "Dockerfile, Docker Images and Registry": "DC-U2-T1", "Kubernetes": "DC-U4-T1",
             "Informed search": "AI-U2-T1", "Configuration": "WT-U1-T1"}


class FakeEmbedder(CachedEmbedder):
    """Hashed bag of stemmed words, standing in for the local bge model."""

    def __init__(self):
        super().__init__("fake-embed", 64)

    def _embed_batch(self, texts, task):
        out = []
        for t in texts:
            v = np.zeros(self.dim)
            for w in tokenize(t, query=True):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
            out.append(v.tolist())
        return out


class ScriptedLLM:
    model = "scripted"

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def generate_json(self, prompt, schema, images=None):
        assert schema is LLMAnswer
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMAnswer(answer=reply)


def forbidden(what):
    def fail(*args, **kwargs):
        raise AssertionError(f"{what} must not be called here")
    return fail


@pytest.fixture
def built(monkeypatch):
    """Wraps the real build_retriever: records each build and pins the ranking settings a local .env could change."""
    real_build, calls = web.build_retriever, []

    def pinned(embedder, backend=None, **kwargs):
        calls.append({"embedder": embedder, "backend": backend})
        r = real_build(embedder, backend=backend, **kwargs)
        r.external_weight, r.rrf_k, r.candidates = 0.95, 60, 50
        return r
    monkeypatch.setattr(web, "build_retriever", pinned)
    return calls


@pytest.fixture
def client(workspace, monkeypatch, built):
    write_export()
    cached_retriever, cached_llm = web.get_retriever, web.get_llm
    cached_retriever.cache_clear()
    cached_llm.cache_clear()
    monkeypatch.setattr(web, "make_embedder", forbidden("the embedding model"))   # tests that need it re-patch it
    with TestClient(web.app, base_url="http://127.0.0.1") as c:
        yield c
    cached_retriever.cache_clear()
    cached_llm.cache_clear()


def ids(response):
    return [h["chunk_id"] for h in response.json()["hits"]]


# ---------------------------------------------------------------- page + health

def test_home_page_lists_every_subject(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    for subject in ("AI", "DC", "MLDL", "WT"):
        assert f"<option>{subject}</option>" in r.text
    assert r.text.count("<option") == r.text.count("</option>")   # the AI option used to be missing its opening tag


def test_health_loads_nothing(client, monkeypatch):
    monkeypatch.setattr(web, "build_retriever", forbidden("build_retriever"))
    monkeypatch.setattr(web, "GeminiLLM", forbidden("GeminiLLM"))
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "scope": "local", "question_bank": True, "data_loaded": False, "answers_configured": False,
                           "offline_demo": False, "primary_source": "DDU"}
    monkeypatch.setattr(config, "GEMINI_API_KEY", SECRET)
    r = client.get("/health")
    assert r.json()["answers_configured"] is True and SECRET not in r.text   # a yes/no, never the key


def test_health_reports_loaded_data_after_a_search(client):
    assert client.post("/search", json={"query": "deadlock"}).status_code == 200
    assert client.get("/health").json()["data_loaded"] is True


def test_rejects_foreign_host_header(workspace):
    with TestClient(web.app, base_url="http://attacker.example") as c:
        assert c.get("/health").status_code == 400


# ---------------------------------------------------------------- search: real retrieval over the synthetic bank

def test_keyword_search_is_offline_and_prefers_ddu(client, built):
    r = client.post("/search", json={"query": "  important questions on deadlock  ", "subject": "DC"})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "important questions on deadlock" and body["mode"] == "bm25"
    assert ids(r)[:2] == ["DC_2025_09_MID-Q2b", "BITM-DC_2019_11_END-Q4a"]   # identical text: DDU first
    top = body["hits"][0]
    assert top["citation"] == "[DC_2025_09_MID, Q2b]" and top["marks"] == 6 and top["source"] == "DDU"
    assert top["bm25_rank"] == 1 and top["dense_rank"] is None and "index_text" not in top
    assert len(built) == 1 and built[0]["embedder"] is None and built[0]["backend"] == "memory"   # no embedding model loaded


@pytest.mark.parametrize("body, expected", [
    ({"source": "external"}, {"BITM-DC_2019_11_END-Q4a"}),
    ({"subject": "dc", "year_to": 2020}, {"BITM-DC_2019_11_END-Q4a"}),
    ({"subject": "AI"}, {"AI_2025_07_MID-Q2a"}),
    ({"source": "DDU", "subject": "DC", "year_from": 2025}, {"DC_2025_09_MID-Q2b", "DC_2025_10_MID-Q3a-OR"}),
])
def test_search_filters(client, body, expected):
    r = client.post("/search", json={"query": "banker deadlock dockerfile search", "k": 10, **body})
    assert r.status_code == 200 and set(ids(r)) == expected


def test_k_limits_hits_and_missing_marks_are_null(client):
    assert len(ids(client.post("/search", json={"query": "banker deadlock dockerfile", "k": 1}))) == 1
    hit = client.post("/search", json={"query": "connection string web.config", "subject": "WT"}).json()["hits"][0]
    assert hit["chunk_id"] == "WT_2025_09_MID-Q1a" and hit["marks"] is None


def test_no_match_returns_empty_list(client):
    r = client.post("/search", json={"query": "quantum chromodynamics", "subject": "DC"})
    assert r.status_code == 200 and r.json()["hits"] == []


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_dense_modes_use_the_local_embedder_once(client, monkeypatch, built, mode):
    providers = []

    def make(provider=None):
        providers.append(provider)
        return FakeEmbedder()
    monkeypatch.setattr(web, "make_embedder", make)
    for _ in range(2):
        r = client.post("/search", json={"query": "important questions on deadlock", "subject": "DC", "mode": mode})
        assert r.status_code == 200 and ids(r)[0] == "DC_2025_09_MID-Q2b"
        assert r.json()["hits"][0]["dense_rank"] == 1
    assert providers == ["local"] and len(built) == 1   # built once, then cached; never the Gemini embedder


# ---------------------------------------------------------------- validation happens before any work

@pytest.mark.parametrize("body", [
    {},
    {"query": ""},
    {"query": "   "},
    {"query": "x" * 1001},
    {"query": "deadlock", "k": 0},
    {"query": "deadlock", "k": 21},
    {"query": "deadlock", "mode": "magic"},
    {"query": "deadlock", "subject": "DC' OR 1=1"},
    {"query": "deadlock", "source": "x" * 17},
    {"query": "deadlock", "year_from": 1800},
    {"query": "deadlock", "year_from": 2025, "year_to": 2020},
])
@pytest.mark.parametrize("endpoint", ["/search", "/ask"])
def test_invalid_requests_are_rejected_without_work(client, monkeypatch, endpoint, body):
    monkeypatch.setattr(web, "build_retriever", forbidden("build_retriever"))
    monkeypatch.setattr(web, "get_llm", forbidden("get_llm"))
    assert client.post(endpoint, json=body).status_code == 422


def test_validation_messages_are_readable(client):
    msg = client.post("/search", json={"query": "deadlock", "year_from": 2025, "year_to": 2020}).json()["detail"][0]["msg"]
    assert "start year must not exceed the end year" in msg
    msg = client.post("/search", json={"query": "  "}).json()["detail"][0]["msg"]
    assert "Enter a question or topic" in msg


def test_non_json_body_is_rejected(client, monkeypatch):
    monkeypatch.setattr(web, "get_llm", forbidden("get_llm"))
    r = client.post("/ask", content=b'{"query": "deadlock"}', headers={"Content-Type": "text/plain"})
    assert r.status_code == 422   # a cross-site "simple" POST cannot trigger a Gemini call


# ---------------------------------------------------------------- cited answers

def test_ask_returns_checked_citations(client, monkeypatch):
    llm = ScriptedLLM(["- Banker's algorithm for deadlock avoidance, asked for 6 marks [DC_2025_09_MID, Q2b]."])
    monkeypatch.setattr(web, "get_llm", lambda: llm)
    r = client.post("/ask", json={"query": "deadlock", "subject": "DC", "source": "DDU"})
    assert r.status_code == 200
    body = r.json()
    assert body["grounded"] is True and body["attempts"] == 1
    assert body["citations"] == ["DC_2025_09_MID-Q2b"] and "[DC_2025_09_MID, Q2b]" in body["answer"]
    assert body["prompt_version"].startswith("answer_v")
    assert ids(r)[0] == "DC_2025_09_MID-Q2b" and all(h["source"] == "DDU" for h in body["hits"])
    assert "[DC_2025_09_MID, Q2b]" in llm.prompts[0] and "Banker's algorithm" in llm.prompts[0]   # retrieved text reached the model


def test_ask_retries_an_invented_citation(client, monkeypatch):
    llm = ScriptedLLM(["- Asked for 6 marks [DC_2023_11_END, Q9].", "- Banker's algorithm, 6 marks [DC_2025_09_MID, Q2b]."])
    monkeypatch.setattr(web, "get_llm", lambda: llm)
    body = client.post("/ask", json={"query": "deadlock", "subject": "DC"}).json()
    assert body["grounded"] is True and body["attempts"] == 2 and body["citations"] == ["DC_2025_09_MID-Q2b"]
    assert "DC_2023_11_END" not in body["answer"]


def test_ask_refuses_unverifiable_answer_but_keeps_sources(client, monkeypatch):
    llm = ScriptedLLM(["- Deadlock is asked every single year."] * config.ANSWER_MAX_ATTEMPTS)
    monkeypatch.setattr(web, "get_llm", lambda: llm)
    body = client.post("/ask", json={"query": "deadlock", "subject": "DC"}).json()
    assert body["grounded"] is False and body["citations"] == [] and body["answer"].startswith("Could not")
    assert "every single year" not in body["answer"] and ids_from(body)[0] == "DC_2025_09_MID-Q2b"


def test_ask_with_no_hits_skips_the_model(client, monkeypatch):
    llm = ScriptedLLM([])
    monkeypatch.setattr(web, "get_llm", lambda: llm)
    body = client.post("/ask", json={"query": "quantum chromodynamics", "subject": "DC"}).json()
    assert body["grounded"] is False and body["hits"] == [] and llm.prompts == []


def ids_from(body):
    return [h["chunk_id"] for h in body["hits"]]


# ---------------------------------------------------------------- failures: clear status, no leaked secrets or paths

def test_ask_without_api_key_fails_fast(client, monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="exam_prep.web")
    monkeypatch.setattr(web, "build_retriever", forbidden("build_retriever"))   # real get_llm -> GeminiLLM raises first
    r = client.post("/ask", json={"query": "deadlock"})
    assert r.status_code == 503 and "Keyword search still works" in r.json()["detail"]
    assert "GEMINI_API_KEY" not in r.text and "LLMError" in caplog.text


def test_llm_error_does_not_leak_the_key(client, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(config, "GEMINI_API_KEY", SECRET)
    llm = ScriptedLLM([LLMError(f"Gemini API error 400: API key {SECRET} not valid")])
    monkeypatch.setattr(web, "get_llm", lambda: llm)
    r = client.post("/ask", json={"query": "deadlock", "subject": "DC"})
    assert r.status_code == 503 and r.json()["detail"].startswith("The answer service is unavailable")
    assert SECRET not in r.text and SECRET not in caplog.text
    assert "LLMError" in caplog.text   # the log still says what failed and where


def test_unexpected_error_is_a_generic_500(client, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    llm = ScriptedLLM([KeyError(f"token={SECRET}")])
    monkeypatch.setattr(web, "get_llm", lambda: llm)
    r = client.post("/ask", json={"query": "deadlock", "subject": "DC"})
    assert r.status_code == 500 and r.json() == {"detail": "Unexpected error; see the server log."}
    assert SECRET not in r.text and SECRET not in caplog.text and "KeyError" in caplog.text


def test_missing_question_bank_is_503_without_paths(client, workspace):
    config.EXPORT_FILE.unlink()
    r = client.post("/search", json={"query": "deadlock"})
    assert r.status_code == 503 and "python -m exam_prep export" in r.json()["detail"]
    assert str(workspace) not in r.text and "questions_tagged" not in r.text


def test_missing_embedding_package_is_503(client, monkeypatch):
    def no_fastembed(provider=None):
        raise ModuleNotFoundError("No module named 'fastembed'")
    monkeypatch.setattr(web, "make_embedder", no_fastembed)
    r = client.post("/search", json={"query": "deadlock", "mode": "hybrid"})
    assert r.status_code == 503 and "requirements-web.txt" in r.json()["detail"]
    assert client.post("/search", json={"query": "deadlock"}).status_code == 200   # keyword search still works


# ---------------------------------------------------------------- Phase D slice: /predict /practice /grade /agent

class SchemaLLM:
    """Returns scripted pydantic replies of whatever schema is asked for."""
    model = "scripted"

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def generate_json(self, prompt, schema, images=None):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        assert isinstance(reply, schema)
        return reply


def practice_reply(refs=("DC_2025_09_MID-Q2b",), marks=6.0, text="A system has 12 tape drives and three processes. Is the state safe?"):
    return LLMPractice(question=text, marks=marks, topic_ids=["DC-U3-T2"], reference_ids=list(refs),
                       rubric=[LLMRubricPoint(criterion="Need matrix", marks=2), LLMRubricPoint(criterion="Safe sequence", marks=4)])


RUBRIC = [{"criterion": "Need matrix", "marks": 2}, {"criterion": "Safe sequence", "marks": 4}]
GRADE_BODY = {"question": "Is the state safe?", "rubric": RUBRIC, "answer": "Need = Max - Allocation; P2, P1, P3 is safe."}


def grade_reply(s1, s2):
    return LLMGrade(scores=[LLMPointScore(point=1, score=s1, feedback="ok"), LLMPointScore(point=2, score=s2, feedback="ok")],
                    overall_feedback="Show each step.")


def test_home_page_has_practice_with_ai_notices(client):
    text = client.get("/").text
    assert 'id="practice"' in text and 'id="grade"' in text and "/predict" in text
    assert "not validated against human graders" in text and "not a prediction" in text


def test_predict_counts_ddu_history_only(client, monkeypatch):
    monkeypatch.setattr(web, "get_llm", forbidden("get_llm"))
    r = client.post("/predict", json={"subject": "dc"})
    assert r.status_code == 200
    body = r.json()
    assert (body["subject"], body["ddu_papers"], body["ddu_questions"], body["excluded_non_ddu"]) == ("DC", 3, 3, 1)
    deadlock = next(t for t in body["topics"] if t["topic_id"] == "DC-U3-T2")
    assert deadlock["name"] == "Deadlock" and deadlock["questions"] == 1 and deadlock["years"] == [2025]   # BITM excluded
    assert body["status"] == "insufficient_history" and body["is_prediction"] is False and body["validated"] is False


@pytest.mark.parametrize("body, status, text", [
    ({"subject": "DC", "method": "trained"}, 409, "at least 2 different years"),
    ({"subject": "OS"}, 404, "No DDU questions for OS"),
])
def test_predict_refusals_are_mapped(client, body, status, text):
    r = client.post("/predict", json=body)
    assert r.status_code == status and text in r.json()["detail"]


@pytest.mark.parametrize("endpoint, body", [
    ("/predict", {}),
    ("/predict", {"subject": "DC", "method": "magic"}),
    ("/predict", {"subject": "DC", "top_n": 0}),
    ("/practice", {"topic": "deadlock"}),
    ("/practice", {"topic": "   ", "subject": "DC"}),
    ("/practice", {"topic": "deadlock", "subject": "DC", "marks": 0}),
    ("/practice", {"topic": "deadlock", "subject": "DC", "k": 11}),
    ("/grade", {**GRADE_BODY, "rubric": []}),
    ("/grade", {**GRADE_BODY, "rubric": [{"criterion": "x", "marks": 0}]}),
    ("/grade", {**GRADE_BODY, "rubric": [{"criterion": "x", "marks": 1}] * 13}),
    ("/grade", {**GRADE_BODY, "rubric": [{"criterion": "  ", "marks": 2}]}),
    ("/grade", {**GRADE_BODY, "answer": "   "}),
    ("/grade", {**GRADE_BODY, "answer": "x" * 8001}),
    ("/agent", {"operation": "delete_everything", "args": {}}),
    ("/agent", {"operation": "predict_topics", "args": "DC"}),
    ("/agent", {"operation": "predict_topics", "args": {"subject": "DC' OR 1=1"}}),
    ("/agent", {"operation": "grade_answer", "args": {"question": "Q?", "answer": "a", "rubric": [{"criterion": "c", "marks": -1}]}}),
])
def test_phase_d_bad_requests_are_422_without_work(client, monkeypatch, endpoint, body):
    monkeypatch.setattr(web, "build_retriever", forbidden("build_retriever"))
    monkeypatch.setattr(web, "get_llm", forbidden("get_llm"))
    monkeypatch.setattr(agent, "get_graph", forbidden("get_graph"))
    assert client.post(endpoint, json=body).status_code == 422


def test_practice_returns_a_checked_ai_question(client, monkeypatch):
    llm = SchemaLLM([practice_reply()])
    monkeypatch.setattr(web, "get_llm", lambda: llm)
    r = client.post("/practice", json={"topic": "deadlock", "subject": "DC", "marks": 6})
    assert r.status_code == 200
    body = r.json()
    assert body["ai_generated"] is True and "language model" in body["notice"] and body["marks"] == 6
    assert [h["citation"] for h in body["references"]] == ["[DC_2025_09_MID, Q2b]"]
    assert body["topic_names"] == {"DC-U3-T2": "Deadlock"} and sum(p["marks"] for p in body["rubric"]) == 6
    assert "DC_2025_09_MID-Q2b" in llm.prompts[0]


def test_practice_with_invented_references_is_refused(client, monkeypatch):
    bad = practice_reply(refs=("DC_2023_11_END-Q9",), text="LEAKED invented question about deadlock detection")
    monkeypatch.setattr(web, "get_llm", lambda: SchemaLLM([bad] * 10))
    r = client.post("/practice", json={"topic": "deadlock", "subject": "DC"})
    assert r.status_code == 502 and r.json()["detail"].startswith("Could not generate a practice question")
    assert "LEAKED" not in r.text and "DC_2023_11_END" not in r.text


def test_practice_without_matches_or_key(client, monkeypatch):
    real_get_llm, llm = web.get_llm, SchemaLLM([])
    monkeypatch.setattr(web, "get_llm", lambda: llm)
    r = client.post("/practice", json={"topic": "quantum chromodynamics", "subject": "DC"})
    assert r.status_code == 404 and llm.prompts == []
    monkeypatch.setattr(web, "get_llm", real_get_llm)   # the workspace fixture set GEMINI_API_KEY to None
    web.get_retriever.cache_clear()
    monkeypatch.setattr(web, "build_retriever", forbidden("build_retriever"))   # real get_llm fails first: no key
    r = client.post("/practice", json={"topic": "deadlock", "subject": "DC"})
    assert r.status_code == 503 and "GEMINI_API_KEY" not in r.text


def test_grade_is_bounded_and_marked_as_ai(client, monkeypatch):
    llm = SchemaLLM([grade_reply(3, 4), grade_reply(1.5, 4)])   # 3 > 2 on point 1 is retried
    monkeypatch.setattr(web, "get_llm", lambda: llm)
    r = client.post("/grade", json=GRADE_BODY)
    assert r.status_code == 200
    body = r.json()
    assert (body["total"], body["max_total"], body["attempts"]) == (5.5, 6, 2)
    assert body["grader"] == "ai" and body["human_validated"] is False and "human graders" in body["notice"]
    assert all(0 <= p["score"] <= p["max_marks"] for p in body["points"])


def test_grade_that_never_validates_is_502(client, monkeypatch):
    monkeypatch.setattr(web, "get_llm", lambda: SchemaLLM([grade_reply(-1, 9)] * 10))
    r = client.post("/grade", json=GRADE_BODY)
    assert r.status_code == 502 and r.json()["detail"].startswith("Could not produce a grade")


def test_agent_without_langgraph_is_503(client, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "langgraph", None)
    monkeypatch.setitem(sys.modules, "langgraph.graph", None)
    agent.get_graph.cache_clear()
    monkeypatch.setattr(web, "get_llm", forbidden("get_llm"))
    r = client.post("/agent", json={"operation": "predict_topics", "args": {"subject": "DC"}})
    agent.get_graph.cache_clear()
    assert r.status_code == 503 and "requirements-agent.txt" in r.json()["detail"]
    assert client.post("/predict", json={"subject": "DC"}).status_code == 200   # the same tool still works directly


def test_agent_dispatches_through_the_graph(client, monkeypatch, fake_graph_lib):
    monkeypatch.setattr(agent, "get_graph", lambda: agent.build_graph(fake_graph_lib))
    monkeypatch.setattr(web, "get_llm", forbidden("get_llm"))
    r = client.post("/agent", json={"operation": "predict_topics", "args": {"subject": "DC", "top_n": 1}})
    assert r.status_code == 200 and r.json()["operation"] == "predict_topics"
    assert len(r.json()["result"]["topics"]) == 1
    r = client.post("/agent", json={"operation": "search_papers", "args": {"query": "deadlock", "subject": "DC"}})
    assert r.status_code == 200 and r.json()["result"]["hits"][0]["chunk_id"] == "DC_2025_09_MID-Q2b"
    llm = SchemaLLM([grade_reply(2, 4)])
    monkeypatch.setattr(web, "get_llm", lambda: llm)
    r = client.post("/agent", json={"operation": "grade_answer", "args": GRADE_BODY})
    assert r.status_code == 200 and r.json()["result"]["total"] == 6
