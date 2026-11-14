"""Local search UI and API for the question bank.

Run from exam-prep-engine/:  uvicorn exam_prep.web:app --port 8000   (uvicorn binds to 127.0.0.1 by default)

    GET  /         search page (web.html)
    GET  /health   cheap status: never loads the question bank, an embedding model or an API client
    POST /search   retrieval only; mode=bm25 is fully offline, dense/hybrid load the local bge model (fastembed)
    POST /ask      retrieval + Gemini answer whose citations are checked in code (answer.py)
    POST /predict  DDU topic history (counts, not a forecast; a trained predictor is refused with 409)
    POST /practice AI-written practice question + rubric, references and marks checked in code
    POST /grade    AI rubric grading, scores bounded per point; not validated against human graders
    POST /agent    {"operation": ..., "args": {...}} through the LangGraph router; 503 without LangGraph

Failures map to short fixed messages. Exception text (which can carry API responses) is never returned or logged:
the log gets the exception type and the file:line it came from.
"""
import logging, traceback
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from threading import Lock
from fastapi import FastAPI, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from . import agent, config
from .agent import GradeArgs, Operation, PracticeArgs, PredictArgs, SearchArgs as SearchRequest, serialise_hits
from .answer import answer
from .embed import make_embedder
from .intelligence import IntelligenceError
from .llm import GeminiLLM, LLMError
from .retrieval import build_retriever

log = logging.getLogger(__name__)
build_lock = Lock()    # lru_cache alone can build the same retriever twice under concurrent first requests
search_lock = Lock()   # retrieval: the embedding cache file is appended to and is not thread-safe
llm_lock = Lock()      # one Gemini call chain at a time, so the free-tier RPM limiter holds; /search never waits on it


class AgentRequest(BaseModel):
    operation: Operation
    args: dict = Field(default_factory=dict)


@lru_cache(maxsize=2)
def get_retriever(dense: bool):
    # Keyword mode works without credentials, model downloads or network calls.
    return build_retriever(make_embedder('local') if dense else None, backend='memory')


def retriever_for(mode: str):
    with build_lock:
        return get_retriever(mode != 'bm25')


@lru_cache(maxsize=1)
def get_llm():
    return GeminiLLM()


class SerialisedRetriever:
    """Takes search_lock only around retrieval, so a slow Gemini answer does not hold up other searches."""

    def __init__(self, retriever):
        self.retriever = retriever

    def search(self, *args, **kwargs):
        with search_lock:
            return self.retriever.search(*args, **kwargs)


def all_chunks():
    retriever = retriever_for('bm25')   # the keyword retriever holds the whole bank without loading an embedding model
    with search_lock:
        return list(retriever.backend.chunks.values())


def services() -> agent.Services:
    # Lambdas look the module functions up per call, so tests can monkeypatch get_llm / retriever_for.
    return agent.Services(retriever_for=lambda mode: SerialisedRetriever(retriever_for(mode)),
                          chunks=lambda: all_chunks(), llm=lambda: get_llm())


def to_http_error(endpoint: str, exc: Exception) -> HTTPException:
    frame = traceback.extract_tb(exc.__traceback__)[-1] if exc.__traceback__ else None
    where = f'{Path(frame.filename).name}:{frame.lineno}' if frame else '?'
    log.warning('%s failed: %s at %s', endpoint, type(exc).__name__, where)
    if isinstance(exc, IntelligenceError):   # .public is written by our code, never model or API text
        return HTTPException(exc.status, exc.public)
    if isinstance(exc, LLMError):
        return HTTPException(503, 'The answer service is unavailable (API key, quota or network). Keyword search still works.')
    if isinstance(exc, OSError) and not config.EXPORT_FILE.exists():
        return HTTPException(503, 'The question bank is missing. Run "python -m exam_prep export" first.')
    if isinstance(exc, ImportError):
        return HTTPException(503, 'A required package is not installed (see requirements-web.txt). Keyword search may still work.')
    if isinstance(exc, OSError):
        return HTTPException(503, 'Local data or model files could not be read. Keyword search may still work.')
    return HTTPException(500, 'Unexpected error; see the server log.')


app = FastAPI(title='Exam-Prep Engine', version='0.1.0')
app.add_middleware(TrustedHostMiddleware, allowed_hosts=config.WEB_ALLOWED_HOSTS)


@app.get('/', response_class=HTMLResponse)
def home():
    return Path(__file__).with_name('web.html').read_text(encoding='utf-8')


@app.get('/health')
def health():
    return {'status': 'ok', 'scope': 'local',
            'question_bank': config.EXPORT_FILE.exists(),
            'data_loaded': bool(get_retriever.cache_info().currsize),
            'answers_configured': bool(config.GEMINI_API_KEY)}


@app.post('/search')
def search(request: SearchRequest):
    try:
        retriever = SerialisedRetriever(retriever_for(request.mode))
        hits = retriever.search(request.query, request.filters(), request.k, request.mode)
    except Exception as exc:
        raise to_http_error('search', exc) from None
    return {'query': request.query, 'mode': request.mode, 'hits': serialise_hits(hits)}


@app.post('/ask')
def ask(request: SearchRequest):
    try:
        with llm_lock:
            llm = get_llm()   # fail fast on a missing key before loading the question bank
            result = answer(request.query, SerialisedRetriever(retriever_for(request.mode)), llm,
                            request.filters(), k=request.k, mode=request.mode)
    except Exception as exc:
        raise to_http_error('ask', exc) from None
    return {'query': request.query, 'mode': request.mode, 'answer': result.text, 'grounded': result.grounded,
            'citations': result.citations, 'attempts': result.attempts, 'prompt_version': result.prompt_version,
            'hits': serialise_hits(result.hits)}


def call_tool(endpoint: str, operation: str, args) -> dict:
    try:
        with llm_lock if operation in agent.NEEDS_LLM else nullcontext():
            return agent.run_operation(operation, args, services())
    except Exception as exc:
        raise to_http_error(endpoint, exc) from None


@app.post('/predict')
def predict_route(request: PredictArgs):
    return call_tool('predict', 'predict_topics', request)


@app.post('/practice')
def practice_route(request: PracticeArgs):
    return call_tool('practice', 'generate_question', request)


@app.post('/grade')
def grade_route(request: GradeArgs):
    return call_tool('grade', 'grade_answer', request)


@app.post('/agent')
def agent_route(request: AgentRequest):
    try:
        args = agent.parse_args(request.operation, request.args)   # 422 before any work
        graph = agent.get_graph()                                  # 503 without LangGraph
        with llm_lock if request.operation in agent.NEEDS_LLM else nullcontext():
            return agent.run_agent(request.operation, args, services(), graph)
    except Exception as exc:
        raise to_http_error('agent', exc) from None
