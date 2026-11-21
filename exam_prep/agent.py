"""Explicit-operation router (Phase E slice) over four tools: search_papers, predict_topics, generate_question, grade_answer.

The caller names the operation; no language model decides the route. With LangGraph installed the graph is

    START -> route (checks the operation, validates its arguments) -> one tool node -> END

Without LangGraph (requirements-agent.txt), `get_graph()` raises AgentUnavailable and POST /agent answers 503. The same
tools stay reachable through `run_operation` (no graph), which is what /search, /predict, /practice and /grade use.
Every tool returns plain JSON-ready dicts. Errors are IntelligenceError subclasses whose `public` text is ours only.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Callable, Literal, Optional, TypedDict, get_args
from pydantic import BaseModel, Field, ValidationError, model_validator
from . import config
from .intelligence import MAX_MARKS, MAX_RUBRIC_POINTS, IntelligenceError, Practice, RubricPoint, generate_practice, grade_answer, predict_topics
from .retrieval import Chunk, Filters

CODE = r'^[A-Za-z0-9_-]*$'   # subject / source / exam codes: DC, MLDL, DDU, external, BITM, mid
Operation = Literal['search_papers', 'predict_topics', 'generate_question', 'grade_answer']
OPERATIONS: tuple[str, ...] = get_args(Operation)
NEEDS_LLM = frozenset({'generate_question', 'grade_answer'})
SearchMode = Literal['bm25', 'dense', 'hybrid']


class UnknownOperation(IntelligenceError):
    status = 422


class BadArguments(IntelligenceError):
    status = 422


class AgentUnavailable(IntelligenceError):
    status = 503


# ---------------------------------------------------------------- argument models (also the web request bodies)

class SearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    subject: Optional[str] = Field(default=None, max_length=16, pattern=CODE)
    source: Optional[str] = Field(default=None, max_length=16, pattern=CODE)
    year_from: Optional[int] = Field(default=None, ge=1900, le=2100)
    year_to: Optional[int] = Field(default=None, ge=1900, le=2100)
    mode: SearchMode = 'bm25'
    k: int = Field(default=5, ge=1, le=20)

    @model_validator(mode='after')
    def validate_query(self):
        self.query = self.query.strip()
        if not self.query:
            raise ValueError('Enter a question or topic.')
        if self.year_from and self.year_to and self.year_from > self.year_to:
            raise ValueError('The start year must not exceed the end year.')
        return self

    def filters(self):
        return Filters(subject=self.subject or None, source=self.source or None,
                       year_from=self.year_from, year_to=self.year_to)


class PredictArgs(BaseModel):
    subject: str = Field(min_length=1, max_length=16, pattern=CODE)
    method: Literal['frequency', 'trained'] = 'frequency'
    exam_type: Optional[str] = Field(default=None, max_length=16, pattern=CODE)
    top_n: int = Field(default=20, ge=1, le=200)


class PracticeArgs(BaseModel):
    topic: str = Field(min_length=1, max_length=300)
    subject: str = Field(min_length=1, max_length=16, pattern=CODE)
    marks: Optional[float] = Field(default=None, gt=0, le=MAX_MARKS)
    source: Optional[str] = Field(default=None, max_length=16, pattern=CODE)
    mode: SearchMode = 'bm25'
    k: int = Field(default=5, ge=1, le=10)
    variant: int = Field(default=0, ge=0, le=1000)

    @model_validator(mode='after')
    def validate_topic(self):
        self.topic = self.topic.strip()
        if not self.topic:
            raise ValueError('Enter a topic.')
        return self


class GradeArgs(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    rubric: list[RubricPoint] = Field(min_length=1, max_length=MAX_RUBRIC_POINTS)
    answer: str = Field(min_length=1, max_length=8000)

    @model_validator(mode='after')
    def validate_text(self):
        self.question, self.answer = self.question.strip(), self.answer.strip()
        if not self.question or not self.answer:
            raise ValueError('Enter both the question and your answer.')
        if any(not p.criterion.strip() for p in self.rubric):
            raise ValueError('Every rubric point needs a criterion.')
        return self


ARG_MODELS: dict[str, type[BaseModel]] = {'search_papers': SearchArgs, 'predict_topics': PredictArgs,
                                          'generate_question': PracticeArgs, 'grade_answer': GradeArgs}


def unknown_operation() -> UnknownOperation:
    return UnknownOperation(f"Unknown operation. Use one of: {', '.join(OPERATIONS)}.")


def parse_args(operation: str, args: Any) -> BaseModel:
    """Validated arguments, or a 422-class error that names fields only (never echoes submitted values)."""
    model = ARG_MODELS.get(operation)
    if model is None:
        raise unknown_operation()
    if isinstance(args, model):
        return args
    try:
        return model.model_validate(args or {})
    except ValidationError as exc:
        fields = sorted({'.'.join(map(str, e['loc'])) or 'args' for e in exc.errors()})
        raise BadArguments(f"Invalid arguments for {operation}: {', '.join(fields[:6])}.",
                           [e['msg'] for e in exc.errors()]) from None


# ---------------------------------------------------------------- tools

@dataclass
class Services:
    retriever_for: Callable[[str], Any]   # mode -> object with .search(query, filters, k, mode)
    chunks: Callable[[], list[Chunk]]     # the whole question bank (DDU + external; predict_topics filters to DDU)
    llm: Callable[[], Any]                # called only by the LLM tools, first, so a missing key fails before retrieval


def serialise_hits(hits) -> list[dict]:
    return [{**h.chunk.model_dump(exclude={'index_text'}), 'citation': h.chunk.citation, 'score': h.score,
             'bm25_rank': h.bm25_rank, 'dense_rank': h.dense_rank} for h in hits]


def practice_dict(p: Practice) -> dict:
    return {'subject': p.subject, 'query': p.query, 'question': p.question, 'marks': p.marks, 'topic_ids': p.topic_ids,
            'topic_names': p.topic_names, 'rubric': [r.model_dump() for r in p.rubric],
            'references': serialise_hits(p.references), 'attempts': p.attempts, 'prompt_version': p.prompt_version,
            'model': p.model, 'ai_generated': True, 'notice': p.notice}


def tool_search_papers(a: SearchArgs, s: Services) -> dict:
    hits = s.retriever_for(a.mode).search(a.query, a.filters(), a.k, a.mode)
    return {'query': a.query, 'mode': a.mode, 'hits': serialise_hits(hits)}


def tool_predict_topics(a: PredictArgs, s: Services) -> dict:
    return asdict(predict_topics(s.chunks(), a.subject, a.method, a.exam_type, a.top_n))


def tool_generate_question(a: PracticeArgs, s: Services) -> dict:
    if config.OFFLINE:
        from .offline import generate_practice_offline
        return generate_practice_offline(a.topic, s.retriever_for('bm25'), a.subject, marks=a.marks, source=a.source,
                                         k=a.k, variant=a.variant)
    llm = s.llm()
    return practice_dict(generate_practice(a.topic, s.retriever_for(a.mode), llm, a.subject, marks=a.marks,
                                           source=a.source, k=a.k, mode=a.mode, variant=a.variant))


def tool_grade_answer(a: GradeArgs, s: Services) -> dict:
    if config.OFFLINE:
        from .offline import grade_keywords
        return grade_keywords(a.rubric, a.answer)
    return asdict(grade_answer(a.question, a.rubric, a.answer, s.llm()))


TOOLS: dict[str, Callable[[BaseModel, Services], dict]] = {
    'search_papers': tool_search_papers, 'predict_topics': tool_predict_topics,
    'generate_question': tool_generate_question, 'grade_answer': tool_grade_answer}


def run_operation(operation: str, args: Any, services: Services) -> dict:
    """Direct dispatch without LangGraph."""
    parsed = parse_args(operation, args)
    return TOOLS[operation](parsed, services)


# ---------------------------------------------------------------- LangGraph

class AgentState(TypedDict, total=False):
    operation: str
    args: Any
    services: Any
    result: dict


def route_node(state: AgentState) -> dict:
    op = state.get('operation')
    if op not in TOOLS:
        raise unknown_operation()
    return {'args': parse_args(op, state.get('args'))}


def pick_tool(state: AgentState) -> str:
    return state['operation']


def tool_node(name: str):
    def node(state: AgentState) -> dict:
        return {'result': TOOLS[name](state['args'], state['services'])}   # looked up per call, so tests can swap tools
    node.__name__ = f'{name}_node'
    return node


def build_graph(lib=None):
    """`lib` is a module with StateGraph, START and END (langgraph.graph by default; tests pass a stand-in)."""
    if lib is None:
        try:
            import langgraph.graph as lib
        except ImportError:
            raise AgentUnavailable('The agent router needs LangGraph (pip install -r requirements-agent.txt). '
                                   '/search, /predict, /practice and /grade still work.') from None
    g = lib.StateGraph(AgentState)
    g.add_node('route', route_node)
    for name in OPERATIONS:
        g.add_node(name, tool_node(name))
        g.add_edge(name, lib.END)
    g.add_edge(lib.START, 'route')
    g.add_conditional_edges('route', pick_tool, {name: name for name in OPERATIONS})
    return g.compile()


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()   # lru_cache does not cache exceptions: a missing LangGraph is re-checked on each call


def run_agent(operation: str, args: Any, services: Services, graph=None) -> dict:
    graph = graph or get_graph()
    out = graph.invoke({'operation': operation, 'args': args, 'services': services})
    return {'operation': operation, 'result': out['result']}
