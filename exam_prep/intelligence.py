"""Phase D slice: topic frequency, practice questions and rubric grading.

Three operations, each explicit about what it is and is not:

    topic_frequency    Counts how often each syllabus topic was asked in DDU papers. Descriptive history only:
                       there is no trained predictor and no time-based backtest. Other universities' papers are
                       excluded, as the project rules require for anything that looks like prediction.
    generate_practice  Gemini writes one new question from retrieved past questions. Reference IDs, topic IDs,
                       marks and the rubric are checked in code; failures are retried with feedback, then refused.
    grade_answer       Gemini scores a student answer per rubric point. Every score is checked against that
                       point's maximum. This is AI grading; it has NOT been validated against human graders.
"""
import re
from collections import defaultdict
from dataclasses import dataclass, field
from string import Template
from typing import Iterable, Literal, Optional
from pydantic import BaseModel, Field, ValidationError
from . import config
from .llm import LLM, load_prompt
from .retrieval import Chunk, Filters, Hit, Mode, Retriever

MIN_YEARS_FOR_BACKTEST = 2       # a time-based backtest needs at least one earlier year to train on and one later year to test
MARKS_TOLERANCE = 0.01
MAX_MARKS = 100.0
MAX_RUBRIC_POINTS = 12
MIN_QUESTION_CHARS = 15

FREQUENCY_METHOD = "historical_frequency"
AI_GRADING_NOTICE = ("AI grading by a language model against the rubric. It has not been validated against human "
                     "graders; treat the scores as practice feedback, not as an exam mark.")
AI_GENERATION_NOTICE = ("Practice question written by a language model from the cited past questions. It is not an "
                        "official question and the rubric is the model's, checked only for structure (IDs and marks).")


# ---------------------------------------------------------------- errors (public messages are composed by our code only)

class IntelligenceError(Exception):
    """`public` is safe to show users: it never contains model output or API error text."""
    status = 422

    def __init__(self, public: str, problems: Optional[list[str]] = None):
        super().__init__(public)
        self.public = public
        self.problems = problems or []


class NotEnoughData(IntelligenceError):
    status = 404


class InsufficientHistory(IntelligenceError):
    status = 409


class PredictorUnavailable(IntelligenceError):
    status = 409


class GenerationFailed(IntelligenceError):
    status = 502


class GradingFailed(IntelligenceError):
    status = 502


# ---------------------------------------------------------------- 1. historical topic frequency (DDU only)

@dataclass
class TopicCount:
    topic_id: str
    name: str
    questions: int          # DDU questions tagged with the topic (OR alternatives count separately)
    papers: int             # DDU papers containing at least one such question
    paper_share: float      # papers / all DDU papers of the subject in scope
    marks_printed: float    # printed marks, split evenly when a question has two topics
    years: list[int]


@dataclass
class FrequencyReport:
    subject: str
    exam_type: Optional[str]
    ddu_papers: int
    ddu_questions: int
    untagged_questions: int
    excluded_non_ddu: int
    years: list[int]
    topics: list[TopicCount]
    status: Literal["insufficient_history", "not_backtested"]
    notice: str
    method: str = FREQUENCY_METHOD
    is_prediction: bool = False   # always False: these are counts of the past
    validated: bool = False       # no backtest has been run


def topic_names(chunks: Iterable[Chunk]) -> dict[str, str]:
    """topic_id -> name. export.py writes `topics` in the same order as `topic_ids` when every ID is in the taxonomy."""
    names: dict[str, str] = {}
    for c in chunks:
        if len(c.topics) == len(c.topic_ids):
            for tid, name in zip(c.topic_ids, c.topics):
                names.setdefault(tid, name)
    return names


def history_status(subject: str, years: list[int]) -> tuple[str, str]:
    if len(years) < MIN_YEARS_FOR_BACKTEST:
        span = ", ".join(map(str, years)) or "none"
        return "insufficient_history", (
            f"Only {len(years)} year of DDU {subject} papers ({span}). That is not enough history to train or backtest "
            f"a topic predictor, so these are counts of what was asked, not a forecast of the next exam.")
    return "not_backtested", (
        f"{len(years)} years of DDU {subject} papers. No trained predictor or time-based backtest exists yet, so these "
        f"are historical counts, not a validated forecast.")


def topic_frequency(chunks: Iterable[Chunk], subject: str, exam_type: Optional[str] = None,
                    top_n: Optional[int] = None) -> FrequencyReport:
    subject = subject.upper()
    exam_type = exam_type.lower() if exam_type else None
    in_scope = [c for c in chunks if c.subject == subject and (exam_type is None or c.exam_type == exam_type)]
    ddu = [c for c in in_scope if c.source == "DDU"]
    if not ddu:
        scope = f" {exam_type}-semester" if exam_type else ""
        raise NotEnoughData(f"No DDU{scope} questions for {subject} in the question bank.")
    papers = {c.paper_id for c in ddu}
    years = sorted({c.year for c in ddu})
    names = topic_names(ddu)
    n_questions: dict[str, int] = defaultdict(int)
    n_papers: dict[str, set] = defaultdict(set)
    marks: dict[str, float] = defaultdict(float)
    t_years: dict[str, set] = defaultdict(set)
    untagged = 0
    for c in ddu:
        ids = list(dict.fromkeys(c.topic_ids))
        if not ids:
            untagged += 1
            continue
        for tid in ids:
            n_questions[tid] += 1
            n_papers[tid].add(c.paper_id)
            t_years[tid].add(c.year)
            if c.marks is not None:
                marks[tid] += c.marks / len(ids)
    counts = [TopicCount(topic_id=t, name=names.get(t, t), questions=n_questions[t], papers=len(n_papers[t]),
                         paper_share=round(len(n_papers[t]) / len(papers), 4), marks_printed=round(marks[t], 2),
                         years=sorted(t_years[t])) for t in n_questions]
    counts.sort(key=lambda t: (-t.papers, -t.questions, -t.marks_printed, t.topic_id))
    status, notice = history_status(subject, years)
    return FrequencyReport(subject=subject, exam_type=exam_type, ddu_papers=len(papers), ddu_questions=len(ddu),
                           untagged_questions=untagged, excluded_non_ddu=len(in_scope) - len(ddu), years=years,
                           topics=counts[:top_n] if top_n else counts, status=status, notice=notice)


def predict_topics(chunks: Iterable[Chunk], subject: str, method: Literal["frequency", "trained"] = "frequency",
                   exam_type: Optional[str] = None, top_n: Optional[int] = None) -> FrequencyReport:
    """`frequency` returns the descriptive counts. `trained` is refused: with one year there is nothing to backtest,
    and with more years no trained, backtested model exists yet, so claiming one would be false."""
    chunks = list(chunks)
    if method == "frequency":
        return topic_frequency(chunks, subject, exam_type, top_n)
    subject = subject.upper()
    years = sorted({c.year for c in chunks if c.subject == subject and c.source == "DDU"})
    if not years:
        raise NotEnoughData(f"No DDU questions for {subject} in the question bank.")
    if len(years) < MIN_YEARS_FOR_BACKTEST:
        raise InsufficientHistory(
            f"A trained topic predictor needs DDU papers from at least {MIN_YEARS_FOR_BACKTEST} different years to "
            f"backtest; {subject} has {len(years)} ({', '.join(map(str, years))}). Use method=frequency for historical counts.")
    raise PredictorUnavailable("No trained topic predictor has been built and backtested yet. Use method=frequency "
                               "for historical counts.")


# ---------------------------------------------------------------- shared rubric model

class RubricPoint(BaseModel):
    criterion: str = Field(min_length=1, max_length=500)
    marks: float = Field(gt=0, le=MAX_MARKS)


def rubric_problems(rubric: list, total: float) -> list[str]:
    problems = []
    if not rubric:
        return ["the rubric is empty"]
    if len(rubric) > MAX_RUBRIC_POINTS:
        problems.append(f"the rubric has {len(rubric)} points; use at most {MAX_RUBRIC_POINTS}")
    for i, p in enumerate(rubric, 1):
        if not p.criterion.strip():
            problems.append(f"rubric point {i} has no criterion")
        if not p.marks > 0:
            problems.append(f"rubric point {i} must carry positive marks (got {p.marks:g})")
    s = sum(p.marks for p in rubric)
    if abs(s - total) > MARKS_TOLERANCE:
        problems.append(f"rubric marks add up to {s:g} but the question is worth {total:g}")
    return problems


# ---------------------------------------------------------------- 2. practice question generation

class LLMRubricPoint(BaseModel):
    criterion: str
    marks: float


class LLMPractice(BaseModel):
    question: str
    marks: float
    topic_ids: list[str]
    reference_ids: list[str]
    rubric: list[LLMRubricPoint]


@dataclass
class Practice:
    subject: str
    query: str
    question: str
    marks: float
    topic_ids: list[str]
    topic_names: dict[str, str]
    references: list[Hit]          # the cited past questions, in citation order
    rubric: list[RubricPoint]
    attempts: int
    prompt_version: str
    model: str
    candidates: list[Hit] = field(default_factory=list)   # everything retrieved and shown to the model
    notice: str = AI_GENERATION_NOTICE


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def format_references(hits: list[Hit]) -> str:
    lines = []
    for h in hits:
        c = h.chunk
        marks = f"{c.marks:g} marks" if c.marks is not None else "marks not printed"
        names = dict(zip(c.topic_ids, c.topics)) if len(c.topics) == len(c.topic_ids) else {}
        topics = "; ".join(f"{t} {names.get(t, '')}".strip() for t in c.topic_ids) or "untagged"
        lines.append(f"{c.chunk_id} | {c.source} {c.year} {c.exam_type} | {marks} | topics: {topics} | {' '.join(c.text.split())}")
    return "\n".join(lines)


def check_practice(out: LLMPractice, hits: list[Hit], marks: Optional[float] = None) -> list[str]:
    """Empty list = valid. Topic IDs must come from the past questions the model cites, so a topic is never invented."""
    problems = []
    by_id = {h.chunk.chunk_id: h.chunk for h in hits}
    if len(out.question.strip()) < MIN_QUESTION_CHARS:
        problems.append("the question is empty or too short")
    if not 0 < out.marks <= MAX_MARKS:
        problems.append(f"marks must be positive and at most {MAX_MARKS:g} (got {out.marks:g})")
    elif marks is not None and abs(out.marks - marks) > MARKS_TOLERANCE:
        problems.append(f"the question must be worth exactly {marks:g} marks (got {out.marks:g})")
    refs = list(dict.fromkeys(out.reference_ids))
    if not refs:
        problems.append("reference_ids is empty; cite at least one past question ID")
    bad_refs = [r for r in refs if r not in by_id]
    if bad_refs:
        problems.append(f"reference_ids not in the list of past questions: {', '.join(bad_refs[:5])}")
    allowed = {t for r in refs if r in by_id for t in by_id[r].topic_ids}
    topics = list(dict.fromkeys(out.topic_ids))
    if allowed and not topics:
        problems.append("topic_ids is empty; use topic IDs of the cited past questions")
    bad_topics = [t for t in topics if t not in allowed]
    if bad_topics:
        problems.append(f"topic_ids not on the cited past questions: {', '.join(bad_topics[:5])}"
                        + ("" if allowed else " (the cited questions are untagged, so topic_ids must be empty)"))
    if 0 < out.marks <= MAX_MARKS:
        problems.extend(rubric_problems(out.rubric, out.marks))
    q = _norm(out.question)
    if q and any(q == _norm(c.text) for c in by_id.values()):
        problems.append("the question copies a past question word for word; write a new one")
    return problems


def generate_practice(topic: str, retriever: Retriever, llm: LLM, subject: str, marks: Optional[float] = None,
                      source: Optional[str] = None, k: int = 5, mode: Mode = "bm25", variant: int = 0,
                      max_attempts: int = config.INTELLIGENCE_MAX_ATTEMPTS) -> Practice:
    subject = subject.upper()
    hits = retriever.search(topic, Filters(subject=subject, source=source), k=k, mode=mode)
    if not hits:
        raise NotEnoughData(f"No past {subject} questions match this topic; try other words or another subject.")
    template, version = load_prompt("practice")
    marks_line = f"The new question must be worth exactly {marks:g} marks." if marks is not None else \
        "Choose marks typical of the cited past questions."
    feedback, problems = "", []
    for attempt in range(1, max_attempts + 1):
        prompt = Template(template).safe_substitute(subject=subject, topic=topic, marks_line=marks_line,
                                                    references=format_references(hits), variant=variant, feedback=feedback)
        out = llm.generate_json(prompt, LLMPractice)
        problems = check_practice(out, hits, marks)
        if not problems:
            by_id = {h.chunk.chunk_id: h for h in hits}
            refs = [by_id[r] for r in dict.fromkeys(out.reference_ids)]
            topic_ids = list(dict.fromkeys(out.topic_ids))
            names = topic_names(h.chunk for h in refs)
            return Practice(subject=subject, query=topic, question=out.question.strip(), marks=out.marks,
                            topic_ids=topic_ids, topic_names={t: names.get(t, t) for t in topic_ids}, references=refs,
                            rubric=[RubricPoint(criterion=p.criterion.strip(), marks=p.marks) for p in out.rubric],
                            attempts=attempt, prompt_version=version, model=getattr(llm, "model", ""), candidates=hits)
        feedback = ("\nYour previous reply failed validation. Fix these problems:\n- " + "\n- ".join(problems[:8])
                    + f"\nAllowed reference IDs: {', '.join(h.chunk.chunk_id for h in hits)}")
    raise GenerationFailed(f"Could not generate a practice question that passed validation in {max_attempts} attempts. "
                           "Try again with different words or marks.", problems)


# ---------------------------------------------------------------- 3. rubric grading (AI, not human-validated)

class LLMPointScore(BaseModel):
    point: int            # 1-based rubric point number
    score: float
    feedback: str


class LLMGrade(BaseModel):
    scores: list[LLMPointScore]
    overall_feedback: str


@dataclass
class PointGrade:
    point: int
    criterion: str
    max_marks: float
    score: float
    feedback: str


@dataclass
class Grade:
    points: list[PointGrade]
    total: float
    max_total: float
    overall_feedback: str
    attempts: int
    prompt_version: str
    model: str
    grader: str = "ai"
    human_validated: bool = False
    notice: str = AI_GRADING_NOTICE


ANSWER_FENCE = re.compile(r"<<<|>>>")


def format_rubric(rubric: list[RubricPoint]) -> str:
    return "\n".join(f"{i}. ({p.marks:g} marks) {' '.join(p.criterion.split())}" for i, p in enumerate(rubric, 1))


def check_grade(out: LLMGrade, rubric: list[RubricPoint]) -> list[str]:
    problems = []
    seen: dict[int, int] = defaultdict(int)
    for s in out.scores:
        seen[s.point] += 1
        if not 1 <= s.point <= len(rubric):
            problems.append(f"point {s.point} is not a rubric point (1-{len(rubric)})")
            continue
        top = rubric[s.point - 1].marks
        if not 0 <= s.score <= top:
            problems.append(f"point {s.point}: score {s.score:g} is outside 0-{top:g}")
        if not s.feedback.strip():
            problems.append(f"point {s.point}: feedback is empty")
    missing = [i for i in range(1, len(rubric) + 1) if not seen.get(i)]
    if missing:
        problems.append(f"no score for rubric point(s) {', '.join(map(str, missing))}")
    dupes = [p for p, n in seen.items() if n > 1]
    if dupes:
        problems.append(f"more than one score for rubric point(s) {', '.join(map(str, sorted(dupes)))}")
    return problems


def grade_answer(question: str, rubric: list[RubricPoint], student_answer: str, llm: LLM,
                 max_attempts: int = config.INTELLIGENCE_MAX_ATTEMPTS) -> Grade:
    try:
        rubric = [p if isinstance(p, RubricPoint) else RubricPoint.model_validate(p) for p in rubric]
    except ValidationError:
        raise IntelligenceError("Every rubric point needs a criterion and positive marks (at most "
                                f"{MAX_MARKS:g}).") from None
    total = sum(p.marks for p in rubric)
    bad = rubric_problems(rubric, total)
    if bad or not question.strip() or not student_answer.strip():
        raise IntelligenceError("The question, answer and rubric must all be filled in, and every rubric point "
                                "must carry positive marks.", bad)
    template, version = load_prompt("grade")
    answer_text = ANSWER_FENCE.sub("", student_answer.strip())   # the answer cannot close its own fence
    feedback, problems = "", []
    for attempt in range(1, max_attempts + 1):
        prompt = Template(template).safe_substitute(question=question.strip(), rubric=format_rubric(rubric),
                                                    total=f"{total:g}", answer=answer_text, feedback=feedback)
        out = llm.generate_json(prompt, LLMGrade)
        problems = check_grade(out, rubric)
        if not problems:
            by_point = {s.point: s for s in out.scores}
            points = [PointGrade(point=i, criterion=p.criterion, max_marks=p.marks, score=by_point[i].score,
                                 feedback=by_point[i].feedback.strip()) for i, p in enumerate(rubric, 1)]
            return Grade(points=points, total=round(sum(p.score for p in points), 2), max_total=round(total, 2),
                         overall_feedback=out.overall_feedback.strip(), attempts=attempt, prompt_version=version,
                         model=getattr(llm, "model", ""))
        feedback = "\nYour previous reply failed validation. Fix these problems:\n- " + "\n- ".join(problems[:8])
    raise GradingFailed(f"Could not produce a grade that passed validation in {max_attempts} attempts. Please try again.",
                        problems)
