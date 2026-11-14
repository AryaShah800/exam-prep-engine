"""Data models shared by every pipeline step. Each step reads and writes JSONL rows of these models."""
from typing import Literal, Optional
from pydantic import BaseModel, Field

ExamType = Literal["mid", "end", "remedial", "other"]


class PaperMeta(BaseModel):
    paper_id: str                      # e.g. OS_2025_11_END, or BITM-AI_2019_11_END for another university
    subject: str                       # short code, e.g. OS
    source: str = "DDU"                # university the paper comes from; only DDU papers count for prediction/stats
    year: int
    month: Optional[int] = None
    exam_type: ExamType = "end"
    source_file: str


class PageText(BaseModel):
    paper_id: str
    page: int                          # 1-based
    method: Literal["text", "ocr"]     # PyMuPDF text layer, or Gemini vision on a rendered page
    text: str
    figures: list[str] = Field(default_factory=list)   # descriptions of diagrams/tables on the page (OCR only)


class Question(BaseModel):
    question_id: str                   # <paper_id>-Q<no>[<sub_part>], unique
    paper_id: str
    subject: str
    source: str = "DDU"
    exam_type: ExamType
    year: int
    month: Optional[int] = None
    question_no: str                   # "1", "2", ...
    sub_part: Optional[str] = None     # "a", "b", "i", ...
    marks: Optional[float] = None
    is_or_alternative: bool = False    # True for the option printed after "OR"
    text: str
    page: Optional[int] = None
    split_method: Literal["rules", "llm"] = "rules"
    co: Optional[str] = None           # course outcome printed beside the question, e.g. "CO3" (DDU layout)
    bloom: Optional[str] = None        # Bloom level letter printed with it: R U A N E C


class Topic(BaseModel):
    topic_id: str                      # <SUBJECT>-U<unit>-T<n>
    unit_no: int
    unit_title: str
    name: str
    keywords: list[str] = Field(default_factory=list)


class Taxonomy(BaseModel):
    subject: str
    subject_name: str = ""
    topics: list[Topic]

    def ids(self) -> set[str]:
        return {t.topic_id for t in self.topics}

    def by_id(self) -> dict[str, Topic]:
        return {t.topic_id: t for t in self.topics}


class TopicTag(BaseModel):
    question_id: str
    topic_ids: list[str]
    confidence: Literal["high", "medium", "low"] = "medium"
    needs_review: bool = False
    off_syllabus: bool = False         # question tests something this syllabus doesn't cover (common for other universities)
    source: Literal["llm", "human", "claude"] = "llm"   # human = Arya's hand check; claude = AI second-model review
    model: str = ""
    prompt_version: str = ""


# ---- Structured-output models sent to Gemini (kept small: flat fields, no defaults the model must guess) ----

class LLMPage(BaseModel):
    text: str
    figures: list[str]


class LLMQuestion(BaseModel):
    question_no: str
    sub_part: Optional[str]
    marks: Optional[float]
    is_or_alternative: bool
    text: str


class LLMSplit(BaseModel):
    questions: list[LLMQuestion]


class LLMTopic(BaseModel):
    name: str
    keywords: list[str]


class LLMUnit(BaseModel):
    unit_no: int
    title: str
    topics: list[LLMTopic]


class LLMSyllabus(BaseModel):
    subject_name: str
    units: list[LLMUnit]


class LLMTag(BaseModel):
    question_id: str
    topic_ids: list[str]
    confidence: Literal["high", "medium", "low"]
    off_syllabus: bool = False         # no topic in the list covers what the question tests


class LLMTagBatch(BaseModel):
    tags: list[LLMTag]
