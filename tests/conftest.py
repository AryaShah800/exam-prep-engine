"""Shared fixtures: a temp data folder, a DDU-style sample paper (page 1 digital, page 2 scanned) and FakeLLM."""
import re
from pathlib import Path
import pymupdf, pytest
from exam_prep import config
from exam_prep.schema import LLMPage, LLMQuestion, LLMSplit, LLMSyllabus, LLMTag, LLMTagBatch, LLMTopic, LLMUnit

PAGE1 = """DHARMSINH DESAI UNIVERSITY, NADIAD
B.Tech Sem V (IT) End Semester Examination, November 2025
Subject: Operating Systems        Max Marks: 60        Time: 3 hours
Instructions: (1) Figures to the right indicate full marks. (2) Assume suitable data.

Q.1 Attempt any four.                                                    [8]
(a) Define a process. Differentiate between a process and a thread.      [2]
(b) What is a context switch?                                            [2]
(c) List the necessary conditions for deadlock.                          [2]
(d) What is thrashing?                                                   [2]
(e) Explain the purpose of the TLB.                                      [2]
Q.2 (a) Explain the Banker's algorithm for deadlock avoidance with an example. [6]
(b) Consider the following processes with arrival time and burst time:
P1 0 8, P2 1 4, P3 2 9, P4 3 5
Draw the Gantt chart and compute average waiting time using SJF (preemptive). [6]
OR
(b) Compare FCFS, SJF and Round Robin scheduling algorithms.              [6]
Q.3 Explain paging with a neat diagram.                                  [8]
OR
Q.3 Explain segmentation and compare it with paging.                     [8]
Page 1 of 2"""

PAGE2_OCR = """Q.4 (a) Explain the readers-writers problem using semaphores. [6]
(b) What is a page fault? Calculate page faults for the reference string 7 0 1 2 0 3 0 4 with 3 frames using LRU. [6]
Page 2 of 2"""

TAXONOMY_UNITS = [
    LLMUnit(unit_no=1, title="Processes and Threads", topics=[LLMTopic(name="Process concept", keywords=["process", "PCB"]), LLMTopic(name="Threads", keywords=["thread"]), LLMTopic(name="Context switching", keywords=["context switch"])]),
    LLMUnit(unit_no=2, title="CPU Scheduling", topics=[LLMTopic(name="Scheduling algorithms", keywords=["FCFS", "SJF", "Round Robin", "Gantt"])]),
    LLMUnit(unit_no=3, title="Synchronization and Deadlocks", topics=[LLMTopic(name="Semaphores and classic problems", keywords=["semaphore", "readers-writers"]), LLMTopic(name="Deadlock conditions", keywords=["deadlock", "conditions"]), LLMTopic(name="Banker's algorithm", keywords=["Banker"])]),
    LLMUnit(unit_no=4, title="Memory Management", topics=[LLMTopic(name="Paging and TLB", keywords=["paging", "TLB"]), LLMTopic(name="Segmentation", keywords=["segmentation"]), LLMTopic(name="Page replacement", keywords=["page fault", "LRU"]), LLMTopic(name="Thrashing", keywords=["thrashing"])]),
]

# keyword -> topic id; first match wins. "thrashing" deliberately maps to an ID that is not in the taxonomy.
KEYWORD_TOPICS = [("Banker", "OS-U3-T3"), ("readers-writers", "OS-U3-T1"), ("deadlock", "OS-U3-T2"), ("context switch", "OS-U1-T3"), ("Gantt", "OS-U2-T1"), ("Round Robin", "OS-U2-T1"), ("page fault", "OS-U4-T3"), ("TLB", "OS-U4-T1"), ("segmentation", "OS-U4-T2"), ("paging", "OS-U4-T1"), ("thrashing", "OS-U9-T9"), ("process", "OS-U1-T1")]


class FakeLLM:
    """Same interface as GeminiLLM; answers from canned data and records every call."""

    def __init__(self, model: str = "fake-model"):
        self.model = model
        self.calls: list[tuple[str, str, int]] = []

    def generate_json(self, prompt, schema, images=None):
        self.calls.append((schema.__name__, prompt, len(images or [])))
        if schema is LLMPage:
            return LLMPage(text=PAGE2_OCR, figures=[])
        if schema is LLMSyllabus:
            return LLMSyllabus(subject_name="Operating Systems", units=TAXONOMY_UNITS)
        if schema is LLMTagBatch:
            questions = re.findall(r"^\[([^\]]+)\][^\n]*?(.*)$", prompt.split("Questions:", 1)[1], re.M)
            tags = []
            for qid, text in questions:
                tid = next((t for kw, t in KEYWORD_TOPICS if kw.lower() in text.lower()), "OS-U1-T1")
                tags.append(LLMTag(question_id=qid, topic_ids=[tid], confidence="high"))
            return LLMTagBatch(tags=tags)
        if schema is LLMSplit:
            return LLMSplit(questions=[
                LLMQuestion(question_no="1", sub_part="a", marks=2, is_or_alternative=False, text="Define a process. Differentiate between a process and a thread."),
                LLMQuestion(question_no="2", sub_part="a", marks=6, is_or_alternative=False, text="Explain the Banker's algorithm for deadlock avoidance with an example."),
                LLMQuestion(question_no="5", sub_part=None, marks=10, is_or_alternative=False, text="Explain the history of UNIX in detail."),   # not in the paper
            ])
        raise AssertionError(f"unexpected schema {schema}")


def make_sample_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((40, 50), PAGE1, fontsize=8)
    # Page 2: text rendered to an image, so it has no text layer, like a scanned paper.
    tmp = pymupdf.open()
    tp = tmp.new_page()
    tp.insert_text((40, 50), PAGE2_OCR, fontsize=9)
    png = tp.get_pixmap(dpi=100).tobytes("png")
    page2 = doc.new_page()
    page2.insert_image(page2.rect, stream=png)
    doc.save(path)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    original = config.ROOT
    config.set_root(tmp_path)
    monkeypatch.setattr(config, "GEMINI_API_KEY", None)
    make_sample_pdf(config.RAW_DIR / "OS" / "OS_2025_11_END.pdf")
    yield tmp_path
    config.set_root(original)


@pytest.fixture
def fake_llm():
    return FakeLLM()


class FakeGraphLib:
    """Stand-in for langgraph.graph (StateGraph, START, END) so the router's wiring is tested without LangGraph.
    invoke() walks START -> nodes -> END, merging each node's returned dict into the state, and records the path."""
    START, END = "__start__", "__end__"

    class StateGraph:
        def __init__(self, schema):
            self.nodes, self.edges, self.branches = {}, {}, {}

        def add_node(self, name, fn):
            assert name not in self.nodes, f"duplicate node {name}"
            self.nodes[name] = fn

        def add_edge(self, src, dst):
            self.edges[src] = dst

        def add_conditional_edges(self, src, pick, path_map):
            self.branches[src] = (pick, path_map)

        def compile(self):
            return self

        def invoke(self, state):
            state, node, path = dict(state), self.edges[FakeGraphLib.START], []
            while node != FakeGraphLib.END:
                path.append(node)
                state.update(self.nodes[node](state) or {})
                if node in self.branches:
                    pick, path_map = self.branches[node]
                    node = path_map[pick(state)]
                else:
                    node = self.edges[node]
            state["path"] = path
            return state


@pytest.fixture
def fake_graph_lib():
    return FakeGraphLib
